"""Versioned validation windows stored as MLflow datasets and real array artifacts."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, InputTag
import numpy as np
import torch

from dataset import PerformanceWindowDataset, load_performances_by_ids
from support.mlflow_local import ROOT, TRACKING_URI
from tokenizer import TokenizerConfig
from support.training_state import batch_fingerprint, load_checkpoint, split_manifest


DEFAULT_POINTER = ROOT / "evaluation_datasets" / "default.json"


def default_evaluation_uri():
    if DEFAULT_POINTER.exists():
        return json.loads(DEFAULT_POINTER.read_text())["uri"]
    return None


def save_dataset(directory, name, batches, validation_data, source_run_id):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    x = np.stack([a.numpy() for a, _ in batches])
    y = np.stack([b.numpy() for _, b in batches])
    np.savez_compressed(directory / "windows.npz", x=x, y=y)
    metadata = {
        "version": 1, "name": name, "source_run_id": source_run_id,
        "sha256": hashlib.sha256((directory / "windows.npz").read_bytes()).hexdigest(),
        "validation_sha256": batch_fingerprint(batches),
        "shape": list(x.shape), "target_tokens": int(y.size),
        "vocabulary": validation_data.vocabulary,
        "tokenizer_config": asdict(validation_data.tokenizer_config),
        "target_tpq": validation_data.target_tpq,
        "performances": split_manifest(validation_data)["performances"],
    }
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def read_dataset(directory, validation_data, config):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["version"] != 1:
        raise ValueError("Unsupported evaluation dataset version")
    if hashlib.sha256((directory / "windows.npz").read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Evaluation artifact checksum mismatch")
    expected_shape = [config.eval_batches, config.batch_size, config.context_length]
    if metadata["shape"] != expected_shape:
        raise ValueError("Evaluation dataset batch/context shape differs; use compatible settings or a separate dataset version")
    if (metadata["vocabulary"] != validation_data.vocabulary or
            metadata["tokenizer_config"] != asdict(validation_data.tokenizer_config) or
            metadata["target_tpq"] != validation_data.target_tpq):
        raise ValueError("Evaluation dataset tokenizer/vocabulary differs")
    expected_performances = json.loads(json.dumps(split_manifest(validation_data)["performances"]))
    if metadata["performances"] != expected_performances:
        raise ValueError("Evaluation dataset held-out performances or token hashes differ")
    with np.load(directory / "windows.npz", allow_pickle=False) as arrays:
        x, y = arrays["x"], arrays["y"]
    if x.dtype != np.int64 or y.dtype != np.int64 or list(x.shape) != expected_shape or y.shape != x.shape:
        raise ValueError("Invalid evaluation token arrays")
    if (x.min() < 0 or y.min() < 0 or x.max() >= len(metadata["vocabulary"])
            or y.max() >= len(metadata["vocabulary"]) or not np.array_equal(x[:, :, 1:], y[:, :, :-1])):
        raise ValueError("Invalid evaluation targets or token IDs")
    batches = [(torch.from_numpy(a.copy()), torch.from_numpy(b.copy())) for a, b in zip(x, y)]
    if batch_fingerprint(batches) != metadata["validation_sha256"]:
        raise ValueError("Evaluation window fingerprint mismatch")
    if y.size != metadata["target_tokens"]:
        raise ValueError("Evaluation target count mismatch")
    return batches, metadata


def attach_dataset(client, run_id, uri, batches, metadata):
    source_id, path = parse_uri(uri)
    artifact_uri = client.get_run(source_id).info.artifact_uri.rstrip("/") + "/" + path + "/windows.npz"
    features = np.concatenate([x.numpy() for x, _ in batches])
    targets = np.concatenate([y.numpy() for _, y in batches])
    dataset = mlflow.data.from_numpy(features, targets=targets, source=artifact_uri,
                                     name=metadata["name"], digest=metadata["sha256"][:32])
    client.log_inputs(run_id, datasets=[DatasetInput(dataset=Dataset(**dataset.to_dict()), tags=[
        InputTag("mlflow.data.context", "validation"),
        InputTag("validation_sha256", metadata["validation_sha256"]),
    ])])
    for key, value in {"eval_dataset_name": metadata["name"], "eval_dataset_digest": dataset.digest,
                       "eval_dataset_uri": uri}.items():
        client.log_param(run_id, key, value)
    return dataset


def parse_uri(uri):
    if not uri.startswith("runs:/"):
        raise ValueError("Evaluation dataset URI must be runs:/<run-id>/<artifact-directory>")
    parts = uri[len("runs:/"):].split("/", 1)
    if len(parts) != 2 or not all(parts) or ".." in parts[1].split("/"):
        raise ValueError("Invalid evaluation dataset URI")
    return parts


def load_dataset(uri, validation_data, config, client):
    source_id, path = parse_uri(uri)
    with tempfile.TemporaryDirectory() as temporary:
        directory = client.download_artifacts(source_id, path, dst_path=temporary)
        batches, metadata = read_dataset(directory, validation_data, config)
    if DEFAULT_POINTER.exists():
        pinned = json.loads(DEFAULT_POINTER.read_text())
        if pinned["uri"] == uri and pinned["sha256"] != metadata["sha256"]:
            raise ValueError("Default evaluation dataset version changed")
    return batches, metadata


def publish_checkpoint_dataset(checkpoint_path, name, *, set_default=False):
    # Publication happens after training; reconstruct the same original sample
    # once, verify its saved fingerprint, then future runs read the array artifact.
    from train import TrainConfig, make_loader

    checkpoint = load_checkpoint(checkpoint_path)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    run_id = checkpoint["mlflow_run_id"]
    if client.get_run(run_id).info.status not in {"FINISHED", "KILLED", "FAILED"}:
        raise ValueError("Publish only after the source training run has stopped")
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Dataset name must be a single directory name")
    artifact_path = f"evaluation_datasets/{name}"
    if client.list_artifacts(run_id, artifact_path):
        raise ValueError("Dataset version already exists; do not overwrite an evaluation version")
    config = TrainConfig(**checkpoint["train_config"])
    saved = checkpoint["split"]["validation"]["performances"]
    rows = load_performances_by_ids([p["id"] for p in saved])
    validation_data = PerformanceWindowDataset(rows, config.context_length, config.target_tpq,
                                               TokenizerConfig(**checkpoint["tokenizer_config"]))
    if (split_manifest(validation_data)["performances"] != saved or
            validation_data.vocabulary != checkpoint["vocabulary"]):
        raise ValueError("Validation source data changed since training")
    batches = list(make_loader(validation_data, config.eval_batches, config.batch_size, config.seed + 1))
    if batch_fingerprint(batches) != checkpoint["validation_sha256"]:
        raise ValueError("Reconstructed validation sample differs from training")
    directory = ROOT / "evaluation_datasets" / name
    metadata = save_dataset(directory, name, batches, validation_data, run_id)
    read_dataset(directory, validation_data, config)
    client.log_artifacts(run_id, str(directory), artifact_path=artifact_path)
    uri = f"runs:/{run_id}/{artifact_path}"
    attach_dataset(client, run_id, uri, batches, metadata)
    if set_default:
        DEFAULT_POINTER.write_text(json.dumps({"uri": uri, "name": name,
                                               "sha256": metadata["sha256"]}, indent=2) + "\n")
    return uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--set-default", action="store_true")
    args = parser.parse_args()
    print(publish_checkpoint_dataset(args.checkpoint, args.name, set_default=args.set_default))
