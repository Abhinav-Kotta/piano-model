"""Checkpoint compatibility and exact reconstruction of data/window state."""

import hashlib

import torch

from dataset import PerformanceWindowDataset, composition_key


FIXED_FIELDS = (
    "max_performances", "num_compositions", "performances_per_composition", "max_scan",
    "batch_size", "context_length", "target_tpq", "time_step", "velocity_bins",
    "learning_rate", "weight_decay", "validation_fraction", "eval_every", "eval_batches", "seed",
)
SCHEDULER_FIELDS = ("lr_factor", "lr_patience", "lr_threshold", "min_lr")
MODEL_FIELDS = ("d_model", "d_ff", "num_heads", "num_layers")
ARCHITECTURE_DEFAULTS = {"output_mode": "flat", "position_encoding": "learned"}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "optimizer_state_dict", "model_config", "train_config",
                "vocabulary", "split", "step", "tokenizer_config", "tokenizer_version"}
    if required - checkpoint.keys() or checkpoint["tokenizer_version"] != 2:
        raise ValueError("Resume requires a coarse-token training checkpoint (tokenizer version 2)")
    return checkpoint


def validate_resume(config, checkpoint):
    if config.steps <= checkpoint["step"]:
        raise ValueError(f"steps is the TOTAL target and must exceed checkpoint step {checkpoint['step']}")
    fields = FIXED_FIELDS + (SCHEDULER_FIELDS if "scheduler_state_dict" in checkpoint
                            and not config.reset_scheduler else ())
    for name in fields:
        if getattr(config, name) != checkpoint["train_config"][name]:
            raise ValueError(f"Cannot change {name} when resuming; keep the saved configuration")
    for name in MODEL_FIELDS:
        if getattr(config, name) != checkpoint["model_config"][name]:
            raise ValueError(f"Cannot change {name} when resuming; use data_from for a fresh architecture experiment")
    for name, default in ARCHITECTURE_DEFAULTS.items():
        if getattr(config, name) != checkpoint["model_config"].get(name, default):
            raise ValueError(f"Cannot change {name} when resuming; use data_from for a fresh architecture experiment")
    # Changing augmentation is an explicit new experiment, with the existing
    # model/data/optimizer preserved. Continuing unchanged augmentation must
    # restore its RNG rather than silently repeat old transformations.
    if (config.pitch_shift > 0 and config.pitch_shift == checkpoint["train_config"].get("pitch_shift", 0)
            and "augmentation_rng_state" not in checkpoint):
        raise ValueError("Checkpoint is missing augmentation RNG state")


def split_manifest(dataset):
    return {
        "performances": [
            {"id": sample_id, "composition_key": key, "tokens": len(ids),
             "token_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
            for sample_id, key, ids in zip(dataset.performance_ids, dataset.composition_keys, dataset.sequences)
        ],
        "skipped_ids": dataset.skipped_ids,
        "windows": len(dataset),
    }


def restore_datasets(rows, manifest, config, tokenizer_config):
    by_id = {}
    for row in rows:
        key = str(row["id"])
        if key in by_id:
            raise ValueError(f"Duplicate performance ID: {key}")
        by_id[key] = row
    expected = [p["id"] for split in manifest.values() for p in split["performances"]]
    if len(expected) != len(set(expected)) or set(by_id) != set(expected):
        raise ValueError("Resume data IDs do not match the saved split")
    datasets = []
    for split in ("train", "validation"):
        saved = manifest[split]
        dataset = PerformanceWindowDataset([by_id[p["id"]] for p in saved["performances"]],
                                          config.context_length, config.target_tpq, tokenizer_config)
        actual = split_manifest(dataset)
        if actual["performances"] != saved["performances"] or actual["windows"] != saved["windows"]:
            raise ValueError(f"Resume {split} token hashes/metadata differ from the saved split")
        dataset.skipped_ids = saved["skipped_ids"]
        datasets.append(dataset)
    if set(datasets[0].composition_keys) & set(datasets[1].composition_keys):
        raise ValueError("Saved train and validation compositions overlap")
    return tuple(datasets)


def build_with_saved_validation(rows, checkpoint, config, tokenizer_config):
    """Keep exact held-out performances/encoding while allowing a new context."""
    saved = checkpoint["split"]["validation"]["performances"]
    heldout_keys = {tuple(p["composition_key"]) for p in saved}
    by_id = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate performance ID: {sample_id}")
        by_id[sample_id] = row
    if any(p["id"] not in by_id for p in saved):
        raise ValueError("Missing saved validation performance")
    train_rows = [row for row in by_id.values() if composition_key(row) not in heldout_keys]
    validation_rows = [by_id[p["id"]] for p in saved]
    datasets = tuple(PerformanceWindowDataset(group, config.context_length, config.target_tpq,
                                              tokenizer_config)
                     for group in (train_rows, validation_rows))
    if (checkpoint["vocabulary"] != datasets[0].vocabulary or
            checkpoint["target_tpq"] != config.target_tpq):
        raise ValueError("Saved validation vocabulary/TPQ must match")
    if split_manifest(datasets[1])["performances"] != saved:
        raise ValueError("Saved validation token hashes/metadata differ or a performance is too short")
    if set(datasets[0].composition_keys) & heldout_keys:
        raise ValueError("Train and saved validation compositions overlap")
    return datasets


def batch_fingerprint(batches):
    digest = hashlib.sha256()
    for x, y in batches:
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
