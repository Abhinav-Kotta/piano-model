"""Train on varied MIDI windows and track the experiment locally with MLflow."""

import argparse
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import math
from pathlib import Path
import platform
import time

import mlflow
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import (TARGET_TPQ, RandomWindowBatchSampler, build_datasets,
                     load_performances, load_performances_by_ids)
from support.diagnostics import evaluate, fit_baselines
from support.training_state import (FIXED_FIELDS, SCHEDULER_FIELDS, MODEL_FIELDS, ARCHITECTURE_DEFAULTS, batch_fingerprint, load_checkpoint,
                            restore_datasets, restore_rng, rng_state, split_manifest, validate_resume,
                            build_with_saved_validation)
from support.mlflow_local import ARTIFACT_ROOT, EXPERIMENT, ROOT, TRACKING_URI, configure_tracking, log_source_files
from model import MusicTransformer
from tokenizer import TokenizerConfig
from support.augmentation import PitchAugmenter
from support.event_types import event_type_ids
from support.evaluation_dataset import default_evaluation_uri, load_dataset as load_evaluation_dataset, attach_dataset


@dataclass
class TrainConfig:
    steps: int = 1000
    max_performances: int = 64
    num_compositions: int = 32
    performances_per_composition: int = 2
    max_scan: int = 5000
    batch_size: int = 8
    context_length: int = 256
    target_tpq: int = TARGET_TPQ
    time_step: int = 10
    velocity_bins: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    validation_fraction: float = 0.2
    eval_every: int = 50
    eval_batches: int = 8
    log_every: int = 10
    seed: int = 42
    device: str = "auto"
    experiment: str = "piano-coarse-diverse"
    run_name: str | None = None
    resume: str | None = None
    validation_from: str | None = None
    lr_factor: float = 0.8
    lr_patience: int = 30
    lr_threshold: float = 0.0005
    min_lr: float = 1e-5
    pitch_shift: int = 0
    d_model: int = 128
    d_ff: int = 512
    num_heads: int = 4
    num_layers: int = 4
    data_from: str | None = None
    eval_dataset: str | None = None
    output_mode: str = "flat"
    position_encoding: str = "learned"
    reset_scheduler: bool = False
    resume_learning_rate: float | None = None

    def validate(self):
        if self.reset_scheduler and not self.resume:
            raise ValueError("reset_scheduler requires resume")
        if self.resume_learning_rate is not None:
            if not self.resume or not self.reset_scheduler:
                raise ValueError("resume_learning_rate requires resume and reset_scheduler")
            if (not math.isfinite(self.resume_learning_rate) or
                    not self.min_lr <= self.resume_learning_rate <= self.learning_rate):
                raise ValueError("resume_learning_rate must be finite, at least min_lr, and at most learning_rate")
        if not isinstance(self.pitch_shift, int) or not 0 <= self.pitch_shift <= 12:
            raise ValueError("pitch_shift must be an integer from 0 to 12 semitones")
        if self.resume and (self.validation_from or self.data_from):
            raise ValueError("Use data_from/validation_from for fresh runs; resume already preserves its saved split")
        if self.data_from and self.validation_from:
            raise ValueError("data_from already preserves both splits; do not combine with validation_from")
        for name in ("steps", "batch_size", "context_length", "target_tpq",
                     "eval_every", "eval_batches", "log_every", *MODEL_FIELDS):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.output_mode not in ("flat", "type_value") or self.position_encoding not in ("learned", "rope"):
            raise ValueError("Unknown output mode or positional encoding")
        if self.position_encoding == "rope" and (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.max_performances < 2:
            raise ValueError("max_performances must be at least two")
        if self.num_compositions < 2 or self.performances_per_composition < 1:
            raise ValueError("Need at least two compositions and one performance per composition")
        if self.max_performances < self.num_compositions or self.max_scan < self.num_compositions:
            raise ValueError("max_performances and max_scan must be at least num_compositions")
        TokenizerConfig(time_step=self.time_step, velocity_bins=self.velocity_bins)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if not 0 < self.lr_factor < 1 or self.lr_patience < 0:
            raise ValueError("lr_factor must be between zero and one; lr_patience must be nonnegative")
        if not math.isfinite(self.lr_threshold) or self.lr_threshold < 0:
            raise ValueError("lr_threshold must be finite and nonnegative")
        if not 0 < self.min_lr <= self.learning_rate:
            raise ValueError("min_lr must be positive and at most learning_rate")


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset, batches, batch_size, seed):
    return DataLoader(dataset, batch_sampler=RandomWindowBatchSampler(
        dataset, num_batches=batches, batch_size=batch_size, seed=seed
    ), generator=torch.Generator().manual_seed(seed))


def train(config, rows=None, *, tracking_uri=TRACKING_URI,
          artifact_root=ARTIFACT_ROOT, checkpoint_root=ROOT / "checkpoints"):
    """rows can supply in-memory MIDI records for offline integration tests."""
    if config.eval_dataset is None and tracking_uri == TRACKING_URI:
        config = replace(config, eval_dataset=default_evaluation_uri())
    config.validate()
    checkpoint = load_checkpoint(config.resume) if config.resume else None
    validation_source = load_checkpoint(config.validation_from) if config.validation_from else None
    data_source = load_checkpoint(config.data_from) if config.data_from else None
    if checkpoint:
        validate_resume(config, checkpoint)
    start_step = checkpoint["step"] if checkpoint else 0
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    client = configure_tracking(config.experiment, tracking_uri, artifact_root)
    with mlflow.start_run(run_name=config.run_name) as run:
        run_id = run.info.run_id
        print(f"Run: {run_id} | Device: {device}", flush=True)
        if checkpoint:
            mlflow.set_tags({"resumed_from_run_id": checkpoint["mlflow_run_id"],
                             "resume_step": str(start_step),
                             "scheduler_restored": str("scheduler_state_dict" in checkpoint and not config.reset_scheduler),
                             "scheduler_reset": str(config.reset_scheduler)})
            print(f"Resuming step {start_step}; target is {config.steps} total steps", flush=True)
            previous_shift = checkpoint["train_config"].get("pitch_shift", 0)
            mlflow.set_tags({"previous_pitch_shift": str(previous_shift),
                             "augmentation_changed": str(previous_shift != config.pitch_shift)})
            if previous_shift != config.pitch_shift:
                print(f"Pitch augmentation changed: ±{previous_shift} -> ±{config.pitch_shift} semitones; "
                      "starting a new augmentation RNG stream", flush=True)
        if validation_source:
            mlflow.set_tags({"validation_from_run_id": validation_source["mlflow_run_id"],
                             "validation_selection": "saved_compositions_and_performances"})
        if data_source:
            mlflow.set_tags({"data_from_run_id": data_source["mlflow_run_id"],
                             "validation_selection": "saved_split_and_windows",
                             "initialization": "fresh_weights"})
        mlflow.log_params({k: v for k, v in asdict(config).items() if v is not None})
        mlflow.set_tags({"data_source": "SyMuPe/PianoCoRe" if rows is None else "provided_rows",
                         "resolved_device": str(device), "split_unit": "composition",
                         "tokenizer_version": "2", "selection": "composition_first"})
        tokenizer_config = TokenizerConfig(time_step=config.time_step, velocity_bins=config.velocity_bins)
        mlflow.log_dict({"version": 2, "target_tpq": config.target_tpq,
                         **asdict(tokenizer_config)}, "tokenizer.json")
        mlflow.log_dict({"python": platform.python_version(), **{
            name: version(name) for name in ("torch", "mlflow", "datasets", "symusic", "pyarrow")
        }}, "environment.json")
        log_source_files()

        print("Loading and tokenizing performances...", flush=True)
        if rows is None:
            saved_source = checkpoint or data_source
            if saved_source:
                ids = [p["id"] for name in ("train", "validation")
                       for p in saved_source["split"][name]["performances"]]
                rows = load_performances_by_ids(ids)
            else:
                rows = load_performances(config.max_performances, config.num_compositions,
                                         config.performances_per_composition, config.max_scan, config.seed,
                                         heldout_performances=(validation_source["split"]["validation"]["performances"]
                                                               if validation_source else None))
        if checkpoint or data_source:
            saved_source = checkpoint or data_source
            train_data, validation_data = restore_datasets(rows, saved_source["split"], config, tokenizer_config)
            if (saved_source["vocabulary"] != train_data.vocabulary or
                    saved_source["tokenizer_config"] != asdict(tokenizer_config)):
                raise ValueError("Checkpoint vocabulary/tokenizer does not match")
        elif validation_source:
            train_data, validation_data = build_with_saved_validation(
                rows, validation_source, config, tokenizer_config)
        else:
            train_data, validation_data = build_datasets(
                rows, context_length=config.context_length, target_tpq=config.target_tpq,
                validation_fraction=config.validation_fraction, seed=config.seed,
                tokenizer_config=tokenizer_config,
            )
        manifest = {"train": split_manifest(train_data), "validation": split_manifest(validation_data)}
        mlflow.log_dict(manifest, "split.json")
        for split, dataset in (("train", train_data), ("validation", validation_data)):
            print(f"{split}: {len(dataset.sequences)} performances, "
                  f"{len(set(dataset.composition_keys))} compositions", flush=True)
            mlflow.log_params({f"{split}_performances": len(dataset.sequences),
                               f"{split}_compositions": len(set(dataset.composition_keys)),
                               f"{split}_windows": len(dataset)})
        train_loader = make_loader(train_data, config.steps - start_step, config.batch_size, config.seed)
        augmenter = PitchAugmenter(train_data.vocabulary, config.pitch_shift, config.seed + 2)
        if checkpoint and checkpoint["train_config"].get("pitch_shift", 0) == config.pitch_shift:
            if "augmentation_rng_state" in checkpoint:
                augmenter.generator.set_state(checkpoint["augmentation_rng_state"])
        if checkpoint:
            if "sampler_rng_state" in checkpoint:
                train_loader.batch_sampler.generator.set_state(checkpoint["sampler_rng_state"])
            else:
                # Old checkpoints predate sampler state. Replay indices only,
                # not MIDI windows or model updates, to advance its private RNG.
                old_sampler = RandomWindowBatchSampler(train_data, start_step, config.batch_size, config.seed)
                for _ in old_sampler:
                    pass
                train_loader.batch_sampler.generator.set_state(old_sampler.generator.get_state())
        # Reuse the same sampled validation windows throughout the run.
        if config.eval_dataset and config.eval_dataset != "sampled":
            validation_batches, evaluation_metadata = load_evaluation_dataset(
                config.eval_dataset, validation_data, config, client)
            attach_dataset(client, run_id, config.eval_dataset, validation_batches, evaluation_metadata)
        else:
            validation_batches = list(make_loader(
                validation_data, config.eval_batches, config.batch_size, config.seed + 1
            ))
        validation_hash = batch_fingerprint(validation_batches)
        saved_source = checkpoint or data_source
        if saved_source and saved_source.get("validation_sha256", validation_hash) != validation_hash:
            raise ValueError("Validation windows differ from the checkpoint")
        mlflow.log_param("validation_sha256", validation_hash)
        baselines = fit_baselines(train_data)
        mlflow.log_dict({group: {"token_id": token_id, "token": train_data.vocabulary[token_id]}
                         for group, token_id in baselines.items()}, "diagnostic_baselines.json")
        model_config = dict(vocab_size=len(train_data.vocabulary), context_length=config.context_length,
                            **{name: getattr(config, name) for name in (*MODEL_FIELDS, *ARCHITECTURE_DEFAULTS)})
        if config.output_mode == "type_value":
            model_config["token_type_ids"] = event_type_ids(train_data.vocabulary)
        model = MusicTransformer(**model_config).to(device)
        mlflow.log_params({key: value for key, value in model_config.items() if key != "token_type_ids"})
        mlflow.log_dict(model_config, "model_config.json")
        mlflow.log_param("parameters", sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience,
            threshold=config.lr_threshold, threshold_mode="abs", min_lr=config.min_lr,
        )
        if checkpoint:
            if model_config != {**ARCHITECTURE_DEFAULTS, **checkpoint["model_config"]}:
                raise ValueError("Checkpoint model configuration does not match")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if config.reset_scheduler:
                previous_rates = [group["lr"] for group in optimizer.param_groups]
                if config.resume_learning_rate is not None:
                    for group in optimizer.param_groups:
                        group["lr"] = config.resume_learning_rate
                if any(group["lr"] < config.min_lr for group in optimizer.param_groups):
                    raise ValueError("Saved optimizer LR is below min_lr; specify resume_learning_rate")
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience,
                    threshold=config.lr_threshold, threshold_mode="abs", min_lr=config.min_lr,
                )
                mlflow.log_dict({"previous_optimizer_rates": previous_rates,
                                 "new_optimizer_rates": scheduler.get_last_lr(),
                                 "previous_scheduler": checkpoint.get("scheduler_state_dict"),
                                 "new_settings": {name: getattr(config, name) for name in SCHEDULER_FIELDS},
                                 "optimizer_moments_preserved": True}, "scheduler_change.json")
                print(f"Scheduler reset: LR {previous_rates} -> {scheduler.get_last_lr()}; "
                      f"factor={config.lr_factor}, patience={config.lr_patience}", flush=True)
            elif "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        output_dir = Path(checkpoint_root) / run_id
        output_dir.mkdir(parents=True, exist_ok=False)

        def save_checkpoint(filename, step, val_loss):
            path = output_dir / filename
            torch.save({
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config, "train_config": asdict(config),
                "vocabulary": train_data.vocabulary, "target_tpq": config.target_tpq,
                "tokenizer_config": asdict(tokenizer_config), "tokenizer_version": 2,
                "split": manifest, "step": step, "val_loss": val_loss, "mlflow_run_id": run_id,
                "scheduler_state_dict": scheduler.state_dict(),
                "sampler_rng_state": train_loader.batch_sampler.generator.get_state(),
                "torch_rng_state": rng_state(device), "validation_sha256": validation_hash,
                "augmentation_rng_state": augmenter.generator.get_state(),
            }, path)
            mlflow.log_artifact(str(path), artifact_path="checkpoints")
            return path

        metrics = evaluate(model, validation_batches, device, train_data.vocabulary, baselines)
        if not checkpoint or "scheduler_state_dict" not in checkpoint or config.reset_scheduler:
            scheduler.step(metrics["val_loss"])
        mlflow.log_metrics({**metrics, "learning_rate": optimizer.param_groups[0]["lr"]}, step=start_step)
        best_loss, best_step = metrics["val_loss"], start_step
        if checkpoint and "torch_rng_state" in checkpoint:
            restore_rng(checkpoint["torch_rng_state"], device)
        save_checkpoint("best.pt", start_step, best_loss)
        print(f"Step {start_step} | val_loss={best_loss:.4f}", flush=True)
        model.train()
        loss_sum, correct, tokens, interval_steps = 0.0, 0, 0, 0
        augmented_windows, absolute_shift, sampled_windows = 0, 0, 0
        started = time.monotonic()
        # Give DataLoader its own base-seed RNG, so constructing its iterator
        # does not consume the restored RNG used by stochastic model layers.
        train_loader.generator = torch.Generator().manual_seed(config.seed)
        for step, (x, y) in enumerate(train_loader, start_step + 1):
            x, y, shifts = augmenter(x, y)
            augmented_windows += int((shifts != 0).sum())
            absolute_shift += int(shifts.abs().sum())
            sampled_windows += len(shifts)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            tokens += y.numel()
            interval_steps += 1
            if step == start_step + 1 or step % config.log_every == 0 or step == config.steps:
                mlflow.log_metrics({"train_loss": loss_sum / interval_steps,
                                    "train_accuracy": correct / tokens,
                                    "train_transposed_window_fraction": augmented_windows / sampled_windows,
                                    "train_mean_absolute_pitch_shift": absolute_shift / sampled_windows}, step=step)
                print(f"Step {step} | train_loss={loss_sum / interval_steps:.4f}", flush=True)
                loss_sum, correct, tokens, interval_steps = 0.0, 0, 0, 0
                augmented_windows, absolute_shift, sampled_windows = 0, 0, 0
            if step % config.eval_every == 0 or step == config.steps:
                metrics = evaluate(model, validation_batches, device, train_data.vocabulary, baselines)
                previous_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(metrics["val_loss"])
                learning_rate = optimizer.param_groups[0]["lr"]
                mlflow.log_metrics({**metrics, "learning_rate": learning_rate}, step=step)
                if learning_rate != previous_lr:
                    print(f"Learning rate reduced: {previous_lr:g} -> {learning_rate:g}", flush=True)
                print(f"Step {step} | val_loss={metrics['val_loss']:.4f} | "
                      f"val_accuracy={metrics['val_accuracy']:.1%}", flush=True)
                if metrics["val_loss"] < best_loss:
                    best_loss, best_step = metrics["val_loss"], step
                    save_checkpoint("best.pt", step, best_loss)
        save_checkpoint("last.pt", config.steps, metrics["val_loss"])
        mlflow.log_metrics({"best_val_loss": best_loss, "best_step": best_step,
                            "training_seconds": time.monotonic() - started}, step=config.steps)
        print(f"Finished. Checkpoints: {output_dir}", flush=True)
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for name in ("steps", "max_performances", "batch_size", "context_length", "target_tpq",
                 "eval_every", "eval_batches", "log_every", "seed", "time_step", "velocity_bins",
                 "num_compositions", "performances_per_composition", "max_scan", "lr_patience", "pitch_shift",
                 *MODEL_FIELDS):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=argparse.SUPPRESS)
    for name in ("learning_rate", "weight_decay", "validation_fraction", "lr_factor", "lr_threshold", "min_lr"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--experiment", default=defaults.experiment)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="Checkpoint path; --steps is the TOTAL desired step count")
    parser.add_argument("--reset-scheduler", action="store_true",
                        help="Explicit schedule experiment on resume; reset plateau history, preserving optimizer moments")
    parser.add_argument("--resume-learning-rate", type=float,
                        help="Starting optimizer rate for this resume; requires --reset-scheduler, never inherited")
    parser.add_argument("--validation-from", help="Fresh model, preserving a checkpoint's held-out compositions and performances")
    parser.add_argument("--data-from", help="Fresh model, inheriting a checkpoint's training settings and exact train/validation split")
    parser.add_argument("--output-mode", choices=("flat", "type_value"), default=argparse.SUPPRESS)
    parser.add_argument("--position-encoding", choices=("learned", "rope"), default=argparse.SUPPRESS)
    parser.add_argument("--eval-dataset", default=argparse.SUPPRESS,
                        help="MLflow runs:/... evaluation dataset directory; defaults to the published workspace dataset. Use sampled to explicitly opt out.")
    args = parser.parse_args()
    values = vars(args)
    try:
        if args.resume or args.data_from:
            checkpoint = load_checkpoint(args.resume or args.data_from)
            if args.resume and "steps" not in values:
                parser.error("--resume requires an explicit --steps TOTAL")
            for name in FIXED_FIELDS + SCHEDULER_FIELDS + ("pitch_shift", "eval_dataset"):
                if name in checkpoint["train_config"]:
                    values.setdefault(name, checkpoint["train_config"][name])
            if args.resume:
                for name in MODEL_FIELDS:
                    values.setdefault(name, checkpoint["model_config"][name])
                for name, default in ARCHITECTURE_DEFAULTS.items():
                    values.setdefault(name, checkpoint["model_config"].get(name, default))
        config = TrainConfig(**values)
        config.validate()
        if args.resume:
            validate_resume(config, checkpoint)
    except ValueError as error:
        parser.error(str(error))
    train(config)


if __name__ == "__main__":
    main()
