# Piano MIDI models

Two PyTorch models for piano MIDI generation, trained on PianoCoRe with
composition-disjoint validation and local MLflow tracking.

- **Event Transformer** (`model.py`, `train.py`): velocity → note-on, note-off,
  and time-shift tokens. Uses 32 velocity bins and a 10-tick grid at 480 TPQ.
  Supports RoPE, type/value output heads, and pitch augmentation.
- **Moonbeam** (`moonbeam/`): complete-note Transformer with musical relative
  attention and a GRU attribute decoder. See [Moonbeam usage](moonbeam/README.md).

The repository contains source only. Training data, weights, run databases,
reports, and virtual environments are excluded from Git.

## Setup

Use Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an NVIDIA environment with a working CUDA PyTorch installation, install
`requirements-gpu.txt` instead to preserve that installation. Training downloads
selected MIDI records from `SyMuPe/PianoCoRe`; the first run needs network access.

## Train the event Transformer

A small initial run:

```sh
python train.py --steps 1000 --eval-dataset sampled
```

The larger event-model configuration used in our experiments:

```sh
python train.py --steps 50000 \
  --d-model 256 --d-ff 1024 --num-heads 4 --num-layers 4 \
  --context-length 512 --position-encoding rope --output-mode type_value \
  --pitch-shift 3 --num-compositions 500 --max-performances 1000 \
  --eval-dataset sampled
```

The device is selected automatically; override with `--device cpu`, `mps`, or
`cuda`. Data selection and the validation split are deterministic for a given
seed. Fresh commands create new splits and do not reproduce the historical
six-composition holdout.

Resume using the path printed by training:

```sh
python train.py --resume checkpoints/<run-id>/last.pt --steps 100000
```

`--steps` is the total target, including completed steps. Resume restores model,
optimizer, scheduler, split, validation windows, and random state. `best.pt`
contains the best validation checkpoint; `last.pt` is the latest continuation
checkpoint. Use `--data-from PATH` for fresh weights with a saved split/settings.
Run `python train.py --help` for all options.

## Tracking and evaluation

```sh
python -m support.mlflow_local --port 5001
```

Open [MLflow](http://127.0.0.1:5001). Runs are stored in `mlflow.db`, artifacts in
`mlartifacts/`, and checkpoints in `checkpoints/`. The server binds to localhost;
forward its port over SSH when training remotely.

Validation reports loss, exact next-token accuracy, per-event metrics, top-5
accuracy, and timing/velocity near misses. These are prediction metrics, not
listening-quality scores. Moonbeam uses different targets and metrics, so its
accuracy is not directly comparable with the event model.

`support/evaluation_dataset.py` optionally saves fixed validation windows to
MLflow. A local `evaluation_datasets/default.json` can select those windows;
`--eval-dataset sampled` explicitly uses the current run's sampled validation.

## Layout

```text
model.py          Event Transformer
train.py          Event-model training and resume
tokenizer.py     Event encoding and quantization
dataset.py       MIDI loading, composition splits, window sampling
moonbeam/         Complete-note model, tokenizer, and trainer
support/          Tracking, evaluation, augmentation, and checkpoint helpers
```
