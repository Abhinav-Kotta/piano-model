# Moonbeam

An independent implementation of Moonbeam's note-level architecture: numerical
musical embeddings, six-group musical relative attention, RMSNorm/SwiGLU
Transformer blocks, and a GRU that predicts the next note's attributes.
It is an adaptation, not a checkpoint-compatible reproduction of the authors'
model. No pretrained weights are included.

## Train

Run from the repository root after installing its requirements:

```sh
python -m moonbeam.train --steps 1000
```

The compact training model has 4.55M parameters, four Transformer layers at
width 192, and a one-layer GRU. The `gpu-114m` preset has 113.83M parameters,
12 Transformer layers at width 768, and a two-layer GRU. `gpu-315m` is also
available. GPU presets use CUDA BF16, microbatch 2, effective batch 8, activation
checkpointing, and 500 warmup updates.

Check CUDA execution and memory before training:

```sh
python -m moonbeam.train --preset gpu-114m --preflight
```

A fresh regularized GPU run:

```sh
python -m moonbeam.train --preset gpu-114m --steps 50000 \
  --num-compositions 500 --max-performances 1000 \
  --dropout 0.1 --weight-decay 0.1 \
  --eval-batches 64 --validation-sampling coverage \
  --early-stopping-patience 60 --early-stopping-min-delta 0.0005
```

These regularization settings are experimental. The command creates a new
composition-disjoint split. Coverage evaluation samples every validation
performance across fixed windows; it does not evaluate every note. The window
budget (`eval-batches × batch-size`) must cover all validation performances.
50,000 steps is a maximum: patience 60 stops after 3,000 updates without a
qualifying improvement when evaluation runs every 50 updates.

Use `--micro-batch-size 1` if GPU memory is insufficient. Larger microbatches and
`--no-gradient-checkpointing` may improve throughput when memory permits.
Parameters and optimizer state remain FP32; BF16 autocast covers the Transformer
and projections, while the GRU runs in FP32.

## Reuse data and resume

`--data-from PATH` reuses a baseline/Moonbeam checkpoint's split with fresh
weights. It also accepts a portable JSON split manifest. `--midi-records PATH`
loads local serialized PianoCoRe records instead of downloading them. Neither
file is included in this repository.

Checkpoints are saved under `checkpoints/moonbeam/<run-id>/`:

- `best.pt`: weights at the lowest validation loss.
- `last.pt`: model, optimizer, scheduler, split, RNG, and early-stopping state.

```sh
python -m moonbeam.train \
  --resume checkpoints/moonbeam/<run-id>/last.pt --steps 100000
```

The step count is a total target. Resume inherits saved configuration and checks
MIDI/validation hashes. Device, logging cadence, microbatch size, and activation
checkpointing may change; architecture, data encoding, and regularization may
not. To change dropout or restart after early stopping, start a fresh experiment.
Use `python -m moonbeam.train --help` for all options. MLflow is shared with the
root trainer; run `python -m support.mlflow_local --port 5001` for its dashboard.

## Representation and metrics

Each input row contains:

```text
[absolute onset, duration, octave, pitch class, instrument, velocity]
```

Timing is quantized to 10 ms; velocity retains MIDI values 1–127. Training
supports onset gaps up to 40.96 seconds and durations up to 20.48 seconds.
Out-of-range values raise errors rather than clipping. The tokenizer follows
the MIDI tempo map, removes initial silence, and preserves elapsed timing.
Pedal/controllers and track identity are not modeled.

The default decoder order is onset gap → duration → velocity → octave → pitch
class → instrument. `--attribute-order paper` moves velocity last. Training uses
teacher forcing: earlier target attributes condition later predictions.

- `val_attribute_accuracy`: exact matches across all six attributes, including
  the nearly constant piano instrument and non-padding special targets.
- `val_music_attribute_accuracy`: mean accuracy of the five non-instrument
  attributes on actual notes.
- `val_note_accuracy_teacher_forced`: all six attributes match on a note.

These metrics do not measure generated musical quality. Timing and velocity
near misses receive no partial credit in the exact-match metrics.

BOS, EOS, and PAD rows repeat -1, -2, and -3 in all six columns. Notes are sorted
by onset; equal onsets form chords. Padding is on the right, and pieces are not
packed together. A sequence of 512 rows produces 511 next-note targets.

## Smoke check and generation

```sh
python -m moonbeam.model --smoke --device cpu
```

```python
from moonbeam.model import MoonbeamModel
from moonbeam.tokenizer import NoteTokenizer

model = MoonbeamModel.load("checkpoints/moonbeam/<run-id>/best.pt")
codec = NoteTokenizer(model.config)
prefix = codec.load_midi("prompt.mid", include_eos=False)[:128]
notes = model.generate(prefix[None], max_new_notes=128)
codec.decode(notes[0]).dump_midi("generated.mid")
```

Generation uses a sliding context. KV caching, infilling, conditional controls,
distributed training, and upstream checkpoint import are not implemented.

Architecture sources: [Moonbeam paper](https://arxiv.org/abs/2505.15559) and
[reference implementation](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model).
