"""Train the compact Moonbeam model on PianoCoRe with local MLflow tracking."""
from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import argparse
import hashlib
import json
from importlib.metadata import version
from itertools import accumulate
import math
from pathlib import Path
import platform
import tempfile
import time

import mlflow
import torch
from symusic import Score
from torch.nn import functional as F

from dataset import (composition_key, load_performances, load_performances_by_ids,
                     split_by_composition, RandomWindowBatchSampler)
from support.mlflow_local import ROOT, TRACKING_URI, ARTIFACT_ROOT, configure_tracking
from support.training_state import rng_state, restore_rng
from .model import MoonbeamConfig, MoonbeamModel, ORDERS, PAD
from .tokenizer import NoteTokenizer


PRESETS = {
    'gpu-114m': dict(d_model=768, d_ff=2688, num_layers=12, num_heads=12,
                    num_kv_heads=6, decoder_dim=768, decoder_layers=2),
    'gpu-315m': dict(d_model=1536, d_ff=5376, num_layers=9, num_heads=12,
                    num_kv_heads=6, decoder_dim=1024, decoder_layers=2),
}
GPU_DEFAULTS = dict(device='cuda', precision='bf16', micro_batch_size=2,
                    warmup_steps=500, gradient_checkpointing=True)


@dataclass(frozen=True)
class TrainConfig(MoonbeamConfig):
    # Cover the measured PianoCoRe baseline split without clipping long rests/notes.
    max_delta: int = 4096
    max_duration: int = 2048
    steps: int = 1000
    batch_size: int = 8
    micro_batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.
    pitch_shift: int = 3
    max_performances: int = 64
    num_compositions: int = 32
    performances_per_composition: int = 2
    max_scan: int = 5000
    validation_fraction: float = .2
    eval_every: int = 50
    eval_batches: int = 8
    validation_sampling: str = 'random'
    log_every: int = 10
    checkpoint_every: int = 500
    lr_factor: float = .8
    lr_patience: int = 30
    lr_threshold: float = .0005
    min_lr: float = 1e-5
    seed: int = 42
    device: str = 'auto'
    experiment: str = 'piano-moonbeam'
    run_name: str | None = None
    resume: str | None = None
    data_from: str | None = None
    midi_records: str | None = None
    precision: str = 'fp32'
    gradient_checkpointing: bool = False
    warmup_steps: int = 0
    early_stopping_patience: int = 0  # Scheduled validation checks; zero disables.
    early_stopping_min_delta: float = .0005

    def __post_init__(self):
        super().__post_init__()
        for name in ('steps', 'batch_size', 'micro_batch_size', 'max_performances',
                     'num_compositions', 'performances_per_composition', 'max_scan',
                     'eval_every', 'eval_batches', 'log_every', 'checkpoint_every'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.micro_batch_size > self.batch_size:
            raise ValueError('micro_batch_size must be <= batch_size')
        if not 0 <= self.pitch_shift <= 12 or self.lr_patience < 0:
            raise ValueError('Require pitch_shift in 0..12 and lr_patience >= 0')
        if not 0 < self.validation_fraction < 1 or not 0 < self.lr_factor < 1:
            raise ValueError('validation_fraction and lr_factor must be between 0 and 1')
        for name in ('learning_rate', 'weight_decay', 'lr_threshold', 'min_lr', 'early_stopping_min_delta'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < self.min_lr <= self.learning_rate:
            raise ValueError('Require 0 < min_lr <= learning_rate')
        if self.num_compositions < 2 or min(self.max_performances, self.max_scan) < self.num_compositions:
            raise ValueError('Need at least two compositions and sufficient performance/scan budgets')
        if self.resume and self.data_from:
            raise ValueError('Use either resume or data_from')
        if self.device not in ('auto', 'cpu', 'mps', 'cuda'):
            raise ValueError('Unknown device')
        if self.precision not in ('fp32', 'bf16') or self.warmup_steps < 0:
            raise ValueError('Require precision fp32/bf16 and warmup_steps >= 0')
        if type(self.early_stopping_patience) is not int or self.early_stopping_patience < 0:
            raise ValueError('early_stopping_patience must be a nonnegative integer')
        if self.validation_sampling not in ('random', 'coverage'):
            raise ValueError('validation_sampling must be random or coverage')

    def model_config(self):
        return MoonbeamConfig(**{f.name: getattr(self, f.name) for f in fields(MoonbeamConfig)})


def load_training_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != 'moonbeam_training_v1':
        raise ValueError('Resume needs Moonbeam last.pt, not baseline or model-only weights')
    return saved


def load_data_source(path):
    return json.loads(Path(path).read_text()) if Path(path).suffix == '.json' else torch.load(
        path, map_location='cpu', weights_only=True)


def resolve_device(config):
    device = torch.device(config.device if config.device != 'auto' else
                          'cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use a CUDA-enabled PyTorch install on the rented GPU')
    if config.precision == 'bf16':
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('BF16 training is supported here on CUDA/CPU; use fp32 on MPS')
        if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support BF16; use --precision fp32')
    return device


def fingerprint(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


class NoteWindows:
    """Uniform performance sampling, then uniform window sampling; no piece joins."""
    def __init__(self, rows, config):
        self.sequences, self.performances = [], []
        self.context_length = config.context_length
        tokenizer = NoteTokenizer(config)
        for row in rows:
            try:
                sequence = tokenizer.encode(Score.from_midi(row['performance_midi_bytes']))
            except ValueError as error:
                raise ValueError(f"Performance {row['id']}: {error}. Adjust --max-duration/--max-delta if needed; no records were silently dropped.") from error
            self.sequences.append(sequence)
            self.performances.append({'id': str(row['id']), 'composition_key': list(composition_key(row)),
                                      'notes': len(sequence) - 2, 'note_sha256': fingerprint(sequence),
                                      'midi_sha256': hashlib.sha256(row['performance_midi_bytes']).hexdigest()})
        if not self.sequences:
            raise ValueError('Each split needs at least one performance')
        self.window_counts = [max(1, len(s) - config.context_length + 1) for s in self.sequences]
        self.window_ends = list(accumulate(self.window_counts))
        self.window_starts = [0, *self.window_ends[:-1]]

    def __getitem__(self, index):
        i = bisect_right(self.window_ends, index)
        start = index - self.window_starts[i]
        sequence = self.sequences[i][start:start + self.context_length]
        result = torch.full((self.context_length, 6), PAD, dtype=torch.long)
        result[:len(sequence)] = sequence
        return result

    def manifest(self):
        return {'performances': self.performances, 'windows': self.window_ends[-1]}


def build_data(rows, config, source=None):
    rows = list(rows)
    by_id = {str(row['id']): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate performance IDs')
    if source:
        groups = []
        expected = [p['id'] for name in ('train', 'validation') for p in source['split'][name]['performances']]
        if len(set(expected)) != len(expected) or set(expected) != set(by_id):
            raise ValueError('Data IDs differ from the saved split')
        for name in ('train', 'validation'):
            selected = []
            for p in source['split'][name]['performances']:
                row = by_id[p['id']]
                if list(composition_key(row)) != list(p['composition_key']):
                    raise ValueError(f"Composition changed for {p['id']}")
                selected.append(row)
            groups.append(selected)
    else:
        groups = split_by_composition(rows, config.validation_fraction, config.seed)
    if {composition_key(r) for r in groups[0]} & {composition_key(r) for r in groups[1]}:
        raise ValueError('Training and validation compositions overlap')
    datasets = tuple(NoteWindows(group, config.model_config()) for group in groups)
    manifest = {name: data.manifest() for name, data in zip(('train', 'validation'), datasets)}
    if source and source.get('format') == 'moonbeam_training_v1':
        for name in manifest:
            actual, previous = manifest[name]['performances'], source['split'][name]['performances']
            changed_midi = any(a['midi_sha256'] != b['midi_sha256'] for a, b in zip(actual, previous))
            if changed_midi or (config.resume and actual != previous):
                raise ValueError(f'{name} MIDI/encoding changed since the saved run')
    return (*datasets, manifest)


def augment(batch, maximum, generator):
    batch = batch.clone()
    for sequence in batch:
        valid = sequence[:, 0] >= 0
        pitches = sequence[valid, 2] * 12 + sequence[valid, 3]
        if len(pitches):
            low, high = max(-maximum, -int(pitches.min())), min(maximum, 127 - int(pitches.max()))
            shift = int(torch.randint(low, high + 1, (), generator=generator))
            pitches = pitches + shift
            sequence[valid, 2], sequence[valid, 3] = pitches // 12, pitches % 12
    return batch


def validation_indices(data, config):
    """Freeze random windows or cover every performance with evenly spaced starts."""
    if config.validation_sampling == 'random':
        return list(RandomWindowBatchSampler(data, config.eval_batches, config.batch_size, config.seed + 1))
    budget = config.eval_batches * config.batch_size
    performances = len(data.sequences)
    if budget < performances:
        raise ValueError('Coverage validation needs eval_batches * batch_size >= validation performances')
    indices = []
    for i, count in enumerate(data.window_counts):
        samples = budget // performances + int(i < budget % performances)
        for j in range(samples):
            offset = j * (count - 1) // (samples - 1) if samples > 1 else (count - 1) // 2
            indices.append(data.window_starts[i] + offset)
    return [indices[i:i + config.batch_size] for i in range(0, budget, config.batch_size)]


def autocast_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


@torch.no_grad()
def evaluate(model, batches, device, micro_batch_size, precision='fp32'):
    model.eval()
    total_loss, total, correct, note_count, exact = 0., 0, 0, 0, 0
    field_correct = torch.zeros(6, dtype=torch.long)
    field_loss = torch.zeros(6, dtype=torch.float64)
    for batch in batches:
        for part in batch.split(micro_batch_size):
            part = part.to(device)
            with autocast_context(device, precision):
                result = model(part, return_logits=True)
            labels, logits = result['labels'], result['logits']
            valid = labels != -100
            n = int(valid.sum())
            total_loss += float(result['loss']) * n
            total += n
            matches = logits.argmax(-1) == labels
            correct += int((matches & valid).sum())
            normal = part[:, 1:, 0] >= 0
            note_count += int(normal.sum())
            exact += int((matches.all(-1) & normal).sum())
            losses = F.cross_entropy(logits.flatten(0, 2).float(), labels.flatten(),
                                     ignore_index=-100, reduction='none').reshape_as(labels)
            field_correct += (matches & normal[..., None]).sum((0, 1)).cpu()
            field_loss += (losses * normal[..., None]).sum((0, 1)).cpu().double()
    if not math.isfinite(total_loss) or not note_count:
        raise FloatingPointError('Validation has non-finite loss or no note targets')
    metrics = {'val_loss': total_loss / total, 'val_attribute_accuracy': correct / total,
               'val_note_accuracy_teacher_forced': exact / note_count}
    names = ('delta_onset', 'duration', 'octave', 'pitch_class', 'instrument', 'velocity')
    for j, field in enumerate(ORDERS[model.config.attribute_order]):
        metrics[f'val_{names[field]}_accuracy'] = int(field_correct[j]) / note_count
        metrics[f'val_{names[field]}_loss'] = float(field_loss[j]) / note_count
    # Piano instrument is nearly constant; expose a score without that easy field.
    metrics['val_music_attribute_accuracy'] = sum(metrics[f'val_{name}_accuracy']
                                                  for name in names if name != 'instrument') / 5
    return metrics


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def train(config, rows=None, *, tracking_uri=TRACKING_URI, artifact_root=ARTIFACT_ROOT,
          checkpoint_root=ROOT / 'checkpoints' / 'moonbeam'):
    """Run training; optional MIDI rows and stores permit isolated offline checks."""
    saved = load_training_checkpoint(config.resume) if config.resume else None
    source = saved or (load_data_source(config.data_from) if config.data_from else None)
    if source and 'split' not in source:
        raise ValueError('data_from requires a training checkpoint with a saved split')
    start = saved['step'] if saved else 0
    if config.steps <= start:
        raise ValueError('--steps is the TOTAL target and must exceed the saved step')
    if saved and config.early_stopping_patience and saved.get('stopped_early', False):
        raise ValueError('This run already reached early stopping; start a fresh experiment')
    if saved:
        mutable = {'steps', 'device', 'experiment', 'run_name', 'resume', 'data_from',
                   'log_every', 'checkpoint_every', 'midi_records',
                   'micro_batch_size', 'gradient_checkpointing'}
        for key, value in asdict(config).items():
            old_value = saved['train_config'].get(key, getattr(TrainConfig(), key))
            if key not in mutable and value != old_value:
                raise ValueError(f'Cannot change {key} during resume')
    device = resolve_device(config)
    torch.manual_seed(config.seed)
    configure_tracking(config.experiment, tracking_uri, artifact_root)
    with mlflow.start_run(run_name=config.run_name) as run:
        run_id = run.info.run_id
        print(f'Run: {run_id} | device: {device} | target: {config.steps} updates', flush=True)
        mlflow.log_params({k: v for k, v in asdict(config).items() if v is not None})
        mlflow.set_tags({'architecture': 'moonbeam', 'representation': 'complete_notes_v1',
                         'data_source': 'local_midi_records' if config.midi_records else
                                        'SyMuPe/PianoCoRe' if rows is None else 'provided_rows',
                         'split_unit': 'composition', 'resolved_device': str(device),
                         'initialization': 'resume' if saved else 'fresh'})
        if device.type == 'cuda':
            mlflow.log_dict({'gpu': torch.cuda.get_device_name(device), 'cuda': torch.version.cuda,
                             'cudnn': torch.backends.cudnn.version(),
                             'memory_bytes': torch.cuda.get_device_properties(device).total_memory}, 'gpu.json')
        if source:
            mlflow.set_tag('resumed_from_run_id' if saved else 'data_from_run_id', source.get('mlflow_run_id', 'unknown'))
        if saved:
            execution_changes = {
                name: {'previous': saved['train_config'].get(name, getattr(TrainConfig(), name)),
                       'current': getattr(config, name)}
                for name in ('micro_batch_size', 'gradient_checkpointing')
                if saved['train_config'].get(name, getattr(TrainConfig(), name)) != getattr(config, name)
            }
            if execution_changes:
                mlflow.log_dict(execution_changes, 'execution_changes.json')
                print('Resuming with changed execution settings; effective batch and optimizer state are preserved. '
                      'Floating-point results may differ slightly.', flush=True)
        mlflow.log_dict({'python': platform.python_version(), **{
            name: version(name) for name in ('torch', 'mlflow', 'datasets', 'symusic')}}, 'environment.json')
        paths = [ROOT / name for name in ('dataset.py', 'tokenizer.py', 'requirements.txt')]
        paths += list((ROOT / 'moonbeam').glob('*.py')) + [ROOT / 'moonbeam' / 'README.md']
        paths += [ROOT / 'support' / name for name in ('__init__.py', 'mlflow_local.py', 'training_state.py')]
        for path in paths:
            mlflow.log_artifact(str(path), str(Path('source') / path.relative_to(ROOT).parent))
        print('Loading MIDI and building complete-note windows...', flush=True)
        if rows is not None and config.midi_records:
            raise ValueError('Supply either rows or midi_records, not both')
        if config.midi_records:
            rows = torch.load(config.midi_records, map_location='cpu', weights_only=True)
        if rows is None:
            if source:
                rows = load_performances_by_ids([p['id'] for name in ('train', 'validation')
                                                for p in source['split'][name]['performances']])
            else:
                rows = load_performances(config.max_performances, config.num_compositions,
                                         config.performances_per_composition, config.max_scan, config.seed)
        train_data, validation_data, manifest = build_data(rows, config, source)
        mlflow.log_dict(manifest, 'split.json')
        for name, data in (('train', train_data), ('validation', validation_data)):
            count = len({tuple(p['composition_key']) for p in data.performances})
            mlflow.log_params({f'{name}_performances': len(data.sequences), f'{name}_compositions': count,
                               f'{name}_windows': data.window_ends[-1]})
            print(f'{name}: {len(data.sequences)} performances, {count} compositions', flush=True)
        sampler = RandomWindowBatchSampler(train_data, config.steps - start, config.batch_size, config.seed)
        augmentation_rng = torch.Generator().manual_seed(config.seed + 2)
        val_indices = validation_indices(validation_data, config)
        validation = torch.stack([torch.stack([validation_data[i] for i in indices]) for indices in val_indices])
        sample_manifest = []
        for batch in val_indices:
            for index in batch:
                performance = bisect_right(validation_data.window_ends, index)
                sample_manifest.append({'performance_id': validation_data.performances[performance]['id'],
                                        'start': index - validation_data.window_starts[performance]})
        mlflow.log_dict({'sampling': config.validation_sampling, 'windows': sample_manifest}, 'validation_sampling.json')
        mlflow.log_param('validation_sampled_performances', len({p['performance_id'] for p in sample_manifest}))
        validation_hash = fingerprint(validation)
        if saved:
            if validation_hash != saved['validation_sha256']:
                raise ValueError('Validation windows differ from checkpoint')
            sampler.generator.set_state(saved['sampler_rng_state'])
            augmentation_rng.set_state(saved['augmentation_rng_state'])
        elif source and source.get('validation_sha256') and source.get('format') == 'moonbeam_training_v1':
            # Comparison runs keep the source's exact fixed sample when window settings match.
            same_windows = all(getattr(config, key) == source['train_config'][key]
                               for key in ('context_length', 'batch_size', 'eval_batches', 'seed', 'time_step_ms'))
            same_windows = same_windows and config.validation_sampling == source['train_config'].get('validation_sampling', 'random')
            if same_windows and validation_hash != source['validation_sha256']:
                raise ValueError('Comparison validation windows differ from source')
        mlflow.log_param('validation_sha256', validation_hash)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'windows.pt'
            torch.save(validation, path)
            mlflow.log_artifact(str(path), 'validation')
        mlflow.log_input(mlflow.data.from_numpy(validation.numpy(), name='moonbeam-fixed-note-windows',
                                              digest=validation_hash[:32]), context='validation')
        model = MoonbeamModel(config.model_config()).to(device)
        model.gradient_checkpointing = config.gradient_checkpointing
        mlflow.log_dict(asdict(model.config), 'model_config.json')
        mlflow.log_param('parameters', sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                      fused=device.type == 'cuda')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.lr_factor, patience=config.lr_patience, threshold=config.lr_threshold,
            threshold_mode='abs', min_lr=config.min_lr)
        if saved:
            model.load_state_dict(saved['model_state_dict'])
            optimizer.load_state_dict(saved['optimizer_state_dict'])
            scheduler.load_state_dict(saved['scheduler_state_dict'])
        output = Path(checkpoint_root) / run_id
        output.mkdir(parents=True, exist_ok=False)
        metrics = evaluate(model, validation, device, config.micro_batch_size, config.precision)
        if saved:
            best_loss, best_step, best_weights = saved['best_loss'], saved['best_step'], saved['best_model_state_dict']
            restore_rng(saved['torch_rng_state'], device)
        else:
            if not config.warmup_steps:
                scheduler.step(metrics['val_loss'])
            best_loss, best_step = metrics['val_loss'], 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mlflow.log_metrics({**metrics, 'learning_rate': optimizer.param_groups[0]['lr']}, step=start)
        early_stopping_best = saved.get('early_stopping_best', best_loss) if saved else best_loss
        early_stopping_bad_evals = saved.get('early_stopping_bad_evals', 0) if saved else 0
        stopped_early = False

        def save_best():
            atomic_save({'format': 'moonbeam_local_v1', 'config': asdict(model.config),
                         'model_state_dict': best_weights, 'step': best_step, 'val_loss': best_loss}, output / 'best.pt')
            mlflow.log_artifact(str(output / 'best.pt'), 'checkpoints')

        def save_last(step, validation_step):
            atomic_save({'format': 'moonbeam_training_v1', 'config': asdict(model.config),
                         'train_config': asdict(config), 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                         'step': step, 'validation_step': validation_step, 'val_loss': metrics['val_loss'],
                         'best_loss': best_loss, 'best_step': best_step, 'best_model_state_dict': best_weights,
                         'early_stopping_best': early_stopping_best,
                         'early_stopping_bad_evals': early_stopping_bad_evals, 'stopped_early': stopped_early,
                         'split': manifest, 'validation_sha256': validation_hash,
                         'sampler_rng_state': sampler.generator.get_state(),
                         'augmentation_rng_state': augmentation_rng.get_state(), 'torch_rng_state': rng_state(device),
                         'mlflow_run_id': run_id}, output / 'last.pt')
            mlflow.log_artifact(str(output / 'last.pt'), 'checkpoints')

        save_best()
        validation_step = start
        save_last(start, validation_step)
        started = time.monotonic()
        loss_sum, target_sum = 0., 0
        for step, indices in enumerate(sampler, start + 1):
            model.train()
            if step <= config.warmup_steps:
                for group in optimizer.param_groups:
                    group['lr'] = config.learning_rate * step / config.warmup_steps
            batch = augment(torch.stack([train_data[i] for i in indices]), config.pitch_shift, augmentation_rng)
            targets = int((batch[:, 1:, 0] != PAD).sum()) * 6
            optimizer.zero_grad(set_to_none=True)
            for part in batch.split(config.micro_batch_size):
                count = int((part[:, 1:, 0] != PAD).sum()) * 6
                with autocast_context(device, config.precision):
                    loss = model(part.to(device))['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                (loss * (count / targets)).backward()
                loss_sum += float(loss.detach()) * count
                target_sum += count
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % config.log_every == 0 or step == config.steps:
                mlflow.log_metrics({'train_loss': loss_sum / target_sum, 'gradient_norm': float(grad_norm),
                                    'learning_rate': optimizer.param_groups[0]['lr']}, step=step)
                print(f'Step {step} | train_loss={loss_sum / target_sum:.4f}', flush=True)
                loss_sum, target_sum = 0., 0
            if step % config.eval_every == 0 or step == config.steps:
                metrics = evaluate(model, validation, device, config.micro_batch_size, config.precision)
                validation_step = step
                # An off-cadence final evaluation must not alter the resume schedule.
                if step % config.eval_every == 0 and step >= config.warmup_steps:
                    scheduler.step(metrics['val_loss'])
                mlflow.log_metrics({**metrics, 'learning_rate': optimizer.param_groups[0]['lr']}, step=step)
                print(f"Step {step} | val_loss={metrics['val_loss']:.4f} | attribute_accuracy={metrics['val_attribute_accuracy']:.1%}", flush=True)
                if metrics['val_loss'] < best_loss:
                    best_loss, best_step = metrics['val_loss'], step
                    best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    save_best()
                # Match scheduler cadence so an off-cadence stop/resume does not consume patience.
                if config.early_stopping_patience and step % config.eval_every == 0 and step >= config.warmup_steps:
                    if metrics['val_loss'] < early_stopping_best - config.early_stopping_min_delta:
                        early_stopping_best = metrics['val_loss']
                        early_stopping_bad_evals = 0
                    else:
                        early_stopping_bad_evals += 1
                    stopped_early = early_stopping_bad_evals >= config.early_stopping_patience
                    mlflow.log_metric('early_stopping_bad_evals', early_stopping_bad_evals, step=step)
            if step % config.checkpoint_every == 0 or step == config.steps or stopped_early:
                save_last(step, validation_step)
            if stopped_early:
                print(f'Early stopping at step {step}: no qualifying validation-loss improvement for '
                      f'{early_stopping_bad_evals} checks. Best step: {best_step}; loss: {best_loss:.4f}', flush=True)
                break
        mlflow.log_metrics({'best_val_loss': best_loss, 'best_step': best_step,
                            'completed_steps': step, 'stopped_early': int(stopped_early),
                            'training_seconds': time.monotonic() - started}, step=step)
        mlflow.set_tag('stop_reason', 'early_stopping' if stopped_early else 'step_limit')
        print(f'Finished. Checkpoints: {output}', flush=True)
    return run_id


def preflight(config):
    """Check the selected architecture/device at full context without starting a run."""
    device = resolve_device(config)
    torch.manual_seed(config.seed)
    model = MoonbeamModel(config.model_config()).to(device)
    model.gradient_checkpointing = config.gradient_checkpointing
    notes = torch.zeros(config.micro_batch_size, config.context_length, 6, dtype=torch.long, device=device)
    notes[..., 0] = torch.arange(config.context_length, device=device)
    notes[..., 1], notes[..., 2], notes[..., 5] = min(50, config.max_duration), 5, 80
    notes[:, 0] = -1
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=device.type == 'cuda')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config.precision):
            loss = model(notes)['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite preflight loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()),
                      'device': str(device), 'precision': config.precision,
                      'micro_batch_size': config.micro_batch_size, 'context_length': config.context_length,
                      'seconds_per_microbatch_update': (time.monotonic() - started) / 3,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
                      'loss': float(loss.detach())}, indent=2))


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument('--preset', choices=tuple(PRESETS), help='Fresh-run architecture and CUDA settings; explicit flags override')
    parser.add_argument('--preflight', action='store_true', help='Run three synthetic full-context updates; no MLflow run or saved weights')
    for field in fields(TrainConfig):
        name, default = field.name, getattr(defaults, field.name)
        kwargs = {'default': argparse.SUPPRESS, 'type': type(default) if default is not None else str}
        if isinstance(default, bool):
            kwargs = {'default': argparse.SUPPRESS, 'action': argparse.BooleanOptionalAction}
        if name == 'device':
            kwargs['choices'] = ('auto', 'cpu', 'mps', 'cuda')
        if name == 'attribute_order':
            kwargs['choices'] = tuple(ORDERS)
        if name == 'precision':
            kwargs['choices'] = ('fp32', 'bf16')
        if name == 'validation_sampling':
            kwargs['choices'] = ('random', 'coverage')
        if name == 'steps':
            kwargs['help'] = 'Maximum total optimizer updates, including those already completed on resume'
        if name == 'early_stopping_patience':
            kwargs['help'] = 'Stop after this many scheduled validation checks without improvement; 0 disables'
        if name == 'data_from':
            kwargs['help'] = 'Reuse a baseline/Moonbeam checkpoint split with fresh weights; training settings are not inherited'
        if name == 'resume':
            kwargs['help'] = 'Moonbeam last.pt; saved settings are inherited; requires explicit --steps'
        parser.add_argument('--' + name.replace('_', '-'), **kwargs)
    values = vars(parser.parse_args(argv))
    preset = values.pop('preset')
    check_only = values.pop('preflight')
    try:
        if preset:
            if values.get('resume'):
                parser.error('--preset is for fresh models; resume already inherits the architecture')
            values = {**PRESETS[preset], **GPU_DEFAULTS, **values}
        if values.get('resume'):
            if 'steps' not in values:
                parser.error('--resume requires --steps TOTAL')
            saved = load_training_checkpoint(values['resume'])
            inherited = dict(saved['train_config'])
            inherited.update(resume=values['resume'], data_from=None, run_name=None, device='auto')
            inherited.update(values)
            values = inherited
        config = TrainConfig(**values)
        if check_only:
            preflight(config)
            parser.exit()
        return config
    except (ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    train(parse_config())
