"""Note-level causal Transformer with FME, musical relative attention, and a GRU.

Independent implementation of the core ideas in arXiv:2505.15559. This is not
checkpoint-compatible with the authors' LLaMA fork or our event-token model.
See README.md for the local choices and tensor contract.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


BOS, EOS, PAD = -1, -2, -3
FIELDS = ('onset', 'duration', 'octave', 'pitch_class', 'instrument', 'velocity')
BASES = (199999., 1031., 19., 20., 199999., 131.)
ORDERS = {'velocity_first': (0, 1, 5, 2, 3, 4), 'paper': (0, 1, 2, 3, 4, 5)}


@dataclass(frozen=True)
class MoonbeamConfig:
    d_model: int = 192
    d_ff: int = 512
    num_layers: int = 4
    num_heads: int = 6
    num_kv_heads: int = 6
    context_length: int = 512  # Complete notes, plus special rows.
    decoder_dim: int = 192
    decoder_layers: int = 1
    time_step_ms: int = 10
    max_delta: int = 1024
    max_duration: int = 1024
    attribute_order: str = 'velocity_first'
    norm_eps: float = 1e-5
    dropout: float = 0.

    def __post_init__(self):
        for name in ('d_model', 'd_ff', 'num_layers', 'num_heads', 'num_kv_heads',
                     'context_length', 'decoder_dim', 'decoder_layers', 'time_step_ms',
                     'max_delta', 'max_duration'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.context_length < 2:
            raise ValueError('context_length must be at least 2')
        if self.num_heads % 6 or self.num_kv_heads % 6 or self.num_heads % self.num_kv_heads:
            raise ValueError('Attention requires six equal musical head groups; Q heads must be a multiple of KV heads')
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2 or self.d_model % 12:
            raise ValueError('Head dimensions and each of the six attribute embedding dimensions must be even')
        if self.attribute_order not in ORDERS:
            raise ValueError(f'attribute_order must be one of {tuple(ORDERS)}')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be finite and in [0, 1)')

    @property
    def value_sizes(self):
        return (self.max_delta + 1, self.max_duration + 1, 11, 12, 129, 128)


def validate_notes(notes, config):
    """Validate absolute-onset rows, with optional BOS/EOS and right-side PAD."""
    if notes.dtype != torch.long or notes.ndim != 3 or notes.shape[-1] != 6:
        raise ValueError('notes must be int64 [batch, length, 6]')
    if not notes.shape[0] or not notes.shape[1]:
        raise ValueError('notes must contain at least one row per sequence')
    first = notes[..., 0]
    normal = first >= 0
    special = (first == BOS) | (first == EOS) | (first == PAD)
    if not (normal | special).all() or (special & (notes != first.unsqueeze(-1)).any(-1)).any():
        raise ValueError('Special rows must contain six identical BOS, EOS, or PAD values')
    valid = first != PAD
    if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError('Padding must be on the right; empty sequences are unsupported')
    if (first[:, 1:] == BOS).any() or ((first == EOS).sum(1) > 1).any():
        raise ValueError('BOS is only allowed first and EOS is only allowed once')
    if (normal & ((first == EOS).cumsum(1) > 0)).any():
        raise ValueError('Notes cannot follow EOS')
    for i, size in enumerate(config.value_sizes[1:], 1):
        minimum = 1 if i in (1, 5) else 0
        if (normal & ((notes[..., i] < minimum) | (notes[..., i] >= size))).any():
            raise ValueError(f'{FIELDS[i]} outside supported range [{minimum}, {size - 1}]')
    if (normal & (notes[..., 2] * 12 + notes[..., 3] > 127)).any():
        raise ValueError('Octave and pitch class must describe a MIDI pitch from 0 to 127')
    if ((normal[:, 1:] & normal[:, :-1]) & (first[:, 1:] < first[:, :-1])).any():
        raise ValueError('Notes must be sorted by onset')
    return valid


class FundamentalMusicEmbedding(nn.Module):
    """Numerical attribute -> paired sin/cos features -> learned affine map."""
    def __init__(self, dim, base):
        super().__init__()
        self.register_buffer('frequencies', base ** (-torch.arange(0, dim, 2).float() / dim))
        self.translation = nn.Parameter(torch.zeros(dim))
        self.projection = nn.Linear(dim, dim)

    def forward(self, values):
        angles = values.float().unsqueeze(-1) * self.frequencies.float()
        waves = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        return self.projection(waves.to(self.projection.weight.dtype) + self.translation)


class NoteEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.d_model // 6
        self.attributes = nn.ModuleList([
            nn.Embedding(129, dim) if i == 4 else FundamentalMusicEmbedding(dim, BASES[i])
            for i in range(6)])
        self.special = nn.Embedding(3, config.d_model, padding_idx=2)

    def forward(self, notes):
        values = notes.clamp_min(0)
        embedded = torch.cat([layer(values[..., i]) for i, layer in enumerate(self.attributes)], -1)
        special_ids = (-notes[..., 0] - 1).clamp(0, 2)
        return torch.where((notes[..., 0] < 0).unsqueeze(-1), self.special(special_ids), embedded)


class MusicalRelativeAttention(nn.Module):
    """Rotate six head groups by onset, duration, octave, pitch class, onset, velocity."""
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads = config.num_heads, config.num_kv_heads
        self.head_dim = config.d_model // self.heads
        self.q = nn.Linear(config.d_model, self.heads * self.head_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        powers = torch.arange(0, self.head_dim, 2).float() / self.head_dim
        self.register_buffer('frequencies', torch.tensor(BASES)[:, None] ** (-powers))

    def rotate(self, x, notes):
        batch, heads, length, dim = x.shape
        coordinates = notes[..., (0, 1, 2, 3, 0, 5)].clamp_min(0).float().transpose(1, 2)
        angles = coordinates.unsqueeze(-1) * self.frequencies.float()[None, :, None, :]
        cosine, sine = angles.cos().unsqueeze(2).to(x.dtype), angles.sin().unsqueeze(2).to(x.dtype)
        grouped = x.reshape(batch, 6, heads // 6, length, dim)
        a, b = grouped[..., :dim // 2], grouped[..., dim // 2:]
        return torch.cat((a * cosine - b * sine, b * cosine + a * sine), -1).reshape_as(x)

    def forward(self, x, notes, allowed):
        batch, length, _ = x.shape
        def split(layer, heads):
            return layer(x).reshape(batch, length, heads, self.head_dim).transpose(1, 2)
        q = self.rotate(split(self.q, self.heads), notes)
        k = self.rotate(split(self.k, self.kv_heads), notes)
        v = split(self.v, self.kv_heads)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
        return self.out(attended.transpose(1, 2).reshape(batch, length, -1))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = MusicalRelativeAttention(config)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, notes, allowed):
        x = x + self.dropout(self.attention(self.norm1(x), notes, allowed))
        normalized = self.norm2(x)
        return x + self.dropout(self.down(F.silu(self.gate(normalized)) * self.up(normalized)))


class AttributeDecoder(nn.Module):
    """Shared output vocabulary; teacher forcing sees only earlier attributes."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.order = ORDERS[config.attribute_order]
        offsets, cursor = [], 0
        for size in config.value_sizes:
            offsets.append(cursor)
            cursor += size + 2  # Per-field BOS and EOS classes.
        self.register_buffer('offsets', torch.tensor(offsets))
        self.register_buffer('sizes', torch.tensor(config.value_sizes))
        self.start_id, self.vocab_size = cursor, cursor + 1
        self.embedding = nn.Embedding(self.vocab_size, config.decoder_dim)
        self.summary = nn.Linear(config.d_model, config.decoder_dim, bias=False)
        self.gru = nn.GRU(config.decoder_dim, config.decoder_dim, config.decoder_layers, batch_first=True)
        self.projection = nn.Linear(config.decoder_dim, config.decoder_dim)
        self.output = nn.Linear(config.decoder_dim, self.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def labels(self, previous, following):
        values = following.clone()
        normal = following[..., 0] >= 0
        delta = following[..., 0] - previous[..., 0].clamp_min(0)
        if (normal & ((delta < 0) | (delta > self.config.max_delta))).any():
            raise ValueError(f'Onset gap exceeds max_delta={self.config.max_delta}; increase the config or split the music')
        values[..., 0] = torch.where(normal, delta, following[..., 0])
        encoded = values + self.offsets
        encoded = torch.where(values == EOS, self.offsets + self.sizes + 1, encoded)
        encoded = torch.where(values == PAD, -100, encoded)
        return encoded[..., self.order]

    def initial_state(self, hidden):
        return self.summary(hidden).unsqueeze(0).expand(self.config.decoder_layers, -1, -1).contiguous()

    def forward(self, hidden, labels):
        leading = hidden.shape[:-1]
        flat_labels = labels.reshape(-1, 6)
        start = flat_labels.new_full((len(flat_labels), 1), self.start_id)
        inputs = torch.cat((start, flat_labels[:, :-1].clamp_min(0)), dim=1)
        # Keep the recurrent kernel in FP32 for CUDA/cuDNN portability under BF16 AMP.
        # The larger Transformer and output projections still use autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            outputs, _ = self.gru(self.dropout(self.embedding(inputs)), self.initial_state(
                hidden.reshape(-1, hidden.shape[-1]).float()))
        return self.output(self.dropout(self.projection(outputs))).reshape(*leading, 6, self.vocab_size)

    def sample(self, hidden, temperature=1., top_p=.95, generator=None, instrument=0):
        """Return relative-onset notes. EOS is chosen at the first attribute only."""
        batch = hidden.shape[0]
        state = self.initial_state(hidden)
        token = torch.full((batch, 1), self.start_id, device=hidden.device, dtype=torch.long)
        values = token.new_zeros((batch, 6))
        finished = token.new_zeros(batch, dtype=torch.bool)
        for field in self.order:
            decoded, state = self.gru(self.embedding(token), state)
            logits = self.output(self.projection(decoded[:, 0])).float()
            offset, size = int(self.offsets[field]), int(self.sizes[field])
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            allowed[:, offset:offset + size] = True
            if field == 0:
                allowed[:, offset + size + 1] = True
            if field in (1, 5):
                allowed[:, offset] = False  # Positive duration and velocity.
            if field == 3:
                pitches = values[:, 2:3] * 12 + torch.arange(12, device=hidden.device)
                allowed[:, offset:offset + size] &= pitches <= 127
            if field == 4 and instrument is not None:
                allowed[:, offset:offset + size] = False
                allowed[:, offset + instrument] = True
            logits = logits.masked_fill(~allowed, -torch.inf)
            if temperature == 0:
                selected = logits.argmax(-1)
            else:
                probabilities = (logits / temperature).softmax(-1)
                probabilities, indices = probabilities.sort(dim=-1, descending=True)
                discard = probabilities.cumsum(-1) - probabilities >= top_p
                probabilities = probabilities.masked_fill(discard, 0.)
                # CPU sampling supports a reproducible CPU generator on MPS too.
                draw = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(hidden.device)
                selected = indices.gather(-1, draw).squeeze(-1)
            values[:, field] = selected - offset
            if field == 0:
                finished = selected == offset + size + 1
            token = selected[:, None]
        values[finished] = EOS
        return values


class MoonbeamModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MoonbeamConfig()
        self.embedding = NoteEmbedding(self.config)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.num_layers)])
        self.norm = nn.RMSNorm(self.config.d_model, eps=self.config.norm_eps)
        self.decoder = AttributeDecoder(self.config)
        self.gradient_checkpointing = False

    def encode(self, notes):
        valid = validate_notes(notes, self.config)
        length = notes.shape[1]
        if length > self.config.context_length:
            raise ValueError(f'Input exceeds {self.config.context_length} note positions')
        causal = torch.ones(length, length, dtype=torch.bool, device=notes.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        hidden = self.embedding_dropout(self.embedding(notes))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, notes, allowed, use_reentrant=False)
            else:
                hidden = block(hidden, notes, allowed)
        return self.norm(hidden).masked_fill(~valid.unsqueeze(-1), 0.)

    def forward(self, notes, *, return_logits=False):
        """Train on whole sequences; shift note targets internally to prevent leakage."""
        if notes.ndim != 3 or notes.shape[1] < 2:
            raise ValueError('Training needs at least two rows: history and a next-note target')
        hidden = self.encode(notes)
        labels = self.decoder.labels(notes[:, :-1], notes[:, 1:])
        if not (labels != -100).any():
            raise ValueError('The batch has no non-padding targets')
        logits = self.decoder(hidden[:, :-1], labels)
        loss = F.cross_entropy(logits.flatten(0, 2).float(), labels.flatten(), ignore_index=-100)
        result = {'loss': loss}
        if return_logits:
            result.update(logits=logits, labels=labels)
        return result

    @torch.no_grad()
    def generate(self, prefix, max_new_notes=128, temperature=1., top_p=.95, generator=None, instrument=0):
        """Generate complete notes; input/output use absolute onsets and right padding."""
        valid = validate_notes(prefix, self.config)
        if type(max_new_notes) is not int or max_new_notes < 0:
            raise ValueError('max_new_notes must be a nonnegative integer')
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError('Require temperature >= 0 and 0 < top_p <= 1')
        if instrument is not None and (type(instrument) is not int or not 0 <= instrument <= 128):
            raise ValueError('instrument must be None or a MIDI program 0..127 (128 for drums)')
        sequences = [row[:int(mask.sum())].clone() for row, mask in zip(prefix, valid)]
        training = self.training
        self.eval()
        try:
            for _ in range(max_new_notes):
                active = [i for i, seq in enumerate(sequences) if int(seq[-1, 0]) != EOS]
                if not active:
                    break
                contexts = [sequences[i][-self.config.context_length:] for i in active]
                lengths = [len(c) for c in contexts]
                batch = prefix.new_full((len(active), max(lengths), 6), PAD)
                for i, context in enumerate(contexts):
                    batch[i, :len(context)] = context
                hidden = self.encode(batch)[torch.arange(len(active), device=prefix.device),
                                            torch.tensor(lengths, device=prefix.device) - 1]
                sampled = self.decoder.sample(hidden, temperature, top_p, generator, instrument)
                for i, index in enumerate(active):
                    note = sampled[i].clone()
                    if note[0] >= 0:
                        note[0] += sequences[index][-1, 0].clamp_min(0)
                    sequences[index] = torch.cat((sequences[index], note[None]), 0)
            result = prefix.new_full((len(sequences), max(map(len, sequences)), 6), PAD)
            for i, seq in enumerate(sequences):
                result[i, :len(seq)] = seq
            return result
        finally:
            self.train(training)

    def save(self, path):
        torch.save({'format': 'moonbeam_local_v1', 'config': asdict(self.config),
                    'model_state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('format') not in ('moonbeam_local_v1', 'moonbeam_training_v1'):
            raise ValueError('Expected a local Moonbeam architecture checkpoint')
        model = cls(MoonbeamConfig(**saved['config']))
        model.load_state_dict(saved['model_state_dict'])
        return model.to(device)


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--attribute-order', choices=tuple(ORDERS), default='velocity_first')
    parser.add_argument('--smoke', action='store_true', help='Run one synthetic optimizer step; no dataset training or files')
    args = parser.parse_args()
    model = MoonbeamModel(MoonbeamConfig(attribute_order=args.attribute_order)).to(args.device)
    print(json.dumps(asdict(model.config), indent=2))
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    if args.smoke:
        notes = torch.tensor([[[BOS] * 6, [0, 48, 5, 0, 0, 78],
                               [24, 48, 5, 4, 0, 82], [EOS] * 6]], device=args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss = model(notes)['loss']
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        print(f'Synthetic forward/backward/update passed; loss={loss.item():.4f}')


if __name__ == '__main__':
    main()
