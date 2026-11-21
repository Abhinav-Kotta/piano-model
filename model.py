import torch
import torch.nn as nn
from torch.nn import functional as F


class MusicEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=128, context_length=256, position_encoding="learned"):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        # Consume the same initialization RNG in both variants, so their shared
        # trunk starts identically in paired experiments.
        if position_encoding == "rope":
            self.position_embedding = None

    def forward(self, token_ids, position_offset=0):
        # token_ids shape: [batch_size, sequence_length]
        batch_size, sequence_length = token_ids.shape

        if position_offset < 0 or sequence_length + position_offset > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        positions = torch.arange(
            position_offset, sequence_length + position_offset,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(token_ids)
        if self.position_embedding is None:
            return token_vectors
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors

def apply_rope(x, position_offset=0):
    """Rotate adjacent query/key coordinate pairs; positions are token indices."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    frequency = 10000.0 ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    positions = torch.arange(position_offset, position_offset + x.shape[-2],
                             device=x.device, dtype=torch.float32)
    angles = positions[:, None] * frequency[None, :]
    cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4, position_encoding="learned"):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.position_encoding = position_encoding
        if position_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x, position_offset=0):
        batch_size, sequence_length, d_model = x.shape

        # Each projection has shape [batch, length, d_model].
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split into heads: [batch, num_heads, length, head_dim].
        q = q.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if self.position_encoding == "rope":
            q, k = apply_rope(q, position_offset), apply_rope(k, position_offset)

        # Compare every query with every key.
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # True means this position is in the future and must be blocked.
        future_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(future_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        # Weighted combination of value vectors.
        attended = weights @ v

        # Combine heads back into [batch, length, d_model].
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, d_model
        )

        return self.output_projection(attended)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_ff=512, position_encoding="learned"):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, position_encoding)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, position_offset=0):
        x = x + self.attention(self.norm1(x), position_offset)
        x = x + self.mlp(self.norm2(x))
        return x


class TypeValueHead(nn.Module):
    """p(token | history) = p(type | history) p(value | type, history).

    Each type has a separate value head. No target type is fed to the model;
    inference ranks all joint token probabilities in the original ID order.
    """
    def __init__(self, flat_head, token_type_ids):
        super().__init__()
        if len(token_type_ids) != flat_head.out_features or set(token_type_ids) != set(range(7)):
            raise ValueError("Type/value output needs a complete seven-family vocabulary")
        self.type_head = nn.Linear(flat_head.in_features, 7)
        nn.init.zeros_(self.type_head.weight)
        nn.init.zeros_(self.type_head.bias)
        ids = [i for group in range(7) for i, g in enumerate(token_type_ids) if g == group]
        self.register_buffer("token_order", torch.tensor(ids).argsort())
        self.value_heads = nn.ModuleList()
        for group in range(7):
            indices = [i for i, g in enumerate(token_type_ids) if g == group]
            head = nn.Linear(flat_head.in_features, len(indices))
            with torch.no_grad():
                head.weight.copy_(flat_head.weight[indices])
                head.bias.copy_(flat_head.bias[indices])
            self.value_heads.append(head)

    def forward(self, x):
        types = F.log_softmax(self.type_head(x), dim=-1)
        values = [types[..., i:i + 1] + F.log_softmax(head(x), dim=-1)
                  for i, head in enumerate(self.value_heads)]
        return torch.cat(values, dim=-1).index_select(-1, self.token_order)


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        context_length=256,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        output_mode="flat",
        position_encoding="learned",
        token_type_ids=None,
    ):
        super().__init__()
        if output_mode not in ("flat", "type_value") or position_encoding not in ("learned", "rope"):
            raise ValueError("Unknown output mode or positional encoding")

        self.embedding = MusicEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            context_length=context_length,
            position_encoding=position_encoding,
        )

        # Each block gets its own independent parameters.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                position_encoding=position_encoding,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        if output_mode == "type_value":
            if token_type_ids is None:
                raise ValueError("type_value requires token_type_ids from the saved vocabulary")
            with torch.random.fork_rng(devices=[]):
                self.output_head = TypeValueHead(self.output_head, token_type_ids)

    def forward(self, token_ids, position_offset=0):
        x = self.embedding(token_ids, position_offset)

        for block in self.blocks:
            x = block(x, position_offset)

        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits
