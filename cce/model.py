"""A small decoder-only transformer language model with a big vocabulary.

The backbone is deliberately tiny so that the logits matrix dominates memory,
which is the regime Cut Cross-Entropy is about. The classifier head is a plain
``(V, D)`` matrix with no bias - the paper's ``C`` - and it is *not* tied to the
token embedding, so the pipeline is exactly

    hidden = backbone(tokens)          # (N, D)
    loss   = cross_entropy(hidden @ C^T, targets)

with ``hidden @ C^T`` the ``(N, V)`` matrix that the chunked path never builds.
A separate embedding of the same size as the head would add nothing to the
comparison and would muddle the parameter count.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn


class Config(NamedTuple):
    vocab_size: int = 65536
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_seq_len: int = 128
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        b, t, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        attn = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.head_dim, x.dtype))
        attn = attn + mx.where(mask, mx.array(0.0, x.dtype), mx.array(-1e9, x.dtype))
        attn = mx.softmax(attn, axis=-1)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)
        x = x + self.out(y)
        h = self.ln2(x)
        return x + self.down(nn.gelu(self.up(h)))


class TinyLM(nn.Module):
    """A 2-layer decoder-only transformer with a very wide classifier head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Start with a uniform distribution over the vocabulary rather than the
        # default random head, so the first loss is log(|V|) and any improvement
        # is the model learning rather than the initialisation shrinking.
        self.head.weight = mx.zeros((cfg.vocab_size, cfg.d_model), mx.float32)

    def hidden(self, tokens: mx.array) -> mx.array:
        """Backbone output ``(B, T, D)`` for a batch of token ids."""
        b, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds {self.cfg.max_seq_len}")
        x = self.embed(tokens) + self.pos(mx.arange(t))[None]
        # Causal mask, true where attention is allowed.
        mask = mx.arange(t)[:, None] >= mx.arange(t)[None, :]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)

    def __call__(self, tokens: mx.array) -> mx.array:
        """The full ``(B, T, V)`` logits. This is what CCE avoids building."""
        return self.hidden(tokens) @ self.head.weight.T

    @property
    def n_params(self) -> int:
        return sum(v.size for _, v in nn.utils.tree_flatten(self.parameters()))


class ConfigSmall(Config):
    pass
