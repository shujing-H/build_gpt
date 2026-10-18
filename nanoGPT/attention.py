"""
Causal self-attention (notebook v4 → multi-head), written explicitly for learning.

Each head has its own slice of the fused Q/K/V projection. Attention is computed
per head with a causal mask, then heads are concatenated and mixed with c_proj.

model.py uses the same layout but F.scaled_dot_product_attention for speed.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttentionHead(nn.Module):
    """Single head: separate Q/K/V linears (notebook-style, n_head=1 only)."""

    def __init__(self, n_embd: int, head_size: int, bias: bool = False):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        tril = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        weight = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        weight = weight.masked_fill(~tril, float("-inf"))
        weight = F.softmax(weight, dim=-1)

        return weight @ v


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    One Linear produces Q, K, V for all heads; reshape to (B, n_head, T, head_size),
    run scaled dot-product attention per head, concat, then output projection.
    """

    def __init__(self, n_embd: int, n_head: int, bias: bool = False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape  # (batch_size, sequence_length, embedding_dim)

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        # (B, n_head, T, head_size)

        tril = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        weight = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        weight = weight.masked_fill(~tril, float("-inf"))
        weight = F.softmax(weight, dim=-1)

        y = weight @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(y)
