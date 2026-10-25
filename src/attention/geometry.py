"""Attention geometry and small projection helpers (Llama / Mistral / Qwen2 family)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class AttnGeometry:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    scaling: float

    @property
    def group(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def q_out(self) -> int:
        return self.n_heads * self.head_dim


def get_attn_geometry(model, layer, seq_len: Optional[int] = None) -> AttnGeometry:
    cfg = model.config
    attn = layer.self_attn
    if hasattr(attn, "q_norm") or hasattr(attn, "k_norm"):
        raise NotImplementedError("QK-norm models (Qwen3, Gemma3, OLMo2, ...) are out of scope for Q-pass v1")
    prf = getattr(cfg, "partial_rotary_factor", 1.0)
    if prf is not None and float(prf) != 1.0:
        raise NotImplementedError("partial RoPE is not supported")
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    if attn.q_proj.out_features != n_heads * head_dim or attn.k_proj.out_features != n_kv * head_dim:
        raise RuntimeError("Unexpected q/k projection shapes for the detected geometry")
    scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
    sw = getattr(cfg, "sliding_window", None)
    if seq_len is not None and sw and getattr(cfg, "use_sliding_window", True) and sw < seq_len:
        warnings.warn("Sliding-window attention is not modelled by the QK statistics (full causal assumed).")
    return AttnGeometry(n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim, scaling=scaling)


def project_heads(x: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor], n: int, h: int) -> torch.Tensor:
    """x (..., d) @ W^T (+ b) -> (..., n, h)  (nn.Linear layout, out index = head * h + r)."""
    y = x @ W.t()
    if b is not None:
        y = y + b
    return y.reshape(*x.shape[:-1], n, h)


def expand_kv(k: torch.Tensor, group: int) -> torch.Tensor:
    """(T, n_kv, h) -> (n_heads, T, h); head j uses kv head j // group (HF repeat_kv)."""
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
    return k.permute(1, 0, 2).contiguous()
