"""RoPE helpers in the Hugging Face `rotate_half` layout.

HF applies   P_t v = v * cos_t + rotate_half(v) * sin_t,
with cos_t[a] == cos_t[a + h/2] (frequencies duplicated over the two halves) and, for
scaled variants (YaRN/longrope), cos/sin already multiplied by the attention factor.
So P_t = a * R_t with R_t orthogonal and the transpose is
    P_t^T u = u * cos_t - rotate_half(u) * sin_t.
Only `P_t` linear per position is needed by the statistics code; no relative-position
identity is assumed.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h2 = x.shape[-1] // 2
    return torch.cat((-x[..., h2:], x[..., :h2]), dim=-1)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """P_t x on the last dimension."""
    return x * cos + rotate_half(x) * sin


def rope_apply_T(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """P_t^T x on the last dimension. Applied to the rows of a matrix M it returns M P_t."""
    return x * cos - rotate_half(x) * sin


def rope_diag_var(m: torch.Tensor, e2: torch.Tensor, eab: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diag(P^T V P) where V = Cov(u) is given by first/second moments of u in rotated space.

    m   = E[u]                (..., h)
    e2  = E[u * u]            (..., h)
    eab = E[u_a * u_{a+h/2}]  (..., h/2)
    """
    h2 = m.shape[-1] // 2
    ma, mb = m[..., :h2], m[..., h2:]
    vaa = e2[..., :h2] - ma * ma
    vbb = e2[..., h2:] - mb * mb
    vab = eab - ma * mb
    c = cos[..., :h2]
    s = sin[..., :h2]
    ya = c * c * vaa + s * s * vbb + 2.0 * c * s * vab
    yb = c * c * vbb + s * s * vaa - 2.0 * c * s * vab
    return torch.cat((ya, yb), dim=-1).clamp_min(0.0)


def _find_rotary(model):
    inner = getattr(model, "model", model)
    rot = getattr(inner, "rotary_emb", None)
    if rot is not None:
        return rot
    layers = getattr(inner, "layers", None)
    if layers is not None and len(layers) > 0:
        rot = getattr(layers[0].self_attn, "rotary_emb", None)
        if rot is not None:
            return rot
    raise RuntimeError("Could not find a rotary embedding module (model.model.rotary_emb or self_attn.rotary_emb)")


@torch.no_grad()
def compute_rope_tables(model, seq_len: int, device, dtype=torch.float32):
    """Return (cos, sin) of shape (1, seq_len, head_dim) from the model's own rotary module."""
    rot = _find_rotary(model)
    pos = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)
    cos, sin = rot(dummy, pos)
    return cos.to(dtype), sin.to(dtype)
