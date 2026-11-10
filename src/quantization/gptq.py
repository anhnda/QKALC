"""GPTQ (Frantar et al., ICLR 2023) on the explicit integer lattice used by CLC / QK correction.

Output is an `IntegerQuantizedTensorState` so that every later stage (CLC, Q-pass, K-pass) runs
unchanged on top of GPTQ:

    W_eff[o, i] = (code[o, i] - zp[o, i]) * scale[o, i]        (in_scales == 1, as for RTN)

Lattice semantics (important for the later flip stages):
  * `pre_round[o, i]` is the *error-compensated* value  w'_oi / scale + zp  at the moment GPTQ rounds
    column i (w' = weight after the OBS updates of all previously quantized columns). GPTQ picks the
    nearest level to it, so "complementary flip" = move to the other adjacent level of pre_round,
    exactly as for RTN/AWQ.
  * `float_weights` is the ORIGINAL weight W (not the compensated one). CLC computes its mean-shift
    target as  sum_i mu_i (W - W_hat)_oi , which must use the original W to be the true output shift.
  * scale / zero_point are stored per element, so act-order with dynamic groups (non-contiguous
    groups in the original column order, i.e. needs g_idx in a real kernel) is representable.

Differences from the reference implementation (github.com/IST-DASLab/gptq), on purpose:
  * group parameters use the same asymmetric min-max rule as `base.groupwise_quantize`
    (range NOT forced to contain 0), so RTN / AWQ / GPTQ share one lattice definition;
  * "dead" input channels (zero calibration energy) keep their weights (the reference sets them to 0,
    which changes the model on inputs where that channel is non-zero); they are RTN-rounded and
    do not propagate error;
  * a dynamic group that straddles a lazy-batch block is built from exactly-updated columns;
  * damping is retried x10 if the Cholesky factorisation fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from src.quantization.state import IntegerQuantizedTensorState


@dataclass
class GPTQConfig:
    damp: float = 0.01            # relative damping: lambda = damp * mean(diag H)
    blocksize: int = 128          # lazy-batch block size
    act_order: bool = True        # process columns by decreasing diag(H)
    static_groups: bool = False   # group params from the original W, contiguous groups (kernel friendly)
    dtype: torch.dtype = torch.float32


def _minmax_params(w: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row asymmetric min-max params over the given columns (same rule as groupwise_quantize)."""
    max_int = 2 ** bits - 1
    w_min = w.min(dim=1).values
    w_max = w.max(dim=1).values
    scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)
    return scale, zp


@torch.no_grad()
def _upper_cholesky_of_inverse(H: torch.Tensor, damp: float, max_tries: int = 6) -> Tuple[torch.Tensor, float]:
    """U upper-triangular with U^T U = (H + lambda I)^-1. Returns (U, relative damping used)."""
    d = H.shape[0]
    diag_mean = float(torch.diagonal(H).mean())
    if not math.isfinite(diag_mean) or diag_mean <= 0:
        diag_mean = 1.0
    idx = torch.arange(d, device=H.device)
    rel = float(damp)
    for _ in range(max_tries):
        Hd = H.clone()
        Hd[idx, idx] += rel * diag_mean
        L, info = torch.linalg.cholesky_ex(Hd)
        del Hd
        if int(info) == 0:
            Hinv = torch.cholesky_inverse(L)
            del L
            U, info2 = torch.linalg.cholesky_ex(Hinv, upper=True)
            del Hinv
            if int(info2) == 0:
                return U, rel
        rel = max(rel, 1e-6) * 10.0
    raise RuntimeError(f"GPTQ: Cholesky failed even with relative damping {rel:g}")


@torch.no_grad()
def gptq_quantize(
    w: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    group_size: int,
    cfg: Optional[GPTQConfig] = None,
) -> Tuple[IntegerQuantizedTensorState, dict]:
    """GPTQ of one weight matrix w (out, d) with input second moment H (d, d) ~ E[x x^T].

    Returns (lattice state in the ORIGINAL column order, info dict).
    """
    cfg = cfg or GPTQConfig()
    dt = cfg.dtype
    W0 = w.detach().to(dt).clone()   # own copy: `w` may alias module.weight, which is overwritten later
    out_f, d = W0.shape
    dev = W0.device
    max_int = 2 ** bits - 1
    gs = d if (group_size is None or group_size <= 0 or group_size >= d) else int(group_size)
    n_groups = (d + gs - 1) // gs
    static = cfg.static_groups or gs >= d

    H = H.detach().to(dev, dt).clone()
    W = W0.clone()

    dead = torch.diagonal(H) <= 0
    n_dead = int(dead.sum())
    if n_dead:
        di = dead.nonzero().flatten()
        H[di, :] = 0.0
        H[:, di] = 0.0
        H[di, di] = 1.0

    # static group params: contiguous groups of the ORIGINAL column order, from the original W
    g_scale = g_zp = None
    if static:
        g_scale = torch.empty(out_f, n_groups, device=dev, dtype=dt)
        g_zp = torch.empty(out_f, n_groups, device=dev, dtype=dt)
        for g in range(n_groups):
            s, z = _minmax_params(W[:, g * gs:min((g + 1) * gs, d)], bits)
            g_scale[:, g], g_zp[:, g] = s, z

    perm = None
    if cfg.act_order:
        perm = torch.argsort(torch.diagonal(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
    perm_list = perm.tolist() if perm is not None else None

    Hinv, damp_used = _upper_cholesky_of_inverse(H, cfg.damp)
    del H

    codes = torch.empty(out_f, d, device=dev, dtype=dt)
    pres = torch.empty_like(codes)
    scl = torch.empty_like(codes)
    zps = torch.empty_like(codes)
    loss = torch.zeros((), device=dev, dtype=torch.float64)

    bs = max(1, int(cfg.blocksize))
    cur_s = cur_z = None
    for i1 in range(0, d, bs):
        i2 = min(i1 + bs, d)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if static:
                orig = perm_list[col] if perm_list is not None else col
                g = orig // gs
                s, z = g_scale[:, g], g_zp[:, g]
            else:
                if col % gs == 0:
                    end = min(col + gs, d)
                    if end <= i2:
                        seg = W1[:, i:i + (end - col)]
                    else:
                        # columns beyond this block have not yet received this block's lazy update
                        rest = W[:, i2:end] - Err1[:, :i] @ Hinv[i1:col, i2:end]
                        seg = torch.cat([W1[:, i:], rest], dim=1)
                    cur_s, cur_z = _minmax_params(seg, bits)
                s, z = cur_s, cur_z

            wc = W1[:, i]
            pre = wc / s + z
            q = torch.round(pre).clamp(0, max_int)
            deq = (q - z) * s
            codes[:, col] = q
            pres[:, col] = pre
            scl[:, col] = s
            zps[:, col] = z

            dii = Hinv1[i, i]
            err = (wc - deq) / dii
            loss += (((wc - deq) ** 2).sum() / (dii * dii)).double() * 0.5   # no host sync
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        if i2 < d:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        del W1, Err1

    if perm is not None:
        inv = torch.argsort(perm)
        codes, pres, scl, zps = codes[:, inv], pres[:, inv], scl[:, inv], zps[:, inv]
    del W, Hinv

    state = IntegerQuantizedTensorState(
        float_weights=W0,
        pre_round=pres.contiguous(),
        integer_weights=codes.contiguous(),
        scale=scl.contiguous(),
        zero_point=zps.contiguous(),
        max_int=max_int,
        min_int=0,
        in_features=d,
        padded_in_features=d,
        original_dtype=dt,
    )
    info = {"gptq_loss": float(loss), "gptq_damp": float(damp_used), "gptq_dead": n_dead}
    return state, info
