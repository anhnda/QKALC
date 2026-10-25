"""Base weight quantizers (RTN / AWQ-style) on an explicit integer lattice.

Extracted from the original CLC repo (`src/quantization/awq.py`):
  * `groupwise_quantize`   == `AWQQuantizerXL.quantize_weight_groupwise_raw`
  * `awq_search_scales`    == `AWQQuantizerXL.search_best_scale` (same grid / salience rule)

Differences from the repo:
  * everything is done in float32 (the repo built states in the weight dtype, i.e. bf16,
    which makes `pre_round` too coarse to decide rounding directions reliably);
  * AWQ search runs on activation rows sampled during a block-sequential pass instead of
    hooks over the whole model.

The lattice is kept so that later stages (CLC, QK correction) can flip integer codes:
    W_eff[o, i] = (code[o, i] - zp[o, i]) * scale[o, i] / in_scale[i]
For RTN `in_scale == 1`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.quantization.state import IntegerQuantizedTensorState


@torch.no_grad()
def groupwise_quantize(w: torch.Tensor, bits: int, group_size: int) -> IntegerQuantizedTensorState:
    """Asymmetric min-max group-wise RTN. `group_size <= 0` means one group per row."""
    w = w.float()
    out_features, in_features = w.shape
    gs = in_features if group_size is None or group_size <= 0 else group_size
    n_groups = (in_features + gs - 1) // gs
    padded = n_groups * gs

    if padded > in_features:
        w_p = torch.zeros(out_features, padded, device=w.device, dtype=w.dtype)
        w_p[:, :in_features] = w
    else:
        w_p = w

    w_g = w_p.reshape(out_features, n_groups, gs)
    w_min = w_g.min(dim=2, keepdim=True)[0]
    w_max = w_g.max(dim=2, keepdim=True)[0]
    max_int = 2 ** bits - 1

    scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)

    scale_flat = scale.expand(out_features, n_groups, gs).reshape(out_features, padded).contiguous()
    zp_flat = zp.expand(out_features, n_groups, gs).reshape(out_features, padded).contiguous()

    pre_round = w_p / scale_flat + zp_flat
    w_int = torch.round(pre_round).clamp(0, max_int)

    return IntegerQuantizedTensorState(
        float_weights=w_p,
        pre_round=pre_round,
        integer_weights=w_int,
        scale=scale_flat,
        zero_point=zp_flat,
        max_int=max_int,
        min_int=0,
        in_features=in_features,
        padded_in_features=padded,
        original_dtype=torch.float32,
    )


@torch.no_grad()
def awq_search_scales(
    w: torch.Tensor,
    x_rows: torch.Tensor,
    salience: torch.Tensor,
    bits: int,
    group_size: int,
    n_grid: int = 20,
):
    """Grid search over s = salience**alpha minimising ||X W^T - X Q(W s)/s ^T||^2 (repo semantics)."""
    w = w.float()
    x = x_rows.float()
    sal = salience.float().clamp(min=1e-5)
    in_features = w.shape[1]
    y_ref = x @ w.t()

    best_err = float("inf")
    best_alpha = 0.0
    best_scales = torch.ones(in_features, device=w.device)
    for g in range(n_grid + 1):
        alpha = g / n_grid
        s = sal.pow(alpha)
        st = groupwise_quantize(w * s[None, :], bits, group_size)
        w_rec = st.dequantize()[:, :in_features] / s[None, :]
        err = (y_ref - x @ w_rec.t()).pow(2).mean().item()
        if err < best_err:
            best_err, best_alpha, best_scales = err, alpha, s.clone()
        del st, w_rec
    return best_scales, best_alpha, best_err


class ActStats:
    """Streaming activation statistics for one Linear input (mean, E[x^2], sampled rows)."""

    def __init__(self, rows_per_call: int = 0):
        self.rows_per_call = int(rows_per_call)
        self.sum: Optional[torch.Tensor] = None
        self.sumsq: Optional[torch.Tensor] = None
        self.n = 0
        self.rows: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.sum is None:
            self.sum = torch.zeros(x.shape[1], device=x.device)
            self.sumsq = torch.zeros(x.shape[1], device=x.device)
        self.sum += x.sum(0)
        self.sumsq += (x * x).sum(0)
        self.n += x.shape[0]
        if self.rows_per_call > 0:
            idx = torch.randperm(x.shape[0], device=x.device)[: self.rows_per_call]
            self.rows.append(x[idx].clone())

    def mean(self) -> torch.Tensor:
        return self.sum / max(self.n, 1)

    def salience(self) -> torch.Tensor:
        return self.sumsq / max(self.n, 1)

    def sample_rows(self, max_rows: int) -> torch.Tensor:
        rows = torch.cat(self.rows, dim=0)
        if rows.shape[0] > max_rows:
            rows = rows[torch.randperm(rows.shape[0], device=rows.device)[:max_rows]]
        return rows


def rows_per_call_for(max_rows: int, n_calls: int) -> int:
    return int(math.ceil(max_rows / max(n_calls, 1)))


@dataclass
class LatticeState:
    """Integer lattice of one Linear plus the per-input-channel AWQ scales (ones for RTN)."""

    state: IntegerQuantizedTensorState
    in_scales: torch.Tensor

    def effective_weight(self) -> torch.Tensor:
        d = self.state.in_features
        return self.state.dequantize()[:, :d] / self.in_scales[None, :]


@torch.no_grad()
def quantize_linear(
    module: nn.Linear,
    method: str,
    bits: int,
    group_size: int,
    act: Optional[ActStats] = None,
    clc=None,
    awq_n_grid: int = 20,
    awq_max_rows: int = 2048,
    keep_state: bool = False,
):
    """Quantize `module.weight` in place (fake-quant, effective weight written in module dtype).

    method: "rtn" | "awq".  `clc` is an optional `CLCCorrection` applied on the lattice
    (repo semantics: activation means divided by the AWQ input scales).
    Returns (LatticeState | None, info dict).
    """
    w = module.weight.data.float()
    dtype = module.weight.dtype
    in_features = w.shape[1]
    in_scales = torch.ones(in_features, device=w.device)
    info: dict = {}

    if method == "awq":
        if act is None or act.n == 0:
            raise RuntimeError("AWQ needs activation statistics")
        x_rows = act.sample_rows(awq_max_rows).to(w.device)
        in_scales, alpha, err = awq_search_scales(w, x_rows, act.salience().to(w.device), bits, group_size, awq_n_grid)
        info.update(awq_alpha=alpha, awq_err=err)
        del x_rows
    elif method != "rtn":
        raise ValueError(f"Unknown base method: {method}")

    state = groupwise_quantize(w * in_scales[None, :], bits, group_size)

    if clc is not None:
        if act is not None and act.n > 0:
            raw_mean = act.mean().to(w.device)
        else:
            raw_mean = torch.zeros(in_features, device=w.device)
        post_mean = clc.prepare_activation_means(raw_mean) / in_scales
        _, outlier_pct, flip_stats = clc.apply(state, post_mean)
        info.update(clc_flips=int(flip_stats["total"]), clc_outlier_pct=float(outlier_pct))

    lat = LatticeState(state=state, in_scales=in_scales)
    module.weight.data.copy_(lat.effective_weight().to(dtype))
    return (lat if keep_state else None), info
