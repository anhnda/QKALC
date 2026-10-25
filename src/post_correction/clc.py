from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.quantization.state import IntegerQuantizedTensorState


def find_knee_point(values, tolerance_offset: float = 0.0) -> int:
    n = len(values)
    if n < 3:
        return n // 2

    if torch.is_tensor(values):
        y = values.detach().cpu().float().numpy()
    else:
        y = np.asarray(values, dtype=np.float32)

    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return n // 2

    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0.0, 1.0, n)
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    distances = np.abs(y_norm - y_line)

    knee_idx = int(np.argmax(distances))
    if knee_idx < n - 1:
        offset_indices = int(tolerance_offset * n)
        knee_idx = min(knee_idx + offset_indices, n - 1)
        knee_idx = max(knee_idx, 0)
    return knee_idx


def compute_james_stein_mean(raw_means: torch.Tensor, variance_estimate: Optional[torch.Tensor] = None) -> torch.Tensor:
    p = len(raw_means)
    if p < 3:
        return raw_means

    grand_mean = raw_means.mean()
    deviations = raw_means - grand_mean
    sum_sq_dev = (deviations ** 2).sum()
    if sum_sq_dev < 1e-10:
        return raw_means

    if variance_estimate is None:
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp(min=1e-8)

    shrinkage_factor = ((p - 2) * variance_estimate) / sum_sq_dev
    shrinkage_factor = shrinkage_factor.clamp(0, 1)
    return grand_mean + (1 - shrinkage_factor) * deviations


@dataclass
class CLCConfig:
    knee_tolerance: float = 0.0
    max_flip_percent: float = 0.05
    use_james_stein: bool = True


class CLCCorrection:
    def __init__(self, config: Optional[CLCConfig] = None):
        self.config = config or CLCConfig()

    @staticmethod
    def empty_flip_stats() -> dict:
        return {
            "total": 0,
            "per_channel_mean": 0.0,
            "per_channel_median": 0.0,
            "per_channel_min": 0.0,
            "per_channel_max": 0.0,
            "per_channel_std": 0.0,
            "per_channel_p25": 0.0,
            "per_channel_p75": 0.0,
            "per_channel_p90": 0.0,
            "per_channel_p95": 0.0,
            "per_channel_p99": 0.0,
            "per_channel_zero_pct": 100.0,
        }

    def prepare_activation_means(self, raw_means: torch.Tensor) -> torch.Tensor:
        if self.config.use_james_stein:
            return compute_james_stein_mean(raw_means)
        return raw_means

    @torch.no_grad()
    def compute_dynamic_outlier_threshold(self, activation_means: torch.Tensor, debug: bool = False):
        sorted_means, _ = torch.sort(activation_means.abs(), descending=True)
        n = len(sorted_means)
        first_half = sorted_means[: n // 2]

        if len(first_half) < 3:
            threshold_idx = int(0.05 * n)
            threshold = sorted_means[threshold_idx].item()
            return threshold, 0.05

        knee_idx = find_knee_point(first_half, tolerance_offset=self.config.knee_tolerance)
        threshold = sorted_means[knee_idx].item()
        num_outliers = (activation_means.abs() >= threshold).sum().item()
        outlier_percent = num_outliers / n

        if debug:
            print(f"    DEBUG threshold={threshold:.6f}, outlier_percent={outlier_percent*100:.2f}%")

        return threshold, outlier_percent

    @torch.no_grad()
    def apply(
        self,
        quant_state: IntegerQuantizedTensorState,
        activation_means: torch.Tensor,
        debug: bool = False,
    ):
        device = quant_state.integer_weights.device
        act_padded = torch.zeros(quant_state.padded_in_features, device=device, dtype=quant_state.scale.dtype)
        act_padded[: quant_state.in_features] = activation_means

        w_quant = quant_state.dequantize()
        w_diff = quant_state.float_weights - w_quant
        current_error = (w_diff * act_padded.unsqueeze(0)).sum(dim=1)

        flip_dir = torch.sign(quant_state.pre_round - quant_state.integer_weights)
        flip_dir[flip_dir == 0] = 1.0
        flip_impacts = act_padded.unsqueeze(0) * flip_dir * quant_state.scale

        target_sign = torch.sign(current_error).unsqueeze(1)
        valid_mask = torch.sign(flip_impacts) == target_sign

        w_int_proposed = quant_state.integer_weights + flip_dir
        in_range = (w_int_proposed >= quant_state.min_int) & (w_int_proposed <= quant_state.max_int)
        valid_mask = valid_mask & in_range

        outlier_threshold, outlier_percent = self.compute_dynamic_outlier_threshold(act_padded, debug=debug)
        is_outlier = act_padded.abs() > outlier_threshold
        valid_mask = valid_mask & (~is_outlier).unsqueeze(0)

        rounding_costs = (quant_state.pre_round - quant_state.integer_weights).abs()
        rounding_costs_masked = rounding_costs.clone()
        rounding_costs_masked[~valid_mask] = -1.0

        sorted_indices = torch.argsort(rounding_costs_masked, dim=1, descending=True)
        sorted_impacts = torch.gather(flip_impacts, 1, sorted_indices)
        sorted_validity = torch.gather(valid_mask.long(), 1, sorted_indices)
        sorted_impacts = sorted_impacts * sorted_validity

        cumsum_impacts = torch.cumsum(sorted_impacts, dim=1)
        residuals = torch.abs(current_error.unsqueeze(1) - cumsum_impacts)
        all_residuals = torch.cat([torch.abs(current_error).unsqueeze(1), residuals], dim=1)
        best_k = torch.argmin(all_residuals, dim=1)

        idx_range = torch.arange(quant_state.padded_in_features, device=device).unsqueeze(0)
        flip_mask_sorted = idx_range < best_k.unsqueeze(1)
        final_flips_sorted = flip_mask_sorted & sorted_validity.bool()

        sorted_flip_dir = torch.gather(flip_dir, 1, sorted_indices)
        sorted_flip_dir[~final_flips_sorted] = 0.0

        max_flips_per_output = int(self.config.max_flip_percent * quant_state.in_features)
        cumsum_flips = final_flips_sorted.long().cumsum(dim=1)
        within_limit = cumsum_flips <= max_flips_per_output
        sorted_flip_dir[~within_limit] = 0.0

        quant_state.integer_weights.scatter_add_(1, sorted_indices, sorted_flip_dir)
        quant_state.integer_weights.clamp_(quant_state.min_int, quant_state.max_int)

        num_flips_total = int((sorted_flip_dir != 0).sum().item())
        flips_per_channel = (sorted_flip_dir != 0).sum(dim=0).float()
        if quant_state.padded_in_features > quant_state.in_features:
            flips_per_channel = flips_per_channel[: quant_state.in_features]

        flip_stats = self.empty_flip_stats()
        if flips_per_channel.numel() > 0:
            flip_stats = {
                "total": num_flips_total,
                "per_channel_mean": flips_per_channel.mean().item(),
                "per_channel_median": flips_per_channel.median().item(),
                "per_channel_min": flips_per_channel.min().item(),
                "per_channel_max": flips_per_channel.max().item(),
                "per_channel_std": flips_per_channel.std().item(),
                "per_channel_p25": torch.quantile(flips_per_channel, 0.25).item(),
                "per_channel_p75": torch.quantile(flips_per_channel, 0.75).item(),
                "per_channel_p90": torch.quantile(flips_per_channel, 0.90).item(),
                "per_channel_p95": torch.quantile(flips_per_channel, 0.95).item(),
                "per_channel_p99": torch.quantile(flips_per_channel, 0.99).item(),
                "per_channel_zero_pct": (flips_per_channel == 0).float().mean().item() * 100,
            }

        return quant_state.dequantize_truncated(), outlier_percent, flip_stats

