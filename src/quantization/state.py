from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IntegerQuantizedTensorState:
    float_weights: torch.Tensor
    pre_round: torch.Tensor
    integer_weights: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    max_int: int
    min_int: int
    in_features: int
    padded_in_features: int
    original_dtype: torch.dtype

    def dequantize(self) -> torch.Tensor:
        return (self.integer_weights - self.zero_point) * self.scale

    def dequantize_truncated(self) -> torch.Tensor:
        dequantized = self.dequantize()
        if self.padded_in_features > self.in_features:
            dequantized = dequantized[:, : self.in_features]
        return dequantized.to(self.original_dtype)
