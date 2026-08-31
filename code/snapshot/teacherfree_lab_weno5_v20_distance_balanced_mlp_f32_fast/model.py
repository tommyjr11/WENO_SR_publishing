#!/usr/bin/env python3
"""Reflection-symmetric WENO5 model with an FP32 MLP and FP64 solver I/O."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import weno5_core as W


SCALE_LOG10_MIN = -4.0


def remap_scale_feature(features: torch.Tensor) -> torch.Tensor:
    """Preserve V20's scale-channel mapping exactly."""
    old_scale = features[..., 4]
    log10_relative = 16.0 * old_scale - 16.0
    new_scale = torch.clamp(
        (log10_relative - SCALE_LOG10_MIN) / (-SCALE_LOG10_MIN),
        0.0,
        1.0,
    )
    return torch.cat(
        (features[..., :4], new_scale.unsqueeze(-1)), dim=-1
    )


class ReflectionSymmetricBadnessMLPFloat32(W.SharedBadnessMLP):
    """Keep only the trainable MLP in FP32.

    WENO features are formed by the FP64 solver. The V9 scale remapping is
    evaluated in FP64, then the five inputs are cast to FP32 immediately before
    the first affine layer. Softmax ratios are cast back to the caller dtype
    before reflection averaging and WENO normalization.
    """

    def __init__(self, seed: int = 41) -> None:
        # Reproduce V20's seeded FP64 initialization, then quantize it. The
        # zero final layer still gives the exact linear-d initial operator.
        super().__init__(seed=seed)
        self.float()

    @staticmethod
    def reflect_features(features: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                features[..., 2],
                features[..., 1],
                features[..., 0],
                features[..., 3],
                features[..., 4],
            ),
            dim=-1,
        )

    @staticmethod
    def reflect_ratios(ratios: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (ratios[..., 2], ratios[..., 1], ratios[..., 0]), dim=-1
        )

    def raw_forward(self, features: torch.Tensor) -> torch.Tensor:
        output_dtype = features.dtype
        h = remap_scale_feature(features).to(dtype=torch.float32)
        h = h @ self.w1 + self.b1
        h = h * torch.sigmoid(h)
        h = h @ self.w2 + self.b2
        h = h * torch.sigmoid(h)
        h = h @ self.w3 + self.b3
        h = h * torch.sigmoid(h)
        raw = h @ self.w4 + self.b4
        badness = 6.0 * torch.tanh(raw / 6.0)
        return torch.softmax(badness, dim=-1).to(dtype=output_dtype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        direct = self.raw_forward(features)
        reflected = self.raw_forward(self.reflect_features(features))
        return 0.5 * (direct + self.reflect_ratios(reflected))


@torch.no_grad()
def reflection_defect(
    model: ReflectionSymmetricBadnessMLPFloat32,
    features: torch.Tensor,
) -> float:
    left = model(model.reflect_features(features))
    right = model.reflect_ratios(model(features))
    return float(torch.max(torch.abs(left - right)))


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> ReflectionSymmetricBadnessMLPFloat32:
    data = np.load(path, allow_pickle=True)
    expected = W.expected_shapes()
    missing = [name for name in expected if name not in data.files]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    wrong = {
        name: data[name].shape
        for name, shape in expected.items()
        if data[name].shape != shape
    }
    if wrong:
        raise ValueError(f"{path} has incompatible WENO5 MLP shapes: {wrong}")

    model = ReflectionSymmetricBadnessMLPFloat32(seed=0).to(device)
    with torch.no_grad():
        for name in expected:
            parameter = getattr(model, name)
            parameter.copy_(
                torch.as_tensor(
                    data[name][0],
                    device=device,
                    dtype=torch.float32,
                )
            )
    return model
