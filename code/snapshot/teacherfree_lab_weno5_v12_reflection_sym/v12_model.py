#!/usr/bin/env python3
"""WENO5 MLP with reflection equivariance enforced by construction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v9_exact_fvm.v9_model import V9BadnessMLP


class ReflectionSymmetricBadnessMLP(V9BadnessMLP):
    """Evaluate both stencil orientations and average in output space.

    For x=(delta0,delta1,delta2,gamma,scale), reflection is
    Px=(delta2,delta1,delta0,gamma,scale).  The second MLP output is reflected
    back before averaging, so forward(Px) == P forward(x) up to roundoff.
    Both branches share exactly the same trainable parameters.
    """

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
        """One orientation of the shared V9 MLP, without symmetrization."""
        return super().forward(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        direct = self.raw_forward(features)
        reflected = self.raw_forward(self.reflect_features(features))
        return 0.5 * (direct + self.reflect_ratios(reflected))


@torch.no_grad()
def reflection_defect(
    model: ReflectionSymmetricBadnessMLP, features: torch.Tensor
) -> float:
    left = model(model.reflect_features(features))
    right = model.reflect_ratios(model(features))
    return float(torch.max(torch.abs(left - right)))


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> ReflectionSymmetricBadnessMLP:
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
    model = ReflectionSymmetricBadnessMLP(seed=0).to(device)
    with torch.no_grad():
        for name in expected:
            getattr(model, name).copy_(torch.as_tensor(data[name][0], device=device))
    return model
