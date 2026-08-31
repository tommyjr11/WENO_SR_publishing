#!/usr/bin/env python3
"""V9 model with a deployment-compatible but better-resolved scale channel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from teacherfree_lab_weno5 import weno5_core as W


SCALE_LOG10_MIN = -4.0


def remap_scale_feature(features: torch.Tensor) -> torch.Tensor:
    """Map relative variation 1e-4..1 onto the full scale-feature interval.

    Existing callers provide the old feature, which maps log10(relative scale)
    from [-16, 0] to [0, 1]. Recovering that logarithm makes V9 usable by all
    existing differentiable losses without changing their stencil code.
    """
    old_scale = features[..., 4]
    log10_relative = 16.0 * old_scale - 16.0
    new_scale = torch.clamp(
        (log10_relative - SCALE_LOG10_MIN) / (-SCALE_LOG10_MIN), 0.0, 1.0
    )
    return torch.cat((features[..., :4], new_scale.unsqueeze(-1)), dim=-1)


class V9BadnessMLP(W.SharedBadnessMLP):
    """Same 5->10->6->6->3 checkpoint shape with the V9 feature mapping."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(remap_scale_feature(features))


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> V9BadnessMLP:
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
    model = V9BadnessMLP(seed=0).to(device)
    with torch.no_grad():
        for name in expected:
            getattr(model, name).copy_(torch.as_tensor(data[name][0], device=device))
    return model
