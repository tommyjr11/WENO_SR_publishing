#!/usr/bin/env python3
"""Algebraic, precision, symmetry, and checkpoint tests for this package."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import rk3_advection as A
import weno5_core as W
from model import (
    ReflectionSymmetricBadnessMLPFloat32,
    load_checkpoint,
    reflection_defect,
)

torch.set_default_dtype(torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = W.torch_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    W.check_weno5_coefficients()
    generator = torch.Generator(device=device)
    generator.manual_seed(20260727)
    model = ReflectionSymmetricBadnessMLPFloat32(seed=17).to(device)
    if {parameter.dtype for parameter in model.parameters()} != {
        torch.float32
    }:
        raise AssertionError("MLP parameters are not all float32")

    stencils = torch.randn(
        (4096, 5), generator=generator, device=device, dtype=torch.float64
    )
    features = W.weno5_features(stencils)
    ratios = model(features)
    if ratios.dtype != torch.float64:
        raise AssertionError("mixed MLP output did not return to float64")
    linear_errors = {}
    for lr in W.LR_VALUES:
        omega = W.omega_from_ratio(ratios, lr)
        target = W.optimal_d(lr, device).expand_as(omega)
        linear_errors[str(lr)] = float(
            torch.max(torch.abs(omega - target)).detach()
        )
    if max(linear_errors.values()) > 2.0e-15:
        raise AssertionError(
            f"zero final layer is not linear-d: {linear_errors}"
        )

    with torch.no_grad():
        model.w4.normal_(mean=0.0, std=0.1, generator=generator)
        model.b4.normal_(mean=0.0, std=0.03, generator=generator)
    defect = reflection_defect(model, features)
    if defect > 2.0e-15:
        raise AssertionError(f"reflection defect={defect}")

    state = torch.randn(
        (2, 48), generator=generator, device=device, dtype=torch.float64
    )
    velocities = A.balanced_velocities(2, device)
    mass_before = torch.sum(state, dim=1)
    updated = A.ssprk3_step_signed(
        model, state, 0.2 / 48.0, 48.0, velocities
    )
    if updated.dtype != torch.float64:
        raise AssertionError("SSPRK3 state is not float64")
    mass_error = float(
        torch.max(
            torch.abs(torch.sum(updated, dim=1) - mass_before)
        ).detach()
    )
    if mass_error > 2.0e-12:
        raise AssertionError(f"periodic mass error={mass_error}")

    positive = A.ssprk3_step_signed(
        model,
        state[:1],
        0.2 / 48.0,
        48.0,
        torch.ones(1, device=device, dtype=torch.float64),
    )
    negative_reflected = A.ssprk3_step_signed(
        model,
        torch.flip(state[:1], dims=(-1,)),
        0.2 / 48.0,
        48.0,
        -torch.ones(1, device=device, dtype=torch.float64),
    )
    bidirectional_error = float(
        torch.max(
            torch.abs(
                torch.flip(positive, dims=(-1,)) - negative_reflected
            )
        ).detach()
    )
    if bidirectional_error > 2.0e-12:
        raise AssertionError(
            f"bidirectional reflection error={bidirectional_error}"
        )

    loss = torch.mean(torch.square(updated))
    loss.backward()
    gradient_norm = float(
        torch.sqrt(
            sum(
                torch.sum(parameter.grad.square())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    )
    if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise AssertionError("end-to-end gradient did not reach the MLP")

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "model.npz"
        W.save_checkpoint(
            checkpoint,
            model,
            {"precision": "mlp_float32_state_float64"},
        )
        with np.load(checkpoint, allow_pickle=True) as data:
            checkpoint_dtypes = {
                name: str(data[name].dtype)
                for name in W.expected_shapes()
            }
        if set(checkpoint_dtypes.values()) != {"float32"}:
            raise AssertionError(
                f"checkpoint parameters are not FP32: {checkpoint_dtypes}"
            )
        loaded = load_checkpoint(checkpoint, device)
        checkpoint_error = float(
            torch.max(torch.abs(loaded(features) - model(features)))
        )
        if checkpoint_error != 0.0:
            raise AssertionError(
                f"checkpoint round-trip error={checkpoint_error}"
            )

    print(
        json.dumps(
            {
                "device": str(device),
                "mlp_parameter_dtype": "float32",
                "solver_state_dtype": str(updated.dtype),
                "linear_d_errors": linear_errors,
                "reflection_defect": defect,
                "periodic_mass_error": mass_error,
                "bidirectional_error": bidirectional_error,
                "gradient_norm": gradient_norm,
                "checkpoint_dtypes": checkpoint_dtypes,
                "checkpoint_roundtrip_error": checkpoint_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
