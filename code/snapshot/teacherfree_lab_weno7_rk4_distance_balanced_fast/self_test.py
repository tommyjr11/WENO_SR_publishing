#!/usr/bin/env python3
"""Fast algebraic and differentiability checks for the standalone package."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import rk4_advection as A
import weno7_core as W


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = W.torch_device(args.device)
    W.check_weno7_coefficients()
    order = A.check_shu_rk4_order()

    generator = torch.Generator(device=device)
    generator.manual_seed(20260723)
    q = torch.randn(
        (4096, 7), generator=generator, device=device, dtype=torch.float64
    )
    feature_error = torch.max(
        torch.abs(
            W.weno7_features(torch.flip(q, dims=(-1,)))
            - W.ReflectionSymmetricBadnessMLP.reflect_features(
                W.weno7_features(q)
            )
        )
    )
    if float(feature_error) > 2.0e-14:
        raise AssertionError(f"feature reflection error={float(feature_error)}")

    model = W.ReflectionSymmetricBadnessMLP(seed=17).to(device)
    features = W.weno7_features(q)
    ratio = model(features)
    if not torch.equal(ratio, torch.full_like(ratio, 0.25)):
        raise AssertionError("zero output layer does not produce uniform ratios")
    for lr in W.LR_VALUES:
        omega = W.omega_from_ratio(ratio, lr)
        target = W.optimal_d(lr, device).expand_as(omega)
        if float(torch.max(torch.abs(omega - target)).detach()) > 2.0e-15:
            raise AssertionError(f"linear initialization failed for lr={lr}")

    with torch.no_grad():
        model.w4.normal_(generator=generator)
        model.b4.normal_(generator=generator)
    defect = W.reflection_defect(model, features)
    if defect > 3.0e-15:
        raise AssertionError(f"reflection-equivariance defect={defect}")

    plus = A.reconstruct_q(model, q, 1)
    minus_reflected = A.reconstruct_q(
        model, torch.flip(q, dims=(-1,)), 2
    )
    reconstruction_error = float(
        torch.max(torch.abs(plus - minus_reflected)).detach()
    )
    if reconstruction_error > 2.0e-13:
        raise AssertionError(
            f"i+/i- reconstruction reflection error={reconstruction_error}"
        )

    state = torch.randn(
        (2, 48), generator=generator, device=device, dtype=torch.float64
    )
    velocity = A.balanced_velocities(2, device)
    mass_before = torch.sum(state, dim=1)
    updated = A.shu_rk4_step_signed(
        model, state, 0.2 / 48.0, 48.0, velocity
    )
    mass_error = float(
        torch.max(
            torch.abs(torch.sum(updated, dim=1) - mass_before)
        ).detach()
    )
    if mass_error > 2.0e-12:
        raise AssertionError(f"periodic mass conservation error={mass_error}")

    reflected_initial = torch.flip(state[:1], dims=(-1,))
    positive = A.shu_rk4_step_signed(
        model,
        state[:1],
        0.2 / 48.0,
        48.0,
        torch.ones(1, device=device),
    )
    negative_reflected = A.shu_rk4_step_signed(
        model,
        reflected_initial,
        0.2 / 48.0,
        48.0,
        -torch.ones(1, device=device),
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
            f"bidirectional RK reflection error={bidirectional_error}"
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
        W.save_checkpoint(checkpoint, model, {"self_test": True})
        loaded = W.load_checkpoint(checkpoint, device)
        loaded_output = loaded(features)
        checkpoint_error = float(
            torch.max(
                torch.abs(loaded_output - model(features))
            ).detach()
        )
        if checkpoint_error != 0.0:
            raise AssertionError(
                f"checkpoint round-trip error={checkpoint_error}"
            )
        data = np.load(checkpoint)
        if any(
            data[name].dtype != np.float64
            for name in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
        ):
            raise AssertionError("checkpoint parameters are not float64")

    report = {
        "device": str(device),
        "rk_local_order": order,
        "feature_reflection_error": float(feature_error),
        "model_reflection_defect": defect,
        "reconstruction_reflection_error": reconstruction_error,
        "periodic_mass_error": mass_error,
        "bidirectional_rk_reflection_error": bidirectional_error,
        "gradient_norm": gradient_norm,
        "checkpoint_roundtrip_error": checkpoint_error,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
