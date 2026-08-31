#!/usr/bin/env python3
"""Compare the fast execution path against the original loss and gradients."""
from __future__ import annotations

import argparse
import json

import torch

import fast_losses
import fvm_profiles
import losses
import rk3_advection as A
import weno5_core as W
from model import ReflectionSymmetricBadnessMLPFloat32

torch.set_default_dtype(torch.float64)


def max_parameter_gradient_difference(
    left: torch.nn.Module, right: torch.nn.Module
) -> tuple[float, float]:
    absolute_max = 0.0
    relative_max = 0.0
    for left_parameter, right_parameter in zip(
        left.parameters(), right.parameters(), strict=True
    ):
        if left_parameter.grad is None or right_parameter.grad is None:
            raise AssertionError("a parameter gradient is missing")
        difference = torch.max(
            torch.abs(left_parameter.grad - right_parameter.grad)
        )
        scale = torch.maximum(
            torch.max(torch.abs(left_parameter.grad)),
            torch.max(torch.abs(right_parameter.grad)),
        )
        absolute_max = max(absolute_max, float(difference))
        relative_max = max(
            relative_max,
            float(difference / torch.clamp(scale, min=1.0e-30)),
        )
    return absolute_max, relative_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--chunk-steps", type=int, default=2)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "jit", "default", "reduce-overhead"),
        default="none",
    )
    args = parser.parse_args()

    if args.batch < 2 or args.batch % 2:
        raise ValueError("batch must be positive and even")
    device = W.torch_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    generator = torch.Generator(device=device)
    generator.manual_seed(20260727)
    profile = fvm_profiles.make_profiles(
        args.batch, args.grid, device, generator
    )
    velocities = A.balanced_velocities(args.batch, device)
    cfl = 0.5
    dt = cfl / float(args.grid)
    edge_steps = min(4, args.steps)

    exact_states, exact_points = fast_losses.precompute_exact_trajectory(
        profile,
        args.grid,
        args.steps,
        dt,
        velocities,
        target_chunk=3,
    )
    scalar_states = []
    scalar_points = []
    for step in range(args.steps + 1):
        time = float(step) * dt
        scalar_states.append(
            losses.exact_cell_average_signed(
                profile, args.grid, time, velocities
            )
        )
        if step < args.steps:
            scalar_points.append(
                losses.exact_point_targets_signed(
                    profile,
                    args.grid,
                    time,
                    A.LR_OFFSETS,
                    velocities,
                )
            )
    scalar_states_tensor = torch.stack(scalar_states)
    scalar_points_tensor = torch.stack(scalar_points)
    state_target_error = float(
        torch.max(torch.abs(exact_states - scalar_states_tensor))
    )
    point_target_error = float(
        torch.max(torch.abs(exact_points - scalar_points_tensor))
    )

    original_model = ReflectionSymmetricBadnessMLPFloat32(seed=91).to(
        device
    )
    with torch.no_grad():
        original_model.w4.normal_(
            mean=0.0, std=0.08, generator=generator
        )
        original_model.b4.normal_(
            mean=0.0, std=0.03, generator=generator
        )
    fast_model = ReflectionSymmetricBadnessMLPFloat32(seed=0).to(device)
    fast_model.load_state_dict(original_model.state_dict())
    if {parameter.dtype for parameter in fast_model.parameters()} != {
        torch.float32
    }:
        raise AssertionError("the mixed-precision MLP is not entirely FP32")

    common = {
        "state_lambda": 1.0,
        "face_path_lambda": 0.04,
        "exact_recon_lambda": 0.15,
        "flat_d2_lambda": 0.05,
        "flat_tolerance": 2.0e-3,
        "tv_lambda": 0.03,
        "global_guard_lambda": 1.0,
        "local_guard_lambda": 1.0,
        "local_window": 8,
        "cvar_fraction": 0.25,
        "guard_tolerance": 0.0,
    }
    original_primary, primary_stats = (
        losses.checkpointed_autoregressive_trajectory_loss(
            original_model,
            profile,
            args.grid,
            args.steps,
            cfl,
            velocities,
            **common,
        )
    )
    edge_common = dict(common)
    edge_common.update(
        {
            "state_lambda": 0.0,
            "face_path_lambda": 0.0,
            "exact_recon_lambda": 0.0,
            "flat_d2_lambda": 0.0,
            "tv_lambda": 0.0,
        }
    )
    original_edge, edge_stats = (
        losses.checkpointed_autoregressive_trajectory_loss(
            original_model,
            profile,
            args.grid,
            edge_steps,
            cfl,
            velocities,
            **edge_common,
        )
    )
    original_total = original_primary + 0.25 * original_edge
    original_total.backward()

    step_operator = fast_losses.TrajectoryStep(
        fast_model,
        dt=dt,
        dxinv=float(args.grid),
        flat_tolerance=common["flat_tolerance"],
        local_window=common["local_window"],
        cvar_fraction=common["cvar_fraction"],
        guard_tolerance=common["guard_tolerance"],
    )
    classical_step = fast_losses.ClassicalStep(
        dt=dt, dxinv=float(args.grid)
    )
    if args.compile_mode == "jit":
        step_operator = fast_losses.ShapeTracedCallable(
            step_operator, "trajectory"
        )
        classical_step = fast_losses.ShapeTracedCallable(
            classical_step, "classical"
        )
    elif args.compile_mode != "none":
        step_operator = torch.compile(
            step_operator, mode=args.compile_mode, dynamic=False
        )
        classical_step = torch.compile(
            classical_step, mode=args.compile_mode, dynamic=False
        )
    fast_total, fast_stats = (
        fast_losses.fast_autoregressive_trajectory_loss(
            step_operator,
            classical_step,
            profile,
            args.grid,
            args.steps,
            cfl,
            velocities,
            state_lambda=1.0,
            face_path_lambda=common["face_path_lambda"],
            exact_recon_lambda=common["exact_recon_lambda"],
            flat_d2_lambda=common["flat_d2_lambda"],
            tv_lambda=common["tv_lambda"],
            global_guard_lambda=common["global_guard_lambda"],
            local_guard_lambda=common["local_guard_lambda"],
            edge_steps=edge_steps,
            edge_lambda=0.25,
            chunk_steps=args.chunk_steps,
            target_chunk=3,
        )
    )
    fast_total.backward()

    loss_absolute_error = abs(
        float((fast_total - original_total).detach())
    )
    loss_relative_error = loss_absolute_error / max(
        abs(float(original_total.detach())), 1.0e-30
    )
    gradient_absolute_error, gradient_relative_error = (
        max_parameter_gradient_difference(original_model, fast_model)
    )

    component_errors = {
        name: abs(float(fast_stats[name]) - float(primary_stats[name]))
        for name in fast_losses.PRIMARY_COMPONENTS
    }
    component_errors.update(
        {
            "edge_global_js_guard": abs(
                float(fast_stats["edge_global_js_guard"])
                - float(edge_stats["global_js_guard"])
            ),
            "edge_local_js_guard": abs(
                float(fast_stats["edge_local_js_guard"])
                - float(edge_stats["local_js_guard"])
            ),
        }
    )
    max_component_error = max(component_errors.values())

    target_tolerance = 2.0e-11 if device.type == "cpu" else 2.0e-10
    if device.type == "cuda" and args.compile_mode == "jit":
        arithmetic_tolerance = 2.0e-8
        gradient_tolerance = 5.0e-6
    else:
        arithmetic_tolerance = target_tolerance
        gradient_tolerance = 2.0e-6
    if state_target_error > target_tolerance:
        raise AssertionError(
            f"vectorized cell-average targets differ by {state_target_error}"
        )
    if point_target_error > target_tolerance:
        raise AssertionError(
            f"vectorized point targets differ by {point_target_error}"
        )
    if loss_relative_error > arithmetic_tolerance:
        raise AssertionError(
            f"total loss relative error={loss_relative_error}"
        )
    if gradient_relative_error > gradient_tolerance:
        raise AssertionError(
            f"parameter gradient relative error={gradient_relative_error}"
        )
    if max_component_error > arithmetic_tolerance:
        raise AssertionError(
            f"loss component max error={max_component_error}"
        )

    print(
        json.dumps(
            {
                "device": str(device),
                "compile_mode": args.compile_mode,
                "state_target_max_abs": state_target_error,
                "point_target_max_abs": point_target_error,
                "loss_absolute_error": loss_absolute_error,
                "loss_relative_error": loss_relative_error,
                "gradient_max_abs": gradient_absolute_error,
                "gradient_max_relative": gradient_relative_error,
                "component_max_abs": max_component_error,
                "component_errors": component_errors,
                "target_tolerance": target_tolerance,
                "arithmetic_tolerance": arithmetic_tolerance,
                "gradient_tolerance": gradient_tolerance,
                "status": "equivalent",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
