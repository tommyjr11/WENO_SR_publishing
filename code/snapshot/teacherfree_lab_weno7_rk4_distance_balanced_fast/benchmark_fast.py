#!/usr/bin/env python3
"""Benchmark one original and one execution-optimized optimizer loss."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import fast_losses
import fvm_profiles
import losses
import rk4_advection as A
import weno7_core as W

torch.set_default_dtype(torch.float64)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--target-chunk", type=int, default=32)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "jit", "default", "reduce-overhead"),
        default="none",
    )
    parser.add_argument(
        "--skip-original", action="store_true", default=False
    )
    args = parser.parse_args()
    if args.batch < 2 or args.batch % 2:
        raise ValueError("batch must be positive and even")

    device = W.torch_device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260727)
    profile = fvm_profiles.make_profiles(
        args.batch, args.grid, device, generator
    )
    velocities = A.balanced_velocities(args.batch, device)
    cfl = 0.5
    edge_steps = min(40, args.steps)
    model = W.ReflectionSymmetricBadnessMLP(seed=37).to(device)
    with torch.no_grad():
        model.w4.normal_(mean=0.0, std=0.05, generator=generator)

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

    def original_iteration() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        primary, _ = losses.checkpointed_autoregressive_trajectory_loss(
            model,
            profile,
            args.grid,
            args.steps,
            cfl,
            velocities,
            **common,
        )
        edge, _ = losses.checkpointed_autoregressive_trajectory_loss(
            model,
            profile,
            args.grid,
            edge_steps,
            cfl,
            velocities,
            **edge_common,
        )
        total = primary + 0.25 * edge
        total.backward()
        return total

    step_module = fast_losses.TrajectoryStep(
        model,
        dt=cfl / float(args.grid),
        dxinv=float(args.grid),
        flat_tolerance=common["flat_tolerance"],
        local_window=common["local_window"],
        cvar_fraction=common["cvar_fraction"],
        guard_tolerance=common["guard_tolerance"],
    )
    classical_module = fast_losses.ClassicalStep(
        dt=cfl / float(args.grid), dxinv=float(args.grid)
    )
    if args.compile_mode == "none":
        step_operator = step_module
        classical_operator = classical_module
    elif args.compile_mode == "jit":
        step_operator = fast_losses.ShapeTracedCallable(
            step_module, "trajectory"
        )
        classical_operator = fast_losses.ShapeTracedCallable(
            classical_module, "classical"
        )
    else:
        step_operator = torch.compile(
            step_module, mode=args.compile_mode, dynamic=False
        )
        classical_operator = torch.compile(
            classical_module, mode=args.compile_mode, dynamic=False
        )

    def fast_iteration() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        total, _ = fast_losses.fast_autoregressive_trajectory_loss(
            step_operator,
            classical_operator,
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
            target_chunk=args.target_chunk,
        )
        total.backward()
        return total

    def measure(callable_) -> tuple[list[float], float]:
        callable_()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(args.repeats):
            synchronize(device)
            start = time.perf_counter()
            callable_()
            synchronize(device)
            samples.append(time.perf_counter() - start)
        peak = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        )
        return samples, peak

    original_times: list[float] = []
    original_peak = 0.0
    if not args.skip_original:
        original_times, original_peak = measure(original_iteration)
    fast_times, fast_peak = measure(fast_iteration)
    original_median = (
        statistics.median(original_times) if original_times else None
    )
    fast_median = statistics.median(fast_times)
    speedup = (
        original_median / fast_median
        if original_median is not None
        else None
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "grid": args.grid,
                "batch": args.batch,
                "rk_steps": args.steps,
                "compile_mode": args.compile_mode,
                "original_seconds": original_times,
                "original_median_seconds": original_median,
                "original_peak_mib": original_peak,
                "fast_seconds": fast_times,
                "fast_median_seconds": fast_median,
                "fast_peak_mib": fast_peak,
                "speedup": speedup,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
