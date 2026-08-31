#!/usr/bin/env python3
"""Run the formal Titarev--Toro thin-strip benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import warp_weno5_helpers as wh
from for_paper_results import config
from for_paper_results.common import interior, state_health, write_json
from for_paper_results.solvers import euler_methods


METHODS = config.EULER_METHODS


def sine_integral(x_left: np.ndarray, x_right: np.ndarray) -> np.ndarray:
    k = 20.0 * np.pi
    return (
        x_right
        - x_left
        + 0.1 * (np.cos(k * x_left) - np.cos(k * x_right)) / k
    )


def make_initial_state(params, x_min: float, x_max: float) -> np.ndarray:
    """Exact finite-volume averages, including a possible cut shock cell."""
    g = params.ghost
    nx_total = params.nx + 2 * g
    ny_total = params.ny + 2 * g
    dx = (x_max - x_min) / float(params.nx)
    x_left = x_min + (np.arange(nx_total, dtype=np.float64) - g) * dx
    x_right = x_left + dx
    shock = -4.5

    left_length = np.clip(np.minimum(x_right, shock) - x_left, 0.0, dx)
    right_start = np.maximum(x_left, shock)
    right_length = np.clip(x_right - right_start, 0.0, dx)
    right_mass = np.where(
        right_length > 0.0,
        sine_integral(right_start, x_right),
        0.0,
    )

    left = wh.primitive_to_conserved(
        1.515695, 0.523346, 0.0, 1.805, params.gamma
    )
    right_constant = np.array(
        [0.0, 0.0, 0.0, 1.0 / (params.gamma - 1.0)], dtype=np.float64
    )
    cell = (
        left_length[:, None] * left[None, :]
        + right_length[:, None] * right_constant[None, :]
    ) / dx
    cell[:, 0] += right_mass / dx
    return np.broadcast_to(cell[None, :, :], (ny_total, nx_total, 4)).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--nx", type=int, default=1001)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--t-end", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ny < 10:
        raise ValueError(
            f"Titarev--Toro thin-strip validation requires --ny >= 10; got {args.ny}"
        )
    config.ensure_output_dirs()
    config.validate_models()
    x_length = args.x_max - args.x_min
    dx = x_length / float(args.nx)
    method = config.METHODS[args.method]
    if method.family.startswith("weno5"):
        params = euler_methods.make_weno5_params(
            args.nx,
            args.ny,
            x_length,
            args.ny * dx,
            args.cfl,
            args.t_end,
            mixed=args.method == "weno5_sr_f32",
        )
    else:
        params = euler_methods.make_weno7_params(
            args.nx,
            args.ny,
            args.x_min,
            args.x_max,
            0.0,
            args.ny * dx,
            args.cfl,
            args.t_end,
        )
    out_dir = args.out_dir or (
        config.RAW / "titarev_toro_cfl08" / f"N{args.nx}x{args.ny}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    initial = make_initial_state(params, args.x_min, args.x_max)
    print(
        f"titarev_start method={args.method} grid={args.nx}x{args.ny} "
        f"x=[{args.x_min},{args.x_max}] cfl={args.cfl} t_end={args.t_end} "
        f"integrator={method.time_integrator} space=characteristic solver=hllc "
        "boundary=transmissive cutoff=False",
        flush=True,
    )

    started = time.perf_counter()
    if method.family.startswith("weno5"):
        final, solver_summary = euler_methods.run_weno5(
            args.method,
            initial,
            params,
            device=args.device,
            boundary="transmissive",
            report_interval=args.report_interval,
        )
    else:
        final, solver_summary = euler_methods.run_weno7(
            args.method,
            initial,
            params,
            device=args.device,
            boundary="transmissive",
            report_interval=args.report_interval,
        )
    solve_seconds = time.perf_counter() - started
    health = state_health(final, params.ghost, args.nx, args.ny)
    complete_time = abs(float(solver_summary["t"]) - args.t_end) < 1.0e-12
    metadata = {
        "benchmark": "Titarev_Toro_shock_density_wave",
        "method": args.method,
        "label": config.METHODS[args.method].label,
        "model": str(config.METHODS[args.method].model)
        if config.METHODS[args.method].model
        else None,
        "nx": args.nx,
        "ny": args.ny,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "dx": dx,
        "cfl": args.cfl,
        "t_end": args.t_end,
        "t": float(solver_summary["t"]),
        "steps": int(solver_summary["steps"]),
        "dt_min": float(solver_summary["dt_min"]),
        "dt_max": float(solver_summary["dt_max"]),
        "dt_mean": float(solver_summary["dt_mean"]),
        "initialization": "exact_finite_volume_cell_average",
        "riemann_solver": "hllc",
        "weno_space": "characteristic",
        "time_integrator": method.time_integrator,
        "boundary": "transmissive",
        "eno_cutoff": False,
        "reflection_symmetrized_mlp": "_sr_" in args.method,
        "solve_seconds": solve_seconds,
        "complete_time": complete_time,
        **health,
    }
    metadata["complete"] = bool(metadata["complete"] and complete_time)
    state = interior(final, params.ghost, args.nx, args.ny)
    x = args.x_min + (np.arange(args.nx, dtype=np.float64) + 0.5) * dx
    np.savez(
        out_dir / f"{args.method}.npz",
        state=state,
        x=x,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    write_json(out_dir / f"{args.method}.json", metadata)
    print(f"saved_results={out_dir / f'{args.method}.npz'}", flush=True)
    if not metadata["complete"]:
        raise RuntimeError(f"Titarev--Toro validation failed: {metadata}")


if __name__ == "__main__":
    main()
