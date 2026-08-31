#!/usr/bin/env python3
"""Run the Lax and left Woodward--Colella shock-tube benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import warp_weno5_helpers as wh
from for_paper_results import config
from for_paper_results.common import interior, state_health, write_json
from for_paper_results.solvers import euler_methods


@dataclass(frozen=True)
class ShockTube:
    key: str
    title: str
    x_min: float
    x_max: float
    discontinuity: float
    t_end: float
    left: tuple[float, float, float]
    right: tuple[float, float, float]


BENCHMARKS = {
    "lax": ShockTube(
        key="lax",
        title="Lax shock tube",
        x_min=-5.0,
        x_max=5.0,
        discontinuity=0.0,
        t_end=1.3,
        left=(0.445, 0.698, 3.528),
        right=(0.500, 0.000, 0.571),
    ),
    "woodward_colella_half": ShockTube(
        key="woodward_colella_half",
        title="Left half of the Woodward--Colella blast wave",
        x_min=0.0,
        x_max=1.0,
        discontinuity=0.5,
        t_end=0.012,
        left=(1.0, 0.0, 1000.0),
        right=(1.0, 0.0, 0.01),
    ),
}


def make_initial_state(params, problem: ShockTube) -> np.ndarray:
    """Construct exact finite-volume averages, including a cut interface cell."""
    g = params.ghost
    nx_total = params.nx + 2 * g
    ny_total = params.ny + 2 * g
    dx = (problem.x_max - problem.x_min) / float(params.nx)
    x_left = problem.x_min + (np.arange(nx_total, dtype=np.float64) - g) * dx
    x_right = x_left + dx

    left_length = np.clip(
        np.minimum(x_right, problem.discontinuity) - x_left, 0.0, dx
    )
    right_length = dx - left_length
    rho_l, u_l, p_l = problem.left
    rho_r, u_r, p_r = problem.right
    conserved_l = wh.primitive_to_conserved(rho_l, u_l, 0.0, p_l, params.gamma)
    conserved_r = wh.primitive_to_conserved(rho_r, u_r, 0.0, p_r, params.gamma)
    cell = (
        left_length[:, None] * conserved_l[None, :]
        + right_length[:, None] * conserved_r[None, :]
    ) / dx
    return np.broadcast_to(cell[None, :, :], (ny_total, nx_total, 4)).copy()


def primitive_1d(conserved: np.ndarray, gamma: float) -> tuple[np.ndarray, ...]:
    rho = conserved[:, 0]
    if not np.all(np.isfinite(conserved)) or np.any(rho <= 0.0):
        raise RuntimeError("invalid conserved state; refusing diagnostic state repair")
    velocity = conserved[:, 1] / rho
    transverse_velocity = conserved[:, 2] / rho
    pressure = (gamma - 1.0) * (
        conserved[:, 3]
        - 0.5 * rho * (velocity * velocity + transverse_velocity * transverse_velocity)
    )
    if np.any(pressure <= 0.0) or not np.all(np.isfinite(pressure)):
        raise RuntimeError("invalid pressure; refusing diagnostic state repair")
    return rho, velocity, pressure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--method", choices=config.EULER_METHODS, required=True)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nx <= 0 or args.ny < 10:
        raise ValueError("shock-tube tests require nx > 0 and ny >= 10")
    if not 0.0 < args.cfl <= 0.8:
        raise ValueError(f"shock-tube CFL must lie in (0, 0.8]; got {args.cfl}")

    config.ensure_output_dirs()
    config.validate_models()
    problem = BENCHMARKS[args.benchmark]
    method = config.METHODS[args.method]
    x_length = problem.x_max - problem.x_min
    dx = x_length / float(args.nx)
    y_length = args.ny * dx
    if method.family.startswith("weno5"):
        params = euler_methods.make_weno5_params(
            args.nx,
            args.ny,
            x_length,
            y_length,
            args.cfl,
            problem.t_end,
            mixed=args.method == "weno5_sr_f32",
        )
    else:
        params = euler_methods.make_weno7_params(
            args.nx,
            args.ny,
            problem.x_min,
            problem.x_max,
            0.0,
            y_length,
            args.cfl,
            problem.t_end,
        )

    out_dir = args.out_dir or (
        config.RAW / "shock_tubes_cfl08" / problem.key / f"N{args.nx}x{args.ny}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    initial = make_initial_state(params, problem)
    print(
        f"shock_tube_start benchmark={problem.key} method={args.method} "
        f"grid={args.nx}x{args.ny} domain=[{problem.x_min},{problem.x_max}] "
        f"x0={problem.discontinuity} cfl={args.cfl} t_end={problem.t_end} "
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
    complete_time = abs(float(solver_summary["t"]) - problem.t_end) < 1.0e-12
    state = interior(final, params.ghost, args.nx, args.ny)
    conserved_1d = state.mean(axis=0)
    diagnostic_error = None
    try:
        rho, velocity, pressure = primitive_1d(conserved_1d, params.gamma)
    except RuntimeError as exc:
        diagnostic_error = str(exc)
        rho = velocity = pressure = None
    x = problem.x_min + (np.arange(args.nx, dtype=np.float64) + 0.5) * dx
    metadata = {
        "benchmark": problem.key,
        "title": problem.title,
        "method": args.method,
        "label": method.label,
        "model": str(method.model) if method.model else None,
        "nx": args.nx,
        "ny": args.ny,
        "x_min": problem.x_min,
        "x_max": problem.x_max,
        "discontinuity": problem.discontinuity,
        "left_primitive": list(problem.left),
        "right_primitive": list(problem.right),
        "dx": dx,
        "cfl": args.cfl,
        "t_end": problem.t_end,
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
        "state_repair": False,
        "solve_seconds": solve_seconds,
        "complete_time": complete_time,
        "diagnostic_error": diagnostic_error,
        **health,
    }
    metadata["complete"] = bool(
        metadata["complete"] and complete_time and diagnostic_error is None
    )
    if diagnostic_error is not None:
        failed_path = out_dir / f"{args.method}.failed.json"
        write_json(failed_path, metadata)
        print(f"saved_failure={failed_path}", flush=True)
        raise RuntimeError(
            f"shock-tube validation failed without state repair: {metadata}"
        )
    result_path = out_dir / f"{args.method}.npz"
    np.savez(
        result_path,
        state=state,
        conserved_1d=conserved_1d,
        rho=rho,
        velocity=velocity,
        pressure=pressure,
        x=x,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    write_json(out_dir / f"{args.method}.json", metadata)
    print(f"saved_results={result_path}", flush=True)
    if not metadata["complete"]:
        raise RuntimeError(f"shock-tube validation failed: {metadata}")


if __name__ == "__main__":
    main()
