#!/usr/bin/env python3
"""Run the order-matched WENO7-Z method for C.3--C.6."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from for_paper_results.common import state_health
from for_paper_results.run_quadrant import CASES, make_quadrant_state

from . import euler_z


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=50)
    args = parser.parse_args()
    if args.nx != args.ny:
        raise ValueError("quadrant tests require square meshes")
    if args.init_quadrature != 15:
        raise ValueError("formal quadrant tests require 15-point initialization")

    definition = CASES[args.case]
    t_end = float(definition["t_end"] if args.t_end is None else args.t_end)
    params = euler_z.make_weno7_params(
        args.nx, args.ny, 0.0, 1.0, 0.0, 1.0, args.cfl, t_end,
    )
    initial = make_quadrant_state(params, definition, args.init_quadrature)
    print(
        f"quadrant_z_start case={args.case} method=weno7_z_p3 "
        f"grid={args.nx}x{args.ny} cfl={args.cfl} t_end={t_end} "
        "quadrature=15x15 characteristic=True solver=hllc",
        flush=True,
    )
    started = time.perf_counter()
    final, summary = euler_z.run_weno7(
        initial, params, device=args.device, boundary="outflow",
        report_interval=args.report_interval,
    )
    elapsed = time.perf_counter() - started
    health = state_health(final, params.ghost, args.nx, args.ny)
    complete_time = abs(float(summary["t"]) - t_end) < 1.0e-12
    metadata = {
        "benchmark": "two-dimensional Riemann problem",
        "case": args.case,
        "configuration": definition["label"],
        "method": "weno7_z_p3",
        "label": "WENO7-Z-RK4",
        "nx": args.nx,
        "ny": args.ny,
        "cfl": args.cfl,
        "t_end": t_end,
        "t": float(summary["t"]),
        "steps": int(summary["steps"]),
        "dt_min": float(summary["dt_min"]),
        "dt_max": float(summary["dt_max"]),
        "dt_mean": float(summary["dt_mean"]),
        "wall_seconds_including_setup": elapsed,
        "time_integrator": "four-stage fourth-order downwind TVD-RK",
        "weno_space": "characteristic",
        "riemann_solver": "hllc",
        "boundary": "outflow",
        "initialization": "15x15_Gauss_Legendre_cell_average",
        "weno_z_tau": "abs(-beta0-3*beta1+3*beta2+beta3)",
        "weno_z_p": 3,
        "weno_z_epsilon": float(params.dx**3),
        "weno_z_epsilon_convention": (
            "paper epsilon=h^3; kernels use 240*h^3 because beta carries "
            "a common factor 240"
        ),
        "smoothness_evaluation": "nonnegative difference-square form",
        "complete_time": complete_time,
        **health,
    }
    metadata["complete"] = bool(metadata["complete"] and complete_time)
    state = euler_z.interior(final, params.ghost, args.nx, args.ny)
    out = ROOT / f"raw/riemann/{args.case}/N{args.nx}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "weno7_z_p3.npz",
        state=state,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    (out / "weno7_z_p3.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)
    if not metadata["complete"]:
        raise RuntimeError(f"incomplete WENO7-Z quadrant result: {metadata}")


if __name__ == "__main__":
    main()
