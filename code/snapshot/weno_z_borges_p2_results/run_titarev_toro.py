#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from for_paper_results.common import state_health, write_json
from for_paper_results.run_weno5_titarev_toro import make_initial_state
from . import euler_z


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Titarev--Toro benchmark with p_Z=2 (WENO5) and p_Z=3 (WENO7)"
    )
    parser.add_argument("--nx", type=int, choices=(1001, 2000), required=True)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--t-end", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=500)
    args = parser.parse_args()
    if args.ny < 10:
        raise ValueError("formal Titarev--Toro run requires ny >= 10")

    x_length = args.x_max - args.x_min
    dx = x_length / args.nx
    out = ROOT / f"raw/titarev_toro_cfl08/N{args.nx}x{args.ny}"
    out.mkdir(parents=True, exist_ok=True)
    for order in (5, 7):
        if order == 5:
            params = euler_z.make_weno5_params(
                args.nx, args.ny, x_length, args.ny * dx, args.cfl, args.t_end,
            )
            run = euler_z.run_weno5
            integrator = "SSPRK3"
        else:
            params = euler_z.make_weno7_params(
                args.nx, args.ny, args.x_min, args.x_max,
                0.0, args.ny * dx, args.cfl, args.t_end,
            )
            run = euler_z.run_weno7
            integrator = "Shu-RK4"

        initial = make_initial_state(params, args.x_min, args.x_max)
        started = time.perf_counter()
        final, summary = run(
            initial,
            params,
            device=args.device,
            boundary="transmissive",
            report_interval=args.report_interval,
        )
        elapsed = time.perf_counter() - started
        health = state_health(final, params.ghost, args.nx, args.ny)
        complete_time = abs(float(summary["t"]) - args.t_end) < 1.0e-12
        state = euler_z.interior(final, params.ghost, args.nx, args.ny)
        x = args.x_min + (np.arange(args.nx, dtype=np.float64) + 0.5) * dx
        power = 2 if order == 5 else 3
        key = f"weno{order}_z_p{power}"
        metadata = {
            "benchmark": "Titarev_Toro_shock_density_wave",
            "method": key,
            "label": f"WENO{order}-Z (p={power})-RK{3 if order == 5 else 4}",
            "nx": args.nx,
            "ny": args.ny,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "dx": dx,
            "cfl": args.cfl,
            "t_end": args.t_end,
            "t": float(summary["t"]),
            "steps": int(summary["steps"]),
            "dt_min": float(summary["dt_min"]),
            "dt_max": float(summary["dt_max"]),
            "dt_mean": float(summary["dt_mean"]),
            "initialization": "exact_finite_volume_cell_average",
            "riemann_solver": "hllc",
            "weno_space": "characteristic",
            "time_integrator": integrator,
            "boundary": "transmissive",
            "eno_cutoff": False,
            "weno_z_p": power,
            "weno_z_tau": (
                "abs(beta0-beta2)" if order == 5
                else "abs(-beta0-3*beta1+3*beta2+beta3)"
            ),
            "weno_z_epsilon": float(dx**power),
            "weno_z_epsilon_convention": f"paper epsilon=dx^{power}",
            "solve_seconds": elapsed,
            "complete_time": complete_time,
            **health,
        }
        metadata["complete"] = bool(metadata["complete"] and complete_time)
        np.savez(
            out / f"{key}.npz",
            state=state,
            x=x,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
        write_json(out / f"{key}.json", metadata)
        print(json.dumps(metadata, sort_keys=True), flush=True)
        if not metadata["complete"]:
            raise RuntimeError(f"incomplete Titarev--Toro result: {metadata}")


if __name__ == "__main__":
    main()
