#!/usr/bin/env python3
"""Run raw-beta WENO7-Z p=1/p=2 on the formal Double-Mach test."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from for_paper_results.common import state_health
from run_double_mach_compare import make_initial_state

from . import euler_z
from . import weno7_double_mach_z


ROOT = Path(__file__).resolve().parent


def report(method: str):
    def callback(step: int, t: float, dt: float, stats: dict[str, float]) -> None:
        print(
            f"{method} step={step:05d} t={t:.8e} dt={dt:.4e} "
            f"rho=[{stats['rho_min']:.5e},{stats['rho_max']:.5e}] "
            f"p=[{stats['p_min']:.5e},{stats['p_max']:.5e}] "
            f"nan={int(stats['nan_count'])}",
            flush=True,
        )
    return callback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--powers", default="1,2")
    parser.add_argument("--nx", type=int, default=1200)
    parser.add_argument("--ny", type=int, default=300)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    args = parser.parse_args()
    powers = [int(value) for value in args.powers.split(",") if value.strip()]
    if not powers or set(powers) - {1, 2}:
        raise ValueError("--powers must contain only 1 and/or 2")
    if args.init_quadrature != 15:
        raise ValueError("formal Double-Mach test requires 15-point initialization")

    out = ROOT / f"raw_beta_h3_p12_audit/double_mach/N{args.nx}x{args.ny}"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for power in powers:
        method = f"weno7_z_p{power}"
        params = euler_z.make_weno7_params(
            args.nx, args.ny, 0.0, 4.0, 0.0, 1.0,
            args.cfl, args.t_end,
        )
        initial = make_initial_state(params, args.init_quadrature)
        started = time.perf_counter()
        try:
            final, dts, steps, reached = weno7_double_mach_z.run_to_time(
                initial,
                params,
                args.t_end,
                args.device,
                weight_kind=4 if power == 1 else 5,
                report_interval=args.report_interval,
                report=report(method),
            )
            exception = ""
        except Exception as error:  # Preserve the other exponent audit on failure.
            final = initial
            dts = []
            steps = 0
            reached = 0.0
            exception = f"{type(error).__name__}: {error}"
            print(f"{method} exception: {exception}", flush=True)
        wall_seconds = time.perf_counter() - started
        state = euler_z.interior(final, params.ghost, args.nx, args.ny)
        health = state_health(final, params.ghost, args.nx, args.ny)
        complete_time = abs(float(reached) - args.t_end) < 1.0e-12
        metadata = {
            "benchmark": "double-Mach reflection",
            "method": method,
            "label": "WENO7-Z-RK4",
            "weno_z_p": power,
            "nx": args.nx,
            "ny": args.ny,
            "cfl": args.cfl,
            "t_end": args.t_end,
            "t": float(reached),
            "steps": int(steps),
            "wall_seconds": float(wall_seconds),
            "dt_min": float(min(dts)) if dts else 0.0,
            "dt_max": float(max(dts)) if dts else 0.0,
            "dt_mean": float(np.mean(dts)) if dts else 0.0,
            "time_integrator": "four-stage fourth-order downwind TVD-RK",
            "weno_space": "characteristic",
            "riemann_solver": "hllc",
            "boundary": "double-mach time-dependent exact/reflective/outflow",
            "initialization": "15x15_Gauss_Legendre_cell_average",
            "beta_convention": "beta_raw_with_common_factor_240",
            "weno_z_tau": "abs(-beta0-3*beta1+3*beta2+beta3)",
            "weno_z_epsilon": float(params.dx**3),
            "weno_z_epsilon_convention": "epsilon=dx^3 without beta rescaling",
            "complete_time": complete_time,
            "exception": exception,
            **health,
        }
        metadata["complete"] = bool(metadata["complete"] and complete_time and not exception)
        np.savez(
            out / f"{method}.npz",
            state=state,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
        (out / f"{method}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(metadata)
        print(json.dumps(metadata, sort_keys=True), flush=True)

    (out / "completion_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(out / "completion_summary.json")


if __name__ == "__main__":
    main()
