#!/usr/bin/env python3
"""Run WENO5-Z and WENO7-Z on the formal Double-Mach reflection test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from for_paper_results.common import state_health
from run_double_mach_compare import make_initial_state

from . import euler_z
from . import weno5_double_mach_z
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
    parser.add_argument("--methods", default="weno5_z_p2,weno7_z_p3")
    parser.add_argument("--nx", type=int, default=1200)
    parser.add_argument("--ny", type=int, default=300)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    args = parser.parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(methods) - {"weno5_z_p2", "weno7_z_p3"}
    if unknown:
        raise ValueError(f"unsupported method(s): {sorted(unknown)}")
    if args.init_quadrature != 15:
        raise ValueError("formal Double-Mach test requires 15-point initialization")

    out = ROOT / f"raw/double_mach/N{args.nx}x{args.ny}"
    out.mkdir(parents=True, exist_ok=True)
    for method in methods:
        if method == "weno5_z_p2":
            params = euler_z.make_weno5_params(
                args.nx, args.ny, 4.0, 1.0, args.cfl, args.t_end,
            )
            initial = make_initial_state(params, args.init_quadrature)
            final, dts, steps, t = weno5_double_mach_z.run_to_time(
                initial, params, args.t_end, args.device,
                report_interval=args.report_interval, report=report(method),
            )
            order, exponent, integrator = 5, 2, "SSPRK3"
        else:
            params = euler_z.make_weno7_params(
                args.nx, args.ny, 0.0, 4.0, 0.0, 1.0,
                args.cfl, args.t_end,
            )
            initial = make_initial_state(params, args.init_quadrature)
            final, dts, steps, t = weno7_double_mach_z.run_to_time(
                initial, params, args.t_end, args.device,
                report_interval=args.report_interval, report=report(method),
            )
            order, exponent, integrator = 7, 3, "four-stage fourth-order downwind TVD-RK"
        state = euler_z.interior(final, params.ghost, args.nx, args.ny)
        health = state_health(final, params.ghost, args.nx, args.ny)
        complete_time = abs(float(t) - args.t_end) < 1.0e-12
        metadata = {
            "benchmark": "double-Mach reflection",
            "method": method,
            "label": f"WENO{order}-Z-{'RK3' if order == 5 else 'RK4'}",
            "nx": args.nx,
            "ny": args.ny,
            "cfl": args.cfl,
            "t_end": args.t_end,
            "t": float(t),
            "steps": int(steps),
            "dt_min": float(min(dts)) if dts else 0.0,
            "dt_max": float(max(dts)) if dts else 0.0,
            "dt_mean": float(np.mean(dts)) if dts else 0.0,
            "time_integrator": integrator,
            "weno_space": "characteristic",
            "riemann_solver": "hllc",
            "boundary": "double-mach time-dependent exact/reflective/outflow",
            "initialization": "15x15_Gauss_Legendre_cell_average",
            "weno_z_tau": (
                "abs(beta0-beta2)" if order == 5
                else "abs(-beta0-3*beta1+3*beta2+beta3)"
            ),
            "weno_z_p": exponent,
            "weno_z_epsilon": float(params.dx**exponent),
            "weno_z_epsilon_convention": (
                "paper epsilon=dx^2"
                if order == 5 else
                "paper epsilon=dx^3; kernels use 240*dx^3 because beta carries a common factor 240"
            ),
            "smoothness_evaluation": (
                "nonnegative difference-square form" if order == 7
                else "Jiang-Shu difference-square form"
            ),
            "complete_time": complete_time,
            **health,
        }
        metadata["complete"] = bool(metadata["complete"] and complete_time)
        np.savez(
            out / f"{method}.npz", state=state,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
        (out / f"{method}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, sort_keys=True), flush=True)
        if not metadata["complete"]:
            raise RuntimeError(f"incomplete WENO-Z Double-Mach result: {metadata}")


if __name__ == "__main__":
    main()
