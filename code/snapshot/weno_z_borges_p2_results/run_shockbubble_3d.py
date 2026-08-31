#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import warp as wp

from warp_weno5_3d_rk3.binary_io import write_step

ROOT = Path(__file__).resolve().parent


def finite_diagnostics(solver: object) -> dict[str, float | int]:
    values = solver.primitive_host()
    finite = np.isfinite(values)
    rho = values[..., 0]
    pressure = values[..., 4]
    finite_rho = rho[np.isfinite(rho)]
    finite_pressure = pressure[np.isfinite(pressure)]
    return {
        "nan_count": int(np.count_nonzero(~finite)),
        "rho_min": float(np.min(finite_rho)) if finite_rho.size else float("nan"),
        "rho_max": float(np.max(finite_rho)) if finite_rho.size else float("nan"),
        "p_min": float(np.min(finite_pressure)) if finite_pressure.size else float("nan"),
        "p_max": float(np.max(finite_pressure)) if finite_pressure.size else float("nan"),
    }


def is_healthy(diagnostics: dict[str, float | int]) -> bool:
    return (
        diagnostics["nan_count"] == 0
        and np.isfinite(diagnostics["rho_min"])
        and np.isfinite(diagnostics["p_min"])
        and diagnostics["rho_min"] > 0.0
        and diagnostics["p_min"] > 0.0
    )


def build_solver(method: str, nx: int, ny: int, nz: int, cfl: float, t_end: float, device: str):
    if method == "weno5_z_p2":
        from warp_weno5_3d_rk3.config import ShockBubbleConfig
        from .weno5_3d_solver_z import Weno5ZRk3Solver

        config = replace(
            ShockBubbleConfig(), nx=nx, ny=ny, nz=nz, cfl=cfl, t_end=t_end,
        )
        solver = Weno5ZRk3Solver(config, device=device, strict_sync=False)
        label = "WENO5-Z-RK3"
        integrator = "SSPRK3"
        tau = "abs(beta0-beta2)"
        exponent = 2
    else:
        from warp_weno7_3d_rk4.config import ShockBubbleConfig
        from .weno7_3d_solver_z import Weno7ZRk4Solver

        config = replace(
            ShockBubbleConfig(), nx=nx, ny=ny, nz=nz, cfl=cfl, t_end=t_end,
        )
        solver = Weno7ZRk4Solver(config, device=device, strict_sync=False)
        label = "WENO7-Z-RK4"
        integrator = "four-stage-fourth-order-downwind-TVD-RK"
        tau = "abs(-beta0-3*beta1+3*beta2+beta3)"
        exponent = 3
    return solver, config, label, integrator, tau, exponent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated 3-D Ma=3 order-matched WENO-Z shock-bubble benchmark"
    )
    parser.add_argument("--method", choices=("weno5_z_p2", "weno7_z_p3"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nx", type=int, default=224)
    parser.add_argument("--ny", type=int, default=88)
    parser.add_argument("--nz", type=int, default=88)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--t-end", type=float, default=1.0e-4)
    parser.add_argument("--report-interval", type=int, default=25)
    parser.add_argument("--health-interval", type=int, default=25)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    wp.init()
    solver, config, label, integrator, tau, exponent = build_solver(
        args.method, args.nx, args.ny, args.nz, args.cfl, args.t_end, args.device,
    )
    out_dir = args.out_dir or (
        ROOT / f"raw/shockbubble_3d/N{args.nx}x{args.ny}x{args.nz}/{args.method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    solver.initialize()
    initial = finite_diagnostics(solver)
    print(f"initialized method={args.method} device={solver.device} diagnostics={initial}", flush=True)
    failure = None
    last_diagnostics = initial
    while solver.time != config.t_end:
        solver.advance()
        row = solver.dt_trace[-1]
        scalar_finite = all(
            np.isfinite(float(row[key]))
            for key in ("raw_dt", "dt", "time_end", "max_speed")
        )
        check_now = (
            solver.step == 1
            or solver.step % args.health_interval == 0
            or solver.time == config.t_end
            or not scalar_finite
        )
        if check_now:
            last_diagnostics = finite_diagnostics(solver)
            if not scalar_finite or not is_healthy(last_diagnostics):
                failure = f"nonphysical state at step {solver.step}"
        if (
            solver.step == 1
            or solver.step % args.report_interval == 0
            or solver.time == config.t_end
            or failure is not None
        ):
            print(
                f"step={solver.step:05d} t={row['time_end']:.8e} dt={row['dt']:.8e} "
                f"max_speed={row['max_speed']:.8e} diagnostics={last_diagnostics}",
                flush=True,
            )
        if failure is not None:
            break

    wp.synchronize_device(solver.device)
    elapsed = time.perf_counter() - started
    final = finite_diagnostics(solver)
    complete = solver.time == config.t_end and is_healthy(final)
    output = out_dir / f"step_{solver.step:04d}.bin"
    write_step(output, solver.time, solver.primitive_host())

    trace = out_dir / "dt_trace.csv"
    with trace.open("w", newline="", encoding="ascii") as stream:
        fields = ("step", "time_start", "raw_dt", "dt", "time_end", "max_speed")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(solver.dt_trace)

    manifest = {
        "benchmark": "three-dimensional Ma=3 shock-bubble",
        "method": args.method,
        "label": label,
        "config": asdict(config),
        "device": str(solver.device),
        "reconstruction": f"characteristic WENO-Z, p_Z={exponent}",
        "weno_z_tau": tau,
        "weno_z_p": exponent,
        "weno_z_epsilon": f"h^{exponent}_directionwise",
        "weno_z_epsilon_x": float(config.dx**exponent),
        "weno_z_epsilon_y": float(config.dy**exponent),
        "weno_z_epsilon_z": float(config.dz**exponent),
        "weno_z_epsilon_convention": f"paper epsilon=h^{exponent}",
        "smoothness_evaluation": (
            "nonnegative difference-square form"
            if args.method == "weno7_z_p3"
            else "Jiang-Shu difference-square form"
        ),
        "riemann_solver": "evilin",
        "boundary": "transmissive",
        "time_integrator": integrator,
        "initialization": "3x3x3 Gauss-Legendre cell average",
        "step": solver.step,
        "time": solver.time,
        "t_end": config.t_end,
        "complete": complete,
        "failure": failure,
        "elapsed_seconds": elapsed,
        "output": str(output),
        "initial_diagnostics": initial,
        "final_diagnostics": final,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if not complete:
        raise RuntimeError(f"incomplete three-dimensional WENO-Z run: {manifest}")


if __name__ == "__main__":
    main()
