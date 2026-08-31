#!/usr/bin/env python3
"""Run the smooth periodic WENO5/RK3 accuracy test on [-1, 1]^2."""

from __future__ import annotations

import argparse
import math

import numpy as np

import warp_weno5_helpers as wh
from weno5_rk3_warp import allocate_warp_arrays, launch_weno_rk3_step


def smooth_rho_cell_average_x(x_center: float, dx: float, t: float) -> float:
    xq = x_center + 0.5 * dx * wh.GAUSS15_XI
    rho = 2.0 + np.sin(np.pi * (xq - t)) ** 4
    return float(0.5 * np.sum(wh.GAUSS15_W * rho))


def make_smooth_periodic_state(params: wh.Params, x_min: float, y_min: float, t: float) -> np.ndarray:
    del y_min
    ny_total, nx_total, _ = params.padded_shape
    u = np.zeros(params.padded_shape, dtype=np.float64)
    rho_x = np.empty(nx_total, dtype=np.float64)

    for i in range(nx_total):
        x = x_min + (i - params.ghost + 0.5) * params.dx
        rho_x[i] = smooth_rho_cell_average_x(x, params.dx, t)

    energy_x = 0.5 * rho_x + 1.0 / (params.gamma - 1.0)
    for j in range(ny_total):
        u[j, :, 0] = rho_x
        u[j, :, 1] = rho_x
        u[j, :, 2] = 0.0
        u[j, :, 3] = energy_x

    return u


def density_error(warp_u: np.ndarray, exact_u: np.ndarray, params: wh.Params) -> dict[str, float]:
    gc = params.ghost
    pred = warp_u[gc : gc + params.ny, gc : gc + params.nx, 0]
    exact = exact_u[gc : gc + params.ny, gc : gc + params.nx, 0]
    diff = pred - exact
    abs_diff = np.abs(diff)
    dxdy = params.dx * params.dy
    return {
        "l1_mean": float(np.mean(abs_diff)),
        "l2_rms": float(np.sqrt(np.mean(diff * diff))),
        "linf": float(np.max(abs_diff)),
        "l1_integral": float(np.sum(abs_diff) * dxdy),
        "l2_integral": float(np.sqrt(np.sum(diff * diff) * dxdy)),
    }


def run_solver(
    label: str,
    u0_host: np.ndarray,
    exact_host: np.ndarray,
    params: wh.Params,
    args: argparse.Namespace,
    fixed_dt: float | None = None,
) -> tuple[dict[str, float], dict[str, float], float, int]:
    arrays = allocate_warp_arrays(u0_host, params, args.device)
    t = 0.0
    step = 0
    min_cfl_dt = float("inf")
    min_used_dt = float("inf")

    while t < args.t_end:
        if fixed_dt is None:
            dt_cfl = wh.compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, args.device)
            min_cfl_dt = min(min_cfl_dt, dt_cfl)
            dt = dt_cfl
        else:
            dt = fixed_dt

        if t + dt > args.t_end:
            dt = args.t_end - t
        else:
            min_used_dt = min(min_used_dt, dt)

        if dt <= 0.0:
            break

        step += 1
        launch_weno_rk3_step(
            arrays,
            params,
            dt,
            args.device,
            args.weno_space == "characteristic",
            "periodic",
        )
        t += dt

        if args.report_interval > 0 and (step == 1 or step % args.report_interval == 0 or t >= args.t_end):
            print(f"{label}: step={step} t={t:.16e} dt={dt:.16e}")

    final_host = arrays["u"].numpy()
    stats = wh.interior_stats(final_host, params)
    err = density_error(final_host, exact_host, params)
    if fixed_dt is not None:
        min_cfl_dt = fixed_dt
    if min_used_dt == float("inf"):
        min_used_dt = min_cfl_dt
    return stats, err, min_cfl_dt, step


def print_result(label: str, stats: dict[str, float], err: dict[str, float], dt: float, steps: int) -> None:
    print(
        f"{label}_done: steps={steps} reference_dt={dt:.16e} "
        f"mass={stats['mass']:.16e} rho=[{stats['rho_min']:.16e},{stats['rho_max']:.16e}] "
        f"p=[{stats['p_min']:.16e},{stats['p_max']:.16e}] nan={int(stats['nan_count'])} "
        f"rho_neg={int(stats['rho_neg'])} p_neg={int(stats['p_neg'])}"
    )
    print(
        f"{label}_density_error: "
        f"L1_mean={err['l1_mean']:.16e} "
        f"L2_rms={err['l2_rms']:.16e} "
        f"Linf={err['linf']:.16e}"
    )
    print(
        f"{label}_density_integral_norms: "
        f"L1={err['l1_integral']:.16e} "
        f"L2={err['l2_integral']:.16e}"
    )


def run(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp = wh.wp
    wp.init()
    wp.set_device(args.device)

    params = wh.Params(
        nx=args.nx,
        ny=args.ny,
        x_length=args.x_max - args.x_min,
        y_length=args.y_max - args.y_min,
        cfl=args.cfl,
        t_end=args.t_end,
    )
    u0_host = make_smooth_periodic_state(params, args.x_min, args.y_min, 0.0)
    exact_host = make_smooth_periodic_state(params, args.x_min, args.y_min, args.t_end)

    print(
        f"start: case=smooth_periodic nx={params.nx} ny={params.ny} "
        f"domain=[{args.x_min},{args.x_max}]x[{args.y_min},{args.y_max}] "
        f"T={args.t_end:.16e} cfl={args.cfl:.16e} exact=15point_gauss_cell_average"
    )

    cfl_stats, cfl_err, min_cfl_dt, cfl_steps = run_solver("cfl", u0_host, exact_host, params, args)
    print_result("cfl", cfl_stats, cfl_err, min_cfl_dt, cfl_steps)

    half_dt = 0.5 * min_cfl_dt
    half_stats, half_err, _, half_steps = run_solver("half_min_dt", u0_host, exact_host, params, args, half_dt)
    print_result("half_min_dt", half_stats, half_err, half_dt, half_steps)

    print(
        "error_ratio_half_over_cfl: "
        f"L1_mean={half_err['l1_mean'] / cfl_err['l1_mean']:.16e} "
        f"L2_rms={half_err['l2_rms'] / cfl_err['l2_rms']:.16e} "
        f"Linf={half_err['linf'] / cfl_err['linf']:.16e}"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--ny", type=int, default=500)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=-1.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weno-space", choices=("characteristic", "conserved"), default="characteristic")
    parser.add_argument("--report-interval", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
