#!/usr/bin/env python3
"""Run the formal shock--helium-bubble benchmark.

Only the established characteristic HLLC adapters are used. WENO5 methods use
SSPRK3 and WENO7 methods use Shu's SSP-RK4. Learned methods retain the
reflection-symmetric inference and MLP precision selected in
:mod:`for_paper_results.config`.
"""

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

GAMMA = 1.4
SHOCK_MACH = 3.0
RHO_AIR = 1.29
P_AIR = 101325.0
RHO_HELIUM = 0.214
U_POST = 736.911
X_SHOCK = 0.005
BUBBLE_X = 0.035
BUBBLE_Y = 0.0445
BUBBLE_RADIUS = 0.025


def post_shock_primitive(gamma: float) -> tuple[float, float, float]:
    rho_post = (
        ((gamma + 1.0) * SHOCK_MACH * SHOCK_MACH)
        / ((gamma - 1.0) * SHOCK_MACH * SHOCK_MACH + 2.0)
        * RHO_AIR
    )
    p_post = (
        ((2.0 * gamma) * SHOCK_MACH * SHOCK_MACH - (gamma - 1.0))
        / (gamma + 1.0)
        * P_AIR
    )
    return rho_post, U_POST, p_post


def proportional_ny(nx: int, x_length: float, y_length: float) -> int:
    return max(1, int(round(nx * y_length / x_length)))


def make_initial_state(params) -> np.ndarray:
    """Vectorized equivalent of the trusted 15x15 Gauss initialization."""
    g = params.ghost
    nx_total = params.nx + 2 * g
    ny_total = params.ny + 2 * g
    xi = wh.GAUSS15_XI
    weights = wh.GAUSS15_W

    x_min = float(getattr(params, "x_min", 0.0))
    y_min = float(getattr(params, "y_min", 0.0))
    x_centers = x_min + (np.arange(nx_total, dtype=np.float64) - g + 0.5) * params.dx
    y_centers = y_min + (np.arange(ny_total, dtype=np.float64) - g + 0.5) * params.dy
    xq = x_centers[:, None] + 0.5 * params.dx * xi[None, :]

    air = wh.primitive_to_conserved(RHO_AIR, 0.0, 0.0, P_AIR, params.gamma)
    rho_post, u_post, p_post = post_shock_primitive(params.gamma)
    post = wh.primitive_to_conserved(
        rho_post, u_post, 0.0, p_post, params.gamma
    )
    helium = wh.primitive_to_conserved(
        RHO_HELIUM, 0.0, 0.0, P_AIR, params.gamma
    )

    post_fraction = 0.5 * np.sum(
        (xq < X_SHOCK) * weights[None, :], axis=1
    )
    state = air[None, None, :] + post_fraction[None, :, None] * (
        post - air
    )[None, None, :]
    state = np.broadcast_to(state, (ny_total, nx_total, 4)).copy()

    x_radius2 = (xq - BUBBLE_X) ** 2
    gauss_weight_2d = weights[:, None] * weights[None, :]
    chunk_rows = 16
    for j0 in range(0, ny_total, chunk_rows):
        j1 = min(j0 + chunk_rows, ny_total)
        yq = (
            y_centers[j0:j1, None]
            + 0.5 * params.dy * xi[None, :]
        )
        y_radius2 = (yq - BUBBLE_Y) ** 2
        inside = (
            y_radius2[:, None, :, None]
            + x_radius2[None, :, None, :]
            <= BUBBLE_RADIUS**2
        )
        bubble_fraction = 0.25 * np.sum(
            inside * gauss_weight_2d[None, None, :, :], axis=(2, 3)
        )
        state[j0:j1] += bubble_fraction[..., None] * (
            helium - air
        )[None, None, :]

    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=0)
    parser.add_argument("--x-length", type=float, default=0.225)
    parser.add_argument("--y-length", type=float, default=0.089)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    config.validate_models()
    ny = args.ny or proportional_ny(args.nx, args.x_length, args.y_length)
    method = config.METHODS[args.method]
    if method.family.startswith("weno5"):
        params = euler_methods.make_weno5_params(
            args.nx,
            ny,
            args.x_length,
            args.y_length,
            args.cfl,
            args.t_end,
            mixed=args.method == "weno5_sr_f32",
        )
    else:
        params = euler_methods.make_weno7_params(
            args.nx,
            ny,
            0.0,
            args.x_length,
            0.0,
            args.y_length,
            args.cfl,
            args.t_end,
        )
    out_dir = args.out_dir or (
        config.RAW / "shockbubble_ma3_t0001_cfl04" / f"N{args.nx}x{ny}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{args.method}.npz"
    out_json = out_dir / f"{args.method}.json"

    init_started = time.perf_counter()
    initial = make_initial_state(params)
    init_seconds = time.perf_counter() - init_started
    print(
        f"shockbubble_start mach={SHOCK_MACH:.1f} method={args.method} grid={args.nx}x{ny} "
        f"cfl={args.cfl} t_end={args.t_end:.8e} init_seconds={init_seconds:.3f} "
        f"integrator={method.time_integrator} space=characteristic solver=hllc "
        "boundary=transmissive cutoff=False "
        "quadrature=15x15",
        flush=True,
    )

    solve_started = time.perf_counter()
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
    solve_seconds = time.perf_counter() - solve_started
    health = state_health(final, params.ghost, args.nx, ny)
    complete_time = abs(float(solver_summary["t"]) - args.t_end) < 1.0e-12
    metadata = {
        "benchmark": "shock_helium_bubble",
        "shock_mach": SHOCK_MACH,
        "rho_air": RHO_AIR,
        "p_air": P_AIR,
        "rho_helium": RHO_HELIUM,
        "rho_post": post_shock_primitive(params.gamma)[0],
        "u_post": U_POST,
        "p_post": post_shock_primitive(params.gamma)[2],
        "x_shock": X_SHOCK,
        "bubble_center": [BUBBLE_X, BUBBLE_Y],
        "bubble_radius": BUBBLE_RADIUS,
        "method": args.method,
        "label": config.METHODS[args.method].label,
        "model": str(config.METHODS[args.method].model)
        if config.METHODS[args.method].model
        else None,
        "nx": args.nx,
        "ny": ny,
        "x_length": args.x_length,
        "y_length": args.y_length,
        "dx": params.dx,
        "dy": params.dy,
        "cfl": args.cfl,
        "t_end": args.t_end,
        "t": float(solver_summary["t"]),
        "steps": int(solver_summary["steps"]),
        "dt_min": float(solver_summary["dt_min"]),
        "dt_max": float(solver_summary["dt_max"]),
        "dt_mean": float(solver_summary["dt_mean"]),
        "initialization": "15x15_Gauss_Legendre_cell_average",
        "riemann_solver": "hllc",
        "weno_space": "characteristic",
        "time_integrator": method.time_integrator,
        "boundary": "transmissive",
        "eno_cutoff": False,
        "reflection_symmetrized_mlp": "_sr_" in args.method,
        "init_seconds": init_seconds,
        "solve_seconds": solve_seconds,
        "complete_time": complete_time,
        **health,
    }
    metadata["complete"] = bool(metadata["complete"] and complete_time)
    state = interior(final, params.ghost, args.nx, ny)
    x = (np.arange(args.nx, dtype=np.float64) + 0.5) * params.dx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * params.dy
    np.savez(
        out_npz,
        state=state,
        x=x,
        y=y,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    write_json(out_json, metadata)
    print(f"saved_results={out_npz}", flush=True)
    print(f"saved_metadata={out_json}", flush=True)
    if not metadata["complete"]:
        raise RuntimeError(f"shock-bubble validation failed: {metadata}")


if __name__ == "__main__":
    main()
