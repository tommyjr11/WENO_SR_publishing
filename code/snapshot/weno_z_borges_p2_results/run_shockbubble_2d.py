#!/usr/bin/env python3
"""Run order-matched WENO-Z variants of the established 2-D shock-bubble tests."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import warp_weno5_helpers as trusted_init
from for_paper_results.common import state_health

from . import euler_z


ROOT = Path(__file__).resolve().parent
GAMMA = 1.4
RHO_AIR = 1.29
P_AIR = 101325.0
RHO_HELIUM = 0.214
X_SHOCK = 0.005
BUBBLE_X = 0.035
BUBBLE_Y = 0.0445
BUBBLE_RADIUS = 0.025


@dataclass(frozen=True)
class Case:
    key: str
    mach: float
    u_post: float
    t_end: float


CASES = {
    "ma122": Case("ma122", 1.22, 110.6273, 6.0e-4),
    "ma30": Case("ma30", 3.0, 736.911, 1.0e-4),
}


def post_shock(case: Case) -> tuple[float, float, float]:
    rho = (
        ((GAMMA + 1.0) * case.mach * case.mach)
        / ((GAMMA - 1.0) * case.mach * case.mach + 2.0)
        * RHO_AIR
    )
    pressure = (
        ((2.0 * GAMMA) * case.mach * case.mach - (GAMMA - 1.0))
        / (GAMMA + 1.0)
        * P_AIR
    )
    return rho, case.u_post, pressure


def initial_state(params, case: Case) -> np.ndarray:
    """Vectorized copy of the trusted 15x15 Gauss initialization."""
    ghost = params.ghost
    nx_total = params.nx + 2 * ghost
    ny_total = params.ny + 2 * ghost
    xi = trusted_init.GAUSS15_XI
    weights = trusted_init.GAUSS15_W
    x_min = float(getattr(params, "x_min", 0.0))
    y_min = float(getattr(params, "y_min", 0.0))
    x_center = x_min + (np.arange(nx_total) - ghost + 0.5) * params.dx
    y_center = y_min + (np.arange(ny_total) - ghost + 0.5) * params.dy
    xq = x_center[:, None] + 0.5 * params.dx * xi[None, :]

    air = trusted_init.primitive_to_conserved(RHO_AIR, 0.0, 0.0, P_AIR, GAMMA)
    rho_post, u_post, p_post = post_shock(case)
    post = trusted_init.primitive_to_conserved(rho_post, u_post, 0.0, p_post, GAMMA)
    helium = trusted_init.primitive_to_conserved(RHO_HELIUM, 0.0, 0.0, P_AIR, GAMMA)

    post_fraction = 0.5 * np.sum((xq < X_SHOCK) * weights[None, :], axis=1)
    state = air[None, None, :] + post_fraction[None, :, None] * (post - air)[None, None, :]
    state = np.broadcast_to(state, (ny_total, nx_total, 4)).copy()

    x_radius2 = (xq - BUBBLE_X) ** 2
    gauss_weight_2d = weights[:, None] * weights[None, :]
    for j0 in range(0, ny_total, 16):
        j1 = min(j0 + 16, ny_total)
        yq = y_center[j0:j1, None] + 0.5 * params.dy * xi[None, :]
        y_radius2 = (yq - BUBBLE_Y) ** 2
        inside = (
            y_radius2[:, None, :, None] + x_radius2[None, :, None, :]
            <= BUBBLE_RADIUS**2
        )
        fraction = 0.25 * np.sum(
            inside * gauss_weight_2d[None, None, :, :], axis=(2, 3)
        )
        state[j0:j1] += fraction[..., None] * (helium - air)[None, None, :]
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--method", choices=("weno5_z_p2", "weno7_z_p3"), required=True)
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=396)
    parser.add_argument("--x-length", type=float, default=0.225)
    parser.add_argument("--y-length", type=float, default=0.089)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    args = parser.parse_args()
    case = CASES[args.case]
    t_end = case.t_end if args.t_end is None else args.t_end
    order = 5 if args.method.startswith("weno5") else 7
    if order == 5:
        params = euler_z.make_weno5_params(
            args.nx, args.ny, args.x_length, args.y_length, args.cfl, t_end
        )
        run = euler_z.run_weno5
        exponent = 2
    else:
        params = euler_z.make_weno7_params(
            args.nx, args.ny, 0.0, args.x_length, 0.0, args.y_length,
            args.cfl, t_end,
        )
        run = euler_z.run_weno7
        exponent = 3

    out = ROOT / f"raw/shockbubble_2d/{case.key}/N{args.nx}x{args.ny}"
    out.mkdir(parents=True, exist_ok=True)
    init_started = time.perf_counter()
    initial = initial_state(params, case)
    init_seconds = time.perf_counter() - init_started
    print(
        f"shockbubble_z_start case={case.key} mach={case.mach:g} method={args.method} "
        f"grid={args.nx}x{args.ny} cfl={args.cfl} t_end={t_end:.8e} "
        f"quadrature=15x15 characteristic=True solver=hllc init_seconds={init_seconds:.3f}",
        flush=True,
    )
    solve_started = time.perf_counter()
    final, summary = run(
        initial, params, device=args.device, boundary="transmissive",
        report_interval=args.report_interval,
    )
    solve_seconds = time.perf_counter() - solve_started
    health = state_health(final, params.ghost, args.nx, args.ny)
    complete_time = abs(float(summary["t"]) - t_end) < 1.0e-12
    metadata = {
        "benchmark": "shock_helium_bubble",
        "case": case.key,
        "shock_mach": case.mach,
        "method": args.method,
        "label": f"WENO{order}-Z-{'RK3' if order == 5 else 'RK4'}",
        "nx": args.nx,
        "ny": args.ny,
        "x_length": args.x_length,
        "y_length": args.y_length,
        "dx": params.dx,
        "dy": params.dy,
        "cfl": args.cfl,
        "t_end": t_end,
        "t": float(summary["t"]),
        "steps": int(summary["steps"]),
        "dt_min": float(summary["dt_min"]),
        "dt_max": float(summary["dt_max"]),
        "dt_mean": float(summary["dt_mean"]),
        "time_integrator": "SSPRK3" if order == 5 else "Shu-RK4",
        "weno_space": "characteristic",
        "riemann_solver": "hllc",
        "boundary": "transmissive",
        "initialization": "15x15_Gauss_Legendre_cell_average",
        "weno_z_tau": (
            "abs(beta0-beta2)" if order == 5
            else "abs(-beta0-3*beta1+3*beta2+beta3)"
        ),
        "weno_z_p": exponent,
        "weno_z_epsilon": f"h^{exponent}_directionwise",
        "weno_z_epsilon_x": float(params.dx**exponent),
        "weno_z_epsilon_y": float(params.dy**exponent),
        "weno_z_epsilon_convention": f"paper epsilon=h^{exponent}",
        "smoothness_evaluation": (
            "nonnegative difference-square form" if order == 7
            else "Jiang-Shu difference-square form"
        ),
        "init_seconds": init_seconds,
        "solve_seconds": solve_seconds,
        "complete_time": complete_time,
        **health,
    }
    metadata["complete"] = bool(metadata["complete"] and complete_time)
    state = euler_z.interior(final, params.ghost, args.nx, args.ny)
    x = (np.arange(args.nx, dtype=np.float64) + 0.5) * params.dx
    y = (np.arange(args.ny, dtype=np.float64) + 0.5) * params.dy
    np.savez(
        out / f"{args.method}.npz", state=state, x=x, y=y,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    (out / f"{args.method}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)
    if not metadata["complete"]:
        raise RuntimeError(f"incomplete WENO-Z shock-bubble result: {metadata}")


if __name__ == "__main__":
    main()
