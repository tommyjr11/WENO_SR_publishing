#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import pretrain_weno7_offline as sod_exact
from for_paper_results import exact_riemann
from for_paper_results.run_sod import make_weno7_sod_state
from teacherfree_lab_weno5 import warp_sod_validation as sod5
from . import euler_z


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Problem:
    key: str
    x_min: float
    x_max: float
    x0: float
    t_end: float
    nx: int
    ny: int
    cfl: float
    left: tuple[float, float, float]
    right: tuple[float, float, float]


PROBLEMS = {
    "sod": Problem("sod", 0.0, 1.0, 0.5, 0.2, 51, 11, 0.4,
                   (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)),
    "lax": Problem("lax", -5.0, 5.0, 0.0, 1.3, 200, 10, 0.8,
                   (0.445, 0.698, 3.528), (0.5, 0.0, 0.571)),
}


def sod_cell_average_conserved(
    centers: np.ndarray,
    dx: float,
    t: float,
    discontinuity: float,
    quadrature: int = 15,
) -> np.ndarray:
    """Match the paper Sod script's unsplit Gauss-cell-average convention."""
    xi, wi = np.polynomial.legendre.leggauss(quadrature)
    state = np.zeros((centers.size, 4), dtype=np.float64)
    for node, weight in zip(xi, wi):
        xq = centers + 0.5 * dx * float(node) - discontinuity
        rho, velocity, pressure = sod_exact.exact_sod_primitive(xq, t, 1.4)
        state[:, 0] += 0.5 * float(weight) * rho
        state[:, 1] += 0.5 * float(weight) * rho * velocity
        state[:, 3] += 0.5 * float(weight) * (
            pressure / 0.4 + 0.5 * rho * velocity * velocity
        )
    return state


def initial_state(params, problem: Problem, order: int) -> np.ndarray:
    if problem.key == "sod":
        if order == 5:
            return sod5.make_exact_sod_state(params, 0.0, "x")
        return make_weno7_sod_state(params, 15, problem.x0)

    g = params.ghost
    dx = (problem.x_max - problem.x_min) / problem.nx
    centers = problem.x_min + (
        np.arange(problem.nx + 2 * g, dtype=np.float64) - g + 0.5
    ) * dx
    cell = exact_riemann.cell_average_conserved(
        centers, dx, 0.0, problem.x0, problem.left, problem.right,
        quadrature=15,
    )
    return np.broadcast_to(
        cell[None, :, :], (problem.ny + 2 * g, problem.nx + 2 * g, 4)
    ).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=tuple(PROBLEMS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=0)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--out-tag", default=None)
    args = parser.parse_args()
    problem = PROBLEMS[args.problem]
    if args.cfl is not None:
        problem = replace(problem, cfl=args.cfl)
    if args.nx is not None:
        problem = replace(problem, nx=args.nx)
    if args.ny is not None:
        problem = replace(problem, ny=args.ny)
    out = ROOT / f"raw/riemann_1d/{problem.key}"
    if args.out_tag:
        out = out / args.out_tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for order in (5, 7):
        dx = (problem.x_max - problem.x_min) / problem.nx
        y_length = problem.ny * dx
        if order == 5:
            params = euler_z.make_weno5_params(
                problem.nx, problem.ny, problem.x_max - problem.x_min,
                y_length, problem.cfl, problem.t_end,
            )
            run = euler_z.run_weno5
            boundary = "transmissive"
        else:
            params = euler_z.make_weno7_params(
                problem.nx, problem.ny, problem.x_min, problem.x_max,
                0.0, y_length, problem.cfl, problem.t_end,
            )
            run = euler_z.run_weno7
            boundary = "outflow"
        initial = initial_state(params, problem, order)
        started = time.perf_counter()
        final, summary = run(
            initial, params, device=args.device, boundary=boundary,
            report_interval=args.report_interval,
        )
        elapsed = time.perf_counter() - started
        state = euler_z.interior(final, params.ghost, problem.nx, problem.ny)
        conserved = state.mean(axis=0)
        primitive = euler_z.primitive(conserved)
        x = problem.x_min + (np.arange(problem.nx) + 0.5) * dx
        if problem.key == "sod":
            exact = sod_cell_average_conserved(
                x, dx, problem.t_end, problem.x0, quadrature=15,
            )
        else:
            exact = exact_riemann.cell_average_conserved(
                x, dx, problem.t_end, problem.x0, problem.left, problem.right,
                quadrature=15,
            )
        exact_primitive = euler_z.primitive(exact)
        delta = primitive[:, 0] - exact_primitive[:, 0]
        power = 2 if order == 5 else 3
        key = f"weno{order}_z_p{power}"
        row = {
            "problem": problem.key, "method": key, "nx": problem.nx,
            "ny": problem.ny, "cfl": problem.cfl, "t_end": problem.t_end,
            "steps": int(summary["steps"]), "t": float(summary["t"]),
            "rho_l1": float(np.mean(np.abs(delta))),
            "rho_l2": float(np.sqrt(np.mean(delta * delta))),
            "rho_linf": float(np.max(np.abs(delta))),
            "rho_min": float(primitive[:, 0].min()),
            "p_min": float(primitive[:, 3].min()),
            "nan_count": int(np.isnan(state).sum()),
            "seconds": elapsed, "weno_z_p": power,
            "weno_z_tau": (
                "abs(beta0-beta2)" if order == 5
                else "abs(-beta0-3*beta1+3*beta2+beta3)"
            ),
            "weno_z_epsilon": float(dx**power),
            "weno_z_epsilon_convention": f"paper epsilon=dx^{power}",
        }
        np.savez(
            out / f"{key}.npz", x=x, state=state, conserved_1d=conserved,
            rho=primitive[:, 0], velocity=primitive[:, 1],
            pressure=primitive[:, 3], exact_conserved=exact,
            metadata_json=np.array(json.dumps(row, sort_keys=True)),
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
