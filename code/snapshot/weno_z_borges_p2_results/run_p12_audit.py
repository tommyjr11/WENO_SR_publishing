#!/usr/bin/env python3
"""WENO7-Z p=1/2 audit with raw beta and epsilon=h^3."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from for_paper_results import exact_riemann
from for_paper_results.common import state_health
from for_paper_results.run_quadrant import CASES, make_quadrant_state
from . import euler_z
from .run_riemann_1d import PROBLEMS, initial_state, sod_cell_average_conserved


ROOT = Path(__file__).resolve().parent / "raw_beta_h3_p12_audit"


def weight_kind(power: int) -> int:
    return 4 if power == 1 else 5


def run_riemann(problem_key: str, power: int, device: str) -> dict[str, object]:
    base = PROBLEMS[problem_key]
    problem = replace(base, nx=100, ny=10, cfl=0.8) if problem_key == "sod" else base
    dx = (problem.x_max - problem.x_min) / problem.nx
    params = euler_z.make_weno7_params(
        problem.nx, problem.ny, problem.x_min, problem.x_max,
        0.0, problem.ny * dx, problem.cfl, problem.t_end,
    )
    initial = initial_state(params, problem, 7)
    started = time.perf_counter()
    final, summary = euler_z.run_weno7(
        initial, params, device=device, boundary="outflow",
        report_interval=0, weight_kind=weight_kind(power),
    )
    elapsed = time.perf_counter() - started
    state = euler_z.interior(final, params.ghost, problem.nx, problem.ny)
    conserved = state.mean(axis=0)
    primitive = euler_z.primitive(conserved)
    x = problem.x_min + (np.arange(problem.nx, dtype=np.float64) + 0.5) * dx
    if problem_key == "sod":
        exact = sod_cell_average_conserved(x, dx, problem.t_end, problem.x0, quadrature=15)
    else:
        exact = exact_riemann.cell_average_conserved(
            x, dx, problem.t_end, problem.x0, problem.left, problem.right,
            quadrature=15,
        )
    error = primitive[:, 0] - exact[:, 0]
    row: dict[str, object] = {
        "problem": problem_key,
        "method": f"weno7_z_p{power}_rawbeta_epsh3",
        "label": f"WENO7-Z(p={power})-RK4",
        "nx": problem.nx,
        "cfl": problem.cfl,
        "t_end": problem.t_end,
        "t": float(summary["t"]),
        "steps": int(summary["steps"]),
        "rho_l1": float(np.mean(np.abs(error))),
        "rho_l2": float(np.sqrt(np.mean(error * error))),
        "rho_linf": float(np.max(np.abs(error))),
        "rho_min": float(primitive[:, 0].min()),
        "p_min": float(primitive[:, 3].min()),
        "nan_count": int(np.isnan(state).sum()),
        "seconds": elapsed,
        "beta_convention": "beta_raw_with_common_factor_240",
        "epsilon": float(dx**3),
        "epsilon_convention": "epsilon=dx^3_without_factor_240",
        "weno_z_power": power,
        "weno_z_tau": "abs(-beta0-3*beta1+3*beta2+beta3)",
    }
    out = ROOT / "riemann_1d" / problem_key
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"weno7_z_p{power}.npz", x=x, rho=primitive[:, 0],
        velocity=primitive[:, 1], pressure=primitive[:, 3],
        conserved_1d=conserved, exact_conserved=exact,
        metadata_json=np.array(json.dumps(row, sort_keys=True)),
    )
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def run_c3(power: int, device: str, report_interval: int) -> dict[str, object]:
    definition = CASES["c3"]
    nx = ny = 400
    cfl = 0.4
    t_end = float(definition["t_end"])
    params = euler_z.make_weno7_params(nx, ny, 0.0, 1.0, 0.0, 1.0, cfl, t_end)
    initial = make_quadrant_state(params, definition, 15)
    print(
        f"c3_start method=WENO7-Z(p={power}) grid=400x400 cfl=0.4 "
        f"t_end={t_end} characteristic=True solver=hllc",
        flush=True,
    )
    started = time.perf_counter()
    final, summary = euler_z.run_weno7(
        initial, params, device=device, boundary="outflow",
        report_interval=report_interval, weight_kind=weight_kind(power),
    )
    elapsed = time.perf_counter() - started
    health = state_health(final, params.ghost, nx, ny)
    complete_time = abs(float(summary["t"]) - t_end) < 1.0e-12
    metadata: dict[str, object] = {
        "benchmark": "two-dimensional Riemann C.3",
        "method": f"weno7_z_p{power}_rawbeta_epsh3",
        "label": f"WENO7-Z(p={power})-RK4",
        "nx": nx,
        "ny": ny,
        "cfl": cfl,
        "t_end": t_end,
        "t": float(summary["t"]),
        "steps": int(summary["steps"]),
        "wall_seconds": elapsed,
        "weno_space": "characteristic",
        "riemann_solver": "hllc",
        "boundary": "outflow",
        "initialization": "15x15_Gauss_Legendre_cell_average",
        "beta_convention": "beta_raw_with_common_factor_240",
        "epsilon": float(params.dx**3),
        "epsilon_convention": "epsilon=dx^3_without_factor_240",
        "weno_z_power": power,
        "weno_z_tau": "abs(-beta0-3*beta1+3*beta2+beta3)",
        "complete_time": complete_time,
        **health,
    }
    metadata["complete"] = bool(metadata["complete"] and complete_time)
    state = euler_z.interior(final, params.ghost, nx, ny)
    out = ROOT / "riemann" / "c3" / "N400"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"weno7_z_p{power}.npz", state=state,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    (out / f"weno7_z_p{power}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("riemann", "c3", "all"), default="all")
    parser.add_argument("--powers", type=int, nargs="+", choices=(1, 2), default=(1, 2))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=25)
    args = parser.parse_args()
    if args.part in ("riemann", "all"):
        rows = [
            run_riemann(problem, power, args.device)
            for problem in ("sod", "lax")
            for power in args.powers
        ]
        out = ROOT / "riemann_1d" / "sod_lax_l1.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.part in ("c3", "all"):
        rows = [run_c3(power, args.device, args.report_interval) for power in args.powers]
        out = ROOT / "riemann" / "c3" / "N400" / "completion.csv"
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
