#!/usr/bin/env python3
"""Controlled WENO7-JS/WENO7-Z comparison with beta normalized by 240."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from for_paper_results import exact_riemann
from . import euler_z
from . import weno7_point_rk4_z as W7
from .run_riemann_1d import PROBLEMS, initial_state, sod_cell_average_conserved


ROOT = Path(__file__).resolve().parent
OUT_NORMALIZED = ROOT / "beta_normalized_audit"
OUT_RAW_TINY = ROOT / "raw_beta_tiny_epsilon_audit"


def run_problem(problem_key: str, device: str, convention: str) -> list[dict[str, object]]:
    base = PROBLEMS[problem_key]
    problem = replace(base, nx=100, ny=10, cfl=0.8) if problem_key == "sod" else base
    dx = (problem.x_max - problem.x_min) / problem.nx
    y_length = problem.ny * dx
    params = euler_z.make_weno7_params(
        problem.nx,
        problem.ny,
        problem.x_min,
        problem.x_max,
        0.0,
        y_length,
        problem.cfl,
        problem.t_end,
    )
    initial = initial_state(params, problem, 7)
    x = problem.x_min + (np.arange(problem.nx, dtype=np.float64) + 0.5) * dx
    if problem_key == "sod":
        exact = sod_cell_average_conserved(x, dx, problem.t_end, problem.x0, quadrature=15)
    else:
        exact = exact_riemann.cell_average_conserved(
            x,
            dx,
            problem.t_end,
            problem.x0,
            problem.left,
            problem.right,
            quadrature=15,
        )
    exact_rho = exact[:, 0]
    normalized = convention == "normalized"
    out_root = OUT_NORMALIZED if normalized else OUT_RAW_TINY
    problem_out = out_root / problem_key
    problem_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    variants = (
        ((0, "weno7_js_beta_normalized"), (1, "weno7_z_beta_normalized"))
        if normalized
        else ((2, "weno7_js_eps1e40"), (3, "weno7_z_eps1e40"))
    )
    for weight_kind, method in variants:
        label = "WENO7-JS-RK4" if weight_kind in (0, 2) else "WENO7-Z-RK4"
        started = time.perf_counter()
        final, summary = W7.run_from_initial(
            initial,
            params,
            device=device,
            riemann_solver="hllc",
            characteristic=True,
            boundary="outflow",
            report_interval=0,
            weight_kind=weight_kind,
        )
        elapsed = time.perf_counter() - started
        state = euler_z.interior(final, params.ghost, problem.nx, problem.ny)
        conserved = state.mean(axis=0)
        primitive = euler_z.primitive(conserved)
        error = primitive[:, 0] - exact_rho
        row: dict[str, object] = {
            "problem": problem_key,
            "method": method,
            "label": label,
            "nx": problem.nx,
            "cfl": problem.cfl,
            "t_end": problem.t_end,
            "steps": int(summary["steps"]),
            "rho_l1": float(np.mean(np.abs(error))),
            "rho_l2": float(np.sqrt(np.mean(error * error))),
            "rho_linf": float(np.max(np.abs(error))),
            "rho_min": float(primitive[:, 0].min()),
            "p_min": float(primitive[:, 3].min()),
            "nan_count": int(np.isnan(state).sum()),
            "seconds": elapsed,
            "beta_convention": "beta_raw/240" if normalized else "beta_raw",
            "epsilon": float(dx**3) if normalized else 1.0e-40,
            "epsilon_convention": "shared epsilon=dx^3" if normalized else "shared epsilon=1e-40",
            "z_power": 3 if weight_kind in (1, 3) else 0,
            "z_tau": (
                "abs(-beta0-3*beta1+3*beta2+beta3)" if weight_kind in (1, 3) else "n/a"
            ),
        }
        np.savez(
            problem_out / f"{method}.npz",
            x=x,
            rho=primitive[:, 0],
            velocity=primitive[:, 1],
            pressure=primitive[:, 3],
            conserved_1d=conserved,
            exact_conserved=exact,
            metadata_json=np.array(json.dumps(row, sort_keys=True)),
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def write_tables(rows: list[dict[str, object]], convention: str) -> None:
    normalized = convention == "normalized"
    out_root = OUT_NORMALIZED if normalized else OUT_RAW_TINY
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "sod_lax_l1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    indexed = {(str(row["problem"]), str(row["method"])): row for row in rows}
    lines = [
        "# WENO7 beta-normalized audit" if normalized else "# WENO7 raw-beta tiny-epsilon audit",
        "",
        (
            "Both methods use beta_hat=beta_raw/240 and the same epsilon=dx^3."
            if normalized
            else "Both methods use the original beta_raw and the same epsilon=1e-40."
        ),
        "All other settings are identical: characteristic FVM, HLLC, RK4, CFL=0.8.",
        "",
        "| Test | WENO7-JS-RK4 L1 | WENO7-Z-RK4 L1 | Z relative to JS |",
        "|---|---:|---:|---:|",
    ]
    js_key = "weno7_js_beta_normalized" if normalized else "weno7_js_eps1e40"
    z_key = "weno7_z_beta_normalized" if normalized else "weno7_z_eps1e40"
    for problem in ("sod", "lax"):
        js = float(indexed[(problem, js_key)]["rho_l1"])
        z = float(indexed[(problem, z_key)]["rho_l1"])
        lines.append(f"| {problem.title()} | {js:.9e} | {z:.9e} | {(z / js - 1.0) * 100.0:+.3f}% |")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--convention", choices=("normalized", "raw-tiny"), default="normalized")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for problem in ("sod", "lax"):
        rows.extend(run_problem(problem, args.device, args.convention))
    write_tables(rows, args.convention)


if __name__ == "__main__":
    main()
