#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from weno7_point_rk4_shu import point_rk4 as vortex7

from for_paper_results import config
from for_paper_results.common import interior, rho_errors, state_health, write_json
from for_paper_results.solvers import euler_methods


def parse_grids(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--grids requires positive comma-separated integers")
    return values


def parse_methods(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in values if item not in config.EULER_METHODS]
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"--methods contains unknown or empty entries: {unknown or values}"
        )
    return values


def pad_periodic(interior_state: np.ndarray, ghost: int) -> np.ndarray:
    return np.pad(interior_state, ((ghost, ghost), (ghost, ghost), (0, 0)), mode="wrap")


def add_orders(rows: list[dict]) -> None:
    for method in config.EULER_METHODS:
        subset = sorted((row for row in rows if row["method"] == method), key=lambda row: row["N"])
        for index, row in enumerate(subset):
            for norm in ("rho_l1", "rho_l2"):
                if index == 0 or row[norm] <= 0.0 or subset[index - 1][norm] <= 0.0:
                    row[f"{norm}_order"] = float("nan")
                else:
                    previous = subset[index - 1]
                    row[f"{norm}_order"] = float(
                        np.log(previous[norm] / row[norm])
                        / np.log(row["N"] / previous["N"])
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=parse_grids, default=[25, 50, 100, 200])
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quadrature", type=int, default=15)
    parser.add_argument("--report-interval", type=int, default=0)
    parser.add_argument(
        "--out-tag",
        default="vortex",
        help="output directory name below for_paper_results/raw",
    )
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=list(config.EULER_METHODS),
        help="comma-separated methods to recompute; unselected validated rows are retained",
    )
    args = parser.parse_args()
    if args.quadrature != 15:
        raise ValueError("formal vortex tests require 15x15 Gauss-Legendre averages")
    if args.t_end != 2.0:
        raise ValueError("formal vortex tests require t_end=2")
    if args.cfl <= 0.0:
        raise ValueError("CFL must be positive")
    config.ensure_output_dirs()
    config.validate_models()
    out_dir = config.RAW / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for n in args.grids:
        p7 = euler_methods.make_weno7_params(
            n, n, -10.0, 10.0, -10.0, 10.0, args.cfl, args.t_end,
        )
        initial7 = vortex7.cell_average_state(p7, 0.0, args.quadrature)
        exact7 = vortex7.cell_average_state(p7, args.t_end, args.quadrature)
        initial_interior = interior(initial7, p7.ghost, n, n)
        exact_interior = interior(exact7, p7.ghost, n, n)
        exact_rho = exact_interior[..., 0]

        for key in args.methods:
            if key.startswith("weno5"):
                mixed = key == "weno5_sr_f32"
                params = euler_methods.make_weno5_params(
                    n, n, 20.0, 20.0, args.cfl, args.t_end, mixed,
                )
                initial = pad_periodic(initial_interior, params.ghost)
                final, summary = euler_methods.run_weno5(
                    key, initial, params, device=args.device, boundary="periodic",
                    report_interval=args.report_interval,
                )
            else:
                params = p7
                final, summary = euler_methods.run_weno7(
                    key, initial7.copy(), params, device=args.device, boundary="periodic",
                    report_interval=args.report_interval,
                )
            state = interior(final, params.ghost, n, n)
            health = state_health(final, params.ghost, n, n)
            complete_time = abs(float(summary["t"]) - args.t_end) < 1.0e-12
            row = {
                "method": key, "N": n, "dx": 20.0 / n,
                "domain": "[-10,10]^2", "cfl": args.cfl,
                "t_end": args.t_end, "quadrature": args.quadrature,
                "initialization": "15x15_Gauss_finite_volume_cell_average",
                "exact_solution": "periodic_isentropic_vortex_center_(t,t)",
                "boundary": "periodic", "riemann_solver": "hllc",
                "weno_space": "characteristic", "eno_cutoff": False,
                "time_integrator": config.METHODS[key].time_integrator,
                **rho_errors(state[..., 0], exact_rho), **health,
                "complete_time": complete_time, "steps": int(summary["steps"]),
                "t": float(summary["t"]),
            }
            row["complete"] = bool(row["complete"] and complete_time)
            np.savez(
                out_dir / f"{key}_N{n}.npz", state=state, exact=exact_interior,
                metadata_json=np.array(json.dumps(row, sort_keys=True)),
            )
            write_json(out_dir / f"{key}_N{n}.json", row)
            rows.append(row)
            print(row, flush=True)

    selected = set(args.methods)
    for key in config.EULER_METHODS:
        if key in selected:
            continue
        for n in args.grids:
            path = out_dir / f"{key}_N{n}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"cannot retain unselected vortex result; missing {path}"
                )
            row = json.loads(path.read_text(encoding="utf-8"))
            if not row.get("complete", False):
                raise RuntimeError(
                    f"cannot retain incomplete vortex result from {path}"
                )
            rows.append(row)

    method_order = {key: index for index, key in enumerate(config.EULER_METHODS)}
    rows.sort(key=lambda row: (method_order[row["method"]], int(row["N"])))
    add_orders(rows)
    fields = ["method", "N", "dx", "cfl", "t_end", "quadrature",
              "complete", "steps", "t", "rho_l1",
              "rho_l1_order", "rho_l2", "rho_l2_order", "rho_linf",
              "rho_min", "rho_max", "p_min", "p_max", "nan_count"]
    with (out_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "summary.json", {"configuration": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
