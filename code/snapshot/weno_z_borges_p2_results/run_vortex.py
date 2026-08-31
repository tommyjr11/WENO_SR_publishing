#!/usr/bin/env python3
"""Run the paper WENO-Z variants on the periodic isentropic vortex."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from weno7_point_rk4_shu import point_rk4 as vortex7

from for_paper_results.common import interior, rho_errors, state_health

from . import euler_z


ROOT = Path(__file__).resolve().parent
METHODS = ("weno5_z_p2", "weno7_z_p3")


def pad_periodic(state: np.ndarray, ghost: int) -> np.ndarray:
    return np.pad(state, ((ghost, ghost), (ghost, ghost), (0, 0)), mode="wrap")


def add_orders(rows: list[dict[str, object]]) -> None:
    for method in METHODS:
        subset = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["N"]),
        )
        for index, row in enumerate(subset):
            for norm in ("rho_l1", "rho_l2"):
                if index == 0:
                    row[f"{norm}_order"] = float("nan")
                    continue
                previous = subset[index - 1]
                row[f"{norm}_order"] = float(
                    np.log(float(previous[norm]) / float(row[norm]))
                    / np.log(float(row["N"]) / float(previous["N"]))
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", default="25,50,100,200")
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--quadrature", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=0)
    args = parser.parse_args()
    grids = [int(value) for value in args.grids.split(",") if value.strip()]
    if grids != [25, 50, 100, 200]:
        raise ValueError("formal vortex audit requires grids 25,50,100,200")
    if args.quadrature != 15 or args.t_end != 2.0 or args.cfl != 0.4:
        raise ValueError("formal vortex audit requires quadrature=15, t_end=2, CFL=0.4")

    out = ROOT / "raw/vortex_cfl04"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for n in grids:
        params7 = euler_z.make_weno7_params(
            n, n, -10.0, 10.0, -10.0, 10.0, args.cfl, args.t_end,
        )
        initial7 = vortex7.cell_average_state(params7, 0.0, args.quadrature)
        exact7 = vortex7.cell_average_state(params7, args.t_end, args.quadrature)
        initial_interior = interior(initial7, params7.ghost, n, n)
        exact_interior = interior(exact7, params7.ghost, n, n)
        exact_rho = exact_interior[..., 0]

        for method in METHODS:
            if method == "weno5_z_p2":
                params = euler_z.make_weno5_params(
                    n, n, 20.0, 20.0, args.cfl, args.t_end,
                )
                initial = pad_periodic(initial_interior, params.ghost)
                final, summary = euler_z.run_weno5(
                    initial,
                    params,
                    device=args.device,
                    boundary="periodic",
                    report_interval=args.report_interval,
                )
                power = 2
                epsilon = float(params.dx**2)
                integrator = "SSPRK3"
                tau = "abs(beta0-beta2)"
            else:
                params = params7
                final, summary = euler_z.run_weno7(
                    initial7.copy(),
                    params,
                    device=args.device,
                    boundary="periodic",
                    report_interval=args.report_interval,
                    weight_kind=1,
                )
                power = 3
                epsilon = float(params.dx**3)
                integrator = "four-stage fourth-order downwind TVD-RK"
                tau = "abs(-beta0-3*beta1+3*beta2+beta3)"

            state = interior(final, params.ghost, n, n)
            health = state_health(final, params.ghost, n, n)
            complete_time = abs(float(summary["t"]) - args.t_end) < 1.0e-12
            row: dict[str, object] = {
                "method": method,
                "N": n,
                "dx": 20.0 / n,
                "domain": "[-10,10]^2",
                "cfl": args.cfl,
                "t_end": args.t_end,
                "quadrature": args.quadrature,
                "initialization": "15x15_Gauss_finite_volume_cell_average",
                "exact_solution": "periodic_isentropic_vortex_center_(t,t)",
                "boundary": "periodic",
                "riemann_solver": "hllc",
                "weno_space": "characteristic",
                "time_integrator": integrator,
                "weno_z_p": power,
                "weno_z_epsilon": epsilon,
                "weno_z_epsilon_convention": f"paper epsilon=dx^{power}",
                "weno_z_tau": tau,
                **rho_errors(state[..., 0], exact_rho),
                **health,
                "complete_time": complete_time,
                "steps": int(summary["steps"]),
                "t": float(summary["t"]),
            }
            row["complete"] = bool(row["complete"] and complete_time)
            np.savez(
                out / f"{method}_N{n}.npz",
                state=state,
                exact=exact_interior,
                metadata_json=np.array(json.dumps(row, sort_keys=True)),
            )
            (out / f"{method}_N{n}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    add_orders(rows)
    rows.sort(key=lambda row: (METHODS.index(str(row["method"])), int(row["N"])))
    fields = [
        "method", "N", "dx", "cfl", "t_end", "complete", "steps", "t",
        "rho_l1", "rho_l1_order", "rho_l2", "rho_l2_order", "rho_linf",
        "rho_min", "rho_max", "p_min", "p_max", "nan_count", "weno_z_p",
        "weno_z_epsilon",
    ]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(bool(row["complete"]) for row in rows):
        raise RuntimeError("one or more WENO-Z vortex runs did not complete")
    print(out / "metrics.csv")


if __name__ == "__main__":
    main()
