#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

import pretrain_weno7_offline as sod_exact
from teacherfree_lab_weno5 import warp_sod_validation as sod5

from for_paper_results import config
from for_paper_results.common import interior, rho_errors, state_health, write_json
from for_paper_results.solvers import euler_methods


def exact_sod_cell_averages(
    centers: np.ndarray,
    dx: float,
    t: float,
    gamma: float,
    quadrature: int,
    discontinuity: float = 0.5,
) -> np.ndarray:
    xi, wi = np.polynomial.legendre.leggauss(quadrature)
    state = np.zeros((centers.size, 3), dtype=np.float64)
    for node, weight in zip(xi, wi):
        xq = centers + 0.5 * dx * float(node) - discontinuity
        rho, velocity, pressure = sod_exact.exact_sod_primitive(xq, t, gamma)
        state += 0.5 * float(weight) * sod_exact.primitive_to_conserved_1d(
            rho, velocity, pressure, gamma,
        )
    return state


def make_weno7_sod_state(params, quadrature: int, discontinuity: float = 0.5) -> np.ndarray:
    g = params.ghost
    centers = params.x_min + (np.arange(params.nx + 2 * g) - g + 0.5) * params.dx
    state1d = exact_sod_cell_averages(
        centers, params.dx, 0.0, params.gamma, quadrature, discontinuity,
    )
    state = np.zeros((params.ny + 2 * g, params.nx + 2 * g, 4), dtype=np.float64)
    state[..., 0] = state1d[None, :, 0]
    state[..., 1] = state1d[None, :, 1]
    state[..., 3] = state1d[None, :, 2]
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=0)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--out-tag", default=None)
    args = parser.parse_args()
    if args.init_quadrature != 15:
        raise ValueError("paper Sod tests require 15-point Gauss-Legendre initialisation")
    config.ensure_output_dirs()
    config.validate_models()
    out_dir = config.RAW / "sod" / args.out_tag if args.out_tag else config.RAW / "sod"
    out_dir.mkdir(parents=True, exist_ok=True)

    discontinuity = 0.5
    reference_params = euler_methods.make_weno7_params(
        args.nx, args.ny, 0.0, 1.0, 0.0, args.ny / args.nx,
        args.cfl, args.t_end,
    )
    x = reference_params.x_min + (np.arange(args.nx) + 0.5) * reference_params.dx
    exact_rho = exact_sod_cell_averages(
        x, reference_params.dx, args.t_end, reference_params.gamma,
        args.init_quadrature, discontinuity,
    )[:, 0]
    rows: list[dict] = []
    for key in config.EULER_METHODS:
        if key.startswith("weno5"):
            mixed = key == "weno5_sr_f32"
            params = euler_methods.make_weno5_params(
                args.nx, args.ny, 1.0, args.ny / args.nx,
                args.cfl, args.t_end, mixed,
            )
            initial = sod5.make_exact_sod_state(params, 0.0, "x")
            final, summary = euler_methods.run_weno5(
                key, initial, params, device=args.device, boundary="transmissive",
                report_interval=args.report_interval,
            )
        else:
            params = euler_methods.make_weno7_params(
                args.nx, args.ny, 0.0, 1.0, 0.0, args.ny / args.nx,
                args.cfl, args.t_end,
            )
            initial = make_weno7_sod_state(
                params, args.init_quadrature, discontinuity,
            )
            final, summary = euler_methods.run_weno7(
                key, initial, params, device=args.device, boundary="outflow",
                report_interval=args.report_interval,
            )
        state = interior(final, params.ghost, args.nx, args.ny)
        rho = state[..., 0].mean(axis=0)
        health = state_health(final, params.ghost, args.nx, args.ny)
        complete_time = abs(float(summary["t"]) - args.t_end) < 1.0e-12
        row = {
            "method": key, "nx": args.nx, "ny": args.ny,
            "init_quadrature": args.init_quadrature,
            "x_min": 0.0, "x_max": 1.0,
            "discontinuity": discontinuity,
            **rho_errors(rho, exact_rho), **health,
            "complete_time": complete_time, "steps": int(summary["steps"]),
            "t": float(summary["t"]),
        }
        row["complete"] = bool(row["complete"] and complete_time)
        np.savez(
            out_dir / f"{key}.npz", x=x, rho=rho, exact_rho=exact_rho,
            state=state, metadata_json=np.array(json.dumps(row, sort_keys=True)),
        )
        write_json(out_dir / f"{key}.json", row)
        rows.append(row)
        print(row, flush=True)

    fields = ["method", "nx", "ny", "init_quadrature", "complete", "steps", "t",
              "rho_l1", "rho_l2", "rho_linf",
              "rho_min", "rho_max", "p_min", "p_max", "nan_count"]
    with (out_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "summary.json", {"configuration": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
