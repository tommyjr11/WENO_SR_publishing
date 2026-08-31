#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from weno7_point_rk4_shu import run_quadrant_point_rk4 as q7

from for_paper_results import config
from for_paper_results.common import interior, state_health, write_json
from for_paper_results.run_vortex import pad_periodic
from for_paper_results.solvers import euler_methods


# State numbering follows the C.3--C.6 table used in the paper:
# 1 = upper right, 2 = upper left, 3 = lower left, 4 = lower right.
CASES = {
    "c3": {
        "label": "C.3 (q400)",
        "t_end": 0.50,
        "states": (
            (1.5, 0.0, 0.0, 1.5),
            (0.5323, 1.206, 0.0, 0.3),
            (0.138, 1.206, 1.206, 0.029),
            (0.5323, 0.0, 1.206, 0.3),
        ),
    },
    "c4": {
        "label": "C.4",
        "t_end": 0.25,
        "states": (
            (1.1, 0.0, 0.0, 1.1),
            (0.5065, 0.8939, 0.0, 0.35),
            (1.1, 0.8939, 0.8939, 1.1),
            (0.5065, 0.0, 0.8939, 0.35),
        ),
    },
    "c5": {
        "label": "C.5",
        "t_end": 0.23,
        "states": (
            (1.0, -0.75, -0.5, 1.0),
            (2.0, -0.75, 0.5, 1.0),
            (1.0, 0.75, 0.5, 1.0),
            (3.0, 0.75, -0.5, 1.0),
        ),
    },
    "c6": {
        "label": "C.6",
        "t_end": 0.30,
        "states": (
            (1.0, 0.75, -0.5, 1.0),
            (2.0, 0.75, 0.5, 1.0),
            (1.0, -0.75, 0.5, 1.0),
            (3.0, -0.75, -0.5, 1.0),
        ),
    },
}


def quadrant_primitive(
    x: np.ndarray,
    y: np.ndarray,
    states: tuple[tuple[float, float, float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(states) != 4:
        raise ValueError("a quadrant problem requires four primitive states")
    rho = np.empty_like(x, dtype=np.float64)
    vx = np.empty_like(x, dtype=np.float64)
    vy = np.empty_like(x, dtype=np.float64)
    pressure = np.empty_like(x, dtype=np.float64)
    regions = (
        (x >= 0.5) & (y >= 0.5),
        (x < 0.5) & (y >= 0.5),
        (x < 0.5) & (y < 0.5),
        (x >= 0.5) & (y < 0.5),
    )
    for mask, (rho_q, vx_q, vy_q, pressure_q) in zip(regions, states):
        rho[mask] = rho_q
        vx[mask] = vx_q
        vy[mask] = vy_q
        pressure[mask] = pressure_q
    return rho, vx, vy, pressure


def make_quadrant_state(params, definition: dict, quadrature: int) -> np.ndarray:
    if quadrature < 1:
        raise ValueError("quadrature order must be at least one")
    xi, wi = np.polynomial.legendre.leggauss(quadrature)
    g = params.ghost
    jj, ii = np.indices((params.ny + 2 * g, params.nx + 2 * g))
    xc = params.x_min + (ii - g + 0.5) * params.dx
    yc = params.y_min + (jj - g + 0.5) * params.dy
    state = np.zeros(params.padded_shape, dtype=np.float64)
    for sx, wx in zip(xi, wi):
        x = xc + 0.5 * params.dx * float(sx)
        for sy, wy in zip(xi, wi):
            y = yc + 0.5 * params.dy * float(sy)
            rho, vx, vy, pressure = quadrant_primitive(x, y, definition["states"])
            state += 0.25 * float(wx) * float(wy) * q7.solver.conserved_from_primitive(
                rho, vx, vy, pressure, params.gamma,
            )
    return state


def parse_methods(text: str) -> list[str]:
    methods = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [method for method in methods if method not in config.METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown method(s): {unknown}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--methods", type=parse_methods, default=list(config.EULER_METHODS))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=50)
    args = parser.parse_args()
    if args.nx != args.ny:
        raise ValueError("paper quadrant tests require square meshes")
    config.ensure_output_dirs()
    config.validate_models()
    definition = CASES[args.case]
    t_end = definition["t_end"] if args.t_end is None else args.t_end
    methods = ["weno7_js"] if args.reference_only else args.methods
    suffix = "reference1200" if args.reference_only else f"N{args.nx}"
    out_dir = config.RAW / args.case / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    p7 = euler_methods.make_weno7_params(
        args.nx, args.ny, 0.0, 1.0, 0.0, 1.0, args.cfl, t_end,
    )
    if args.init_quadrature != 15:
        raise ValueError("paper Riemann tests require 15-point initialisation")
    initial7 = make_quadrant_state(p7, definition, args.init_quadrature)
    initial_interior = interior(initial7, p7.ghost, args.nx, args.ny)

    rows: list[dict] = []
    failed_methods: list[str] = []
    for key in methods:
        started = time.perf_counter()
        if key.startswith("weno5"):
            mixed = key == "weno5_sr_f32"
            params = euler_methods.make_weno5_params(
                args.nx, args.ny, 1.0, 1.0, args.cfl, t_end, mixed,
            )
            initial = pad_periodic(initial_interior, params.ghost)
            final, summary = euler_methods.run_weno5(
                key, initial, params, device=args.device, boundary="transmissive",
                report_interval=args.report_interval,
            )
        else:
            params = p7
            final, summary = euler_methods.run_weno7(
                key, initial7.copy(), params, device=args.device, boundary="outflow",
                report_interval=args.report_interval,
            )
        elapsed = time.perf_counter() - started
        state = interior(final, params.ghost, args.nx, args.ny)
        health = state_health(final, params.ghost, args.nx, args.ny)
        complete_time = abs(float(summary["t"]) - t_end) < 1.0e-12
        row = {
            "method": key, "case": args.case, "configuration": definition["label"],
            "nx": args.nx, "ny": args.ny, "cfl": args.cfl, "t_end": t_end,
            "t": float(summary["t"]), "steps": int(summary["steps"]),
            "wall_seconds_including_setup": elapsed,
            **health, "complete_time": complete_time,
            "riemann_solver": "hllc", "weno_space": "characteristic",
            "eno_cutoff": False,
        }
        row["complete"] = bool(row["complete"] and complete_time)
        np.savez(
            out_dir / f"{key}.npz", state=state,
            metadata_json=np.array(json.dumps(row, sort_keys=True)),
        )
        write_json(out_dir / f"{key}.json", row)
        rows.append(row)
        print(row, flush=True)
        if not row["complete"]:
            failed_methods.append(key)
    row_by_method = {row["method"]: row for row in rows}
    if not args.reference_only:
        for key in config.EULER_METHODS:
            if key in row_by_method:
                continue
            prior_path = out_dir / f"{key}.json"
            if not prior_path.is_file():
                continue
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            same_configuration = (
                prior.get("case") == args.case
                and prior.get("nx") == args.nx
                and prior.get("ny") == args.ny
                and abs(float(prior.get("cfl", -1.0)) - args.cfl) < 1.0e-15
                and abs(float(prior.get("t_end", -1.0)) - t_end) < 1.0e-15
            )
            if same_configuration and prior.get("complete", False):
                row_by_method[key] = prior
        rows = [
            row_by_method[key]
            for key in config.EULER_METHODS
            if key in row_by_method
        ]
    write_json(out_dir / "summary.json", {"configuration": vars(args), "rows": rows})
    if failed_methods:
        raise RuntimeError(
            "formal validation failed for: " + ", ".join(failed_methods)
        )


if __name__ == "__main__":
    main()
