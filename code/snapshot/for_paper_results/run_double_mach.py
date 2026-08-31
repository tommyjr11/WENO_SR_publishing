#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import warp_weno5_helpers as wh5
from run_double_mach_compare import make_initial_state
from run_weno5_circle_mlp_compare import load_mlp_params as load_weno5_f64
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32 import warp_weno5_helpers_mlp_f32 as wh5_f32
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_circle_mlp_compare_mlp_f32 import (
    load_mlp_params as load_weno5_f32,
)

from for_paper_results import config
from for_paper_results.common import interior, state_health, write_json
from for_paper_results.solvers import (
    weno5_hllc_mixed,
    weno7_double_mach,
    weno7_warp_beta,
)
from teacherfree_lab_weno5_v20_distance_balanced import weno5_hllc_refsym


BASELINE_DIR = config.ROOT / "plots/WENO5_MLP/weno_double_reflective_1200"


def density_errors(state: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = state[..., 0] - reference[..., 0]
    return {
        "rho_l1": float(np.mean(np.abs(delta))),
        "rho_l2": float(np.sqrt(np.mean(delta * delta))),
        "rho_linf": float(np.max(np.abs(delta))),
    }


def report(method: str):
    def callback(step: int, t: float, dt: float, stats: dict[str, float]) -> None:
        print(
            f"{method} step={step:05d} t={t:.8e} dt={dt:.4e} "
            f"rho=[{stats['rho_min']:.5e},{stats['rho_max']:.5e}] "
            f"p=[{stats['p_min']:.5e},{stats['p_max']:.5e}] "
            f"nan={int(stats['nan_count'])}",
            flush=True,
        )

    return callback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=1200)
    parser.add_argument("--ny", type=int, default=300)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument(
        "--weno7-beta-backend",
        choices=("warp", "torch"),
        default="warp",
        help="WENO7-SR beta evaluator; both use the same FP64 checkpoint map",
    )
    parser.add_argument(
        "--methods",
        default="weno5_sr_f64,weno5_sr_f32,weno7_js,weno7_sr_f64",
        help="comma-separated paper methods to run",
    )
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    supported = {
        "weno5_sr_f64", "weno5_sr_f32", "weno7_js", "weno7_sr_f64",
    }
    unknown = set(methods) - supported
    if unknown:
        raise ValueError(f"unsupported double-Mach method(s): {sorted(unknown)}")
    if args.init_quadrature != 15:
        raise ValueError("paper double-Mach tests require 15-point initialisation")
    if (args.nx, args.ny) != (1200, 300):
        raise ValueError("paper double-Mach comparison is fixed at 1200x300")
    config.ensure_output_dirs()
    config.validate_models()
    for filename in ("weno5_classical.npy", "weno7_ader4.npy"):
        if not (BASELINE_DIR / filename).is_file():
            raise FileNotFoundError(BASELINE_DIR / filename)

    baseline5_full = np.load(BASELINE_DIR / "weno5_classical.npy", mmap_mode="r")
    baseline7_full = np.load(BASELINE_DIR / "weno7_ader4.npy", mmap_mode="r")
    baseline5 = np.asarray(baseline5_full[3 : 3 + args.ny, 3 : 3 + args.nx, :])
    baseline7 = np.asarray(baseline7_full[4 : 4 + args.ny, 4 : 4 + args.nx, :])
    out_dir = config.RAW / "double_mach"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    method_runs = []
    if {"weno5_sr_f64", "weno5_sr_f32"} & set(methods):
        params64 = wh5.Params(
            nx=args.nx, ny=args.ny, x_length=4.0, y_length=1.0,
            cfl=args.cfl, t_end=args.t_end,
        )
        initial64 = make_initial_state(params64, args.init_quadrature)

    if "weno5_sr_f64" in methods:
        params_model64 = load_weno5_f64(
            config.METHODS["weno5_sr_f64"].model, args.device,
        )
        final64, dts64, steps64, t64 = weno5_hllc_refsym.run_to_time(
            initial64,
            params64,
            args.t_end,
            args.device,
            params_model64,
            "double-mach",
            report_interval=args.report_interval,
            report=report("weno5_sr_f64"),
        )
        method_runs.append(
            ("weno5_sr_f64", final64, params64, t64, steps64, dts64)
        )

    if "weno5_sr_f32" in methods:
        params32 = wh5_f32.Params(
            nx=args.nx, ny=args.ny, x_length=4.0, y_length=1.0,
            cfl=args.cfl, t_end=args.t_end,
        )
        initial32 = make_initial_state(params32, args.init_quadrature)
        params_model32 = load_weno5_f32(
            config.METHODS["weno5_sr_f32"].model, args.device,
        )
        final32, dts32, steps32, t32 = weno5_hllc_mixed.run_to_time(
            initial32,
            params32,
            args.t_end,
            args.device,
            params_model32,
            "double-mach",
            report_interval=args.report_interval,
            report=report("weno5_sr_f32"),
        )
        method_runs.append(
            ("weno5_sr_f32", final32, params32, t32, steps32, dts32)
        )

    if {"weno7_js", "weno7_sr_f64"} & set(methods):
        params7 = weno7_double_mach.Params(
            nx=args.nx,
            ny=args.ny,
            x_min=0.0,
            x_max=4.0,
            y_min=0.0,
            y_max=1.0,
            cfl=args.cfl,
            t_end=args.t_end,
        )
        initial7 = make_initial_state(params7, args.init_quadrature)

    if "weno7_js" in methods:
        final7_js, dts7_js, steps7_js, t7_js = (
            weno7_double_mach.run_to_time(
                initial7,
                params7,
                args.t_end,
                args.device,
                None,
                report_interval=args.report_interval,
                report=report("weno7_js"),
            )
        )
        method_runs.append(
            ("weno7_js", final7_js, params7, t7_js, steps7_js, dts7_js)
        )

    if "weno7_sr_f64" in methods:
        beta_class = (
            weno7_warp_beta.WarpWeno7PointBeta
            if args.weno7_beta_backend == "warp"
            else weno7_double_mach.TorchWeno7PointBeta
        )
        beta7 = beta_class(
            config.METHODS["weno7_sr_f64"].model, args.device, params7.gamma
        )
        final7_sr, dts7_sr, steps7_sr, t7_sr = (
            weno7_double_mach.run_to_time(
                initial7,
                params7,
                args.t_end,
                args.device,
                beta7,
                report_interval=args.report_interval,
                report=report("weno7_sr_f64"),
            )
        )
        method_runs.append(
            ("weno7_sr_f64", final7_sr, params7, t7_sr, steps7_sr, dts7_sr)
        )

    failed_methods: list[str] = []
    for key, final, params, t, steps, dts in method_runs:
        state = interior(final, params.ghost, args.nx, args.ny)
        health = state_health(final, params.ghost, args.nx, args.ny)
        complete_time = abs(float(t) - args.t_end) < 1.0e-12
        valid_run = bool(health["complete"] and complete_time)
        if valid_run:
            versus5 = density_errors(state, baseline5)
            versus7 = density_errors(state, baseline7)
        else:
            versus5 = {"rho_l1": None, "rho_l2": None, "rho_linf": None}
            versus7 = {"rho_l1": None, "rho_l2": None, "rho_linf": None}
        row: dict[str, object] = {
            "method": key,
            "nx": args.nx,
            "ny": args.ny,
            "cfl": args.cfl,
            "t_end": args.t_end,
            "t": float(t),
            "steps": int(steps),
            "dt_min": float(min(dts)) if dts else 0.0,
            "dt_max": float(max(dts)) if dts else 0.0,
            "dt_mean": float(np.mean(dts)) if dts else 0.0,
            "complete_time": complete_time,
            "riemann_solver": "hllc_direct_minmax_tiny1e-16",
            "weno_space": "characteristic",
            "eno_cutoff": False,
            "time_integrator": (
                "SSP-RK4" if key.startswith("weno7") else "SSPRK3"
            ),
            "reflection_symmetrization": (
                "0.5*(M(x)+P*M(P*x))"
                if key in {"weno5_sr_f64", "weno5_sr_f32", "weno7_sr_f64"}
                else None
            ),
            "weno7_beta_backend": (
                args.weno7_beta_backend if key == "weno7_sr_f64" else None
            ),
            "rho_l1_vs_weno5_js": versus5["rho_l1"],
            "rho_l2_vs_weno5_js": versus5["rho_l2"],
            "rho_linf_vs_weno5_js": versus5["rho_linf"],
            "rho_l1_vs_weno7_js": versus7["rho_l1"],
            "rho_l2_vs_weno7_js": versus7["rho_l2"],
            "rho_linf_vs_weno7_js": versus7["rho_linf"],
            **health,
        }
        row["complete"] = valid_run
        np.savez(
            out_dir / f"{key}.npz",
            state=state,
            metadata_json=np.array(json.dumps(row, sort_keys=True)),
        )
        write_json(out_dir / f"{key}.json", row)
        rows.append(row)
        print(row, flush=True)
        if not row["complete"]:
            failed_methods.append(key)

    row_map = {str(row["method"]): row for row in rows}
    for key in supported - set(row_map):
        row_path = out_dir / f"{key}.json"
        if row_path.is_file():
            previous = json.loads(row_path.read_text(encoding="utf-8"))
            if previous.get("complete"):
                row_map[key] = previous
    rows = [
        row_map[key]
        for key in (
            "weno5_sr_f64", "weno5_sr_f32", "weno7_js", "weno7_sr_f64"
        )
        if key in row_map
    ]

    fields = [
        "method", "nx", "ny", "cfl", "t_end", "complete", "steps",
        "rho_l1_vs_weno5_js", "rho_l2_vs_weno5_js", "rho_linf_vs_weno5_js",
        "rho_l1_vs_weno7_js", "rho_l2_vs_weno7_js", "rho_linf_vs_weno7_js",
        "rho_min", "rho_max", "p_min", "p_max", "nan_count",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out_dir / "summary.json",
        {
            "configuration": vars(args),
            "baseline_directory": str(BASELINE_DIR),
            "baseline_weno5": str(BASELINE_DIR / "weno5_classical.npy"),
            "baseline_weno7": str(BASELINE_DIR / "weno7_ader4.npy"),
            "rows": rows,
        },
    )
    if failed_methods:
        raise RuntimeError(
            "double-Mach validation failed for: " + ", ".join(failed_methods)
        )


if __name__ == "__main__":
    main()
