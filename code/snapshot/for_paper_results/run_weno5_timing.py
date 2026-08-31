#!/usr/bin/env python3
"""Time classical and learned WENO5 paths on Riemann configuration C.4."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time

import numpy as np

from run_weno5_circle_mlp_compare import load_mlp_params as load_f64
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_circle_mlp_compare_mlp_f32 import (
    load_mlp_params as load_f32,
)
from for_paper_results import config
from for_paper_results.common import interior, state_health, write_json
from for_paper_results.run_quadrant import CASES, make_quadrant_state
from for_paper_results.run_vortex import pad_periodic
from for_paper_results.solvers import euler_methods, weno5_hllc, weno5_hllc_mixed
from teacherfree_lab_weno5_v20_distance_balanced import weno5_hllc_refsym


def latex_table(rows: list[dict]) -> None:
    path = config.TABLES / "weno5_precision_timing.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{WENO5 wall time on the $400^2$ Riemann configuration C.4. "
        "Compilation, model loading, initialisation, diagnostics, and output are excluded.}",
        "\\label{tab:weno5_precision_timing}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Problem & Method & Mean (s) & Std. dev. (s) & Cost/JS & Overhead \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        if index and row["case"] != rows[index - 1]["case"]:
            lines.append("\\midrule")
        lines.append(
            f"{row['case_label']} & {row['label']} & {row['mean_seconds']:.3f} & "
            f"{row['std_seconds']:.3f} & {row['cost_vs_classical']:.2f}$\\times$ & "
            f"{row['overhead_percent']:.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--methods",
        default="weno5_js,weno5_sr_f64,weno5_sr_f32",
        help="comma-separated methods to retime; unselected validated rows are retained",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    selected = {item.strip() for item in args.methods.split(",") if item.strip()}
    known = {"weno5_js", "weno5_sr_f64", "weno5_sr_f32"}
    if not selected or selected - known:
        raise ValueError(f"invalid --methods selection: {sorted(selected - known)}")

    config.ensure_output_dirs()
    config.validate_models()
    variants = (
        ("weno5_js", "WENO5-JS", False, weno5_hllc, None),
        ("weno5_sr_f64", "FP64 MLP", False, weno5_hllc_refsym, load_f64),
        ("weno5_sr_f32", "FP32 MLP", True, weno5_hllc_mixed, load_f32),
    )
    cases = (("c4", "C.4", CASES["c4"]),)
    rows: list[dict] = []
    out = config.RAW / "timing"
    old_rows: dict[tuple[str, str], dict] = {}
    summary_path = out / "summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        old_rows = {
            (row["case"], row["method"]): row for row in prior.get("rows", [])
        }
    nx = ny = 400
    for case, case_label, definition in cases:
        t_end = float(definition["t_end"])
        p7 = euler_methods.make_weno7_params(
            nx, ny, 0.0, 1.0, 0.0, 1.0, 0.4, t_end,
        )
        initial7 = make_quadrant_state(p7, definition, 15)
        initial_interior = interior(initial7, p7.ghost, nx, ny)
        case_rows: list[dict] = []
        for key, label, mixed, adapter, loader in variants:
            if key not in selected:
                prior = old_rows.get((case, key))
                if prior is None or not prior.get("complete", False):
                    raise RuntimeError(
                        f"cannot retain missing or incomplete timing row for {case}/{key}"
                    )
                case_rows.append(prior)
                rows.append(prior)
                continue
            params = euler_methods.make_weno5_params(
                nx, ny, 1.0, 1.0, 0.4, t_end, mixed,
            )
            initial = pad_periodic(initial_interior, params.ghost)
            mlp_params = None if loader is None else loader(
                config.METHODS[key].model, args.device,
            )

            warm, _, _, warm_t = adapter.run_to_time(
                initial, params, t_end, args.device, mlp_params, "transmissive",
                report_interval=0, report=None,
            )
            warm_health = state_health(warm, params.ghost, nx, ny)
            if not warm_health["complete"] or abs(warm_t - t_end) >= 1.0e-12:
                raise RuntimeError(
                    f"warm-up failed for {case}/{key}: {warm_health}, t={warm_t}"
                )

            samples: list[float] = []
            final_health = warm_health
            for repeat in range(args.repeats):
                started = time.perf_counter()
                final, _, _, final_t = adapter.run_to_time(
                    initial, params, t_end, args.device, mlp_params, "transmissive",
                    report_interval=0, report=None,
                )
                elapsed = time.perf_counter() - started
                final_health = state_health(final, params.ghost, nx, ny)
                if not final_health["complete"] or abs(final_t - t_end) >= 1.0e-12:
                    raise RuntimeError(
                        f"timed repeat {repeat + 1} failed for {case}/{key}: "
                        f"{final_health}, t={final_t}"
                    )
                samples.append(elapsed)
                print(
                    f"{case} {key} repeat={repeat + 1} seconds={elapsed:.6f}",
                    flush=True,
                )

            row = {
                "case": case,
                "case_label": case_label,
                "method": key,
                "label": label,
                "samples_seconds": samples,
                "mean_seconds": statistics.fmean(samples),
                "std_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "complete": True,
                **final_health,
            }
            case_rows.append(row)
            rows.append(row)
        classical_time = next(
            row["mean_seconds"]
            for row in case_rows
            if row["method"] == "weno5_js"
        )
        for row in case_rows:
            row["cost_vs_classical"] = row["mean_seconds"] / classical_time
            row["overhead_percent"] = 100.0 * (row["cost_vs_classical"] - 1.0)

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", {"repeats": args.repeats, "rows": rows})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case", "method", "label", "mean_seconds", "std_seconds",
                "cost_vs_classical", "overhead_percent", "complete",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in writer.fieldnames})
    latex_table(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
