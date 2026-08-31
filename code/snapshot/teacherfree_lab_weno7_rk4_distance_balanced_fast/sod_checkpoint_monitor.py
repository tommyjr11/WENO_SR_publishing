#!/usr/bin/env python3
"""Evaluate every completed WENO7 checkpoint with the trusted 2D Sod path."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sod_eval
import validate_sod
from warp_sod.point_rk4_mlp import TorchWeno7PointBeta


STEP_PATTERN = re.compile(r"model_step_(\d+)\.npz$")


def load_rows(path: Path) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["step"]): dict(row)
            for row in csv.DictReader(stream)
            if row.get("step")
        }


def ready_checkpoints(
    run_dir: Path, interval: int
) -> list[tuple[int, Path]]:
    ready: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("model_step_*.npz"):
        match = STEP_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step > 0 and step % interval == 0:
            ready.append((step, path))
    return sorted(ready)


def save_trends(path: Path, rows: list[dict[str, object]]) -> None:
    steps = np.asarray([int(row["step"]) for row in rows])
    gains = np.asarray(
        [float(row["gain_l2_percent"]) for row in rows], dtype=np.float64
    )
    figure, axis = plt.subplots(
        figsize=(7.2, 4.0), constrained_layout=True
    )
    axis.axhline(0.0, color="0.5", lw=0.8)
    axis.plot(steps, gains, color="#0072B2", marker="o", markersize=3.0)
    axis.set_xlabel("Training step")
    axis.set_ylabel(r"Density $L_2$ gain over WENO7-JS (\%)")
    axis.grid(alpha=0.20)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    if args.interval <= 0 or args.poll_seconds <= 0:
        raise ValueError("interval and poll-seconds must be positive")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    out_dir = run_dir / "sod_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "sod_metrics.csv"
    rows = load_rows(metrics_path)
    params = sod_eval.make_sod_params(
        args.nx, args.ny, args.t_end, args.cfl
    )
    print(
        f"sod_monitor_start run={run_dir} interval={args.interval} "
        f"grid={args.nx}x{args.ny} cfl={args.cfl} t_end={args.t_end} "
        f"space=characteristic solver={args.solver} cutoff=False "
        f"mlp_forward=0.5*(M(x)+P4*M(P6*x)) existing={len(rows)}",
        flush=True,
    )
    print("SOD_CLASSICAL_START", flush=True)
    classical = sod_eval.eval_classical(
        params,
        args.device,
        solver=args.solver,
        report_interval=args.report_interval,
    )
    exact_average = sod_eval.reference_density(params, args.t_end)
    print(
        f"SOD_CLASSICAL_DONE L2={float(classical['l2']):.6e}",
        flush=True,
    )

    while True:
        pending = [
            item
            for item in ready_checkpoints(run_dir, args.interval)
            if item[0] not in rows
        ]
        for step, checkpoint in pending:
            print(
                f"SOD_START step={step:06d} checkpoint={checkpoint}",
                flush=True,
            )
            try:
                provider = TorchWeno7PointBeta(
                    checkpoint, args.device, params.gamma
                )
                result = sod_eval.eval_mlp(
                    params,
                    args.device,
                    provider,
                    solver=args.solver,
                    eno_cutoff=False,
                    report_interval=args.report_interval,
                )
            except Exception as error:
                print(
                    f"SOD_RETRY step={step:06d} "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                break

            gain_l1 = 100.0 * (
                1.0 - float(result["l1"]) / float(classical["l1"])
            )
            gain_l2 = 100.0 * (
                1.0 - float(result["l2"]) / float(classical["l2"])
            )
            row: dict[str, object] = {
                "step": step,
                "mlp_l1": result["l1"],
                "classical_l1": classical["l1"],
                "gain_l1_percent": gain_l1,
                "mlp_l2": result["l2"],
                "classical_l2": classical["l2"],
                "gain_l2_percent": gain_l2,
                "mlp_linf": result["linf"],
                "classical_linf": classical["linf"],
                "finite": result["finite"],
                "solver": args.solver,
                "characteristic": True,
                "eno_cutoff": False,
                "nx": args.nx,
                "ny": args.ny,
                "cfl": args.cfl,
                "t_end": args.t_end,
                "checkpoint": str(checkpoint),
                "reflection_formula": "0.5*(M(x)+P4*M(P6*x))",
            }
            step_dir = out_dir / f"step_{step:06d}"
            validate_sod.save_plot(
                step_dir / "sod_density.png",
                params,
                step,
                exact_average,
                classical,
                result,
                args.solver,
            )
            (step_dir / "summary.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows[step] = row
            ordered = [rows[key] for key in sorted(rows)]
            validate_sod.write_csv(metrics_path, ordered)
            save_trends(out_dir / "sod_metrics_trends.png", ordered)
            print(
                f"SOD_DONE step={step:06d} gain_L2={gain_l2:.2f}% "
                f"failed={int(not bool(result['finite']))}",
                flush=True,
            )

        if args.max_step in rows:
            print(f"sod_monitor_complete step={args.max_step:06d}", flush=True)
            return
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=250)
    parser.add_argument("--max-step", type=int, default=200000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.25)
    parser.add_argument("--solver", choices=("hllc", "evilin"), default="hllc")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--report-interval", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
