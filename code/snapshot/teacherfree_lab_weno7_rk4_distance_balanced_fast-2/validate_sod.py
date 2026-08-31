#!/usr/bin/env python3
"""Run trusted characteristic WENO7/Shu-RK4 Sod model selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sod_eval
from warp_sod.point_rk4_mlp import TorchWeno7PointBeta


def save_plot(
    path: Path,
    params,
    step: int,
    exact_average: np.ndarray,
    classical: dict[str, object],
    model: dict[str, object],
    solver: str,
) -> None:
    centers = params.x_min + (
        np.arange(params.nx) + 0.5
    ) * params.dx
    dense_x = np.linspace(params.x_min, params.x_max, 5001)
    dense_density, _, _ = sod_eval.sod_exact.exact_primitive(
        dense_x, params.t_end, params.gamma
    )
    figure, axis = plt.subplots(
        figsize=(8.0, 4.3), constrained_layout=True
    )
    axis.plot(dense_x, dense_density, color="black", lw=1.8, label="Exact")
    axis.plot(
        centers,
        classical["density"],
        color="0.45",
        ls="--",
        marker="o",
        markersize=3.2,
        markerfacecolor="none",
        markeredgewidth=0.6,
        label=f"WENO7-JS (L2={float(classical['l2']):.3e})",
    )
    axis.plot(
        centers,
        model["density"],
        color="#0072B2",
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.6,
        label=f"WENO7-SR step {step} (L2={float(model['l2']):.3e})",
    )
    axis.scatter(
        centers,
        exact_average,
        s=8,
        color="black",
        alpha=0.28,
        label="Exact cell averages",
    )
    gain = 100.0 * (
        1.0 - float(model["l2"]) / float(classical["l2"])
    )
    axis.set_xlabel("$x$")
    axis.set_ylabel("Density")
    axis.set_title(
        f"Sod, characteristic {solver.upper()}, Shu RK4, "
        f"N={params.nx}, step={step}, gain={gain:.2f}%"
    )
    axis.grid(alpha=0.20)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--t-end", type=float, default=0.25)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--solver", choices=("hllc", "evilin"), default="hllc")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--eno-cutoff",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--report-interval", type=int, default=0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else run_dir / "sod_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    params = sod_eval.make_sod_params(
        args.nx, args.ny, args.t_end, args.cfl
    )
    print(
        f"classical characteristic solver={args.solver} "
        f"N={args.nx}x{args.ny}",
        flush=True,
    )
    classical = sod_eval.eval_classical(
        params,
        args.device,
        solver=args.solver,
        report_interval=args.report_interval,
    )
    exact_average = sod_eval.reference_density(params, args.t_end)
    rows: list[dict[str, object]] = []
    for step in args.steps:
        checkpoint = (
            run_dir / "checkpoints" / f"model_step_{step:06d}.npz"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"MLP step={step} checkpoint={checkpoint}", flush=True)
        provider = TorchWeno7PointBeta(
            checkpoint, args.device, params.gamma
        )
        result = sod_eval.eval_mlp(
            params,
            args.device,
            provider,
            solver=args.solver,
            eno_cutoff=args.eno_cutoff,
            report_interval=args.report_interval,
        )
        gain_l2 = 100.0 * (
            1.0 - float(result["l2"]) / float(classical["l2"])
        )
        gain_l1 = 100.0 * (
            1.0 - float(result["l1"]) / float(classical["l1"])
        )
        row = {
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
            "eno_cutoff": args.eno_cutoff,
            "nx": args.nx,
            "ny": args.ny,
            "cfl": args.cfl,
            "t_end": args.t_end,
            "checkpoint": str(checkpoint),
        }
        rows.append(row)
        step_dir = out_dir / f"step_{step:06d}"
        save_plot(
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
        print(
            f"  L2={float(result['l2']):.6e} "
            f"JS={float(classical['l2']):.6e} "
            f"gain={gain_l2:.2f}%",
            flush=True,
        )
    write_csv(out_dir / "sod_metrics.csv", rows)
    print(f"complete: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
