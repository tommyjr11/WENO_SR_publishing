#!/usr/bin/env python3
"""Evaluate multiple WENO7 checkpoints on the unseen GSTE advection test."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import rk4_advection as A
import weno7_core as W

X_MIN = -1.0
X_MAX = 1.0
LENGTH = X_MAX - X_MIN
COMPONENTS = (
    ("Gaussian", -0.7, (-0.84, -0.56)),
    ("Square", -0.3, (-0.44, -0.16)),
    ("Triangle", 0.1, (-0.04, 0.24)),
    ("Semi-ellipse", 0.5, (0.36, 0.64)),
)
COLORS = {
    0.2: "#0072B2",
    0.4: "#009E73",
    0.6: "#E69F00",
    0.8: "#CC79A7",
}


def wrap(x: np.ndarray) -> np.ndarray:
    return X_MIN + np.mod(x - X_MIN, LENGTH)


def gste_point(x: np.ndarray) -> np.ndarray:
    x = wrap(np.asarray(x, dtype=np.float64))
    delta = 0.005
    beta = math.log(2.0) / (36.0 * delta * delta)
    z = -0.7
    a = 0.5
    alpha = 10.0

    def gaussian(center: float) -> np.ndarray:
        return np.exp(-beta * np.square(x - center))

    def ellipse(center: float) -> np.ndarray:
        return np.sqrt(
            np.maximum(1.0 - np.square(alpha * (x - center)), 0.0)
        )

    output = np.zeros_like(x)
    mask = (-0.8 < x) & (x < -0.6)
    output[mask] = (
        gaussian(z - delta)[mask]
        + 4.0 * gaussian(z)[mask]
        + gaussian(z + delta)[mask]
    ) / 6.0
    mask = (-0.4 < x) & (x < -0.2)
    output[mask] = 1.0
    mask = (0.0 < x) & (x < 0.2)
    output[mask] = 1.0 - np.abs(10.0 * (x[mask] - 0.1))
    mask = (0.4 < x) & (x < 0.6)
    output[mask] = (
        ellipse(a - delta)[mask]
        + 4.0 * ellipse(a)[mask]
        + ellipse(a + delta)[mask]
    ) / 6.0
    return output


def cell_averages(
    nx: int, time: float, quadrature: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    dx = LENGTH / float(nx)
    centers = X_MIN + (np.arange(nx) + 0.5) * dx
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    points = centers[:, None] + 0.5 * dx * nodes[None, :] - time
    averages = 0.5 * np.sum(
        weights[None, :] * gste_point(points), axis=1
    )
    return centers, averages


@torch.no_grad()
def integrate(
    model,
    initial: np.ndarray,
    cfl_limit: float,
    t_end: float,
    device: torch.device,
    *,
    eno_cutoff: bool,
    report_interval: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    nx = int(initial.size)
    dx = LENGTH / float(nx)
    n_steps = int(math.ceil(t_end / (cfl_limit * dx)))
    dt = t_end / float(n_steps)
    actual_cfl = dt / dx
    state = torch.as_tensor(
        initial, device=device, dtype=torch.float64
    ).reshape(1, nx)
    velocity = torch.ones(1, device=device, dtype=torch.float64)
    if hasattr(model, "eval"):
        model.eval()
    for step in range(1, n_steps + 1):
        state = A.shu_rk4_step_signed(
            model,
            state,
            dt,
            1.0 / dx,
            velocity,
            eno_cutoff=eno_cutoff,
        )
        if report_interval > 0 and step % report_interval == 0:
            print(
                f"  progress {step}/{n_steps} "
                f"range=[{float(state.min()):.5g},{float(state.max()):.5g}]",
                flush=True,
            )
        if not bool(torch.all(torch.isfinite(state))):
            break
    final = state[0].detach().cpu().numpy()
    return final, {
        "steps": n_steps,
        "dt": dt,
        "cfl": actual_cfl,
        "t_end": t_end,
        "finite": bool(np.all(np.isfinite(final))),
    }


def metrics(
    final: np.ndarray,
    exact: np.ndarray,
    meta: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    difference = final - exact
    finite = bool(meta["finite"])
    return {
        **meta,
        "l1": float(np.mean(np.abs(difference))) if finite else float("nan"),
        "l2": (
            float(np.sqrt(np.mean(np.square(difference))))
            if finite
            else float("nan")
        ),
        "linf": (
            float(np.max(np.abs(difference))) if finite else float("nan")
        ),
        "tv": (
            float(np.sum(np.abs(final - np.roll(final, 1))))
            if finite
            else float("nan")
        ),
        "minimum": float(np.min(final)) if finite else float("nan"),
        "maximum": float(np.max(final)) if finite else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def material_coordinates(x: np.ndarray, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    coordinate = wrap(x - t_end)
    order = np.argsort(coordinate)
    return coordinate[order], order


def plot_checkpoint(
    step_dir: Path,
    step: int,
    x: np.ndarray,
    t_end: float,
    results: dict[float, dict[str, np.ndarray | dict[str, object]]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.45,
        }
    )
    dense_x = np.linspace(X_MIN, X_MAX, 20001)
    dense_exact = gste_point(dense_x)
    material_x, order = material_coordinates(x, t_end)

    cfl_values = sorted(results)
    columns = 2 if len(cfl_values) > 1 else 1
    rows = int(math.ceil(len(cfl_values) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0 if columns == 2 else 6.4, 3.5 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, cfl in zip(axes.flat, cfl_values):
        item = results[cfl]
        classical = np.asarray(item["classical"])[order]
        model = np.asarray(item["model"])[order]
        axis.plot(dense_x, dense_exact, color="black", lw=1.8, label="Exact")
        axis.plot(
            material_x,
            classical,
            color="0.42",
            ls="--",
            marker="o",
            markersize=2.0,
            markerfacecolor="none",
            markeredgewidth=0.45,
            label="WENO7-JS",
        )
        axis.plot(
            material_x,
            model,
            color=COLORS.get(cfl, "#0072B2"),
            marker="o",
            markersize=2.0,
            markerfacecolor="white",
            markeredgewidth=0.5,
            label=f"WENO7-SR step {step}",
        )
        metric = item["model_metrics"]
        axis.set_title(
            f"CFL = {cfl:g}, L1/JS = {float(metric['l1_ratio_js']):.3f}"
        )
        axis.set_xlim(X_MIN, X_MAX)
        axis.set_ylim(-0.12, 1.12)
        axis.grid(alpha=0.20)
        axis.legend(frameon=False, loc="upper right")
    for axis in axes.flat[len(cfl_values) :]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("$u$")
    for axis in axes[-1, :]:
        axis.set_xlabel("material coordinate")
    figure.suptitle(
        f"GSTE, WENO7-SR reflection-symmetric model, step {step}, "
        f"N={x.size}, t={t_end:g}"
    )
    figure.tight_layout()
    save_figure(figure, step_dir / "profiles_by_cfl")

    for cfl in sorted(results):
        item = results[cfl]
        classical = np.asarray(item["classical"])[order]
        model = np.asarray(item["model"])[order]
        figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
        for axis, (name, _center, limits) in zip(axes.flat, COMPONENTS):
            axis.plot(
                dense_x, dense_exact, color="black", lw=1.8, label="Exact"
            )
            axis.plot(
                material_x,
                classical,
                color="0.42",
                ls="--",
                marker="o",
                markersize=3.0,
                markerfacecolor="none",
                markeredgewidth=0.6,
                label="WENO7-JS",
            )
            axis.plot(
                material_x,
                model,
                color=COLORS.get(cfl, "#0072B2"),
                marker="o",
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.6,
                label=f"WENO7-SR step {step}",
            )
            axis.set_xlim(*limits)
            axis.set_ylim(-0.10, 1.08)
            axis.set_title(name)
            axis.set_xlabel("material coordinate")
            axis.set_ylabel("$u$")
            axis.grid(alpha=0.20)
        axes[0, 0].legend(frameon=False, loc="best")
        figure.suptitle(
            f"GSTE component details, step {step}, CFL = {cfl:g}"
        )
        figure.tight_layout()
        tag = str(cfl).replace(".", "p")
        save_figure(figure, step_dir / f"components_cfl_{tag}")


def plot_ranking(rows: list[dict[str, object]], out_dir: Path) -> None:
    if not rows:
        return
    steps = np.array([int(row["step"]) for row in rows])
    mean_ratio = np.array([float(row["mean_l1_ratio_js"]) for row in rows])
    worst_ratio = np.array([float(row["worst_l1_ratio_js"]) for row in rows])
    figure, axis = plt.subplots(figsize=(7.5, 4.1), constrained_layout=True)
    axis.plot(steps, mean_ratio, "o-", label="Mean over CFL")
    axis.plot(steps, worst_ratio, "s-", label="Worst CFL")
    axis.axhline(1.0, color="black", lw=1.0, ls="--", label="WENO7-JS")
    axis.set_xlabel("Training step")
    axis.set_ylabel("$L_1 / L_{1,\\mathrm{JS}}$")
    axis.set_title("GSTE checkpoint selection")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    save_figure(figure, out_dir / "checkpoint_ranking")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument(
        "--cfls", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8)
    )
    parser.add_argument("--quadrature", type=int, default=15)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--eno-cutoff",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--report-interval", type=int, default=0)
    args = parser.parse_args()

    W.check_weno7_coefficients()
    A.check_shu_rk4_order()
    device = W.torch_device(args.device)
    run_dir = args.run_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else run_dir / "gste_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    x, initial = cell_averages(args.nx, 0.0, args.quadrature)
    _, exact = cell_averages(args.nx, args.t_end, args.quadrature)

    classical_results: dict[float, tuple[np.ndarray, dict[str, object]]] = {}
    for cfl in args.cfls:
        print(f"classical WENO7-JS CFL={cfl:g}", flush=True)
        final, meta = integrate(
            "classical",
            initial,
            cfl,
            args.t_end,
            device,
            eno_cutoff=False,
            report_interval=args.report_interval,
        )
        classical_metric = metrics(final, exact, meta)
        classical_results[float(cfl)] = (final, classical_metric)

    all_metric_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for step in args.steps:
        checkpoint_path = (
            run_dir / "checkpoints" / f"model_step_{step:06d}.npz"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model = W.load_checkpoint(checkpoint_path, device)
        model.eval()
        step_dir = out_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        per_cfl: dict[
            float, dict[str, np.ndarray | dict[str, object]]
        ] = {}
        ratios = []
        stable = True
        for cfl in args.cfls:
            cfl = float(cfl)
            print(f"model step={step} CFL={cfl:g}", flush=True)
            final, meta = integrate(
                model,
                initial,
                cfl,
                args.t_end,
                device,
                eno_cutoff=args.eno_cutoff,
                report_interval=args.report_interval,
            )
            model_metric = metrics(final, exact, meta)
            classical, classical_metric = classical_results[cfl]
            ratio = float(model_metric["l1"]) / max(
                float(classical_metric["l1"]), 1.0e-300
            )
            model_metric.update(
                {
                    "step": step,
                    "cfl_requested": cfl,
                    "l1_ratio_js": ratio,
                    "eno_cutoff": args.eno_cutoff,
                    "checkpoint": str(checkpoint_path),
                }
            )
            ratios.append(ratio)
            stable = stable and bool(model_metric["finite"])
            per_cfl[cfl] = {
                "classical": classical,
                "model": final,
                "classical_metrics": classical_metric,
                "model_metrics": model_metric,
            }
            np.savez(
                step_dir / f"result_cfl_{str(cfl).replace('.', 'p')}.npz",
                x=x,
                initial=initial,
                exact=exact,
                classical=classical,
                model=final,
                model_metrics_json=np.array(
                    json.dumps(model_metric, sort_keys=True)
                ),
                classical_metrics_json=np.array(
                    json.dumps(classical_metric, sort_keys=True)
                ),
            )
            all_metric_rows.append(model_metric)
            print(
                f"  L1={float(model_metric['l1']):.6e} "
                f"JS={float(classical_metric['l1']):.6e} "
                f"ratio={ratio:.4f} TV={float(model_metric['tv']):.5f} "
                f"range=[{float(model_metric['minimum']):.5f},"
                f"{float(model_metric['maximum']):.5f}]",
                flush=True,
            )
        plot_checkpoint(step_dir, step, x, args.t_end, per_cfl)
        step_metrics = [
            row for row in all_metric_rows if int(row["step"]) == step
        ]
        write_csv(step_dir / "metrics.csv", step_metrics)
        ranking_rows.append(
            {
                "step": step,
                "stable_all_cfl": stable,
                "mean_l1_ratio_js": float(np.mean(ratios)),
                "worst_l1_ratio_js": float(np.max(ratios)),
                "best_l1_ratio_js": float(np.min(ratios)),
                "checkpoint": str(checkpoint_path),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranking_rows.sort(
        key=lambda row: (
            not bool(row["stable_all_cfl"]),
            float(row["worst_l1_ratio_js"]),
            float(row["mean_l1_ratio_js"]),
        )
    )
    write_csv(out_dir / "all_metrics.csv", all_metric_rows)
    write_csv(out_dir / "model_ranking.csv", ranking_rows)
    plot_ranking(sorted(ranking_rows, key=lambda row: int(row["step"])), out_dir)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "steps": args.steps,
                "nx": args.nx,
                "t_end": args.t_end,
                "cfls": args.cfls,
                "quadrature": args.quadrature,
                "eno_cutoff": args.eno_cutoff,
                "time_integrator": "Shu fourth-order TVD RK with L_tilde",
                "model_forward": "reflection symmetric",
                "GSTE_used_in_training": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"complete: {out_dir}", flush=True)
    print("ranking:", flush=True)
    for row in ranking_rows:
        print(
            f"  step={int(row['step']):06d} "
            f"worst={float(row['worst_l1_ratio_js']):.4f} "
            f"mean={float(row['mean_l1_ratio_js']):.4f} "
            f"stable={row['stable_all_cfl']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
