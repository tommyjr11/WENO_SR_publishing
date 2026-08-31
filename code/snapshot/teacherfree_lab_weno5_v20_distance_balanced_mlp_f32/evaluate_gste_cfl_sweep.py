#!/usr/bin/env python3
"""Evaluate a reflection-symmetric V20 MLP-FP32 checkpoint on held-out GSTE."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from for_paper_results.run_gste import cell_averages, gste_point, integrate_ssprk3
from teacherfree_lab_weno5 import apost_advect as adv5
from teacherfree_lab_weno5_v20_distance_balanced_mlp_f32.v20_mlp_f32_model import (
    load_checkpoint,
)


WINDOWS = (
    ("Gaussian", -0.83, -0.57),
    ("Square", -0.43, -0.17),
    ("Triangle", -0.03, 0.23),
    ("Semi-ellipse", 0.37, 0.63),
)
COLORS = {0.2: "#0072B2", 0.4: "#009E73", 0.6: "#D55E00", 0.8: "#CC79A7"}


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def metrics(method: str, cfl: float, x: np.ndarray, final: np.ndarray,
            exact: np.ndarray, meta: dict) -> dict[str, object]:
    difference = final - exact
    ellipse = (x >= 0.40) & (x <= 0.60)
    plateau = (x >= -0.38) & (x <= -0.22)
    return {
        "method": method,
        "cfl_requested": cfl,
        "cfl": float(meta["cfl"]),
        "steps": int(meta["steps"]),
        "l1": float(np.mean(np.abs(difference))),
        "l2": float(np.sqrt(np.mean(np.square(difference)))),
        "linf": float(np.max(np.abs(difference))),
        "tv": float(np.sum(np.abs(final - np.roll(final, 1)))),
        "min": float(np.min(final)),
        "max": float(np.max(final)),
        "ellipse_peak": float(np.max(final[ellipse])),
        "ellipse_min": float(np.min(final[ellipse])),
        "plateau_mean": float(np.mean(final[plateau])),
        "plateau_ripple": float(np.max(final[plateau]) - np.min(final[plateau])),
        "complete": bool(np.all(np.isfinite(final))),
    }


def plot_one(cfl: float, data: dict[str, np.ndarray], out_dir: Path,
             nx: int, t_end: float, model_label: str) -> None:
    exact_x = np.linspace(-1.0, 1.0, 20001)
    exact_line = gste_point(exact_x - t_end)
    color = COLORS.get(cfl, "#0072B2")
    values = np.concatenate((data["js"], data["v12"]))
    ymin = min(-0.04, float(np.min(values)) - 0.02)
    ymax = max(1.05, float(np.max(values)) + 0.02)

    fig, ax = plt.subplots(figsize=(9.0, 4.35), constrained_layout=True)
    ax.plot(exact_x, exact_line, color="black", lw=1.8, label="Exact")
    ax.plot(data["x"], data["js"], "--o", color="#666666", lw=1.2,
            ms=2.6, mfc="white", mew=0.65, label="WENO5-JS")
    ax.plot(data["x"], data["v12"], "-o", color=color, lw=1.45,
            ms=2.6, mfc="white", mew=0.65, label=model_label)
    ax.set(xlim=(-1.0, 1.0), ylim=(ymin, ymax), xlabel="$x$", ylabel="$u$")
    ax.set_title(f"GSTE advection ($N={nx}$, $t={t_end:g}$, CFL $={cfl:g}$)")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, out_dir / f"cfl{int(round(10*cfl)):02d}_full")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True)
    for ax, (title, xmin, xmax) in zip(axes.flat, WINDOWS):
        exact_mask = (exact_x >= xmin) & (exact_x <= xmax)
        local = (data["x"] >= xmin) & (data["x"] <= xmax)
        ax.plot(exact_x[exact_mask], exact_line[exact_mask], color="black", lw=1.8,
                label="Exact")
        ax.plot(data["x"][local], data["js"][local], "--o", color="#666666",
                lw=1.2, ms=3.7, mfc="white", mew=0.7, label="WENO5-JS")
        ax.plot(data["x"][local], data["v12"][local], "-o", color=color,
                lw=1.45, ms=3.7, mfc="white", mew=0.7, label=model_label)
        ax.set(xlim=(xmin, xmax), title=title, xlabel="$x$", ylabel="$u$")
        ax.grid(alpha=0.20)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"GSTE component details: CFL $={cfl:g}$", fontsize=12)
    save_figure(fig, out_dir / f"cfl{int(round(10*cfl)):02d}_components")


def plot_sweep(results: dict[float, dict[str, np.ndarray]], out_dir: Path,
               nx: int, t_end: float, model_label: str) -> None:
    exact_x = np.linspace(-1.0, 1.0, 20001)
    exact_line = gste_point(exact_x - t_end)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    for ax, cfl in zip(axes.flat, sorted(results)):
        data = results[cfl]
        color = COLORS.get(cfl, "#0072B2")
        ax.plot(exact_x, exact_line, color="black", lw=1.7, label="Exact")
        ax.plot(data["x"], data["js"], "--", color="#777777", lw=1.15,
                label="WENO5-JS")
        ax.plot(data["x"], data["v12"], "-o", color=color, lw=1.3,
                ms=2.0, mfc="white", mew=0.55, label=model_label)
        ax.set(xlim=(-1.0, 1.0), ylim=(-0.07, 1.08),
               title=f"CFL $={cfl:g}$", xlabel="$x$", ylabel="$u$")
        ax.grid(alpha=0.20)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"GSTE CFL sweep ($N={nx}$, $t={t_end:g}$, SSPRK3)", fontsize=12)
    save_figure(fig, out_dir / "cfl_sweep_comparison")

    fig, ax = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    ax.plot(exact_x, exact_line, color="black", lw=1.8, label="Exact")
    for cfl in sorted(results):
        data = results[cfl]
        ax.plot(data["x"], data["v12"], lw=1.25, color=COLORS.get(cfl),
                label=f"{model_label}, CFL={cfl:g}")
    ax.set(xlim=(-1.0, 1.0), ylim=(-0.07, 1.08), xlabel="$x$", ylabel="$u$")
    ax.set_title(f"{model_label} CFL sensitivity")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, out_dir / "v20_mlp_f32_cfl_overlay")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    step_text = args.model.stem.rsplit("_", 1)[-1]
    step = int(step_text)
    step_label = f"{step // 1000}k" if step % 1000 == 0 else f"{step:,}"
    model_label = f"WENO5-V20 MLP-FP32 reflection-sym @ {step_label}"

    if args.reuse_existing:
        rows: list[dict[str, object]] = []
        with (args.out_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["method"] != "WENO5-JS":
                    row["method"] = model_label
                rows.append(row)
        results: dict[float, dict[str, np.ndarray]] = {}
        for cfl in args.cfls:
            archive = np.load(args.out_dir / f"cfl{int(round(10*cfl)):02d}.npz")
            results[cfl] = {key: archive[key] for key in archive.files}
            plot_one(cfl, results[cfl], args.out_dir, args.nx, args.t_end, model_label)
        with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_sweep(results, args.out_dir, args.nx, args.t_end, model_label)
        print(f"GSTE_REDRAW_COMPLETE out={args.out_dir}", flush=True)
        return

    device = torch.device(args.device)
    model = load_checkpoint(args.model, device)
    model.eval()
    x, initial = cell_averages(args.nx, 0.0, args.quadrature)
    _, exact = cell_averages(args.nx, args.t_end, args.quadrature)
    rows: list[dict[str, object]] = []
    results: dict[float, dict[str, np.ndarray]] = {}

    for cfl in args.cfls:
        print(f"GSTE_START cfl={cfl:g}", flush=True)
        js, js_meta = integrate_ssprk3(
            "classical", adv5.ssprk3, initial, cfl, args.t_end, device
        )
        v12, v12_meta = integrate_ssprk3(
            model, adv5.ssprk3, initial, cfl, args.t_end, device
        )
        js_row = metrics("WENO5-JS", cfl, x, js, exact, js_meta)
        v12_row = metrics(model_label, cfl, x, v12, exact, v12_meta)
        rows.extend((js_row, v12_row))
        results[cfl] = {
            "x": x,
            "initial": initial,
            "exact": exact,
            "js": js,
            "v12": v12,
        }
        np.savez_compressed(
            args.out_dir / f"cfl{int(round(10*cfl)):02d}.npz", **results[cfl]
        )
        plot_one(cfl, results[cfl], args.out_dir, args.nx, args.t_end, model_label)
        print("GSTE_DONE", js_row, flush=True)
        print("GSTE_DONE", v12_row, flush=True)

    fields = tuple(rows[0])
    with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    plot_sweep(results, args.out_dir, args.nx, args.t_end, model_label)
    print(f"GSTE_COMPLETE out={args.out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--quadrature", type=int, default=15)
    parser.add_argument("--cfls", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Redraw existing NPZ/CSV outputs without rerunning GSTE.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
