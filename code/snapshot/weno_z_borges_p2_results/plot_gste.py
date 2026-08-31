#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/gste/N200_t10_cfl06"
OUT = ROOT / "figures/gste/N200_t10_cfl06"
TABLE = ROOT / "tables/gste/N200_t10_cfl06/gste_errors_with_weno_z_rminus1.csv"
PAPER_RAW = ROOT.parent / "for_paper_results/raw/gste"


STYLE = {
    "weno5_js": ("WENO5-JS-RK3", "#6E6E6E", "--", "o"),
    "weno5_sr_f64": ("WENO5-SR-RK3", "#0072B2", "-", "s"),
    "weno5_sr_f32": ("WENO5-SR-FP32-RK3", "#009E73", "-.", "^"),
    "weno5_z_p2": ("WENO5-Z-RK3", "#56B4E9", ":", "P"),
    "weno7_js": ("WENO7-JS-RK4", "#A65628", "--", "D"),
    "weno7_sr_f64": ("WENO7-SR-RK4", "#CC79A7", "-", "v"),
    "weno7_z_p3": ("WENO7-Z-RK4", "#E69F00", ":", "X"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for key in STYLE:
        path = (RAW if "_z_" in key else PAPER_RAW) / f"{key}.npz"
        data[key] = np.load(path)
    x = data["weno5_z_p2"]["x"]
    exact_cells = data["weno5_z_p2"]["exact"]
    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    dense_x = np.linspace(-1.0, 1.0, 20001)
    from for_paper_results.run_gste import gste_point
    ax.plot(dense_x, gste_point(dense_x - 10.0), color="black", lw=1.25, label="Exact")
    for key, values in data.items():
        label, color, linestyle, marker = STYLE[key]
        ax.plot(x, values["final"], color=color, ls=linestyle, lw=1.0,
                marker=marker, ms=2.2, markevery=4, mfc="white", mew=0.55,
                label=label)
    ax.set(xlabel=r"$x$", ylabel=r"$u$", xlim=(-1.0, 1.0), ylim=(-0.08, 1.08))
    ax.grid(alpha=0.18, lw=0.5)
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=7.5)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"gste_all_methods_with_weno_z_rminus1.{suffix}", bbox_inches="tight")
    plt.close(fig)

    windows = ((-0.82, -0.58, "Gaussian"), (-0.42, -0.18, "Square"),
               (-0.02, 0.22, "Triangle"), (0.38, 0.62, "Semi-ellipse"))
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7))
    for ax, (lo, hi, title) in zip(axes.flat, windows):
        mask = (x >= lo) & (x <= hi)
        xd = np.linspace(lo, hi, 1800)
        ax.plot(xd, gste_point(xd - 10.0), color="black", lw=1.2, label="Exact")
        for key, values in data.items():
            label, color, linestyle, marker = STYLE[key]
            ax.plot(x[mask], values["final"][mask], color=color, ls=linestyle,
                    lw=0.95, marker=marker, ms=2.7, markevery=2,
                    mfc="white", mew=0.55, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(lo, hi)
        ax.grid(alpha=0.18, lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.985), fontsize=7.2)
    fig.supxlabel(r"$x$")
    fig.supylabel(r"$u$")
    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.08, top=0.84,
        hspace=0.36, wspace=0.20,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"gste_all_methods_with_weno_z_rminus1_components.{suffix}", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for key, values in data.items():
        error = np.asarray(values["final"]) - exact_cells
        label = STYLE[key][0]
        rows.append({
            "method": key,
            "label": label,
            "nx": x.size,
            "t_end": 10.0,
            "cfl": 0.6,
            "l1": float(np.mean(np.abs(error))),
            "l2": float(np.sqrt(np.mean(error * error))),
            "linf": float(np.max(np.abs(error))),
            "tv": float(np.sum(np.abs(np.asarray(values["final"]) - np.roll(values["final"], 1)))),
        })
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "gste_all_methods_with_weno_z_rminus1.png")
    print(TABLE)


if __name__ == "__main__":
    main()
