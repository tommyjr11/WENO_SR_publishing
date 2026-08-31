#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results import config
from for_paper_results.run_gste import gste_point


PAPER_METHODS = (
    "weno5_js",
    "weno5_sr_f64",
    "weno5_sr_f32",
    "weno7_js",
    "weno7_sr_f64",
    "weno9_tt",
    "weno11_tt",
)
METHODS = PAPER_METHODS + ("weno5_offline137k",)

LABELS = {
    "weno5_js": "WENO5-JS-SSPRK3",
    "weno5_offline137k": "WENO5-SR-Offline137k-SSPRK3",
    "weno5_sr_f64": "WENO5-SR-SSPRK3",
    "weno5_sr_f32": "WENO5-SR-FP32-SSPRK3",
    "weno7_js": "WENO7-JS-SSPRK3",
    "weno7_sr_f64": "WENO7-SR-SSPRK3",
    "weno9_tt": "WENO9-TT",
    "weno11_tt": "WENO11-TT",
}
COMPARISON_LABELS = {
    "weno5_js": "WENO5-JS",
    "weno5_offline137k": "Offline137k + cutoff",
    "weno5_sr_f64": "Distance-balanced FP64",
    "weno5_sr_f32": "Teacher-free FP32 + cutoff",
}

COLORS = {
    key: config.METHODS[key].color for key in config.EULER_METHODS
}
COLORS.update({"weno9_tt": "#E69F00", "weno11_tt": "#56B4E9"})
COLORS["weno5_offline137k"] = "#7A5195"


def parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if not methods or unknown:
        raise ValueError(f"invalid GSTE methods: {unknown or methods}")
    return methods


def read_metrics(path: Path, methods: tuple[str, ...]) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["method"]: row for row in csv.DictReader(stream)}
    missing = [
        key for key in methods
        if key not in rows or rows[key].get("complete", "").lower() not in ("true", "1")
    ]
    if missing:
        raise RuntimeError(f"incomplete GSTE results for {missing}")
    return rows


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tag", default="N200_t10")
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--methods", default=",".join(PAPER_METHODS))
    args = parser.parse_args()
    methods = parse_methods(args.methods)

    raw = config.RAW / "gste" / args.input_tag
    rows = read_metrics(raw / "metrics.csv", methods)
    summary = json.loads((raw / "summary.json").read_text(encoding="utf-8"))
    nx = int(summary["nx"])
    t_end = float(summary["t_end"])
    output_tag = args.output_tag or args.input_tag
    out_dir = config.FIGURES / "gste" / output_tag / "individual"

    loaded = {key: np.load(raw / f"{key}.npz") for key in methods}
    all_values = np.concatenate([loaded[key]["final"] for key in methods])
    ymin = min(-0.08, float(np.min(all_values)) - 0.025)
    ymax = max(1.08, float(np.max(all_values)) + 0.025)
    exact_x = np.linspace(-1.0, 1.0, 20001, dtype=np.float64)
    exact = gste_point(exact_x - t_end)
    eno_cutoff = bool(summary.get("eno_cutoff", False))

    def method_label(key: str) -> str:
        suffix = " + ENO cutoff" if eno_cutoff and key.startswith("weno5_") and key != "weno5_js" else ""
        return LABELS[key] + suffix

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
    })
    for key in methods:
        data = loaded[key]
        row = rows[key]
        fig, ax = plt.subplots(figsize=(7.6, 3.7), constrained_layout=True)
        ax.plot(exact_x, exact, color="black", lw=1.8, label="Exact")
        ax.plot(
            data["x"], data["final"], color=COLORS[key], lw=1.35,
            marker="o", markersize=2.2, markerfacecolor="none",
            markeredgewidth=0.55, label=method_label(key),
        )
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"GSTE advection: {method_label(key)} ($N={nx}$, $t={t_end:g}$)")
        ax.grid(alpha=0.20)
        ax.legend(frameon=False, loc="upper right")
        ax.text(
            0.015, 0.965,
            f"$L_1={float(row['l1']):.3e}$\n$L_2={float(row['l2']):.3e}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        )
        save(fig, out_dir / key)

    if len(methods) > 1:
        fig, ax = plt.subplots(figsize=(8.8, 4.3), constrained_layout=True)
        ax.plot(exact_x, exact, color="black", lw=1.9, label="Exact")
        for key in methods:
            data = loaded[key]
            ax.plot(
                data["x"], data["final"], color=COLORS[key], lw=1.25,
                marker="o", markersize=1.8, markerfacecolor="none",
                markeredgewidth=0.45,
                label=COMPARISON_LABELS.get(key, method_label(key)),
            )
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.set_title(f"GSTE advection comparison ($N={nx}$, $t={t_end:g}$)")
        ax.grid(alpha=0.20)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
        )
        save(fig, out_dir.parent / "comparison")


if __name__ == "__main__":
    main()
