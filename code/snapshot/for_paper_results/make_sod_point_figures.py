#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pretrain_weno7_offline as sod_exact
from for_paper_results import config


MARKERS = {
    "weno5_js": "o",
    "weno5_sr_f64": "s",
    "weno5_sr_f32": "^",
    "weno7_js": "D",
    "weno7_sr_f64": "v",
}


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex_sci(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value)))) if value else 0
    mantissa = value / (10.0 ** exponent) if value else 0.0
    return f"${mantissa:.3f}\\times10^{{{exponent}}}$"


def write_table(rows: list[dict[str, str]]) -> None:
    path = config.TABLES / "sod_point_errors.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Density errors for the one-dimensional Sod problem at $t=0.2$.}",
        "\\label{tab:sod_point_errors}",
        "\\begin{tabular}{rlrrr}",
        "\\toprule",
        "$N$ & Method & $L_1(\\rho)$ & $L_2(\\rho)$ & $L_\\infty(\\rho)$ \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        if index and index % len(config.EULER_METHODS) == 0:
            lines.append("\\midrule")
        lines.append(
            f"{row['nx']} & {config.METHODS[row['method']].label} & "
            f"{latex_sci(float(row['rho_l1']))} & {latex_sci(float(row['rho_l2']))} & "
            f"{latex_sci(float(row['rho_linf']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "lines.linewidth": 1.4,
    })
    exact_x = np.linspace(0.0, 1.0, 10000, dtype=np.float64)
    exact_rho, _, _ = sod_exact.exact_sod_primitive(exact_x - 0.5, 0.2, 1.4)
    raw51 = config.RAW / "sod/N51_t020"
    with (raw51 / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        combined_rows = list(csv.DictReader(stream))
    complete = {
        row["method"] for row in combined_rows
        if row["complete"].lower() in ("true", "1")
    }
    if complete != set(config.EULER_METHODS):
        raise RuntimeError(f"incomplete Sod results in {raw51}")

    fig = plt.figure(figsize=(10.8, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.15, 1.0))
    axes51 = [fig.add_subplot(grid[0, :])]
    axes51.extend(fig.add_subplot(grid[1, column]) for column in range(3))
    for ax in axes51:
        ax.plot(exact_x, exact_rho, "k-", lw=1.9, label="Exact pointwise solution")
    for key in config.EULER_METHODS:
        data = np.load(raw51 / f"{key}.npz")
        method = config.METHODS[key]
        for ax in axes51:
            ax.plot(
                data["x"], data["rho"], linestyle="none", marker=MARKERS[key],
                ms=3.8, markerfacecolor="none", markeredgewidth=0.85,
                color=method.color, label=method.label,
            )
    axes51[0].set_xlim(0.0, 1.0)
    axes51[0].set_ylim(0.08, 1.04)
    axes51[0].set_ylabel("Density $\\rho$")
    axes51[0].set_title("Sod shock tube, $N=51$, $t=0.2$")
    axes51[0].legend(ncol=3, frameon=False, loc="lower left", fontsize=7)

    zooms = (
        ("Post-rarefaction plateau", 0.49, 0.66, 0.405, 0.445),
        ("Contact", 0.64, 0.73, 0.245, 0.445),
        ("Shock and right plateau", 0.80, 0.93, 0.10, 0.29),
    )
    for ax, (title, xmin, xmax, ymin, ymax) in zip(axes51[1:], zooms):
        ax.set_title(title)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Density $\\rho$")
    for ax in axes51:
        ax.set_xlabel("$x$")
        ax.grid(alpha=0.22)
    save(fig, config.FIGURES / "sod/sod_density_points")

    combined_path = config.RAW / "sod/sod_point_errors.csv"
    fields = ("nx", "ny", "method", "rho_l1", "rho_l2", "rho_linf", "complete")
    with combined_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({field: row[field] for field in fields})
    write_table(combined_rows)


if __name__ == "__main__":
    main()
