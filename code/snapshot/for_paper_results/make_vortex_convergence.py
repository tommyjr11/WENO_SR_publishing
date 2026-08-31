#!/usr/bin/env python3
"""Plot and tabulate the five-method isentropic-vortex convergence study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
import numpy as np

from for_paper_results import config


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def format_order(value: str) -> str:
    number = float(value)
    return "--" if not np.isfinite(number) else f"{number:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--expected-cfl", type=float, default=None)
    args = parser.parse_args()

    rows = read_rows(args.raw_dir / "metrics.csv")
    cfl_values = {float(row["cfl"]) for row in rows}
    if len(cfl_values) != 1:
        raise RuntimeError(f"mixed CFL values in vortex table: {cfl_values}")
    cfl = cfl_values.pop()
    if args.expected_cfl is not None and abs(cfl - args.expected_cfl) > 1.0e-14:
        raise RuntimeError(f"expected CFL {args.expected_cfl}, found {cfl}")
    expected_grids = [25, 50, 100, 200]
    for key in config.EULER_METHODS:
        subset = sorted(
            (row for row in rows if row["method"] == key),
            key=lambda row: int(row["N"]),
        )
        if [int(row["N"]) for row in subset] != expected_grids:
            raise RuntimeError(f"incomplete grid series for {key}")
        if not all(row["complete"].lower() in ("true", "1") for row in subset):
            raise RuntimeError(f"nonphysical/incomplete vortex result for {key}")
        if not all(abs(float(row["cfl"]) - cfl) < 1.0e-14 for row in subset):
            raise RuntimeError(f"unexpected CFL in vortex result for {key}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for key in config.EULER_METHODS:
        subset = sorted(
            (row for row in rows if row["method"] == key),
            key=lambda row: int(row["N"]),
        )
        grids = np.array([int(row["N"]) for row in subset])
        method = config.METHODS[key]
        for ax, metric, title in zip(
            axes,
            ("rho_l1", "rho_l2"),
            ("Density $L_1$ error", "Density $L_2$ error"),
        ):
            ax.loglog(
                grids,
                [float(row[metric]) for row in subset],
                marker="o",
                markersize=4.0,
                color=method.color,
                linestyle=method.linestyle,
                label=method.label,
            )
            ax.set_title(title)
            ax.set_xlabel("Grid size $N$")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(FixedLocator(expected_grids))
            ax.set_xticklabels([str(n) for n in expected_grids])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("Error")
    axes[1].set_ylabel("Error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Periodic isentropic vortex at $t=2$: characteristic HLLC, CFL $={cfl:g}$"
    )
    save_figure(fig, args.figure_dir / "vortex_convergence")

    args.table_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.table_dir / "vortex_convergence.md"
    lines = [
        "# Periodic isentropic-vortex convergence",
        "",
        f"Domain `[-10,10]^2`, final time `t=2`, CFL `{cfl:g}`, periodic boundary, "
        "characteristic HLLC, and 15x15 Gauss cell averages for both the initial "
        "and exact final states.",
        "",
        "| Method | N | L1(rho) | order | L2(rho) | order |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in config.EULER_METHODS:
        subset = sorted(
            (row for row in rows if row["method"] == key),
            key=lambda row: int(row["N"]),
        )
        for row in subset:
            lines.append(
                f"| {config.METHODS[key].label} | {row['N']} | "
                f"{float(row['rho_l1']):.8e} | {format_order(row['rho_l1_order'])} | "
                f"{float(row['rho_l2']):.8e} | {format_order(row['rho_l2_order'])} |"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex_path = args.table_dir / "vortex_convergence.tex"
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Density errors and observed orders for the periodic "
            rf"isentropic vortex at $t=2$ and CFL ${cfl:g}$.}}"
        ),
        r"\label{tab:vortex_convergence}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & $N$ & $L_1(\rho)$ & order & $L_2(\rho)$ & order \\",
        r"\midrule",
    ]
    for method_index, key in enumerate(config.EULER_METHODS):
        subset = sorted(
            (row for row in rows if row["method"] == key),
            key=lambda row: int(row["N"]),
        )
        for row in subset:
            latex_lines.append(
                f"{config.METHODS[key].label} & {row['N']} & "
                f"{float(row['rho_l1']):.3e} & {format_order(row['rho_l1_order'])} & "
                f"{float(row['rho_l2']):.3e} & {format_order(row['rho_l2_order'])} \\\\"
            )
        if method_index + 1 < len(config.EULER_METHODS):
            latex_lines.append(r"\midrule")
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    print(f"figure={args.figure_dir / 'vortex_convergence.png'}", flush=True)
    print(f"table={markdown_path}", flush=True)


if __name__ == "__main__":
    main()
