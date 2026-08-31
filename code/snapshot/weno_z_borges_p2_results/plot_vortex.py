#!/usr/bin/env python3
"""Merge the five paper methods and two WENO-Z vortex convergence series."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
import numpy as np

from for_paper_results import config


ROOT = Path(__file__).resolve().parent
BASE_RAW = ROOT.parent / "for_paper_results/raw/vortex_cfl04/metrics.csv"
Z_RAW = ROOT / "raw/vortex_cfl04/metrics.csv"
OUT = ROOT / "figures/vortex_cfl04"
TABLES = ROOT / "tables/vortex_cfl04"
ORDER = (
    "weno5_js", "weno5_z_p2", "weno5_sr_f64", "weno5_sr_f32",
    "weno7_js", "weno7_z_p3", "weno7_sr_f64",
)
LABELS = {
    **{key: config.METHODS[key].label for key in config.EULER_METHODS},
    "weno5_z_p2": "WENO5-Z-RK3",
    "weno7_z_p3": "WENO7-Z-RK4",
}
STYLES = {
    **{
        key: (config.METHODS[key].color, config.METHODS[key].linestyle, "o")
        for key in config.EULER_METHODS
    },
    "weno5_z_p2": ("#56B4E9", ":", "P"),
    "weno7_z_p3": ("#E69F00", ":", "X"),
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def order_text(value: str) -> str:
    number = float(value)
    return "--" if not np.isfinite(number) else f"{number:.3f}"


def main() -> None:
    rows = read(BASE_RAW) + read(Z_RAW)
    grids = [25, 50, 100, 200]
    for method in ORDER:
        subset = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["N"]),
        )
        if [int(row["N"]) for row in subset] != grids:
            raise RuntimeError(f"incomplete vortex series for {method}")
        if not all(row["complete"].lower() in ("true", "1") for row in subset):
            raise RuntimeError(f"incomplete/nonphysical vortex result for {method}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.2,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), constrained_layout=True)
    for method in ORDER:
        subset = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["N"]),
        )
        color, linestyle, marker = STYLES[method]
        for axis, metric, title in zip(
            axes,
            ("rho_l1", "rho_l2"),
            (r"Density $L_1$ error", r"Density $L_2$ error"),
        ):
            axis.loglog(
                grids,
                [float(row[metric]) for row in subset],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.2,
                linewidth=1.25,
                label=LABELS[method],
            )
            axis.set_title(title)
            axis.set_xlabel(r"Grid size $N$")
            axis.set_xscale("log", base=2)
            axis.xaxis.set_major_locator(FixedLocator(grids))
            axis.set_xticklabels([str(n) for n in grids])
            axis.xaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.grid(True, which="both", alpha=0.22)
            axis.set_ylabel("Error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        r"Periodic isentropic vortex at $t=2$: characteristic HLLC, CFL $=0.4$"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 350}), ("pdf", {})):
        fig.savefig(OUT / f"vortex_convergence_with_weno_z.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    TABLES.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# Periodic isentropic-vortex convergence with WENO-Z",
        "",
        "| Method | N | L1(rho) | order | L2(rho) | order |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Density errors and observed orders for the periodic isentropic vortex at $t=2$ and CFL $0.4$.}",
        r"\label{tab:vortex_convergence}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.7pt}",
        r"\renewcommand{\arraystretch}{0.90}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & $N$ & $L_1(\rho)$ & order & $L_2(\rho)$ & order \\",
        r"\midrule",
    ]
    for method_index, method in enumerate(ORDER):
        subset = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["N"]),
        )
        for row in subset:
            markdown.append(
                f"| {LABELS[method]} | {row['N']} | {float(row['rho_l1']):.8e} | "
                f"{order_text(row['rho_l1_order'])} | {float(row['rho_l2']):.8e} | "
                f"{order_text(row['rho_l2_order'])} |"
            )
            latex.append(
                f"{LABELS[method]} & {row['N']} & {float(row['rho_l1']):.3e} & "
                f"{order_text(row['rho_l1_order'])} & {float(row['rho_l2']):.3e} & "
                f"{order_text(row['rho_l2_order'])} \\\\"
            )
        if method_index + 1 < len(ORDER):
            latex.append(r"\midrule")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "vortex_convergence_with_weno_z.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (TABLES / "vortex_convergence_with_weno_z.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )
    print(OUT / "vortex_convergence_with_weno_z.png")
    print(TABLES / "vortex_convergence_with_weno_z.tex")


if __name__ == "__main__":
    main()
