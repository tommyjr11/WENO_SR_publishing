#!/usr/bin/env python3
"""Create the paper-layout Lax figure with order-matched WENO-Z curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results import exact_riemann
from .plot_riemann_1d_zoom import METHODS, result_path
from .run_riemann_1d import PROBLEMS


ROOT = Path(__file__).resolve().parent


def load_conserved(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["x"], dtype=np.float64),
            np.asarray(data["conserved_1d"], dtype=np.float64),
        )


def primitive(conserved: np.ndarray) -> dict[str, np.ndarray]:
    rho = conserved[:, 0]
    velocity = conserved[:, 1] / rho
    transverse = conserved[:, 2] / rho
    pressure = 0.4 * (
        conserved[:, 3]
        - 0.5 * rho * (velocity * velocity + transverse * transverse)
    )
    if (
        not np.all(np.isfinite(conserved))
        or np.any(rho <= 0.0)
        or np.any(pressure <= 0.0)
    ):
        raise ValueError("invalid Lax state; refusing plotting-time repair")
    return {"rho": rho, "velocity": velocity, "pressure": pressure}


def error_metrics(numerical: np.ndarray, exact: np.ndarray, dx: float) -> dict[str, float]:
    delta = numerical - exact
    return {
        "l1_mean": float(np.mean(np.abs(delta))),
        "l1_integral": float(dx * np.sum(np.abs(delta))),
        "l2_mean": float(np.sqrt(np.mean(delta * delta))),
        "linf": float(np.max(np.abs(delta))),
    }


def main() -> None:
    problem = PROBLEMS["lax"]
    dx = (problem.x_max - problem.x_min) / problem.nx
    centers = problem.x_min + (np.arange(problem.nx) + 0.5) * dx
    results = {
        method: load_conserved(result_path("lax", method, None))
        for method, *_ in METHODS
    }
    for method, (x, _) in results.items():
        if not np.array_equal(x, centers):
            raise RuntimeError(f"{method}: grid does not match the formal Lax grid")

    exact_conserved = exact_riemann.cell_average_conserved(
        centers, dx, problem.t_end, problem.x0, problem.left, problem.right,
        quadrature=15,
    )
    exact_coarse = primitive(exact_conserved)
    exact_x = np.linspace(problem.x_min, problem.x_max, 10001)
    exact_rho, exact_velocity, exact_pressure = exact_riemann.sample_points(
        exact_x, problem.t_end, problem.x0, problem.left, problem.right,
    )
    exact_points = {
        "rho": exact_rho,
        "velocity": exact_velocity,
        "pressure": exact_pressure,
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": 7.0,
    })
    fields = (
        ("rho", r"Density $\rho$"),
        ("velocity", r"Velocity $u$"),
        ("pressure", r"Pressure $p$"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(11.6, 8.0), constrained_layout=True)
    for row, (field, ylabel) in enumerate(fields):
        for column, axis in enumerate(axes[row]):
            axis.plot(
                exact_x, exact_points[field], color="black", lw=1.45,
                label="Exact Riemann solution", zorder=2,
            )
            for method, label, color, _, marker in METHODS:
                values = primitive(results[method][1])[field]
                axis.plot(
                    centers, values, color=color, linestyle="none",
                    marker=marker, markevery=3, markersize=3.0,
                    markerfacecolor="white", markeredgewidth=0.7,
                    label=label, zorder=3,
                )
            axis.set_ylabel(ylabel if column == 0 else "")
            axis.grid(alpha=0.20, linewidth=0.55)
            axis.set_xlim(
                (problem.x_min, problem.x_max) if column == 0 else (1.75, 3.45)
            )
            if row == 0:
                axis.set_title("Full domain" if column == 0 else "Contact/shock zoom")
            if row == 2:
                axis.set_xlabel(r"$x$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        rf"Lax shock tube: $N={problem.nx}$, $t={problem.t_end:g}$, "
        rf"CFL $={problem.cfl:g}$; characteristic HLLC"
    )
    out_dir = ROOT / "figures/riemann_1d/lax"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "lax_primitive_compare_with_weno_z_rminus1"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    rows = []
    for method, label, *_ in METHODS:
        values = primitive(results[method][1])
        row: dict[str, object] = {
            "problem": "lax",
            "method": method,
            "label": label,
            "nx": problem.nx,
            "t_end": problem.t_end,
            "cfl": problem.cfl,
            "reference": "15-point Gauss exact cell averages",
        }
        for field, _ in fields:
            row.update({
                f"{field}_{name}": value
                for name, value in error_metrics(
                    values[field], exact_coarse[field], dx,
                ).items()
            })
        rows.append(row)
    table = ROOT / "tables/riemann_1d/lax/lax_errors_vs_exact_fv.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(stem.with_suffix(".png"))
    print(table)


if __name__ == "__main__":
    main()
