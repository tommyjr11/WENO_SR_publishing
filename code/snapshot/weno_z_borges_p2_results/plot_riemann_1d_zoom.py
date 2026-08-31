#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results import exact_riemann
import pretrain_weno7_offline as sod_exact
from .run_riemann_1d import PROBLEMS, sod_cell_average_conserved


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "for_paper_results/raw"

METHODS = (
    ("weno5_js", "WENO5-JS-RK3", "#666666", "--", "o"),
    ("weno5_z_p2", "WENO5-Z-RK3", "#56B4E9", "-", "P"),
    ("weno5_sr_f64", "WENO5-SR-RK3", "#0072B2", "-", "s"),
    ("weno5_sr_f32", "WENO5-SR-FP32-RK3", "#009E73", "-.", "^"),
    ("weno7_js", "WENO7-JS-RK4", "#A65628", "--", "D"),
    ("weno7_z_p3", "WENO7-Z-RK4", "#E69F00", "-", "X"),
    ("weno7_sr_f64", "WENO7-SR-RK4", "#CC79A7", "-", "v"),
)

ZOOMS = {
    "sod": (
        ("Post-rarefaction plateau", 0.49, 0.66, 0.405, 0.445),
        ("Contact", 0.64, 0.73, 0.245, 0.445),
        ("Shock and right plateau", 0.80, 0.93, 0.10, 0.29),
    ),
    "lax": (
        ("Rarefaction", -3.65, -1.85, None, None),
        ("Contact discontinuity", 1.72, 2.24, None, None),
        ("Shock", 2.94, 3.48, None, None),
    ),
}


def result_path(problem_key: str, method: str, input_tag: str | None) -> Path:
    if "_z_" in method:
        base = ROOT / f"raw/riemann_1d/{problem_key}"
        if input_tag:
            base = base / input_tag
        return base / f"{method}.npz"
    old = (
        BASE / "sod/N51_t020"
        if problem_key == "sod"
        else BASE / "shock_tubes_cfl08/lax/N200x10"
    )
    if input_tag:
        old = BASE / problem_key / input_tag
    return old / f"{method}.npz"


def load_density(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        if "rho" in data:
            rho = np.asarray(data["rho"], dtype=np.float64)
        elif "conserved_1d" in data:
            rho = np.asarray(data["conserved_1d"], dtype=np.float64)[:, 0]
        else:
            rho = np.asarray(data["state"], dtype=np.float64).mean(axis=0)[:, 0]
    return x, rho


def padded_limits(
    xmin: float,
    xmax: float,
    exact_x: np.ndarray,
    exact_rho: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    values = [exact_rho[(exact_x >= xmin) & (exact_x <= xmax)]]
    for x, rho in results.values():
        values.append(rho[(x >= xmin) & (x <= xmax)])
    selected = np.concatenate([value for value in values if value.size])
    lower = float(np.min(selected))
    upper = float(np.max(selected))
    padding = max(0.025 * (upper - lower), 0.005)
    return lower - padding, upper + padding


def write_errors(
    problem_key: str,
    problem,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_tag: str | None,
) -> Path:
    dx = (problem.x_max - problem.x_min) / problem.nx
    centers = problem.x_min + (np.arange(problem.nx) + 0.5) * dx
    if problem_key == "sod":
        exact = sod_cell_average_conserved(
            centers, dx, problem.t_end, problem.x0, quadrature=15,
        )[:, 0]
    else:
        exact = exact_riemann.cell_average_conserved(
            centers,
            dx,
            problem.t_end,
            problem.x0,
            problem.left,
            problem.right,
            quadrature=15,
        )[:, 0]

    labels = {method: label for method, label, *_ in METHODS}
    rows = []
    for method, (x, rho) in results.items():
        if not np.array_equal(x, centers):
            raise RuntimeError(f"{method}: numerical grid does not match exact FV grid")
        error = rho - exact
        rows.append({
            "problem": problem_key,
            "method": method,
            "label": labels[method],
            "nx": problem.nx,
            "t_end": problem.t_end,
            "cfl": problem.cfl,
            "exact_reference": "15-point Gauss exact cell averages",
            "rho_l1_mean": float(np.mean(np.abs(error))),
            "rho_l1_integral": float(dx * np.sum(np.abs(error))),
            "rho_l2": float(np.sqrt(np.mean(error * error))),
            "rho_linf": float(np.max(np.abs(error))),
        })

    out_dir = ROOT / f"tables/riemann_1d/{problem_key}"
    if output_tag:
        out_dir = out_dir / output_tag
    out = out_dir / f"{problem_key}_density_errors_vs_exact_fv.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def plot(
    problem_key: str,
    *,
    cfl: float | None = None,
    nx: int | None = None,
    ny: int | None = None,
    input_tag: str | None = None,
    output_tag: str | None = None,
    paper_style: bool = False,
) -> tuple[Path, Path]:
    problem = PROBLEMS[problem_key]
    if cfl is not None:
        problem = replace(problem, cfl=cfl)
    if nx is not None:
        problem = replace(problem, nx=nx)
    if ny is not None:
        problem = replace(problem, ny=ny)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, *_ in METHODS:
        path = result_path(problem_key, method, input_tag)
        if not path.is_file():
            raise FileNotFoundError(path)
        results[method] = load_density(path)

    exact_x = np.linspace(problem.x_min, problem.x_max, 20001, dtype=np.float64)
    if problem_key == "sod":
        exact_rho = sod_exact.exact_sod_primitive(
            exact_x - problem.x0, problem.t_end, 1.4,
        )[0]
    else:
        exact_rho = exact_riemann.sample_points(
            exact_x,
            problem.t_end,
            problem.x0,
            problem.left,
            problem.right,
        )[0]

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
        "legend.fontsize": 7.2,
    })
    fig = plt.figure(figsize=(10.8, 6.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.12, 1.0))
    full = fig.add_subplot(grid[0, :])
    zoom_axes = [fig.add_subplot(grid[1, col]) for col in range(3)]

    full.plot(exact_x, exact_rho, color="black", lw=1.75, label="Exact")
    for method, label, color, linestyle, marker in METHODS:
        x, rho = results[method]
        numerical_linestyle = "none" if paper_style else linestyle
        full.plot(
            x, rho, color=color, ls=numerical_linestyle, lw=0.9,
            marker=marker, ms=3.8 if paper_style else 3.0,
            markevery=max(1, problem.nx // 70),
            mfc="white", mew=0.8 if paper_style else 0.65, label=label,
        )
    full.set_xlim(problem.x_min, problem.x_max)
    full.set_xlabel(r"$x$")
    full.set_ylabel(r"Density $\rho$")
    title = "Sod shock tube" if problem_key == "sod" else "Lax shock tube"
    full.set_title(
        rf"{title}: $N={problem.nx}$, $t={problem.t_end:g}$, "
        rf"$\mathrm{{CFL}}={problem.cfl:g}$"
    )
    full.grid(alpha=0.18, lw=0.45)
    full.legend(ncol=4, frameon=False, loc="best", columnspacing=1.25)
    if paper_style and problem_key == "sod":
        full.set_ylim(0.08, 1.04)

    for ax, zoom in zip(zoom_axes, ZOOMS[problem_key]):
        zoom_title, xmin, xmax, ymin, ymax = zoom
        ax.plot(exact_x, exact_rho, color="black", lw=1.65)
        for method, _, color, linestyle, marker in METHODS:
            x, rho = results[method]
            ax.plot(
                x, rho, color=color,
                ls="none" if paper_style else linestyle, lw=0.9,
                marker=marker, ms=4.2 if paper_style else 3.9,
                mfc="white", mew=0.8 if paper_style else 0.7,
            )
        if ymin is None or ymax is None:
            ymin, ymax = padded_limits(xmin, xmax, exact_x, exact_rho, results)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(zoom_title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"Density $\rho$")
        ax.grid(alpha=0.2, lw=0.45)

    out_dir = ROOT / f"figures/riemann_1d/{problem_key}"
    if output_tag:
        out_dir = out_dir / output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_kind = "points" if paper_style else "zoom"
    stem = out_dir / f"{problem_key}_density_{figure_kind}_with_weno_z_rminus1"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    table = write_errors(problem_key, problem, results, output_tag)
    return stem.with_suffix(".png"), table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=tuple(PROBLEMS), required=True)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--input-tag", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--paper-style", action="store_true")
    args = parser.parse_args()
    figure, table = plot(
        args.problem,
        cfl=args.cfl,
        nx=args.nx,
        ny=args.ny,
        input_tag=args.input_tag,
        output_tag=args.output_tag,
        paper_style=args.paper_style,
    )
    print(figure)
    print(table)


if __name__ == "__main__":
    main()
