#!/usr/bin/env python3
"""Create publication figures for the Lax and half blast-wave tests."""

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
from for_paper_results import exact_riemann


TITLES = {
    "lax": "Lax shock tube",
    "woodward_colella_half": "Left half of the Woodward--Colella blast wave",
}
ZOOM_RANGES = {
    "lax": (1.75, 3.45),
    "woodward_colella_half": (0.70, 0.81),
}
MARKERS = {
    "weno5_js": "o",
    "weno5_sr_f64": "s",
    "weno5_sr_f32": "^",
    "weno7_js": "D",
    "weno7_sr_f64": "v",
}


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def load_result(path: Path) -> dict[str, object]:
    with np.load(path) as data:
        return {
            "x": np.asarray(data["x"], dtype=np.float64),
            "conserved": np.asarray(data["conserved_1d"], dtype=np.float64),
            "metadata": json.loads(str(data["metadata_json"].item())),
        }


def primitive(conserved: np.ndarray, gamma: float = 1.4) -> dict[str, np.ndarray]:
    rho = conserved[:, 0]
    if np.any(rho <= 0.0) or not np.all(np.isfinite(conserved)):
        raise ValueError("invalid conserved result; refusing plotting-time repair")
    velocity = conserved[:, 1] / rho
    transverse = conserved[:, 2] / rho
    pressure = (gamma - 1.0) * (
        conserved[:, 3] - 0.5 * rho * (velocity * velocity + transverse * transverse)
    )
    if np.any(pressure <= 0.0) or not np.all(np.isfinite(pressure)):
        raise ValueError("invalid primitive result; refusing plotting-time repair")
    return {"rho": rho, "velocity": velocity, "pressure": pressure}


def restrict_conserved(reference: np.ndarray, target_nx: int) -> np.ndarray:
    if reference.shape[0] % target_nx != 0:
        raise ValueError(
            f"reference size {reference.shape[0]} is not divisible by target {target_nx}"
        )
    ratio = reference.shape[0] // target_nx
    return reference.reshape(target_nx, ratio, 4).mean(axis=1)


def errors(numerical: np.ndarray, reference: np.ndarray, dx: float) -> dict[str, float]:
    delta = numerical - reference
    return {
        "l1_mean": float(np.mean(np.abs(delta))),
        "l1_integral": float(dx * np.sum(np.abs(delta))),
        "l2_mean": float(np.sqrt(np.mean(delta * delta))),
        "linf": float(np.max(np.abs(delta))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(TITLES), required=True)
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {
        key: load_result(args.main_dir / f"{key}.npz")
        for key in config.EULER_METHODS
        if (args.main_dir / f"{key}.npz").is_file()
    }
    missing = [key for key in config.EULER_METHODS if key not in results]
    if missing and not args.allow_missing:
        raise FileNotFoundError(f"missing N=200 results: {missing}")
    if missing:
        print(f"warning: excluding failed/missing methods from figure: {missing}", flush=True)
    reference = load_result(args.reference) if args.reference and args.reference.is_file() else None

    settings = {
        (
            result["metadata"]["benchmark"],
            int(result["metadata"]["nx"]),
            float(result["metadata"]["cfl"]),
            float(result["metadata"]["t_end"]),
        )
        for result in results.values()
    }
    if len(settings) != 1:
        raise ValueError(f"inconsistent result settings: {settings}")
    benchmark, nx, cfl, t_end = settings.pop()
    if benchmark != args.benchmark or nx != 200:
        raise ValueError(
            f"unexpected formal configuration: benchmark={benchmark}, nx={nx}, cfl={cfl}"
        )

    first_meta = next(iter(results.values()))["metadata"]
    x_min = float(first_meta["x_min"])
    x_max = float(first_meta["x_max"])
    dx = (x_max - x_min) / nx
    discontinuity = float(first_meta["discontinuity"])
    left = tuple(float(value) for value in first_meta["left_primitive"])
    right = tuple(float(value) for value in first_meta["right_primitive"])
    centers = next(iter(results.values()))["x"]
    exact_conserved = exact_riemann.cell_average_conserved(
        centers, dx, t_end, discontinuity, left, right,
    )
    exact_coarse = primitive(exact_conserved)
    exact_x = np.linspace(x_min, x_max, 10001, dtype=np.float64)
    exact_rho, exact_velocity, exact_pressure = exact_riemann.sample_points(
        exact_x, t_end, discontinuity, left, right,
    )
    exact_points = {
        "rho": exact_rho,
        "velocity": exact_velocity,
        "pressure": exact_pressure,
    }

    if reference is not None:
        ref_meta = reference["metadata"]
        if (
            ref_meta["benchmark"] != benchmark
            or abs(float(ref_meta["t_end"]) - t_end) > 1.0e-14
        ):
            raise ValueError(f"incompatible numerical reference metadata: {ref_meta}")
        ref_nx = int(ref_meta["nx"])
        ref_dx = (x_max - x_min) / ref_nx
        ref_exact = primitive(
            exact_riemann.cell_average_conserved(
                reference["x"], ref_dx, t_end, discontinuity, left, right,
            )
        )
        ref_values = primitive(reference["conserved"])
        print(
            f"numerical_reference_check N={ref_nx} "
            f"rho_L1_exact={errors(ref_values['rho'], ref_exact['rho'], ref_dx)['l1_mean']:.6e}",
            flush=True,
        )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 7.7,
        "lines.linewidth": 1.05,
    })
    fields = (
        ("rho", "Density $\\rho$"),
        ("velocity", "Velocity $u$"),
        ("pressure", "Pressure $p$"),
    )
    fig, axes = plt.subplots(
        3, 2, figsize=(11.6, 8.0), constrained_layout=True,
    )
    zoom_min, zoom_max = ZOOM_RANGES[benchmark]
    for row_index, (field, ylabel) in enumerate(fields):
        for column_index, ax in enumerate(axes[row_index]):
            ax.plot(
                exact_x,
                exact_points[field],
                color="black",
                lw=1.45,
                zorder=2,
                label="Exact Riemann solution",
            )
            for key in config.EULER_METHODS:
                if key not in results:
                    continue
                method = config.METHODS[key]
                values = primitive(results[key]["conserved"])[field]
                ax.plot(
                    results[key]["x"],
                    values,
                    color=method.color,
                    linestyle=method.linestyle,
                    marker=MARKERS[key],
                    markevery=3,
                    markersize=2.8,
                    markerfacecolor="white",
                    markeredgewidth=0.65,
                    label=method.label,
                    zorder=3,
                )
            ax.set_ylabel(ylabel if column_index == 0 else "")
            ax.grid(True, alpha=0.20, linewidth=0.55)
            ax.set_xlim((x_min, x_max) if column_index == 0 else (zoom_min, zoom_max))
            if row_index == 0:
                ax.set_title("Full domain" if column_index == 0 else "Contact/shock zoom")
            if row_index == 2:
                ax.set_xlabel("$x$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"{TITLES[benchmark]}: $N={nx}$, $t={t_end:g}$, CFL $={cfl:g}$; "
        "characteristic HLLC"
    )
    stem = args.figure_dir / f"{benchmark}_primitive_compare"
    save_figure(fig, stem)

    rows: list[dict[str, object]] = []
    profiles = {"x": centers}
    for field, _ in fields:
        profiles[f"exact_{field}"] = exact_coarse[field]
    for key in config.EULER_METHODS:
        if key not in results:
            continue
        values = primitive(results[key]["conserved"])
        row: dict[str, object] = {
            "benchmark": benchmark,
            "method": key,
            "label": config.METHODS[key].label,
            "nx": nx,
            "reference_method": "exact_riemann_cell_average",
            "cfl": cfl,
            "t_end": t_end,
        }
        for field, _ in fields:
            row.update({
                f"{field}_{name}": value
                for name, value in errors(values[field], exact_coarse[field], dx).items()
            })
            profiles[f"{key}_{field}"] = values[field]
        rows.append(row)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.table_dir / f"{benchmark}_errors_vs_exact.csv"
    fieldnames = list(rows[0])
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    profile_path = args.table_dir / f"{benchmark}_profiles_N{nx}.csv"
    profile_keys = list(profiles)
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(profile_keys)
        writer.writerows(zip(*(profiles[key] for key in profile_keys)))

    print(f"figure={stem.with_suffix('.png')}", flush=True)
    print(f"table={table_path}", flush=True)
    for row in rows:
        print(
            f"{row['method']}: rho_L1={row['rho_l1_mean']:.6e} "
            f"u_L1={row['velocity_l1_mean']:.6e} p_L1={row['pressure_l1_mean']:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
