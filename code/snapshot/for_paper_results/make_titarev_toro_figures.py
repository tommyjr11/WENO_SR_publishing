#!/usr/bin/env python3
"""Plot formal WENO5/WENO7 Titarev--Toro density comparisons."""

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


METHODS = config.EULER_METHODS
DEFAULT_REFERENCE = config.ROOT / (
    "plots/WENO5_MLP/weno5_titarev_toro_reference_10001_rk3_classical/"
    "reference_10001_results.npz"
)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_method(path: Path) -> dict[str, object]:
    with np.load(path) as data:
        return {
            "state": np.asarray(data["state"], dtype=np.float64),
            "x": np.asarray(data["x"], dtype=np.float64),
            "metadata": json.loads(str(data["metadata_json"].item())),
        }


def metrics(numerical: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = numerical - reference
    return {
        "l1": float(np.mean(np.abs(delta))),
        "l2": float(np.sqrt(np.mean(delta * delta))),
        "linf": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-dir", type=Path,
        default=config.RAW / "titarev_toro_cfl08" / "N1001x10",
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--figure-dir", type=Path,
        default=config.FIGURES / "titarev_toro_cfl08",
    )
    parser.add_argument("--table-dir", type=Path, default=config.TABLES)
    args = parser.parse_args()
    results = {
        key: load_method(args.main_dir / f"{key}.npz")
        for key in METHODS
        if (args.main_dir / f"{key}.npz").is_file()
    }
    if not results:
        raise FileNotFoundError(f"no completed Titarev--Toro results in {args.main_dir}")

    settings = {
        (
            int(result["metadata"]["nx"]),
            int(result["metadata"]["ny"]),
            float(result["metadata"]["cfl"]),
            float(result["metadata"]["t_end"]),
        )
        for result in results.values()
    }
    if len(settings) != 1:
        raise ValueError(f"inconsistent Titarev--Toro result settings: {settings}")
    nx, ny, cfl, t_end = settings.pop()
    if ny < 10:
        raise ValueError(f"Titarev--Toro plotting requires ny >= 10; got {ny}")

    reference = None
    if args.reference.is_file():
        with np.load(args.reference) as data:
            reference = {
                "x": np.asarray(data["x"], dtype=np.float64),
                "rho": np.asarray(data["rho"], dtype=np.float64),
            }

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.1), constrained_layout=True)
    if reference is not None:
        for ax in axes:
            ax.plot(
                reference["x"], reference["rho"], color="black", lw=1.15,
                label="WENO5-JS reference, $N=10001$",
            )
    rows: list[dict[str, object]] = []
    profile_columns: dict[str, np.ndarray] = {}
    common_x = None
    same_grid_js = None
    if "weno5_js" in results:
        same_grid_js = results["weno5_js"]["state"][..., 0].mean(axis=0)

    for key in METHODS:
        if key not in results:
            continue
        result = results[key]
        x = result["x"]
        rho = result["state"][..., 0].mean(axis=0)
        common_x = x
        profile_columns[key] = rho
        method = config.METHODS[key]
        for ax in axes:
            ax.plot(
                x, rho, color=method.color, linestyle=method.linestyle,
                lw=1.0, label=method.label,
            )
        row: dict[str, object] = {"method": key}
        if reference is not None:
            ref_on_grid = np.interp(x, reference["x"], reference["rho"])
            row.update({f"vs_reference_{name}": value for name, value in metrics(rho, ref_on_grid).items()})
        if same_grid_js is not None:
            row.update({f"vs_weno5_js_{name}": value for name, value in metrics(rho, same_grid_js).items()})
        rows.append(row)

    axes[0].set_xlim(-5.0, 5.0)
    axes[0].set_title("Full domain")
    axes[1].set_xlim(-1.0, 2.5)
    axes[1].set_title("Resolved entropy-wave train")
    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("Density $\\rho$")
        ax.grid(True, alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=3, frameon=False,
    )
    fig.suptitle(
        f"Titarev--Toro: ${nx}\\times{ny}$, $t={t_end:g}$, "
        f"CFL $={cfl:g}$, characteristic HLLC, method-matched SSP-RK"
    )
    save_figure(fig, args.figure_dir / "titarev_density_compare")

    table_path = args.table_dir / "titarev_toro_cfl08_density_errors.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({name for row in rows for name in row}, key=lambda name: (name != "method", name))
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if common_x is not None:
        profile_path = args.table_dir / "titarev_toro_cfl08_profiles.csv"
        with profile_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            keys = [key for key in METHODS if key in profile_columns]
            writer.writerow(["x", *keys])
            writer.writerows(zip(common_x, *(profile_columns[key] for key in keys)))
    print(f"titarev_figures={args.figure_dir}", flush=True)
    print(f"titarev_table={table_path}", flush=True)


if __name__ == "__main__":
    main()
