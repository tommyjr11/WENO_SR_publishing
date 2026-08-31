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


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
REFERENCE = REPO / (
    "plots/WENO5_MLP/weno5_titarev_toro_reference_10001_rk3_classical/"
    "reference_10001_results.npz"
)
METHODS = (
    ("weno5_js", "WENO5-JS-RK3", "#666666", "--"),
    ("weno5_z_p2", "WENO5-Z-RK3", "#56B4E9", "-"),
    ("weno5_sr_f64", "WENO5-SR-RK3", "#0072B2", "-"),
    ("weno5_sr_f32", "WENO5-SR-FP32-RK3", "#009E73", "-."),
    ("weno7_js", "WENO7-JS-RK4", "#A65628", "--"),
    ("weno7_z_p3", "WENO7-Z-RK4", "#E69F00", "-"),
    ("weno7_sr_f64", "WENO7-SR-RK4", "#CC79A7", "-"),
)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        state = np.asarray(data["state"], dtype=np.float64)
        metadata = json.loads(str(data["metadata_json"].item()))
    return x, state[..., 0].mean(axis=0), metadata


def errors(numerical: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = numerical - reference
    return {
        "rho_l1": float(np.mean(np.abs(delta))),
        "rho_l2": float(np.sqrt(np.mean(delta * delta))),
        "rho_linf": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, choices=(1001, 2000), required=True)
    parser.add_argument("--ny", type=int, default=10)
    args = parser.parse_args()
    old = REPO / f"for_paper_results/raw/titarev_toro_cfl08/N{args.nx}x{args.ny}"
    new = ROOT / f"raw/titarev_toro_cfl08/N{args.nx}x{args.ny}"
    records = {}
    for key, *_ in METHODS:
        path = (new if "_z_" in key else old) / f"{key}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        records[key] = load(path)
    with np.load(REFERENCE) as data:
        ref_x = np.asarray(data["x"], dtype=np.float64)
        ref_rho = np.asarray(data["rho"], dtype=np.float64)

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
    })
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.3), constrained_layout=True)
    for axis in axes:
        axis.plot(ref_x, ref_rho, color="black", lw=1.2,
                  label="WENO5-JS reference, $N=10001$")
    rows = []
    for key, label, color, linestyle in METHODS:
        x, rho, metadata = records[key]
        for axis in axes:
            axis.plot(x, rho, color=color, ls=linestyle, lw=0.95, label=label)
        ref_on_grid = np.interp(x, ref_x, ref_rho)
        rows.append({
            "method": key,
            "label": label,
            "nx": args.nx,
            "ny": args.ny,
            "cfl": metadata["cfl"],
            "t_end": metadata["t_end"],
            "reference": "WENO5-JS N=10001 pointwise interpolation",
            **errors(rho, ref_on_grid),
        })
    axes[0].set_xlim(-5.0, 5.0)
    axes[0].set_title("Full domain")
    axes[1].set_xlim(-1.0, 2.5)
    axes[1].set_title("Resolved entropy-wave train")
    for axis in axes:
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"Density $\rho$")
        axis.grid(alpha=0.2, lw=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, fontsize=7.2)
    fig.suptitle(
        rf"Titarev--Toro: $N={args.nx}$, $t=5$, "
        r"CFL $=0.8$, characteristic HLLC"
    )
    out = ROOT / f"figures/titarev_toro_cfl08/N{args.nx}x{args.ny}"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "titarev_density_with_weno_z_rminus1"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    table = ROOT / f"tables/titarev_toro_cfl08/N{args.nx}x{args.ny}_density_errors.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(stem.with_suffix(".png"))
    print(table)


if __name__ == "__main__":
    main()
