#!/usr/bin/env python3
"""Run V20 WENO5-SR on the standard double-Mach-reflection problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import warp_weno5_helpers as wh
from run_double_mach_compare import make_initial_state
from teacherfree_lab_weno5_v12_reflection_sym.warp_v12.run_weno5_circle_mlp_compare_v12 import (
    load_mlp_params,
)
from teacherfree_lab_weno5_v20_distance_balanced import weno5_hllc_refsym


REFLECTION_FORMULA = "0.5*(M(x)+P*M(P*x))"


def primitive_fields(
    state: np.ndarray,
    params: wh.Params,
) -> tuple[np.ndarray, np.ndarray]:
    gc = params.ghost
    conserved = state[
        gc : gc + params.ny,
        gc : gc + params.nx,
        :,
    ]
    primitive = wh.primitive_from_conserved(conserved, params.gamma)
    return primitive[..., 0], primitive[..., 3]


def plot_density(
    density: np.ndarray,
    params: wh.Params,
    out_path: Path,
    title: str,
) -> None:
    x = (np.arange(params.nx, dtype=np.float64) + 0.5) * params.dx
    y = (np.arange(params.ny, dtype=np.float64) + 0.5) * params.dy
    xx, yy = np.meshgrid(x, y)
    rho_min = float(np.min(density))
    rho_max = float(np.max(density))
    levels = np.linspace(rho_min, rho_max, 120)
    contour_levels = np.linspace(1.4, 22.9, 43)

    fig, ax = plt.subplots(figsize=(13.5, 3.8), constrained_layout=True)
    image = ax.contourf(
        xx,
        yy,
        density,
        levels=levels,
        cmap="turbo",
        extend="both",
    )
    ax.contour(
        xx,
        yy,
        density,
        levels=contour_levels,
        colors="k",
        linewidths=0.18,
        alpha=0.55,
    )
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(0.0, params.x_length)
    ax.set_ylim(0.0, params.y_length)
    ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(image, ax=ax, pad=0.012, shrink=0.92)
    colorbar.set_label(r"Density $\rho$")
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def report(
    step: int,
    t: float,
    dt: float,
    stats: dict[str, float],
) -> None:
    print(
        f"step={step:05d} t={t:.8e} dt={dt:.4e} "
        f"rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
        f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] "
        f"nan={int(stats['nan_count'])} "
        f"rho_neg={int(stats['rho_neg'])} p_neg={int(stats['p_neg'])}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    if args.init_quadrature != 15:
        raise ValueError("double-Mach validation requires 15-point initialization")
    wh.require_warp()
    wh.wp.init()
    wh.wp.set_device(args.device)

    params = wh.Params(
        nx=args.nx,
        ny=args.ny,
        x_length=4.0,
        y_length=1.0,
        cfl=args.cfl,
        t_end=args.t_end,
    )
    initial = make_initial_state(params, args.init_quadrature)
    mlp_params = load_mlp_params(args.model, args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"double_mach_start model={args.model} nx={args.nx} ny={args.ny} "
        f"cfl={args.cfl} t_end={args.t_end} "
        f"init_quadrature={args.init_quadrature} "
        "weno_space=characteristic boundary=double-mach "
        f"riemann_solver=hllc eno_cutoff={args.eno_cutoff} "
        f"mlp_forward={REFLECTION_FORMULA}",
        flush=True,
    )
    state, dt_values, steps, t = weno5_hllc_refsym.run_to_time(
        initial,
        params,
        args.t_end,
        args.device,
        mlp_params,
        "double-mach",
        eno_cutoff=args.eno_cutoff,
        report_interval=args.report_interval,
        report=report,
    )
    density, pressure = primitive_fields(state, params)
    health = wh.interior_stats(state, params)
    summary: dict[str, object] = {
        "model": str(args.model),
        "test": "double-mach-reflection",
        "riemann_solver": "hllc_direct_minmax_tiny1e-16",
        "weno_space": "characteristic",
        "boundary": "double-mach",
        "eno_cutoff": args.eno_cutoff,
        "mlp_forward": REFLECTION_FORMULA,
        "nx": args.nx,
        "ny": args.ny,
        "x_length": 4.0,
        "y_length": 1.0,
        "cfl": args.cfl,
        "init_quadrature": args.init_quadrature,
        "t": t,
        "t_end": args.t_end,
        "complete_time": abs(t - args.t_end) < 1.0e-12,
        "steps": steps,
        "dt_min": float(np.min(dt_values)) if dt_values else 0.0,
        "dt_max": float(np.max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
        **health,
    }

    np.savez(
        args.out_dir / "mlp_double_mach_results.npz",
        initial=initial,
        mlp=state,
        dt_values=np.asarray(dt_values),
        metadata_json=np.array(json.dumps(summary, sort_keys=True)),
    )
    with (args.out_dir / "summary.txt").open("w", encoding="utf-8") as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
    plot_density(
        density,
        params,
        args.out_dir / "mlp_double_mach_density.png",
        (
            "WENO5-SR V20 @ 12250, double-Mach reflection, "
            f"HLLC, {args.nx}x{args.ny}, t={t:.3f}"
        ),
    )
    print("summary:", flush=True)
    for key, value in summary.items():
        print(f"  {key}: {value}", flush=True)
    print(
        f"plot={args.out_dir / 'mlp_double_mach_density.png'}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "teacherfree_lab_weno5_v20_distance_balanced/runs/"
            "apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/"
            "model_step_012250.npz"
        ),
    )
    parser.add_argument("--nx", type=int, default=1200)
    parser.add_argument("--ny", type=int, default=300)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument(
        "--eno-cutoff",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "plots/WENO5_MLP/teacherfree_weno5_v20_refsym/double_mach/"
            "apost_weno5_v20_distance_balanced_model_step_012250_"
            "double_mach_1200x300_t02_hllc_cfl04"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
