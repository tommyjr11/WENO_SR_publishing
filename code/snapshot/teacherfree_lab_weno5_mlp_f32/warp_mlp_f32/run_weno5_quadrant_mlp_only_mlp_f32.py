#!/usr/bin/env python3
"""Run only the MLP WENO5 solver on the 2D quadrant Riemann problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32 import warp_weno5_helpers_mlp_f32 as wh
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_circle_mlp_compare_mlp_f32 import load_mlp_params
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_quadrant_mlp_compare_mlp_f32 import density_field, pressure_field
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.weno5_rk3_forward_mlp_f32 import run_forward_to_time


wp = wh.wp


def make_quadrant_state_fast(params: wh.Params, quadrant_case: str = "case12") -> np.ndarray:
    g = params.ghost
    jj, ii = np.indices((params.ny + 2 * g, params.nx + 2 * g))
    x = (ii - g + 0.5) * params.dx
    y = (jj - g + 0.5) * params.dy

    rho = np.empty_like(x, dtype=np.float64)
    vx = np.empty_like(x, dtype=np.float64)
    vy = np.empty_like(x, dtype=np.float64)
    p = np.empty_like(x, dtype=np.float64)

    lower_left = (x < 0.5) & (y < 0.5)
    upper_left = (x < 0.5) & (y >= 0.5)
    lower_right = (x >= 0.5) & (y < 0.5)
    upper_right = (x >= 0.5) & (y >= 0.5)

    if quadrant_case == "case6":
        rho[lower_left], vx[lower_left], vy[lower_left], p[lower_left] = 0.8, 0.0, 0.0, 1.0
        rho[upper_left], vx[upper_left], vy[upper_left], p[upper_left] = 1.0, 0.7276, 0.0, 1.0
        rho[lower_right], vx[lower_right], vy[lower_right], p[lower_right] = 1.0, 0.0, 0.7276, 1.0
        rho[upper_right], vx[upper_right], vy[upper_right], p[upper_right] = 0.5315, 0.0, 0.0, 0.4
    else:
        rho[lower_left], vx[lower_left], vy[lower_left], p[lower_left] = 0.138, 1.206, 1.206, 0.029
        rho[upper_left], vx[upper_left], vy[upper_left], p[upper_left] = 0.5323, 1.206, 0.0, 0.3
        rho[lower_right], vx[lower_right], vy[lower_right], p[lower_right] = 0.5323, 0.0, 1.206, 0.3
        rho[upper_right], vx[upper_right], vy[upper_right], p[upper_right] = 1.5, 0.0, 0.0, 1.5

    u = np.empty((params.ny + 2 * g, params.nx + 2 * g, 4), dtype=np.float64)
    u[..., 0] = rho
    u[..., 1] = rho * vx
    u[..., 2] = rho * vy
    u[..., 3] = p / (params.gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return u


def density_levels_for_case(quadrant_case: str) -> tuple[np.ndarray, str]:
    if quadrant_case == "case6":
        return np.arange(0.54, 1.70 + 1.0e-12, 0.04), "rho054_170_step004"
    return np.arange(0.16, 1.71 + 1.0e-12, 0.05), "rho016_171_step005"


def plot_qstyle(u: np.ndarray, params: wh.Params, out_path: Path, title: str, quadrant_case: str) -> None:
    rho = density_field(u, params)
    p = pressure_field(u, params)
    pri = wh.primitive_from_conserved(
        u[params.ghost : params.ghost + params.ny, params.ghost : params.ghost + params.nx, :],
        params.gamma,
    )
    vx = pri[..., 1]
    vy = pri[..., 2]

    x = np.linspace(0.0, 1.0, params.nx)
    y = np.linspace(0.0, 1.0, params.ny)
    x_grid, y_grid = np.meshgrid(x, y)
    rho_levels, _ = density_levels_for_case(quadrant_case)
    skip = max(1, params.nx // 30)

    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.contourf(x_grid, y_grid, p, levels=300, cmap="jet")
    ax.contour(x_grid, y_grid, rho, levels=rho_levels, colors="k", linewidths=0.3)
    ax.quiver(
        x_grid[::skip, ::skip],
        y_grid[::skip, ::skip],
        vx[::skip, ::skip],
        vy[::skip, ::skip],
        color="white",
        scale=40,
        width=0.002,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(c, ax=ax, label="Pressure")
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp.init()
    wp.set_device(args.device)

    params = wh.Params(nx=args.nx, ny=args.ny, x_length=1.0, y_length=1.0, cfl=args.cfl, t_end=args.t_end)
    mlp_params = load_mlp_params(args.model, args.device)
    initial = make_quadrant_state_fast(params, args.quadrant_case)
    print(
        f"run_start model={args.model} solver={args.riemann_solver} nx={params.nx} ny={params.ny} "
        f"cfl={params.cfl} t_end={params.t_end} quadrant_case={args.quadrant_case}",
        flush=True,
    )

    def report_step(step: int, t: float, dt: float, stats: dict[str, float]) -> None:
        print(
            f"step={step:04d} t={t:.8e} dt={dt:.8e} "
            f"rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
            f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] nan={int(stats['nan_count'])}",
            flush=True,
        )
        if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
            print("failure: NaN/negative rho/p detected, stopping early", flush=True)

    u, dt_values, steps, t = run_forward_to_time(
        initial,
        params,
        args.t_end,
        args.device,
        args.weno_space == "characteristic",
        mlp_params,
        args.eno_cutoff,
        "transmissive",
        args.riemann_solver,
        args.report_interval,
        report_step,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rho = density_field(u, params)
    p = pressure_field(u, params)
    summary = {
        "model": str(args.model),
        "quadrant_case": args.quadrant_case,
        "riemann_solver": args.riemann_solver,
        "nx": params.nx,
        "ny": params.ny,
        "cfl": params.cfl,
        "t": t,
        "steps": steps,
        "dt_min": float(np.min(dt_values)) if dt_values else 0.0,
        "dt_max": float(np.max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "p_min": float(np.min(p)),
        "p_max": float(np.max(p)),
        "nan_count": int(np.count_nonzero(~np.isfinite(u))),
    }
    with (args.out_dir / "summary.txt").open("w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    np.savez(args.out_dir / "mlp_quadrant_results.npz", initial=initial, mlp=u, dt_values=np.array(dt_values), **summary)

    _, rho_tag = density_levels_for_case(args.quadrant_case)
    plot_path = args.out_dir / f"mlp_pressure_rho_quiver_{rho_tag}.png"
    plot_qstyle(
        u,
        params,
        plot_path,
        f"MLP {args.quadrant_case} {args.riemann_solver} {params.nx}x{params.ny} t={t:.3f}",
        args.quadrant_case,
    )
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"plot={plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weno-space", choices=("characteristic", "conserved"), default="characteristic")
    parser.add_argument("--riemann-solver", choices=("force", "evilin"), default="evilin")
    parser.add_argument("--quadrant-case", choices=("case12", "case6"), default="case12")
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
