#!/usr/bin/env python3
"""Compare classical and MLP WENO5/RK3 on the 2D quadrant Riemann case."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import warp_weno5_helpers as wh
from run_weno5_circle_mlp_compare import advance_one_step, load_mlp_params


wp = wh.wp


def quadrant_riemann_conserved(x: float, y: float, gamma: float) -> np.ndarray:
    if x < 0.5:
        if y < 0.5:
            return wh.primitive_to_conserved(0.138, 1.206, 1.206, 0.029, gamma)
        return wh.primitive_to_conserved(0.5323, 1.206, 0.0, 0.3, gamma)
    if y < 0.5:
        return wh.primitive_to_conserved(0.5323, 0.0, 1.206, 0.3, gamma)
    return wh.primitive_to_conserved(1.5, 0.0, 0.0, 1.5, gamma)


def cell_average_quadrant_riemann(xc: float, yc: float, dx: float, dy: float, gamma: float) -> np.ndarray:
    state = np.zeros(4, dtype=np.float64)
    for xi, wx in zip(wh.GAUSS15_XI, wh.GAUSS15_W):
        x = xc + 0.5 * dx * float(xi)
        for eta, wy in zip(wh.GAUSS15_XI, wh.GAUSS15_W):
            y = yc + 0.5 * dy * float(eta)
            state += float(wx) * float(wy) * quadrant_riemann_conserved(x, y, gamma)
    return 0.25 * state


def make_quadrant_state(params: wh.Params) -> np.ndarray:
    u = np.zeros(params.padded_shape, dtype=np.float64)
    for j in range(params.ny + 2 * params.ghost):
        y = (j - params.ghost + 0.5) * params.dy
        for i in range(params.nx + 2 * params.ghost):
            x = (i - params.ghost + 0.5) * params.dx
            u[j, i, :] = cell_average_quadrant_riemann(x, y, params.dx, params.dy, params.gamma)
    return u


def density_field(u: np.ndarray, params: wh.Params) -> np.ndarray:
    g = params.ghost
    return u[g : g + params.ny, g : g + params.nx, 0]


def pressure_field(u: np.ndarray, params: wh.Params) -> np.ndarray:
    g = params.ghost
    q = u[g : g + params.ny, g : g + params.nx, :]
    pri = wh.primitive_from_conserved(q, params.gamma)
    return pri[..., 3]


def mass(u: np.ndarray, params: wh.Params) -> float:
    return float(np.sum(density_field(u, params)) * params.dx * params.dy)


def plot_density(initial: np.ndarray, classical: np.ndarray, mlp: np.ndarray, params: wh.Params, t: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rho0 = density_field(initial, params)
    rhoc = density_field(classical, params)
    rhom = density_field(mlp, params)
    diff = rhom - rhoc
    extent = [0.0, params.x_length, 0.0, params.y_length]
    vmin = float(min(np.min(rho0), np.min(rhoc), np.min(rhom)))
    vmax = float(max(np.max(rho0), np.max(rhoc), np.max(rhom)))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for ax, data, title in zip(axes, (rho0, rhoc, rhom), ("initial", "classical WENO5-RK3", "MLP WENO5-RK3")):
        im = ax.imshow(data, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"Quadrant Riemann density, t={t:.6f}")
    fig.savefig(out_dir / "density_compare.png", dpi=180)
    plt.close(fig)

    lim = float(np.max(np.abs(diff)))
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    im = ax.imshow(diff, origin="lower", extent=extent, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="equal")
    ax.set_title("MLP - classical density")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_dir / "density_difference.png", dpi=180)
    plt.close(fig)

    x = (np.arange(params.nx) + 0.5) * params.dx
    y = (np.arange(params.ny) + 0.5) * params.dy
    mid_y = params.ny // 2
    mid_x = params.nx // 2
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    axes[0].plot(x, rhoc[mid_y, :], "k--", lw=1.6, label="classical")
    axes[0].plot(x, rhom[mid_y, :], "r-", lw=1.2, label="mlp")
    axes[0].set_title("density centerline y=0.5")
    axes[0].set_xlabel("x")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(y, rhoc[:, mid_x], "k--", lw=1.6, label="classical")
    axes[1].plot(y, rhom[:, mid_x], "r-", lw=1.2, label="mlp")
    axes[1].set_title("density centerline x=0.5")
    axes[1].set_xlabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(out_dir / "centerline_density.png", dpi=180)
    plt.close(fig)


def save_summary(initial: np.ndarray, classical: np.ndarray, mlp: np.ndarray, params: wh.Params, t: float, steps: int, dt_values: list[float], out_dir: Path) -> None:
    rhoc = density_field(classical, params)
    rhom = density_field(mlp, params)
    pc = pressure_field(classical, params)
    pm = pressure_field(mlp, params)
    diff = rhom - rhoc
    summary = {
        "t": t,
        "steps": steps,
        "dt_min": float(np.min(dt_values)) if dt_values else 0.0,
        "dt_max": float(np.max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
        "initial_mass": mass(initial, params),
        "classical_mass": mass(classical, params),
        "mlp_mass": mass(mlp, params),
        "rho_diff_l1": float(np.mean(np.abs(diff))),
        "rho_diff_l2": float(np.sqrt(np.mean(diff * diff))),
        "rho_diff_linf": float(np.max(np.abs(diff))),
        "classical_rho_min": float(np.min(rhoc)),
        "classical_rho_max": float(np.max(rhoc)),
        "classical_p_min": float(np.min(pc)),
        "classical_p_max": float(np.max(pc)),
        "mlp_rho_min": float(np.min(rhom)),
        "mlp_rho_max": float(np.max(rhom)),
        "mlp_p_min": float(np.min(pm)),
        "mlp_p_max": float(np.max(pm)),
        "mlp_nan_count": float(np.count_nonzero(~np.isfinite(mlp))),
        "classical_nan_count": float(np.count_nonzero(~np.isfinite(classical))),
    }
    with (out_dir / "summary.txt").open("w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    np.savez(out_dir / "quadrant_results.npz", initial=initial, classical=classical, mlp=mlp, dt_values=np.array(dt_values), **summary)
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def run(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp.init()
    wp.set_device(args.device)

    params = wh.Params(nx=args.nx, ny=args.ny, x_length=1.0, y_length=1.0, cfl=args.cfl, t_end=args.t_end)
    mlp_params = load_mlp_params(args.model, args.device)
    u0 = make_quadrant_state(params)
    u_classical = u0.copy()
    u_mlp = u0.copy()
    t = 0.0
    steps = 0
    dt_values: list[float] = []

    while t < args.t_end - 1.0e-14:
        dt_c = wh.compute_dt(u_classical, params)
        dt_m = wh.compute_dt(u_mlp, params)
        dt = min(dt_c, dt_m, args.t_end - t)
        characteristic_weno = args.weno_space == "characteristic"
        u_classical = advance_one_step(u_classical, params, dt, args.device, None, False, "transmissive", characteristic_weno, args.riemann_solver)
        u_mlp = advance_one_step(u_mlp, params, dt, args.device, mlp_params, args.eno_cutoff, "transmissive", characteristic_weno, args.riemann_solver)
        t += dt
        steps += 1
        dt_values.append(dt)
        if steps == 1 or steps % args.report_interval == 0 or t >= args.t_end - 1.0e-14:
            stats_c = wh.interior_stats(u_classical, params)
            stats_m = wh.interior_stats(u_mlp, params)
            print(
                f"step={steps:04d} t={t:.8e} dt={dt:.8e} "
                f"classical rho=[{stats_c['rho_min']:.5e},{stats_c['rho_max']:.5e}] p_min={stats_c['p_min']:.5e} "
                f"mlp rho=[{stats_m['rho_min']:.5e},{stats_m['rho_max']:.5e}] p_min={stats_m['p_min']:.5e}"
            )
            if stats_c["nan_count"] or stats_m["nan_count"] or stats_c["rho_neg"] or stats_m["rho_neg"] or stats_c["p_neg"] or stats_m["p_neg"]:
                print("failure: NaN/negative rho/p detected; saving partial result.")
                break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_density(u0, u_classical, u_mlp, params, t, args.out_dir)
    save_summary(u0, u_classical, u_mlp, params, t, steps, dt_values, args.out_dir)
    print(f"plots: {args.out_dir / 'density_compare.png'}")
    print(f"plots: {args.out_dir / 'density_difference.png'}")
    print(f"plots: {args.out_dir / 'centerline_density.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--weno-space", choices=("characteristic", "conserved"), default="characteristic")
    parser.add_argument("--riemann-solver", choices=("force", "evilin"), default="force")
    parser.add_argument("--report-interval", type=int, default=25)
    parser.add_argument("--model", type=Path, default=Path("plots/weno5_mlp_cpu_c025_smooth20/model_best_circle_l2.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/weno5_quadrant_256_t05_c025_smooth20_bestcircle"))
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
