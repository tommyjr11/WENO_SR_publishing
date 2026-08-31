#!/usr/bin/env python3
"""Run WENO7 point-value Shu-RK4 on the 2D quadrant Riemann problem."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from weno7_point_rk4_shu import point_rk4 as solver


def quadrant_primitive(x: np.ndarray, y: np.ndarray, quadrant_case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return rho, vx, vy, p


def make_quadrant_state(params: solver.Params, quadrant_case: str, quadrature: int) -> np.ndarray:
    if quadrature < 1:
        raise ValueError("--init-quadrature must be at least 1")
    xi, wi = np.polynomial.legendre.leggauss(quadrature)
    g = params.ghost
    jj, ii = np.indices((params.ny + 2 * g, params.nx + 2 * g))
    xc = params.x_min + (ii - g + 0.5) * params.dx
    yc = params.y_min + (jj - g + 0.5) * params.dy
    u = np.zeros(params.padded_shape, dtype=np.float64)
    for sx, wx in zip(xi, wi):
        x = xc + 0.5 * params.dx * float(sx)
        for sy, wy in zip(xi, wi):
            y = yc + 0.5 * params.dy * float(sy)
            rho, vx, vy, p = quadrant_primitive(x, y, quadrant_case)
            u += 0.25 * float(wx) * float(wy) * solver.conserved_from_primitive(rho, vx, vy, p, params.gamma)
    return u


def primitive_interior(u: np.ndarray, params: solver.Params) -> np.ndarray:
    g = params.ghost
    return solver.primitive_from_conserved(u[g : g + params.ny, g : g + params.nx, :], params.gamma)


def plot_qstyle(u: np.ndarray, params: solver.Params, out_path: Path, title: str) -> None:
    pri = primitive_interior(u, params)
    rho = pri[..., 0]
    vx = pri[..., 1]
    vy = pri[..., 2]
    p = pri[..., 3]
    x = np.linspace(params.x_min + 0.5 * params.dx, params.x_max - 0.5 * params.dx, params.nx)
    y = np.linspace(params.y_min + 0.5 * params.dy, params.y_max - 0.5 * params.dy, params.ny)
    x_grid, y_grid = np.meshgrid(x, y)
    rho_levels = np.arange(0.16, 1.71, 0.05)
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
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim([params.x_min, params.x_max])
    ax.set_ylim([params.y_min, params.y_max])
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(c, ax=ax, label="Pressure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, header: dict[str, object], summary: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key, value in header.items():
            f.write(f"{key}: {value}\n")
        f.write("\n[result]\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def run(args: argparse.Namespace) -> None:
    params = solver.Params(
        nx=args.nx,
        ny=args.ny,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        cfl=args.cfl,
        t_end=args.t_end,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial = make_quadrant_state(params, args.quadrant_case, args.init_quadrature)
    print(
        f"run_start nx={args.nx} ny={args.ny} t_end={args.t_end} cfl={args.cfl} "
        f"case={args.quadrant_case} solver={args.riemann_solver} boundary={args.boundary} "
        f"init_quadrature={args.init_quadrature} device={args.device}",
        flush=True,
    )
    final, summary = solver.run_from_initial(
        initial,
        params,
        device=args.device,
        riemann_solver=args.riemann_solver,
        characteristic=args.weno_space == "characteristic",
        report_interval=args.report_interval,
        boundary=args.boundary,
    )
    np.savez(args.out_dir / "quadrant_point_rk4_results.npz", initial=initial, final=final, **summary)
    plot_qstyle(
        final,
        params,
        args.out_dir / "point_rk4_pressure_rho_quiver_rho016_171_step005.png",
        f"WENO7 point-RK4 {args.quadrant_case} {args.riemann_solver} {args.nx}x{args.ny} t={summary['t']:.3f}",
    )
    write_summary(
        args.out_dir / "summary.txt",
        {
            "method": "WENO7 point-value Shu-RK4",
            "quadrant_case": args.quadrant_case,
            "riemann_solver": args.riemann_solver,
            "boundary": args.boundary,
            "nx": args.nx,
            "ny": args.ny,
            "cfl": args.cfl,
            "t_end": args.t_end,
            "init_quadrature": args.init_quadrature,
            "weno_space": args.weno_space,
            "device": args.device,
        },
        summary,
    )
    print(f"summary={args.out_dir / 'summary.txt'}", flush=True)
    print(f"plot={args.out_dir / 'point_rk4_pressure_rho_quiver_rho016_171_step005.png'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.5)
    parser.add_argument("--quadrant-case", choices=("case12", "case6"), default="case12")
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--riemann-solver", choices=("evilin", "hllc"), default="evilin")
    parser.add_argument("--weno-space", choices=("characteristic", "conserved"), default="characteristic")
    parser.add_argument("--boundary", choices=("outflow", "periodic"), default="outflow")
    parser.add_argument("--report-interval", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

