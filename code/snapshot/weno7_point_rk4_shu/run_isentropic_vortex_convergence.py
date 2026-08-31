#!/usr/bin/env python3
"""Run WENO7 point-value Shu-RK4 convergence on the 2D isentropic vortex."""

from __future__ import annotations

import argparse
import csv
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


def parse_grids(text: str) -> list[int]:
    grids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not grids:
        raise argparse.ArgumentTypeError("--grids must contain at least one integer")
    if any(n <= 0 for n in grids):
        raise argparse.ArgumentTypeError("grid sizes must be positive")
    return grids


def interior(u: np.ndarray, params: solver.Params) -> np.ndarray:
    g = params.ghost
    return u[g : g + params.ny, g : g + params.nx, :]


def error_metrics(num: np.ndarray, exact: np.ndarray, params: solver.Params) -> dict[str, float]:
    diff = interior(num, params) - interior(exact, params)
    area = params.dx * params.dy
    rho_diff = diff[..., 0]
    cons_l2 = np.sqrt(np.sum(diff * diff) * area)
    return {
        "rho_l1": float(np.mean(np.abs(rho_diff))),
        "rho_l2": float(np.sqrt(np.mean(rho_diff * rho_diff))),
        "rho_linf": float(np.max(np.abs(rho_diff))),
        "cons_l1": float(np.mean(np.sum(np.abs(diff), axis=-1))),
        "cons_l2": float(cons_l2),
        "cons_linf": float(np.max(np.linalg.norm(diff, axis=-1))),
    }


def add_orders(rows: list[dict[str, float]]) -> None:
    keys = ["rho_l1", "rho_l2", "rho_linf", "cons_l1", "cons_l2", "cons_linf"]
    for idx, row in enumerate(rows):
        if idx == 0:
            for key in keys:
                row[f"{key}_order"] = float("nan")
            continue
        prev = rows[idx - 1]
        h_prev = float(prev["dx"])
        h = float(row["dx"])
        for key in keys:
            e_prev = float(prev[key])
            e = float(row[key])
            if e > 0.0 and e_prev > 0.0 and h > 0.0 and h_prev > 0.0:
                row[f"{key}_order"] = float(np.log(e_prev / e) / np.log(h_prev / h))
            else:
                row[f"{key}_order"] = float("nan")


def primitive_interior(u: np.ndarray, params: solver.Params) -> np.ndarray:
    return solver.primitive_from_conserved(interior(u, params), params.gamma)


def plot_density(u: np.ndarray, params: solver.Params, out_path: Path, title: str) -> None:
    pri = primitive_interior(u, params)
    rho = pri[..., 0]
    x = np.linspace(params.x_min + 0.5 * params.dx, params.x_max - 0.5 * params.dx, params.nx)
    y = np.linspace(params.y_min + 0.5 * params.dy, params.y_max - 0.5 * params.dy, params.ny)
    xg, yg = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(6.0, 5.4), constrained_layout=True)
    im = ax.contourf(xg, yg, rho, levels=80, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, label=r"$\rho$")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_rho_error(num: np.ndarray, exact: np.ndarray, params: solver.Params, out_path: Path) -> None:
    diff = primitive_interior(num, params)[..., 0] - primitive_interior(exact, params)[..., 0]
    x = np.linspace(params.x_min + 0.5 * params.dx, params.x_max - 0.5 * params.dx, params.nx)
    y = np.linspace(params.y_min + 0.5 * params.dy, params.y_max - 0.5 * params.dy, params.ny)
    xg, yg = np.meshgrid(x, y)
    scale = float(np.nanmax(np.abs(diff)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    fig, ax = plt.subplots(figsize=(6.0, 5.4), constrained_layout=True)
    im = ax.contourf(xg, yg, diff, levels=101, cmap="coolwarm", vmin=-scale, vmax=scale)
    ax.set_title(r"$\rho-\rho_{\rm exact}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax, label="density error")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_convergence(rows: list[dict[str, float]], out_path: Path) -> None:
    if len(rows) < 2:
        return
    h = np.array([float(row["dx"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.2, 4.7), constrained_layout=True)
    for key, label, marker in [
        ("rho_l1", r"$L_1(\rho)$", "o"),
        ("rho_l2", r"$L_2(\rho)$", "s"),
        ("rho_linf", r"$L_\infty(\rho)$", "^"),
    ]:
        err = np.array([float(row[key]) for row in rows], dtype=np.float64)
        ax.loglog(h, err, marker=marker, linewidth=1.8, label=label)
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel("error")
    ax.set_title("WENO7 point-value Shu-RK4 convergence")
    ax.legend()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def write_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    fieldnames = [
        "N",
        "dx",
        "steps",
        "t",
        "mass",
        "mass_error",
        "rho_min",
        "rho_max",
        "p_min",
        "p_max",
        "nan_count",
        "rho_neg",
        "p_neg",
        "rho_l1",
        "rho_l1_order",
        "rho_l2",
        "rho_l2_order",
        "rho_linf",
        "rho_linf_order",
        "cons_l1",
        "cons_l1_order",
        "cons_l2",
        "cons_l2_order",
        "cons_linf",
        "cons_linf_order",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(rows: list[dict[str, float]], out_path: Path, args: argparse.Namespace) -> None:
    lines = [
        "WENO7 point-value Shu-RK4 isentropic vortex convergence",
        f"grids={','.join(str(int(row['N'])) for row in rows)}",
        f"domain=[{args.x_min},{args.x_max}]x[{args.y_min},{args.y_max}] t_end={args.t_end} cfl={args.cfl}",
        f"init_quadrature={args.init_quadrature} exact_quadrature={args.exact_quadrature}",
        f"riemann_solver={args.riemann_solver} weno_space={args.weno_space} boundary={args.boundary} device={args.device}",
        "",
    ]
    for row in rows:
        lines.append(
            "N={N:d} steps={steps:d} rho_l2={rho_l2:.8e} order={rho_l2_order:.4f} "
            "rho_linf={rho_linf:.8e} mass_error={mass_error:.8e} nan={nan_count:.0f}".format(
                N=int(row["N"]),
                steps=int(row["steps"]),
                rho_l2=float(row["rho_l2"]),
                rho_l2_order=float(row["rho_l2_order"]),
                rho_linf=float(row["rho_linf"]),
                mass_error=float(row["mass_error"]),
                nan_count=float(row["nan_count"]),
            )
        )
    out_path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    characteristic = args.weno_space == "characteristic"
    rows: list[dict[str, float]] = []
    for n in args.grids:
        params = solver.Params(
            nx=n,
            ny=n,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            cfl=args.cfl,
            t_end=args.t_end,
        )
        print(f"run_grid_start N={n} dx={params.dx:.8e}", flush=True)
        initial_exact = solver.cell_average_state(params, 0.0, args.exact_quadrature)
        final, summary = solver.run_case(
            params,
            device=args.device,
            init_quadrature=args.init_quadrature,
            riemann_solver=args.riemann_solver,
            characteristic=characteristic,
            report_interval=args.report_interval,
        )
        exact = solver.cell_average_state(params, args.t_end, args.exact_quadrature)
        metrics = error_metrics(final, exact, params)
        mass0 = solver.interior_stats(initial_exact, params)["mass"]
        row: dict[str, float] = {
            "N": float(n),
            "dx": float(params.dx),
            **{key: float(value) for key, value in summary.items() if isinstance(value, (int, float))},
            **metrics,
        }
        row["mass_error"] = float(row["mass"] - mass0)
        rows.append(row)
        print(
            f"run_grid_done N={n} steps={int(row['steps'])} rho_l2={row['rho_l2']:.8e} "
            f"rho_linf={row['rho_linf']:.8e} mass_error={row['mass_error']:.8e}",
            flush=True,
        )
        if args.save_fields:
            np.save(out_dir / f"solution_N{n}.npy", final)
            np.save(out_dir / f"exact_N{n}.npy", exact)
        plot_density(final, params, out_dir / f"density_N{n}.png", f"WENO7 point-RK4 density, N={n}")
        plot_rho_error(final, exact, params, out_dir / f"rho_error_N{n}.png")

    add_orders(rows)
    write_csv(rows, out_dir / "convergence.csv")
    write_summary(rows, out_dir / "summary.txt", args)
    plot_convergence(rows, out_dir / "convergence_order.png")
    print(f"summary={out_dir / 'summary.txt'}", flush=True)
    print(f"csv={out_dir / 'convergence.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=parse_grids, default=[25, 50, 100, 200])
    parser.add_argument("--x-min", type=float, default=-10.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-min", type=float, default=-10.0)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--exact-quadrature", type=int, default=15)
    parser.add_argument("--riemann-solver", choices=("evilin", "hllc"), default="evilin")
    parser.add_argument("--weno-space", choices=("characteristic", "conserved"), default="characteristic")
    parser.add_argument("--boundary", choices=("periodic",), default="periodic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--report-interval", type=int, default=0)
    parser.add_argument("--save-fields", action="store_true")
    parser.add_argument("--out-dir", default="plots/WENO7_point_RK4_shu/isentropic_vortex_convergence")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

