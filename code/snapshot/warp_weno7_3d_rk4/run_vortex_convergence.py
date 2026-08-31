from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from warp_weno5_3d_rk3 import kernels as B

from .binary_io import write_step
from .config import ShockBubbleConfig
from .solver import Weno7Rk4Solver


PLANES = ("xy", "yz", "xz")


def periodic_delta(value: np.ndarray, length: float = 20.0) -> np.ndarray:
    return value - length * np.round(value / length)


def vortex_conserved(a: np.ndarray, b: np.ndarray, t: float, plane: str) -> np.ndarray:
    gamma = 1.4
    beta = 5.0
    da = periodic_delta(a - t)
    db = periodic_delta(b - t)
    radius2 = da * da + db * db
    delta_t = -((gamma - 1.0) * beta * beta) / (8.0 * gamma * np.pi * np.pi) * np.exp(1.0 - radius2)
    temperature = 1.0 + delta_t
    rho = np.power(temperature, 1.0 / (gamma - 1.0))
    pressure = np.power(temperature, gamma / (gamma - 1.0))
    factor = beta / (2.0 * np.pi) * np.exp(0.5 * (1.0 - radius2))
    velocity_a = 1.0 - factor * db
    velocity_b = 1.0 + factor * da

    u = np.zeros_like(rho)
    v = np.zeros_like(rho)
    w = np.zeros_like(rho)
    if plane == "xy":
        u = velocity_a
        v = velocity_b
    elif plane == "yz":
        v = velocity_a
        w = velocity_b
    elif plane == "xz":
        u = velocity_a
        w = velocity_b
    else:
        raise ValueError(f"unknown plane: {plane}")

    result = np.empty(rho.shape + (5,), dtype=np.float64)
    result[..., 0] = rho
    result[..., 1] = rho * u
    result[..., 2] = rho * v
    result[..., 3] = rho * w
    result[..., 4] = pressure / (gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w)
    return result


def grid_shape(plane: str, n: int, inactive: int) -> tuple[int, int, int]:
    if plane == "xy":
        return n, n, inactive
    if plane == "yz":
        return inactive, n, n
    if plane == "xz":
        return n, inactive, n
    raise ValueError(f"unknown plane: {plane}")


def cell_average_state(config: ShockBubbleConfig, plane: str, t: float, quadrature: int) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    nodes = nodes.astype(np.float64)
    weights = weights.astype(np.float64)

    if plane == "xy":
        a_center = config.x_start + (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx
        b_center = config.y_start + (np.arange(config.ny, dtype=np.float64) + 0.5) * config.dy
        a_center, b_center = np.meshgrid(a_center, b_center)
        da, db = config.dx, config.dy
    elif plane == "yz":
        a_center = config.y_start + (np.arange(config.ny, dtype=np.float64) + 0.5) * config.dy
        b_center = config.z_start + (np.arange(config.nz, dtype=np.float64) + 0.5) * config.dz
        a_center, b_center = np.meshgrid(a_center, b_center)
        da, db = config.dy, config.dz
    elif plane == "xz":
        a_center = config.x_start + (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx
        b_center = config.z_start + (np.arange(config.nz, dtype=np.float64) + 0.5) * config.dz
        a_center, b_center = np.meshgrid(a_center, b_center)
        da, db = config.dx, config.dz
    else:
        raise ValueError(f"unknown plane: {plane}")

    average2d = np.zeros(a_center.shape + (5,), dtype=np.float64)
    for node_a, weight_a in zip(nodes, weights, strict=True):
        a = a_center + 0.5 * da * float(node_a)
        for node_b, weight_b in zip(nodes, weights, strict=True):
            b = b_center + 0.5 * db * float(node_b)
            average2d += 0.25 * float(weight_a) * float(weight_b) * vortex_conserved(a, b, t, plane)

    if plane == "xy":
        return np.broadcast_to(average2d[None, :, :, :], (config.nz, config.ny, config.nx, 5)).copy()
    if plane == "yz":
        return np.broadcast_to(average2d[:, :, None, :], (config.nz, config.ny, config.nx, 5)).copy()
    return np.broadcast_to(average2d[:, None, :, :], (config.nz, config.ny, config.nx, 5)).copy()


def padded_periodic(interior: np.ndarray, ghost: int) -> np.ndarray:
    return np.pad(interior, ((ghost, ghost), (ghost, ghost), (ghost, ghost), (0, 0)), mode="wrap")


def primitive_from_conserved(state: np.ndarray) -> np.ndarray:
    rho = state[..., 0]
    u = state[..., 1] / rho
    v = state[..., 2] / rho
    w = state[..., 3] / rho
    pressure = 0.4 * (state[..., 4] - 0.5 * rho * (u * u + v * v + w * w))
    return np.stack((rho, u, v, w, pressure), axis=-1)


def write_trace(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="ascii") as stream:
        fields = ("step", "time_start", "raw_dt", "dt", "time_end", "max_speed")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    plane: str,
    n: int,
    inactive: int,
    device: str,
    cfl: float,
    t_end: float,
    quadrature: int,
    out_dir: Path,
    report_interval: int,
) -> dict[str, object]:
    nx, ny, nz = grid_shape(plane, n, inactive)
    config = replace(
        ShockBubbleConfig(),
        nx=nx,
        ny=ny,
        nz=nz,
        x_start=-10.0,
        x_end=10.0,
        y_start=-10.0,
        y_end=10.0,
        z_start=-10.0,
        z_end=10.0,
        cfl=cfl,
        t_end=t_end,
    )
    initial = cell_average_state(config, plane, 0.0, quadrature)
    exact = cell_average_state(config, plane, t_end, quadrature)
    solver = Weno7Rk4Solver(config, device=device, boundary="periodic")
    solver.q.assign(padded_periodic(initial, config.ghost))
    solver._launch(B.conserved_to_primitive_kernel, config.padded_shape[:3], [solver.q, solver.primitive])

    wall_start = time.perf_counter()
    while solver.time != t_end:
        if not solver.advance():
            break
        if report_interval > 0 and (solver.step == 1 or solver.step % report_interval == 0 or solver.time == t_end):
            print(
                f"plane={plane} N={n} step={solver.step} t={solver.time:.16e} "
                f"dt={solver.dt_trace[-1]['dt']:.6e}",
                flush=True,
            )
    wp.synchronize_device(solver.device)
    elapsed = time.perf_counter() - wall_start

    g = config.ghost
    numerical = solver.q.numpy()[g : g + nz, g : g + ny, g : g + nx, :]
    rho_error = numerical[..., 0] - exact[..., 0]
    rho_l1 = float(np.mean(np.abs(rho_error)))
    rho_l2 = float(np.sqrt(np.mean(rho_error * rho_error)))
    mass_initial = float(np.sum(initial[..., 0]) * config.dx * config.dy * config.dz)
    mass_final = float(np.sum(numerical[..., 0]) * config.dx * config.dy * config.dz)
    diagnostics = solver.diagnostics()
    complete = bool(
        solver.time == t_end
        and diagnostics["nan_count"] == 0
        and diagnostics["rho_min"] > 0.0
        and diagnostics["p_min"] > 0.0
    )

    case_dir = out_dir / plane / f"N{n}x{n}x{inactive}"
    case_dir.mkdir(parents=True, exist_ok=True)
    final_path = case_dir / f"step_{solver.step:04d}.bin"
    write_step(final_path, solver.time, solver.primitive_host())
    write_trace(case_dir / "dt_trace.csv", solver.dt_trace)
    np.savez_compressed(
        case_dir / "density_comparison.npz",
        numerical=numerical[..., 0],
        exact=exact[..., 0],
        error=rho_error,
    )

    row: dict[str, object] = {
        "plane": plane,
        "N": n,
        "inactive_cells": inactive,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx_active": 20.0 / n,
        "cfl": cfl,
        "t_end": t_end,
        "quadrature": quadrature,
        "boundary": "periodic_xyz",
        "time_integrator": "four-stage-fourth-order-downwind-tvd-rk",
        "spatial_method": "characteristic-weno7-js-dorder0",
        "riemann_solver": "evilin",
        "steps": solver.step,
        "time": solver.time,
        "elapsed_seconds": elapsed,
        "rho_l1": rho_l1,
        "rho_l2": rho_l2,
        "rho_l1_order": float("nan"),
        "rho_l2_order": float("nan"),
        "mass_initial": mass_initial,
        "mass_final": mass_final,
        "mass_abs_change": abs(mass_final - mass_initial),
        "complete": complete,
        **diagnostics,
        "output": str(final_path),
    }
    (case_dir / "metrics.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="ascii")
    if not complete:
        raise RuntimeError(f"vortex case failed health checks: {row}")
    print(
        f"DONE plane={plane} N={n} steps={solver.step} L1={rho_l1:.8e} "
        f"L2={rho_l2:.8e} wall={elapsed:.2f}s",
        flush=True,
    )
    return row


def add_orders(rows: list[dict[str, object]]) -> None:
    for plane in PLANES:
        subset = sorted((row for row in rows if row["plane"] == plane), key=lambda row: int(row["N"]))
        for index in range(1, len(subset)):
            previous = subset[index - 1]
            current = subset[index]
            ratio = float(current["N"]) / float(previous["N"])
            for norm in ("rho_l1", "rho_l2"):
                current[f"{norm}_order"] = float(
                    np.log(float(previous[norm]) / float(current[norm])) / np.log(ratio)
                )


def write_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    fields = (
        "plane", "N", "inactive_cells", "nx", "ny", "nz", "dx_active", "cfl", "t_end",
        "quadrature", "steps", "time", "elapsed_seconds", "rho_l1", "rho_l1_order",
        "rho_l2", "rho_l2_order", "mass_abs_change", "rho_min", "rho_max", "p_min", "p_max",
        "nan_count", "complete",
    )
    with (out_dir / "convergence.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Three-dimensional WENO7-JS--RK4 vortex verification",
        "",
        "All values compare finite-volume density cell averages against a 15x15 Gauss exact average at t=2.",
        "",
        "| plane | N | rho L1 | order | rho L2 | order | steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        l1_order = "-" if not np.isfinite(float(row["rho_l1_order"])) else f"{float(row['rho_l1_order']):.4f}"
        l2_order = "-" if not np.isfinite(float(row["rho_l2_order"])) else f"{float(row['rho_l2_order']):.4f}"
        lines.append(
            f"| {row['plane']} | {row['N']} | {float(row['rho_l1']):.8e} | {l1_order} | "
            f"{float(row['rho_l2']):.8e} | {l2_order} | {row['steps']} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="ascii")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    colors = {"xy": "#0072B2", "yz": "#D55E00", "xz": "#009E73"}
    minimum_n = min(int(row["N"]) for row in rows)
    maximum_n = max(int(row["N"]) for row in rows)
    for axis, norm, label in zip(axes, ("rho_l1", "rho_l2"), (r"$L_1(\rho)$", r"$L_2(\rho)$"), strict=True):
        for plane in PLANES:
            subset = sorted((row for row in rows if row["plane"] == plane), key=lambda row: int(row["N"]))
            n_values = np.array([int(row["N"]) for row in subset])
            errors = np.array([float(row[norm]) for row in subset])
            axis.loglog(n_values, errors, marker="o", linewidth=1.6, color=colors[plane], label=plane.upper())
        reference_n = np.array([float(minimum_n), float(maximum_n)])
        anchor = min(float(row[norm]) for row in rows if int(row["N"]) == minimum_n)
        axis.loglog(
            reference_n,
            anchor * (reference_n / float(minimum_n)) ** -4.0,
            "k--",
            linewidth=1.1,
            label="fourth order",
        )
        axis.set_xlabel("Active cells per direction, N")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("3-D WENO7-JS--RK4 periodic isentropic-vortex convergence")
    fig.savefig(out_dir / "convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify 3-D WENO7-JS--RK4 with planar isentropic vortices")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planes", nargs="+", choices=PLANES, default=list(PLANES))
    parser.add_argument("--grids", nargs="+", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--inactive-cells", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--quadrature", type=int, default=15)
    parser.add_argument("--report-interval", type=int, default=25)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("warp_weno7_3d_rk4/runs/isentropic_vortex_cfl04"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quadrature != 15:
        raise ValueError("formal verification requires 15x15 Gauss cell averages")
    if args.cfl != 0.4 or args.t_end != 2.0:
        raise ValueError("formal verification requires CFL=0.4 and t_end=2")
    if args.inactive_cells < 7:
        raise ValueError("the inactive direction must contain at least seven cells")
    if any(n < 7 for n in args.grids):
        raise ValueError("all active grids must contain at least seven cells")

    wp.init()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for plane in args.planes:
        for n in args.grids:
            rows.append(
                run_case(
                    plane,
                    n,
                    args.inactive_cells,
                    args.device,
                    args.cfl,
                    args.t_end,
                    args.quadrature,
                    args.out_dir,
                    args.report_interval,
                )
            )
    add_orders(rows)
    write_summary(rows, args.out_dir)
    print(args.out_dir / "convergence.csv")
    print(args.out_dir / "SUMMARY.md")
    print(args.out_dir / "convergence.png")


if __name__ == "__main__":
    main()
