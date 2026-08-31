#!/usr/bin/env python3
"""Trusted WENO5/RK3 runner with a symmetric FP32 MLP and FP64 solver."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import warp_weno5_helpers as wh
from warp_mlp_f32.weno5_rk3_diff_mlp_f32 import (
    allocate_diff_arrays,
    launch_weno5_rk3_diff_step,
)


wp = wh.wp


def circle_sod_conserved(x: float, y: float, gamma: float) -> np.ndarray:
    r = np.sqrt((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0))
    if r < 0.4:
        return wh.primitive_to_conserved(1.0, 0.0, 0.0, 1.0, gamma)
    return wh.primitive_to_conserved(0.125, 0.0, 0.0, 0.1, gamma)


def cell_average_circle_sod(xc: float, yc: float, dx: float, dy: float, gamma: float) -> np.ndarray:
    state = np.zeros(4, dtype=np.float64)
    for xi, wx in zip(wh.GAUSS15_XI, wh.GAUSS15_W):
        x = xc + 0.5 * dx * float(xi)
        for eta, wy in zip(wh.GAUSS15_XI, wh.GAUSS15_W):
            y = yc + 0.5 * dy * float(eta)
            state += float(wx) * float(wy) * circle_sod_conserved(x, y, gamma)
    return 0.25 * state


def make_circle_sod_state(params: wh.Params) -> np.ndarray:
    u = np.zeros(params.padded_shape, dtype=np.float64)
    for j in range(params.ny + 2 * params.ghost):
        y = (j - params.ghost + 0.5) * params.dy
        for i in range(params.nx + 2 * params.ghost):
            x = (i - params.ghost + 0.5) * params.dx
            u[j, i, :] = cell_average_circle_sod(x, y, params.dx, params.dy, params.gamma)
    return u


def load_mlp_params(model_path: Path, device: str) -> dict[str, object]:
    data = np.load(model_path, allow_pickle=True)
    required = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"Model {model_path} is missing arrays: {missing}")
    expected_shapes = {
        "w1": (1, 5, 10),
        "b1": (1, 10),
        "w2": (1, 10, 6),
        "b2": (1, 6),
        "w3": (1, 6, 6),
        "b3": (1, 6),
        "w4": (1, 6, 3),
        "b4": (1, 3),
    }
    wrong = {name: data[name].shape for name, shape in expected_shapes.items() if data[name].shape != shape}
    if wrong:
        raise ValueError(
            f"Model {model_path} uses old/incompatible MLP shapes: {wrong}. "
            "Retrain with the shared direct beta-ratio 5->10->6->6->3 WENO5-NN architecture."
        )
    if "meta_json" in data.files:
        meta = str(data["meta_json"])
        if "shared_direct_beta_ratio_5_10_6_6_3" not in meta:
            print(
                "warning: model metadata does not say it was trained with the "
                "shared direct beta-ratio 5->10->6->6->3 WENO5-NN architecture; retrain before trusting this comparison."
            )
    return {
        name: wp.array(data[name], dtype=wp.float32, device=device, requires_grad=False)
        for name in required
    }


def advance_one_step(
    u_host: np.ndarray,
    params: wh.Params,
    dt: float,
    device: str,
    mlp_params: dict[str, object] | None,
    eno_cutoff: bool,
    boundary: str,
    characteristic_weno: bool = True,
    riemann_solver: str = "force",
) -> np.ndarray:
    arrays = allocate_diff_arrays(u_host, u_host, params, device)
    launch_weno5_rk3_diff_step(arrays, params, dt, device, characteristic_weno, mlp_params, eno_cutoff, boundary, riemann_solver)
    wp.synchronize()
    return arrays["u3"].numpy()


def density_field(u: np.ndarray, params: wh.Params) -> np.ndarray:
    g = params.ghost
    return u[g : g + params.ny, g : g + params.nx, 0]


def grid_centers(params: wh.Params) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(params.nx) + 0.5) * params.dx
    y = (np.arange(params.ny) + 0.5) * params.dy
    return x, y


def resample_field_to_params(field: np.ndarray, source: wh.Params, target: wh.Params) -> np.ndarray:
    src_x, src_y = grid_centers(source)
    dst_x, dst_y = grid_centers(target)
    tmp = np.empty((source.ny, target.nx), dtype=np.float64)
    for j in range(source.ny):
        tmp[j, :] = np.interp(dst_x, src_x, field[j, :])
    out = np.empty((target.ny, target.nx), dtype=np.float64)
    for i in range(target.nx):
        out[:, i] = np.interp(dst_y, src_y, tmp[:, i])
    return out


def run_classical_solution(
    initial: np.ndarray,
    params: wh.Params,
    t_end: float,
    device: str,
    boundary: str,
    report_interval: int,
    label: str,
    characteristic_weno: bool = True,
) -> tuple[np.ndarray, float, int, list[float]]:
    u = initial.copy()
    t = 0.0
    steps = 0
    dt_values: list[float] = []
    while t < t_end - 1.0e-14:
        dt = min(wh.compute_dt(u, params), t_end - t)
        u = advance_one_step(u, params, dt, device, None, False, boundary, characteristic_weno)
        t += dt
        steps += 1
        dt_values.append(dt)
        if steps == 1 or steps % report_interval == 0 or t >= t_end - 1.0e-14:
            stats = wh.interior_stats(u, params)
            print(
                f"{label} step={steps:04d} t={t:.8e} dt={dt:.8e} "
                f"rho=[{stats['rho_min']:.5e},{stats['rho_max']:.5e}] p_min={stats['p_min']:.5e}"
            )
            if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
                break
    return u, t, steps, dt_values


def plot_results(
    classical: np.ndarray,
    mlp: np.ndarray,
    initial: np.ndarray,
    params: wh.Params,
    t: float,
    out_dir: Path,
    reference: np.ndarray | None = None,
    reference_params: wh.Params | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rho_c = density_field(classical, params)
    rho_m = density_field(mlp, params)
    rho_0 = density_field(initial, params)
    rho_ref = None
    rho_ref_on_coarse = None
    if reference is not None and reference_params is not None:
        rho_ref = density_field(reference, reference_params)
        rho_ref_on_coarse = resample_field_to_params(rho_ref, reference_params, params)
    diff = rho_m - rho_c
    extent = [0.0, params.x_length, 0.0, params.y_length]
    ref_extent = extent if reference_params is None else [0.0, reference_params.x_length, 0.0, reference_params.y_length]

    vmin = float(min(np.min(rho_c), np.min(rho_m), np.min(rho_ref) if rho_ref is not None else np.inf))
    vmax = float(max(np.max(rho_c), np.max(rho_m), np.max(rho_ref) if rho_ref is not None else -np.inf))

    panels: list[tuple[np.ndarray, str, list[float]]] = [(rho_0, "initial", extent)]
    if rho_ref is not None:
        panels.append((rho_ref, f"reference classical {reference_params.nx}x{reference_params.ny}", ref_extent))
    panels.extend([(rho_c, "classical WENO5-RK3", extent), (rho_m, "MLP WENO5-RK3", extent)])

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.2), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for ax, (data, title, image_extent) in zip(axes_arr, panels):
        im = ax.imshow(data, origin="lower", extent=image_extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"Circle Sod density, t={t:.6f}")
    fig.savefig(out_dir / "density_compare.png", dpi=180)
    plt.close(fig)

    if rho_ref_on_coarse is None:
        diff_panels = [(diff, "MLP - classical")]
    else:
        diff_panels = [
            (rho_c - rho_ref_on_coarse, "classical - reference"),
            (rho_m - rho_ref_on_coarse, "MLP - reference"),
            (diff, "MLP - classical"),
        ]
    lim = float(max(np.max(np.abs(data)) for data, _ in diff_panels))
    fig, axes = plt.subplots(1, len(diff_panels), figsize=(5.2 * len(diff_panels), 4.4), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for ax, (data, title) in zip(axes_arr, diff_panels):
        im = ax.imshow(data, origin="lower", extent=extent, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out_dir / "density_difference.png", dpi=180)
    plt.close(fig)

    x, _ = grid_centers(params)
    mid = params.ny // 2
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    ax.plot(x, rho_0[mid, :], "k:", lw=1.2, label="initial")
    if rho_ref is not None and reference_params is not None:
        ref_x, _ = grid_centers(reference_params)
        ref_mid = reference_params.ny // 2
        ax.plot(ref_x, rho_ref[ref_mid, :], "k-", lw=2.0, label="reference")
    ax.plot(x, rho_c[mid, :], "--", lw=1.7, label="classical")
    ax.plot(x, rho_m[mid, :], "-", lw=1.5, label="mlp")
    ax.set_xlabel("x at y=1")
    ax.set_ylabel("rho")
    ax.set_title(f"Centerline density, t={t:.6f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / "centerline_density.png", dpi=180)
    plt.close(fig)


def save_summary(
    classical: np.ndarray,
    mlp: np.ndarray,
    initial: np.ndarray,
    params: wh.Params,
    t: float,
    steps: int,
    dt_values: list[float],
    out_dir: Path,
    reference: np.ndarray | None = None,
    reference_params: wh.Params | None = None,
    reference_steps: int = 0,
    reference_dt_values: list[float] | None = None,
) -> None:
    rho_c = density_field(classical, params)
    rho_m = density_field(mlp, params)
    diff = rho_m - rho_c
    stats_c = wh.interior_stats(classical, params)
    stats_m = wh.interior_stats(mlp, params)
    stats_0 = wh.interior_stats(initial, params)
    summary = {
        "t": t,
        "steps": steps,
        "dt_min": float(np.min(dt_values)),
        "dt_max": float(np.max(dt_values)),
        "dt_mean": float(np.mean(dt_values)),
        "rho_diff_l1": float(np.mean(np.abs(diff))),
        "rho_diff_l2": float(np.sqrt(np.mean(diff * diff))),
        "rho_diff_linf": float(np.max(np.abs(diff))),
        "initial_mass": stats_0["mass"],
        "classical_mass": stats_c["mass"],
        "mlp_mass": stats_m["mass"],
        "classical_rho_min": stats_c["rho_min"],
        "classical_rho_max": stats_c["rho_max"],
        "classical_p_min": stats_c["p_min"],
        "classical_p_max": stats_c["p_max"],
        "mlp_rho_min": stats_m["rho_min"],
        "mlp_rho_max": stats_m["rho_max"],
        "mlp_p_min": stats_m["p_min"],
        "mlp_p_max": stats_m["p_max"],
        "mlp_nan_count": stats_m["nan_count"],
        "classical_nan_count": stats_c["nan_count"],
    }
    if reference is not None and reference_params is not None:
        rho_ref = density_field(reference, reference_params)
        rho_ref_on_coarse = resample_field_to_params(rho_ref, reference_params, params)
        c_ref = rho_c - rho_ref_on_coarse
        m_ref = rho_m - rho_ref_on_coarse
        summary.update(
            {
                "reference_nx": reference_params.nx,
                "reference_ny": reference_params.ny,
                "reference_steps": reference_steps,
                "reference_dt_min": float(np.min(reference_dt_values)) if reference_dt_values else 0.0,
                "reference_dt_max": float(np.max(reference_dt_values)) if reference_dt_values else 0.0,
                "classical_vs_reference_l1": float(np.mean(np.abs(c_ref))),
                "classical_vs_reference_l2": float(np.sqrt(np.mean(c_ref * c_ref))),
                "classical_vs_reference_linf": float(np.max(np.abs(c_ref))),
                "mlp_vs_reference_l1": float(np.mean(np.abs(m_ref))),
                "mlp_vs_reference_l2": float(np.sqrt(np.mean(m_ref * m_ref))),
                "mlp_vs_reference_linf": float(np.max(np.abs(m_ref))),
            }
        )
    with (out_dir / "summary.txt").open("w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    np.savez(
        out_dir / "circle_sod_results.npz",
        initial=initial,
        classical=classical,
        mlp=mlp,
        reference=reference if reference is not None else np.array([], dtype=np.float64),
        dt=np.array(dt_values, dtype=np.float64),
        reference_dt=np.array(reference_dt_values or [], dtype=np.float64),
        summary=np.array(summary, dtype=object),
    )
    print("summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def run(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp.init()
    wp.set_device(args.device)

    params = wh.Params(nx=args.nx, ny=args.ny, x_length=2.0, y_length=2.0, cfl=args.cfl, t_end=args.t_end)
    mlp_params = load_mlp_params(args.model, args.device)

    u0 = make_circle_sod_state(params)
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
        u_classical = advance_one_step(u_classical, params, dt, args.device, None, False, args.boundary, characteristic_weno, args.riemann_solver)
        u_mlp = advance_one_step(u_mlp, params, dt, args.device, mlp_params, args.eno_cutoff, args.boundary, characteristic_weno, args.riemann_solver)
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
                args.out_dir.mkdir(parents=True, exist_ok=True)
                plot_results(u_classical, u_mlp, u0, params, t, args.out_dir)
                save_summary(u_classical, u_mlp, u0, params, t, steps, dt_values, args.out_dir)
                with (args.out_dir / "failure.txt").open("w") as f:
                    f.write("Simulation produced NaN/negative rho/p before reaching t_end.\n")
                    f.write(f"failed_step: {steps}\n")
                    f.write(f"failed_time: {t}\n")
                    f.write(f"requested_t_end: {args.t_end}\n")
                    f.write(f"classical_stats: {stats_c}\n")
                    f.write(f"mlp_stats: {stats_m}\n")
                print(f"failure: saved partial comparison to {args.out_dir}")
                return

    reference = None
    reference_params = None
    reference_steps = 0
    reference_dt_values: list[float] = []
    if args.reference_nx > 0:
        reference_ny = args.reference_ny if args.reference_ny > 0 else args.reference_nx
        reference_params = wh.Params(nx=args.reference_nx, ny=reference_ny, x_length=2.0, y_length=2.0, cfl=args.cfl, t_end=args.t_end)
        reference_initial = make_circle_sod_state(reference_params)
        reference, _, reference_steps, reference_dt_values = run_classical_solution(
            reference_initial,
            reference_params,
            args.t_end,
            args.device,
            args.boundary,
            args.reference_report_interval,
            "reference",
            args.weno_space == "characteristic",
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_results(u_classical, u_mlp, u0, params, t, args.out_dir, reference, reference_params)
    save_summary(
        u_classical,
        u_mlp,
        u0,
        params,
        t,
        steps,
        dt_values,
        args.out_dir,
        reference,
        reference_params,
        reference_steps,
        reference_dt_values,
    )
    print(f"plots: {args.out_dir / 'density_compare.png'}")
    print(f"plots: {args.out_dir / 'density_difference.png'}")
    print(f"plots: {args.out_dir / 'centerline_density.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument("--t-end", type=float, default=0.25)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--boundary", choices=("periodic", "transmissive"), default="transmissive")
    parser.add_argument("--weno-space", choices=("characteristic", "conserved"), default="characteristic")
    parser.add_argument("--riemann-solver", choices=("force", "evilin"), default="force")
    parser.add_argument("--report-interval", type=int, default=10)
    parser.add_argument("--reference-nx", type=int, default=0, help="Run a classical high-resolution reference with this nx; 0 disables reference.")
    parser.add_argument("--reference-ny", type=int, default=0, help="Reference ny; defaults to --reference-nx when omitted.")
    parser.add_argument("--reference-report-interval", type=int, default=25)
    parser.add_argument("--model", type=Path, default=Path("plots/weno5_mlp_train_100_jsratio_corr_4_16_3/model_latest.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots/weno5_circle_100_t025_jsratio_corr_4_16_3"))
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=True, help="Apply the ENO cutoff layer for the MLP run.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
