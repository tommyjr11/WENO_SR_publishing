#!/usr/bin/env python3
"""Double Mach reflection comparison for WENO5 classical, WENO5-MLP, and WENO7/ADER4.

The test uses the Woodward-Colella initial data on [0,4]x[0,1].  The bottom
boundary is inflow/exact before x0=1/6 and reflecting after x0.  The left and
top boundaries are filled from the time-dependent incident shock, while the
right boundary is outflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import warp_weno5_helpers as wh5
import warp_weno7_ader4_helpers as wh7
import weno5_rk3_warp as weno5
import weno7_ader4_warp as weno7


GAUSS5_XI = np.array(
    [
        -0.906179845938663992797626878299392759799957275390625,
        -0.53846931010568310771446931539685465395450592041015625,
        0.0,
        0.53846931010568310771446931539685465395450592041015625,
        0.906179845938663992797626878299392759799957275390625,
    ],
    dtype=np.float64,
)
GAUSS5_W = np.array(
    [
        0.2369268850561890835142640407196017913520336151123046875,
        0.47862867049936646804129151441133581101894378662109375,
        0.5688888888888888888888888888888888888888888888888888889,
        0.47862867049936646804129151441133581101894378662109375,
        0.2369268850561890835142640407196017913520336151123046875,
    ],
    dtype=np.float64,
)


def double_mach_conserved_state(x: float, y: float, t: float, gamma: float = 1.4) -> np.ndarray:
    root3 = np.sqrt(3.0)
    x0 = 1.0 / 6.0
    shock_x = x0 + y / root3 + 20.0 * t / root3
    if x < shock_x:
        return wh5.primitive_to_conserved(8.0, 8.25 * root3 * 0.5, -8.25 * 0.5, 116.5, gamma)
    return wh5.primitive_to_conserved(1.4, 0.0, 0.0, 1.0, gamma)


def quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order == 1:
        return np.array([0.0], dtype=np.float64), np.array([2.0], dtype=np.float64)
    if order == 5:
        return GAUSS5_XI, GAUSS5_W
    if order == 15:
        return wh5.GAUSS15_XI, wh5.GAUSS15_W
    raise ValueError("--init-quadrature must be 1, 5, or 15")


def cell_average_state(xc: float, yc: float, dx: float, dy: float, t: float, gamma: float, order: int) -> np.ndarray:
    xi, wi = quadrature_rule(order)
    state = np.zeros(4, dtype=np.float64)
    for ax, wx in zip(xi, wi):
        x = xc + 0.5 * dx * float(ax)
        for ay, wy in zip(xi, wi):
            y = yc + 0.5 * dy * float(ay)
            state += float(wx) * float(wy) * double_mach_conserved_state(x, y, t, gamma)
    return 0.25 * state


def make_initial_state(params, order: int) -> np.ndarray:
    state = np.zeros(params.padded_shape, dtype=np.float64)
    ny_total, nx_total, _ = state.shape
    for j in range(ny_total):
        y = (j - params.ghost + 0.5) * params.dy
        for i in range(nx_total):
            x = (i - params.ghost + 0.5) * params.dx
            state[j, i, :] = cell_average_state(x, y, params.dx, params.dy, 0.0, params.gamma, order)
    return state


def run_weno7_solution(initial: np.ndarray, params: wh7.Params, args: argparse.Namespace, label: str) -> tuple[np.ndarray, float, list[float]]:
    arrays = weno7.allocate_warp_arrays(initial, params, args.device)
    t = 0.0
    dt_values: list[float] = []
    max_steps = args.steps if args.steps > 0 else 10_000_000
    for step in range(1, max_steps + 1):
        dt = wh7.compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, args.device)
        if args.t_end > 0.0 and t + dt > args.t_end:
            dt = args.t_end - t
        if dt <= 0.0:
            break
        weno7.launch_weno7_ader4_step(arrays, params, dt, args.device, "double-mach", t, args.riemann_solver)
        t += dt
        dt_values.append(dt)
        reached_t_end = args.t_end > 0.0 and t >= args.t_end - 1.0e-14
        should_report = step == 1 or reached_t_end or (args.report_interval > 0 and step % args.report_interval == 0)
        if should_report:
            host = arrays["u"].numpy()
            stats = wh7.interior_stats(host, params)
            print(
                f"{label} step={step:05d} t={t:.16e} dt={dt:.16e} "
                f"rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
                f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] "
                f"nan={int(stats['nan_count'])} rho_neg={int(stats['rho_neg'])} p_neg={int(stats['p_neg'])}",
                flush=True,
            )
            if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
                break
        if reached_t_end:
            break
    return arrays["u"].numpy(), t, dt_values


def primitive_fields(u: np.ndarray, params, helper) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = params.ghost
    pri = helper.primitive_from_conserved(u[g : g + params.ny, g : g + params.nx, :], params.gamma)
    return pri[..., 0], pri[..., 1], pri[..., 2], pri[..., 3]


def plot_density(
    u: np.ndarray,
    params,
    helper,
    out_path: Path,
    title: str,
    rho_limits: tuple[float, float] | None = None,
) -> None:
    rho, _, _, pressure = primitive_fields(u, params, helper)
    x = (np.arange(params.nx, dtype=np.float64) + 0.5) * params.dx
    y = (np.arange(params.ny, dtype=np.float64) + 0.5) * params.dy
    xg, yg = np.meshgrid(x, y)
    finite_rho = np.isfinite(rho)
    if not np.any(finite_rho):
        fig, ax = plt.subplots(figsize=(13.5, 3.8))
        ax.text(0.5, 0.5, "solution contains only NaN/Inf", ha="center", va="center", fontsize=16)
        ax.set_title(title)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"saved_plot={out_path}", flush=True)
        return
    if rho_limits is None:
        rho_min = float(np.nanmin(rho))
        rho_max = float(np.nanmax(rho))
    else:
        rho_min, rho_max = rho_limits
    levels = np.linspace(rho_min, rho_max, 80)
    contour_levels = np.linspace(rho_min, rho_max, 36)

    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    im = ax.contourf(xg, yg, rho, levels=levels, cmap="turbo", extend="both")
    ax.contour(xg, yg, rho, levels=contour_levels, colors="k", linewidths=0.18, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, params.x_length)
    ax.set_ylim(0.0, params.y_length)
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax, pad=0.012, shrink=0.90)
    cbar.set_label(r"density $\rho$")
    text = (
        f"rho=[{np.nanmin(rho):.3g},{np.nanmax(rho):.3g}], "
        f"p_min={np.nanmin(pressure):.3g}"
    )
    ax.text(0.01, 0.02, text, transform=ax.transAxes, fontsize=8.5, color="white", bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={out_path}", flush=True)


def plot_combined(results: dict[str, tuple[np.ndarray, object, object]], out_path: Path, rho_limits: tuple[float, float] | None = None) -> None:
    results = {
        label: item
        for label, item in results.items()
        if np.any(np.isfinite(primitive_fields(item[0], item[1], item[2])[0]))
    }
    if not results:
        return
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13.5, 3.25 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    all_rho = []
    for u, params, helper in results.values():
        all_rho.append(primitive_fields(u, params, helper)[0])
    if rho_limits is None:
        rho_min = float(min(np.nanmin(r) for r in all_rho))
        rho_max = float(max(np.nanmax(r) for r in all_rho))
    else:
        rho_min, rho_max = rho_limits
    levels = np.linspace(rho_min, rho_max, 80)
    contour_levels = np.linspace(rho_min, rho_max, 36)
    last_im = None
    for ax, (label, (u, params, helper)) in zip(axes, results.items()):
        rho, _, _, _ = primitive_fields(u, params, helper)
        x = (np.arange(params.nx, dtype=np.float64) + 0.5) * params.dx
        y = (np.arange(params.ny, dtype=np.float64) + 0.5) * params.dy
        xg, yg = np.meshgrid(x, y)
        last_im = ax.contourf(xg, yg, rho, levels=levels, cmap="turbo", extend="both")
        ax.contour(xg, yg, rho, levels=contour_levels, colors="k", linewidths=0.16, alpha=0.42)
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0.0, params.x_length)
        ax.set_ylim(0.0, params.y_length)
        ax.set_aspect("equal", adjustable="box")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, pad=0.012, shrink=0.92)
        cbar.set_label(r"density $\rho$")
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={out_path}", flush=True)


def save_summary(path: Path, label: str, u: np.ndarray, params, helper, t: float, dt_values: list[float]) -> None:
    stats = helper.interior_stats(u, params)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"label: {label}\n")
        f.write(f"t: {t:.16e}\n")
        f.write(f"steps: {len(dt_values)}\n")
        f.write(f"dt_min: {float(np.min(dt_values)) if dt_values else 0.0:.16e}\n")
        f.write(f"dt_max: {float(np.max(dt_values)) if dt_values else 0.0:.16e}\n")
        f.write(f"dt_mean: {float(np.mean(dt_values)) if dt_values else 0.0:.16e}\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("plots/weno5_offline_power2_5_10_6_6_3_400k_disc15/checkpoints/model_step_137000.npz"))
    parser.add_argument("--nx", type=int, default=800)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=0, help="0 means run to --t-end")
    parser.add_argument("--init-quadrature", type=int, choices=(1, 5, 15), default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--weno-space", choices=("characteristic", "conserved"), default="characteristic")
    parser.add_argument("--riemann-solver", choices=("evilin", "hllc"), default="evilin")
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-weno5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-weno7", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/weno_double_mach_800x200_cfl04"))
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    wh5.require_warp()
    wp = wh5.wp
    wp.init()
    wp.set_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    
    params7 = wh7.Params(nx=args.nx, ny=args.ny, x_length=4.0, y_length=1.0, cfl=args.cfl, t_end=args.t_end)
    params5 = wh5.Params(nx=args.nx, ny=args.ny, x_length=4.0, y_length=1.0, cfl=args.cfl, t_end=args.t_end)

    print(
        f"double_mach_start nx={args.nx} ny={args.ny} cfl={args.cfl} t_end={args.t_end} "
        f"init_quadrature={args.init_quadrature} riemann_solver={args.riemann_solver} model={args.model}",
        flush=True,
    )

    results: dict[str, tuple[np.ndarray, object, object]] = {}
    run_args5 = SimpleNamespace(
        steps=args.steps if args.steps > 0 else 10_000_000,
        dt_mode="warp",
        t_end=args.t_end,
        device=args.device,
        weno_space=args.weno_space,
        boundary="double-mach",
        eno_cutoff=args.eno_cutoff,
        riemann_solver=args.riemann_solver,
        report_interval=args.report_interval,
    )
    
    if args.run_weno7:
        initial7 = make_initial_state(params7, args.init_quadrature)
        stats7 = wh7.interior_stats(initial7, params7)
        print(f"initial_weno7 rho=[{stats7['rho_min']:.6e},{stats7['rho_max']:.6e}] p=[{stats7['p_min']:.6e},{stats7['p_max']:.6e}]", flush=True)
        w7, t7, dt7 = run_weno7_solution(initial7, params7, args, "weno7_ader4")
        np.save(args.out_dir / "weno7_ader4.npy", w7)
        save_summary(args.out_dir / "weno7_ader4_summary.txt", "WENO7 ADER4 HEOC classical", w7, params7, wh7, t7, dt7)
        plot_density(w7, params7, wh7, args.out_dir / "weno7_ader4_density.png", f"WENO7 ADER4 HEOC classical, t={t7:.3f}")
        results["WENO7 ADER4 HEOC classical"] = (w7, params7, wh7)

    if args.run_weno5:
        initial5 = make_initial_state(params5, args.init_quadrature)
        stats5 = wh5.interior_stats(initial5, params5)
        print(f"initial_weno5 rho=[{stats5['rho_min']:.6e},{stats5['rho_max']:.6e}] p=[{stats5['p_min']:.6e},{stats5['p_max']:.6e}]", flush=True)

        classical5, t5, _, dt5 = weno5.run_solution(initial5, params5, run_args5, None, "weno5_classical")
        np.save(args.out_dir / "weno5_classical.npy", classical5)
        save_summary(args.out_dir / "weno5_classical_summary.txt", "WENO5 RK3 classical", classical5, params5, wh5, t5, dt5)
        plot_density(classical5, params5, wh5, args.out_dir / "weno5_classical_density.png", f"WENO5 RK3 classical, t={t5:.3f}")
        results["WENO5 RK3 classical"] = (classical5, params5, wh5)

        mlp_params = weno5.load_mlp_params(args.model, args.device)
        mlp5, tm, _, dtm = weno5.run_solution(initial5, params5, run_args5, mlp_params, "weno5_mlp")
        np.save(args.out_dir / "weno5_mlp.npy", mlp5)
        save_summary(args.out_dir / "weno5_mlp_summary.txt", "WENO5 RK3 MLP", mlp5, params5, wh5, tm, dtm)
        plot_density(mlp5, params5, wh5, args.out_dir / "weno5_mlp_density.png", f"WENO5 RK3 MLP, t={tm:.3f}")
        results["WENO5 RK3 MLP"] = (mlp5, params5, wh5)

    

    plot_combined(results, args.out_dir / "density_compare_all.png")
    print(f"done out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
