#!/usr/bin/env python3
"""Trusted 2D Warp/RK3 Sod validation for teacherfree WENO5 checkpoints.

This module deliberately uses the old deployed WENO5 Warp forward runner
(`run_weno5_circle_mlp_compare.advance_one_step`) instead of any lightweight
Python/Euler substitute. The initial condition is a very thin 2D planar Sod
strip, so it is cheap enough to run during training every eval interval.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32 import warp_weno5_helpers_mlp_f32 as wh
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_circle_mlp_compare_mlp_f32 import (
    advance_one_step,
    density_field,
)

wp = wh.wp

_PREPARED_DEVICES: set[str] = set()


def prepare_warp(device: str) -> None:
    wh.require_warp()
    if device not in _PREPARED_DEVICES:
        wp.init()
        wp.set_device(device)
        _PREPARED_DEVICES.add(device)


def wp_params_from_payload(payload: dict[str, np.ndarray], device: str) -> dict[str, object]:
    required = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    return {
        name: wp.array(payload[name], dtype=wp.float32, device=device, requires_grad=False)
        for name in required
    }


def _sod_states() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)


def _primitive_to_conserved_axis(
    rho: float,
    normal_vel: float,
    pressure: float,
    gamma: float,
    axis: str,
) -> np.ndarray:
    energy = 0.5 * rho * normal_vel * normal_vel + pressure / (gamma - 1.0)
    if axis == "y":
        return np.array([rho, 0.0, rho * normal_vel, energy], dtype=np.float64)
    return np.array([rho, rho * normal_vel, 0.0, energy], dtype=np.float64)


def _riemann_pressure_function(
    p: float,
    rho: float,
    pressure: float,
    sound: float,
    gamma: float,
) -> tuple[float, float]:
    if p > pressure:
        a = 2.0 / ((gamma + 1.0) * rho)
        b = (gamma - 1.0) / (gamma + 1.0) * pressure
        root = np.sqrt(a / (p + b))
        value = (p - pressure) * root
        deriv = root * (1.0 - 0.5 * (p - pressure) / (p + b))
        return float(value), float(deriv)

    exponent = (gamma - 1.0) / (2.0 * gamma)
    ratio = max(p / pressure, 1.0e-300)
    value = 2.0 * sound / (gamma - 1.0) * (ratio**exponent - 1.0)
    deriv = (1.0 / (rho * sound)) * ratio ** (-(gamma + 1.0) / (2.0 * gamma))
    return float(value), float(deriv)


@lru_cache(maxsize=128)
def _solve_riemann_star(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    gamma: float,
) -> tuple[float, float]:
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    p_guess = 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r)
    p = max(float(p_guess), 1.0e-12)

    for _ in range(80):
        f_l, df_l = _riemann_pressure_function(p, rho_l, p_l, a_l, gamma)
        f_r, df_r = _riemann_pressure_function(p, rho_r, p_r, a_r, gamma)
        residual = f_l + f_r + u_r - u_l
        p_new = p - residual / (df_l + df_r)
        if p_new <= 0.0 or not np.isfinite(p_new):
            p_new = 0.5 * p
        if abs(p_new - p) <= 1.0e-12 * (0.5 * (p_new + p) + 1.0e-12):
            p = p_new
            break
        p = p_new

    f_l, _ = _riemann_pressure_function(p, rho_l, p_l, a_l, gamma)
    f_r, _ = _riemann_pressure_function(p, rho_r, p_r, a_r, gamma)
    u = 0.5 * (u_l + u_r + f_r - f_l)
    return float(max(p, 1.0e-14)), float(u)


def _sample_sod_primitive(xi: float, gamma: float) -> tuple[float, float, float]:
    left, right = _sod_states()
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    p_star, u_star = _solve_riemann_star(left, right, gamma)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    if xi <= u_star:
        if p_star > p_l:
            shock_speed = u_l - a_l * np.sqrt((gp1 / (2.0 * gamma)) * (p_star / p_l) + gm1 / (2.0 * gamma))
            if xi <= shock_speed:
                return rho_l, u_l, p_l
            ratio = p_star / p_l
            rho_star = rho_l * ((ratio + gm1 / gp1) / ((gm1 / gp1) * ratio + 1.0))
            return float(rho_star), u_star, p_star

        a_star = a_l * (p_star / p_l) ** (gm1 / (2.0 * gamma))
        head_speed = u_l - a_l
        tail_speed = u_star - a_star
        if xi <= head_speed:
            return rho_l, u_l, p_l
        if xi >= tail_speed:
            rho_star = rho_l * (p_star / p_l) ** (1.0 / gamma)
            return float(rho_star), u_star, p_star
        u = 2.0 / gp1 * (a_l + 0.5 * gm1 * u_l + xi)
        a = 2.0 / gp1 * (a_l + 0.5 * gm1 * (u_l - xi))
        rho = rho_l * (a / a_l) ** (2.0 / gm1)
        pressure = p_l * (a / a_l) ** (2.0 * gamma / gm1)
        return float(rho), float(u), float(pressure)

    if p_star > p_r:
        shock_speed = u_r + a_r * np.sqrt((gp1 / (2.0 * gamma)) * (p_star / p_r) + gm1 / (2.0 * gamma))
        if xi >= shock_speed:
            return rho_r, u_r, p_r
        ratio = p_star / p_r
        rho_star = rho_r * ((ratio + gm1 / gp1) / ((gm1 / gp1) * ratio + 1.0))
        return float(rho_star), u_star, p_star

    a_star = a_r * (p_star / p_r) ** (gm1 / (2.0 * gamma))
    head_speed = u_r + a_r
    tail_speed = u_star + a_star
    if xi >= head_speed:
        return rho_r, u_r, p_r
    if xi <= tail_speed:
        rho_star = rho_r * (p_star / p_r) ** (1.0 / gamma)
        return float(rho_star), u_star, p_star
    u = 2.0 / gp1 * (-a_r + 0.5 * gm1 * u_r + xi)
    a = 2.0 / gp1 * (a_r - 0.5 * gm1 * (u_r - xi))
    rho = rho_r * (a / a_r) ** (2.0 / gm1)
    pressure = p_r * (a / a_r) ** (2.0 * gamma / gm1)
    return float(rho), float(u), float(pressure)


def _exact_sod_conserved(coord: float, t: float, gamma: float, axis: str) -> np.ndarray:
    left, right = _sod_states()
    if t <= 0.0:
        rho, vel, pressure = left if coord < 0.0 else right
    else:
        rho, vel, pressure = _sample_sod_primitive(coord / t, gamma)
    return _primitive_to_conserved_axis(rho, vel, pressure, gamma, axis)


def _cell_average_sod(coord_center: float, h: float, t: float, gamma: float, axis: str) -> np.ndarray:
    state = np.zeros(4, dtype=np.float64)
    for xi, weight in zip(wh.GAUSS15_XI, wh.GAUSS15_W):
        coord = coord_center + 0.5 * h * float(xi)
        state += float(weight) * _exact_sod_conserved(coord, t, gamma, axis)
    return 0.5 * state


def make_exact_sod_state(params: wh.Params, t: float, axis: str = "x") -> np.ndarray:
    ny_total, nx_total, _ = params.padded_shape
    u = np.zeros(params.padded_shape, dtype=np.float64)
    if axis == "y":
        y0 = -0.5 * params.y_length
        con_y = np.empty((ny_total, 4), dtype=np.float64)
        for j in range(ny_total):
            y = y0 + (j - params.ghost + 0.5) * params.dy
            con_y[j, :] = _cell_average_sod(y, params.dy, t, params.gamma, axis)
        for i in range(nx_total):
            u[:, i, :] = con_y
    else:
        x0 = -0.5 * params.x_length
        con_x = np.empty((nx_total, 4), dtype=np.float64)
        for i in range(nx_total):
            x = x0 + (i - params.ghost + 0.5) * params.dx
            con_x[i, :] = _cell_average_sod(x, params.dx, t, params.gamma, axis)
        for j in range(ny_total):
            u[j, :, :] = con_x
    return u


def _density_row(u: np.ndarray, params: wh.Params, axis: str) -> np.ndarray:
    g = params.ghost
    rho = density_field(u, params)
    if axis == "y":
        return rho[:, g : g + params.nx].mean(axis=1)
    return rho[g : g + params.ny, :].mean(axis=0)


def _sod_x_grid(params: wh.Params, axis: str) -> np.ndarray:
    if axis == "y":
        start = -0.5 * params.y_length
        return start + (np.arange(params.ny) + 0.5) * params.dy
    start = -0.5 * params.x_length
    return start + (np.arange(params.nx) + 0.5) * params.dx


def _row_metrics(row: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    diff = row - reference
    if not np.all(np.isfinite(diff)):
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(np.abs(diff))),
        float(np.sqrt(np.mean(diff * diff))),
        float(np.max(np.abs(diff))),
    )


def _plot_sod_density(
    x: np.ndarray,
    exact: np.ndarray,
    classical: np.ndarray,
    mlp: np.ndarray,
    l2_c: float,
    l2_m: float,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, exact, "k-", lw=1.5, label="exact")
    ax.plot(x, classical, "ro", ms=3, mfc="none", label=f"classical WENO5 (L2={l2_c:.2e})")
    ax.plot(x, mlp, "bs", ms=3, mfc="none", label=f"MLP (L2={l2_m:.2e})")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _write_weno7_style_outputs(
    step_dir: Path,
    params: wh.Params,
    axis: str,
    u0: np.ndarray,
    u_classical: np.ndarray,
    u_mlp: np.ndarray,
    reference: np.ndarray,
    mlp_t: float,
    mlp_steps: int,
    mlp_dt_values: list[float],
    c_l2: float,
    m_l2: float,
    raw_step: int,
) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    x_grid = _sod_x_grid(params, axis)
    rho0 = _density_row(u0, params, axis)
    rho_ref = _density_row(reference, params, axis)
    rho_c = _density_row(u_classical, params, axis)
    rho_m = _density_row(u_mlp, params, axis)
    _plot_sod_density(
        x_grid,
        rho_ref,
        rho_c,
        rho_m,
        c_l2,
        m_l2,
        step_dir / "sod_density.png",
        f"WENO5 teacher-free Sod step {raw_step}",
    )
    np.savez(
        step_dir / "circle_sod_results.npz",
        x=x_grid,
        initial_rho=rho0,
        reference_rho=rho_ref,
        classical_rho=rho_c,
        mlp_rho=rho_m,
        initial=u0,
        reference=reference,
        classical=u_classical,
        mlp=u_mlp,
        dt_values=np.array(mlp_dt_values, dtype=np.float64),
    )
    stats_c = wh.interior_stats(u_classical, params)
    stats_m = wh.interior_stats(u_mlp, params)
    with (step_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("validation_kind: weno7_style_sod_2d_warp\n")
        f.write(f"raw_step: {raw_step}\n")
        f.write(f"nx: {params.nx}\nny: {params.ny}\n")
        f.write(f"x_min: {-0.5 * params.x_length:.16e}\n")
        f.write(f"x_max: {0.5 * params.x_length:.16e}\n")
        f.write(f"dx: {params.dx:.16e}\n")
        f.write(f"dy: {params.dy:.16e}\n")
        f.write(f"t: {mlp_t:.16e}\n")
        f.write(f"steps: {mlp_steps}\n")
        f.write(f"dt_min: {float(np.min(mlp_dt_values)) if mlp_dt_values else 0.0:.16e}\n")
        f.write(f"dt_max: {float(np.max(mlp_dt_values)) if mlp_dt_values else 0.0:.16e}\n")
        f.write(f"classical_vs_reference_l2: {c_l2:.16e}\n")
        f.write(f"mlp_vs_reference_l2: {m_l2:.16e}\n")
        f.write(f"classical_rho_min: {float(stats_c['rho_min']):.16e}\n")
        f.write(f"classical_rho_max: {float(stats_c['rho_max']):.16e}\n")
        f.write(f"classical_p_min: {float(stats_c['p_min']):.16e}\n")
        f.write(f"mlp_rho_min: {float(stats_m['rho_min']):.16e}\n")
        f.write(f"mlp_rho_max: {float(stats_m['rho_max']):.16e}\n")
        f.write(f"mlp_p_min: {float(stats_m['p_min']):.16e}\n")
        f.write(f"mlp_nan_count: {float(stats_m['nan_count']):.16e}\n")


def _advance_solution(
    initial: np.ndarray,
    params: wh.Params,
    t_end: float,
    device: str,
    mlp_params: dict[str, object] | None,
    eno_cutoff: bool,
    boundary: str,
    characteristic_weno: bool,
    riemann_solver: str,
) -> tuple[np.ndarray, float, int, list[float], bool]:
    u = initial.copy()
    t = 0.0
    steps = 0
    dt_values: list[float] = []
    failed = False
    while t < t_end - 1.0e-14:
        dt_cfl = wh.compute_dt(u, params)
        if not np.isfinite(dt_cfl) or dt_cfl <= 0.0:
            failed = True
            break
        dt = min(dt_cfl, t_end - t)
        u = advance_one_step(
            u,
            params,
            dt,
            device,
            mlp_params,
            eno_cutoff,
            boundary,
            characteristic_weno,
            riemann_solver,
        )
        t += dt
        steps += 1
        dt_values.append(dt)
        stats = wh.interior_stats(u, params)
        if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
            failed = True
            break
    return u, t, steps, dt_values, failed


def run_warp_sod_validation(
    accepted: int,
    raw_step: int,
    mlp_params: dict[str, object],
    out_dir: Path,
    device: str,
    nx: int = 100,
    ny: int = 10,
    cfl: float = 0.4,
    t_end: float = 0.25,
    axis: str = "x",
    eno_cutoff: bool = True,
    boundary: str = "transmissive",
    weno_space: str = "characteristic",
    riemann_solver: str = "evilin",
) -> dict[str, object]:
    params = wh.Params(nx=nx, ny=ny, x_length=1.0, y_length=ny / float(nx), cfl=cfl, t_end=t_end)
    u0 = make_exact_sod_state(params, 0.0, axis)
    reference = make_exact_sod_state(params, t_end, axis)
    characteristic_weno = weno_space == "characteristic"

    u_classical, classical_t, classical_steps, classical_dt_values, classical_failed = _advance_solution(
        u0,
        params,
        t_end,
        device,
        None,
        False,
        boundary,
        characteristic_weno,
        riemann_solver,
    )
    u_mlp, mlp_t, mlp_steps, mlp_dt_values, mlp_failed = _advance_solution(
        u0,
        params,
        t_end,
        device,
        mlp_params,
        eno_cutoff,
        boundary,
        characteristic_weno,
        riemann_solver,
    )

    failed = classical_failed or mlp_failed
    full_mlp = (not mlp_failed) and abs(mlp_t - t_end) < 1.0e-12
    step_dir = out_dir / "circle_validation" / f"step_{accepted:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    rho_c = _density_row(u_classical, params, axis)
    rho_m = _density_row(u_mlp, params, axis)
    diff = rho_m - rho_c
    same_time = full_mlp and (not classical_failed) and abs(classical_t - mlp_t) < 1.0e-12
    if same_time:
        rho_diff_l1 = float(np.mean(np.abs(diff)))
        rho_diff_l2 = float(np.sqrt(np.mean(diff * diff)))
        rho_diff_linf = float(np.max(np.abs(diff)))
    else:
        rho_diff_l1 = float("nan")
        rho_diff_l2 = float("nan")
        rho_diff_linf = float("nan")

    reference_metrics = {
        "reference_nx": float(params.nx),
        "reference_ny": float(params.ny),
        "reference_steps": 0.0,
        "classical_vs_reference_l1": float("nan"),
        "classical_vs_reference_l2": float("nan"),
        "classical_vs_reference_linf": float("nan"),
        "mlp_vs_reference_l1": float("nan"),
        "mlp_vs_reference_l2": float("nan"),
        "mlp_vs_reference_linf": float("nan"),
        "gain_vs_reference_l1": float("nan"),
        "gain_vs_reference_l2": float("nan"),
        "gain_vs_reference_linf": float("nan"),
        "rel_gain_vs_reference_l1": float("nan"),
        "rel_gain_vs_reference_l2": float("nan"),
        "rel_gain_vs_reference_linf": float("nan"),
    }
    rho_ref = _density_row(reference, params, axis)
    c_l1, c_l2, c_linf = _row_metrics(rho_c, rho_ref)
    if full_mlp:
        m_l1, m_l2, m_linf = _row_metrics(rho_m, rho_ref)
        gain_l1 = c_l1 - m_l1
        gain_l2 = c_l2 - m_l2
        gain_linf = c_linf - m_linf
        rel_gain_l1 = gain_l1 / max(c_l1, 1.0e-300)
        rel_gain_l2 = gain_l2 / max(c_l2, 1.0e-300)
        rel_gain_linf = gain_linf / max(c_linf, 1.0e-300)
    else:
        m_l1 = m_l2 = m_linf = float("nan")
        gain_l1 = gain_l2 = gain_linf = float("nan")
        rel_gain_l1 = rel_gain_l2 = rel_gain_linf = float("nan")
    reference_metrics.update(
        {
            "classical_vs_reference_l1": c_l1,
            "classical_vs_reference_l2": c_l2,
            "classical_vs_reference_linf": c_linf,
            "mlp_vs_reference_l1": m_l1,
            "mlp_vs_reference_l2": m_l2,
            "mlp_vs_reference_linf": m_linf,
            "gain_vs_reference_l1": gain_l1,
            "gain_vs_reference_l2": gain_l2,
            "gain_vs_reference_linf": gain_linf,
            "rel_gain_vs_reference_l1": rel_gain_l1,
            "rel_gain_vs_reference_l2": rel_gain_l2,
            "rel_gain_vs_reference_linf": rel_gain_linf,
        }
    )

    stats_c = wh.interior_stats(u_classical, params)
    stats_m = wh.interior_stats(u_mlp, params)
    _write_weno7_style_outputs(
        step_dir,
        params,
        axis,
        u0,
        u_classical,
        u_mlp,
        reference,
        mlp_t,
        mlp_steps,
        mlp_dt_values if mlp_dt_values else classical_dt_values,
        c_l2,
        m_l2,
        raw_step,
    )
    return {
        "raw_step": raw_step,
        "step": accepted,
        "validation_kind": "weno7-style-sod-2d-warp",
        "axis": axis,
        "riemann_solver": riemann_solver,
        "weno_space": weno_space,
        "x_min": -0.5 * params.x_length,
        "x_max": 0.5 * params.x_length,
        "dx": params.dx,
        "dy": params.dy,
        "t": mlp_t,
        "steps": mlp_steps,
        "classical_t": classical_t,
        "classical_steps": classical_steps,
        "mlp_t": mlp_t,
        "mlp_steps": mlp_steps,
        "dt_min": float(np.min(mlp_dt_values)) if mlp_dt_values else 0.0,
        "dt_max": float(np.max(mlp_dt_values)) if mlp_dt_values else 0.0,
        "dt_mean": float(np.mean(mlp_dt_values)) if mlp_dt_values else 0.0,
        "classical_dt_min": float(np.min(classical_dt_values)) if classical_dt_values else 0.0,
        "classical_dt_max": float(np.max(classical_dt_values)) if classical_dt_values else 0.0,
        "classical_dt_mean": float(np.mean(classical_dt_values)) if classical_dt_values else 0.0,
        "rho_diff_l1": rho_diff_l1,
        "rho_diff_l2": rho_diff_l2,
        "rho_diff_linf": rho_diff_linf,
        **reference_metrics,
        "classical_rho_min": float(stats_c["rho_min"]),
        "classical_rho_max": float(stats_c["rho_max"]),
        "classical_p_min": float(stats_c["p_min"]),
        "mlp_rho_min": float(stats_m["rho_min"]),
        "mlp_rho_max": float(stats_m["rho_max"]),
        "mlp_p_min": float(stats_m["p_min"]),
        "mlp_nan_count": float(stats_m["nan_count"]),
        "mlp_rho_neg": float(stats_m["rho_neg"]),
        "mlp_p_neg": float(stats_m["p_neg"]),
        "failed": float(failed),
        "plot_dir": str(step_dir),
    }


def _ordered_fields(records: list[dict[str, object]]) -> list[str]:
    preferred = [
        "raw_step",
        "step",
        "validation_kind",
        "axis",
        "t",
        "steps",
        "classical_t",
        "classical_steps",
        "mlp_t",
        "mlp_steps",
        "dt_min",
        "dt_max",
        "dt_mean",
        "classical_dt_min",
        "classical_dt_max",
        "classical_dt_mean",
        "rho_diff_l1",
        "rho_diff_l2",
        "rho_diff_linf",
        "classical_vs_reference_l1",
        "classical_vs_reference_l2",
        "classical_vs_reference_linf",
        "mlp_vs_reference_l1",
        "mlp_vs_reference_l2",
        "mlp_vs_reference_linf",
        "gain_vs_reference_l1",
        "gain_vs_reference_l2",
        "gain_vs_reference_linf",
        "rel_gain_vs_reference_l1",
        "rel_gain_vs_reference_l2",
        "rel_gain_vs_reference_linf",
        "classical_rho_min",
        "classical_rho_max",
        "classical_p_min",
        "mlp_rho_min",
        "mlp_rho_max",
        "mlp_p_min",
        "mlp_nan_count",
        "mlp_rho_neg",
        "mlp_p_neg",
        "failed",
        "plot_dir",
    ]
    keys = {key for record in records for key in record}
    return [key for key in preferred if key in keys] + sorted(keys.difference(preferred))


def write_warp_sod_outputs(out_dir: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = _ordered_fields(records)
    with (out_dir / "circle_validation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    np.savez(out_dir / "circle_validation_metrics.npz", records=np.array(records, dtype=object))

    steps = np.array([float(r.get("step", np.nan)) for r in records], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    for key, label in (
        ("rho_diff_l2", "MLP - classical rho L2"),
        ("mlp_vs_reference_l2", "MLP - exact rho L2"),
        ("classical_vs_reference_l2", "classical - exact rho L2"),
    ):
        vals = np.array([float(r.get(key, np.nan)) for r in records], dtype=np.float64)
        axes[0 if key == "rho_diff_l2" else 1].plot(steps, vals, marker="o", label=label)
    axes[0].set_ylabel("rho L2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_ylabel("rho L2")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    gain = np.array([float(r.get("rel_gain_vs_reference_l2", np.nan)) for r in records], dtype=np.float64)
    failed = np.array([float(r.get("failed", np.nan)) for r in records], dtype=np.float64)
    axes[2].plot(steps, gain, marker="o", label="relative L2 gain")
    axes[2].plot(steps, failed, marker="x", label="failed")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("gain / failed")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.savefig(out_dir / "circle_validation_trends.png", dpi=180)
    plt.close(fig)
