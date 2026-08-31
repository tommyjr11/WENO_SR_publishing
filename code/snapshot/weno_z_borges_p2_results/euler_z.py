"""Adapters for the isolated two-dimensional WENO-Z Euler solvers."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import weno5_helpers_z as W5
from . import weno5_hllc_z_min as W5H
from . import weno7_point_rk4_z as W7


def make_weno5_params(nx: int, ny: int, x_length: float, y_length: float,
                      cfl: float, t_end: float) -> W5.Params:
    return W5.Params(
        nx=nx, ny=ny, x_length=x_length, y_length=y_length,
        cfl=cfl, t_end=t_end,
    )


def make_weno7_params(nx: int, ny: int, x_min: float, x_max: float,
                      y_min: float, y_max: float, cfl: float,
                      t_end: float) -> W7.Params:
    return W7.Params(
        nx=nx, ny=ny, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, cfl=cfl, t_end=t_end,
    )


def run_weno5(initial: np.ndarray, params: W5.Params, *, device: str,
              boundary: str, report_interval: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    def report(step: int, t: float, dt: float, stats: dict[str, float]) -> None:
        print(
            f"weno5_z_p2 step={step:05d} t={t:.8e} dt={dt:.4e} "
            f"rho=[{stats['rho_min']:.5e},{stats['rho_max']:.5e}] "
            f"p=[{stats['p_min']:.5e},{stats['p_max']:.5e}] "
            f"nan={int(stats['nan_count'])}", flush=True,
        )

    final, dts, steps, time = W5H.run_to_time(
        initial, params, params.t_end, device, boundary,
        report_interval=report_interval,
        report=report if report_interval else None,
    )
    return final, {
        "method": "weno5_z_p2", "steps": steps, "t": time,
        "dt_min": min(dts) if dts else 0.0,
        "dt_max": max(dts) if dts else 0.0,
        "dt_mean": float(np.mean(dts)) if dts else 0.0,
    }


def run_weno7(initial: np.ndarray, params: W7.Params, *, device: str,
              boundary: str, report_interval: int = 0,
              weight_kind: int = 1) -> tuple[np.ndarray, dict[str, Any]]:
    return W7.run_from_initial(
        initial, params, device=device, riemann_solver="hllc",
        characteristic=True, boundary=boundary,
        report_interval=report_interval, weight_kind=weight_kind,
    )


def interior(final: np.ndarray, ghost: int, nx: int, ny: int) -> np.ndarray:
    return final[ghost : ghost + ny, ghost : ghost + nx, :]


def primitive(conserved: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    rho = conserved[..., 0]
    u = conserved[..., 1] / rho
    v = conserved[..., 2] / rho
    p = (gamma - 1.0) * (
        conserved[..., 3] - 0.5 * rho * (u * u + v * v)
    )
    return np.stack((rho, u, v, p), axis=-1)
