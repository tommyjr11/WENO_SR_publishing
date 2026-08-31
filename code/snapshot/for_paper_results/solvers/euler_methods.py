"""Unified adapters for the five Euler methods used in the paper."""
from __future__ import annotations

from typing import Any

import numpy as np

import warp_weno5_helpers as wh5
from run_weno5_circle_mlp_compare import load_mlp_params as load_weno5_f64
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32 import warp_weno5_helpers_mlp_f32 as wh5mixed
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_circle_mlp_compare_mlp_f32 import (
    load_mlp_params as load_weno5_f32,
)
from weno7_point_rk4_shu import point_rk4 as weno7_base
from teacherfree_lab_weno7_rk4_distance_balanced_fast.warp_sod import (
    point_rk4_mlp as weno7_mlp,
)
from teacherfree_lab_weno7_rk4_distance_balanced_fast.warp_sod.point_rk4_mlp import (
    TorchWeno7PointBeta,
)

from for_paper_results import config
from for_paper_results.solvers import weno5_hllc, weno5_hllc_mixed
from teacherfree_lab_weno5_v20_distance_balanced import weno5_hllc_refsym


def make_weno5_params(nx: int, ny: int, x_length: float, y_length: float,
                      cfl: float, t_end: float, mixed: bool = False):
    cls = wh5mixed.Params if mixed else wh5.Params
    return cls(nx=nx, ny=ny, x_length=x_length, y_length=y_length, cfl=cfl, t_end=t_end)


def make_weno7_params(nx: int, ny: int, x_min: float, x_max: float,
                      y_min: float, y_max: float, cfl: float, t_end: float):
    return weno7_base.Params(
        nx=nx, ny=ny, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, cfl=cfl, t_end=t_end,
    )


def run_weno5(method_key: str, initial: np.ndarray, params, *, device: str,
              boundary: str, report_interval: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    method = config.METHODS[method_key]
    if method_key == "weno5_js":
        adapter = weno5_hllc
        mlp_params = None
    elif method_key == "weno5_sr_f64":
        adapter = weno5_hllc_refsym
        mlp_params = load_weno5_f64(method.model, device)
    elif method_key == "weno5_sr_f32":
        adapter = weno5_hllc_mixed
        mlp_params = load_weno5_f32(method.model, device)
    else:
        raise ValueError(f"not a WENO5 paper method: {method_key}")

    def report(step: int, t: float, dt: float, stats: dict[str, float]) -> None:
        print(
            f"{method_key} step={step:05d} t={t:.8e} dt={dt:.4e} "
            f"rho=[{stats['rho_min']:.5e},{stats['rho_max']:.5e}] "
            f"p=[{stats['p_min']:.5e},{stats['p_max']:.5e}] nan={int(stats['nan_count'])}",
            flush=True,
        )

    final, dts, steps, t = adapter.run_to_time(
        initial, params, params.t_end, device, mlp_params, boundary,
        report_interval=report_interval, report=report if report_interval else None,
    )
    return final, {
        "method": method_key,
        "riemann_solver": "hllc",
        "weno_space": "characteristic",
        "eno_cutoff": False,
        "steps": steps,
        "t": t,
        "dt_min": min(dts) if dts else 0.0,
        "dt_max": max(dts) if dts else 0.0,
        "dt_mean": float(np.mean(dts)) if dts else 0.0,
    }


def run_weno7(method_key: str, initial: np.ndarray, params: weno7_base.Params, *,
              device: str, boundary: str, report_interval: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    if method_key == "weno7_js":
        return weno7_base.run_from_initial(
            initial, params, device=device, riemann_solver="hllc", characteristic=True,
            boundary=boundary, report_interval=report_interval,
        )
    if method_key == "weno7_sr_f64":
        beta = TorchWeno7PointBeta(
            config.METHODS[method_key].model, device, params.gamma
        )
        return weno7_mlp.run_from_initial_mlp(
            initial, params, device=device, riemann_solver="hllc", beta_model=beta,
            boundary=boundary, eno_cutoff=False, report_interval=report_interval,
        )
    raise ValueError(f"not a WENO7 paper method: {method_key}")
