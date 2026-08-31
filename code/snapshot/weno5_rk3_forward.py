#!/usr/bin/env python3
"""Lightweight forward-only WENO5/RK3 launcher.

This module reuses the already-tested WENO5/RK3 kernels, but keeps the state on
the device, reuses RK scratch arrays, and avoids all training/Tape/loss arrays.
It is intended for fast visualization and validation runs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import warp_weno5_helpers as wh
from weno5_rk3_diff import allocate_stage, launch_weno5_rk3_diff_step, zeros_like_state


wp = wh.wp


def allocate_forward_arrays(u0_host: np.ndarray, params: wh.Params, device: str) -> dict[str, object]:
    """Allocate reusable forward buffers with gradients disabled."""
    return {
        "u0": wp.array(u0_host, dtype=wp.float64, device=device, requires_grad=False),
        "u1": zeros_like_state(params, device),
        "u2": zeros_like_state(params, device),
        "u3": zeros_like_state(params, device),
        "reg_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=False),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
        "s1": allocate_stage(params, device, "s1"),
        "s2": allocate_stage(params, device, "s2"),
        "s3": allocate_stage(params, device, "s3"),
    }


def advance_forward_inplace(
    arrays: dict[str, object],
    params: wh.Params,
    dt: float,
    device: str,
    characteristic_weno: bool,
    mlp_params: dict[str, object] | None,
    eno_cutoff: bool,
    boundary: str,
    riemann_solver: str = "force",
) -> None:
    """Advance one RK3 step and make arrays['u0'] point at the new state."""
    launch_weno5_rk3_diff_step(
        arrays,
        params,
        dt,
        device,
        characteristic_weno,
        mlp_params,
        eno_cutoff,
        boundary,
        riemann_solver,
    )
    arrays["u0"], arrays["u3"] = arrays["u3"], arrays["u0"]


def run_forward_to_time(
    u0_host: np.ndarray,
    params: wh.Params,
    t_end: float,
    device: str,
    characteristic_weno: bool,
    mlp_params: dict[str, object] | None,
    eno_cutoff: bool = False,
    boundary: str = "periodic",
    riemann_solver: str = "force",
    report_interval: int = 50,
    report: Callable[[int, float, float, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[float], int, float]:
    """Run a forward solve to t_end and return the final host state."""
    arrays = allocate_forward_arrays(u0_host, params, device)
    t = 0.0
    steps = 0
    dt_values: list[float] = []
    while t < t_end - 1.0e-14:
        dt_cfl = wh.compute_dt_from_warp_array(arrays["u0"], arrays["speed"], params, device)
        dt = min(dt_cfl, t_end - t)
        advance_forward_inplace(
            arrays,
            params,
            dt,
            device,
            characteristic_weno,
            mlp_params,
            eno_cutoff,
            boundary,
            riemann_solver,
        )
        t += dt
        steps += 1
        dt_values.append(dt)

        should_report = steps == 1 or (report_interval > 0 and steps % report_interval == 0) or t >= t_end - 1.0e-14
        if should_report:
            state_host = arrays["u0"].numpy()
            stats = wh.interior_stats(state_host, params)
            if report is not None:
                report(steps, t, dt, stats)
            if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
                break

    final_host = arrays["u0"].numpy()
    return final_host, dt_values, steps, t
