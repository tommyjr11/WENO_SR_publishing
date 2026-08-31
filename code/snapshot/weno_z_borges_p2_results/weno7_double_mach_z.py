"""WENO7-Z adapter for the trusted time-dependent Double-Mach RK4 path."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from for_paper_results.solvers import weno7_double_mach as trusted

from . import weno7_helpers_z as wh
from . import weno7_point_rk4_z as base


# The trusted adapter contains only boundary and RK-stage orchestration.  Its
# module globals are replaced by the isolated WENO-Z spatial implementation.
trusted.base = base
trusted.wh = wh

Params = base.Params


def _compute_rhs_classical(
    arrays: dict[str, object],
    params: Params,
    src_name: str,
    rhs_name: str,
    dt: float,
    stage_time: float,
    device: str,
    *,
    reverse_upwind: bool,
    weight_kind: int,
) -> None:
    """Trusted Double-Mach boundary path with an explicit Z exponent mode."""
    nx = params.nx
    ny = params.ny
    src = arrays[src_name]
    trusted._apply_boundary(src, params, stage_time, device)
    gamma = wh.wp.float64(params.gamma)
    reverse_i = 1 if reverse_upwind else 0

    wh.wp.launch(
        base.compute_x_stage_point_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[
            src,
            arrays["x_l"],
            arrays["x_r"],
            nx,
            ny,
            wh.wp.float64(params.dx),
            gamma,
            1,
            weight_kind,
        ],
        device=device,
    )
    for loca, flux_name in ((1, "fx1"), (2, "fx2")):
        wh.wp.launch(
            base.compute_x_flux_point_kernel,
            dim=(ny, nx + 1),
            inputs=[
                arrays[flux_name],
                arrays["x_l"],
                arrays["x_r"],
                wh.wp.float64(dt / params.dx),
                wh.wp.float64(params.dy),
                nx,
                ny,
                loca,
                gamma,
                1,
                reverse_i,
                weight_kind,
            ],
            device=device,
        )

    wh.wp.launch(
        base.compute_y_stage_point_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[
            src,
            arrays["y_l"],
            arrays["y_r"],
            nx,
            ny,
            wh.wp.float64(params.dy),
            gamma,
            1,
            weight_kind,
        ],
        device=device,
    )
    for loca, flux_name in ((1, "fy1"), (2, "fy2")):
        wh.wp.launch(
            base.compute_y_flux_point_kernel,
            dim=(ny + 1, nx),
            inputs=[
                arrays[flux_name],
                arrays["y_l"],
                arrays["y_r"],
                wh.wp.float64(dt / params.dy),
                wh.wp.float64(params.dx),
                nx,
                ny,
                loca,
                gamma,
                1,
                reverse_i,
                weight_kind,
            ],
            device=device,
        )
    trusted._finish_rhs(arrays, params, rhs_name, device)


def _launch_shu_rk4_step(
    arrays: dict[str, object],
    params: Params,
    dt: float,
    time: float,
    device: str,
    weight_kind: int,
) -> None:
    n0, n1, _ = params.padded_shape

    def rhs(src_name: str, rhs_name: str, stage_time: float, reverse: bool) -> None:
        _compute_rhs_classical(
            arrays,
            params,
            src_name,
            rhs_name,
            dt,
            stage_time,
            device,
            reverse_upwind=reverse,
            weight_kind=weight_kind,
        )

    rhs("u", "rhs0", time, False)
    wh.wp.launch(
        base.shu_stage1_kernel,
        dim=(n0, n1),
        inputs=[arrays["u"], arrays["rhs0"], arrays["u1"], n0, n1, wh.wp.float64(dt)],
        device=device,
    )
    rhs("u", "rhs_t0", time, True)
    rhs("u1", "rhs1", time + 0.5 * dt, False)
    wh.wp.launch(
        base.shu_stage2_kernel,
        dim=(n0, n1),
        inputs=[
            arrays["u"], arrays["rhs_t0"], arrays["u1"], arrays["rhs1"],
            arrays["u2"], n0, n1, wh.wp.float64(dt),
        ],
        device=device,
    )
    rhs("u1", "rhs_t1", time + 0.5 * dt, True)
    rhs("u2", "rhs2", time + 0.5 * dt, False)
    wh.wp.launch(
        base.shu_stage3_kernel,
        dim=(n0, n1),
        inputs=[
            arrays["u"], arrays["rhs_t0"], arrays["u1"], arrays["rhs_t1"],
            arrays["u2"], arrays["rhs2"], arrays["u3"], n0, n1,
            wh.wp.float64(dt),
        ],
        device=device,
    )
    rhs("u3", "rhs3", time + dt, False)
    wh.wp.launch(
        base.shu_final_kernel,
        dim=(n0, n1),
        inputs=[
            arrays["u"], arrays["rhs0"], arrays["u1"], arrays["rhs1"],
            arrays["u2"], arrays["u3"], arrays["rhs3"], arrays["u"],
            n0, n1, wh.wp.float64(dt),
        ],
        device=device,
    )


def run_to_time(
    initial: np.ndarray,
    params: Params,
    t_end: float,
    device: str,
    *,
    weight_kind: int = 1,
    report_interval: int = 100,
    report: Callable[[int, float, float, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[float], int, float]:
    wh.require_warp()
    wh.wp.init()
    wh.wp.set_device(device)
    arrays = base.allocate_arrays(initial, params, device)
    t = 0.0
    step = 0
    dt_values: list[float] = []
    while t < t_end - 1.0e-14:
        dt_cfl = base.compute_dt_from_warp_array(
            arrays["u"], arrays["speed"], params, device
        )
        dt = min(dt_cfl, t_end - t)
        if not np.isfinite(dt) or dt <= 0.0:
            break

        _launch_shu_rk4_step(arrays, params, dt, t, device, weight_kind)
        t += dt
        step += 1
        dt_values.append(float(dt))

        should_report = (
            step == 1
            or (report_interval > 0 and step % report_interval == 0)
            or t >= t_end - 1.0e-14
        )
        if should_report:
            stats = base.interior_stats(arrays["u"].numpy(), params)
            if report is not None:
                report(step, t, dt, stats)
            if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
                break

    return arrays["u"].numpy(), dt_values, step, t
