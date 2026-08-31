"""Isolated HLLC adapter for FP32-MLP/FP64-state WENO5 Warp inference."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import weno5_rk3_warp as trusted_weno5

from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32 import warp_weno5_helpers_mlp_f32 as wh
from teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast.warp_mlp_f32 import (
    weno5_rk3_diff_mlp_f32 as core,
)
from for_paper_results.solvers.weno5_hllc import hllc_flux


wp = wh.wp


if wp is not None:

    @wp.kernel
    def compute_x_flux_mlp_hllc_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        reg_loss: wp.array(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float32), b1: wp.array2d(dtype=wp.float32),
        w2: wp.array3d(dtype=wp.float32), b2: wp.array2d(dtype=wp.float32),
        w3: wp.array3d(dtype=wp.float32), b3: wp.array2d(dtype=wp.float32),
        w4: wp.array3d(dtype=wp.float32), b4: wp.array2d(dtype=wp.float32),
        dt_dx: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        reg_norm: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            ul = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            ur = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            penalty = wp.float64(0.0)
            for comp in range(4):
                right = core.weno5_gauss_value_mlp(
                    temp_l[j + 1, i + 1, comp], temp_l[j + 2, i + 1, comp],
                    temp_l[j + 3, i + 1, comp], temp_l[j + 4, i + 1, comp],
                    temp_l[j + 5, i + 1, comp], loca,
                    w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
                )
                left = core.weno5_gauss_value_mlp(
                    temp_r[j + 1, i, comp], temp_r[j + 2, i, comp],
                    temp_r[j + 3, i, comp], temp_r[j + 4, i, comp],
                    temp_r[j + 5, i, comp], loca,
                    w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
                )
                ur[comp] = right[0]
                ul[comp] = left[0]
                penalty = penalty + right[1] + left[1]
            f = hllc_flux(ul, ur, 1, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = f[comp] * dt_dx
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


    @wp.kernel
    def compute_y_flux_mlp_hllc_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        reg_loss: wp.array(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float32), b1: wp.array2d(dtype=wp.float32),
        w2: wp.array3d(dtype=wp.float32), b2: wp.array2d(dtype=wp.float32),
        w3: wp.array3d(dtype=wp.float32), b3: wp.array2d(dtype=wp.float32),
        w4: wp.array3d(dtype=wp.float32), b4: wp.array2d(dtype=wp.float32),
        dt_dy: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        reg_norm: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            ul = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            ur = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            penalty = wp.float64(0.0)
            for comp in range(4):
                right = core.weno5_gauss_value_mlp(
                    temp_l[j + 1, i + 1, comp], temp_l[j + 1, i + 2, comp],
                    temp_l[j + 1, i + 3, comp], temp_l[j + 1, i + 4, comp],
                    temp_l[j + 1, i + 5, comp], loca,
                    w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
                )
                left = core.weno5_gauss_value_mlp(
                    temp_r[j, i + 1, comp], temp_r[j, i + 2, comp],
                    temp_r[j, i + 3, comp], temp_r[j, i + 4, comp],
                    temp_r[j, i + 5, comp], loca,
                    w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
                )
                ur[comp] = right[0]
                ul[comp] = left[0]
                penalty = penalty + right[1] + left[1]
            f = hllc_flux(ul, ur, 2, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = f[comp] * dt_dy
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


def _normal_inputs(stage: dict[str, object], params: wh.Params,
                   mlp_params: dict[str, object], reg_norm: float, axis: str) -> list[object]:
    return [
        stage["bc"], stage[f"{axis}_l"], stage[f"{axis}_r"], stage["reg_loss"],
        mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"],
        mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
        params.nx, params.ny, wp.float64(reg_norm), 0, wp.float64(params.gamma),
    ]


def launch_stage(u_stage: object, u0: object, u_out: object, stage: dict[str, object],
                 params: wh.Params, dt: float, rk: int, device: str,
                 mlp_params: dict[str, object], boundary: str,
                 boundary_time: float = 0.0) -> None:
    nx, ny, gc = params.nx, params.ny, params.ghost
    nx_total, ny_total = nx + 2 * gc, ny + 2 * gc
    gamma = wp.float64(params.gamma)
    dt_dx = wp.float64(dt / params.dx)
    dt_dy = wp.float64(dt / params.dy)
    reg_norm = 1.0 / max(1.0, 3.0 * 32.0 * float(nx * ny))
    if boundary == "double-mach":
        wp.launch(core.copy_transmissive_boundary_kernel, dim=(ny_total, nx_total),
                  inputs=[u_stage, stage["bc"], nx, ny, gc], device=device)
        wp.launch(
            trusted_weno5.apply_double_mach_boundary_kernel,
            dim=(ny_total, nx_total),
            inputs=[stage["bc"], nx, ny, gc, wp.float64(params.dx),
                    wp.float64(params.dy), wp.float64(boundary_time), gamma],
            device=device,
        )
    elif boundary == "transmissive":
        boundary_kernel = core.copy_transmissive_boundary_kernel
        wp.launch(boundary_kernel, dim=(ny_total, nx_total),
                  inputs=[u_stage, stage["bc"], nx, ny, gc], device=device)
    elif boundary == "periodic":
        boundary_kernel = core.copy_periodic_boundary_kernel
        wp.launch(boundary_kernel, dim=(ny_total, nx_total),
                  inputs=[u_stage, stage["bc"], nx, ny, gc], device=device)
    else:
        raise ValueError(f"unsupported boundary {boundary!r}")
    wp.launch(core.compute_x_stage_weno_mlp_characteristic_kernel, dim=(ny + 6, nx + 2),
              inputs=_normal_inputs(stage, params, mlp_params, reg_norm, "x"), device=device)
    wp.launch(core.compute_y_stage_weno_mlp_characteristic_kernel, dim=(ny + 2, nx + 6),
              inputs=_normal_inputs(stage, params, mlp_params, reg_norm, "y"), device=device)
    base = [stage["reg_loss"], mlp_params["w1"], mlp_params["b1"],
            mlp_params["w2"], mlp_params["b2"], mlp_params["w3"], mlp_params["b3"],
            mlp_params["w4"], mlp_params["b4"]]
    for name, loca in (("fx1", 1), ("fx2", 2)):
        wp.launch(compute_x_flux_mlp_hllc_kernel, dim=(ny, nx + 1),
                  inputs=[stage[name], stage["x_l"], stage["x_r"], *base, dt_dx,
                          nx, ny, loca, gamma, wp.float64(reg_norm), 0], device=device)
    for name, loca in (("fy1", 1), ("fy2", 2)):
        wp.launch(compute_y_flux_mlp_hllc_kernel, dim=(ny + 1, nx),
                  inputs=[stage[name], stage["y_l"], stage["y_r"], *base, dt_dy,
                          nx, ny, loca, gamma, wp.float64(reg_norm), 0], device=device)
    wp.launch(core.update_rk3_out_kernel, dim=(ny, nx),
              inputs=[stage["bc"], u0, u_out, stage["fx1"], stage["fx2"],
                      stage["fy1"], stage["fy2"], nx, ny, gc, rk], device=device)


def allocate_forward(u0: np.ndarray, params: wh.Params, device: str) -> dict[str, object]:
    arrays = {
        "u0": wp.array(u0, dtype=wp.float64, device=device),
        "u1": wp.zeros(params.padded_shape, dtype=wp.float64, device=device),
        "u2": wp.zeros(params.padded_shape, dtype=wp.float64, device=device),
        "u3": wp.zeros(params.padded_shape, dtype=wp.float64, device=device),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
    }
    for name in ("s1", "s2", "s3"):
        arrays[name] = core.allocate_stage(params, device, name)
        arrays[name]["reg_loss"] = wp.zeros(1, dtype=wp.float64, device=device)
    return arrays


def run_to_time(
    u0: np.ndarray,
    params: wh.Params,
    t_end: float,
    device: str,
    mlp_params: dict[str, object],
    boundary: str,
    report_interval: int = 0,
    report: Callable[[int, float, float, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[float], int, float]:
    arrays = allocate_forward(u0, params, device)
    t = 0.0
    step = 0
    dts: list[float] = []
    while t < t_end - 1.0e-14:
        dt = min(wh.compute_dt_from_warp_array(arrays["u0"], arrays["speed"], params, device), t_end - t)
        if not np.isfinite(dt) or dt <= 0.0:
            break
        launch_stage(arrays["u0"], arrays["u0"], arrays["u1"], arrays["s1"], params, dt, 1, device, mlp_params, boundary, t)
        launch_stage(arrays["u1"], arrays["u0"], arrays["u2"], arrays["s2"], params, dt, 2, device, mlp_params, boundary, t + dt)
        launch_stage(arrays["u2"], arrays["u0"], arrays["u3"], arrays["s3"], params, dt, 3, device, mlp_params, boundary, t + 0.5 * dt)
        arrays["u0"], arrays["u3"] = arrays["u3"], arrays["u0"]
        t += dt
        step += 1
        dts.append(float(dt))
        if report is not None and (step == 1 or (report_interval and step % report_interval == 0) or t >= t_end - 1.0e-14):
            report(step, t, dt, wh.interior_stats(arrays["u0"].numpy(), params))
    return arrays["u0"].numpy(), dts, step, t
