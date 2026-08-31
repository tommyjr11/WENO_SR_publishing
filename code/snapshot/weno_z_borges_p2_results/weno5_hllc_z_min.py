"""Minimal FP64 WENO5-Z/HLLC/SSPRK3 forward solver.

This module intentionally contains no MLP or differentiable-training kernels.
It preserves the trusted two-dimensional characteristic reconstruction,
transverse Gaussian reconstruction, HLLC flux, boundary conditions, and RK3
update while replacing only the nonlinear WENO weights through
``weno5_helpers_z``.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import weno5_helpers_z as wh


wp = wh.wp


if wp is not None:

    @wp.kernel
    def copy_periodic_boundary_kernel(
        src: wp.array3d(dtype=wp.float64),
        dst: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
    ):
        j, i = wp.tid()
        nx_total = nx + 2 * gc
        ny_total = ny + 2 * gc
        if j < ny_total and i < nx_total:
            src_i = i
            src_j = j
            if i < gc:
                src_i = i + nx
            elif i >= nx + gc:
                src_i = i - nx
            if j < gc:
                src_j = j + ny
            elif j >= ny + gc:
                src_j = j - ny
            for comp in range(4):
                dst[j, i, comp] = src[src_j, src_i, comp]


    @wp.kernel
    def copy_transmissive_boundary_kernel(
        src: wp.array3d(dtype=wp.float64),
        dst: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
    ):
        j, i = wp.tid()
        nx_total = nx + 2 * gc
        ny_total = ny + 2 * gc
        if j < ny_total and i < nx_total:
            src_i = i
            src_j = j
            if i < gc:
                src_i = gc
            elif i >= nx + gc:
                src_i = nx + gc - 1
            if j < gc:
                src_j = gc
            elif j >= ny + gc:
                src_j = ny + gc - 1
            for comp in range(4):
                dst[j, i, comp] = src[src_j, src_i, comp]


    @wp.func
    def hllc_flux(
        ul0: wp.vec4d,
        ur0: wp.vec4d,
        direction: int,
        gamma: wp.float64,
    ) -> wp.vec4d:
        tiny = wp.float64(1.0e-16)
        wl = wh.con_to_pri(ul0, gamma)
        wr = wh.con_to_pri(ur0, gamma)
        rho_l = wp.max(wl[0], tiny)
        rho_r = wp.max(wr[0], tiny)
        p_l = wp.max(wl[3], tiny)
        p_r = wp.max(wr[3], tiny)
        a_l = wp.sqrt(gamma * p_l / rho_l)
        a_r = wp.sqrt(gamma * p_r / rho_r)
        un_l = wl[1]
        un_r = wr[1]
        if direction == 2:
            un_l = wl[2]
            un_r = wr[2]

        s_l = wp.min(un_l - a_l, un_r - a_r)
        s_r = wp.max(un_l + a_l, un_r + a_r)
        f_l = wh.pri_to_flux(wl, direction, gamma)
        f_r = wh.pri_to_flux(wr, direction, gamma)
        if wp.float64(0.0) <= s_l:
            return f_l
        if wp.float64(0.0) >= s_r:
            return f_r

        denom = rho_l * (s_l - un_l) - rho_r * (s_r - un_r)
        if wp.abs(denom) < tiny:
            if denom < wp.float64(0.0):
                denom = -tiny
            else:
                denom = tiny
        s_star = (
            p_r - p_l
            + rho_l * un_l * (s_l - un_l)
            - rho_r * un_r * (s_r - un_r)
        ) / denom

        u_side = ul0
        w_side = wl
        f_side = f_l
        s_side = s_l
        if s_star < wp.float64(0.0):
            u_side = ur0
            w_side = wr
            f_side = f_r
            s_side = s_r

        rho = wp.max(w_side[0], tiny)
        u = w_side[1]
        v = w_side[2]
        p = wp.max(w_side[3], tiny)
        un = u
        if direction == 2:
            un = v

        denom_star = s_side - s_star
        if wp.abs(denom_star) < tiny:
            if denom_star < wp.float64(0.0):
                denom_star = -tiny
            else:
                denom_star = tiny
        factor = rho * (s_side - un) / denom_star
        rho_star = factor
        mx_star = factor * s_star
        my_star = factor * v
        if direction == 2:
            mx_star = factor * u
            my_star = factor * s_star

        denom_side = s_side - un
        if wp.abs(denom_side) < tiny:
            if denom_side < wp.float64(0.0):
                denom_side = -tiny
            else:
                denom_side = tiny
        e_star = factor * (
            (u_side[3] / rho)
            + (s_star - un) * (s_star + p / (rho * denom_side))
        )
        u_star = wp.vec4d(rho_star, mx_star, my_star, e_star)
        return wp.vec4d(
            f_side[0] + s_side * (u_star[0] - u_side[0]),
            f_side[1] + s_side * (u_star[1] - u_side[1]),
            f_side[2] + s_side * (u_star[2] - u_side[2]),
            f_side[3] + s_side * (u_star[3] - u_side[3]),
        )


    @wp.kernel
    def compute_x_flux_hllc_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        dt_dx: wp.float64,
        nx: int,
        ny: int,
        h: wp.float64,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            ul = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            ur = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                ur[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 2, i + 1, comp],
                    temp_l[j + 3, i + 1, comp],
                    temp_l[j + 4, i + 1, comp],
                    temp_l[j + 5, i + 1, comp],
                    loca, h,
                )
                ul[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j + 1, i, comp],
                    temp_r[j + 2, i, comp],
                    temp_r[j + 3, i, comp],
                    temp_r[j + 4, i, comp],
                    temp_r[j + 5, i, comp],
                    loca, h,
                )
            flux = hllc_flux(ul, ur, 1, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = flux[comp] * dt_dx


    @wp.kernel
    def compute_y_flux_hllc_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        dt_dy: wp.float64,
        nx: int,
        ny: int,
        h: wp.float64,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            ul = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            ur = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                ur[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 1, i + 2, comp],
                    temp_l[j + 1, i + 3, comp],
                    temp_l[j + 1, i + 4, comp],
                    temp_l[j + 1, i + 5, comp],
                    loca, h,
                )
                ul[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j, i + 1, comp],
                    temp_r[j, i + 2, comp],
                    temp_r[j, i + 3, comp],
                    temp_r[j, i + 4, comp],
                    temp_r[j, i + 5, comp],
                    loca, h,
                )
            flux = hllc_flux(ul, ur, 2, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = flux[comp] * dt_dy


    @wp.kernel
    def update_rk3_out_kernel(
        u_stage: wp.array3d(dtype=wp.float64),
        u0: wp.array3d(dtype=wp.float64),
        u_out: wp.array3d(dtype=wp.float64),
        flux_x1: wp.array3d(dtype=wp.float64),
        flux_x2: wp.array3d(dtype=wp.float64),
        flux_y1: wp.array3d(dtype=wp.float64),
        flux_y2: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        rk: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            for comp in range(4):
                fx_r = flux_x1[j, i + 1, comp] + flux_x2[j, i + 1, comp]
                fx_l = flux_x1[j, i, comp] + flux_x2[j, i, comp]
                fy_t = flux_y1[j + 1, i, comp] + flux_y2[j + 1, i, comp]
                fy_b = flux_y1[j, i, comp] + flux_y2[j, i, comp]
                rhs = (
                    u_stage[jj, ii, comp]
                    - wp.float64(0.5) * (fx_r - fx_l)
                    - wp.float64(0.5) * (fy_t - fy_b)
                )
                if rk == 1:
                    u_out[jj, ii, comp] = rhs
                elif rk == 2:
                    u_out[jj, ii, comp] = (
                        wp.float64(0.75) * u0[jj, ii, comp]
                        + wp.float64(0.25) * rhs
                    )
                else:
                    u_out[jj, ii, comp] = (
                        (wp.float64(1.0) / wp.float64(3.0)) * u0[jj, ii, comp]
                        + (wp.float64(2.0) / wp.float64(3.0)) * rhs
                    )


def _state(params: wh.Params, device: str) -> object:
    return wp.zeros(params.padded_shape, dtype=wp.float64, device=device)


def _stage(params: wh.Params, device: str) -> dict[str, object]:
    return {
        "bc": _state(params, device),
        "x_l": _state(params, device),
        "x_r": _state(params, device),
        "y_l": _state(params, device),
        "y_r": _state(params, device),
        "fx1": _state(params, device),
        "fx2": _state(params, device),
        "fy1": _state(params, device),
        "fy2": _state(params, device),
    }


def _launch_stage(
    u_stage: object,
    u0: object,
    u_out: object,
    stage: dict[str, object],
    params: wh.Params,
    dt: float,
    rk: int,
    device: str,
    boundary: str,
) -> None:
    nx, ny, gc = params.nx, params.ny, params.ghost
    nx_total, ny_total = nx + 2 * gc, ny + 2 * gc
    dt_dx = wp.float64(dt / params.dx)
    dt_dy = wp.float64(dt / params.dy)
    gamma = wp.float64(params.gamma)

    if boundary == "transmissive":
        boundary_kernel = copy_transmissive_boundary_kernel
    elif boundary == "periodic":
        boundary_kernel = copy_periodic_boundary_kernel
    else:
        raise ValueError(f"unsupported boundary {boundary!r}")
    wp.launch(
        boundary_kernel,
        dim=(ny_total, nx_total),
        inputs=[u_stage, stage["bc"], nx, ny, gc],
        device=device,
    )

    wp.launch(
        wh.compute_x_stage_weno_kernel,
        dim=(ny + 6, nx + 2),
        inputs=[stage["bc"], stage["x_l"], stage["x_r"], nx, ny, wp.float64(params.dx), 1, gamma],
        device=device,
    )
    for name, loca in (("fx1", 1), ("fx2", 2)):
        wp.launch(
            compute_x_flux_hllc_kernel,
            dim=(ny, nx + 1),
            inputs=[stage[name], stage["x_l"], stage["x_r"], dt_dx, nx, ny, wp.float64(params.dy), loca, gamma],
            device=device,
        )

    wp.launch(
        wh.compute_y_stage_weno_kernel,
        dim=(ny + 2, nx + 6),
        inputs=[stage["bc"], stage["y_l"], stage["y_r"], nx, ny, wp.float64(params.dy), 1, gamma],
        device=device,
    )
    for name, loca in (("fy1", 1), ("fy2", 2)):
        wp.launch(
            compute_y_flux_hllc_kernel,
            dim=(ny + 1, nx),
            inputs=[stage[name], stage["y_l"], stage["y_r"], dt_dy, nx, ny, wp.float64(params.dx), loca, gamma],
            device=device,
        )

    wp.launch(
        update_rk3_out_kernel,
        dim=(ny, nx),
        inputs=[
            stage["bc"], u0, u_out,
            stage["fx1"], stage["fx2"], stage["fy1"], stage["fy2"],
            nx, ny, gc, rk,
        ],
        device=device,
    )


def run_to_time(
    u0: np.ndarray,
    params: wh.Params,
    t_end: float,
    device: str,
    boundary: str,
    report_interval: int = 0,
    report: Callable[[int, float, float, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[float], int, float]:
    arrays = {
        "u0": wp.array(u0, dtype=wp.float64, device=device),
        "u1": _state(params, device),
        "u2": _state(params, device),
        "u3": _state(params, device),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
        "s1": _stage(params, device),
        "s2": _stage(params, device),
        "s3": _stage(params, device),
    }
    time = 0.0
    step = 0
    dts: list[float] = []
    while time < t_end - 1.0e-14:
        dt = min(
            wh.compute_dt_from_warp_array(arrays["u0"], arrays["speed"], params, device),
            t_end - time,
        )
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError(f"invalid timestep dt={dt} at step={step} time={time}")
        _launch_stage(arrays["u0"], arrays["u0"], arrays["u1"], arrays["s1"], params, dt, 1, device, boundary)
        _launch_stage(arrays["u1"], arrays["u0"], arrays["u2"], arrays["s2"], params, dt, 2, device, boundary)
        _launch_stage(arrays["u2"], arrays["u0"], arrays["u3"], arrays["s3"], params, dt, 3, device, boundary)
        arrays["u0"], arrays["u3"] = arrays["u3"], arrays["u0"]
        time += dt
        step += 1
        dts.append(float(dt))
        if report is not None and (
            step == 1
            or (report_interval and step % report_interval == 0)
            or time >= t_end - 1.0e-14
        ):
            report(step, time, dt, wh.interior_stats(arrays["u0"].numpy(), params))
    return arrays["u0"].numpy(), dts, step, time
