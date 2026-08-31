"""Isolated HLLC forward adapter for the trusted FP64 WENO5 Warp solver.

Only the Riemann flux and its dispatch are new.  Reconstruction, characteristic
projection, Gaussian transverse reconstruction, RK3 stages, boundary kernels,
and timestep calculation are imported unchanged from the validated solver.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

import warp_weno5_helpers as wh
import weno5_rk3_diff as core


wp = wh.wp


if wp is not None:

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
    def hllc_equal_state_kernel(
        states: wp.array2d(dtype=wp.float64),
        fluxes: wp.array2d(dtype=wp.float64),
        direction: int,
        gamma: wp.float64,
    ):
        i = wp.tid()
        q = wp.vec4d(states[i, 0], states[i, 1], states[i, 2], states[i, 3])
        f = hllc_flux(q, q, direction, gamma)
        for c in range(4):
            fluxes[i, c] = f[c]


    @wp.kernel
    def compute_x_flux_classical_hllc_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        dt_dx: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            ul = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            ur = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                ur[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp], temp_l[j + 2, i + 1, comp],
                    temp_l[j + 3, i + 1, comp], temp_l[j + 4, i + 1, comp],
                    temp_l[j + 5, i + 1, comp], loca,
                )
                ul[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j + 1, i, comp], temp_r[j + 2, i, comp],
                    temp_r[j + 3, i, comp], temp_r[j + 4, i, comp],
                    temp_r[j + 5, i, comp], loca,
                )
            f = hllc_flux(ul, ur, 1, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = f[comp] * dt_dx


    @wp.kernel
    def compute_y_flux_classical_hllc_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        dt_dy: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            ul = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            ur = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                ur[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp], temp_l[j + 1, i + 2, comp],
                    temp_l[j + 1, i + 3, comp], temp_l[j + 1, i + 4, comp],
                    temp_l[j + 1, i + 5, comp], loca,
                )
                ul[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j, i + 1, comp], temp_r[j, i + 2, comp],
                    temp_r[j, i + 3, comp], temp_r[j, i + 4, comp],
                    temp_r[j, i + 5, comp], loca,
                )
            f = hllc_flux(ul, ur, 2, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = f[comp] * dt_dy


    @wp.kernel
    def compute_x_flux_mlp_hllc_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        reg_loss: wp.array(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64),
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
        w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64),
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


def _mlp_inputs(stage: dict[str, object], params: wh.Params, mlp_params: dict[str, object],
                reg_norm: float, eno_cutoff: bool, axis: str) -> list[object]:
    inputs = [
        stage["bc"], stage[f"{axis}_l"], stage[f"{axis}_r"], stage["reg_loss"],
        mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"],
        mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
        params.nx, params.ny, wp.float64(reg_norm), 1 if eno_cutoff else 0,
        wp.float64(params.gamma),
    ]
    return inputs


def launch_stage(
    u_stage: object,
    u0: object,
    u_out: object,
    stage: dict[str, object],
    params: wh.Params,
    dt: float,
    rk: int,
    device: str,
    mlp_params: dict[str, object] | None,
    boundary: str,
    eno_cutoff: bool = False,
) -> None:
    nx, ny, gc = params.nx, params.ny, params.ghost
    nx_total, ny_total = nx + 2 * gc, ny + 2 * gc
    dt_dx = wp.float64(dt / params.dx)
    dt_dy = wp.float64(dt / params.dy)
    gamma = wp.float64(params.gamma)
    reg_norm = 1.0 / max(1.0, 3.0 * 32.0 * float(nx * ny))
    stage["reg_loss"] = stage.get("reg_loss", wp.zeros(1, dtype=wp.float64, device=device))

    if boundary == "transmissive":
        boundary_kernel = core.copy_transmissive_boundary_kernel
    elif boundary == "periodic":
        boundary_kernel = core.copy_periodic_boundary_kernel
    else:
        raise ValueError(f"unsupported boundary {boundary!r}")
    wp.launch(boundary_kernel, dim=(ny_total, nx_total),
              inputs=[u_stage, stage["bc"], nx, ny, gc], device=device)

    if mlp_params is None:
        wp.launch(wh.compute_x_stage_weno_kernel, dim=(ny + 6, nx + 2),
                  inputs=[stage["bc"], stage["x_l"], stage["x_r"], nx, ny, 1, gamma], device=device)
        wp.launch(compute_x_flux_classical_hllc_kernel, dim=(ny, nx + 1),
                  inputs=[stage["fx1"], stage["x_l"], stage["x_r"], dt_dx, nx, ny, 1, gamma], device=device)
        wp.launch(compute_x_flux_classical_hllc_kernel, dim=(ny, nx + 1),
                  inputs=[stage["fx2"], stage["x_l"], stage["x_r"], dt_dx, nx, ny, 2, gamma], device=device)
        wp.launch(wh.compute_y_stage_weno_kernel, dim=(ny + 2, nx + 6),
                  inputs=[stage["bc"], stage["y_l"], stage["y_r"], nx, ny, 1, gamma], device=device)
        wp.launch(compute_y_flux_classical_hllc_kernel, dim=(ny + 1, nx),
                  inputs=[stage["fy1"], stage["y_l"], stage["y_r"], dt_dy, nx, ny, 1, gamma], device=device)
        wp.launch(compute_y_flux_classical_hllc_kernel, dim=(ny + 1, nx),
                  inputs=[stage["fy2"], stage["y_l"], stage["y_r"], dt_dy, nx, ny, 2, gamma], device=device)
    else:
        x_inputs = _mlp_inputs(stage, params, mlp_params, reg_norm, eno_cutoff, "x")
        y_inputs = _mlp_inputs(stage, params, mlp_params, reg_norm, eno_cutoff, "y")
        wp.launch(core.compute_x_stage_weno_mlp_characteristic_kernel, dim=(ny + 6, nx + 2),
                  inputs=x_inputs, device=device)
        wp.launch(core.compute_y_stage_weno_mlp_characteristic_kernel, dim=(ny + 2, nx + 6),
                  inputs=y_inputs, device=device)
        base = [stage["reg_loss"], mlp_params["w1"], mlp_params["b1"],
                mlp_params["w2"], mlp_params["b2"], mlp_params["w3"], mlp_params["b3"],
                mlp_params["w4"], mlp_params["b4"]]
        for name, loca in (("fx1", 1), ("fx2", 2)):
            wp.launch(compute_x_flux_mlp_hllc_kernel, dim=(ny, nx + 1),
                      inputs=[stage[name], stage["x_l"], stage["x_r"], *base, dt_dx,
                              nx, ny, loca, gamma, wp.float64(reg_norm), 1 if eno_cutoff else 0], device=device)
        for name, loca in (("fy1", 1), ("fy2", 2)):
            wp.launch(compute_y_flux_mlp_hllc_kernel, dim=(ny + 1, nx),
                      inputs=[stage[name], stage["y_l"], stage["y_r"], *base, dt_dy,
                              nx, ny, loca, gamma, wp.float64(reg_norm), 1 if eno_cutoff else 0], device=device)

    wp.launch(core.update_rk3_out_kernel, dim=(ny, nx),
              inputs=[stage["bc"], u0, u_out, stage["fx1"], stage["fx2"],
                      stage["fy1"], stage["fy2"], nx, ny, gc, rk], device=device)


def allocate_forward(u0: np.ndarray, params: wh.Params, device: str) -> dict[str, object]:
    def state() -> object:
        return wp.zeros(params.padded_shape, dtype=wp.float64, device=device)
    arrays = {
        "u0": wp.array(u0, dtype=wp.float64, device=device),
        "u1": state(), "u2": state(), "u3": state(),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
    }
    for name in ("s1", "s2", "s3"):
        arrays[name] = core.allocate_stage(params, device, name)
        arrays[name]["reg_loss"] = wp.zeros(1, dtype=wp.float64, device=device)
    return arrays


def advance_step(arrays: dict[str, object], params: wh.Params, dt: float, device: str,
                 mlp_params: dict[str, object] | None, boundary: str) -> None:
    launch_stage(arrays["u0"], arrays["u0"], arrays["u1"], arrays["s1"], params, dt, 1,
                 device, mlp_params, boundary)
    launch_stage(arrays["u1"], arrays["u0"], arrays["u2"], arrays["s2"], params, dt, 2,
                 device, mlp_params, boundary)
    launch_stage(arrays["u2"], arrays["u0"], arrays["u3"], arrays["s3"], params, dt, 3,
                 device, mlp_params, boundary)
    arrays["u0"], arrays["u3"] = arrays["u3"], arrays["u0"]


def run_to_time(
    u0: np.ndarray,
    params: wh.Params,
    t_end: float,
    device: str,
    mlp_params: dict[str, object] | None,
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
        advance_step(arrays, params, dt, device, mlp_params, boundary)
        t += dt
        step += 1
        dts.append(float(dt))
        if report is not None and (step == 1 or (report_interval and step % report_interval == 0) or t >= t_end - 1.0e-14):
            report(step, t, dt, wh.interior_stats(arrays["u0"].numpy(), params))
    return arrays["u0"].numpy(), dts, step, t
