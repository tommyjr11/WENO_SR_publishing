"""HLLC forward adapter for reflection-symmetric FP64 WENO5-SR inference.

The reconstruction and RK3 kernels come from the trusted V12 symmetric path.
Only the Riemann flux dispatch and the time-dependent double-Mach boundary
adapter are supplied here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import warp_weno5_helpers as wh
from for_paper_results.solvers.weno5_hllc import hllc_flux
from teacherfree_lab_weno5_v20_distance_balanced import (
    weno5_rk3_diff_v20_deploy as core,
)


wp = wh.wp


if wp is not None:

    @wp.func
    def double_mach_exact_conserved(
        x: wp.float64,
        y: wp.float64,
        t: wp.float64,
        gamma: wp.float64,
    ) -> wp.vec4d:
        """Exact copy of the validated boundary state in weno5_rk3_warp.py."""
        root3 = wp.sqrt(wp.float64(3.0))
        x0 = wp.float64(1.0) / wp.float64(6.0)
        shock_x = x0 + y / root3 + wp.float64(20.0) * t / root3
        if x < shock_x:
            return wh.pri_to_con(
                wp.vec4d(
                    wp.float64(8.0),
                    wp.float64(8.25) * root3 * wp.float64(0.5),
                    -wp.float64(8.25) * wp.float64(0.5),
                    wp.float64(116.5),
                ),
                gamma,
            )
        return wh.pri_to_con(
            wp.vec4d(
                wp.float64(1.4),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(1.0),
            ),
            gamma,
        )


    @wp.kernel
    def apply_double_mach_boundary_kernel(
        u: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        dx: wp.float64,
        dy: wp.float64,
        time: wp.float64,
        gamma: wp.float64,
    ):
        """Exact copy of the validated boundary kernel in weno5_rk3_warp.py."""
        j, i = wp.tid()
        nx_total = nx + 2 * gc
        ny_total = ny + 2 * gc
        if i >= nx_total or j >= ny_total:
            return

        inside_x = i >= gc and i < nx + gc
        inside_y = j >= gc and j < ny + gc
        if inside_x and inside_y:
            return

        x = (wp.float64(i - gc) + wp.float64(0.5)) * dx
        y = (wp.float64(j - gc) + wp.float64(0.5)) * dy
        x0 = wp.float64(1.0) / wp.float64(6.0)

        q = wp.vec4d()
        use_exact = 0
        use_reflect = 0
        src_i = i
        src_j = j

        if j >= ny + gc:
            q = double_mach_exact_conserved(x, y, time, gamma)
            use_exact = 1
        elif j < gc:
            if x < x0:
                q = double_mach_exact_conserved(
                    x, wp.float64(0.0), time, gamma
                )
                use_exact = 1
            else:
                use_reflect = 1
                src_j = 2 * gc - 1 - j
                if i < gc:
                    src_i = gc
                elif i >= nx + gc:
                    src_i = nx + gc - 1
        elif i < gc:
            q = double_mach_exact_conserved(x, y, time, gamma)
            use_exact = 1
        elif i >= nx + gc:
            src_i = nx + gc - 1
            src_j = j

        if use_exact == 1:
            for comp in range(4):
                u[j, i, comp] = q[comp]
        elif use_reflect == 1:
            u[j, i, 0] = u[src_j, src_i, 0]
            u[j, i, 1] = u[src_j, src_i, 1]
            u[j, i, 2] = -u[src_j, src_i, 2]
            u[j, i, 3] = u[src_j, src_i, 3]
        else:
            for comp in range(4):
                u[j, i, comp] = u[src_j, src_i, comp]


    @wp.kernel(enable_backward=False)
    def compute_x_flux_mlp_hllc_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        reg_loss: wp.array(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
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
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 2, i + 1, comp],
                    temp_l[j + 3, i + 1, comp],
                    temp_l[j + 4, i + 1, comp],
                    temp_l[j + 5, i + 1, comp],
                    loca,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                    w4,
                    b4,
                    eno_cutoff,
                )
                left = core.weno5_gauss_value_mlp(
                    temp_r[j + 1, i, comp],
                    temp_r[j + 2, i, comp],
                    temp_r[j + 3, i, comp],
                    temp_r[j + 4, i, comp],
                    temp_r[j + 5, i, comp],
                    loca,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                    w4,
                    b4,
                    eno_cutoff,
                )
                ur[comp] = right[0]
                ul[comp] = left[0]
                penalty = penalty + right[1] + left[1]
            flux = hllc_flux(ul, ur, 1, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = flux[comp] * dt_dx
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_y_flux_mlp_hllc_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        reg_loss: wp.array(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
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
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 1, i + 2, comp],
                    temp_l[j + 1, i + 3, comp],
                    temp_l[j + 1, i + 4, comp],
                    temp_l[j + 1, i + 5, comp],
                    loca,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                    w4,
                    b4,
                    eno_cutoff,
                )
                left = core.weno5_gauss_value_mlp(
                    temp_r[j, i + 1, comp],
                    temp_r[j, i + 2, comp],
                    temp_r[j, i + 3, comp],
                    temp_r[j, i + 4, comp],
                    temp_r[j, i + 5, comp],
                    loca,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                    w4,
                    b4,
                    eno_cutoff,
                )
                ur[comp] = right[0]
                ul[comp] = left[0]
                penalty = penalty + right[1] + left[1]
            flux = hllc_flux(ul, ur, 2, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = flux[comp] * dt_dy
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


def _stage_inputs(
    stage: dict[str, object],
    params: wh.Params,
    mlp_params: dict[str, object],
    reg_norm: float,
    axis: str,
    eno_cutoff: bool,
) -> list[object]:
    return [
        stage["bc"],
        stage[f"{axis}_l"],
        stage[f"{axis}_r"],
        stage["reg_loss"],
        mlp_params["w1"],
        mlp_params["b1"],
        mlp_params["w2"],
        mlp_params["b2"],
        mlp_params["w3"],
        mlp_params["b3"],
        mlp_params["w4"],
        mlp_params["b4"],
        params.nx,
        params.ny,
        wp.float64(reg_norm),
        1 if eno_cutoff else 0,
        wp.float64(params.gamma),
    ]


def launch_stage(
    u_stage: object,
    u0: object,
    u_out: object,
    stage: dict[str, object],
    params: wh.Params,
    dt: float,
    rk: int,
    device: str,
    mlp_params: dict[str, object],
    boundary: str,
    boundary_time: float,
    eno_cutoff: bool,
) -> None:
    nx, ny, gc = params.nx, params.ny, params.ghost
    nx_total, ny_total = nx + 2 * gc, ny + 2 * gc
    gamma = wp.float64(params.gamma)
    dt_dx = wp.float64(dt / params.dx)
    dt_dy = wp.float64(dt / params.dy)
    reg_norm = 1.0 / max(1.0, 3.0 * 32.0 * float(nx * ny))

    if boundary == "periodic":
        wp.launch(
            core.copy_periodic_boundary_kernel,
            dim=(ny_total, nx_total),
            inputs=[u_stage, stage["bc"], nx, ny, gc],
            device=device,
        )
    elif boundary in ("transmissive", "double-mach"):
        wp.launch(
            core.copy_transmissive_boundary_kernel,
            dim=(ny_total, nx_total),
            inputs=[u_stage, stage["bc"], nx, ny, gc],
            device=device,
        )
        if boundary == "double-mach":
            wp.launch(
                apply_double_mach_boundary_kernel,
                dim=(ny_total, nx_total),
                inputs=[
                    stage["bc"],
                    nx,
                    ny,
                    gc,
                    wp.float64(params.dx),
                    wp.float64(params.dy),
                    wp.float64(boundary_time),
                    gamma,
                ],
                device=device,
            )
    else:
        raise ValueError(f"unsupported boundary {boundary!r}")

    wp.launch(
        core.compute_x_stage_weno_mlp_characteristic_kernel,
        dim=(ny + 6, nx + 2),
        inputs=_stage_inputs(
            stage, params, mlp_params, reg_norm, "x", eno_cutoff
        ),
        device=device,
    )
    wp.launch(
        core.compute_y_stage_weno_mlp_characteristic_kernel,
        dim=(ny + 2, nx + 6),
        inputs=_stage_inputs(
            stage, params, mlp_params, reg_norm, "y", eno_cutoff
        ),
        device=device,
    )

    base = [
        stage["reg_loss"],
        mlp_params["w1"],
        mlp_params["b1"],
        mlp_params["w2"],
        mlp_params["b2"],
        mlp_params["w3"],
        mlp_params["b3"],
        mlp_params["w4"],
        mlp_params["b4"],
    ]
    for name, loca in (("fx1", 1), ("fx2", 2)):
        wp.launch(
            compute_x_flux_mlp_hllc_kernel,
            dim=(ny, nx + 1),
            inputs=[
                stage[name],
                stage["x_l"],
                stage["x_r"],
                *base,
                dt_dx,
                nx,
                ny,
                loca,
                gamma,
                wp.float64(reg_norm),
                1 if eno_cutoff else 0,
            ],
            device=device,
        )
    for name, loca in (("fy1", 1), ("fy2", 2)):
        wp.launch(
            compute_y_flux_mlp_hllc_kernel,
            dim=(ny + 1, nx),
            inputs=[
                stage[name],
                stage["y_l"],
                stage["y_r"],
                *base,
                dt_dy,
                nx,
                ny,
                loca,
                gamma,
                wp.float64(reg_norm),
                1 if eno_cutoff else 0,
            ],
            device=device,
        )
    wp.launch(
        core.update_rk3_out_kernel,
        dim=(ny, nx),
        inputs=[
            stage["bc"],
            u0,
            u_out,
            stage["fx1"],
            stage["fx2"],
            stage["fy1"],
            stage["fy2"],
            nx,
            ny,
            gc,
            rk,
        ],
        device=device,
    )


def allocate_forward(
    initial: np.ndarray,
    params: wh.Params,
    device: str,
) -> dict[str, object]:
    arrays = {
        "u0": wp.array(
            initial, dtype=wp.float64, device=device, requires_grad=False
        ),
        "u1": core.zeros_like_state(params, device, requires_grad=False),
        "u2": core.zeros_like_state(params, device, requires_grad=False),
        "u3": core.zeros_like_state(params, device, requires_grad=False),
        "speed": wp.zeros(
            params.nx * params.ny, dtype=wp.float64, device=device
        ),
    }
    for name in ("s1", "s2", "s3"):
        arrays[name] = core.allocate_stage(params, device, name)
        arrays[name]["reg_loss"] = wp.zeros(
            1, dtype=wp.float64, device=device, requires_grad=False
        )
    return arrays


def run_to_time(
    initial: np.ndarray,
    params: wh.Params,
    t_end: float,
    device: str,
    mlp_params: dict[str, object],
    boundary: str,
    eno_cutoff: bool = False,
    report_interval: int = 100,
    report: Callable[[int, float, float, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[float], int, float]:
    arrays = allocate_forward(initial, params, device)
    t = 0.0
    step = 0
    dt_values: list[float] = []
    while t < t_end - 1.0e-14:
        dt_cfl = wh.compute_dt_from_warp_array(
            arrays["u0"], arrays["speed"], params, device
        )
        dt = min(dt_cfl, t_end - t)
        if not np.isfinite(dt) or dt <= 0.0:
            break

        launch_stage(
            arrays["u0"],
            arrays["u0"],
            arrays["u1"],
            arrays["s1"],
            params,
            dt,
            1,
            device,
            mlp_params,
            boundary,
            t,
            eno_cutoff,
        )
        launch_stage(
            arrays["u1"],
            arrays["u0"],
            arrays["u2"],
            arrays["s2"],
            params,
            dt,
            2,
            device,
            mlp_params,
            boundary,
            t + dt,
            eno_cutoff,
        )
        launch_stage(
            arrays["u2"],
            arrays["u0"],
            arrays["u3"],
            arrays["s3"],
            params,
            dt,
            3,
            device,
            mlp_params,
            boundary,
            t + 0.5 * dt,
            eno_cutoff,
        )
        arrays["u0"], arrays["u3"] = arrays["u3"], arrays["u0"]
        t += dt
        step += 1
        dt_values.append(float(dt))

        should_report = (
            step == 1
            or (report_interval > 0 and step % report_interval == 0)
            or t >= t_end - 1.0e-14
        )
        if should_report:
            stats = wh.interior_stats(arrays["u0"].numpy(), params)
            if report is not None:
                report(step, t, dt, stats)
            if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
                break

    return arrays["u0"].numpy(), dt_values, step, t
