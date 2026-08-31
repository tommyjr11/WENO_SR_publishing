"""Double-Mach boundary adapter for the trusted WENO7 Shu-RK4 paths.

The spatial reconstruction, HLLC flux, and Runge--Kutta kernels are imported
unchanged. This module only supplies time-dependent Double-Mach ghost cells
at every RHS evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from teacherfree_lab_weno7_rk4_distance_balanced_fast.warp_sod import (
    point_rk4 as base,
)
from teacherfree_lab_weno7_rk4_distance_balanced_fast.warp_sod import (
    point_rk4_mlp as mlp,
)
from teacherfree_lab_weno7_rk4_distance_balanced_fast.warp_sod import (
    warp_weno7_ader4_helpers_classical_only as wh,
)


wp = wh.wp
Params = base.Params
TorchWeno7PointBeta = mlp.TorchWeno7PointBeta


if wp is not None:

    @wp.func
    def double_mach_exact_conserved(
        x: wp.float64,
        y: wp.float64,
        t: wp.float64,
        gamma: wp.float64,
    ) -> wp.vec4d:
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


def _apply_boundary(
    src: object,
    params: Params,
    time: float,
    device: str,
) -> None:
    nx_total = params.nx + 2 * params.ghost
    ny_total = params.ny + 2 * params.ghost
    wp.launch(
        wh.apply_boundary_kernel,
        dim=(ny_total, nx_total),
        inputs=[src, params.nx, params.ny, params.ghost],
        device=device,
    )
    wp.launch(
        apply_double_mach_boundary_kernel,
        dim=(ny_total, nx_total),
        inputs=[
            src,
            params.nx,
            params.ny,
            params.ghost,
            wp.float64(params.dx),
            wp.float64(params.dy),
            wp.float64(time),
            wp.float64(params.gamma),
        ],
        device=device,
    )


def _finish_rhs(
    arrays: dict[str, object],
    params: Params,
    rhs_name: str,
    device: str,
) -> None:
    wp.launch(
        base.rhs_from_flux_kernel,
        dim=(params.ny, params.nx),
        inputs=[
            arrays[rhs_name],
            arrays["fx1"],
            arrays["fx2"],
            arrays["fy1"],
            arrays["fy2"],
            params.nx,
            params.ny,
            params.ghost,
            wp.float64(1.0 / params.dx),
            wp.float64(1.0 / params.dy),
        ],
        device=device,
    )


def compute_rhs_classical(
    arrays: dict[str, object],
    params: Params,
    src_name: str,
    rhs_name: str,
    dt: float,
    stage_time: float,
    device: str,
    *,
    reverse_upwind: bool,
) -> None:
    nx = params.nx
    ny = params.ny
    src = arrays[src_name]
    _apply_boundary(src, params, stage_time, device)
    gamma = wp.float64(params.gamma)
    reverse_i = 1 if reverse_upwind else 0

    wp.launch(
        base.compute_x_stage_point_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[
            src,
            arrays["x_l"],
            arrays["x_r"],
            nx,
            ny,
            wp.float64(params.dx),
            gamma,
            1,
        ],
        device=device,
    )
    for loca, flux_name in ((1, "fx1"), (2, "fx2")):
        wp.launch(
            base.compute_x_flux_point_kernel,
            dim=(ny, nx + 1),
            inputs=[
                arrays[flux_name],
                arrays["x_l"],
                arrays["x_r"],
                wp.float64(dt / params.dx),
                nx,
                ny,
                loca,
                gamma,
                1,
                reverse_i,
            ],
            device=device,
        )

    wp.launch(
        base.compute_y_stage_point_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[
            src,
            arrays["y_l"],
            arrays["y_r"],
            nx,
            ny,
            wp.float64(params.dy),
            gamma,
            1,
        ],
        device=device,
    )
    for loca, flux_name in ((1, "fy1"), (2, "fy2")):
        wp.launch(
            base.compute_y_flux_point_kernel,
            dim=(ny + 1, nx),
            inputs=[
                arrays[flux_name],
                arrays["y_l"],
                arrays["y_r"],
                wp.float64(dt / params.dy),
                nx,
                ny,
                loca,
                gamma,
                1,
                reverse_i,
            ],
            device=device,
        )
    _finish_rhs(arrays, params, rhs_name, device)


def compute_rhs_mlp(
    arrays: dict[str, object],
    params: Params,
    src_name: str,
    rhs_name: str,
    dt: float,
    stage_time: float,
    device: str,
    *,
    beta_model: TorchWeno7PointBeta,
    reverse_upwind: bool,
) -> None:
    nx = params.nx
    ny = params.ny
    src = arrays[src_name]
    _apply_boundary(src, params, stage_time, device)
    wp.synchronize()

    gamma = wp.float64(params.gamma)
    reverse_i = 1 if reverse_upwind else 0
    beta_model.fill_normal_x_for(arrays, params, src_name)
    wp.launch(
        mlp.compute_x_stage_point_mlp_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[
            src,
            arrays["beta_x"],
            arrays["x_l"],
            arrays["x_r"],
            nx,
            ny,
            wp.float64(params.dx),
            gamma,
            0,
        ],
        device=device,
    )
    wp.synchronize()
    beta_model.fill_cross_x_point(arrays, params)
    for loca, flux_name in ((1, "fx1"), (2, "fx2")):
        wp.launch(
            mlp.compute_x_flux_point_mlp_kernel,
            dim=(ny, nx + 1),
            inputs=[
                arrays[flux_name],
                arrays["x_l"],
                arrays["x_r"],
                arrays["beta_cross_x"],
                wp.float64(dt / params.dx),
                nx,
                ny,
                loca,
                gamma,
                1,
                reverse_i,
                0,
            ],
            device=device,
        )

    beta_model.fill_normal_y_for(arrays, params, src_name)
    wp.launch(
        mlp.compute_y_stage_point_mlp_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[
            src,
            arrays["beta_y"],
            arrays["y_l"],
            arrays["y_r"],
            nx,
            ny,
            wp.float64(params.dy),
            gamma,
            0,
        ],
        device=device,
    )
    wp.synchronize()
    beta_model.fill_cross_y_point(arrays, params)
    for loca, flux_name in ((1, "fy1"), (2, "fy2")):
        wp.launch(
            mlp.compute_y_flux_point_mlp_kernel,
            dim=(ny + 1, nx),
            inputs=[
                arrays[flux_name],
                arrays["y_l"],
                arrays["y_r"],
                arrays["beta_cross_y"],
                wp.float64(dt / params.dy),
                nx,
                ny,
                loca,
                gamma,
                1,
                reverse_i,
                0,
            ],
            device=device,
        )
    _finish_rhs(arrays, params, rhs_name, device)


def _launch_shu_rk4_step(
    arrays: dict[str, object],
    params: Params,
    dt: float,
    time: float,
    device: str,
    beta_model: TorchWeno7PointBeta | None,
) -> None:
    n0, n1, _ = params.padded_shape
    rhs = compute_rhs_classical if beta_model is None else compute_rhs_mlp
    common = {} if beta_model is None else {"beta_model": beta_model}

    rhs(
        arrays, params, "u", "rhs0", dt, time, device,
        reverse_upwind=False, **common,
    )
    wp.launch(
        base.shu_stage1_kernel,
        dim=(n0, n1),
        inputs=[
            arrays["u"], arrays["rhs0"], arrays["u1"],
            n0, n1, wp.float64(dt),
        ],
        device=device,
    )

    rhs(
        arrays, params, "u", "rhs_t0", dt, time, device,
        reverse_upwind=True, **common,
    )
    rhs(
        arrays, params, "u1", "rhs1", dt, time + 0.5 * dt, device,
        reverse_upwind=False, **common,
    )
    wp.launch(
        base.shu_stage2_kernel,
        dim=(n0, n1),
        inputs=[
            arrays["u"], arrays["rhs_t0"], arrays["u1"], arrays["rhs1"],
            arrays["u2"], n0, n1, wp.float64(dt),
        ],
        device=device,
    )

    rhs(
        arrays, params, "u1", "rhs_t1", dt, time + 0.5 * dt, device,
        reverse_upwind=True, **common,
    )
    rhs(
        arrays, params, "u2", "rhs2", dt, time + 0.5 * dt, device,
        reverse_upwind=False, **common,
    )
    wp.launch(
        base.shu_stage3_kernel,
        dim=(n0, n1),
        inputs=[
            arrays["u"], arrays["rhs_t0"], arrays["u1"],
            arrays["rhs_t1"], arrays["u2"], arrays["rhs2"],
            arrays["u3"], n0, n1, wp.float64(dt),
        ],
        device=device,
    )

    rhs(
        arrays, params, "u3", "rhs3", dt, time + dt, device,
        reverse_upwind=False, **common,
    )
    wp.launch(
        base.shu_final_kernel,
        dim=(n0, n1),
        inputs=[
            arrays["u"], arrays["rhs0"], arrays["u1"], arrays["rhs1"],
            arrays["u2"], arrays["u3"], arrays["rhs3"], arrays["u"],
            n0, n1, wp.float64(dt),
        ],
        device=device,
    )


def run_to_time(
    initial: np.ndarray,
    params: Params,
    t_end: float,
    device: str,
    beta_model: TorchWeno7PointBeta | None,
    *,
    report_interval: int = 100,
    report: Callable[[int, float, float, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[float], int, float]:
    wh.require_warp()
    wp.init()
    wp.set_device(device)
    arrays = (
        base.allocate_arrays(initial, params, device)
        if beta_model is None
        else mlp.allocate_arrays_mlp(initial, params, device)
    )
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

        _launch_shu_rk4_step(arrays, params, dt, t, device, beta_model)
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
