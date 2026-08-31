"""Time-dependent Double-Mach boundary adapter for WENO5-Z/HLLC/RK3."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import weno5_helpers_z as wh
from . import weno5_hllc_z_min as base


wp = wh.wp
Params = wh.Params


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
                wp.float64(1.4), wp.float64(0.0),
                wp.float64(0.0), wp.float64(1.0),
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
                q = double_mach_exact_conserved(x, wp.float64(0.0), time, gamma)
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


def _launch_stage(
    u_stage: object,
    u0: object,
    u_out: object,
    stage: dict[str, object],
    params: Params,
    dt: float,
    rk: int,
    device: str,
    boundary_time: float,
) -> None:
    nx, ny, gc = params.nx, params.ny, params.ghost
    nx_total, ny_total = nx + 2 * gc, ny + 2 * gc
    dt_dx = wp.float64(dt / params.dx)
    dt_dy = wp.float64(dt / params.dy)
    gamma = wp.float64(params.gamma)

    wp.launch(
        base.copy_transmissive_boundary_kernel,
        dim=(ny_total, nx_total),
        inputs=[u_stage, stage["bc"], nx, ny, gc],
        device=device,
    )
    wp.launch(
        apply_double_mach_boundary_kernel,
        dim=(ny_total, nx_total),
        inputs=[
            stage["bc"], nx, ny, gc, wp.float64(params.dx),
            wp.float64(params.dy), wp.float64(boundary_time), gamma,
        ],
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
            base.compute_x_flux_hllc_kernel,
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
            base.compute_y_flux_hllc_kernel,
            dim=(ny + 1, nx),
            inputs=[stage[name], stage["y_l"], stage["y_r"], dt_dy, nx, ny, wp.float64(params.dx), loca, gamma],
            device=device,
        )
    wp.launch(
        base.update_rk3_out_kernel,
        dim=(ny, nx),
        inputs=[
            stage["bc"], u0, u_out, stage["fx1"], stage["fx2"],
            stage["fy1"], stage["fy2"], nx, ny, gc, rk,
        ],
        device=device,
    )


def run_to_time(
    initial: np.ndarray,
    params: Params,
    t_end: float,
    device: str,
    *,
    report_interval: int = 100,
    report: Callable[[int, float, float, dict[str, float]], None] | None = None,
) -> tuple[np.ndarray, list[float], int, float]:
    arrays = {
        "u0": wp.array(initial, dtype=wp.float64, device=device),
        "u1": base._state(params, device),
        "u2": base._state(params, device),
        "u3": base._state(params, device),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
        "s1": base._stage(params, device),
        "s2": base._stage(params, device),
        "s3": base._stage(params, device),
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
        _launch_stage(arrays["u0"], arrays["u0"], arrays["u1"], arrays["s1"], params, dt, 1, device, time)
        _launch_stage(arrays["u1"], arrays["u0"], arrays["u2"], arrays["s2"], params, dt, 2, device, time + dt)
        _launch_stage(arrays["u2"], arrays["u0"], arrays["u3"], arrays["s3"], params, dt, 3, device, time + 0.5 * dt)
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
