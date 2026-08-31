"""Torch-beta MLP extension for the WENO7 point-value Shu-RK4 path.

The neural network is evaluated outside Warp by PyTorch.  Warp only receives
the resulting beta arrays and performs the WENO7 point-value reconstruction.
This keeps the MLP out of CUDA code generation.
"""

from __future__ import annotations

import torch

from . import point_rk4 as base
from . import warp_weno7_ader4_helpers_classical_only as wh
from . import warp_weno7_external_overwrite_fullall as ext
from .beta_provider import TorchWeno7Beta as _TorchWeno7Beta


wp = wh.wp
Params = base.Params


class TorchWeno7PointBeta(_TorchWeno7Beta):
    @torch.no_grad()
    def fill_normal_x_for(self, arrays: dict[str, object], params: Params, src_name: str) -> None:
        u = wp.to_torch(arrays[src_name])
        qx = [u[:, k : k + params.nx + 2, :] for k in range(7)]
        wp.to_torch(arrays["beta_x"]).copy_(self._compute_beta(qx, 1))
        self._sync_after_copy()

    @torch.no_grad()
    def fill_normal_y_for(self, arrays: dict[str, object], params: Params, src_name: str) -> None:
        u = wp.to_torch(arrays[src_name])
        qy = [u[k : k + params.ny + 2, :, :] for k in range(7)]
        wp.to_torch(arrays["beta_y"]).copy_(self._compute_beta(qy, 2))
        self._sync_after_copy()

    @torch.no_grad()
    def fill_cross_x_point(self, arrays: dict[str, object], params: Params) -> None:
        out = wp.to_torch(arrays["beta_cross_x"])
        for group, name in ((0, "x_r"), (1, "x_l")):
            a = wp.to_torch(arrays[name])
            q = [a[k + 1 : k + 1 + params.ny, : params.nx + 2, :] for k in range(7)]
            start = group * 128
            out[..., start : start + 32].copy_(self._compute_beta_conservative(q))
        self._sync_after_copy()

    @torch.no_grad()
    def fill_cross_y_point(self, arrays: dict[str, object], params: Params) -> None:
        out = wp.to_torch(arrays["beta_cross_y"])
        for group, name in ((0, "y_r"), (1, "y_l")):
            a = wp.to_torch(arrays[name])
            q = [a[: params.ny + 2, k + 1 : k + 1 + params.nx, :] for k in range(7)]
            start = group * 128
            out[..., start : start + 32].copy_(self._compute_beta_conservative(q))
        self._sync_after_copy()


if wp is not None:

    @wp.kernel
    def compute_x_stage_point_mlp_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_x: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        dx: wp.float64,
        gamma: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 8 and i < nx + 2:
            q0 = wh.vec_from_array(u, j, i)
            q1 = wh.vec_from_array(u, j, i + 1)
            q2 = wh.vec_from_array(u, j, i + 2)
            q3 = wh.vec_from_array(u, j, i + 3)
            q4 = wh.vec_from_array(u, j, i + 4)
            q5 = wh.vec_from_array(u, j, i + 5)
            q6 = wh.vec_from_array(u, j, i + 6)
            wh.write_vec(left, j, i, ext._weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dx, 0, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(right, j, i, ext._weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dx, 0, 1, gamma, beta_x, j, i, eno_cutoff))


    @wp.kernel
    def compute_y_stage_point_mlp_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_y: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        dy: wp.float64,
        gamma: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 8:
            q0 = wh.vec_from_array(u, j, i)
            q1 = wh.vec_from_array(u, j + 1, i)
            q2 = wh.vec_from_array(u, j + 2, i)
            q3 = wh.vec_from_array(u, j + 3, i)
            q4 = wh.vec_from_array(u, j + 4, i)
            q5 = wh.vec_from_array(u, j + 5, i)
            q6 = wh.vec_from_array(u, j + 6, i)
            wh.write_vec(left, j, i, ext._weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dy, 0, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(right, j, i, ext._weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dy, 0, 2, gamma, beta_y, j, i, eno_cutoff))


    @wp.kernel
    def compute_x_flux_point_mlp_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        beta_cross_x: wp.array3d(dtype=wp.float64),
        dt_over_dx: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        solver_kind: int,
        reverse_upwind: int,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            ql0 = wh.vec_from_array(right, j + 1, i)
            ql1 = wh.vec_from_array(right, j + 2, i)
            ql2 = wh.vec_from_array(right, j + 3, i)
            ql3 = wh.vec_from_array(right, j + 4, i)
            ql4 = wh.vec_from_array(right, j + 5, i)
            ql5 = wh.vec_from_array(right, j + 6, i)
            ql6 = wh.vec_from_array(right, j + 7, i)
            qr0 = wh.vec_from_array(left, j + 1, i + 1)
            qr1 = wh.vec_from_array(left, j + 2, i + 1)
            qr2 = wh.vec_from_array(left, j + 3, i + 1)
            qr3 = wh.vec_from_array(left, j + 4, i + 1)
            qr4 = wh.vec_from_array(left, j + 5, i + 1)
            qr5 = wh.vec_from_array(left, j + 6, i + 1)
            qr6 = wh.vec_from_array(left, j + 7, i + 1)
            state_l = ext._weno7_gauss_vec_conservative_external_beta(ql0, ql1, ql2, ql3, ql4, ql5, ql6, loca, wp.float64(1.0), 0, beta_cross_x, j, i, 0, 0, eno_cutoff)
            state_r = ext._weno7_gauss_vec_conservative_external_beta(qr0, qr1, qr2, qr3, qr4, qr5, qr6, loca, wp.float64(1.0), 0, beta_cross_x, j, i + 1, 1, 0, eno_cutoff)
            f = wh.riemann_flux(state_l, state_r, 1, dt_over_dx, gamma, solver_kind)
            if reverse_upwind == 1:
                f = wh.riemann_flux(state_r, state_l, 1, dt_over_dx, gamma, solver_kind)
            wh.write_vec(flux_x, j, i, f)


    @wp.kernel
    def compute_y_flux_point_mlp_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        beta_cross_y: wp.array3d(dtype=wp.float64),
        dt_over_dy: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        solver_kind: int,
        reverse_upwind: int,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            ql0 = wh.vec_from_array(right, j, i + 1)
            ql1 = wh.vec_from_array(right, j, i + 2)
            ql2 = wh.vec_from_array(right, j, i + 3)
            ql3 = wh.vec_from_array(right, j, i + 4)
            ql4 = wh.vec_from_array(right, j, i + 5)
            ql5 = wh.vec_from_array(right, j, i + 6)
            ql6 = wh.vec_from_array(right, j, i + 7)
            qr0 = wh.vec_from_array(left, j + 1, i + 1)
            qr1 = wh.vec_from_array(left, j + 1, i + 2)
            qr2 = wh.vec_from_array(left, j + 1, i + 3)
            qr3 = wh.vec_from_array(left, j + 1, i + 4)
            qr4 = wh.vec_from_array(left, j + 1, i + 5)
            qr5 = wh.vec_from_array(left, j + 1, i + 6)
            qr6 = wh.vec_from_array(left, j + 1, i + 7)
            state_l = ext._weno7_gauss_vec_conservative_external_beta(ql0, ql1, ql2, ql3, ql4, ql5, ql6, loca, wp.float64(1.0), 0, beta_cross_y, j, i, 0, 0, eno_cutoff)
            state_r = ext._weno7_gauss_vec_conservative_external_beta(qr0, qr1, qr2, qr3, qr4, qr5, qr6, loca, wp.float64(1.0), 0, beta_cross_y, j + 1, i, 1, 0, eno_cutoff)
            f = wh.riemann_flux(state_l, state_r, 2, dt_over_dy, gamma, solver_kind)
            if reverse_upwind == 1:
                f = wh.riemann_flux(state_r, state_l, 2, dt_over_dy, gamma, solver_kind)
            wh.write_vec(flux_y, j, i, f)


def allocate_arrays_mlp(u0_host, params: Params, device: str) -> dict[str, object]:
    arrays = base.allocate_arrays(u0_host, params, device)
    arrays.update(ext.allocate_external_beta_arrays(params.nx, params.ny, device))
    return arrays


def compute_rhs_mlp(
    arrays: dict[str, object],
    params: Params,
    src_name: str,
    rhs_name: str,
    dt: float,
    device: str,
    *,
    solver_kind: int,
    beta_model: TorchWeno7PointBeta,
    reverse_upwind: bool,
    boundary: str = "periodic",
    eno_cutoff: bool = False,
) -> None:
    nx = params.nx
    ny = params.ny
    gc = params.ghost
    nx_total = nx + 2 * gc
    ny_total = ny + 2 * gc
    src = arrays[src_name]
    boundary_kernel = wh.apply_periodic_boundary_kernel if boundary == "periodic" else wh.apply_boundary_kernel
    wp.launch(boundary_kernel, dim=(ny_total, nx_total), inputs=[src, nx, ny, gc], device=device)
    wp.synchronize()

    reverse_i = 1 if reverse_upwind else 0
    eno_i = 1 if eno_cutoff else 0
    gamma = wp.float64(params.gamma)

    beta_model.fill_normal_x_for(arrays, params, src_name)
    wp.launch(
        compute_x_stage_point_mlp_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[src, arrays["beta_x"], arrays["x_l"], arrays["x_r"], nx, ny, wp.float64(params.dx), gamma, eno_i],
        device=device,
    )
    wp.synchronize()
    beta_model.fill_cross_x_point(arrays, params)
    for loca, flux_name in ((1, "fx1"), (2, "fx2")):
        wp.launch(
            compute_x_flux_point_mlp_kernel,
            dim=(ny, nx + 1),
            inputs=[arrays[flux_name], arrays["x_l"], arrays["x_r"], arrays["beta_cross_x"], wp.float64(dt / params.dx), nx, ny, loca, gamma, solver_kind, reverse_i, eno_i],
            device=device,
        )

    beta_model.fill_normal_y_for(arrays, params, src_name)
    wp.launch(
        compute_y_stage_point_mlp_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[src, arrays["beta_y"], arrays["y_l"], arrays["y_r"], nx, ny, wp.float64(params.dy), gamma, eno_i],
        device=device,
    )
    wp.synchronize()
    beta_model.fill_cross_y_point(arrays, params)
    for loca, flux_name in ((1, "fy1"), (2, "fy2")):
        wp.launch(
            compute_y_flux_point_mlp_kernel,
            dim=(ny + 1, nx),
            inputs=[arrays[flux_name], arrays["y_l"], arrays["y_r"], arrays["beta_cross_y"], wp.float64(dt / params.dy), nx, ny, loca, gamma, solver_kind, reverse_i, eno_i],
            device=device,
        )

    wp.launch(
        base.rhs_from_flux_kernel,
        dim=(ny, nx),
        inputs=[
            arrays[rhs_name],
            arrays["fx1"],
            arrays["fx2"],
            arrays["fy1"],
            arrays["fy2"],
            nx,
            ny,
            gc,
            wp.float64(1.0 / params.dx),
            wp.float64(1.0 / params.dy),
        ],
        device=device,
    )


def launch_shu_rk4_step_mlp(
    arrays: dict[str, object],
    params: Params,
    dt: float,
    device: str,
    *,
    riemann_solver: str,
    beta_model: TorchWeno7PointBeta,
    boundary: str = "periodic",
    eno_cutoff: bool = False,
) -> None:
    solver_kind = 1 if riemann_solver == "hllc" else 0
    n0, n1, _ = params.padded_shape
    compute_rhs_mlp(arrays, params, "u", "rhs0", dt, device, solver_kind=solver_kind, beta_model=beta_model, reverse_upwind=False, boundary=boundary, eno_cutoff=eno_cutoff)
    wp.launch(base.shu_stage1_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs0"], arrays["u1"], n0, n1, wp.float64(dt)], device=device)

    compute_rhs_mlp(arrays, params, "u", "rhs_t0", dt, device, solver_kind=solver_kind, beta_model=beta_model, reverse_upwind=True, boundary=boundary, eno_cutoff=eno_cutoff)
    compute_rhs_mlp(arrays, params, "u1", "rhs1", dt, device, solver_kind=solver_kind, beta_model=beta_model, reverse_upwind=False, boundary=boundary, eno_cutoff=eno_cutoff)
    wp.launch(base.shu_stage2_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs_t0"], arrays["u1"], arrays["rhs1"], arrays["u2"], n0, n1, wp.float64(dt)], device=device)

    compute_rhs_mlp(arrays, params, "u1", "rhs_t1", dt, device, solver_kind=solver_kind, beta_model=beta_model, reverse_upwind=True, boundary=boundary, eno_cutoff=eno_cutoff)
    compute_rhs_mlp(arrays, params, "u2", "rhs2", dt, device, solver_kind=solver_kind, beta_model=beta_model, reverse_upwind=False, boundary=boundary, eno_cutoff=eno_cutoff)
    wp.launch(base.shu_stage3_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs_t0"], arrays["u1"], arrays["rhs_t1"], arrays["u2"], arrays["rhs2"], arrays["u3"], n0, n1, wp.float64(dt)], device=device)

    compute_rhs_mlp(arrays, params, "u3", "rhs3", dt, device, solver_kind=solver_kind, beta_model=beta_model, reverse_upwind=False, boundary=boundary, eno_cutoff=eno_cutoff)
    wp.launch(base.shu_final_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs0"], arrays["u1"], arrays["rhs1"], arrays["u2"], arrays["u3"], arrays["rhs3"], arrays["u"], n0, n1, wp.float64(dt)], device=device)


def run_from_initial_mlp(
    u0,
    params: Params,
    *,
    device: str,
    riemann_solver: str,
    beta_model: TorchWeno7PointBeta,
    report_interval: int = 0,
    max_steps: int = 10_000_000,
    boundary: str = "periodic",
    eno_cutoff: bool = False,
) -> tuple[object, dict[str, object]]:
    wh.require_warp()
    wp.init()
    wp.set_device(device)
    arrays = allocate_arrays_mlp(u0, params, device)
    t = 0.0
    step = 0
    dt_values: list[float] = []
    while t < params.t_end and step < max_steps:
        dt = base.compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, device)
        if t + dt > params.t_end:
            dt = params.t_end - t
        if dt <= 0.0:
            break
        launch_shu_rk4_step_mlp(
            arrays,
            params,
            dt,
            device,
            riemann_solver=riemann_solver,
            beta_model=beta_model,
            boundary=boundary,
            eno_cutoff=eno_cutoff,
        )
        t += dt
        step += 1
        dt_values.append(float(dt))
        if report_interval > 0 and (step == 1 or step % report_interval == 0 or t >= params.t_end):
            stats_now = base.interior_stats(arrays["u"].numpy(), params)
            print(
                f"MLP-RK4 N={params.nx} step={step:05d} t={t:.8e} dt={dt:.8e} "
                f"rho=[{stats_now['rho_min']:.6e},{stats_now['rho_max']:.6e}] "
                f"p=[{stats_now['p_min']:.6e},{stats_now['p_max']:.6e}] nan={int(stats_now['nan_count'])}",
                flush=True,
            )
            if stats_now["nan_count"] or stats_now["rho_neg"] or stats_now["p_neg"]:
                print("MLP-RK4 failure: NaN/negative rho/p detected, stopping early", flush=True)
                break
    final = arrays["u"].numpy()
    stats = base.interior_stats(final, params)
    summary: dict[str, object] = {
        "steps": int(step),
        "t": float(t),
        "dt_min": float(min(dt_values)) if dt_values else 0.0,
        "dt_max": float(max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(sum(dt_values) / len(dt_values)) if dt_values else 0.0,
        **stats,
    }
    return final, summary
