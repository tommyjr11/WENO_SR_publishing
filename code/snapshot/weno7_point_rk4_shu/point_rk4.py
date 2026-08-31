"""WENO7 point-value finite-volume RHS with Shu's fourth-order TVD RK.

This module is intentionally separate from the existing WENO7/ADER4 path.
It reuses the scalar/characteristic WENO7 and Euler flux helpers, but never
reconstructs spatial derivatives or ADER time derivatives.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from weno7_external_clean import warp_weno7_ader4_helpers_classical_only as wh


wp = wh.wp


@dataclass(frozen=True)
class Params:
    nx: int
    ny: int
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    ghost: int = 4
    gamma: float = 1.4
    cfl: float = 0.4
    t_end: float = 2.0

    @property
    def x_length(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_length(self) -> float:
        return self.y_max - self.y_min

    @property
    def dx(self) -> float:
        return self.x_length / float(self.nx)

    @property
    def dy(self) -> float:
        return self.y_length / float(self.ny)

    @property
    def padded_shape(self) -> tuple[int, int, int]:
        return (self.ny + 2 * self.ghost, self.nx + 2 * self.ghost, 4)


def conserved_from_primitive(rho: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    out = np.empty(rho.shape + (4,), dtype=np.float64)
    out[..., 0] = rho
    out[..., 1] = rho * u
    out[..., 2] = rho * v
    out[..., 3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return out


def primitive_from_conserved(q: np.ndarray, gamma: float) -> np.ndarray:
    rho = np.maximum(q[..., 0], 1.0e-15)
    u = q[..., 1] / rho
    v = q[..., 2] / rho
    p = (gamma - 1.0) * (q[..., 3] - 0.5 * rho * (u * u + v * v))
    return np.stack([rho, u, v, p], axis=-1)


def pdelta(d: np.ndarray, length: float) -> np.ndarray:
    return d - length * np.round(d / length)


def isentropic_vortex_conserved(x: np.ndarray, y: np.ndarray, t: float, gamma: float = 1.4) -> np.ndarray:
    rho_inf = 1.0
    p_inf = 1.0
    u_inf = 1.0
    v_inf = 1.0
    beta = 5.0
    length = 20.0

    dx = pdelta(x - u_inf * t, length)
    dy = pdelta(y - v_inf * t, length)
    r2 = dx * dx + dy * dy
    d_t = -((gamma - 1.0) * beta * beta) / (8.0 * gamma * np.pi * np.pi) * np.exp(1.0 - r2)
    rho = rho_inf + (np.power(1.0 + d_t, 1.0 / (gamma - 1.0)) - 1.0)
    p = p_inf + (np.power(1.0 + d_t, gamma / (gamma - 1.0)) - 1.0)
    factor = beta / (2.0 * np.pi) * np.exp(0.5 * (1.0 - r2))
    u = u_inf - factor * dy
    v = v_inf + factor * dx
    return conserved_from_primitive(rho, u, v, p, gamma)


def cell_average_state(params: Params, t: float, quadrature: int) -> np.ndarray:
    if quadrature < 1:
        raise ValueError("quadrature must be at least 1")
    xi, wi = np.polynomial.legendre.leggauss(quadrature)
    weights = wi.astype(np.float64)
    nodes = xi.astype(np.float64)
    g = params.ghost
    jj, ii = np.indices((params.ny + 2 * g, params.nx + 2 * g))
    xc = params.x_min + (ii - g + 0.5) * params.dx
    yc = params.y_min + (jj - g + 0.5) * params.dy
    out = np.zeros(params.padded_shape, dtype=np.float64)
    for sx, wx in zip(nodes, weights):
        x = xc + 0.5 * params.dx * float(sx)
        for sy, wy in zip(nodes, weights):
            y = yc + 0.5 * params.dy * float(sy)
            out += 0.25 * float(wx) * float(wy) * isentropic_vortex_conserved(x, y, t, params.gamma)
    return out


def interior_stats(u_host: np.ndarray, params: Params) -> dict[str, float]:
    g = params.ghost
    q = u_host[g : g + params.ny, g : g + params.nx, :]
    pri = primitive_from_conserved(q, params.gamma)
    return {
        "mass": float(np.sum(q[..., 0]) * params.dx * params.dy),
        "rho_min": float(np.nanmin(pri[..., 0])),
        "rho_max": float(np.nanmax(pri[..., 0])),
        "p_min": float(np.nanmin(pri[..., 3])),
        "p_max": float(np.nanmax(pri[..., 3])),
        "nan_count": float(np.isnan(q).sum()),
        "rho_neg": float(np.sum(pri[..., 0] <= 0.0)),
        "p_neg": float(np.sum(pri[..., 3] <= 0.0)),
    }


def compute_dt_from_warp_array(u_device: object, speed_workspace: object, params: Params, device: str) -> float:
    wh.require_warp()
    wp.launch(
        wh.compute_max_speed_kernel,
        dim=(params.ny, params.nx),
        inputs=[u_device, speed_workspace, params.nx, params.ny, params.ghost, wp.float64(params.gamma)],
        device=device,
    )
    max_speed = float(np.max(speed_workspace.numpy()))
    if max_speed < 1.0e-15:
        return 1.0e10
    return params.cfl * min(params.dx, params.dy) / max_speed


if wp is not None:

    @wp.kernel
    def copy_array_kernel(src: wp.array3d(dtype=wp.float64), dst: wp.array3d(dtype=wp.float64), n0: int, n1: int):
        j, i = wp.tid()
        if j < n0 and i < n1:
            for comp in range(4):
                dst[j, i, comp] = src[j, i, comp]


    @wp.kernel
    def compute_x_stage_point_kernel(
        u: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        dx: wp.float64,
        gamma: wp.float64,
        characteristic: int,
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
            if characteristic == 1:
                wh.write_vec(left, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 0, 1, gamma))
                wh.write_vec(right, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 0, 1, gamma))
            else:
                wh.write_vec(left, j, i, wh.weno7_lr_vec_conservative(q0, q1, q2, q3, q4, q5, q6, 2, dx, 0))
                wh.write_vec(right, j, i, wh.weno7_lr_vec_conservative(q0, q1, q2, q3, q4, q5, q6, 1, dx, 0))


    @wp.kernel
    def compute_y_stage_point_kernel(
        u: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        dy: wp.float64,
        gamma: wp.float64,
        characteristic: int,
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
            if characteristic == 1:
                wh.write_vec(left, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 0, 2, gamma))
                wh.write_vec(right, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 0, 2, gamma))
            else:
                wh.write_vec(left, j, i, wh.weno7_lr_vec_conservative(q0, q1, q2, q3, q4, q5, q6, 2, dy, 0))
                wh.write_vec(right, j, i, wh.weno7_lr_vec_conservative(q0, q1, q2, q3, q4, q5, q6, 1, dy, 0))


    @wp.kernel
    def compute_x_flux_point_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        dt_over_dx: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        solver_kind: int,
        reverse_upwind: int,
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
            state_l = wh.weno7_gauss_vec_conservative(ql0, ql1, ql2, ql3, ql4, ql5, ql6, loca, wp.float64(1.0), 0)
            state_r = wh.weno7_gauss_vec_conservative(qr0, qr1, qr2, qr3, qr4, qr5, qr6, loca, wp.float64(1.0), 0)
            f = wh.riemann_flux(state_l, state_r, 1, dt_over_dx, gamma, solver_kind)
            if reverse_upwind == 1:
                f = wh.riemann_flux(state_r, state_l, 1, dt_over_dx, gamma, solver_kind)
            wh.write_vec(flux_x, j, i, f)


    @wp.kernel
    def compute_y_flux_point_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        right: wp.array3d(dtype=wp.float64),
        dt_over_dy: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        solver_kind: int,
        reverse_upwind: int,
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
            state_l = wh.weno7_gauss_vec_conservative(ql0, ql1, ql2, ql3, ql4, ql5, ql6, loca, wp.float64(1.0), 0)
            state_r = wh.weno7_gauss_vec_conservative(qr0, qr1, qr2, qr3, qr4, qr5, qr6, loca, wp.float64(1.0), 0)
            f = wh.riemann_flux(state_l, state_r, 2, dt_over_dy, gamma, solver_kind)
            if reverse_upwind == 1:
                f = wh.riemann_flux(state_r, state_l, 2, dt_over_dy, gamma, solver_kind)
            wh.write_vec(flux_y, j, i, f)


    @wp.kernel
    def rhs_from_flux_kernel(
        rhs: wp.array3d(dtype=wp.float64),
        fx1: wp.array3d(dtype=wp.float64),
        fx2: wp.array3d(dtype=wp.float64),
        fy1: wp.array3d(dtype=wp.float64),
        fy2: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        inv_dx: wp.float64,
        inv_dy: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            for comp in range(4):
                fxl = fx1[j, i, comp] + fx2[j, i, comp]
                fxr = fx1[j, i + 1, comp] + fx2[j, i + 1, comp]
                fyb = fy1[j, i, comp] + fy2[j, i, comp]
                fyt = fy1[j + 1, i, comp] + fy2[j + 1, i, comp]
                rhs[jj, ii, comp] = -wp.float64(0.5) * (fxr - fxl) * inv_dx - wp.float64(0.5) * (fyt - fyb) * inv_dy


    @wp.kernel
    def shu_stage1_kernel(
        u0: wp.array3d(dtype=wp.float64),
        rhs0: wp.array3d(dtype=wp.float64),
        u1: wp.array3d(dtype=wp.float64),
        n0: int,
        n1: int,
        dt: wp.float64,
    ):
        j, i = wp.tid()
        if j < n0 and i < n1:
            for comp in range(4):
                u1[j, i, comp] = u0[j, i, comp] + wp.float64(0.5) * dt * rhs0[j, i, comp]


    @wp.kernel
    def shu_stage2_kernel(
        u0: wp.array3d(dtype=wp.float64),
        rhs_t0: wp.array3d(dtype=wp.float64),
        u1: wp.array3d(dtype=wp.float64),
        rhs1: wp.array3d(dtype=wp.float64),
        u2: wp.array3d(dtype=wp.float64),
        n0: int,
        n1: int,
        dt: wp.float64,
    ):
        j, i = wp.tid()
        if j < n0 and i < n1:
            for comp in range(4):
                u2[j, i, comp] = (
                    wp.float64(649.0) / wp.float64(1600.0) * u0[j, i, comp]
                    - wp.float64(10890423.0) / wp.float64(25193600.0) * dt * rhs_t0[j, i, comp]
                    + wp.float64(951.0) / wp.float64(1600.0) * u1[j, i, comp]
                    + wp.float64(5000.0) / wp.float64(7873.0) * dt * rhs1[j, i, comp]
                )


    @wp.kernel
    def shu_stage3_kernel(
        u0: wp.array3d(dtype=wp.float64),
        rhs_t0: wp.array3d(dtype=wp.float64),
        u1: wp.array3d(dtype=wp.float64),
        rhs_t1: wp.array3d(dtype=wp.float64),
        u2: wp.array3d(dtype=wp.float64),
        rhs2: wp.array3d(dtype=wp.float64),
        u3: wp.array3d(dtype=wp.float64),
        n0: int,
        n1: int,
        dt: wp.float64,
    ):
        j, i = wp.tid()
        if j < n0 and i < n1:
            for comp in range(4):
                u3[j, i, comp] = (
                    wp.float64(53989.0) / wp.float64(2500000.0) * u0[j, i, comp]
                    - wp.float64(102261.0) / wp.float64(5000000.0) * dt * rhs_t0[j, i, comp]
                    + wp.float64(4806213.0) / wp.float64(20000000.0) * u1[j, i, comp]
                    - wp.float64(5121.0) / wp.float64(20000.0) * dt * rhs_t1[j, i, comp]
                    + wp.float64(23619.0) / wp.float64(32000.0) * u2[j, i, comp]
                    + wp.float64(7873.0) / wp.float64(10000.0) * dt * rhs2[j, i, comp]
                )


    @wp.kernel
    def shu_final_kernel(
        u0: wp.array3d(dtype=wp.float64),
        rhs0: wp.array3d(dtype=wp.float64),
        u1: wp.array3d(dtype=wp.float64),
        rhs1: wp.array3d(dtype=wp.float64),
        u2: wp.array3d(dtype=wp.float64),
        u3: wp.array3d(dtype=wp.float64),
        rhs3: wp.array3d(dtype=wp.float64),
        out: wp.array3d(dtype=wp.float64),
        n0: int,
        n1: int,
        dt: wp.float64,
    ):
        j, i = wp.tid()
        if j < n0 and i < n1:
            for comp in range(4):
                out[j, i, comp] = (
                    wp.float64(1.0) / wp.float64(5.0) * u0[j, i, comp]
                    + wp.float64(1.0) / wp.float64(10.0) * dt * rhs0[j, i, comp]
                    + wp.float64(6127.0) / wp.float64(30000.0) * u1[j, i, comp]
                    + wp.float64(1.0) / wp.float64(6.0) * dt * rhs1[j, i, comp]
                    + wp.float64(7873.0) / wp.float64(30000.0) * u2[j, i, comp]
                    + wp.float64(1.0) / wp.float64(3.0) * u3[j, i, comp]
                    + wp.float64(1.0) / wp.float64(6.0) * dt * rhs3[j, i, comp]
                )


def allocate_arrays(u0_host: np.ndarray, params: Params, device: str) -> dict[str, object]:
    wh.require_warp()
    shape = params.padded_shape
    fx_shape = (params.ny, params.nx + 1, 4)
    fy_shape = (params.ny + 1, params.nx, 4)
    arrays: dict[str, object] = {
        "u": wp.array(u0_host, dtype=wp.float64, device=device),
        "u1": wp.zeros(shape, dtype=wp.float64, device=device),
        "u2": wp.zeros(shape, dtype=wp.float64, device=device),
        "u3": wp.zeros(shape, dtype=wp.float64, device=device),
        "rhs0": wp.zeros(shape, dtype=wp.float64, device=device),
        "rhs1": wp.zeros(shape, dtype=wp.float64, device=device),
        "rhs2": wp.zeros(shape, dtype=wp.float64, device=device),
        "rhs3": wp.zeros(shape, dtype=wp.float64, device=device),
        "rhs_t0": wp.zeros(shape, dtype=wp.float64, device=device),
        "rhs_t1": wp.zeros(shape, dtype=wp.float64, device=device),
        "x_l": wp.zeros(shape, dtype=wp.float64, device=device),
        "x_r": wp.zeros(shape, dtype=wp.float64, device=device),
        "y_l": wp.zeros(shape, dtype=wp.float64, device=device),
        "y_r": wp.zeros(shape, dtype=wp.float64, device=device),
        "fx1": wp.zeros(fx_shape, dtype=wp.float64, device=device),
        "fx2": wp.zeros(fx_shape, dtype=wp.float64, device=device),
        "fy1": wp.zeros(fy_shape, dtype=wp.float64, device=device),
        "fy2": wp.zeros(fy_shape, dtype=wp.float64, device=device),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
    }
    return arrays


def compute_rhs(
    arrays: dict[str, object],
    params: Params,
    src_name: str,
    rhs_name: str,
    dt: float,
    device: str,
    *,
    solver_kind: int,
    characteristic: bool,
    reverse_upwind: bool,
    boundary: str = "periodic",
) -> None:
    nx = params.nx
    ny = params.ny
    gc = params.ghost
    nx_total = nx + 2 * gc
    ny_total = ny + 2 * gc
    src = arrays[src_name]
    boundary_kernel = wh.apply_periodic_boundary_kernel if boundary == "periodic" else wh.apply_boundary_kernel
    wp.launch(boundary_kernel, dim=(ny_total, nx_total), inputs=[src, nx, ny, gc], device=device)
    characteristic_i = 1 if characteristic else 0
    reverse_i = 1 if reverse_upwind else 0
    gamma = wp.float64(params.gamma)
    wp.launch(
        compute_x_stage_point_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[src, arrays["x_l"], arrays["x_r"], nx, ny, wp.float64(params.dx), gamma, characteristic_i],
        device=device,
    )
    wp.launch(
        compute_x_flux_point_kernel,
        dim=(ny, nx + 1),
        inputs=[arrays["fx1"], arrays["x_l"], arrays["x_r"], wp.float64(dt / params.dx), nx, ny, 1, gamma, solver_kind, reverse_i],
        device=device,
    )
    wp.launch(
        compute_x_flux_point_kernel,
        dim=(ny, nx + 1),
        inputs=[arrays["fx2"], arrays["x_l"], arrays["x_r"], wp.float64(dt / params.dx), nx, ny, 2, gamma, solver_kind, reverse_i],
        device=device,
    )
    wp.launch(
        compute_y_stage_point_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[src, arrays["y_l"], arrays["y_r"], nx, ny, wp.float64(params.dy), gamma, characteristic_i],
        device=device,
    )
    wp.launch(
        compute_y_flux_point_kernel,
        dim=(ny + 1, nx),
        inputs=[arrays["fy1"], arrays["y_l"], arrays["y_r"], wp.float64(dt / params.dy), nx, ny, 1, gamma, solver_kind, reverse_i],
        device=device,
    )
    wp.launch(
        compute_y_flux_point_kernel,
        dim=(ny + 1, nx),
        inputs=[arrays["fy2"], arrays["y_l"], arrays["y_r"], wp.float64(dt / params.dy), nx, ny, 2, gamma, solver_kind, reverse_i],
        device=device,
    )
    wp.launch(
        rhs_from_flux_kernel,
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


def launch_shu_rk4_step(
    arrays: dict[str, object],
    params: Params,
    dt: float,
    device: str,
    *,
    riemann_solver: str = "evilin",
    characteristic: bool = True,
    boundary: str = "periodic",
) -> None:
    solver_kind = 1 if riemann_solver == "hllc" else 0
    n0, n1, _ = params.padded_shape
    compute_rhs(arrays, params, "u", "rhs0", dt, device, solver_kind=solver_kind, characteristic=characteristic, reverse_upwind=False, boundary=boundary)
    wp.launch(shu_stage1_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs0"], arrays["u1"], n0, n1, wp.float64(dt)], device=device)

    compute_rhs(arrays, params, "u", "rhs_t0", dt, device, solver_kind=solver_kind, characteristic=characteristic, reverse_upwind=True, boundary=boundary)
    compute_rhs(arrays, params, "u1", "rhs1", dt, device, solver_kind=solver_kind, characteristic=characteristic, reverse_upwind=False, boundary=boundary)
    wp.launch(shu_stage2_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs_t0"], arrays["u1"], arrays["rhs1"], arrays["u2"], n0, n1, wp.float64(dt)], device=device)

    compute_rhs(arrays, params, "u1", "rhs_t1", dt, device, solver_kind=solver_kind, characteristic=characteristic, reverse_upwind=True, boundary=boundary)
    compute_rhs(arrays, params, "u2", "rhs2", dt, device, solver_kind=solver_kind, characteristic=characteristic, reverse_upwind=False, boundary=boundary)
    wp.launch(shu_stage3_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs_t0"], arrays["u1"], arrays["rhs_t1"], arrays["u2"], arrays["rhs2"], arrays["u3"], n0, n1, wp.float64(dt)], device=device)

    compute_rhs(arrays, params, "u3", "rhs3", dt, device, solver_kind=solver_kind, characteristic=characteristic, reverse_upwind=False, boundary=boundary)
    wp.launch(shu_final_kernel, dim=(n0, n1), inputs=[arrays["u"], arrays["rhs0"], arrays["u1"], arrays["rhs1"], arrays["u2"], arrays["u3"], arrays["rhs3"], arrays["u"], n0, n1, wp.float64(dt)], device=device)


def run_case(
    params: Params,
    *,
    device: str,
    init_quadrature: int,
    riemann_solver: str,
    characteristic: bool,
    report_interval: int = 0,
    max_steps: int = 10_000_000,
    boundary: str = "periodic",
) -> tuple[np.ndarray, dict[str, object]]:
    wh.require_warp()
    wp.init()
    wp.set_device(device)
    u0 = cell_average_state(params, 0.0, init_quadrature)
    return run_from_initial(
        u0,
        params,
        device=device,
        riemann_solver=riemann_solver,
        characteristic=characteristic,
        report_interval=report_interval,
        max_steps=max_steps,
        boundary=boundary,
    )


def run_from_initial(
    u0: np.ndarray,
    params: Params,
    *,
    device: str,
    riemann_solver: str,
    characteristic: bool,
    report_interval: int = 0,
    max_steps: int = 10_000_000,
    boundary: str = "periodic",
) -> tuple[np.ndarray, dict[str, object]]:
    wh.require_warp()
    wp.init()
    wp.set_device(device)
    arrays = allocate_arrays(u0, params, device)
    t = 0.0
    step = 0
    dt_values: list[float] = []
    while t < params.t_end and step < max_steps:
        dt = compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, device)
        if t + dt > params.t_end:
            dt = params.t_end - t
        if dt <= 0.0:
            break
        launch_shu_rk4_step(arrays, params, dt, device, riemann_solver=riemann_solver, characteristic=characteristic, boundary=boundary)
        t += dt
        step += 1
        dt_values.append(float(dt))
        if report_interval > 0 and (step == 1 or step % report_interval == 0 or t >= params.t_end):
            stats_now = interior_stats(arrays["u"].numpy(), params)
            print(
                f"N={params.nx} step={step:05d} t={t:.8e} dt={dt:.8e} "
                f"rho=[{stats_now['rho_min']:.6e},{stats_now['rho_max']:.6e}] "
                f"p=[{stats_now['p_min']:.6e},{stats_now['p_max']:.6e}] nan={int(stats_now['nan_count'])}",
                flush=True,
            )
            if stats_now["nan_count"] or stats_now["rho_neg"] or stats_now["p_neg"]:
                break
    final = arrays["u"].numpy()
    stats = interior_stats(final, params)
    summary: dict[str, object] = {
        "steps": int(step),
        "t": float(t),
        "dt_min": float(np.min(dt_values)) if dt_values else 0.0,
        "dt_max": float(np.max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
        **stats,
    }
    return final, summary
