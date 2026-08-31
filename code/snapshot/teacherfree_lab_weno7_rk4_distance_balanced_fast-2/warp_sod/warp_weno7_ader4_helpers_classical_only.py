"""WENO7 ADER4 Warp helper code for the shock-bubble CUDA comparison.

This file is independent from the WENO5/RK3 helper.  It mirrors the
ADER_TR4 HEOC path: ghost=4, double precision, characteristic WENO7 in the
normal direction, three ADER4 time Gauss points, and EVILIN fluxes.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import numpy as np

try:
    import warp as wp
except ModuleNotFoundError:
    wp = None


GAUSS15_XI = np.array(
    [
        -0.9879925180204853774057482951320707798004150390625,
        -0.93727339240070584036601530897314660251140594482421875,
        -0.84820658341042720618219163952744565904140472412109375,
        -0.724417731360170069621062793885357677936553955078125,
        -0.570972172608538830473889902350492775440216064453125,
        -0.39415134707756338539041962576447986066341400146484375,
        -0.201194093997434542142599411818082444369792938232421875,
        0.0,
        0.201194093997434542142599411818082444369792938232421875,
        0.39415134707756338539041962576447986066341400146484375,
        0.570972172608538830473889902350492775440216064453125,
        0.724417731360170069621062793885357677936553955078125,
        0.84820658341042720618219163952744565904140472412109375,
        0.93727339240070584036601530897314660251140594482421875,
        0.9879925180204853774057482951320707798004150390625,
    ],
    dtype=np.float64,
)

GAUSS15_W = np.array(
    [
        0.0307532419961180671086342641729061142541468143463134765625,
        0.0703660474881085129528202060100738890469074249267578125,
        0.10715922046717189786146917640508036129176616668701171875,
        0.1395706779261540464442958864310639910399913787841796875,
        0.1662692058169938091882755770711810328066349029541015625,
        0.1861610000155619337736112584025249816477298736572265625,
        0.1984314853271114398314267646128428168594837188720703125,
        0.2025782419255609811958862565006711520254611968994140625,
        0.1984314853271114398314267646128428168594837188720703125,
        0.1861610000155619337736112584025249816477298736572265625,
        0.1662692058169938091882755770711810328066349029541015625,
        0.1395706779261540464442958864310639910399913787841796875,
        0.10715922046717189786146917640508036129176616668701171875,
        0.0703660474881085129528202060100738890469074249267578125,
        0.0307532419961180671086342641729061142541468143463134765625,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Params:
    nx: int = 450
    ny: int = 178
    ghost: int = 4
    gamma: float = 1.4
    x_length: float = 0.225
    y_length: float = 0.089
    cfl: float = 0.228
    t_end: float = 0.0002

    @property
    def dx(self) -> float:
        return self.x_length / float(self.nx)

    @property
    def dy(self) -> float:
        return self.y_length / float(self.ny)

    @property
    def padded_shape(self) -> tuple[int, int, int]:
        return (self.ny + 2 * self.ghost, self.nx + 2 * self.ghost, 4)


def primitive_to_conserved(rho: float, vel_x: float, vel_y: float, p: float, gamma: float = 1.4) -> np.ndarray:
    energy = 0.5 * rho * (vel_x * vel_x + vel_y * vel_y) + p / (gamma - 1.0)
    return np.array([rho, rho * vel_x, rho * vel_y, energy], dtype=np.float64)


def shock_bubble_conserved_state(x: float, y: float, t: float = 0.0) -> np.ndarray:
    del t
    gamma = 1.4
    mach = 1.22
    x_shock = 0.005
    rho_air = 1.29
    p_air = 101325.0
    rho_he = 0.214
    rho_post = (((gamma + 1.0) * mach * mach) / ((gamma - 1.0) * mach * mach + 2.0)) * rho_air
    p_post = ((((2.0 * gamma) * mach * mach) - (gamma - 1.0)) / (gamma + 1.0)) * p_air
    u_post = 110.6273

    if x < x_shock:
        con = primitive_to_conserved(rho_post, u_post, 0.0, p_post, gamma)
    else:
        con = primitive_to_conserved(rho_air, 0.0, 0.0, p_air, gamma)

    bubble_xc = 0.035
    bubble_yc = 0.0445
    bubble_r = 0.025
    dx_bubble = x - bubble_xc
    dy_bubble = y - bubble_yc
    if dx_bubble * dx_bubble + dy_bubble * dy_bubble <= bubble_r * bubble_r:
        con = primitive_to_conserved(rho_he, 0.0, 0.0, p_air, gamma)
    return con


def function2d_shock_bubble_conserved(x: float, y: float, comp: int, t: float = 0.0) -> float:
    return float(shock_bubble_conserved_state(x, y, t)[comp])


def cell_average_state_2d_15point(x_center: float, y_center: float, dx: float, dy: float, t: float = 0.0) -> np.ndarray:
    state = np.zeros(4, dtype=np.float64)
    for ix, wx in zip(GAUSS15_XI, GAUSS15_W):
        x = x_center + 0.5 * dx * float(ix)
        for iy, wy in zip(GAUSS15_XI, GAUSS15_W):
            y = y_center + 0.5 * dy * float(iy)
            state += float(wx) * float(wy) * shock_bubble_conserved_state(x, y, t)
    return 0.25 * state


def make_initial_state(params: Params) -> np.ndarray:
    u = np.zeros(params.padded_shape, dtype=np.float64)
    ny_total, nx_total, _ = params.padded_shape
    for j in range(ny_total):
        y = (j - params.ghost + 0.5) * params.dy
        for i in range(nx_total):
            x = (i - params.ghost + 0.5) * params.dx
            u[j, i, :] = cell_average_state_2d_15point(x, y, params.dx, params.dy, 0.0)
    return u


def primitive_from_conserved(u: np.ndarray, gamma: float) -> np.ndarray:
    rho = np.maximum(u[..., 0], 1.0e-15)
    vel_x = u[..., 1] / rho
    vel_y = u[..., 2] / rho
    p = (gamma - 1.0) * (u[..., 3] - 0.5 * rho * (vel_x * vel_x + vel_y * vel_y))
    return np.stack([rho, vel_x, vel_y, p], axis=-1)


def compute_dt(u_host: np.ndarray, params: Params) -> float:
    gc = params.ghost
    interior = u_host[gc : gc + params.ny, gc : gc + params.nx, :]
    pri = primitive_from_conserved(interior, params.gamma)
    rho = np.maximum(pri[..., 0], 1.0e-15)
    p = np.maximum(pri[..., 3], 1.0e-15)
    sound = np.sqrt(params.gamma * p / rho)
    speed = np.maximum(np.abs(pri[..., 1]) + sound, np.abs(pri[..., 2]) + sound)
    max_speed = float(np.max(speed))
    if max_speed < 1.0e-15:
        return 1.0e10
    return params.cfl * min(params.dx, params.dy) / max_speed


def compute_dt_from_warp_array(u_device: object, speed_workspace: object, params: Params, device: str) -> float:
    require_warp()
    wp.launch(
        compute_max_speed_kernel,
        dim=(params.ny, params.nx),
        inputs=[u_device, speed_workspace, params.nx, params.ny, params.ghost, wp.float64(params.gamma)],
        device=device,
    )
    max_speed = float(np.max(speed_workspace.numpy()))
    if max_speed < 1.0e-15:
        return 1.0e10
    return params.cfl * min(params.dx, params.dy) / max_speed


def interior_stats(u_host: np.ndarray, params: Params) -> dict[str, float]:
    gc = params.ghost
    interior = u_host[gc : gc + params.ny, gc : gc + params.nx, :]
    pri = primitive_from_conserved(interior, params.gamma)
    return {
        "mass": float(np.sum(interior[..., 0]) * params.dx * params.dy),
        "rho_min": float(np.min(pri[..., 0])),
        "rho_max": float(np.max(pri[..., 0])),
        "p_min": float(np.min(pri[..., 3])),
        "p_max": float(np.max(pri[..., 3])),
        "nan_count": float(np.isnan(interior).sum()),
        "rho_neg": float(np.sum(pri[..., 0] <= 0.0)),
        "p_neg": float(np.sum(pri[..., 3] <= 0.0)),
    }


def require_warp() -> None:
    if wp is None:
        print("NVIDIA Warp is not installed. Install it with:\n    pip install warp-lang\n", file=sys.stderr)
        raise SystemExit(1)


if wp is not None:

    @wp.func
    def safe_rcp(x: wp.float64) -> wp.float64:
        return wp.float64(1.0) / (wp.float64(1.0e-6) + x)


    @wp.func
    def vec_from_array(a: wp.array3d(dtype=wp.float64), j: int, i: int) -> wp.vec4d:
        return wp.vec4d(a[j, i, 0], a[j, i, 1], a[j, i, 2], a[j, i, 3])


    @wp.func
    def write_vec(a: wp.array3d(dtype=wp.float64), j: int, i: int, q: wp.vec4d):
        a[j, i, 0] = q[0]
        a[j, i, 1] = q[1]
        a[j, i, 2] = q[2]
        a[j, i, 3] = q[3]


    @wp.func
    def con_to_pri(q: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        rho = q[0]
        u = q[1] / rho
        v = q[2] / rho
        p = (gamma - wp.float64(1.0)) * (q[3] - wp.float64(0.5) * rho * (u * u + v * v))
        return wp.vec4d(rho, u, v, p)


    @wp.func
    def pri_to_con(w: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        rho = w[0]
        u = w[1]
        v = w[2]
        p = w[3]
        e = wp.float64(0.5) * rho * (u * u + v * v) + p / (gamma - wp.float64(1.0))
        return wp.vec4d(rho, rho * u, rho * v, e)


    @wp.func
    def pri_to_flux(w: wp.vec4d, direction: int, gamma: wp.float64) -> wp.vec4d:
        rho = w[0]
        u = w[1]
        v = w[2]
        p = w[3]
        e = wp.float64(0.5) * rho * (u * u + v * v) + p / (gamma - wp.float64(1.0))
        if direction == 1:
            return wp.vec4d(rho * u, rho * u * u + p, rho * u * v, u * (p + e))
        return wp.vec4d(rho * v, rho * u * v, rho * v * v + p, v * (p + e))


    @wp.func
    def evilin_state_2d(ul0: wp.vec4d, ur0: wp.vec4d, direction: int, c: wp.float64, gamma: wp.float64) -> wp.vec4d:
        tiny = wp.float64(1.0e-15)
        wl0 = con_to_pri(ul0, gamma)
        wr0 = con_to_pri(ur0, gamma)
        un_l = wl0[1]
        un_r = wr0[1]
        if direction == 2:
            un_l = wl0[2]
            un_r = wr0[2]
        a_l = wp.sqrt(gamma * wl0[3] / wl0[0])
        a_r = wp.sqrt(gamma * wr0[3] / wr0[0])
        smax = wp.max(wp.abs(un_l) + a_l, wp.abs(un_r) + a_r)
        smax = wp.max(smax, tiny)
        dt_dx_loc = c
        dx_dt_loc = wp.float64(1.0) / dt_dx_loc
        local_cfl = c * smax
        fl = pri_to_flux(wl0, direction, gamma)
        fr = pri_to_flux(wr0, direction, gamma)
        qlw = wp.vec4d(
            wp.float64(0.5) * (ul0[0] + ur0[0]) - wp.float64(0.5) * dt_dx_loc * (fr[0] - fl[0]),
            wp.float64(0.5) * (ul0[1] + ur0[1]) - wp.float64(0.5) * dt_dx_loc * (fr[1] - fl[1]),
            wp.float64(0.5) * (ul0[2] + ur0[2]) - wp.float64(0.5) * dt_dx_loc * (fr[2] - fl[2]),
            wp.float64(0.5) * (ul0[3] + ur0[3]) - wp.float64(0.5) * dt_dx_loc * (fr[3] - fl[3]),
        )
        flw = pri_to_flux(con_to_pri(qlw, gamma), direction, gamma)
        ffo = wp.vec4d(
            wp.float64(0.25) * (fl[0] + wp.float64(2.0) * flw[0] + fr[0] - dx_dt_loc * (ur0[0] - ul0[0])),
            wp.float64(0.25) * (fl[1] + wp.float64(2.0) * flw[1] + fr[1] - dx_dt_loc * (ur0[1] - ul0[1])),
            wp.float64(0.25) * (fl[2] + wp.float64(2.0) * flw[2] + fr[2] - dx_dt_loc * (ur0[2] - ul0[2])),
            wp.float64(0.25) * (fl[3] + wp.float64(2.0) * flw[3] + fr[3] - dx_dt_loc * (ur0[3] - ul0[3])),
        )
        theta = (wp.float64(1.0) - local_cfl) / (wp.float64(1.0) + local_cfl)
        fgf = wp.vec4d(
            theta * flw[0] + (wp.float64(1.0) - theta) * ffo[0],
            theta * flw[1] + (wp.float64(1.0) - theta) * ffo[1],
            theta * flw[2] + (wp.float64(1.0) - theta) * ffo[2],
            theta * flw[3] + (wp.float64(1.0) - theta) * ffo[3],
        )
        ulh = wp.vec4d(
            ul0[0] - dt_dx_loc * (fgf[0] - fl[0]),
            ul0[1] - dt_dx_loc * (fgf[1] - fl[1]),
            ul0[2] - dt_dx_loc * (fgf[2] - fl[2]),
            ul0[3] - dt_dx_loc * (fgf[3] - fl[3]),
        )
        urh = wp.vec4d(
            ur0[0] - dt_dx_loc * (fr[0] - fgf[0]),
            ur0[1] - dt_dx_loc * (fr[1] - fgf[1]),
            ur0[2] - dt_dx_loc * (fr[2] - fgf[2]),
            ur0[3] - dt_dx_loc * (fr[3] - fgf[3]),
        )
        wl = con_to_pri(ulh, gamma)
        wr = con_to_pri(urh, gamma)
        rho_bar = wp.max(wp.float64(0.5) * (wl[0] + wr[0]), tiny)
        p_bar = wp.max(wp.float64(0.5) * (wl[3] + wr[3]), tiny)
        a_bar = wp.max(wp.sqrt(gamma * p_bar / rho_bar), tiny)
        c1 = rho_bar * a_bar
        c2 = rho_bar / a_bar
        un_le = wl[1]
        un_re = wr[1]
        if direction == 2:
            un_le = wl[2]
            un_re = wr[2]
        u_star = wp.float64(0.5) * (un_le + un_re) - wp.float64(0.5) * (wr[3] - wl[3]) / c1
        p_star = wp.float64(0.5) * (wl[3] + wr[3]) - wp.float64(0.5) * (un_re - un_le) * c1
        rho_star_l = wp.max(wl[0] + (un_le - u_star) * c2, tiny)
        rho_star_r = wp.max(wr[0] + (u_star - un_re) * c2, tiny)
        p_star = wp.max(p_star, tiny)
        u_bar = wp.float64(0.5) * (un_le + un_re)
        lam1 = u_bar - a_bar
        lam2 = u_bar
        lam3 = u_bar + a_bar
        wface = wl
        if lam1 >= wp.float64(0.0):
            wface = wl
        elif lam2 >= wp.float64(0.0):
            if direction == 1:
                wface = wp.vec4d(rho_star_l, u_star, wl[2], p_star)
            else:
                wface = wp.vec4d(rho_star_l, wl[1], u_star, p_star)
        elif lam3 >= wp.float64(0.0):
            if direction == 1:
                wface = wp.vec4d(rho_star_r, u_star, wr[2], p_star)
            else:
                wface = wp.vec4d(rho_star_r, wr[1], u_star, p_star)
        else:
            wface = wr
        return pri_to_con(wface, gamma)

    @wp.func
    def hllc_flux(ul0: wp.vec4d, ur0: wp.vec4d, direction: int, gamma: wp.float64) -> wp.vec4d:
        tiny = wp.float64(1.0e-16)
        wl = con_to_pri(ul0, gamma)
        wr = con_to_pri(ur0, gamma)
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

        f_l = pri_to_flux(wl, direction, gamma)
        f_r = pri_to_flux(wr, direction, gamma)
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
        s_star = (p_r - p_l + rho_l * un_l * (s_l - un_l) - rho_r * un_r * (s_r - un_r)) / denom

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
        e_star = factor * ((u_side[3] / rho) + (s_star - un) * (s_star + p / (rho * denom_side)))
        u_star = wp.vec4d(rho_star, mx_star, my_star, e_star)

        return wp.vec4d(
            f_side[0] + s_side * (u_star[0] - u_side[0]),
            f_side[1] + s_side * (u_star[1] - u_side[1]),
            f_side[2] + s_side * (u_star[2] - u_side[2]),
            f_side[3] + s_side * (u_star[3] - u_side[3]),
        )

    @wp.func
    def riemann_flux(ul0: wp.vec4d, ur0: wp.vec4d, direction: int, c: wp.float64, gamma: wp.float64, solver_kind: int) -> wp.vec4d:
        if solver_kind == 1:
            return hllc_flux(ul0, ur0, direction, gamma)
        return pri_to_flux(con_to_pri(evilin_state_2d(ul0, ur0, direction, c, gamma), gamma), direction, gamma)


    @wp.func
    def roe_average_state(left: wp.vec4d, right: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        tiny = wp.float64(1.0e-15)
        gm1 = gamma - wp.float64(1.0)
        rho_l = wp.max(left[0], tiny)
        u_l = left[1] / rho_l
        v_l = left[2] / rho_l
        e_l = left[3]
        p_l = wp.max(gm1 * (e_l - wp.float64(0.5) * rho_l * (u_l * u_l + v_l * v_l)), tiny)
        h_l = (e_l + p_l) / rho_l
        rho_r = wp.max(right[0], tiny)
        u_r = right[1] / rho_r
        v_r = right[2] / rho_r
        e_r = right[3]
        p_r = wp.max(gm1 * (e_r - wp.float64(0.5) * rho_r * (u_r * u_r + v_r * v_r)), tiny)
        h_r = (e_r + p_r) / rho_r
        sqrt_l = wp.sqrt(rho_l)
        sqrt_r = wp.sqrt(rho_r)
        inv = wp.float64(1.0) / (sqrt_l + sqrt_r)
        u_roe = (sqrt_l * u_l + sqrt_r * u_r) * inv
        v_roe = (sqrt_l * v_l + sqrt_r * v_r) * inv
        h_roe = (sqrt_l * h_l + sqrt_r * h_r) * inv
        rho_roe = sqrt_l * sqrt_r
        q2_roe = u_roe * u_roe + v_roe * v_roe
        e_roe = (rho_roe * h_roe + wp.float64(0.5) * gm1 * rho_roe * q2_roe) / gamma
        return wp.vec4d(rho_roe, rho_roe * u_roe, rho_roe * v_roe, e_roe)


    @wp.func
    def jacobian_state_values(roe: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        tiny = wp.float64(1.0e-15)
        rho = wp.max(roe[0], tiny)
        u = roe[1] / rho
        v = roe[2] / rho
        gm1 = gamma - wp.float64(1.0)
        q2 = u * u + v * v
        p = wp.max(gm1 * (roe[3] - wp.float64(0.5) * rho * q2), tiny)
        a = wp.sqrt(gamma * p / rho)
        h = wp.float64(0.5) * q2 + a * a / gm1
        return wp.vec4d(u, v, a, h)


    @wp.func
    def con_to_char_x(q: wp.vec4d, roe: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        values = jacobian_state_values(roe, gamma)
        u = values[0]
        v = values[1]
        a = values[2]
        h = values[3]
        gm1 = gamma - wp.float64(1.0)
        a2 = a * a
        scale = gm1 / (wp.float64(2.0) * a2)
        c0 = ((h + a * (u - a) / gm1) * q[0] - (u + a / gm1) * q[1] - v * q[2] + q[3]) * scale
        c1 = ((-wp.float64(2.0) * h + wp.float64(4.0) * a2 / gm1) * q[0] + wp.float64(2.0) * u * q[1] + wp.float64(2.0) * v * q[2] - wp.float64(2.0) * q[3]) * scale
        c2 = (-wp.float64(2.0) * v * a2 / gm1 * q[0] + wp.float64(2.0) * a2 / gm1 * q[2]) * scale
        c3 = ((h - a * (u + a) / gm1) * q[0] + (-u + a / gm1) * q[1] - v * q[2] + q[3]) * scale
        return wp.vec4d(c0, c1, c2, c3)


    @wp.func
    def char_to_con_x(c: wp.vec4d, roe: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        values = jacobian_state_values(roe, gamma)
        u = values[0]
        v = values[1]
        a = values[2]
        h = values[3]
        q2 = u * u + v * v
        return wp.vec4d(
            c[0] + c[1] + c[3],
            (u - a) * c[0] + u * c[1] + (u + a) * c[3],
            v * c[0] + v * c[1] + c[2] + v * c[3],
            (h - u * a) * c[0] + wp.float64(0.5) * q2 * c[1] + v * c[2] + (h + u * a) * c[3],
        )


    @wp.func
    def con_to_char_y(q: wp.vec4d, roe: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        values = jacobian_state_values(roe, gamma)
        tangent = values[0]
        normal = values[1]
        a = values[2]
        h = values[3]
        gm1 = gamma - wp.float64(1.0)
        a2 = a * a
        scale = gm1 / (wp.float64(2.0) * a2)
        c0 = ((h + a * (normal - a) / gm1) * q[0] - tangent * q[1] - (normal + a / gm1) * q[2] + q[3]) * scale
        c1 = ((-wp.float64(2.0) * h + wp.float64(4.0) * a2 / gm1) * q[0] + wp.float64(2.0) * tangent * q[1] + wp.float64(2.0) * normal * q[2] - wp.float64(2.0) * q[3]) * scale
        c2 = (-wp.float64(2.0) * tangent * a2 / gm1 * q[0] + wp.float64(2.0) * a2 / gm1 * q[1]) * scale
        c3 = ((h - a * (normal + a) / gm1) * q[0] - tangent * q[1] + (-normal + a / gm1) * q[2] + q[3]) * scale
        return wp.vec4d(c0, c1, c2, c3)


    @wp.func
    def char_to_con_y(c: wp.vec4d, roe: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        values = jacobian_state_values(roe, gamma)
        u = values[0]
        v = values[1]
        a = values[2]
        h = values[3]
        q2 = u * u + v * v
        return wp.vec4d(
            c[0] + c[1] + c[3],
            u * c[0] + u * c[1] + c[2] + u * c[3],
            (v - a) * c[0] + v * c[1] + (v + a) * c[3],
            (h - v * a) * c[0] + wp.float64(0.5) * q2 * c[1] + u * c[2] + (h + v * a) * c[3],
        )


    @wp.func
    def con_to_char(q: wp.vec4d, roe: wp.vec4d, direction: int, gamma: wp.float64) -> wp.vec4d:
        if direction == 1:
            return con_to_char_x(q, roe, gamma)
        return con_to_char_y(q, roe, gamma)


    @wp.func
    def char_to_con(c: wp.vec4d, roe: wp.vec4d, direction: int, gamma: wp.float64) -> wp.vec4d:
        if direction == 1:
            return char_to_con_x(c, roe, gamma)
        return char_to_con_y(c, roe, gamma)


    @wp.func
    def weno7_beta(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, q5: wp.float64, q6: wp.float64) -> wp.vec4d:
        beta3 = wp.float64(2107.0)*q3*q3 - wp.float64(9402.0)*q4*q3 + wp.float64(7042.0)*q3*q5 - wp.float64(1854.0)*q3*q6 - wp.float64(17246.0)*q4*q5 - wp.float64(3882.0)*q5*q6 + wp.float64(11003.0)*q4*q4 + wp.float64(7043.0)*q5*q5 + wp.float64(547.0)*q6*q6 + wp.float64(4642.0)*q4*q6
        beta2 = wp.float64(267.0)*q5*q5 - wp.float64(1642.0)*q5*q4 + wp.float64(1602.0)*q5*q3 - wp.float64(494.0)*q5*q2 - wp.float64(2522.0)*q3*q2 + wp.float64(3443.0)*q3*q3 + wp.float64(547.0)*q2*q2 - wp.float64(5966.0)*q4*q3 + wp.float64(2843.0)*q4*q4 + wp.float64(1922.0)*q4*q2
        beta1 = wp.float64(267.0)*q1*q1 - wp.float64(494.0)*q4*q1 - wp.float64(1642.0)*q2*q1 + wp.float64(1602.0)*q3*q1 - wp.float64(5966.0)*q3*q2 + wp.float64(3443.0)*q3*q3 - wp.float64(2522.0)*q4*q3 + wp.float64(2843.0)*q2*q2 + wp.float64(547.0)*q4*q4 + wp.float64(1922.0)*q4*q2
        beta0 = wp.float64(2107.0)*q3*q3 + wp.float64(7042.0)*q3*q1 - wp.float64(1854.0)*q3*q0 - wp.float64(9402.0)*q3*q2 - wp.float64(17246.0)*q2*q1 + wp.float64(4642.0)*q2*q0 + wp.float64(547.0)*q0*q0 - wp.float64(3882.0)*q1*q0 + wp.float64(11003.0)*q2*q2 + wp.float64(7043.0)*q1*q1
        return wp.vec4d(beta0, beta1, beta2, beta3)


    @wp.func
    def stencil_weno7(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, location: int, dorder: int) -> wp.float64:
        r = wp.float64(0.0)
        if location == 5:
            if dorder == 0:
                r = wp.float64(25.0)/wp.float64(12.0)*q0 - wp.float64(23.0)/wp.float64(12.0)*q1 + wp.float64(13.0)/wp.float64(12.0)*q2 - wp.float64(1.0)/wp.float64(4.0)*q3
            elif dorder == 1:
                r = -wp.float64(35.0)/wp.float64(12.0)*q0 + wp.float64(23.0)/wp.float64(4.0)*q1 - wp.float64(15.0)/wp.float64(4.0)*q2 + wp.float64(11.0)/wp.float64(12.0)*q3
            elif dorder == 2:
                r = wp.float64(5.0)/wp.float64(2.0)*q0 - wp.float64(13.0)/wp.float64(2.0)*q1 + wp.float64(11.0)/wp.float64(2.0)*q2 - wp.float64(3.0)/wp.float64(2.0)*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 4:
            if dorder == 0:
                r = wp.float64(1.0)/wp.float64(4.0)*q0 + wp.float64(13.0)/wp.float64(12.0)*q1 - wp.float64(5.0)/wp.float64(12.0)*q2 + wp.float64(1.0)/wp.float64(12.0)*q3
            elif dorder == 1:
                r = -wp.float64(11.0)/wp.float64(12.0)*q0 + wp.float64(3.0)/wp.float64(4.0)*q1 + wp.float64(1.0)/wp.float64(4.0)*q2 - wp.float64(1.0)/wp.float64(12.0)*q3
            elif dorder == 2:
                r = wp.float64(3.0)/wp.float64(2.0)*q0 - wp.float64(7.0)/wp.float64(2.0)*q1 + wp.float64(5.0)/wp.float64(2.0)*q2 - wp.float64(1.0)/wp.float64(2.0)*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 3:
            if dorder == 0:
                r = -wp.float64(1.0)/wp.float64(12.0)*q0 + wp.float64(7.0)/wp.float64(12.0)*q1 + wp.float64(7.0)/wp.float64(12.0)*q2 - wp.float64(1.0)/wp.float64(12.0)*q3
            elif dorder == 1:
                r = wp.float64(1.0)/wp.float64(12.0)*q0 - wp.float64(5.0)/wp.float64(4.0)*q1 + wp.float64(5.0)/wp.float64(4.0)*q2 - wp.float64(1.0)/wp.float64(12.0)*q3
            elif dorder == 2:
                r = wp.float64(1.0)/wp.float64(2.0)*q0 - wp.float64(1.0)/wp.float64(2.0)*q1 - wp.float64(1.0)/wp.float64(2.0)*q2 + wp.float64(1.0)/wp.float64(2.0)*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 2:
            if dorder == 0:
                r = wp.float64(1.0)/wp.float64(12.0)*q0 - wp.float64(5.0)/wp.float64(12.0)*q1 + wp.float64(13.0)/wp.float64(12.0)*q2 + wp.float64(1.0)/wp.float64(4.0)*q3
            elif dorder == 1:
                r = wp.float64(1.0)/wp.float64(12.0)*q0 - wp.float64(1.0)/wp.float64(4.0)*q1 - wp.float64(3.0)/wp.float64(4.0)*q2 + wp.float64(11.0)/wp.float64(12.0)*q3
            elif dorder == 2:
                r = -wp.float64(1.0)/wp.float64(2.0)*q0 + wp.float64(5.0)/wp.float64(2.0)*q1 - wp.float64(7.0)/wp.float64(2.0)*q2 + wp.float64(3.0)/wp.float64(2.0)*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        else:
            if dorder == 0:
                r = -wp.float64(1.0)/wp.float64(4.0)*q0 + wp.float64(13.0)/wp.float64(12.0)*q1 - wp.float64(23.0)/wp.float64(12.0)*q2 + wp.float64(25.0)/wp.float64(12.0)*q3
            elif dorder == 1:
                r = -wp.float64(11.0)/wp.float64(12.0)*q0 + wp.float64(15.0)/wp.float64(4.0)*q1 - wp.float64(23.0)/wp.float64(4.0)*q2 + wp.float64(35.0)/wp.float64(12.0)*q3
            elif dorder == 2:
                r = -wp.float64(3.0)/wp.float64(2.0)*q0 + wp.float64(11.0)/wp.float64(2.0)*q1 - wp.float64(13.0)/wp.float64(2.0)*q2 + wp.float64(5.0)/wp.float64(2.0)*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        return r


    @wp.func
    def stencil_weno7_2gauss(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, location: int, dorder: int) -> wp.float64:
        r = wp.float64(0.0)
        s = wp.sqrt(wp.float64(3.0))
        if location == 1:
            if dorder == 0:
                r = (-wp.float64(11.0)*s/wp.float64(216.0))*q0 + (wp.float64(17.0)*s/wp.float64(72.0))*q1 - (wp.float64(35.0)*s/wp.float64(72.0))*q2 + ((wp.float64(65.0)*s + wp.float64(216.0))/wp.float64(216.0))*q3
            elif dorder == 1:
                r = (-(s + wp.float64(2.0))/wp.float64(6.0))*q0 + ((wp.float64(4.0)*s + wp.float64(9.0))/wp.float64(6.0))*q1 + (-(wp.float64(5.0)*s + wp.float64(18.0))/wp.float64(6.0))*q2 + ((wp.float64(2.0)*s + wp.float64(11.0))/wp.float64(6.0))*q3
            elif dorder == 2:
                r = (-(s + wp.float64(6.0))/wp.float64(6.0))*q0 + ((s + wp.float64(8.0))/wp.float64(2.0))*q1 + (-(s + wp.float64(10.0))/wp.float64(2.0))*q2 + ((s + wp.float64(12.0))/wp.float64(6.0))*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 2:
            if dorder == 0:
                r = (wp.float64(7.0)*s/wp.float64(216.0))*q0 - (wp.float64(13.0)*s/wp.float64(72.0))*q1 + ((wp.float64(7.0)*s + wp.float64(72.0))/wp.float64(72.0))*q2 + (wp.float64(11.0)*s/wp.float64(216.0))*q3
            elif dorder == 1:
                r = wp.float64(1.0)/wp.float64(6.0)*q0 + ((s - wp.float64(6.0))/wp.float64(6.0))*q1 + ((wp.float64(3.0) - wp.float64(2.0)*s)/wp.float64(6.0))*q2 + ((s + wp.float64(2.0))/wp.float64(6.0))*q3
            elif dorder == 2:
                r = (-s/wp.float64(6.0))*q0 + ((s + wp.float64(2.0))/wp.float64(2.0))*q1 + (-(s + wp.float64(4.0))/wp.float64(2.0))*q2 + ((s + wp.float64(6.0))/wp.float64(6.0))*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 3:
            if dorder == 0:
                r = (-wp.float64(11.0)*s/wp.float64(216.0))*q0 + ((wp.float64(72.0) - wp.float64(7.0)*s)/wp.float64(72.0))*q1 + (wp.float64(13.0)*s/wp.float64(72.0))*q2 - (wp.float64(7.0)*s/wp.float64(216.0))*q3
            elif dorder == 1:
                r = ((s - wp.float64(2.0))/wp.float64(6.0))*q0 + (-(wp.float64(2.0)*s + wp.float64(3.0))/wp.float64(6.0))*q1 + ((s + wp.float64(6.0))/wp.float64(6.0))*q2 - wp.float64(1.0)/wp.float64(6.0)*q3
            elif dorder == 2:
                r = ((wp.float64(6.0) - s)/wp.float64(6.0))*q0 + ((s - wp.float64(4.0))/wp.float64(2.0))*q1 + ((wp.float64(2.0) - s)/wp.float64(2.0))*q2 + s/wp.float64(6.0)*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 4:
            if dorder == 0:
                r = ((wp.float64(216.0) - wp.float64(65.0)*s)/wp.float64(216.0))*q0 + (wp.float64(35.0)*s/wp.float64(72.0))*q1 - (wp.float64(17.0)*s/wp.float64(72.0))*q2 + (wp.float64(11.0)*s/wp.float64(216.0))*q3
            elif dorder == 1:
                r = ((wp.float64(2.0)*s - wp.float64(11.0))/wp.float64(6.0))*q0 + ((wp.float64(18.0) - wp.float64(5.0)*s)/wp.float64(6.0))*q1 + ((wp.float64(4.0)*s - wp.float64(9.0))/wp.float64(6.0))*q2 + ((wp.float64(2.0) - s)/wp.float64(6.0))*q3
            elif dorder == 2:
                r = ((wp.float64(12.0) - s)/wp.float64(6.0))*q0 + ((s - wp.float64(10.0))/wp.float64(2.0))*q1 + ((wp.float64(8.0) - s)/wp.float64(2.0))*q2 + ((s - wp.float64(6.0))/wp.float64(6.0))*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 5:
            if dorder == 0:
                r = (wp.float64(11.0)*s/wp.float64(216.0))*q0 - (wp.float64(17.0)*s/wp.float64(72.0))*q1 + (wp.float64(35.0)*s/wp.float64(72.0))*q2 + ((wp.float64(216.0) - wp.float64(65.0)*s)/wp.float64(216.0))*q3
            elif dorder == 1:
                r = ((s - wp.float64(2.0))/wp.float64(6.0))*q0 + ((wp.float64(9.0) - wp.float64(4.0)*s)/wp.float64(6.0))*q1 + ((wp.float64(5.0)*s - wp.float64(18.0))/wp.float64(6.0))*q2 + ((wp.float64(11.0) - wp.float64(2.0)*s)/wp.float64(6.0))*q3
            elif dorder == 2:
                r = ((s - wp.float64(6.0))/wp.float64(6.0))*q0 + ((wp.float64(8.0) - s)/wp.float64(2.0))*q1 + ((s - wp.float64(10.0))/wp.float64(2.0))*q2 + ((wp.float64(12.0) - s)/wp.float64(6.0))*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 6:
            if dorder == 0:
                r = (-wp.float64(7.0)*s/wp.float64(216.0))*q0 + (wp.float64(13.0)*s/wp.float64(72.0))*q1 + ((wp.float64(72.0) - wp.float64(7.0)*s)/wp.float64(72.0))*q2 - (wp.float64(11.0)*s/wp.float64(216.0))*q3
            elif dorder == 1:
                r = wp.float64(1.0)/wp.float64(6.0)*q0 + (-(s + wp.float64(6.0))/wp.float64(6.0))*q1 + ((wp.float64(2.0)*s + wp.float64(3.0))/wp.float64(6.0))*q2 + ((wp.float64(2.0) - s)/wp.float64(6.0))*q3
            elif dorder == 2:
                r = s/wp.float64(6.0)*q0 + ((wp.float64(2.0) - s)/wp.float64(2.0))*q1 + ((s - wp.float64(4.0))/wp.float64(2.0))*q2 + ((wp.float64(6.0) - s)/wp.float64(6.0))*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        elif location == 7:
            if dorder == 0:
                r = (wp.float64(11.0)*s/wp.float64(216.0))*q0 + ((wp.float64(7.0)*s + wp.float64(72.0))/wp.float64(72.0))*q1 - (wp.float64(13.0)*s/wp.float64(72.0))*q2 + (wp.float64(7.0)*s/wp.float64(216.0))*q3
            elif dorder == 1:
                r = (-(s + wp.float64(2.0))/wp.float64(6.0))*q0 + ((wp.float64(2.0)*s - wp.float64(3.0))/wp.float64(6.0))*q1 + ((wp.float64(6.0) - s)/wp.float64(6.0))*q2 - wp.float64(1.0)/wp.float64(6.0)*q3
            elif dorder == 2:
                r = ((s + wp.float64(6.0))/wp.float64(6.0))*q0 + (-(s + wp.float64(4.0))/wp.float64(2.0))*q1 + ((s + wp.float64(2.0))/wp.float64(2.0))*q2 - s/wp.float64(6.0)*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        else:
            if dorder == 0:
                r = ((wp.float64(65.0)*s + wp.float64(216.0))/wp.float64(216.0))*q0 - (wp.float64(35.0)*s/wp.float64(72.0))*q1 + (wp.float64(17.0)*s/wp.float64(72.0))*q2 - (wp.float64(11.0)*s/wp.float64(216.0))*q3
            elif dorder == 1:
                r = (-(wp.float64(2.0)*s + wp.float64(11.0))/wp.float64(6.0))*q0 + ((wp.float64(5.0)*s + wp.float64(18.0))/wp.float64(6.0))*q1 - ((wp.float64(4.0)*s + wp.float64(9.0))/wp.float64(6.0))*q2 + ((s + wp.float64(2.0))/wp.float64(6.0))*q3
            elif dorder == 2:
                r = ((s + wp.float64(12.0))/wp.float64(6.0))*q0 - ((s + wp.float64(10.0))/wp.float64(2.0))*q1 + ((s + wp.float64(8.0))/wp.float64(2.0))*q2 - ((s + wp.float64(6.0))/wp.float64(6.0))*q3
            else:
                r = -q0 + wp.float64(3.0)*q1 - wp.float64(3.0)*q2 + q3
        return r


    @wp.func
    def scale_derivative(v: wp.float64, h: wp.float64, dorder: int) -> wp.float64:
        out = v
        if dorder == 1:
            out = v / h
        elif dorder == 2:
            out = v / (h * h)
        elif dorder == 3:
            out = v / (h * h * h)
        return out


    @wp.func
    def weno7_optimal_weights(lr: int, gauss: int) -> wp.vec4d:
        root3 = wp.sqrt(wp.float64(3.0))
        if gauss == 1:
            if lr == 1:
                return wp.vec4d(
                    wp.float64(59.0) / wp.float64(880.0) - wp.float64(5.0) * root3 / wp.float64(16632.0),
                    wp.float64(381.0) / wp.float64(880.0) - wp.float64(587.0) * root3 / wp.float64(194040.0),
                    wp.float64(381.0) / wp.float64(880.0) + wp.float64(587.0) * root3 / wp.float64(194040.0),
                    wp.float64(59.0) / wp.float64(880.0) + wp.float64(5.0) * root3 / wp.float64(16632.0),
                )
            return wp.vec4d(
                wp.float64(59.0) / wp.float64(880.0) + wp.float64(5.0) * root3 / wp.float64(16632.0),
                wp.float64(381.0) / wp.float64(880.0) + wp.float64(587.0) * root3 / wp.float64(194040.0),
                wp.float64(381.0) / wp.float64(880.0) - wp.float64(587.0) * root3 / wp.float64(194040.0),
                wp.float64(59.0) / wp.float64(880.0) - wp.float64(5.0) * root3 / wp.float64(16632.0),
            )
        if lr == 1:
            return wp.vec4d(
                wp.float64(1.0) / wp.float64(35.0),
                wp.float64(12.0) / wp.float64(35.0),
                wp.float64(18.0) / wp.float64(35.0),
                wp.float64(4.0) / wp.float64(35.0),
            )
        return wp.vec4d(
            wp.float64(4.0) / wp.float64(35.0),
            wp.float64(18.0) / wp.float64(35.0),
            wp.float64(12.0) / wp.float64(35.0),
            wp.float64(1.0) / wp.float64(35.0),
        )


    @wp.func
    def weno7_lr_scalar(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, q5: wp.float64, q6: wp.float64, lr: int, h: wp.float64, dorder: int) -> wp.float64:
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        s3 = wp.float64(0.0)
        d0 = wp.float64(0.0)
        d1 = wp.float64(0.0)
        d2 = wp.float64(0.0)
        d3 = wp.float64(0.0)
        if lr == 1:
            s0 = stencil_weno7(q0, q1, q2, q3, 1, dorder)
            s1 = stencil_weno7(q1, q2, q3, q4, 2, dorder)
            s2 = stencil_weno7(q2, q3, q4, q5, 3, dorder)
            s3 = stencil_weno7(q3, q4, q5, q6, 4, dorder)
            d0 = wp.float64(1.0) / wp.float64(35.0)
            d1 = wp.float64(12.0) / wp.float64(35.0)
            d2 = wp.float64(18.0) / wp.float64(35.0)
            d3 = wp.float64(4.0) / wp.float64(35.0)
        else:
            s0 = stencil_weno7(q0, q1, q2, q3, 2, dorder)
            s1 = stencil_weno7(q1, q2, q3, q4, 3, dorder)
            s2 = stencil_weno7(q2, q3, q4, q5, 4, dorder)
            s3 = stencil_weno7(q3, q4, q5, q6, 5, dorder)
            d0 = wp.float64(4.0) / wp.float64(35.0)
            d1 = wp.float64(18.0) / wp.float64(35.0)
            d2 = wp.float64(12.0) / wp.float64(35.0)
            d3 = wp.float64(1.0) / wp.float64(35.0)
        beta = weno7_beta(q0, q1, q2, q3, q4, q5, q6)
        a0 = d0 * safe_rcp(beta[0]) * safe_rcp(beta[0])
        a1 = d1 * safe_rcp(beta[1]) * safe_rcp(beta[1])
        a2 = d2 * safe_rcp(beta[2]) * safe_rcp(beta[2])
        a3 = d3 * safe_rcp(beta[3]) * safe_rcp(beta[3])
        asum = a0 + a1 + a2 + a3
        return scale_derivative((a0 * s0 + a1 * s1 + a2 * s2 + a3 * s3) / asum, h, dorder)


    @wp.func
    def weno7_gauss_lr_scalar(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, q5: wp.float64, q6: wp.float64, lr: int, h: wp.float64, dorder: int) -> wp.float64:
        root3 = wp.sqrt(wp.float64(3.0))
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        s3 = wp.float64(0.0)
        d0 = wp.float64(0.0)
        d1 = wp.float64(0.0)
        d2 = wp.float64(0.0)
        d3 = wp.float64(0.0)
        if lr == 1:
            s0 = stencil_weno7_2gauss(q0, q1, q2, q3, 1, dorder)
            s1 = stencil_weno7_2gauss(q1, q2, q3, q4, 2, dorder)
            s2 = stencil_weno7_2gauss(q2, q3, q4, q5, 3, dorder)
            s3 = stencil_weno7_2gauss(q3, q4, q5, q6, 4, dorder)
            d0 = wp.float64(59.0)/wp.float64(880.0) - wp.float64(5.0)*root3/wp.float64(16632.0)
            d1 = wp.float64(381.0)/wp.float64(880.0) - wp.float64(587.0)*root3/wp.float64(194040.0)
            d2 = wp.float64(587.0)*root3/wp.float64(194040.0) + wp.float64(381.0)/wp.float64(880.0)
            d3 = wp.float64(5.0)*root3/wp.float64(16632.0) + wp.float64(59.0)/wp.float64(880.0)
        else:
            s0 = stencil_weno7_2gauss(q0, q1, q2, q3, 5, dorder)
            s1 = stencil_weno7_2gauss(q1, q2, q3, q4, 6, dorder)
            s2 = stencil_weno7_2gauss(q2, q3, q4, q5, 7, dorder)
            s3 = stencil_weno7_2gauss(q3, q4, q5, q6, 8, dorder)
            d0 = wp.float64(5.0)*root3/wp.float64(16632.0) + wp.float64(59.0)/wp.float64(880.0)
            d1 = wp.float64(587.0)*root3/wp.float64(194040.0) + wp.float64(381.0)/wp.float64(880.0)
            d2 = wp.float64(381.0)/wp.float64(880.0) - wp.float64(587.0)*root3/wp.float64(194040.0)
            d3 = wp.float64(59.0)/wp.float64(880.0) - wp.float64(5.0)*root3/wp.float64(16632.0)
        beta = weno7_beta(q0, q1, q2, q3, q4, q5, q6)
        a0 = d0 * safe_rcp(beta[0]) * safe_rcp(beta[0])
        a1 = d1 * safe_rcp(beta[1]) * safe_rcp(beta[1])
        a2 = d2 * safe_rcp(beta[2]) * safe_rcp(beta[2])
        a3 = d3 * safe_rcp(beta[3]) * safe_rcp(beta[3])
        asum = a0 + a1 + a2 + a3
        return scale_derivative((a0 * s0 + a1 * s1 + a2 * s2 + a3 * s3) / asum, h, dorder)


    @wp.func
    def weno7_lr_vec_conservative(q0: wp.vec4d, q1: wp.vec4d, q2: wp.vec4d, q3: wp.vec4d, q4: wp.vec4d, q5: wp.vec4d, q6: wp.vec4d, lr: int, h: wp.float64, dorder: int) -> wp.vec4d:
        return wp.vec4d(
            weno7_lr_scalar(q0[0], q1[0], q2[0], q3[0], q4[0], q5[0], q6[0], lr, h, dorder),
            weno7_lr_scalar(q0[1], q1[1], q2[1], q3[1], q4[1], q5[1], q6[1], lr, h, dorder),
            weno7_lr_scalar(q0[2], q1[2], q2[2], q3[2], q4[2], q5[2], q6[2], lr, h, dorder),
            weno7_lr_scalar(q0[3], q1[3], q2[3], q3[3], q4[3], q5[3], q6[3], lr, h, dorder),
        )


    @wp.func
    def weno7_gauss_vec_conservative(q0: wp.vec4d, q1: wp.vec4d, q2: wp.vec4d, q3: wp.vec4d, q4: wp.vec4d, q5: wp.vec4d, q6: wp.vec4d, lr: int, h: wp.float64, dorder: int) -> wp.vec4d:
        return wp.vec4d(
            weno7_gauss_lr_scalar(q0[0], q1[0], q2[0], q3[0], q4[0], q5[0], q6[0], lr, h, dorder),
            weno7_gauss_lr_scalar(q0[1], q1[1], q2[1], q3[1], q4[1], q5[1], q6[1], lr, h, dorder),
            weno7_gauss_lr_scalar(q0[2], q1[2], q2[2], q3[2], q4[2], q5[2], q6[2], lr, h, dorder),
            weno7_gauss_lr_scalar(q0[3], q1[3], q2[3], q3[3], q4[3], q5[3], q6[3], lr, h, dorder),
        )


    @wp.func
    def weno7_lr_vec_characteristic(q0: wp.vec4d, q1: wp.vec4d, q2: wp.vec4d, q3: wp.vec4d, q4: wp.vec4d, q5: wp.vec4d, q6: wp.vec4d, lr: int, h: wp.float64, dorder: int, direction: int, gamma: wp.float64) -> wp.vec4d:
        roe = roe_average_state(q3, q4, gamma)
        if lr == 2:
            roe = roe_average_state(q2, q3, gamma)
        c0 = con_to_char(q0, roe, direction, gamma)
        c1 = con_to_char(q1, roe, direction, gamma)
        c2 = con_to_char(q2, roe, direction, gamma)
        c3 = con_to_char(q3, roe, direction, gamma)
        c4 = con_to_char(q4, roe, direction, gamma)
        c5 = con_to_char(q5, roe, direction, gamma)
        c6 = con_to_char(q6, roe, direction, gamma)
        cf = wp.vec4d(
            weno7_lr_scalar(c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0], lr, h, dorder),
            weno7_lr_scalar(c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1], lr, h, dorder),
            weno7_lr_scalar(c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], lr, h, dorder),
            weno7_lr_scalar(c0[3], c1[3], c2[3], c3[3], c4[3], c5[3], c6[3], lr, h, dorder),
        )
        return char_to_con(cf, roe, direction, gamma)


    @wp.func
    def after_dritq_2d(qt: wp.vec4d, qtt: wp.vec4d, qttt: wp.vec4d, u0: wp.vec4d, loca: int, dt: wp.float64) -> wp.vec4d:
        xi = wp.sqrt(wp.float64(15.0)) / wp.float64(5.0)
        tau = wp.float64(0.5) * dt
        if loca == 1:
            tau = wp.float64(0.5) * dt * (wp.float64(1.0) - xi)
        elif loca == 3:
            tau = wp.float64(0.5) * dt * (wp.float64(1.0) + xi)
        return wp.vec4d(
            u0[0] + qt[0] * tau + wp.float64(0.5) * qtt[0] * tau * tau + qttt[0] * tau * tau * tau / wp.float64(6.0),
            u0[1] + qt[1] * tau + wp.float64(0.5) * qtt[1] * tau * tau + qttt[1] * tau * tau * tau / wp.float64(6.0),
            u0[2] + qt[2] * tau + wp.float64(0.5) * qtt[2] * tau * tau + qttt[2] * tau * tau * tau / wp.float64(6.0),
            u0[3] + qt[3] * tau + wp.float64(0.5) * qtt[3] * tau * tau + qttt[3] * tau * tau * tau / wp.float64(6.0),
        )


    @wp.func
    def compute_euler_time_derivatives_2d_order3(
        U: wp.vec4d,
        Ux: wp.vec4d,
        Uy: wp.vec4d,
        Uxx: wp.vec4d,
        Uxy: wp.vec4d,
        Uyy: wp.vec4d,
        Uxxx: wp.vec4d,
        Uxxy: wp.vec4d,
        Uxyy: wp.vec4d,
        Uyyy: wp.vec4d,
        gamma: wp.float64,
    ):

        # Aliases
        rho = U[0]
        m = U[1]
        n = U[2]
        E = U[3]
        inv_rho = wp.float64(1.0) / rho
        p = (gamma - wp.float64(1.0)) * (E - wp.float64(0.5)*(m*m + n*n)*inv_rho)
        rho_x = Ux[0]
        m_x = Ux[1]
        n_x = Ux[2]
        E_x = Ux[3]
        rho_y = Uy[0]
        m_y = Uy[1]
        n_y = Uy[2]
        E_y = Uy[3]
        rho_xx = Uxx[0]
        m_xx = Uxx[1]
        n_xx = Uxx[2]
        E_xx = Uxx[3]
        rho_xy = Uxy[0]
        m_xy = Uxy[1]
        n_xy = Uxy[2]
        E_xy = Uxy[3]
        rho_yy = Uyy[0]
        m_yy = Uyy[1]
        n_yy = Uyy[2]
        E_yy = Uyy[3]
        rho_xxx = Uxxx[0]
        m_xxx = Uxxx[1]
        n_xxx = Uxxx[2]
        E_xxx = Uxxx[3]
        rho_xxy = Uxxy[0]
        m_xxy = Uxxy[1]
        n_xxy = Uxxy[2]
        E_xxy = Uxxy[3]
        rho_xyy = Uxyy[0]
        m_xyy = Uxyy[1]
        n_xyy = Uxyy[2]
        E_xyy = Uxyy[3]
        rho_yyy = Uyyy[0]
        m_yyy = Uyyy[1]
        n_yyy = Uyyy[2]
        E_yyy = Uyyy[3]

        # CSE Temporaries (1193)
        T0 = m_x + n_y
        T1 = gamma - wp.float64(1)
        T2 = E_x*T1
        T3 = ((m)*(m))
        T4 = wp.float64(1.0)*T3
        T5 = ((n)*(n))
        T6 = T3 + T5
        T7 = T1*T6
        T8 = wp.float64(0.5)*T7
        T9 = -T8
        T10 = T4 + T9
        # const double T11 = pow(rho, -2);
        T11 = wp.float64(1.0)/(rho*rho)
        T12 = T11*rho_x
        T13 = wp.float64(1.0)/rho
        T14 = wp.float64(1.0)*gamma
        T15 = T14 - wp.float64(3.0)
        T16 = T15*m_x
        T17 = T16*m
        T18 = wp.float64(1.0)*n_y
        T19 = T18*m
        T20 = T13*T19
        T21 = wp.float64(1.0)*m_y
        T22 = T21*n
        T23 = T13*T22
        T24 = T11*rho_y
        T25 = wp.float64(1.0)*m
        T26 = T25*n
        T27 = T24*T26
        T28 = T13*n
        T29 = T1*n_x
        T30 = wp.float64(1.0)*T29
        T31 = T28*T30
        T32 = -T20 - T23 + T27 + T31
        T33 = E_y*T1
        T34 = wp.float64(1.0)*T5
        T35 = T34 + T9
        T36 = T15*n_y
        T37 = T36*n
        T38 = wp.float64(1.0)*n_x
        T39 = T38*m
        T40 = T13*T39
        T41 = wp.float64(1.0)*m_x
        T42 = T41*n
        T43 = T13*T42
        T44 = T13*m
        T45 = T1*m_y
        T46 = wp.float64(1.0)*T45
        T47 = T44*T46
        T48 = T12*T26
        T49 = -T40 - T43 + T47 + T48
        T50 = T1*T3
        T51 = T13*T50
        T52 = T1*(E - T13*(wp.float64(0.5)*T3 + wp.float64(0.5)*T5))
        T53 = E + T52
        T54 = -T51 + T53
        T55 = -T54
        T56 = T1*T5
        T57 = T13*T56
        T58 = -T57
        T59 = T53 + T58
        T60 = -T59
        T61 = T13*T8
        T62 = -T61
        T63 = wp.float64(1.0)*E + wp.float64(1.0)*T52
        T64 = T62 + T63
        T65 = T13*rho_x
        T66 = T65*m
        T67 = T13*rho_y
        T68 = T67*n
        T69 = wp.float64(1.0)*n
        T70 = T44*T45
        T71 = T69*T70
        T72 = T29*T69
        T73 = T44*T72
        T74 = E_x*T14
        T75 = T74*m
        T76 = E_y*T14
        T77 = T76*n
        T78 = -T75 - T77
        T79 = E_xx*T1
        T80 = E_yy*T1
        T81 = wp.float64(2.0)*n_xy
        T82 = T44*T81
        T83 = wp.float64(2.0)*m_xy
        T84 = T28*T83
        T85 = T11*rho_xy
        T86 = wp.float64(2.0)*m
        T87 = T86*n
        T88 = T85*T87
        T89 = wp.float64(1.0)*T1
        T90 = T89*m_yy
        T91 = T44*T90
        T92 = T89*n_xx
        T93 = T28*T92
        T94 = T15*m_xx
        T95 = T44*T94
        T96 = T15*n_yy
        T97 = T28*T96
        T98 = T11*rho_xx
        T99 = T10*T98
        T100 = T11*rho_yy
        T101 = T100*T35
        T102 = T65*n
        T103 = T67*m
        T104 = T1*T103
        T105 = -T102 + T104 - T45 + n_x
        T106 = T13*T21
        T107 = T1*T102
        T108 = T103 - T107 + T29 - m_y
        T109 = T13*T38
        T110 = T108*T109
        T111 = T67*T69
        T112 = -T15
        T113 = T112*m_x
        T114 = T15*T66
        T115 = T113 + T114
        T116 = -T111 + T115
        T117 = T116 + T18
        T118 = T117*T13
        T119 = T25*T65
        T120 = T112*n_y
        T121 = T15*T68
        T122 = T120 + T121
        T123 = -T119 + T122 + T41
        T124 = T123*T13
        T125 = wp.float64(2.0)*T3
        T126 = wp.float64(1.0)*T7
        T127 = -T126
        T128 = T125 + T127
        T129 = T128*T13
        T130 = T129*rho_x
        T131 = wp.float64(2.0)*n
        T132 = T103*T131 - T19 - T22 + T72
        T133 = T130 + T132 + T17
        T134 = wp.float64(2.0)*T5
        T135 = T127 + T134
        T136 = T13*T135
        T137 = T136*rho_y
        T138 = T102*T86 + T25*T45 - T39 - T42
        T139 = T137 + T138 + T37
        T140 = -T10
        T141 = T12*T140
        T142 = T113*T44
        T143 = T20 + T23 - T27
        T144 = T143 - T31
        T145 = T141 + T142 + T144 + T2
        T146 = T145*T15
        T147 = T15*m
        T148 = T28*T89
        T149 = T105*T148 + T118*T147 + T146
        T150 = -T74
        T151 = wp.float64(3.0)*m_x
        T152 = T1*T44
        T153 = T1*T87
        T154 = T125*T13
        T155 = T1*T154
        T156 = -T155
        T157 = T156 + T64
        T158 = -T157
        T159 = T1*T20 + T150 + T151*T152 - T153*T24 - T158*T65 + T28*T46 + T31
        T160 = T1*T159
        T161 = T0*T15
        T162 = T161*T44
        T163 = T160 - T162
        T164 = ((T15)*(T15))
        T165 = T13*T3
        T166 = T164*T165
        T167 = T1*T55
        T168 = wp.float64(1.0)*T167
        T169 = T13*T4
        T170 = T13*T34
        T171 = T1*T170
        T172 = T171 + T62
        T173 = T168 + T169 + T172
        T174 = -T35
        T175 = T174*T24
        T176 = T40 + T43 - T48
        T177 = T176 - T47
        T178 = T120*T28 + T175 + T177 + T33
        T179 = T1*T178
        T180 = wp.float64(1.0)*T179
        T181 = T1*T69
        T182 = T124*T181
        T183 = wp.float64(1.0)*T108
        T184 = T15*T44
        T185 = T183*T184
        T186 = -T76
        T187 = wp.float64(3.0)*n_y
        T188 = T1*T28
        T189 = T13*T134
        T190 = T1*T189
        T191 = -T190
        T192 = T191 + T64
        T193 = -T192
        T194 = T1*T43 - T12*T153 + T186 + T187*T188 - T193*T67 + T30*T44 + T47
        T195 = T1*T194
        T196 = T0*T13
        T197 = T196*n
        T198 = T197*T89
        T199 = T195 - T198
        T200 = T1*T60
        T201 = wp.float64(1.0)*T200
        T202 = T112*T169
        T203 = T15*T171
        T204 = T202 + T203
        T205 = -T169 - T201 + T204 + T61
        T206 = T105*T44
        T207 = T118*n
        T208 = T178 - T197
        T209 = -T206 - T207 - T208
        T210 = T124*m
        T211 = T196*m
        T212 = T145 - T211
        T213 = -T108*T13*n + T212
        T214 = -T210 - T213
        T215 = T13*T7
        T216 = -T215 + T53
        T217 = wp.float64(2.0)*T66
        T218 = wp.float64(2.0)*T68
        T219 = wp.float64(2.0)*T45
        T220 = T44*n
        T221 = wp.float64(2.0)*T29
        T222 = T221*n
        T223 = T158*m_x + T193*n_y + T216*T217 + T216*T218 + T219*T220 + T222*T44 + T78
        T224 = T1*T223
        T225 = T139*T148
        T226 = -T133*T184 - T225
        T227 = T0*T129
        T228 = T180*n
        T229 = T146*m + T227 + T228
        T230 = T197*T86
        T231 = wp.float64(1.0)*T44
        T232 = wp.float64(1.0)*T28
        T233 = T178*T25
        T234 = T145*T69
        T235 = -T133*T232 - T139*T231 - T230 + T233 + T234
        T236 = T1*T14
        T237 = -T1
        T238 = T112*T237
        T239 = T236 - T238
        T240 = T3 - T56
        T241 = T10*T13
        T242 = T15*T241
        T243 = T1*T64
        T244 = -T243
        T245 = T44*rho_xx
        T246 = T15*T169
        T247 = T13*T35
        T248 = T1*T247
        T249 = wp.float64(1.0)*T248
        T250 = T246 + T249
        T251 = T28*rho_xy
        T252 = T83*n
        T253 = T184*T252
        T254 = T13*m_yy
        T255 = -T5 + T50
        T256 = wp.float64(1.0)*T255
        T257 = -m_x
        T258 = T257 + T66 + T68 - n_y
        T259 = T14*T2
        T260 = T134 + T9
        T261 = T25*T260
        T262 = T125 + T9
        T263 = T262*T69
        T264 = wp.float64(1.0)*n_yy
        T265 = T15*n
        T266 = T265*T44
        T267 = T264*T266
        T268 = wp.float64(3.0)*T1
        T269 = T220*T268
        T270 = T100*T261 + T254*T256 + T258*T259 + T263*T85 + T267 + T269*n_xx
        T271 = T1*T145
        T272 = wp.float64(1.0)*T271
        T273 = T1*T25
        T274 = wp.float64(1.0)*T105
        T275 = T15*T28
        T276 = T118*T273 + T272 + T274*T275
        T277 = T211*T89
        T278 = T160 - T277
        T279 = T15*T178
        T280 = T124*T265
        T281 = T161*T28
        T282 = T108*T89
        T283 = T282*T44
        T284 = T195 - T281 - T283
        T285 = T13*T5
        T286 = T164*T285
        T287 = T1*T169
        T288 = T287 + T62
        T289 = T170 + T201 + T288
        T290 = T112*T170
        T291 = T1*T246
        T292 = T290 + T291
        T293 = -T168 - T170 + T292 + T61
        T294 = T44*T89
        T295 = T133*T294
        T296 = -T139*T275 - T295
        T297 = T0*T136
        T298 = T25*T271
        T299 = T279*n + T297 + T298
        T300 = T15*T170
        T301 = T1*T241
        T302 = wp.float64(1.0)*T301
        T303 = T300 + T302
        T304 = T44*rho_xy
        T305 = T15*T247
        T306 = T28*rho_yy
        T307 = T265*T82
        T308 = wp.float64(1.0)*m_xy
        T309 = T13*T255
        T310 = T14*T33
        T311 = wp.float64(1.0)*m_xx
        T312 = T266*T311
        T313 = T258*T310 + T261*T85 + T263*T98 + T269*m_yy + T308*T309 + T312
        T314 = ((gamma)*(gamma))
        T315 = T165*T314
        T316 = -T237*T54 + T315
        T317 = T285*T314
        T318 = -T237*T59 + T317
        T319 = T0*T158
        T320 = -T319
        T321 = wp.float64(1.0)*T55
        T322 = wp.float64(3.0)*m
        T323 = -T64
        T324 = T18*T59 + T323*T66 + T323*T68 + T41*T54 - T71 - T73 + T75 + T77
        T325 = T14*T324
        T326 = -T325
        T327 = T14*T159
        T328 = T327*m
        T329 = T326 + T328
        T330 = T13*m_x
        T331 = wp.float64(1.0)*T60
        T332 = T327*n
        T333 = T1*T197
        T334 = T333*T86
        T335 = -T334
        T336 = T332 + T335
        T337 = T180*m
        T338 = T271*T69
        T339 = T337 + T338
        T340 = T13*m_y
        T341 = T14*T194
        T342 = T341*m
        T343 = T335 + T342
        T344 = T13*n_x
        T345 = T179*n
        T346 = T341*n
        T347 = T326 + T346
        T348 = T0*T193
        T349 = T181*T44
        T350 = T108*T349
        T351 = -T348 - T350
        T352 = T13*n_y
        T353 = T14*T223
        T354 = T353*T44
        T355 = wp.float64(2.0)*T211
        T356 = T216*T355
        T357 = T325*T44
        T358 = T131*T179
        T359 = T145*T158 + T356 - T357 + T358*T44
        T360 = T28*T353
        T361 = wp.float64(2.0)*T197
        T362 = T216*T361
        T363 = T28*T325
        T364 = T131*T271
        T365 = T178*T193 + T362 - T363 + T364*T44
        T366 = T165*T56
        T367 = wp.float64(1.0)*T98
        T368 = wp.float64(1.0)*T100
        T369 = T15*T321
        T370 = T14*T55
        T371 = T171 + T64
        T372 = T370 + T371
        T373 = T44*m_xx
        T374 = T15*T331
        T375 = T14*T60
        T376 = T287 + T64
        T377 = T375 + T376
        T378 = T28*n_yy
        T379 = -T203
        T380 = T375 + T379
        T381 = T288 + T380
        T382 = -T291
        T383 = T370 + T382
        T384 = T172 + T383
        T385 = ((T1)*(T1))
        T386 = T285*T385
        T387 = T57*gamma
        T388 = T386 - T387
        T389 = T388 + T59
        T390 = T165*T385
        T391 = T51*gamma
        T392 = T390 - T391 + T54
        T393 = T285*gamma
        T394 = T285 + T393
        T395 = T394 + T59
        T396 = T64*gamma
        T397 = T26*T85
        T398 = T165*gamma
        T399 = T165 + T398 + T54
        T400 = T1*T237
        T401 = T314 + T400
        T402 = E_xy*T401
        T403 = T131*T402
        T404 = T395*T91 + T397*(-T248 + T396 + T54) + T397*(-T301 + T396 + T59) + T399*T93 - T403*T44 + T74*(T13*T258*gamma*m - T212) + T76*(T13*T258*gamma*n - T208)
        T405 = E_xxx*T1
        T406 = E_xxy*n
        T407 = E_xyy*m
        T408 = E_yyy*n
        T409 = T44*m_xxy
        T410 = wp.float64(3.0)*T265
        T411 = T44*n_xyy
        T412 = T13*n_xxy
        T413 = wp.float64(1.0)*T240
        T414 = T13*m_xyy
        T415 = T11*rho_xyy
        T416 = T11*rho_xxy
        T417 = T14 - wp.float64(1.0)
        T418 = T417*n_x
        T419 = -T102*T89
        T420 = wp.float64(1.0)*T237
        T421 = -T103*T420 + wp.float64(1.0)*T237*m_y - T418 - T419
        T422 = T417*m_y
        T423 = T103*T89
        T424 = -T102*T420 + T237*T38 - T422 + T423
        T425 = wp.float64(1.0)*n_xy
        T426 = T231*rho_xx
        T427 = T232*rho_xy
        T428 = T38*T67
        T429 = -T18
        T430 = T218 + T429
        T431 = -T41
        T432 = T217 + T431
        T433 = -T430 - T432
        T434 = -T311 + T41*T65 - T425 + T426 + T427 + T428 + T433*T65
        T435 = T2*gamma
        T436 = T231*rho_xy
        T437 = T21*T65
        T438 = T232*rho_yy
        T439 = T18*T67 - T264 - T308 + T433*T67 + T436 + T437 + T438
        T440 = T33*gamma
        T441 = -T171
        T442 = T112*T241 + T441
        T443 = T243 + T442
        T444 = T44*rho_xxx
        T445 = -T287
        T446 = T112*T247 + T445
        T447 = T243 + T446
        T448 = T28*rho_yyy
        T449 = -T239
        T450 = T15*T237 + T236
        T451 = T236*T258
        T452 = -T111*T237 + T18*T237 + T449*m_x + T450*T66 + T451
        T453 = -T119*T237 + T237*T41 + T449*n_y + T450*T68 + T451
        T454 = T112*T165
        T455 = T15*T454 + T173
        T456 = T112*T285
        T457 = T15*T456 + T289
        T458 = T168 + T288 + T382
        T459 = -T300 - T458
        T460 = T172 + T201 + T379
        T461 = -T246 - T460
        T462 = -T302
        T463 = T243 + T290 + T462
        T464 = T170 + T247 + T463
        T465 = T44*rho_xyy
        T466 = -T249
        T467 = T202 + T466
        T468 = T243 + T467
        T469 = T169 + T241 + T468
        T470 = T28*rho_xxy
        T471 = wp.float64(2.0)*n_y
        T472 = T28*T471
        T473 = -T255
        T474 = T112*T13
        T475 = wp.float64(2.0)*gamma
        T476 = T475 - wp.float64(6.0)
        T477 = -T476
        T478 = wp.float64(1.0)*T33
        T479 = wp.float64(1.0)*T175
        T480 = T112*T28
        T481 = T18*T480
        T482 = T478 + T479 + T481
        T483 = -T105
        T484 = T44*T483
        T485 = wp.float64(1.0)*T484
        T486 = -T117
        T487 = T28*T486
        T488 = wp.float64(1.0)*T487
        T489 = -T485 - T488
        T490 = wp.float64(1.0)*T0*T13*n + wp.float64(3.0)*T1*T13*m*m_y + wp.float64(2.0)*T11*T473*rho_y + wp.float64(1.0)*T11*T477*m*n*rho_x - T113*T232 - T176 - T39*T474 - T472 - T482 - T489
        T491 = wp.float64(2.0)*m_x
        T492 = T44*T491
        T493 = wp.float64(1.0)*T2
        T494 = wp.float64(1.0)*T141
        T495 = wp.float64(1.0)*T142
        T496 = T493 + T494 + T495
        T497 = -T123
        T498 = T44*T497
        T499 = wp.float64(1.0)*T498
        T500 = T108*T28
        T501 = wp.float64(1.0)*T500
        T502 = -T499 - T501
        T503 = wp.float64(1.0)*T0*T13*m + wp.float64(3.0)*T1*T13*n*n_x + wp.float64(2.0)*T11*T240*rho_x + wp.float64(1.0)*T11*T477*m*n*rho_y - T143 - T19*T474 - T22*T474 - T492 - T496 - T502
        T504 = wp.float64(6.0)*T1
        T505 = m*n
        T506 = T24*T505
        T507 = -T504*T506
        T508 = -T145
        T509 = T508*T89
        T510 = wp.float64(2.0)*T473
        T511 = wp.float64(2.0)*n_x
        T512 = wp.float64(1.0)*T112
        T513 = T483*T512
        T514 = T294*T486
        T515 = T1*T492 + T12*T510 + T152*T187 + T278 + wp.float64(3.0)*T28*T45 - T28*T511 + T28*T513 + T507 - T509 - T514
        T516 = T12*T505
        T517 = -T504*T516
        T518 = -T178
        T519 = T518*T89
        T520 = wp.float64(2.0)*m_y
        T521 = T108*T512
        T522 = T148*T497
        T523 = T1*T472 + T151*T188 + T199 + wp.float64(2.0)*T24*T240 + wp.float64(3.0)*T29*T44 - T44*T520 + T44*T521 + T517 - T519 - T522
        T524 = T1*T157
        T525 = T1*T202
        T526 = T126*T13
        T527 = -T189 + T526
        T528 = T292 - T300 + T524 - T525 + T527
        T529 = wp.float64(1.0)*T385
        T530 = T15 + T529
        T531 = T28*n_y
        T532 = T14 + wp.float64(2.0)
        T533 = T13*T15
        T534 = T477*n
        T535 = T25*T534
        T536 = T12*T535 - T310 + T39*T533 + T42*T533
        T537 = wp.float64(1.0)*T197 - T478 - T479 - T481 + T485 + T488 + T49 + T528*T67 + T530*T531 + T532*T70 + T536
        T538 = T1*T192
        T539 = T112*T171
        T540 = T538 - T539
        T541 = -T154 + T526
        T542 = T204 - T246 + T540 + T541
        T543 = T44*m_x
        T544 = T28*T29
        T545 = T19*T533 + T22*T533 + T24*T535 - T259
        T546 = wp.float64(1.0)*T211 + T32 - T493 - T494 - T495 + T499 + T501 + T530*T543 + T532*T544 + T542*T65 + T545
        T547 = ((T112)*(T112))
        T548 = T165*T547 + T524
        T549 = T166 + T191 + T541 + T548
        T550 = wp.float64(3.0)*T385
        T551 = wp.float64(2)*T112
        T552 = wp.float64(3.0) - T14
        T553 = T15*T551 + T550 + T552
        T554 = T112*T44
        T555 = T483*T89
        T556 = T28*T555
        T557 = T112*T508 + T163 + T486*T554 - T556
        T558 = T14*T544 + T543*T553 + T545 + T549*T65 + T557
        T559 = T285*T547 + T538
        T560 = T156 + T286 + T527 + T559
        T561 = T112*T518 + T284 + T480*T497
        T562 = T14*T70 + T531*T553 + T536 + T560*T67 + T561
        T563 = T467 + T540
        T564 = -T134
        T565 = -T50 - T564 - T8
        T566 = wp.float64(6.0)*T5
        T567 = -wp.float64(1.5)*T1*T6
        T568 = m*(-T566 - T567)
        T569 = -T216
        T570 = wp.float64(2.0)*T1
        T571 = T569*T570
        T572 = -T135
        T573 = T13*T572
        T574 = T573*T89
        T575 = -T125*T474 + T250 + T571 - T574
        T576 = T259*n
        T577 = T44*n_y
        T578 = wp.float64(5.0) - T14
        T579 = T578*n
        T580 = T577*T579
        T581 = -T385 - T476
        T582 = -T230
        T583 = T25*T518
        T584 = T508*T69
        T585 = -T139
        T586 = -T133
        T587 = T231*T585 + T232*T586 + T582 - T583 - T584
        T588 = T102*T575 + T106*T565 + T24*T568 + T42*T44*T581 + T563*n_x + T576 + T580 + T587
        T589 = T13*rho_xy
        T590 = T1*T157 + T290 - T302 - T525
        T591 = -T125
        T592 = -T56 - T591 - T8
        T593 = -T128
        T594 = T13*T593
        T595 = T594*T89
        T596 = -T134*T474 + T303 + T571 - T595
        T597 = wp.float64(6.0)*T3
        T598 = n*(-T567 - T597)
        T599 = T310*m
        T600 = T543*T579
        T601 = T103*T596 + T109*T592 + T12*T598 + T19*T28*T581 + T587 + T590*m_y + T599 + T600
        T602 = T442 + T548
        T603 = T13*T18
        T604 = wp.float64(3.0)*T57
        T605 = T112*T594 + T242 + T571 + T604
        T606 = T220*T578
        T607 = T14 + wp.float64(3.0)
        T608 = T220*T607
        T609 = -T148*T585
        T610 = T1*T518
        T611 = T610*T69
        T612 = T15*T508
        T613 = -T223
        T614 = T1*T613
        T615 = -T227 + T554*T586 + T609 + T611 + T612*m + T614
        T616 = T24*T598 + T259*m - T29*T608 + T592*T603 + T602*m_x + T605*T66 + T606*m_y + T615
        T617 = T13*rho_xx
        T618 = T446 + T559
        T619 = T13*T41
        T620 = wp.float64(3.0)*T51
        T621 = T112*T573 + T305 + T571 + T620
        T622 = -T294*T586
        T623 = T1*T508
        T624 = T25*T623
        T625 = T15*T518
        T626 = -T297 + T480*T585 + T614 + T622 + T624 + T625*n
        T627 = T12*T568 + T310*n - T45*T608 + T565*T619 + T606*n_x + T618*n_y + T621*T68 + T626
        T628 = T13*rho_yy
        T629 = wp.float64(2.0)*T102
        T630 = -T629
        T631 = wp.float64(1.0)*T15
        T632 = wp.float64(2.0)*T1*m_y - T102*T631 - T103*T268 - T511 - T630
        T633 = wp.float64(2.0)*T103
        T634 = T633 - wp.float64(1.0)*m_y
        T635 = -T629 - T634 + wp.float64(1.0)*n_x
        T636 = T385*T635
        T637 = T102*T512 - T282 - T423 + T513 + T636
        T638 = T103*T570 + T38 - T46 + T630
        T639 = -T16 - T430 - T477*T66
        T640 = wp.float64(1.0)*T137 + T138 + T18*T265
        T641 = T177 - T361 + T482
        T642 = -T13*T640 + T641
        T643 = T231*T638 + T232*T639 + T489 + T642
        T644 = wp.float64(2.0)*E + wp.float64(2.0)*T52
        T645 = -wp.float64(2.0)*T215 + T644
        T646 = -wp.float64(6.0)*T51 + T645
        T647 = -T150 - T152*T471 - T219*T28 - T221*T28 - T504*T543 - T507 - T646*T65
        T648 = T1*T647
        T649 = T13*T89
        T650 = T211*T570
        T651 = T509 + T514
        T652 = T650 + T651
        T653 = T15*T232*T483 - T294*T639 + T480*T638 - T586*T649 + T648 + T652
        T654 = -T218
        T655 = -T217
        T656 = -T115 - T122 - T654 - T655
        T657 = wp.float64(1.0)*T85
        T658 = wp.float64(2.0)*T44
        T659 = -T430 - wp.float64(6.0)*T66 + wp.float64(3.0)*m_x
        T660 = T268*T28
        T661 = T505*T578
        T662 = T1*T505
        T663 = T607*T662
        T664 = wp.float64(1.0)*E_xy
        T665 = -E_xy*T417 + T237*T664
        T666 = -T100*T663 + T109*T656 + T13*T45*(T1*T659 - T111*T112 + wp.float64(2.0)*T113 + wp.float64(3.0)*T114 + T18 + T431 + T654) + T152*T532*m_xy + T152*T83 + T236*T44*n_yy + T275*T311 + T275*T425 + T310*T65 + T330*T632 + T352*T637 - T425*T480 + T565*T657 + T589*T590 + T643*T65 + T653*T67 - T658*n_xx + T660*m_yy + T661*T98 + T665
        T667 = T103*T512 + T419 + T521 - T555 + T636
        T668 = wp.float64(2.0)*T1*n_x - T102*T268 - T103*T631 - T520 + T633
        T669 = -wp.float64(6.0)*T57 + T645
        T670 = -T186 - T188*T491 - T219*T44 - T221*T44 - T504*T531 - T517 - T669*T67
        T671 = T1*T670
        T672 = wp.float64(2.0)*T1*T13*n*rho_x - T30 - T634
        T673 = -T36 - T432 - T477*T68
        T674 = T197*T570
        T675 = T185 + T674
        T676 = T519 + T522
        T677 = -T148*T673 + T554*T672 - T585*T649 + T671 + T675 + T676
        T678 = wp.float64(1.0)*T130 + T132 + T147*T41
        T679 = T144 - T355 + T496
        T680 = -T13*T678 + T679
        T681 = T231*T673 + T232*T672 + T502 + T680
        T682 = wp.float64(2.0)*T28
        T683 = -T432 - wp.float64(6.0)*T68 + wp.float64(3.0)*n_y
        T684 = T268*T44
        T685 = T100*T661 + T106*T656 + T13*T29*(T1*T683 - T112*T119 + wp.float64(2.0)*T120 + wp.float64(3.0)*T121 + T41 + T429 + T655) + T184*T264 + T184*T308 + T188*T532*n_xy + T188*T81 + T236*T28*m_xx + T259*T67 - T308*T554 + T330*T667 + T352*T668 + T563*T589 + T592*T657 + T65*T677 - T663*T98 + T665 + T67*T681 - T682*m_yy + T684*n_xx
        T686 = -T82
        T687 = E_yy*T237
        T688 = wp.float64(1.0)*T687
        T689 = E_xx*T449
        T690 = -T161
        T691 = -T148*T638 + T184*T486 - T211*T477 + T474*T586 + T554*T639 + T556 + T612 + T648
        T692 = T13*T656
        T693 = T44*n_xy
        T694 = T44*m_yy
        T695 = T397*T581 + T661*T85
        T696 = T18*T692 + T259*T65 + T264*T275 + T275*T308 - T308*T480 + T330*(-T112*T114 + T385*T659 + T486*T551 + T529*T68 + T690) + T340*T632 + T344*T667 + T368*T565 + T373*T553 + T530*T693 + T570*T694 + T602*T617 + T643*T67 + T65*T691 + T660*n_xx + T686 + T688 + T689 + T695
        T697 = -T84
        T698 = E_xx*T237
        T699 = wp.float64(1.0)*T698
        T700 = E_yy*T449
        T701 = -T197*T477 + T275*T497 + T283 - T294*T672 + T474*T585 + T480*T673 + T625 + T671
        T702 = T28*m_xy
        T703 = T28*T570
        T704 = T184*T311 + T184*T425 + T268*T694 + T310*T67 + T340*T637 + T344*T668 + T352*(-T112*T121 + T385*T683 + T497*T551 + T529*T66 + T690) + T367*T592 + T378*T553 + T41*T692 - T425*T554 + T530*T702 + T618*T628 + T65*T681 + T67*T701 + T695 + T697 + T699 + T700 + T703*n_xx
        T705 = E_xy*T1
        T706 = T69*T705
        T707 = E_xy*T237
        T708 = T69*T707
        T709 = T25*T687
        T710 = wp.float64(6.0)*n
        T711 = T112*n
        T712 = T44*T711
        T713 = T13*T240
        T714 = E_xx*m
        T715 = T433*gamma
        T716 = T511*m
        T717 = T491*n
        T718 = -wp.float64(3.0)*T1*T6
        T719 = T566 + T718
        T720 = -T719
        T721 = wp.float64(2.0)*T1*m*m_y + wp.float64(6.0)*T13*m*n*rho_x - T534*n_y - T67*T720 - T716 - T717
        T722 = T471*m
        T723 = T520*n
        T724 = T477*m
        T725 = T597 + T718
        T726 = -T725
        T727 = wp.float64(2.0)*T1*n*n_x + wp.float64(6.0)*T13*m*n*rho_y - T65*T726 - T722 - T723 - T724*m_x
        T728 = T518*T86
        T729 = T131*T508
        T730 = wp.float64(6.0)*m
        T731 = T197*T730 + T728 + T729
        T732 = T231*T721 + T232*T727 - T585*T658 - T586*T682 + T731
        T733 = T131*T610
        T734 = E_x*m
        T735 = T475*T734
        T736 = E_y*n
        T737 = T475*T736
        T738 = wp.float64(6.0)*E - wp.float64(9.0)*T1*T13*T6 + wp.float64(6.0)*T52
        T739 = -T738
        T740 = wp.float64(6.0)*T1*T13*m*m_y*n + wp.float64(6.0)*T1*T13*m*n*n_x - T646*m_x - T66*T739 - T669*n_y - T68*T739 - T735 - T737
        T741 = T1*T740
        T742 = T100*T568 - T152*T710*n_xx + T2*T715 + T245*T605 + T251*T575 + T252*T554 - T253 + T254*T510 + T264*T712 - T267 + T450*T714 + T542*n_xy + T549*m_xx + T598*T85 + T643*m_y + T65*(-T148*T721 + wp.float64(2)*T184*T586 - T196*T726 + T508*T724 + T554*T727 + T585*T703 - T733 + T741) + T67*T732 + T677*n_x + T681*n_y + T691*m_x + T706 - T708 - T709 + T713*T81
        T743 = T25*T705
        T744 = T69*T698
        T745 = T25*T707
        T746 = T152*m_yy
        T747 = T13*T473
        T748 = T13*n_xx
        T749 = E_yy*n
        T750 = T44*T570
        T751 = T623*T86
        T752 = wp.float64(2.0)*T240*T748 + T304*T596 + T306*T621 - T307 + T311*T712 - T312 + T33*T715 + T450*T749 + T528*m_xy + T560*n_yy + T568*T85 + T598*T98 + T643*m_x + T65*T732 + T653*m_y + T67*(-T196*T720 + wp.float64(2)*T275*T585 - T294*T727 + T480*T721 + T518*T534 + T586*T750 + T741 - T751) + T681*n_x + T701*n_y - T710*T746 + T711*T82 + T743 - T744 - T745 + T747*T83
        T753 = T13*T6*(T475 - wp.float64(2.0))
        T754 = T11*m*(T4 - wp.float64(3.0)*T56)
        T755 = -T34
        T756 = T11*n
        T757 = T756*(-wp.float64(3.0)*T50 - T755)
        T758 = T314 - T385
        T759 = -T475 + T758 + wp.float64(4.0)
        T760 = E_xxy*T131
        T761 = wp.float64(3.0)*T165
        T762 = T15*T761
        T763 = wp.float64(3.0)*T285
        T764 = T15*T763
        T765 = -T189 - T458 - T764
        T766 = -T154 - T460 - T762
        T767 = T1*T34
        T768 = -T767
        T769 = T13*(T4 + T768)
        T770 = T13*T140
        T771 = T15*T285 + T456 + T457
        T772 = T15*T165 + T454 + T455
        T773 = T285*T550
        T774 = T434*n + T439*m
        T775 = T13*T262
        T776 = T241 - T443 - T775
        T777 = T416*n
        T778 = T11*rho_yyy
        T779 = T13*T260
        T780 = T26*(T247 - T447 - T779)
        T781 = wp.float64(2.0)*T13*T3*T5 - T34*T469 - T4*T464
        T782 = T15*T3
        T783 = -T366
        T784 = T10*T54 + T3*T396 + T783
        T785 = -T784*T89
        T786 = T11*rho_xxx
        T787 = wp.float64(1.0)*T54
        T788 = T15*T787
        T789 = T372 + T788
        T790 = T15*T4
        T791 = T10*T247
        T792 = T447*T767
        T793 = T35*T59 + T396*T5 + T783
        T794 = -T793*T89
        T795 = wp.float64(1.0)*T59
        T796 = T15*T795
        T797 = T377 + T796
        T798 = -T129
        T799 = T459*T89
        T800 = T169*T385
        T801 = T287*gamma
        T802 = -T801
        T803 = T441 + T63
        T804 = T445 + T61 + T803
        T805 = T291 - T370 + T800 + T802 + T804
        T806 = T1*T805
        T807 = T15*T190
        T808 = T170*T385
        T809 = T14*T57
        T810 = -T809
        T811 = T203 - T375 + T804 + T808 + T810
        T812 = T1*T811
        T813 = -T208
        T814 = T1*T813
        T815 = T1*T178 - T28*T452 - T333 - T424*T44 - T814
        T816 = T1*T211
        T817 = -T212
        T818 = T1*T817
        T819 = T1*T145 - T28*T421 - T44*T453 - T816 - T818
        T820 = wp.float64(1.0)*E_yy
        T821 = -T464
        T822 = -T469
        T823 = T301 + T51 + T57
        T824 = -wp.float64(2)*E + T248 - wp.float64(2)*T396 - wp.float64(2)*T52 + T823
        T825 = T824*T89
        T826 = T178*T529
        T827 = T208*T417
        T828 = T401*T87
        T829 = wp.float64(3.0)*T400
        T830 = wp.float64(2.0)*T314 + T829
        T831 = wp.float64(1.0)*T314
        T832 = T258*T831
        T833 = T189*T314 + T193*T237
        T834 = T12*T828 + T14*T197 + T14*T518 - T237*T47 + T237*T76 + T28*T832 - T40*T401 - T401*T43 - T531*T830 + T67*T833
        T835 = -T1*T834
        T836 = T1*T146
        T837 = T15*T212
        T838 = T1*T837
        T839 = T154*T314 + T158*T237
        T840 = T14*T211 + T14*T508 - T20*T401 - T23*T401 - T237*T31 + T237*T74 + T24*T828 + T44*T832 - T543*T830 + T65*T839
        T841 = -T1*T840
        T842 = T14*T258
        T843 = -T44*T842
        T844 = T14*T44
        T845 = T433*T844 + T680 + T843
        T846 = -T28*T842
        T847 = T14*T28
        T848 = T433*T847 + T642 + T846
        T849 = T258*gamma
        T850 = T111 - T113 - T114 + T66*gamma + T849 + m_x
        T851 = T102*gamma + T102 - T104 + T45 - n_x
        T852 = T103*gamma + T108
        T853 = -T120 - T121
        T854 = T119 + T853 + n_y
        T855 = T68*gamma + T849 + T854
        T856 = T10 + T3*gamma
        T857 = T35 + T5*gamma
        T858 = T425*T44
        T859 = T28*T308
        T860 = gamma + wp.float64(1)
        T861 = T91 + T93
        T862 = T106*T851 + T109*T852 - T14*T693 - T14*T702 + T367*T856 + T368*T857 - wp.float64(2.0)*T373 - wp.float64(2.0)*T378 + T603*T855 + T619*T850 + T65*T845 + T67*T848 + T688 + T699 - T858 - T859 + T860*T88 + T861
        T863 = T131*T206
        T864 = T490*T69
        T865 = wp.float64(1.0)*T0*T13*T255 + wp.float64(1.0)*T1*T817*m + wp.float64(1.0)*T178*n - T25*T515 - T298 - T69*T813 - T863 - T864
        T866 = T523*T69
        T867 = wp.float64(1.0)*T1*T813*n + wp.float64(2.0)*T108*T13*m*n + wp.float64(1.0)*T145*m - T196*T413 - T228 - T25*T503 - T25*T817 - T866
        T868 = T13*n_xy
        T869 = T558*T69
        T870 = T14 - wp.float64(2.0)
        T871 = wp.float64(2)*T114 - wp.float64(2)*T16 + T471 + T654
        T872 = T44*T69
        T873 = -wp.float64(1.0)*T0*T13*T870*m*n + T233 + T234 + T582
        T874 = wp.float64(1.0)*T15*T813*m - T233*T870 - T25*T537 - T69*T817 - T869 - T871*T872 - T873
        T875 = T13*m_xy
        T876 = T546*T69
        T877 = T112*T68 + T217 + T36 - T491 + T853
        T878 = wp.float64(1.0)*T13*T877*m*n + wp.float64(1.0)*T15*T817*n - T234*T870 - T25*T562 - T25*T813 - T873 - T876
        T879 = T13*n_yy
        T880 = T197*T25*(T385 + T552)
        T881 = wp.float64(2.0)*T241
        T882 = T146*T69
        T883 = T826*m
        T884 = T69*T837
        T885 = -T392 - T58
        T886 = T54 + T58
        T887 = T394 + T886
        T888 = T74*n
        T889 = T165*T550
        T890 = wp.float64(3.0)*T391
        T891 = -T620
        T892 = T62 + T644
        T893 = T891 + T892
        T894 = T889 - T890 + T893
        T895 = -T604
        T896 = T892 + T895
        T897 = -wp.float64(3.0)*T393 - T763 - T896
        T898 = wp.float64(5.0)*T1
        T899 = T898*n
        T900 = T349*T486
        T901 = T1*T583
        T902 = T1*T584
        T903 = -T901 - T902
        T904 = T1*T580 + T102*T894 + T104*T897 + T336 + T38*T885 + T422*T887 + T483*T795 + T543*T899 + T599 - T888 - T900 + T903
        T905 = T1*T904
        T906 = T25*T333*(wp.float64(4.0) - T14)
        T907 = T181*T503
        T908 = -T388 - T886
        T909 = T399 + T58
        T910 = wp.float64(3.0)*T387
        T911 = T773 + T896 - T910
        T912 = -wp.float64(3.0)*T398 - T761 - T893
        T913 = T181*T498
        T914 = T1*T600 + T103*T911 + T107*T912 + T108*T787 + T21*T908 + T343 + T418*T909 + T576 + T577*T899 - T76*m + T903 - T913
        T915 = -T1*T914
        T916 = wp.float64(4.0)*n
        T917 = T103*T916 + wp.float64(2)*T130 + wp.float64(2)*T17 + T222 - T722 - T723
        T918 = T145*m
        T919 = T601*m
        T920 = T616*n
        T921 = T0*T13*T262*n - T10*T178 + T10*T813 - T220*T917 + T817*m*n - T918*n - T919 - T920
        T922 = wp.float64(4.0)*T102*m + wp.float64(2)*T137 + T219*m + wp.float64(2)*T37 - T716 - T717
        T923 = T588*n
        T924 = T627*m
        T925 = T0*T13*T260*m - T145*T35 - T178*T505 - T220*T922 + T35*T817 + T813*m*n - T923 - T924
        T926 = wp.float64(3.0)*T220
        T927 = T14 + wp.float64(5.0)
        T928 = T220*T927
        T929 = T181*T484
        T930 = T268*T508
        T931 = T930*m
        T932 = T190 + T383 + T62 + T809
        T933 = -wp.float64(1.5)*T215
        T934 = T15*T155
        T935 = T370 + T933 - T934
        T936 = -wp.float64(1.0)*T157*gamma + T604
        T937 = -T525 - T935 - T936
        T938 = -T15*T620 + T157 + T171 + T370
        T939 = T788 + T890 + T938
        T940 = -T324
        T941 = T14*T940
        T942 = -T15*T158 + T370 + T645
        T943 = wp.float64(1.0)*T112*T54 - T936 - T942
        T944 = T29*T926 + T320 + T328 + T45*T928 + T486*T787 - T611 + T66*T943 + T68*T937 - T735 - T736*T831 - T929 - T931 + T932*n_y + T939*m_x + T941
        T945 = -T1*T944
        T946 = -T15 - T164
        T947 = T181*T537
        T948 = T69*T827
        T949 = T13*m_xx
        T950 = T155 + T380 + T62 + T801
        T951 = -T15*T604 + T192 + T287 + T375
        T952 = T796 + T910 + T951
        T953 = T268*T518
        T954 = T953*n
        T955 = T375 - T807 + T933
        T956 = -wp.float64(1.0)*T192*gamma + T620
        T957 = -T539 - T955 - T956
        T958 = -T15*T193 + T375 + T645
        T959 = wp.float64(1.0)*T112*T59 - T956 - T958
        T960 = T29*T928 + T346 + T351 + T45*T926 + T497*T795 - T624 + T66*T957 + T68*T959 - T734*T831 - T737 + T941 + T950*m_x + T952*n_y - T954
        T961 = -T1*T960
        T962 = T181*T562
        T963 = T228*T870
        T964 = T0*T11
        T965 = T10*T11
        T966 = wp.float64(4.0)*T366
        T967 = T569*gamma
        T968 = T14*T64
        T969 = T193*T35 - T5*T968
        T970 = T134*T967 + T572*T795 + T966 + T969
        T971 = -wp.float64(4.0)*T51
        T972 = T63 + T968
        T973 = -T155*gamma + T466 + T800 + T971 + T972
        T974 = wp.float64(3.0)*T248
        T975 = T14*T192 + T156 + T512*T59 + T968 - T974
        T976 = -E - T52
        T977 = -T112*T57 + T192*gamma - T248 - T51 - T57 - T976
        T978 = -wp.float64(2.0)*T1*T13*T35 + T968
        T979 = wp.float64(3.0)*E + wp.float64(3.0)*T52 - wp.float64(2.0)*T569*gamma + T62
        T980 = -T574 - T971 - T978 - T979
        T981 = T189*gamma + T189 + T247 + T803
        T982 = T44*T860
        T983 = T273*T756
        T984 = T1*T729
        T985 = T13*T795
        T986 = T14*T613
        T987 = T193*T518 - T28*T941 + T28*T986 - T362 + T44*T984 + T585*T985 - T586*T983
        T988 = T13*T76*T857 + T24*T970 + T28*T973*m_x + T40*T977 + T516*T980 + T531*T975 - T70*T981 + T888*T982 + T987
        T989 = -T1*T988
        T990 = T10*T158 - T396*T4
        T991 = T125*T967 + T593*T787 + T966 + T990
        T992 = wp.float64(3.0)*T301
        T993 = T14*T157 + T191 + T512*T54 + T968 - T992
        T994 = -wp.float64(4.0)*T57
        T995 = -T190*gamma + T462 + T808 + T972 + T994
        T996 = -T112*T51 + T157*gamma - T823 - T976
        T997 = -wp.float64(2.0)*T1*T10*T13 + T968
        T998 = -T595 - T979 - T994 - T997
        T999 = T154*gamma + T154 + T241 + T445 + T63
        T1000 = T13*T787
        T1001 = T1000*T586 + T158*T508 - T356 + T44*T733 - T44*T941 + T44*T986 - T585*T983
        T1002 = T1001 + T12*T991 + T13*T74*T856 + T23*T996 + T506*T998 + T543*T993 - T544*T999 + T577*T995 + T77*T982
        T1003 = -T1*T1002
        T1004 = -T489 - T641
        T1005 = T69*T696
        T1006 = T116 + T257
        T1007 = -T1006
        T1008 = wp.float64(1.0)*T178
        T1009 = wp.float64(1.0)*T813
        T1010 = T112*T245 - T16*T65 + T18*T65 + wp.float64(2.0)*T251 - T294*rho_yy + T428 - T46*T67 + T638*T67 + T639*T65 - T81 + T90 + T94
        T1011 = -T293
        T1012 = T1*T159 + wp.float64(1.0)*T112*T13*T483*n - T277 - T651
        T1013 = T101 + T106*T483 + T110 + T12*T586 + T24*T585 + T330*T486 + T352*T497 + T686 + T687 + T697 + T698 + T861 + T88 + T95 + T97 + T99
        T1014 = T230 + T583 + T584
        T1015 = -wp.float64(1.0)*T13*T585*m - wp.float64(1.0)*T13*T586*n
        T1016 = -T1014 - T1015
        T1017 = T1016*T65
        T1018 = T626*T67
        T1019 = T13*T483*m + T13*T486*n - T208
        T1020 = T13*T497*m - T213
        T1021 = -T240
        T1022 = T1019*T41 + T1020*T38 + wp.float64(1.0)*T1021*T748 + T265*T858 + T313 - T711*T858 - T743 + T744 + T745
        T1023 = T1011*T308 + T1012*T21 - wp.float64(1.0)*T1013*n + wp.float64(1.0)*T1017 + wp.float64(1.0)*T1018 + T1022 + T18*T561 + T264*T457 + T436*T463 + T438*T447 + T69*T700
        T1024 = T0*T1004 - T1005 - T1007*T1008 - T1009*T117 + wp.float64(1.0)*T1010*T13*m*n - T1023 + wp.float64(1.0)*T105*T145 - T25*T666 - T274*T817
        T1025 = -T502 - T679
        T1026 = T685*T69
        T1027 = wp.float64(1.0)*T145
        T1028 = wp.float64(1.0)*T817
        T1029 = T112*T306 - T148*rho_xx - T30*T65 + wp.float64(2.0)*T304 - T36*T67 + T41*T67 + T437 + T65*T672 + T67*T673 - T83 + T92 + T96
        T1030 = T1*T194 + wp.float64(1.0)*T108*T112*T13*m - T198 - T676
        T1031 = -T205
        T1032 = T615*T65
        T1033 = T1016*T67
        T1034 = T1019*T21 + T1020*T18 + T1021*T13*T425 + T266*T308 + T270 - T308*T712 - T706 + T708 + T709
        T1035 = -wp.float64(1.0)*T1013*m + T1030*T38 + T1031*T425 + wp.float64(1.0)*T1032 + wp.float64(1.0)*T1033 + T1034 + T25*T689 + T311*T455 + T41*T557 + T426*T443 + T427*T468
        T1036 = T0*T1025 - T1008*T108 - T1026 - T1027*T854 - T1028*T123 + wp.float64(1.0)*T1029*T13*m*n - T1035 + wp.float64(1.0)*T108*T813 - T25*T704
        T1037 = T178*T417
        T1038 = T181*T704
        T1039 = T1011*m_xy + T1012*m_y + T1017 + T1018 + T1022 + T304*T463 + T306*T447 + T457*n_yy + T561*n_y + T700*n
        T1040 = T1039*T89
        T1041 = T1*T606
        T1042 = T1*T927
        T1043 = wp.float64(2.0)*T500
        T1044 = T25*T265
        T1045 = wp.float64(6.0)*T333 + T953
        T1046 = T273*n
        T1047 = -T341
        T1048 = T1047 + T108*T684 + T674 + T676
        T1049 = wp.float64(1.0)*T1*T13*T635*gamma*m - T1046*T12 - T1048 + wp.float64(1.0)*T13*T54*rho_y - T44*T555 - T487*T89
        T1050 = -T327
        T1051 = T1050 + T483*T660 + T652
        T1052 = wp.float64(1.0)*T1*T13*T635*gamma*n - T1046*T24 - T1051 + wp.float64(1.0)*T13*T59*rho_x - T498*T89 - T500*T89
        T1053 = -T342
        T1054 = T14*T670
        T1055 = T1*T131
        T1056 = T1*T728 - T148*T586 - T294*T585 + T333*T730 + T984
        T1057 = T1053 + T1054*m + T1055*T498 + T1056 + T158*T183 - T349*T673 + T672*T787
        T1058 = T610*T710
        T1059 = T131*T152
        T1060 = -T941
        T1061 = T1060 - T346
        T1062 = -T0*T669 + T1054*n + T1058 + T1059*T108 + T1061 + T193*T497 - T349*T672 - T585*T660 + T622 + T673*T795 + T751 + T986
        T1063 = T1041*m_yy + T1042*T693*n + T1049*m_x + T1052*m_y + T1057*T65 + T1062*T67 - T188*T999*rho_xx - T237*T69*T79 + T268*T373*n + T29*(-T1043 + T1044*T24 - wp.float64(2.0)*T498 - T65*T787 + T679 + T683*T844) + T304*T995 + T306*T975 + T308*T885 - T402*T86 + T417*T909*n_xx + T436*T977 + T693*T899 + T74*T852 - T749*T830 + T76*T855 + T932*m_xy + T952*n_yy + n_y*(wp.float64(1.0)*T1*T13*T683*gamma*n - T1045 - T108*T750 + wp.float64(1.0)*T11*T385*m*n*rho_x - wp.float64(6.0)*T188*T497 + wp.float64(2.0)*T194*gamma - T67*T796)
        T1064 = -T1*T1063
        T1065 = T1030*n_x + T1031*n_xy + T1032 + T1033 + T1034 + T245*T443 + T251*T468 + T455*m_xx + T557*m_x + T689*m
        T1066 = T220*m_xy
        T1067 = T152*T486
        T1068 = wp.float64(6.0)*T816 + T930
        T1069 = -T332
        T1070 = wp.float64(1.0)*T193
        T1071 = T14*T647
        T1072 = T1056 + T1067*T131 + T1069 + T1070*T483 + T1071*n - T349*T639 + T638*T795
        T1073 = T1060 - T328
        T1074 = -T0*T646 + T1055*T484 + T1071*m + T1073 + T158*T486 - T349*T638 - T586*T684 + T609 + T623*T730 + T639*T787 + T733 + T986
        T1075 = T1041*n_xx + T1042*T1066 + T1049*n_x + T1052*n_y + T1066*T898 + T1072*T67 + T1074*T65 - T152*T981*rho_yy - T237*T25*T80 + T245*T993 + T251*T973 + T269*n_yy - T403 + T417*T887*m_yy + T425*T908 + T427*T996 + T45*(T1044*T12 - wp.float64(2.0)*T484 - wp.float64(2.0)*T487 + T641 + T659*T847 - T67*T795) - T714*T830 + T74*T850 + T76*T851 + T939*m_xx + T950*n_xy + m_x*(wp.float64(1.0)*T1*T13*T659*gamma*m - wp.float64(6.0)*T1067 - T1068 + wp.float64(1.0)*T11*T385*m*n*rho_y + wp.float64(2.0)*T159*gamma - T483*T703 - T65*T788)
        T1076 = T1*T1075
        T1077 = T13*T133
        T1078 = T13*T139
        T1079 = -T13*T640 + T641
        T1080 = -T13*T678 + T679
        T1081 = -T1015 - T731
        T1082 = T112*m*m_xx - T181*n_xx + T252 - T273*m_yy - T304*T916 + T573*rho_yy + T594*rho_xx + T638*m_y + T639*m_x + T65*T727 + T67*T721 + T672*n_x + T673*n_y + T711*n_yy + T81*m
        T1083 = T0*T1081*T13 - T1009*T1077 - T1013*T131*T44 - T1028*T1078 + wp.float64(1.0)*T1039*T13*m + wp.float64(1.0)*T1065*T13*n - T1079*T145 - T1080*T178 + wp.float64(1.0)*T1082*T11*m*n - T231*T752 - T232*T742
        T1084 = T1040*T28
        T1085 = wp.float64(4.0)*T220
        T1086 = E_x*gamma
        T1087 = E_y*gamma
        T1088 = T505*T85
        T1089 = wp.float64(4.0)*T11*T662
        T1090 = wp.float64(2)*T13
        T1091 = T44*T475
        T1092 = T44*T710
        T1093 = T28*T475
        T1094 = E_xx*T839 + E_yy*T833 + T100*T970 + T1057*T344 + T1062*T352 + T1072*T340 + T1074*T330 + T1085*T402 + T1086*T845 + T1087*T848 + T1088*T980 + T1088*T998 + T188*T912*n_xx + T373*T943 + T378*T959 + T65*(T1000*T727 - T1058*T44 + T1089*T585 + T1090*T158*T586 - T1091*T613 + T1091*T940 - T211*T739 + T508*T646 - T721*T983 + T740*T844) + T67*(T1089*T586 + T1090*T193*T585 - T1092*T623 - T1093*T613 + T1093*T940 - T197*T739 + T518*T669 + T721*T985 - T727*T983 + T740*T847) + T693*T911 + T693*T957 + T702*T894 + T702*T937 + T746*T897 + T98*T991
        T1095 = T1*T1094
        T1096 = T1*T4
        T1097 = T1096 + T755
        T1098 = T13*T174
        T1099 = T15*T5
        T1100 = T15*T34
        T1101 = T1096*T443
        T1102 = -T136
        T1103 = T461*T89
        T1104 = wp.float64(1.0)*E_xx
        T1105 = T145*T529
        T1106 = T212*T417
        T1107 = T1*T1106
        T1108 = T1*T279
        T1109 = T15*T208
        T1110 = T1*T1109
        T1111 = wp.float64(2.0)*T247
        T1112 = T1106*T69
        T1113 = T1105*n
        T1114 = T1109*T69
        T1115 = T178*n
        T1116 = T11*T35
        T1117 = T338*T44
        T1118 = T197*T476 - T279
        T1119 = T1039*T15
        T1120 = T139*T533
        T1121 = T271*T86
        T1122 = T1065*T89
        T1123 = T1122*T44
        T1124 = T61 + T644
        T1125 = T1*T220
        T1126 = T14*T399
        T1127 = T243 - wp.float64(1.0)*gamma*(-T167 + T315)
        T1128 = T243 - wp.float64(1.0)*gamma*(-T200 + T317)
        T1129 = T15*T64
        T1130 = T220*m_xxy
        T1131 = T14*T797
        T1132 = T220*n_xyy
        T1133 = T14*T793
        T1134 = T14*T784
        T1135 = -wp.float64(1.0)*T1*T13*T6 + T644
        T1136 = T11*T56
        T1137 = T1136*T790
        T1138 = T169*gamma
        T1139 = T1136*T125*T15
        T1140 = T189*T64
        T1141 = T154*T64
        T1142 = -T824
        T1143 = T14*T285
        T1144 = T14*n
        T1145 = T813*T831
        T1146 = T197*m
        T1147 = -T1014*T385 + T1146*(-wp.float64(2.0)*T385 + T831)
        T1148 = T14*m
        T1149 = T817*T831
        T1150 = T829 + T831
        T1151 = -wp.float64(1.0)*T1*T324*gamma
        T1152 = wp.float64(3.0)*T1*T145*m + T228 - T319 - T325
        T1153 = T1*T1152
        T1154 = wp.float64(3.0)*T1*T178*n + T298 - T325 - T348
        T1155 = T1*T1154
        T1156 = T14*T220
        T1157 = T1156*T818
        T1158 = T1014*T529
        T1159 = T345*T44
        T1160 = T145*T417
        T1161 = T1156*T814
        T1162 = wp.float64(4.0) - T475
        T1163 = T1014*T148
        T1164 = T236*T324
        T1165 = T1014*T294
        T1166 = T28*T64
        T1167 = T1*T1014
        T1168 = T1117*T870 + T365
        T1169 = T0 + T849
        T1170 = T44*T64
        T1171 = T359 + T44*T963
        T1172 = T817*T968
        T1173 = -T968
        T1174 = -T220*T325
        T1175 = T813*T968
        T1176 = T125*T57
        T1177 = -T1014*T349
        T1178 = T1*T86
        T1179 = -wp.float64(6.0)*T0*T1*T13*m*n + T179*T86 + T364
        T1180 = T13*T325
        T1181 = -T1013*T1059 + T1040*T44 + T1122*T28
        T1182 = -T646
        T1183 = T271*T730
        T1184 = T334 + T901 + T902
        T1185 = T1001*T65 - T1104*T316 - T318*T820 + T330*(-T1073 - T319 + wp.float64(1.0)*T486*T54 - T611 - T929 - T931) + T340*(-T1069 - T1184 + wp.float64(1.0)*T483*T59 - T900) + T344*(-T1053 + wp.float64(1.0)*T108*T54 - T1184 - T913) + T352*(-T1061 - T348 - T350 + wp.float64(1.0)*T497*T59 - T624 - T954) + T367*T784 + T368*T793 + T373*T789 + T378*T797 + T381*T693 + T384*T702 - T389*T858 - T392*T859 + T404 + T67*T987
        T1186 = -T1185*T14
        T1187 = -T669
        T1188 = wp.float64(6.0)*T345
        T1189 = T324*gamma
        T1190 = T1013*T216
        T1191 = T325 - T353
        T1192 = T1178*T756


        qt0 = -T0
        qt1 = T10*T12 + T13*T17 - T2 + T32
        qt2 = T13*T37 + T24*T35 - T33 + T49
        qt3 = T13*(T18*T60 + T41*T55 + T64*T66 + T64*T68 + T71 + T73 + T78)
        qtt0 = -T101 + T105*T106 - T110 + T118*m_x + T12*T133 + T124*n_y + T139*T24 + T79 + T80 + T82 + T84 - T88 - T91 - T93 - T95 - T97 - T99
        qtt1 = T13*(E_xx*T239*m + wp.float64(2.0)*E_xy*T1*n + wp.float64(1.0)*E_yy*T1*m + wp.float64(1.0)*T13*T240*n_xy - T18*T214 + T205*n_xy - T209*T21 - T235*T67 - T245*(-T171 - T242 - T244) - T251*(-T244 - T250) - T253 - T270 - T65*(-T224 - T226 - T229) - m_x*(T149 + T163) - m_xx*(-T166 + T173) - n_x*(T180 + T182 - T185 + T199))
        qtt2 = T13*(wp.float64(1.0)*E_xx*T1*n + wp.float64(2.0)*E_xy*T1*m + E_yy*T239*n + wp.float64(1.0)*T13*T240*n_xx - T209*T41 - T214*T38 - T235*T65 + T293*m_xy - T304*(-T244 - T303) - T306*(-T244 - T287 - T305) - T307 - T313 - T67*(-T224 - T296 - T299) - m_y*(T276 + T278) - n_y*(T279 + T280 + T284) - n_yy*(-T286 + T289))
        qtt3 = T13*(wp.float64(1.0)*E_xx*T316 + wp.float64(1.0)*E_yy*T318 - T13*T381*m*n_xy - T13*T384*m_xy*n + wp.float64(1.0)*T13*T389*m*n_xy + wp.float64(1.0)*T13*T392*m_xy*n - T330*(T117*T321 + T181*T206 + T228 + T271*T322 + T320 + T329) - T340*(T105*T331 + T207*T273 + T336 + T339) - T344*(-T108*T321 + T181*T210 + T339 + T343) - T352*(T123*T331 + T298 + wp.float64(3.0)*T345 + T347 + T351) - T367*(-T10*T55 + T3*T64*gamma - T366) - T368*(-T35*T60 - T366 + T5*T64*gamma) - T373*(-T369 + T372) - T378*(-T374 + T377) - T404 - T65*(wp.float64(1.0)*T1*T11*T139*m*n + wp.float64(1.0)*T13*T133*T55 - T354 - T359) - T67*(wp.float64(1.0)*T1*T11*T133*m*n + wp.float64(1.0)*T13*T139*T60 - T360 - T365))
        qttt0 = T13*(E_xx*T452 + E_xy*T421 + E_xy*T424 + E_yy*T453 + T256*T414 + T261*T415 + T263*T416 - T268*T406 - T268*T407 - T268*T408 + T269*m_yyy + T269*n_xxx - T322*T405 + T409*T410 + T410*T411 - T412*T413 + T434*T435 + T439*T440 + T443*T444 + T447*T448 + T455*m_xxx + T457*n_yyy - T459*m_xyy - T461*n_xxy + T464*T465 + T469*T470 + T490*m_xy + T503*n_xy + T515*m_yy + T523*n_xx + T537*m_xy + T546*n_xy + T558*m_xx + T562*n_yy + T588*T589 + T589*T601 + T616*T617 + T627*T628 + T65*T742 + T666*m_y + T67*T752 + T685*n_x + T696*m_x + T704*n_y)
        qttt1 = T13*(-E_xx*(T1*T162 + T148*T424 + T184*T452 - T836 - T838 + T841) + wp.float64(1.0)*E_xxy*T13*T449*m*n - E_xy*(-T1*T827 + T148*T453 + T184*T421 + T197*T529 - T826 + T835) - E_xyy*T753 + E_xyy*(-wp.float64(1.0)*T1*T318 - T1*T770 + T169*T238 + T171*T239) + wp.float64(1.0)*E_y*T1*T13*T774*gamma + wp.float64(1.0)*E_yyy*T13*T449*m*n + T1*T13*n*n_xxx*(T112*T761 + T399*T89 + T769 + T770) - T1024*T340 - T1036*T352 - T1083*T67 + T11*m*n*rho_xxy*(wp.float64(1.0)*T1*T821 - T112*T822 + T13*T593 - T825) + wp.float64(1.0)*T13*T771*m*n_yyy + wp.float64(1.0)*T13*T772*m_xxy*n + T13*m*m_xyy*(wp.float64(1.0)*T1*T13*T140 + wp.float64(1.0)*T385*T395 - T512*T747 - T773) + T13*n*n_xyy*(T1*T797 - T112*T770 + T15*T202 - T457*T89) - T152*T759*T760 - T25*T776*T777 - T254*T865 - T28*T766*n_xyy - T28*m_xxy*(T154*T164 + T798 - T799 + T806) - T330*(-T0*(T0*T13*T476*m - T149) + wp.float64(1.0)*T1*T666*n + T10*T1010*T13 - T1006*T146 - T1013*T147 - T105*T180 + T1065*T15 - T1076 - T117*T837 + T15*T696*m - T274*T827) - T344*(-T0*(-T180 - T182 + T675) - T1*T1013*T69 + T1029*T241 + T1037*T854 + T1038 + T1040 + T1064 - T123*T827 + T146*T183 + T147*T685 + T183*T837) - T368*T925 - T405*(wp.float64(1.0)*T13*T3*T314 - T168 - T171 - T241 - T762) - T415*T781 - T415*(T779*T790 + T791 + T792 + T794) - T435*(wp.float64(1.0)*T1*T13*T439*n + T13*T15*T434*m - T862) - T44*T765*m_xyy - T44*m_xxx*(-T1*T789 + T15*T455 + T203 + T242) - T44*n_xxy*(-T15*T461 + T798 + T807 + T812) - T589*(T148*T627 + T180*T247 + T184*T588 + T247*T827 + T44*T882 + T44*T884 - T69*T964*(T1*T35 + T782) - T922*T965 + T989) - T617*(-T0*T11*m*(T10*T15 + T767) + T1003 + T146*T241 + T148*T601 + T184*T616 + T212*T242 + T228*T44 + T44*T948 - T917*T965) - T65*(wp.float64(1.0)*T1*T13*T752*n + T1*T178*T642 + T10*T1082*T11 - T1013*T129 - T1065*T15*T44 - T1077*T837 - T1078*T827 - T1084 - T1095 + T13*T15*T742*m - T145*(T133*T533 - T146 + T211*T476) - T196*(T196*T725 + T226 + T358 + T476*T918)) - T657*T921 - T664*T815 - T748*(T108*T881 + T147*T523 - T25*T827 - T337 + T69*T836 + T69*T838 + T906 + T907 + T915) - T754*n_xxy - T757*m_yyy - T778*T780 - T786*(((T10)*(T10))*T13 + T443*T782 + T767*T775 + T785) - T819*T820 - T867*T868 - T868*(T147*T546 + T196*(T128 - T15*T767 + T790) + T229 + T241*T877 - T25*T837 + T265*T827 + T961 + T962 + T963) - T874*T875 - T875*(wp.float64(1.0)*T1*T208*T417*m + wp.float64(1.0)*T1*T515*n - T105*T881 + T15*T490*m - T880 - T882 + T883 - T884 - T905) - T878*T879 - T949*(T147*T558 + T164*T212*m + T196*(-T126 - T164*T3 - T591 - T768) + T229 - T241*T871 - T918*T946 + T945 + T947 - T948))
        qttt2 = T13*(wp.float64(1.0)*E_x*T1*T13*T774*gamma + wp.float64(1.0)*E_xxx*T13*T449*m*n - E_xxy*T753 + E_xxy*(-T1*T1098 - wp.float64(1.0)*T1*T316 + T170*T238 + T239*T287) - E_xy*(-T1105 - T1107 + T211*T529 + T275*T424 + T294*T452 + T841) - E_xyy*T1059*T759 + wp.float64(1.0)*E_xyy*T13*T449*m*n - E_yy*(T1*T281 - T1108 - T1110 + T275*T453 + T294*T421 + T835) - E_yyy*T1*(wp.float64(1.0)*T13*T314*T5 - T201 - T247 - T287 - T764) + T1*T13*m*m_yyy*(-T1097*T13 + T1098 + T112*T763 + T395*T89) - T1024*T330 - T1036*T344 - T1083*T65 + T11*m*n*rho_xyy*(wp.float64(1.0)*T1*T822 - T112*T821 + T13*T572 - T825) - T1104*T815 + wp.float64(1.0)*T13*T771*m*n_xyy + wp.float64(1.0)*T13*T772*m_xxx*n + T13*m*m_xxy*(T1*T789 - T1098*T112 + T15*T290 - T455*T89) + T13*n*n_xxy*(wp.float64(1.0)*T1*T13*T174 + wp.float64(1.0)*T385*T399 - T512*T713 - T889) - T254*(-T105*T1111 + T1108*T25 + T1110*T25 - T1112 + T265*T515 + T273*T490 - T338 - T905 + T906) - T26*T776*T786 - T28*T766*n_xxy - T28*m_xyy*(T1102 - T15*T459 + T806 + T934) - T28*n_yyy*(-T1*T797 + T15*T457 + T291 + T305) - T340*(-T0*(-T276 + T650) - T1*T1013*T25 + wp.float64(1.0)*T1*T1065 + wp.float64(1.0)*T1*T696*m + T1007*T145*T417 + T1010*T13*T35 - T1076 - T1106*T117 - T1109*T274 + T15*T666*n - T274*T279) - T352*(-T0*(T1118 - T280 + T283) - T1013*T265 + T1029*T247 + T1064 + T108*T272 + T1106*T183 - T1109*T123 + T1119 + T265*T704 + T273*T685 + T279*T854) - T367*T921 - T409*T765 - T411*(T1102 - T1103 + T164*T189 + T812) - T415*T780 - T416*T781 - T416*(T1100*T775 + T1101 + T785 + T791) - T440*(wp.float64(1.0)*T1*T13*T434*m + T13*T15*T439*n - T862) - T589*(T1003 + T1106*T241 + T1114*T44 - T1116*T917 + T241*T272 - T25*T964*(T1*T10 + T1099) + T275*T601 + T279*T872 + T294*T616) - T628*(-T0*T11*n*(T1096 + T15*T35) + T1112*T44 - T1116*T922 + T1117 + T208*T305 + T247*T279 + T275*T627 + T294*T588 + T989) - T657*T925 - T664*T819 - T67*(wp.float64(1.0)*T1*T13*T742*m + T1*T145*T680 - T1013*T136 - T1077*T1106 + T1082*T11*T35 - T1095 - T1119*T28 - T1120*T208 - T1123 + T13*T15*T752*n - T178*(T1118 + T1120) - T196*(T1115*T476 + T1121 + T196*T719 + T296)) - T748*T867 - T754*n_xxx - T757*m_xyy - T778*(T1096*T779 + T1099*T447 + T13*((T35)*(T35)) + T794) - T865*T875 - T868*T878 - T868*(T108*T1111 + T1107*T69 - T1109*T25 + T1113 - T25*T279 + T265*T503 + T273*T523 - T880 + T915) - T874*T949 - T875*(-T1114 - T196*(T1*T790 - T1100 + T126 + T564) - T247*T871 + T265*T537 + T273*T558 + T298*T870 + T299 + T417*T837*m + T945) - T879*(-T1106*T25 - T1115*T946 + T164*T208*n + T196*(T1096 + T135 - T164*T5) + T247*T877 + T265*T562 + T273*T546 + T299 + T961))
        qttt3 = -T11*(E_xx*(-T0*(-T1*T158 + T169*T314) + wp.float64(1.0)*T1*T13*T424*m*n - T1148*T840 - T1149*m + T1150*T145*m - T1151 - T1153 + wp.float64(1.0)*T452*T55 - T826*n) + E_xxx*m*(-T1127 - wp.float64(3.0)*T167 - T808) + E_xy*(wp.float64(1.0)*T1*T13*T452*m*n - T1113 - T1144*T840 - T1145*m - T1147 + wp.float64(1.0)*T178*T401*m + wp.float64(1.0)*T424*T60) + E_xy*(wp.float64(1.0)*T1*T13*T453*m*n - T1147 - T1148*T834 - T1149*n + wp.float64(1.0)*T145*T401*n + wp.float64(1.0)*T421*T55 - T883) + E_xyy*T86*(T13*T5*T758*gamma - T200 - T386) + E_yy*(-T0*(-T1*T193 + T170*T314) + wp.float64(1.0)*T1*T13*T421*m*n - T1144*T834 - T1145*n + T1150*T178*n - T1151 - T1155 + wp.float64(1.0)*T453*T60 - T529*T918) + T1086*(T0*(-T679 - T843) + wp.float64(1.0)*T1*T434*T55 - T1027*T1169 - T1035 - T1148*T862 + wp.float64(1.0)*T13*T385*T439*m*n + wp.float64(1.0)*T258*T817*gamma) + T1087*(T0*(-T641 - T846) + wp.float64(1.0)*T1*T439*T60 - T1008*T1169 - T1023 - T1144*T862 + wp.float64(1.0)*T13*T385*T434*m*n + wp.float64(1.0)*T258*T813*gamma) + T1125*m_yyy*(T1097*T13 - T1124 - T14*T395 - T895) + T1125*n_xxx*(-T1124 - T1126 - T769 - T891) + T1130*(T1129 - T14*T789 + T374 + T455*T89) + T1130*(-T1135 + wp.float64(2.0)*T15*T55 - T799 + wp.float64(1.0)*T805*gamma) + T1132*(-T1103 - T1135 + wp.float64(2.0)*T15*T60 + wp.float64(1.0)*T811*gamma) + T1132*(T1129 - T1131 + T369 + T457*T89) + T406*(-T1127 - T201 - T889) + T407*(-T1128 - T168 - T773) + T408*(-T1128 - wp.float64(3.0)*T200 - T800) + T412*(wp.float64(1.0)*T1*T5*T64 - T1126*T56 + wp.float64(3.0)*T13*T3*T385*T5 - T240*T331) + T414*(-T1096*T395*gamma + T243*T4 + T255*T321 + T3*T773) + T415*m*(-T1133 + T260*T321 + T35*T64 + T792) + T444*(T11*T262*T767 - T1134*T13 + T241*T64 + T321*T443) + T448*(T1096*T11*T260 - T1133*T13 + T247*T64 + T331*T447) + T465*(T1140 - T1142*T1143 + T171*T469 + T331*T464) + T470*(-T1138*T1142 + T1141 + T287*T464 + T321*T469) + T589*(-T1002*T1144 - T1014*T301 - T1146*(-T192 - T997) + T1154*T872 - T1166*T917 + T1174 + T1175*m + T178*m*(T1173 + T302 + T604) + T294*T920 + T331*T601 + T338*T775) + T589*(-T1014*T248 - T1146*(-T157 - T978) - T1148*T988 + T1152*T872 - T1170*T922 + T1172*n + T1174 + T145*n*(T1173 + T249 + T620) + T148*T924 + T321*T588 + T337*T779) + T617*(-T1002*T1148 + T1152*T241 - T1170*T917 + T1172*m + T1177 + T148*T919 - T196*(T1176 + T990) + T228*T775 - T241*T325 + T321*T616 + T918*(T1173 + T171 + T992)) + T628*(T1115*(T1173 + T287 + T974) - T1144*T988 + T1154*T247 - T1166*T922 + T1175*n + T1177 - T196*(T1176 + T969) - T247*T325 + T294*T923 + T298*T779 + T331*T627) + T65*(-T0*(-T1077*T158 - T1091*T324 + T1182*T145 + T1188*T44 - T1192*T139 + T211*T738 + T354) + T1*T1014*T13*T139 + T1*T1081*T178 + wp.float64(1.0)*T1*T13*T752*m*n - T1039*T1059 - T1065*T158 - T1077*T1152 - T1080*T1189 + T1082*T13*T64*m - T1094*T1148 + wp.float64(1.0)*T1185*gamma*m - T1190*T86 - T145*(T0*T1182 - T1183 + T1191 + T133*T684 + T225 - T358) - T353*T817 + wp.float64(1.0)*T55*T742) + T67*(-T0*(-T1078*T193 + T1092*T271 - T1093*T324 + T1187*T178 - T1192*T133 + T197*T738 + T360) + T1*T1014*T13*T133 + T1*T1081*T145 + wp.float64(1.0)*T1*T13*T742*m*n - T1039*T193 - T1059*T1065 - T1078*T1154 - T1079*T1189 + T1082*T13*T64*n - T1094*T1144 + wp.float64(1.0)*T1185*gamma*n - T1190*T131 - T178*(T0*T1187 - T1121 - T1188 + T1191 + T139*T660 + T295) - T353*T813 + wp.float64(1.0)*T60*T752) + T760*(T13*T3*T758*gamma - T167 - T390) + T777*(T10*T64 + T1101 - T1134 + T262*T331) + m_x*(T1004*T179 - T1007*T1180 + T1010*T1170 - T1013*T158 + T1014*T105*T649 + T1065*T684 - T1075*T844 + T1084 - T1152*T118 + T1186 + T145*(T1050 + T1068 + T486*T684 + T556) + T181*T44*T666 - T196*(T0*T1182 - T1*T863 - T117*T158 - T1183 - T329 - T358) + T321*T696 + T327*T817) + m_xx*(T1152*T184 + T1163 - T1170*T871 + T1171 - T145*T938 + T211*(T190 + T942) + T321*T558 - T357*T870 + T370*T817 + T44*T947 - T844*T944) + m_xxx*(T1129*T165 + T1137 - T1138*T789 + T321*T455) + m_xy*(-T1154*T232 - T1166*T871 - T1167*T184 + T1168 + T152*T869 - T178*(T371 + T383) + T197*(T190 + T63 + T935) + T331*T537 + T370*T813 - T847*T944) + m_xy*(T1037*T309 - T1085*T271 - T1152*T232 + T1157 - T1158*T44 + T181*T44*T515 + T197*(-T154*T385 - T157 - T802) - wp.float64(2.0)*T206*T64 + T321*T490 + T363 - T844*T904) + m_xyy*(T1139 - T1140 + T14*T285*T805 - T331*T459) + m_y*(T1004*T271 + T1005*T152 + T1010*T1166 + T105*T1180 + T1051*T178 - T1075*T847 - T1154*T13*T274 + T1167*T118 + T1181 - T196*(-T105*T1070 - T1178*T207 - T1179 - T332) + T327*T813 + T331*T666) + m_yy*(-T105*T64*T682 + T1155*T231 - T1159*T1162 + T1160*T309 + T1161 + T1163 - T1164*T44 + T152*T864 + T331*T515 + T816*(T1143 + T189 + T192) - T847*T904) + n_x*(T1025*T179 + T1029*T1170 + T1038*T44 + T1048*T145 - T1063*T844 - T108*T1180 + T1152*T13*T183 + T1167*T124 + T1181 - T196*(-T1055*T210 + wp.float64(1.0)*T108*T158 - T1179 - T342) + T321*T685 + T341*T817) + n_xx*(-T1037*T713 + T108*T64*T658 + T1153*T232 + T1157 - T1162*T220*T271 - T1164*T28 + T1165 + T321*T523 + T333*(T1138 + T154 + T157) + T44*T907 - T844*T914) + n_xxy*(T1138*T811 + T1139 - T1141 - T321*T461) + n_xy*(-T1152*T231 - T1167*T275 + T1170*T877 + T1171 - T145*(T376 + T380) + T211*(T155 + T63 + T955) + T321*T546 + T375*T817 + T44*T962 - T844*T960) + n_xy*(T1043*T64 - T1154*T231 - T1158*T28 - wp.float64(4.0)*T1159 - T1160*T713 + T1161 + T152*T866 + T211*(-T189*T385 - T192 - T810) + T331*T503 + T357 - T847*T914) + n_y*(-T1013*T193 - T1014*T13*T282 + T1025*T271 + T1026*T152 + T1029*T1166 + T1039*T660 - T1063*T847 + T1123 - T1154*T124 - T1180*T854 + T1186 + T178*(T1045 + T1047 + T283 + T497*T660) - T196*(T0*T1187 + wp.float64(2.0)*T1*T108*T13*m*n - T1121 - T1188 - T123*T193 - T347) + T331*T704 + T341*T813) + n_yy*(T1154*T275 + T1165 + T1166*T877 + T1168 + T152*T876 - T178*T951 + T197*(T155 + T958) + T331*T562 - T363*T870 + T375*T813 - T847*T960) + n_yyy*(T1129*T285 - T1131*T285 + T1137 + T331*T457))
        return (
            wp.vec4d(qt0, qt1, qt2, qt3),
            wp.vec4d(qtt0, qtt1, qtt2, qtt3),
            wp.vec4d(qttt0, qttt1, qttt2, qttt3),
        )


    @wp.kernel
    def compute_max_speed_kernel(u: wp.array3d(dtype=wp.float64), speed: wp.array(dtype=wp.float64), nx: int, ny: int, gc: int, gamma: wp.float64):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            rho = wp.max(u[jj, ii, 0], wp.float64(1.0e-15))
            vel_x = u[jj, ii, 1] / rho
            vel_y = u[jj, ii, 2] / rho
            pressure = (gamma - wp.float64(1.0)) * (u[jj, ii, 3] - wp.float64(0.5) * rho * (vel_x * vel_x + vel_y * vel_y))
            pressure = wp.max(pressure, wp.float64(1.0e-15))
            sound = wp.sqrt(gamma * pressure / rho)
            speed[j * nx + i] = wp.max(wp.abs(vel_x) + sound, wp.abs(vel_y) + sound)


    @wp.kernel
    def apply_boundary_kernel(u: wp.array3d(dtype=wp.float64), nx: int, ny: int, gc: int):
        j, i = wp.tid()
        nx_total = nx + 2 * gc
        ny_total = ny + 2 * gc
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
        if i < nx_total and j < ny_total:
            if src_i != i or src_j != j:
                for comp in range(4):
                    u[j, i, comp] = u[src_j, src_i, comp]


    @wp.kernel
    def apply_periodic_boundary_kernel(u: wp.array3d(dtype=wp.float64), nx: int, ny: int, gc: int):
        j, i = wp.tid()
        nx_total = nx + 2 * gc
        ny_total = ny + 2 * gc
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
        if i < nx_total and j < ny_total:
            if src_i != i or src_j != j:
                for comp in range(4):
                    u[j, i, comp] = u[src_j, src_i, comp]


    @wp.kernel
    def compute_x_stage_weno7_big_kernel(u: wp.array3d(dtype=wp.float64), l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64), l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64), l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64), l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64), nx: int, ny: int, dx: wp.float64, gamma: wp.float64):
        j, i = wp.tid()
        if j < ny + 8 and i < nx + 2:
            q0 = vec_from_array(u, j, i + 0)
            q1 = vec_from_array(u, j, i + 1)
            q2 = vec_from_array(u, j, i + 2)
            q3 = vec_from_array(u, j, i + 3)
            q4 = vec_from_array(u, j, i + 4)
            q5 = vec_from_array(u, j, i + 5)
            q6 = vec_from_array(u, j, i + 6)
            write_vec(l0, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 0, 1, gamma))
            write_vec(r0, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 0, 1, gamma))
            write_vec(l1, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 1, 1, gamma))
            write_vec(r1, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 1, 1, gamma))
            write_vec(l2, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 2, 1, gamma))
            write_vec(r2, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 2, 1, gamma))
            write_vec(l3, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 3, 1, gamma))
            write_vec(r3, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 3, 1, gamma))


    @wp.kernel
    def compute_y_stage_weno7_big_kernel(u: wp.array3d(dtype=wp.float64), l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64), l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64), l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64), l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64), nx: int, ny: int, dy: wp.float64, gamma: wp.float64):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 8:
            q0 = vec_from_array(u, j + 0, i)
            q1 = vec_from_array(u, j + 1, i)
            q2 = vec_from_array(u, j + 2, i)
            q3 = vec_from_array(u, j + 3, i)
            q4 = vec_from_array(u, j + 4, i)
            q5 = vec_from_array(u, j + 5, i)
            q6 = vec_from_array(u, j + 6, i)
            write_vec(l0, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 0, 2, gamma))
            write_vec(r0, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 0, 2, gamma))
            write_vec(l1, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 1, 2, gamma))
            write_vec(r1, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 1, 2, gamma))
            write_vec(l2, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 2, 2, gamma))
            write_vec(r2, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 2, 2, gamma))
            write_vec(l3, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 3, 2, gamma))
            write_vec(r3, j, i, weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 3, 2, gamma))


    @wp.func
    def cross_x_one_side(a0: wp.array3d(dtype=wp.float64), a1: wp.array3d(dtype=wp.float64), a2: wp.array3d(dtype=wp.float64), a3: wp.array3d(dtype=wp.float64), j: int, i: int, lr: int, dy: wp.float64, dt: wp.float64, gamma: wp.float64, loca: int) -> wp.vec4d:
        q00 = vec_from_array(a0, j + 1, i); q01 = vec_from_array(a0, j + 2, i); q02 = vec_from_array(a0, j + 3, i); q03 = vec_from_array(a0, j + 4, i); q04 = vec_from_array(a0, j + 5, i); q05 = vec_from_array(a0, j + 6, i); q06 = vec_from_array(a0, j + 7, i)
        q10 = vec_from_array(a1, j + 1, i); q11 = vec_from_array(a1, j + 2, i); q12 = vec_from_array(a1, j + 3, i); q13 = vec_from_array(a1, j + 4, i); q14 = vec_from_array(a1, j + 5, i); q15 = vec_from_array(a1, j + 6, i); q16 = vec_from_array(a1, j + 7, i)
        q20 = vec_from_array(a2, j + 1, i); q21 = vec_from_array(a2, j + 2, i); q22 = vec_from_array(a2, j + 3, i); q23 = vec_from_array(a2, j + 4, i); q24 = vec_from_array(a2, j + 5, i); q25 = vec_from_array(a2, j + 6, i); q26 = vec_from_array(a2, j + 7, i)
        q30 = vec_from_array(a3, j + 1, i); q31 = vec_from_array(a3, j + 2, i); q32 = vec_from_array(a3, j + 3, i); q33 = vec_from_array(a3, j + 4, i); q34 = vec_from_array(a3, j + 5, i); q35 = vec_from_array(a3, j + 6, i); q36 = vec_from_array(a3, j + 7, i)
        u = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dy, 0)
        uy = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dy, 1)
        uyy = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dy, 2)
        uyyy = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dy, 3)
        ux = weno7_gauss_vec_conservative(q10, q11, q12, q13, q14, q15, q16, lr, dy, 0)
        uxy = weno7_gauss_vec_conservative(q10, q11, q12, q13, q14, q15, q16, lr, dy, 1)
        uxyy = weno7_gauss_vec_conservative(q10, q11, q12, q13, q14, q15, q16, lr, dy, 2)
        uxx = weno7_gauss_vec_conservative(q20, q21, q22, q23, q24, q25, q26, lr, dy, 0)
        uxxy = weno7_gauss_vec_conservative(q20, q21, q22, q23, q24, q25, q26, lr, dy, 1)
        uxxx = weno7_gauss_vec_conservative(q30, q31, q32, q33, q34, q35, q36, lr, dy, 0)
        qt, qtt, qttt = compute_euler_time_derivatives_2d_order3(u, ux, uy, uxx, uxy, uyy, uxxx, uxxy, uxyy, uyyy, gamma)
        return after_dritq_2d(qt, qtt, qttt, u, loca, dt)


    @wp.func
    def cross_y_one_side(a0: wp.array3d(dtype=wp.float64), a1: wp.array3d(dtype=wp.float64), a2: wp.array3d(dtype=wp.float64), a3: wp.array3d(dtype=wp.float64), j: int, i: int, lr: int, dx: wp.float64, dt: wp.float64, gamma: wp.float64, loca: int) -> wp.vec4d:
        q00 = vec_from_array(a0, j, i + 1); q01 = vec_from_array(a0, j, i + 2); q02 = vec_from_array(a0, j, i + 3); q03 = vec_from_array(a0, j, i + 4); q04 = vec_from_array(a0, j, i + 5); q05 = vec_from_array(a0, j, i + 6); q06 = vec_from_array(a0, j, i + 7)
        q10 = vec_from_array(a1, j, i + 1); q11 = vec_from_array(a1, j, i + 2); q12 = vec_from_array(a1, j, i + 3); q13 = vec_from_array(a1, j, i + 4); q14 = vec_from_array(a1, j, i + 5); q15 = vec_from_array(a1, j, i + 6); q16 = vec_from_array(a1, j, i + 7)
        q20 = vec_from_array(a2, j, i + 1); q21 = vec_from_array(a2, j, i + 2); q22 = vec_from_array(a2, j, i + 3); q23 = vec_from_array(a2, j, i + 4); q24 = vec_from_array(a2, j, i + 5); q25 = vec_from_array(a2, j, i + 6); q26 = vec_from_array(a2, j, i + 7)
        q30 = vec_from_array(a3, j, i + 1); q31 = vec_from_array(a3, j, i + 2); q32 = vec_from_array(a3, j, i + 3); q33 = vec_from_array(a3, j, i + 4); q34 = vec_from_array(a3, j, i + 5); q35 = vec_from_array(a3, j, i + 6); q36 = vec_from_array(a3, j, i + 7)
        u = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dx, 0)
        ux = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dx, 1)
        uxx = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dx, 2)
        uxxx = weno7_gauss_vec_conservative(q00, q01, q02, q03, q04, q05, q06, lr, dx, 3)
        uy = weno7_gauss_vec_conservative(q10, q11, q12, q13, q14, q15, q16, lr, dx, 0)
        uxy = weno7_gauss_vec_conservative(q10, q11, q12, q13, q14, q15, q16, lr, dx, 1)
        uxxy = weno7_gauss_vec_conservative(q10, q11, q12, q13, q14, q15, q16, lr, dx, 2)
        uyy = weno7_gauss_vec_conservative(q20, q21, q22, q23, q24, q25, q26, lr, dx, 0)
        uxyy = weno7_gauss_vec_conservative(q20, q21, q22, q23, q24, q25, q26, lr, dx, 1)
        uyyy = weno7_gauss_vec_conservative(q30, q31, q32, q33, q34, q35, q36, lr, dx, 0)
        qt, qtt, qttt = compute_euler_time_derivatives_2d_order3(u, ux, uy, uxx, uxy, uyy, uxxx, uxxy, uxyy, uyyy, gamma)
        return after_dritq_2d(qt, qtt, qttt, u, loca, dt)


    @wp.kernel
    def compute_x_cross_stage_ader4_kernel(tl1: wp.array3d(dtype=wp.float64), tl2: wp.array3d(dtype=wp.float64), tl3: wp.array3d(dtype=wp.float64), tr1: wp.array3d(dtype=wp.float64), tr2: wp.array3d(dtype=wp.float64), tr3: wp.array3d(dtype=wp.float64), l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64), l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64), l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64), l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64), nx: int, ny: int, dy: wp.float64, dt: wp.float64, loca: int, gamma: wp.float64):
        j, i = wp.tid()
        if j < ny and i < nx + 2:
            write_vec(tl1, j, i, cross_x_one_side(l0, l1, l2, l3, j, i, loca, dy, dt, gamma, 1))
            write_vec(tl2, j, i, cross_x_one_side(l0, l1, l2, l3, j, i, loca, dy, dt, gamma, 2))
            write_vec(tl3, j, i, cross_x_one_side(l0, l1, l2, l3, j, i, loca, dy, dt, gamma, 3))
            write_vec(tr1, j, i, cross_x_one_side(r0, r1, r2, r3, j, i, loca, dy, dt, gamma, 1))
            write_vec(tr2, j, i, cross_x_one_side(r0, r1, r2, r3, j, i, loca, dy, dt, gamma, 2))
            write_vec(tr3, j, i, cross_x_one_side(r0, r1, r2, r3, j, i, loca, dy, dt, gamma, 3))


    @wp.kernel
    def compute_y_cross_stage_ader4_kernel(tl1: wp.array3d(dtype=wp.float64), tl2: wp.array3d(dtype=wp.float64), tl3: wp.array3d(dtype=wp.float64), tr1: wp.array3d(dtype=wp.float64), tr2: wp.array3d(dtype=wp.float64), tr3: wp.array3d(dtype=wp.float64), l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64), l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64), l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64), l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64), nx: int, ny: int, dx: wp.float64, dt: wp.float64, loca: int, gamma: wp.float64):
        j, i = wp.tid()
        if j < ny + 2 and i < nx:
            write_vec(tl1, j, i, cross_y_one_side(l0, l1, l2, l3, j, i, loca, dx, dt, gamma, 1))
            write_vec(tl2, j, i, cross_y_one_side(l0, l1, l2, l3, j, i, loca, dx, dt, gamma, 2))
            write_vec(tl3, j, i, cross_y_one_side(l0, l1, l2, l3, j, i, loca, dx, dt, gamma, 3))
            write_vec(tr1, j, i, cross_y_one_side(r0, r1, r2, r3, j, i, loca, dx, dt, gamma, 1))
            write_vec(tr2, j, i, cross_y_one_side(r0, r1, r2, r3, j, i, loca, dx, dt, gamma, 2))
            write_vec(tr3, j, i, cross_y_one_side(r0, r1, r2, r3, j, i, loca, dx, dt, gamma, 3))


    @wp.kernel
    def compute_x_flux_ader4_kernel(flux_x: wp.array3d(dtype=wp.float64), tl1: wp.array3d(dtype=wp.float64), tl2: wp.array3d(dtype=wp.float64), tl3: wp.array3d(dtype=wp.float64), tr1: wp.array3d(dtype=wp.float64), tr2: wp.array3d(dtype=wp.float64), tr3: wp.array3d(dtype=wp.float64), tempdx_dt: wp.float64, nx: int, ny: int, loca: int, gamma: wp.float64, solver_kind: int):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            left1 = vec_from_array(tl1, j, i + 1); right1 = vec_from_array(tr1, j, i)
            left2 = vec_from_array(tl2, j, i + 1); right2 = vec_from_array(tr2, j, i)
            left3 = vec_from_array(tl3, j, i + 1); right3 = vec_from_array(tr3, j, i)
            f1 = riemann_flux(right1, left1, 1, tempdx_dt, gamma, solver_kind)
            f2 = riemann_flux(right2, left2, 1, tempdx_dt, gamma, solver_kind)
            f3 = riemann_flux(right3, left3, 1, tempdx_dt, gamma, solver_kind)
            total = wp.vec4d(
                tempdx_dt * (wp.float64(5.0) * f1[0] + wp.float64(8.0) * f2[0] + wp.float64(5.0) * f3[0]),
                tempdx_dt * (wp.float64(5.0) * f1[1] + wp.float64(8.0) * f2[1] + wp.float64(5.0) * f3[1]),
                tempdx_dt * (wp.float64(5.0) * f1[2] + wp.float64(8.0) * f2[2] + wp.float64(5.0) * f3[2]),
                tempdx_dt * (wp.float64(5.0) * f1[3] + wp.float64(8.0) * f2[3] + wp.float64(5.0) * f3[3]),
            )
            if loca == 1:
                write_vec(flux_x, j, i, total)
            else:
                old = vec_from_array(flux_x, j, i)
                write_vec(flux_x, j, i, old + total)


    @wp.kernel
    def compute_y_flux_ader4_kernel(flux_y: wp.array3d(dtype=wp.float64), tl1: wp.array3d(dtype=wp.float64), tl2: wp.array3d(dtype=wp.float64), tl3: wp.array3d(dtype=wp.float64), tr1: wp.array3d(dtype=wp.float64), tr2: wp.array3d(dtype=wp.float64), tr3: wp.array3d(dtype=wp.float64), tempdy_dt: wp.float64, nx: int, ny: int, loca: int, gamma: wp.float64, solver_kind: int):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            left1 = vec_from_array(tl1, j + 1, i); right1 = vec_from_array(tr1, j, i)
            left2 = vec_from_array(tl2, j + 1, i); right2 = vec_from_array(tr2, j, i)
            left3 = vec_from_array(tl3, j + 1, i); right3 = vec_from_array(tr3, j, i)
            f1 = riemann_flux(right1, left1, 2, tempdy_dt, gamma, solver_kind)
            f2 = riemann_flux(right2, left2, 2, tempdy_dt, gamma, solver_kind)
            f3 = riemann_flux(right3, left3, 2, tempdy_dt, gamma, solver_kind)
            total = wp.vec4d(
                tempdy_dt * (wp.float64(5.0) * f1[0] + wp.float64(8.0) * f2[0] + wp.float64(5.0) * f3[0]),
                tempdy_dt * (wp.float64(5.0) * f1[1] + wp.float64(8.0) * f2[1] + wp.float64(5.0) * f3[1]),
                tempdy_dt * (wp.float64(5.0) * f1[2] + wp.float64(8.0) * f2[2] + wp.float64(5.0) * f3[2]),
                tempdy_dt * (wp.float64(5.0) * f1[3] + wp.float64(8.0) * f2[3] + wp.float64(5.0) * f3[3]),
            )
            if loca == 1:
                write_vec(flux_y, j, i, total)
            else:
                old = vec_from_array(flux_y, j, i)
                write_vec(flux_y, j, i, old + total)


    @wp.kernel
    def update_ader4_kernel(u: wp.array3d(dtype=wp.float64), flux_x: wp.array3d(dtype=wp.float64), flux_y: wp.array3d(dtype=wp.float64), pri: wp.array3d(dtype=wp.float64), nx: int, ny: int, gc: int, gamma: wp.float64):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = vec_from_array(u, jj, ii)
            fx_l = vec_from_array(flux_x, j, i)
            fx_r = vec_from_array(flux_x, j, i + 1)
            fy_d = vec_from_array(flux_y, j, i)
            fy_u = vec_from_array(flux_y, j + 1, i)
            scale = wp.float64(1.0) / wp.float64(36.0)
            qn = wp.vec4d(
                q[0] - scale * (fx_r[0] - fx_l[0]) - scale * (fy_u[0] - fy_d[0]),
                q[1] - scale * (fx_r[1] - fx_l[1]) - scale * (fy_u[1] - fy_d[1]),
                q[2] - scale * (fx_r[2] - fx_l[2]) - scale * (fy_u[2] - fy_d[2]),
                q[3] - scale * (fx_r[3] - fx_l[3]) - scale * (fy_u[3] - fy_d[3]),
            )
            write_vec(u, jj, ii, qn)
            write_vec(pri, jj, ii, con_to_pri(qn, gamma))
