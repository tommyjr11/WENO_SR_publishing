"""Reusable helpers for the Warp WENO5/RK3 prototype.

The host initialization mirrors the current shock-bubble setup in
ADER_TR_Project/initialize.cu: conservative variables, 3 ghost cells, and
15x15 Gauss-Legendre cell averages.  The Warp kernels use float64 to match the
CUDA project's Double64 configuration more closely.
"""

from __future__ import annotations

import os
import sys
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
    ghost: int = 3
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
    u_air = 0.0
    v_air = 0.0

    rho_he = 0.214
    rho_post = (((gamma + 1.0) * mach * mach) / ((gamma - 1.0) * mach * mach + 2.0)) * rho_air
    p_post = ((((2.0 * gamma) * mach * mach) - (gamma - 1.0)) / (gamma + 1.0)) * p_air
    u_post = 110.6273
    v_post = 0.0

    if x < x_shock:
        con = primitive_to_conserved(rho_post, u_post, v_post, p_post, gamma)
    else:
        con = primitive_to_conserved(rho_air, u_air, v_air, p_air, gamma)

    bubble_xc = 0.035
    bubble_yc = 0.0445
    bubble_r = 0.025
    dx_bubble = x - bubble_xc
    dy_bubble = y - bubble_yc
    if dx_bubble * dx_bubble + dy_bubble * dy_bubble <= bubble_r * bubble_r:
        con = primitive_to_conserved(rho_he, 0.0, 0.0, p_air, gamma)

    return con


def function2d_shock_bubble_conserved(x: float, y: float, comp: int, t: float = 0.0) -> float:
    con = shock_bubble_conserved_state(x, y, t)
    return float(con[comp])


def cell_average_2d_15point(x_center: float, y_center: float, dx: float, dy: float, comp: int, t: float = 0.0) -> float:
    sum_val = 0.0
    for ix, wx in zip(GAUSS15_XI, GAUSS15_W):
        x = x_center + 0.5 * dx * float(ix)
        for iy, wy in zip(GAUSS15_XI, GAUSS15_W):
            y = y_center + 0.5 * dy * float(iy)
            sum_val += float(wx) * float(wy) * function2d_shock_bubble_conserved(x, y, comp, t)
    return 0.25 * sum_val


def cell_average_state_2d_15point(x_center: float, y_center: float, dx: float, dy: float, t: float = 0.0) -> np.ndarray:
    state = np.zeros(4, dtype=np.float64)
    for ix, wx in zip(GAUSS15_XI, GAUSS15_W):
        x = x_center + 0.5 * dx * float(ix)
        for iy, wy in zip(GAUSS15_XI, GAUSS15_W):
            y = y_center + 0.5 * dy * float(iy)
            weight = float(wx) * float(wy)
            state += weight * shock_bubble_conserved_state(x, y, t)
    return 0.25 * state


def make_initial_state(params: Params) -> np.ndarray:
    ny_total, nx_total, _ = params.padded_shape
    u = np.zeros(params.padded_shape, dtype=np.float64)

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
        inputs=[
            u_device,
            speed_workspace,
            params.nx,
            params.ny,
            params.ghost,
            wp.float64(params.gamma),
        ],
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
    mass = float(np.sum(interior[..., 0]) * params.dx * params.dy)
    return {
        "mass": mass,
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
        print(
            "NVIDIA Warp is not installed. Install it with:\n"
            "    pip install warp-lang\n",
            file=sys.stderr,
        )
        raise SystemExit(1)


if wp is not None:
    WENO5_USE_NUMERICAL_WEIGHTS = wp.constant(int(os.environ.get("WENO5_USE_NUMERICAL_WEIGHTS", "1")))

    @wp.func
    def safe_rcp(x: wp.float64) -> wp.float64:
        return wp.float64(1.0) / (wp.float64(1.0e-6) + x)


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
            return wp.vec4d(
                rho * u,
                rho * u * u + p,
                rho * u * v,
                u * (p + e),
            )

        return wp.vec4d(
            rho * v,
            rho * u * v,
            rho * v * v + p,
            v * (p + e),
        )


    @wp.func
    def evilin_state_2d(
        ul0: wp.vec4d,
        ur0: wp.vec4d,
        direction: int,
        c: wp.float64,
        gamma: wp.float64,
    ) -> wp.vec4d:
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
    def force_flux(
        ul0: wp.vec4d,
        ur0: wp.vec4d,
        direction: int,
        c: wp.float64,
        gamma: wp.float64,
    ) -> wp.vec4d:
        wl0 = con_to_pri(ul0, gamma)
        wr0 = con_to_pri(ur0, gamma)
        fl = pri_to_flux(wl0, direction, gamma)
        fr = pri_to_flux(wr0, direction, gamma)

        ri_u = wp.vec4d(
            wp.float64(0.5) * (ul0[0] + ur0[0]) - wp.float64(0.5) * c * (fr[0] - fl[0]),
            wp.float64(0.5) * (ul0[1] + ur0[1]) - wp.float64(0.5) * c * (fr[1] - fl[1]),
            wp.float64(0.5) * (ul0[2] + ur0[2]) - wp.float64(0.5) * c * (fr[2] - fl[2]),
            wp.float64(0.5) * (ul0[3] + ur0[3]) - wp.float64(0.5) * c * (fr[3] - fl[3]),
        )
        ri = pri_to_flux(con_to_pri(ri_u, gamma), direction, gamma)
        dx_dt = wp.float64(1.0) / c

        lf = wp.vec4d(
            wp.float64(0.5) * (fl[0] + fr[0]) - wp.float64(0.5) * dx_dt * (ur0[0] - ul0[0]),
            wp.float64(0.5) * (fl[1] + fr[1]) - wp.float64(0.5) * dx_dt * (ur0[1] - ul0[1]),
            wp.float64(0.5) * (fl[2] + fr[2]) - wp.float64(0.5) * dx_dt * (ur0[2] - ul0[2]),
            wp.float64(0.5) * (fl[3] + fr[3]) - wp.float64(0.5) * dx_dt * (ur0[3] - ul0[3]),
        )

        return wp.vec4d(
            wp.float64(0.5) * (lf[0] + ri[0]),
            wp.float64(0.5) * (lf[1] + ri[1]),
            wp.float64(0.5) * (lf[2] + ri[2]),
            wp.float64(0.5) * (lf[3] + ri[3]),
        )


    @wp.func
    def stencil_2d(u1: wp.float64, u2: wp.float64, u3: wp.float64, location: int) -> wp.float64:
        if location == 1:
            return (wp.float64(1.0) / wp.float64(3.0)) * u1 - (wp.float64(7.0) / wp.float64(6.0)) * u2 + (wp.float64(11.0) / wp.float64(6.0)) * u3
        if location == 2:
            return (-wp.float64(1.0) / wp.float64(6.0)) * u1 + (wp.float64(5.0) / wp.float64(6.0)) * u2 + (wp.float64(1.0) / wp.float64(3.0)) * u3
        if location == 3:
            return (wp.float64(1.0) / wp.float64(3.0)) * u1 + (wp.float64(5.0) / wp.float64(6.0)) * u2 - (wp.float64(1.0) / wp.float64(6.0)) * u3
        return (wp.float64(11.0) / wp.float64(6.0)) * u1 - (wp.float64(7.0) / wp.float64(6.0)) * u2 + (wp.float64(1.0) / wp.float64(3.0)) * u3


    @wp.func
    def weno5_nn_features_nogueira(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64) -> wp.vec4d:
        d20 = q0 - wp.float64(2.0) * q1 + q2
        d21 = q1 - wp.float64(2.0) * q2 + q3
        d22 = q2 - wp.float64(2.0) * q3 + q4

        delta0 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d20) + wp.float64(0.25) * wp.abs(q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2)
        delta1 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d21) + wp.float64(0.25) * wp.abs(q1 - q3)
        delta2 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d22) + wp.float64(0.25) * wp.abs(wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4)

        eps = wp.float64(1.0e-15)
        delta_max = wp.max(wp.max(delta0, delta1), delta2)
        inv_delta_max = wp.float64(1.0) / wp.max(delta_max, eps)

        gamma0 = wp.abs(d20) / (wp.abs(q1 - q0) + wp.abs(q2 - q1) + eps)
        gamma1 = wp.abs(d21) / (wp.abs(q2 - q1) + wp.abs(q3 - q2) + eps)
        gamma2 = wp.abs(d22) / (wp.abs(q3 - q2) + wp.abs(q4 - q3) + eps)
        gamma_s = wp.min(wp.float64(1.0), wp.max(wp.max(gamma0, gamma1), gamma2))
        return wp.vec4d(delta0 * inv_delta_max, delta1 * inv_delta_max, delta2 * inv_delta_max, gamma_s)


    @wp.func
    def weno5_eno_cutoff(weights: wp.vec3d) -> wp.vec3d:
        psi0 = wp.float64(1.0)
        psi1 = wp.float64(1.0)
        psi2 = wp.float64(1.0)
        cutoff = wp.float64(4.0e-7)
        if weights[0] <= cutoff:
            psi0 = wp.float64(0.0)
        if weights[1] <= cutoff:
            psi1 = wp.float64(0.0)
        if weights[2] <= cutoff:
            psi2 = wp.float64(0.0)
        weight_sum = psi0 * weights[0] + psi1 * weights[1] + psi2 * weights[2]
        inv_sum = wp.float64(1.0) / wp.max(weight_sum, wp.float64(1.0e-300))
        return wp.vec3d(psi0 * weights[0] * inv_sum, psi1 * weights[1] * inv_sum, psi2 * weights[2] * inv_sum)


    @wp.func
    def MLP_W_calculate_weno5(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        lr: int,
    ) -> wp.vec3d:
        features = weno5_nn_features_nogueira(q0, q1, q2, q3, q4)
        unused_gate = features[3] * wp.float64(0.0)
        if lr == 1:
            return weno5_eno_cutoff(wp.vec3d(
                wp.float64(1.0) / wp.float64(10.0) + unused_gate,
                wp.float64(3.0) / wp.float64(5.0),
                wp.float64(3.0) / wp.float64(10.0),
            ))
        return weno5_eno_cutoff(wp.vec3d(
            wp.float64(3.0) / wp.float64(10.0) + unused_gate,
            wp.float64(3.0) / wp.float64(5.0),
            wp.float64(1.0) / wp.float64(10.0),
        ))


    @wp.func
    def weno5_lr_value(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        lr: int,
    ) -> wp.float64:
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        d0 = wp.float64(0.0)
        d1 = wp.float64(3.0) / wp.float64(5.0)
        d2 = wp.float64(0.0)

        if lr == 1:
            s0 = stencil_2d(q0, q1, q2, 1)
            s1 = stencil_2d(q1, q2, q3, 2)
            s2 = stencil_2d(q2, q3, q4, 3)
            d0 = wp.float64(1.0) / wp.float64(10.0)
            d2 = wp.float64(3.0) / wp.float64(10.0)
        else:
            s0 = stencil_2d(q0, q1, q2, 2)
            s1 = stencil_2d(q1, q2, q3, 3)
            s2 = stencil_2d(q2, q3, q4, 4)
            d0 = wp.float64(3.0) / wp.float64(10.0)
            d2 = wp.float64(1.0) / wp.float64(10.0)

        if WENO5_USE_NUMERICAL_WEIGHTS == 0:
            mlp_w = MLP_W_calculate_weno5(q0, q1, q2, q3, q4, lr)
            return mlp_w[0] * s0 + mlp_w[1] * s1 + mlp_w[2] * s2

        beta2 = (
            (wp.float64(13.0) / wp.float64(12.0)) * (q2 - wp.float64(2.0) * q3 + q4) * (q2 - wp.float64(2.0) * q3 + q4)
            + wp.float64(0.25) * (wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4) * (wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4)
        )
        beta1 = (
            (wp.float64(13.0) / wp.float64(12.0)) * (q1 - wp.float64(2.0) * q2 + q3) * (q1 - wp.float64(2.0) * q2 + q3)
            + wp.float64(0.25) * (q1 - q3) * (q1 - q3)
        )
        beta0 = (
            (wp.float64(13.0) / wp.float64(12.0)) * (q0 - wp.float64(2.0) * q1 + q2) * (q0 - wp.float64(2.0) * q1 + q2)
            + wp.float64(0.25) * (q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2) * (q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2)
        )

        inv0 = safe_rcp(beta0)
        inv1 = safe_rcp(beta1)
        inv2 = safe_rcp(beta2)

        alpha0 = d0 * inv0 * inv0
        alpha1 = d1 * inv1 * inv1
        alpha2 = d2 * inv2 * inv2
        alpha_sum = alpha0 + alpha1 + alpha2

        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum

        return w0 * s0 + w1 * s1 + w2 * s2


    @wp.func
    def MLP_W_calculate_weno5_gauss(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        lr: int,
    ) -> wp.vec3d:
        root3 = wp.sqrt(wp.float64(3.0))
        features = weno5_nn_features_nogueira(q0, q1, q2, q3, q4)
        unused_gate = features[3] * wp.float64(0.0)
        if lr == 1:
            return weno5_eno_cutoff(wp.vec3d(
                (wp.float64(210.0) + root3) / wp.float64(1080.0) + unused_gate,
                wp.float64(11.0) / wp.float64(18.0),
                (wp.float64(210.0) - root3) / wp.float64(1080.0),
            ))
        return weno5_eno_cutoff(wp.vec3d(
            (wp.float64(210.0) - root3) / wp.float64(1080.0) + unused_gate,
            wp.float64(11.0) / wp.float64(18.0),
            (wp.float64(210.0) + root3) / wp.float64(1080.0),
        ))


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

        c0 = (
            (h + a * (u - a) / gm1) * q[0]
            - (u + a / gm1) * q[1]
            - v * q[2]
            + q[3]
        ) * scale
        c1 = (
            (-wp.float64(2.0) * h + wp.float64(4.0) * a2 / gm1) * q[0]
            + wp.float64(2.0) * u * q[1]
            + wp.float64(2.0) * v * q[2]
            - wp.float64(2.0) * q[3]
        ) * scale
        c2 = (
            -wp.float64(2.0) * v * a2 / gm1 * q[0]
            + wp.float64(2.0) * a2 / gm1 * q[2]
        ) * scale
        c3 = (
            (h - a * (u + a) / gm1) * q[0]
            + (-u + a / gm1) * q[1]
            - v * q[2]
            + q[3]
        ) * scale
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

        c0 = (
            (h + a * (normal - a) / gm1) * q[0]
            - tangent * q[1]
            - (normal + a / gm1) * q[2]
            + q[3]
        ) * scale
        c1 = (
            (-wp.float64(2.0) * h + wp.float64(4.0) * a2 / gm1) * q[0]
            + wp.float64(2.0) * tangent * q[1]
            + wp.float64(2.0) * normal * q[2]
            - wp.float64(2.0) * q[3]
        ) * scale
        c2 = (
            -wp.float64(2.0) * tangent * a2 / gm1 * q[0]
            + wp.float64(2.0) * a2 / gm1 * q[1]
        ) * scale
        c3 = (
            (h - a * (normal + a) / gm1) * q[0]
            - tangent * q[1]
            + (-normal + a / gm1) * q[2]
            + q[3]
        ) * scale
        return wp.vec4d(c0, c1, c2, c3)


    @wp.func
    def char_to_con_y(c: wp.vec4d, roe: wp.vec4d, gamma: wp.float64) -> wp.vec4d:
        values = jacobian_state_values(roe, gamma)
        tangent = values[0]
        normal = values[1]
        a = values[2]
        h = values[3]
        q2 = tangent * tangent + normal * normal

        return wp.vec4d(
            c[0] + c[1] + c[3],
            tangent * c[0] + tangent * c[1] + c[2] + tangent * c[3],
            (normal - a) * c[0] + normal * c[1] + (normal + a) * c[3],
            (h - normal * a) * c[0] + wp.float64(0.5) * q2 * c[1] + tangent * c[2] + (h + normal * a) * c[3],
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
    def weno5_lr_value_characteristic(
        q0: wp.vec4d,
        q1: wp.vec4d,
        q2: wp.vec4d,
        q3: wp.vec4d,
        q4: wp.vec4d,
        lr: int,
        direction: int,
        gamma: wp.float64,
    ) -> wp.vec4d:
        left = q2
        right = q3
        if lr == 2:
            left = q1
            right = q2

        roe = roe_average_state(left, right, gamma)
        c0 = con_to_char(q0, roe, direction, gamma)
        c1 = con_to_char(q1, roe, direction, gamma)
        c2 = con_to_char(q2, roe, direction, gamma)
        c3 = con_to_char(q3, roe, direction, gamma)
        c4 = con_to_char(q4, roe, direction, gamma)

        c_face = wp.vec4d(
            weno5_lr_value(c0[0], c1[0], c2[0], c3[0], c4[0], lr),
            weno5_lr_value(c0[1], c1[1], c2[1], c3[1], c4[1], lr),
            weno5_lr_value(c0[2], c1[2], c2[2], c3[2], c4[2], lr),
            weno5_lr_value(c0[3], c1[3], c2[3], c3[3], c4[3], lr),
        )
        return char_to_con(c_face, roe, direction, gamma)


    @wp.func
    def stencil_gauss(u1: wp.float64, u2: wp.float64, u3: wp.float64, location: int) -> wp.float64:
        if location == 1:
            return wp.float64(-0.144337567297407) * u1 + wp.float64(0.577350269189626) * u2 + wp.float64(0.566987298107780) * u3
        if location == 2:
            return wp.float64(0.144337567297406) * u1 + u2 - wp.float64(0.144337567297407) * u3
        if location == 3:
            return wp.float64(1.433012701892220) * u1 - wp.float64(0.577350269189626) * u2 + wp.float64(0.144337567297406) * u3
        if location == 4:
            return wp.float64(0.144337567297406) * u1 - wp.float64(0.577350269189626) * u2 + wp.float64(1.433012701892220) * u3
        if location == 5:
            return -wp.float64(0.144337567297407) * u1 + u2 + wp.float64(0.144337567297406) * u3
        return wp.float64(0.566987298107781) * u1 + wp.float64(0.577350269189625) * u2 - wp.float64(0.144337567297406) * u3


    @wp.func
    def weno5_gauss_lr_value(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        lr: int,
    ) -> wp.float64:
        root3 = wp.sqrt(wp.float64(3.0))
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        d0 = wp.float64(0.0)
        d1 = wp.float64(11.0) / wp.float64(18.0)
        d2 = wp.float64(0.0)

        if lr == 1:
            s0 = stencil_gauss(q0, q1, q2, 1)
            s1 = stencil_gauss(q1, q2, q3, 2)
            s2 = stencil_gauss(q2, q3, q4, 3)
            d0 = (wp.float64(210.0) + root3) / wp.float64(1080.0)
            d2 = (wp.float64(210.0) - root3) / wp.float64(1080.0)
        else:
            s0 = stencil_gauss(q0, q1, q2, 4)
            s1 = stencil_gauss(q1, q2, q3, 5)
            s2 = stencil_gauss(q2, q3, q4, 6)
            d0 = (wp.float64(210.0) - root3) / wp.float64(1080.0)
            d2 = (wp.float64(210.0) + root3) / wp.float64(1080.0)

        if WENO5_USE_NUMERICAL_WEIGHTS == 0:
            mlp_w = MLP_W_calculate_weno5_gauss(q0, q1, q2, q3, q4, lr)
            return mlp_w[0] * s0 + mlp_w[1] * s1 + mlp_w[2] * s2

        beta2 = (
            (wp.float64(13.0) / wp.float64(12.0)) * (q2 - wp.float64(2.0) * q3 + q4) * (q2 - wp.float64(2.0) * q3 + q4)
            + wp.float64(0.25) * (wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4) * (wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4)
        )
        beta1 = (
            (wp.float64(13.0) / wp.float64(12.0)) * (q1 - wp.float64(2.0) * q2 + q3) * (q1 - wp.float64(2.0) * q2 + q3)
            + wp.float64(0.25) * (q1 - q3) * (q1 - q3)
        )
        beta0 = (
            (wp.float64(13.0) / wp.float64(12.0)) * (q0 - wp.float64(2.0) * q1 + q2) * (q0 - wp.float64(2.0) * q1 + q2)
            + wp.float64(0.25) * (q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2) * (q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2)
        )

        inv0 = safe_rcp(beta0)
        inv1 = safe_rcp(beta1)
        inv2 = safe_rcp(beta2)

        alpha0 = d0 * inv0 * inv0
        alpha1 = d1 * inv1 * inv1
        alpha2 = d2 * inv2 * inv2
        alpha_sum = alpha0 + alpha1 + alpha2

        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum

        return w0 * s0 + w1 * s1 + w2 * s2


    @wp.kernel
    def compute_max_speed_kernel(
        u: wp.array3d(dtype=wp.float64),
        speed: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            rho = wp.max(u[jj, ii, 0], wp.float64(1.0e-15))
            vel_x = u[jj, ii, 1] / rho
            vel_y = u[jj, ii, 2] / rho
            pressure = (gamma - wp.float64(1.0)) * (
                u[jj, ii, 3] - wp.float64(0.5) * rho * (vel_x * vel_x + vel_y * vel_y)
            )
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
    def compute_x_stage_weno_kernel(
        u: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        characteristic: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 6 and i < nx + 2:
            if characteristic != 0:
                q0 = wp.vec4d(u[j, i + 0, 0], u[j, i + 0, 1], u[j, i + 0, 2], u[j, i + 0, 3])
                q1 = wp.vec4d(u[j, i + 1, 0], u[j, i + 1, 1], u[j, i + 1, 2], u[j, i + 1, 3])
                q2 = wp.vec4d(u[j, i + 2, 0], u[j, i + 2, 1], u[j, i + 2, 2], u[j, i + 2, 3])
                q3 = wp.vec4d(u[j, i + 3, 0], u[j, i + 3, 1], u[j, i + 3, 2], u[j, i + 3, 3])
                q4 = wp.vec4d(u[j, i + 4, 0], u[j, i + 4, 1], u[j, i + 4, 2], u[j, i + 4, 3])
                ql = weno5_lr_value_characteristic(q0, q1, q2, q3, q4, 2, 1, gamma)
                qr = weno5_lr_value_characteristic(q0, q1, q2, q3, q4, 1, 1, gamma)
                for comp in range(4):
                    temp_l[j, i, comp] = ql[comp]
                    temp_r[j, i, comp] = qr[comp]
            else:
                for comp in range(4):
                    q0s = u[j, i + 0, comp]
                    q1s = u[j, i + 1, comp]
                    q2s = u[j, i + 2, comp]
                    q3s = u[j, i + 3, comp]
                    q4s = u[j, i + 4, comp]
                    temp_l[j, i, comp] = weno5_lr_value(q0s, q1s, q2s, q3s, q4s, 2)
                    temp_r[j, i, comp] = weno5_lr_value(q0s, q1s, q2s, q3s, q4s, 1)


    @wp.kernel
    def compute_x_point_weno_kernel(
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        loca: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 2:
            for comp in range(4):
                q0 = temp_l[j + 1, i, comp]
                q1 = temp_l[j + 2, i, comp]
                q2 = temp_l[j + 3, i, comp]
                q3 = temp_l[j + 4, i, comp]
                q4 = temp_l[j + 5, i, comp]
                point_l[j, i, comp] = weno5_gauss_lr_value(q0, q1, q2, q3, q4, loca)

                q0 = temp_r[j + 1, i, comp]
                q1 = temp_r[j + 2, i, comp]
                q2 = temp_r[j + 3, i, comp]
                q3 = temp_r[j + 4, i, comp]
                q4 = temp_r[j + 5, i, comp]
                point_r[j, i, comp] = weno5_gauss_lr_value(q0, q1, q2, q3, q4, loca)


    @wp.kernel
    def compute_x_flux_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        tempdx_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            left_state = wp.vec4d(
                point_l[j, i + 1, 0],
                point_l[j, i + 1, 1],
                point_l[j, i + 1, 2],
                point_l[j, i + 1, 3],
            )
            right_state = wp.vec4d(
                point_r[j, i, 0],
                point_r[j, i, 1],
                point_r[j, i, 2],
                point_r[j, i, 3],
            )
            # face_con = evilin_state_2d(right_state, left_state, 1, tempdx_dt, gamma)
            # f = pri_to_flux(con_to_pri(face_con, gamma), 1, gamma)
            f = force_flux(right_state, left_state, 1, tempdx_dt, gamma)

            if loca == 1:
                for comp in range(4):
                    flux_x[j, i, comp] = f[comp] * tempdx_dt
            else:
                for comp in range(4):
                    flux_x[j, i, comp] = flux_x[j, i, comp] + f[comp] * tempdx_dt


    @wp.kernel
    def compute_y_stage_weno_kernel(
        u: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        characteristic: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 6:
            if characteristic != 0:
                q0 = wp.vec4d(u[j + 0, i, 0], u[j + 0, i, 1], u[j + 0, i, 2], u[j + 0, i, 3])
                q1 = wp.vec4d(u[j + 1, i, 0], u[j + 1, i, 1], u[j + 1, i, 2], u[j + 1, i, 3])
                q2 = wp.vec4d(u[j + 2, i, 0], u[j + 2, i, 1], u[j + 2, i, 2], u[j + 2, i, 3])
                q3 = wp.vec4d(u[j + 3, i, 0], u[j + 3, i, 1], u[j + 3, i, 2], u[j + 3, i, 3])
                q4 = wp.vec4d(u[j + 4, i, 0], u[j + 4, i, 1], u[j + 4, i, 2], u[j + 4, i, 3])
                ql = weno5_lr_value_characteristic(q0, q1, q2, q3, q4, 2, 2, gamma)
                qr = weno5_lr_value_characteristic(q0, q1, q2, q3, q4, 1, 2, gamma)
                for comp in range(4):
                    temp_l[j, i, comp] = ql[comp]
                    temp_r[j, i, comp] = qr[comp]
            else:
                for comp in range(4):
                    q0s = u[j + 0, i, comp]
                    q1s = u[j + 1, i, comp]
                    q2s = u[j + 2, i, comp]
                    q3s = u[j + 3, i, comp]
                    q4s = u[j + 4, i, comp]
                    temp_l[j, i, comp] = weno5_lr_value(q0s, q1s, q2s, q3s, q4s, 2)
                    temp_r[j, i, comp] = weno5_lr_value(q0s, q1s, q2s, q3s, q4s, 1)


    @wp.kernel
    def compute_y_point_weno_kernel(
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        loca: int,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx:
            for comp in range(4):
                q0 = temp_l[j, i + 1, comp]
                q1 = temp_l[j, i + 2, comp]
                q2 = temp_l[j, i + 3, comp]
                q3 = temp_l[j, i + 4, comp]
                q4 = temp_l[j, i + 5, comp]
                point_l[j, i, comp] = weno5_gauss_lr_value(q0, q1, q2, q3, q4, loca)

                q0 = temp_r[j, i + 1, comp]
                q1 = temp_r[j, i + 2, comp]
                q2 = temp_r[j, i + 3, comp]
                q3 = temp_r[j, i + 4, comp]
                q4 = temp_r[j, i + 5, comp]
                point_r[j, i, comp] = weno5_gauss_lr_value(q0, q1, q2, q3, q4, loca)


    @wp.kernel
    def compute_y_flux_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        tempdy_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            left_state = wp.vec4d(
                point_l[j + 1, i, 0],
                point_l[j + 1, i, 1],
                point_l[j + 1, i, 2],
                point_l[j + 1, i, 3],
            )
            right_state = wp.vec4d(
                point_r[j, i, 0],
                point_r[j, i, 1],
                point_r[j, i, 2],
                point_r[j, i, 3],
            )
            # face_con = evilin_state_2d(right_state, left_state, 2, tempdy_dt, gamma)
            # f = pri_to_flux(con_to_pri(face_con, gamma), 2, gamma)
            f = force_flux(right_state, left_state, 2, tempdy_dt, gamma)

            if loca == 1:
                for comp in range(4):
                    flux_y[j, i, comp] = f[comp] * tempdy_dt
            else:
                for comp in range(4):
                    flux_y[j, i, comp] = flux_y[j, i, comp] + f[comp] * tempdy_dt


    @wp.kernel
    def update_rk3_kernel(
        u: wp.array3d(dtype=wp.float64),
        u0: wp.array3d(dtype=wp.float64),
        flux_x: wp.array3d(dtype=wp.float64),
        flux_y: wp.array3d(dtype=wp.float64),
        pri: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        rk: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc

            if rk == 1:
                for comp in range(4):
                    u0[jj, ii, comp] = u[jj, ii, comp]

            for comp in range(4):
                rhs = (
                    u[jj, ii, comp]
                    - wp.float64(0.5) * (flux_x[j, i + 1, comp] - flux_x[j, i, comp])
                    - wp.float64(0.5) * (flux_y[j + 1, i, comp] - flux_y[j, i, comp])
                )
                if rk == 1:
                    u[jj, ii, comp] = rhs
                elif rk == 2:
                    u[jj, ii, comp] = wp.float64(0.75) * u0[jj, ii, comp] + wp.float64(0.25) * rhs
                else:
                    u[jj, ii, comp] = (wp.float64(1.0) / wp.float64(3.0)) * u0[jj, ii, comp] + (wp.float64(2.0) / wp.float64(3.0)) * rhs

            if rk == 3:
                q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
                w = con_to_pri(q, gamma)
                for comp in range(4):
                    pri[jj, ii, comp] = w[comp]
