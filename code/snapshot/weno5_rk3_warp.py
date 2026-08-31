#!/usr/bin/env python3
"""Command-line demo for the reusable Warp WENO5/RK3 helpers.

Reusable initialization, primitive/conservative conversion, WENO functions,
flux functions, and kernels live in warp_weno5_helpers.py.  This file only
allocates arrays, launches the RK3 stages, and prints simple diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import warp_weno5_helpers as wh


wp = wh.wp


if wp is not None:
    MLP_INPUTS = 5
    MLP_HIDDEN1 = 10
    MLP_HIDDEN2 = 6
    MLP_HIDDEN3 = 6
    Vec5d = wp.types.vector(length=MLP_INPUTS, dtype=wp.float64)
    VecH1d = wp.types.vector(length=MLP_HIDDEN1, dtype=wp.float64)
    VecH2d = wp.types.vector(length=MLP_HIDDEN2, dtype=wp.float64)
    VecH3d = wp.types.vector(length=MLP_HIDDEN3, dtype=wp.float64)

    HEAD_NORMAL_LR1 = wp.constant(0)
    HEAD_NORMAL_LR2 = wp.constant(1)
    HEAD_GAUSS_LR1 = wp.constant(2)
    HEAD_GAUSS_LR2 = wp.constant(3)

    @wp.func
    def swish(x: wp.float64) -> wp.float64:
        return x / (wp.float64(1.0) + wp.exp(-x))

    @wp.func
    def evilin_flux(right_state: wp.vec4d, left_state: wp.vec4d, direction: int, dt_over_h: wp.float64, gamma: wp.float64) -> wp.vec4d:
        face_con = wh.evilin_state_2d(right_state, left_state, direction, dt_over_h, gamma)
        return wh.pri_to_flux(wh.con_to_pri(face_con, gamma), direction, gamma)

    @wp.func
    def hllc_flux(ul0: wp.vec4d, ur0: wp.vec4d, direction: int, gamma: wp.float64) -> wp.vec4d:
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
    def riemann_flux(right_state: wp.vec4d, left_state: wp.vec4d, direction: int, dt_over_h: wp.float64, gamma: wp.float64, solver_kind: int) -> wp.vec4d:
        if solver_kind == 1:
            return hllc_flux(right_state, left_state, direction, gamma)
        return evilin_flux(right_state, left_state, direction, dt_over_h, gamma)

    @wp.func
    def double_mach_exact_conserved(x: wp.float64, y: wp.float64, t: wp.float64, gamma: wp.float64) -> wp.vec4d:
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
        return wh.pri_to_con(wp.vec4d(wp.float64(1.4), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)), gamma)

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

    @wp.func
    def weno5_raw_sensors(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64) -> wp.vec4d:
        d20 = q0 - wp.float64(2.0) * q1 + q2
        d21 = q1 - wp.float64(2.0) * q2 + q3
        d22 = q2 - wp.float64(2.0) * q3 + q4
        delta0 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d20) + wp.float64(0.25) * wp.abs(q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2)
        delta1 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d21) + wp.float64(0.25) * wp.abs(q1 - q3)
        delta2 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d22) + wp.float64(0.25) * wp.abs(wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4)
        eps = wp.float64(1.0e-15)
        gamma0 = wp.abs(d20) / (wp.abs(q1 - q0) + wp.abs(q2 - q1) + eps)
        gamma1 = wp.abs(d21) / (wp.abs(q2 - q1) + wp.abs(q3 - q2) + eps)
        gamma2 = wp.abs(d22) / (wp.abs(q3 - q2) + wp.abs(q4 - q3) + eps)
        gamma_s = wp.min(wp.float64(1.0), wp.max(wp.max(gamma0, gamma1), gamma2))
        return wp.vec4d(delta0, delta1, delta2, gamma_s)

    @wp.func
    def weno5_nn_features(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64) -> Vec5d:
        raw = weno5_raw_sensors(q0, q1, q2, q3, q4)
        eps = wp.float64(1.0e-15)
        delta0 = raw[0]
        delta1 = raw[1]
        delta2 = raw[2]
        delta_max = wp.max(wp.max(delta0, delta1), delta2)
        inv_delta_max = wp.float64(1.0) / wp.max(delta_max, eps)
        q_scale = wp.max(wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))), wp.abs(q4))
        q_scale = wp.max(q_scale, wp.float64(1.0))
        relative_scale = wp.max(delta_max / q_scale, wp.float64(1.0e-30))
        log10_relative_scale = wp.log(relative_scale) / wp.log(wp.float64(10.0))
        scale_feature = (log10_relative_scale + wp.float64(16.0)) / wp.float64(16.0)
        scale_feature = wp.min(wp.float64(1.0), wp.max(wp.float64(0.0), scale_feature))
        features = Vec5d()
        features[0] = delta0 * inv_delta_max
        features[1] = delta1 * inv_delta_max
        features[2] = delta2 * inv_delta_max
        features[3] = raw[3]
        features[4] = scale_feature
        return features

    @wp.func
    def plateau_detected(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64) -> int:
        raw = weno5_raw_sensors(q0, q1, q2, q3, q4)
        delta_max = wp.max(wp.max(raw[0], raw[1]), raw[2])
        q_scale = wp.max(wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))), wp.abs(q4))
        q_scale = wp.max(q_scale, wp.float64(1.0))
        if delta_max <= wp.float64(1.0e-13) * q_scale:
            return 1
        return 0

    @wp.func
    def optimal_weights(head: int) -> wp.vec3d:
        root3 = wp.sqrt(wp.float64(3.0))
        if head == HEAD_NORMAL_LR1:
            return wp.vec3d(wp.float64(0.1), wp.float64(0.6), wp.float64(0.3))
        if head == HEAD_NORMAL_LR2:
            return wp.vec3d(wp.float64(0.3), wp.float64(0.6), wp.float64(0.1))
        if head == HEAD_GAUSS_LR1:
            return wp.vec3d(
                (wp.float64(210.0) + root3) / wp.float64(1080.0),
                wp.float64(11.0) / wp.float64(18.0),
                (wp.float64(210.0) - root3) / wp.float64(1080.0),
            )
        return wp.vec3d(
            (wp.float64(210.0) - root3) / wp.float64(1080.0),
            wp.float64(11.0) / wp.float64(18.0),
            (wp.float64(210.0) + root3) / wp.float64(1080.0),
        )

    @wp.func
    def mlp_weights(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        head: int,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        eno_cutoff: int,
    ) -> wp.vec3d:
        c = optimal_weights(head)
        if plateau_detected(q0, q1, q2, q3, q4) == 1:
            return c

        x = weno5_nn_features(q0, q1, q2, q3, q4)
        h1 = VecH1d()
        for o in range(MLP_HIDDEN1):
            z = b1[0, o]
            for k in range(MLP_INPUTS):
                z = z + x[k] * w1[0, k, o]
            h1[o] = swish(z)

        h2 = VecH2d()
        for o in range(MLP_HIDDEN2):
            z = b2[0, o]
            for k in range(MLP_HIDDEN1):
                z = z + h1[k] * w2[0, k, o]
            h2[o] = swish(z)

        h3 = VecH3d()
        for o in range(MLP_HIDDEN3):
            z = b3[0, o]
            for k in range(MLP_HIDDEN2):
                z = z + h2[k] * w3[0, k, o]
            h3[o] = swish(z)

        raw0 = b4[0, 0]
        raw1 = b4[0, 1]
        raw2 = b4[0, 2]
        for k in range(MLP_HIDDEN3):
            raw0 = raw0 + h3[k] * w4[0, k, 0]
            raw1 = raw1 + h3[k] * w4[0, k, 1]
            raw2 = raw2 + h3[k] * w4[0, k, 2]

        cap = wp.float64(6.0)
        bad0 = cap * wp.tanh(raw0 / cap)
        bad1 = cap * wp.tanh(raw1 / cap)
        bad2 = cap * wp.tanh(raw2 / cap)
        bad_max = wp.max(wp.max(bad0, bad1), bad2)
        e0 = wp.exp(bad0 - bad_max)
        e1 = wp.exp(bad1 - bad_max)
        e2 = wp.exp(bad2 - bad_max)
        inv_sum = wp.float64(1.0) / (e0 + e1 + e2)
        beta0 = wp.float64(3.0) * e0 * inv_sum
        beta1 = wp.float64(3.0) * e1 * inv_sum
        beta2 = wp.float64(3.0) * e2 * inv_sum
        eps = wp.float64(1.0e-12)
        ib0 = wp.float64(1.0) / (beta0 + eps)
        ib1 = wp.float64(1.0) / (beta1 + eps)
        ib2 = wp.float64(1.0) / (beta2 + eps)
        a0 = c[0] * ib0 * ib0
        a1 = c[1] * ib1 * ib1
        a2 = c[2] * ib2 * ib2
        inv_a_sum = wp.float64(1.0) / wp.max(a0 + a1 + a2, wp.float64(1.0e-300))
        ww0 = a0 * inv_a_sum
        ww1 = a1 * inv_a_sum
        ww2 = a2 * inv_a_sum

        if eno_cutoff == 1:
            psi0 = wp.float64(1.0)
            psi1 = wp.float64(1.0)
            psi2 = wp.float64(1.0)
            cutoff = wp.float64(1.0e-6)
            if ww0 <= cutoff:
                psi0 = wp.float64(0.0)
            if ww1 <= cutoff:
                psi1 = wp.float64(0.0)
            if ww2 <= cutoff:
                psi2 = wp.float64(0.0)
            cut_sum = psi0 * ww0 + psi1 * ww1 + psi2 * ww2
            inv_cut = wp.float64(1.0) / wp.max(cut_sum, wp.float64(1.0e-300))
            ww0 = psi0 * ww0 * inv_cut
            ww1 = psi1 * ww1 * inv_cut
            ww2 = psi2 * ww2 * inv_cut

        return wp.vec3d(ww0, ww1, ww2)

    @wp.func
    def weno5_lr_value_mlp(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        lr: int,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        eno_cutoff: int,
    ) -> wp.float64:
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        head = HEAD_NORMAL_LR1
        if lr == 1:
            s0 = wh.stencil_2d(q0, q1, q2, 1)
            s1 = wh.stencil_2d(q1, q2, q3, 2)
            s2 = wh.stencil_2d(q2, q3, q4, 3)
            head = HEAD_NORMAL_LR1
        else:
            s0 = wh.stencil_2d(q0, q1, q2, 2)
            s1 = wh.stencil_2d(q1, q2, q3, 3)
            s2 = wh.stencil_2d(q2, q3, q4, 4)
            head = HEAD_NORMAL_LR2
        weights = mlp_weights(q0, q1, q2, q3, q4, head, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
        return weights[0] * s0 + weights[1] * s1 + weights[2] * s2

    @wp.func
    def weno5_gauss_value_mlp(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        loca: int,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        eno_cutoff: int,
    ) -> wp.float64:
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        head = HEAD_GAUSS_LR1
        if loca == 1:
            s0 = wh.stencil_gauss(q0, q1, q2, 1)
            s1 = wh.stencil_gauss(q1, q2, q3, 2)
            s2 = wh.stencil_gauss(q2, q3, q4, 3)
            head = HEAD_GAUSS_LR1
        else:
            s0 = wh.stencil_gauss(q0, q1, q2, 4)
            s1 = wh.stencil_gauss(q1, q2, q3, 5)
            s2 = wh.stencil_gauss(q2, q3, q4, 6)
            head = HEAD_GAUSS_LR2
        weights = mlp_weights(q0, q1, q2, q3, q4, head, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
        return weights[0] * s0 + weights[1] * s1 + weights[2] * s2

    @wp.kernel
    def compute_x_stage_weno_mlp_kernel(
        u: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        nx: int,
        ny: int,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 6 and i < nx + 2:
            for comp in range(4):
                q0 = u[j, i + 0, comp]
                q1 = u[j, i + 1, comp]
                q2 = u[j, i + 2, comp]
                q3 = u[j, i + 3, comp]
                q4 = u[j, i + 4, comp]
                temp_l[j, i, comp] = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
                temp_r[j, i, comp] = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)

    @wp.kernel
    def compute_y_stage_weno_mlp_kernel(
        u: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        nx: int,
        ny: int,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 6:
            for comp in range(4):
                q0 = u[j + 0, i, comp]
                q1 = u[j + 1, i, comp]
                q2 = u[j + 2, i, comp]
                q3 = u[j + 3, i, comp]
                q4 = u[j + 4, i, comp]
                temp_l[j, i, comp] = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
                temp_r[j, i, comp] = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)

    @wp.kernel
    def compute_x_stage_weno_mlp_characteristic_kernel(
        u: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        nx: int,
        ny: int,
        eno_cutoff: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 6 and i < nx + 2:
            q0 = wp.vec4d(u[j, i + 0, 0], u[j, i + 0, 1], u[j, i + 0, 2], u[j, i + 0, 3])
            q1 = wp.vec4d(u[j, i + 1, 0], u[j, i + 1, 1], u[j, i + 1, 2], u[j, i + 1, 3])
            q2 = wp.vec4d(u[j, i + 2, 0], u[j, i + 2, 1], u[j, i + 2, 2], u[j, i + 2, 3])
            q3 = wp.vec4d(u[j, i + 3, 0], u[j, i + 3, 1], u[j, i + 3, 2], u[j, i + 3, 3])
            q4 = wp.vec4d(u[j, i + 4, 0], u[j, i + 4, 1], u[j, i + 4, 2], u[j, i + 4, 3])

            roe_l = wh.roe_average_state(q1, q2, gamma)
            c0_l = wh.con_to_char(q0, roe_l, 1, gamma)
            c1_l = wh.con_to_char(q1, roe_l, 1, gamma)
            c2_l = wh.con_to_char(q2, roe_l, 1, gamma)
            c3_l = wh.con_to_char(q3, roe_l, 1, gamma)
            c4_l = wh.con_to_char(q4, roe_l, 1, gamma)
            ql = wh.char_to_con(
                wp.vec4d(
                    weno5_lr_value_mlp(c0_l[0], c1_l[0], c2_l[0], c3_l[0], c4_l[0], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_l[1], c1_l[1], c2_l[1], c3_l[1], c4_l[1], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_l[2], c1_l[2], c2_l[2], c3_l[2], c4_l[2], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_l[3], c1_l[3], c2_l[3], c3_l[3], c4_l[3], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                ),
                roe_l,
                1,
                gamma,
            )

            roe_r = wh.roe_average_state(q2, q3, gamma)
            c0_r = wh.con_to_char(q0, roe_r, 1, gamma)
            c1_r = wh.con_to_char(q1, roe_r, 1, gamma)
            c2_r = wh.con_to_char(q2, roe_r, 1, gamma)
            c3_r = wh.con_to_char(q3, roe_r, 1, gamma)
            c4_r = wh.con_to_char(q4, roe_r, 1, gamma)
            qr = wh.char_to_con(
                wp.vec4d(
                    weno5_lr_value_mlp(c0_r[0], c1_r[0], c2_r[0], c3_r[0], c4_r[0], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_r[1], c1_r[1], c2_r[1], c3_r[1], c4_r[1], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_r[2], c1_r[2], c2_r[2], c3_r[2], c4_r[2], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_r[3], c1_r[3], c2_r[3], c3_r[3], c4_r[3], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                ),
                roe_r,
                1,
                gamma,
            )
            for comp in range(4):
                temp_l[j, i, comp] = ql[comp]
                temp_r[j, i, comp] = qr[comp]

    @wp.kernel
    def compute_y_stage_weno_mlp_characteristic_kernel(
        u: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        nx: int,
        ny: int,
        eno_cutoff: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 6:
            q0 = wp.vec4d(u[j + 0, i, 0], u[j + 0, i, 1], u[j + 0, i, 2], u[j + 0, i, 3])
            q1 = wp.vec4d(u[j + 1, i, 0], u[j + 1, i, 1], u[j + 1, i, 2], u[j + 1, i, 3])
            q2 = wp.vec4d(u[j + 2, i, 0], u[j + 2, i, 1], u[j + 2, i, 2], u[j + 2, i, 3])
            q3 = wp.vec4d(u[j + 3, i, 0], u[j + 3, i, 1], u[j + 3, i, 2], u[j + 3, i, 3])
            q4 = wp.vec4d(u[j + 4, i, 0], u[j + 4, i, 1], u[j + 4, i, 2], u[j + 4, i, 3])

            roe_l = wh.roe_average_state(q1, q2, gamma)
            c0_l = wh.con_to_char(q0, roe_l, 2, gamma)
            c1_l = wh.con_to_char(q1, roe_l, 2, gamma)
            c2_l = wh.con_to_char(q2, roe_l, 2, gamma)
            c3_l = wh.con_to_char(q3, roe_l, 2, gamma)
            c4_l = wh.con_to_char(q4, roe_l, 2, gamma)
            ql = wh.char_to_con(
                wp.vec4d(
                    weno5_lr_value_mlp(c0_l[0], c1_l[0], c2_l[0], c3_l[0], c4_l[0], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_l[1], c1_l[1], c2_l[1], c3_l[1], c4_l[1], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_l[2], c1_l[2], c2_l[2], c3_l[2], c4_l[2], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_l[3], c1_l[3], c2_l[3], c3_l[3], c4_l[3], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                ),
                roe_l,
                2,
                gamma,
            )

            roe_r = wh.roe_average_state(q2, q3, gamma)
            c0_r = wh.con_to_char(q0, roe_r, 2, gamma)
            c1_r = wh.con_to_char(q1, roe_r, 2, gamma)
            c2_r = wh.con_to_char(q2, roe_r, 2, gamma)
            c3_r = wh.con_to_char(q3, roe_r, 2, gamma)
            c4_r = wh.con_to_char(q4, roe_r, 2, gamma)
            qr = wh.char_to_con(
                wp.vec4d(
                    weno5_lr_value_mlp(c0_r[0], c1_r[0], c2_r[0], c3_r[0], c4_r[0], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_r[1], c1_r[1], c2_r[1], c3_r[1], c4_r[1], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_r[2], c1_r[2], c2_r[2], c3_r[2], c4_r[2], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                    weno5_lr_value_mlp(c0_r[3], c1_r[3], c2_r[3], c3_r[3], c4_r[3], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff),
                ),
                roe_r,
                2,
                gamma,
            )
            for comp in range(4):
                temp_l[j, i, comp] = ql[comp]
                temp_r[j, i, comp] = qr[comp]

    @wp.kernel
    def compute_x_point_weno_mlp_kernel(
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        nx: int,
        ny: int,
        loca: int,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 2:
            for comp in range(4):
                point_l[j, i, comp] = weno5_gauss_value_mlp(
                    temp_l[j + 1, i, comp],
                    temp_l[j + 2, i, comp],
                    temp_l[j + 3, i, comp],
                    temp_l[j + 4, i, comp],
                    temp_l[j + 5, i, comp],
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
                point_r[j, i, comp] = weno5_gauss_value_mlp(
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

    @wp.kernel
    def compute_y_point_weno_mlp_kernel(
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
        nx: int,
        ny: int,
        loca: int,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx:
            for comp in range(4):
                point_l[j, i, comp] = weno5_gauss_value_mlp(
                    temp_l[j, i + 1, comp],
                    temp_l[j, i + 2, comp],
                    temp_l[j, i + 3, comp],
                    temp_l[j, i + 4, comp],
                    temp_l[j, i + 5, comp],
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
                point_r[j, i, comp] = weno5_gauss_value_mlp(
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

    @wp.kernel
    def compute_x_flux_evilin_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        tempdx_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        solver_kind: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            left_state = wp.vec4d(point_l[j, i + 1, 0], point_l[j, i + 1, 1], point_l[j, i + 1, 2], point_l[j, i + 1, 3])
            right_state = wp.vec4d(point_r[j, i, 0], point_r[j, i, 1], point_r[j, i, 2], point_r[j, i, 3])
            f = riemann_flux(right_state, left_state, 1, tempdx_dt, gamma, solver_kind)
            if loca == 1:
                for comp in range(4):
                    flux_x[j, i, comp] = f[comp] * tempdx_dt
            else:
                for comp in range(4):
                    flux_x[j, i, comp] = flux_x[j, i, comp] + f[comp] * tempdx_dt

    @wp.kernel
    def compute_y_flux_evilin_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        point_l: wp.array3d(dtype=wp.float64),
        point_r: wp.array3d(dtype=wp.float64),
        tempdy_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        solver_kind: int,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            left_state = wp.vec4d(point_l[j + 1, i, 0], point_l[j + 1, i, 1], point_l[j + 1, i, 2], point_l[j + 1, i, 3])
            right_state = wp.vec4d(point_r[j, i, 0], point_r[j, i, 1], point_r[j, i, 2], point_r[j, i, 3])
            f = riemann_flux(right_state, left_state, 2, tempdy_dt, gamma, solver_kind)
            if loca == 1:
                for comp in range(4):
                    flux_y[j, i, comp] = f[comp] * tempdy_dt
            else:
                for comp in range(4):
                    flux_y[j, i, comp] = flux_y[j, i, comp] + f[comp] * tempdy_dt

def launch_weno_rk3_step(
    arrays: dict[str, object],
    params: wh.Params,
    dt: float,
    device: str,
    characteristic_weno: bool,
    boundary: str,
    mlp_params: dict[str, object] | None = None,
    eno_cutoff: bool = False,
    riemann_solver: str = "evilin",
    boundary_time: float = 0.0,
) -> None:
    wp = wh.wp
    if riemann_solver not in ("evilin", "hllc"):
        raise ValueError("weno5_rk3_warp.py formal path supports --riemann-solver evilin or hllc")
    nx = params.nx
    ny = params.ny
    gc = params.ghost
    nx_total = nx + 2 * gc
    ny_total = ny + 2 * gc
    tempdx_dt = dt / params.dx
    tempdy_dt = dt / params.dy
    characteristic = 1 if characteristic_weno else 0
    eno_cutoff_i = 1 if eno_cutoff else 0
    solver_kind = 1 if riemann_solver == "hllc" else 0

    for rk in (1, 2, 3):
        if boundary == "double-mach":
            stage_time = boundary_time
            if rk == 2:
                stage_time = boundary_time + dt
            elif rk == 3:
                stage_time = boundary_time + 0.5 * dt
            wp.launch(
                apply_double_mach_boundary_kernel,
                dim=(ny_total, nx_total),
                inputs=[
                    arrays["u"],
                    nx,
                    ny,
                    gc,
                    wp.float64(params.dx),
                    wp.float64(params.dy),
                    wp.float64(stage_time),
                    wp.float64(params.gamma),
                ],
                device=device,
            )
        else:
            boundary_kernel = wh.apply_periodic_boundary_kernel if boundary == "periodic" else wh.apply_boundary_kernel
            wp.launch(boundary_kernel, dim=(ny_total, nx_total), inputs=[arrays["u"], nx, ny, gc], device=device)

        if mlp_params is None:
            wp.launch(
                wh.compute_x_stage_weno_kernel,
                dim=(ny + 6, nx + 2),
                inputs=[
                    arrays["u"],
                    arrays["temp_l"],
                    arrays["temp_r"],
                    nx,
                    ny,
                    characteristic,
                    wp.float64(params.gamma),
                ],
                device=device,
            )
        elif characteristic_weno:
            wp.launch(
                compute_x_stage_weno_mlp_characteristic_kernel,
                dim=(ny + 6, nx + 2),
                inputs=[
                    arrays["u"],
                    arrays["temp_l"],
                    arrays["temp_r"],
                    mlp_params["w1"],
                    mlp_params["b1"],
                    mlp_params["w2"],
                    mlp_params["b2"],
                    mlp_params["w3"],
                    mlp_params["b3"],
                    mlp_params["w4"],
                    mlp_params["b4"],
                    nx,
                    ny,
                    eno_cutoff_i,
                    wp.float64(params.gamma),
                ],
                device=device,
            )
        else:
            wp.launch(
                compute_x_stage_weno_mlp_kernel,
                dim=(ny + 6, nx + 2),
                inputs=[
                    arrays["u"],
                    arrays["temp_l"],
                    arrays["temp_r"],
                    mlp_params["w1"],
                    mlp_params["b1"],
                    mlp_params["w2"],
                    mlp_params["b2"],
                    mlp_params["w3"],
                    mlp_params["b3"],
                    mlp_params["w4"],
                    mlp_params["b4"],
                    nx,
                    ny,
                    eno_cutoff_i,
                ],
                device=device,
            )
        for loca in (1, 2):
            if mlp_params is None:
                wp.launch(
                    wh.compute_x_point_weno_kernel,
                    dim=(ny, nx + 2),
                    inputs=[arrays["temp_l"], arrays["temp_r"], arrays["point_l"], arrays["point_r"], nx, ny, loca],
                    device=device,
                )
            else:
                wp.launch(
                    compute_x_point_weno_mlp_kernel,
                    dim=(ny, nx + 2),
                    inputs=[
                        arrays["temp_l"],
                        arrays["temp_r"],
                        arrays["point_l"],
                        arrays["point_r"],
                        mlp_params["w1"],
                        mlp_params["b1"],
                        mlp_params["w2"],
                        mlp_params["b2"],
                        mlp_params["w3"],
                        mlp_params["b3"],
                        mlp_params["w4"],
                        mlp_params["b4"],
                        nx,
                        ny,
                        loca,
                        eno_cutoff_i,
                    ],
                    device=device,
                )
            wp.launch(
                compute_x_flux_evilin_kernel,
                dim=(ny, nx + 1),
                inputs=[
                    arrays["flux_x"],
                    arrays["point_l"],
                    arrays["point_r"],
                    wp.float64(tempdx_dt),
                    nx,
                    ny,
                    loca,
                    wp.float64(params.gamma),
                    solver_kind,
                ],
                device=device,
            )

        if mlp_params is None:
            wp.launch(
                wh.compute_y_stage_weno_kernel,
                dim=(ny + 2, nx + 6),
                inputs=[
                    arrays["u"],
                    arrays["temp_l"],
                    arrays["temp_r"],
                    nx,
                    ny,
                    characteristic,
                    wp.float64(params.gamma),
                ],
                device=device,
            )
        elif characteristic_weno:
            wp.launch(
                compute_y_stage_weno_mlp_characteristic_kernel,
                dim=(ny + 2, nx + 6),
                inputs=[
                    arrays["u"],
                    arrays["temp_l"],
                    arrays["temp_r"],
                    mlp_params["w1"],
                    mlp_params["b1"],
                    mlp_params["w2"],
                    mlp_params["b2"],
                    mlp_params["w3"],
                    mlp_params["b3"],
                    mlp_params["w4"],
                    mlp_params["b4"],
                    nx,
                    ny,
                    eno_cutoff_i,
                    wp.float64(params.gamma),
                ],
                device=device,
            )
        else:
            wp.launch(
                compute_y_stage_weno_mlp_kernel,
                dim=(ny + 2, nx + 6),
                inputs=[
                    arrays["u"],
                    arrays["temp_l"],
                    arrays["temp_r"],
                    mlp_params["w1"],
                    mlp_params["b1"],
                    mlp_params["w2"],
                    mlp_params["b2"],
                    mlp_params["w3"],
                    mlp_params["b3"],
                    mlp_params["w4"],
                    mlp_params["b4"],
                    nx,
                    ny,
                    eno_cutoff_i,
                ],
                device=device,
            )
        for loca in (1, 2):
            if mlp_params is None:
                wp.launch(
                    wh.compute_y_point_weno_kernel,
                    dim=(ny + 2, nx),
                    inputs=[arrays["temp_l"], arrays["temp_r"], arrays["point_l"], arrays["point_r"], nx, ny, loca],
                    device=device,
                )
            else:
                wp.launch(
                    compute_y_point_weno_mlp_kernel,
                    dim=(ny + 2, nx),
                    inputs=[
                        arrays["temp_l"],
                        arrays["temp_r"],
                        arrays["point_l"],
                        arrays["point_r"],
                        mlp_params["w1"],
                        mlp_params["b1"],
                        mlp_params["w2"],
                        mlp_params["b2"],
                        mlp_params["w3"],
                        mlp_params["b3"],
                        mlp_params["w4"],
                        mlp_params["b4"],
                        nx,
                        ny,
                        loca,
                        eno_cutoff_i,
                    ],
                    device=device,
                )
            wp.launch(
                compute_y_flux_evilin_kernel,
                dim=(ny + 1, nx),
                inputs=[
                    arrays["flux_y"],
                    arrays["point_l"],
                    arrays["point_r"],
                    wp.float64(tempdy_dt),
                    nx,
                    ny,
                    loca,
                    wp.float64(params.gamma),
                    solver_kind,
                ],
                device=device,
            )

        wp.launch(
            wh.update_rk3_kernel,
            dim=(ny, nx),
            inputs=[
                arrays["u"],
                arrays["u0"],
                arrays["flux_x"],
                arrays["flux_y"],
                arrays["pri"],
                nx,
                ny,
                gc,
                rk,
                wp.float64(params.gamma),
            ],
            device=device,
        )

    wp.synchronize()


def allocate_warp_arrays(u0_host: np.ndarray, params: wh.Params, device: str) -> dict[str, object]:
    wp = wh.wp
    shape = params.padded_shape
    return {
        "u": wp.array(u0_host, dtype=wp.float64, device=device),
        "u0": wp.zeros(shape, dtype=wp.float64, device=device),
        "pri": wp.zeros(shape, dtype=wp.float64, device=device),
        "temp_l": wp.zeros(shape, dtype=wp.float64, device=device),
        "temp_r": wp.zeros(shape, dtype=wp.float64, device=device),
        "point_l": wp.zeros(shape, dtype=wp.float64, device=device),
        "point_r": wp.zeros(shape, dtype=wp.float64, device=device),
        "flux_x": wp.zeros(shape, dtype=wp.float64, device=device),
        "flux_y": wp.zeros(shape, dtype=wp.float64, device=device),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
    }


def read_cuda_csv_fields(path: str | Path, params: wh.Params) -> tuple[float | None, np.ndarray]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8") as file:
        header = file.readline().strip()
        rows = [line.strip() for line in file if line.strip()]

    time = None
    if header.startswith("# Time:"):
        time = float(header.split(":", 1)[1].strip())

    ny = len(rows)
    data = np.fromstring("\n".join(rows), sep=",", dtype=np.float64)
    if ny == 0 or data.size % (ny * 4) != 0:
        raise ValueError(f"{csv_path} has malformed CSV dimensions")

    nx = data.size // (ny * 4)
    fields = data.reshape(ny, nx, 4)
    if (ny, nx) != (params.ny, params.nx):
        raise ValueError(
            f"{csv_path} shape is {(ny, nx)}, but Warp params expect {(params.ny, params.nx)}"
        )

    return time, fields


def density_comparison(warp_u: np.ndarray, csv_path: str | Path, params: wh.Params) -> dict[str, float]:
    csv_time, cuda_pri = read_cuda_csv_fields(csv_path, params)
    gc = params.ghost
    warp_rho = warp_u[gc : gc + params.ny, gc : gc + params.nx, 0]
    cuda_rho = cuda_pri[..., 0]
    diff = warp_rho - cuda_rho
    abs_diff = np.abs(diff)
    max_flat = int(np.argmax(abs_diff))
    max_j, max_i = np.unravel_index(max_flat, abs_diff.shape)

    eps = 1.0e-300
    return {
        "csv_time": float("nan") if csv_time is None else csv_time,
        "mean_abs": float(np.mean(abs_diff)),
        "rms_abs": float(np.sqrt(np.mean(diff * diff))),
        "linf_abs": float(np.max(abs_diff)),
        "rel_l1": float(np.sum(abs_diff) / max(np.sum(np.abs(cuda_rho)), eps)),
        "rel_l2": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(cuda_rho.ravel()), eps)),
        "rel_linf": float(np.max(abs_diff) / max(np.max(np.abs(cuda_rho)), eps)),
        "max_i": float(max_i),
        "max_j": float(max_j),
        "max_x": float((max_i + 0.5) * params.dx),
        "max_y": float((max_j + 0.5) * params.dy),
        "warp_rho_at_max": float(warp_rho[max_j, max_i]),
        "cuda_rho_at_max": float(cuda_rho[max_j, max_i]),
    }


def print_density_comparison(metrics: dict[str, float], csv_path: str | Path) -> None:
    print(f"compare_csv={csv_path}")
    if not np.isnan(metrics["csv_time"]):
        print(f"csv_time={metrics['csv_time']:.16e}")
    print(
        "density_diff: "
        f"mean_abs={metrics['mean_abs']:.16e} "
        f"rms_abs={metrics['rms_abs']:.16e} "
        f"linf_abs={metrics['linf_abs']:.16e}"
    )
    print(
        "density_rel: "
        f"rel_l1={metrics['rel_l1']:.16e} "
        f"rel_l2={metrics['rel_l2']:.16e} "
        f"rel_linf={metrics['rel_linf']:.16e}"
    )
    print(
        "density_max_location: "
        f"i={int(metrics['max_i'])} j={int(metrics['max_j'])} "
        f"x={metrics['max_x']:.16e} y={metrics['max_y']:.16e} "
        f"warp_rho={metrics['warp_rho_at_max']:.16e} "
        f"cuda_rho={metrics['cuda_rho_at_max']:.16e}"
    )


def load_mlp_params(model_path: str | Path, device: str) -> dict[str, object]:
    data = np.load(model_path, allow_pickle=True)
    required = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"Model {model_path} is missing arrays: {missing}")
    expected_shapes = {
        "w1": (1, 5, 10),
        "b1": (1, 10),
        "w2": (1, 10, 6),
        "b2": (1, 6),
        "w3": (1, 6, 6),
        "b3": (1, 6),
        "w4": (1, 6, 3),
        "b4": (1, 3),
    }
    wrong = {name: data[name].shape for name, shape in expected_shapes.items() if data[name].shape != shape}
    if wrong:
        raise ValueError(
            f"Model {model_path} uses incompatible MLP shapes: {wrong}. "
            "weno5_rk3_warp.py formal MLP path only supports 5->10->6->6->3 checkpoints."
        )
    if "meta_json" in data.files:
        meta = str(data["meta_json"])
        if "shared_direct_beta_ratio_5_10_6_6_3" not in meta:
            print("warning: model metadata does not mention shared_direct_beta_ratio_5_10_6_6_3")
    return {name: wp.array(data[name], dtype=wp.float64, device=device, requires_grad=False) for name in required}


def primitive_fields(u: np.ndarray, params: wh.Params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = params.ghost
    pri = wh.primitive_from_conserved(u[g : g + params.ny, g : g + params.nx, :], params.gamma)
    return pri[..., 0], pri[..., 1], pri[..., 2], pri[..., 3]


def make_physical_grid(params: wh.Params) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(params.nx, dtype=np.float64) + 0.5) * params.dx
    y = (np.arange(params.ny, dtype=np.float64) + 0.5) * params.dy
    return np.meshgrid(x, y)


def plot_pressure_density(
    u: np.ndarray,
    params: wh.Params,
    out_path: Path,
    title: str,
    p_min: float | None = None,
    p_max: float | None = None,
) -> None:
    rho, vx, vy, pressure = primitive_fields(u, params)
    x, y = make_physical_grid(params)
    if p_min is None:
        p_min = float(np.min(pressure))
    if p_max is None:
        p_max = float(np.max(pressure))
    rho_min = float(np.min(rho))
    rho_max = float(np.max(rho))
    if rho_max > rho_min:
        rho_levels = np.linspace(rho_min, rho_max, 32)
    else:
        rho_levels = np.array([rho_min], dtype=np.float64)
    skip = max(1, params.nx // 30)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    bg = ax.contourf(x, y, pressure, levels=300, cmap="jet", vmin=p_min, vmax=p_max)
    if rho_levels.size > 1:
        ax.contour(x, y, rho, levels=rho_levels, colors="k", linewidths=0.25)
    ax.quiver(
        x[::skip, ::skip],
        y[::skip, ::skip],
        vx[::skip, ::skip],
        vy[::skip, ::skip],
        color="white",
        scale=600.0,
        width=0.0012,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim([0.0, params.x_length])
    ax.set_ylim([0.0, params.y_length])
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(bg, ax=ax, label="Pressure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_minus_classical(classical: np.ndarray, mlp: np.ndarray, params: wh.Params, out_path: Path) -> None:
    rho_c, _, _, p_c = primitive_fields(classical, params)
    rho_m, _, _, p_m = primitive_fields(mlp, params)
    x, y = make_physical_grid(params)
    drho = rho_m - rho_c
    dp = p_m - p_c
    lim_rho = max(float(np.max(np.abs(drho))), 1.0e-15)
    lim_p = max(float(np.max(np.abs(dp))), 1.0e-15)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    im0 = axes[0].contourf(x, y, drho, levels=240, cmap="coolwarm", vmin=-lim_rho, vmax=lim_rho)
    axes[0].set_title("MLP - classical density")
    im1 = axes[1].contourf(x, y, dp, levels=240, cmap="coolwarm", vmin=-lim_p, vmax=lim_p)
    axes[1].set_title("MLP - classical pressure")
    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim([0.0, params.x_length])
        ax.set_ylim([0.0, params.y_length])
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im0, ax=axes[0], shrink=0.86)
    fig.colorbar(im1, ax=axes[1], shrink=0.86)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_solution(
    initial: np.ndarray,
    params: wh.Params,
    args: argparse.Namespace,
    mlp_params: dict[str, object] | None,
    label: str,
) -> tuple[np.ndarray, float, int, list[float]]:
    arrays = allocate_warp_arrays(initial, params, args.device)
    t = 0.0
    dt_values: list[float] = []
    for step in range(1, args.steps + 1):
        if args.dt_mode == "warp":
            dt = wh.compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, args.device)
        else:
            dt = wh.compute_dt(arrays["u"].numpy(), params)
        if args.t_end > 0.0 and t + dt > args.t_end:
            dt = args.t_end - t
        if dt <= 0.0:
            break

        launch_weno_rk3_step(
            arrays,
            params,
            dt,
            args.device,
            args.weno_space == "characteristic",
            args.boundary,
            mlp_params,
            args.eno_cutoff if mlp_params is not None else False,
            args.riemann_solver,
            t,
        )
        t += dt
        dt_values.append(dt)

        reached_t_end = args.t_end > 0.0 and t >= args.t_end - 1.0e-14
        should_report = (
            step == 1
            or step == args.steps
            or reached_t_end
            or (args.report_interval > 0 and step % args.report_interval == 0)
        )
        if should_report:
            host = arrays["u"].numpy()
            stats = wh.interior_stats(host, params)
            print(
                f"{label} step={step:04d} t={t:.16e} dt={dt:.16e} "
                f"mass={stats['mass']:.16e} "
                f"rho_min={stats['rho_min']:.6e} p_min={stats['p_min']:.6e} "
                f"nan={int(stats['nan_count'])} rho_neg={int(stats['rho_neg'])} p_neg={int(stats['p_neg'])}",
                flush=True,
            )
            if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
                break

        if reached_t_end:
            break

    return arrays["u"].numpy(), t, len(dt_values), dt_values


def save_outputs(
    out_dir: Path,
    params: wh.Params,
    initial: np.ndarray,
    classical: np.ndarray | None,
    mlp: np.ndarray | None,
    t: float,
    dt_values: list[float],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    active = mlp if mlp is not None else classical
    if active is None:
        return
    summary = {
        "model": "" if args.model is None else str(args.model),
        "riemann_solver": args.riemann_solver,
        "eno_cutoff": bool(args.eno_cutoff and mlp is not None),
        "nx": params.nx,
        "ny": params.ny,
        "x_length": params.x_length,
        "y_length": params.y_length,
        "cfl": params.cfl,
        "t": t,
        "steps": len(dt_values),
        "dt_min": float(np.min(dt_values)) if dt_values else 0.0,
        "dt_max": float(np.max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
        **wh.interior_stats(active, params),
    }
    payload = {"initial": initial, "t": np.array(t), "dt_values": np.array(dt_values)}
    if classical is not None:
        payload["classical"] = classical
    if mlp is not None:
        payload["mlp"] = mlp
    np.savez(out_dir / "shockbubble_results.npz", **payload)
    rho, vx, vy, pressure = primitive_fields(active, params)
    np.savez(out_dir / "primitive_fields_active.npz", rho=rho, vx=vx, vy=vy, p=pressure)
    with (out_dir / "summary.txt").open("w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    all_pressures = []
    if classical is not None:
        all_pressures.append(primitive_fields(classical, params)[3])
    if mlp is not None:
        all_pressures.append(primitive_fields(mlp, params)[3])
    p_min = float(min(np.min(p) for p in all_pressures))
    p_max = float(max(np.max(p) for p in all_pressures))
    if classical is not None:
        plot_pressure_density(classical, params, out_dir / "classical_pressure_density.png", f"classical WENO5-RK3 EVILIN, t={t:.3e}", p_min, p_max)
    if mlp is not None:
        plot_pressure_density(mlp, params, out_dir / "mlp_pressure_density.png", f"MLP WENO5-RK3 EVILIN, t={t:.3e}", p_min, p_max)
    if classical is not None and mlp is not None:
        plot_mlp_minus_classical(classical, mlp, params, out_dir / "mlp_minus_classical_density_pressure.png")
    print(f"saved_results={out_dir / 'shockbubble_results.npz'}")
    print(f"saved_summary={out_dir / 'summary.txt'}")


def run_demo(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp = wh.wp
    wp.init()

    params = wh.Params(nx=args.nx, ny=args.ny, cfl=args.cfl, t_end=args.t_end)
    device = args.device
    wp.set_device(device)

    u0_host = wh.make_initial_state(params)
    initial_stats = wh.interior_stats(u0_host, params)
    print(
        f"start: mode={'mlp' if args.model else 'classical'} weno_space={args.weno_space} "
        f"boundary={args.boundary} riemann_solver={args.riemann_solver} dt_mode={args.dt_mode} "
        f"mass={initial_stats['mass']:.16e}, "
        f"rho=[{initial_stats['rho_min']:.6e},{initial_stats['rho_max']:.6e}], "
        f"p=[{initial_stats['p_min']:.6e},{initial_stats['p_max']:.6e}]"
    )

    mlp_params = load_mlp_params(args.model, device) if args.model else None
    classical = None
    mlp = None
    t = 0.0
    dt_values: list[float] = []

    if mlp_params is None:
        classical, t, _, dt_values = run_solution(u0_host, params, args, None, "classical")
        active = classical
    else:
        classical, t_c, _, _ = run_solution(u0_host, params, args, None, "classical_ref")
        mlp, t, _, dt_values = run_solution(u0_host, params, args, mlp_params, "mlp")
        if abs(t - t_c) > 1.0e-12:
            print(f"warning: classical_ref reached t={t_c:.16e}, mlp reached t={t:.16e}")
        active = mlp

    if args.save_npy:
        np.save(args.save_npy, active)
        print(f"saved {args.save_npy}")

    if args.compare_csv:
        metrics = density_comparison(active, args.compare_csv, params)
        print_density_comparison(metrics, args.compare_csv)

    if args.out_dir:
        save_outputs(Path(args.out_dir), params, u0_host, classical, mlp, t, dt_values, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D WENO5 + RK3 prototype in NVIDIA Warp.")
    parser.add_argument("--nx", type=int, default=450)
    parser.add_argument("--ny", type=int, default=178)
    parser.add_argument("--cfl", type=float, default=0.228)
    parser.add_argument("--steps", type=int, default=1_000_000, help="maximum number of RK3 time steps")
    parser.add_argument("--t-end", type=float, default=0.0002, help="optional final time; 0 disables this stop")
    parser.add_argument("--device", default="cuda", help="Warp device, e.g. cuda or cpu")
    parser.add_argument("--model", type=Path, default=None, help="optional 5->10->6->6->3 MLP checkpoint; omitted means classical WENO5")
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=False, help="apply MLP ENO cutoff after neural weights")
    parser.add_argument("--out-dir", type=Path, default=None, help="optional directory for npz summary and shock-bubble plots")
    parser.add_argument("--save-npy", default="", help="optional output path for the final conserved array")
    parser.add_argument("--compare-csv", default="", help="optional CUDA primitive CSV for density comparison")
    parser.add_argument("--report-interval", type=int, default=100, help="print diagnostics every N steps; 0 prints first and final only")
    parser.add_argument(
        "--riemann-solver",
        choices=("evilin", "hllc"),
        default="evilin",
        help="Riemann solver for the formal Warp path",
    )
    parser.add_argument(
        "--dt-mode",
        choices=("warp", "host"),
        default="warp",
        help="compute CFL dt with a Warp kernel, or copy U to host and use NumPy",
    )
    parser.add_argument(
        "--weno-space",
        choices=("characteristic", "conserved"),
        default="characteristic",
        help="normal-direction WENO space for the first x/y reconstruction",
    )
    parser.add_argument(
        "--boundary",
        choices=("outflow", "periodic", "double-mach"),
        default="outflow",
        help="ghost-cell boundary condition",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
