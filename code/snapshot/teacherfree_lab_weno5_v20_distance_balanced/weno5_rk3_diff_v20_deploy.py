#!/usr/bin/env python3
"""WENO5/RK3 deployment path for reflection-symmetric V20 inference.

The numerical kernels are copied from the validated reflection-symmetric V12
path. V20 constructs the legacy scale feature in ``weno5_core`` and then
applies ``V9BadnessMLP.remap_scale_feature`` before the shared MLP. The
effective feature seen by the learned parameters is therefore
``clip((log10(relative_scale) + 4) / 4, 0, 1)``. Deployment evaluates that
effective feature directly.
"""

from __future__ import annotations

import argparse

import numpy as np

import warp_weno5_helpers as wh
from run_weno5_smooth_periodic import make_smooth_periodic_state


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
    def evilin_flux(
        right_state: wp.vec4d,
        left_state: wp.vec4d,
        direction: int,
        dt_over_h: wp.float64,
        gamma: wp.float64,
    ) -> wp.vec4d:
        face_con = wh.evilin_state_2d(right_state, left_state, direction, dt_over_h, gamma)
        return wh.pri_to_flux(wh.con_to_pri(face_con, gamma), direction, gamma)


    @wp.func
    def smooth_abs_loss(x: wp.float64) -> wp.float64:
        return wp.sqrt(x * x + wp.float64(1.0e-24))


    @wp.func
    def smooth_pos_loss(x: wp.float64) -> wp.float64:
        return wp.float64(0.5) * (x + wp.sqrt(x * x + wp.float64(1.0e-24)))


    @wp.func
    def stable_lp_root_loss(x: wp.float64, lp_order: wp.float64) -> wp.float64:
        x_pos = wp.max(x, wp.float64(0.0))
        eps = wp.float64(1.0e-6)
        inv_order = wp.float64(1.0) / lp_order
        return wp.pow(x_pos + eps, inv_order) - wp.pow(eps, inv_order)


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
        gamma_s = raw[3]
        inv_delta_max = wp.float64(1.0) / wp.max(delta_max, eps)

        q_scale = wp.max(wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))), wp.abs(q4))
        q_scale = wp.max(q_scale, wp.float64(1.0))
        relative_scale = wp.max(delta_max / q_scale, wp.float64(1.0e-30))
        log10_relative_scale = wp.log(relative_scale) / wp.log(wp.float64(10.0))
        scale_feature = (log10_relative_scale + wp.float64(4.0)) / wp.float64(4.0)
        scale_feature = wp.min(wp.float64(1.0), wp.max(wp.float64(0.0), scale_feature))

        features = Vec5d()
        features[0] = delta0 * inv_delta_max
        features[1] = delta1 * inv_delta_max
        features[2] = delta2 * inv_delta_max
        features[3] = gamma_s
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
    def mlp_badness_ratios(
        x: Vec5d,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
    ) -> wp.vec3d:
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

        raw_badness0 = b4[0, 0]
        raw_badness1 = b4[0, 1]
        raw_badness2 = b4[0, 2]
        for k in range(MLP_HIDDEN3):
            raw_badness0 = raw_badness0 + h3[k] * w4[0, k, 0]
            raw_badness1 = raw_badness1 + h3[k] * w4[0, k, 1]
            raw_badness2 = raw_badness2 + h3[k] * w4[0, k, 2]

        badness_cmax = wp.float64(6.0)
        badness0 = badness_cmax * wp.tanh(raw_badness0 / badness_cmax)
        badness1 = badness_cmax * wp.tanh(raw_badness1 / badness_cmax)
        badness2 = badness_cmax * wp.tanh(raw_badness2 / badness_cmax)

        badness_max = wp.max(wp.max(badness0, badness1), badness2)
        e0 = wp.exp(badness0 - badness_max)
        e1 = wp.exp(badness1 - badness_max)
        e2 = wp.exp(badness2 - badness_max)
        inv_sum = wp.float64(1.0) / (e0 + e1 + e2)
        return wp.vec3d(e0 * inv_sum, e1 * inv_sum, e2 * inv_sum)


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

        direct = mlp_badness_ratios(
            weno5_nn_features(q0, q1, q2, q3, q4),
            w1, b1, w2, b2, w3, b3, w4, b4,
        )
        reflected_raw = mlp_badness_ratios(
            weno5_nn_features(q4, q3, q2, q1, q0),
            w1, b1, w2, b2, w3, b3, w4, b4,
        )
        beta_ratio0 = wp.float64(0.5) * (direct[0] + reflected_raw[2])
        beta_ratio1 = wp.float64(0.5) * (direct[1] + reflected_raw[1])
        beta_ratio2 = wp.float64(0.5) * (direct[2] + reflected_raw[0])

        eps_ratio = wp.float64(1.0e-12)
        badness_ratio_scale = wp.float64(3.0)
        scaled_beta0 = badness_ratio_scale * beta_ratio0
        scaled_beta1 = badness_ratio_scale * beta_ratio1
        scaled_beta2 = badness_ratio_scale * beta_ratio2
        inv_beta0 = wp.float64(1.0) / (scaled_beta0 + eps_ratio)
        inv_beta1 = wp.float64(1.0) / (scaled_beta1 + eps_ratio)
        inv_beta2 = wp.float64(1.0) / (scaled_beta2 + eps_ratio)
        inv_beta0_sq = inv_beta0 * inv_beta0
        inv_beta1_sq = inv_beta1 * inv_beta1
        inv_beta2_sq = inv_beta2 * inv_beta2
        alpha0 = c[0] * inv_beta0_sq
        alpha1 = c[1] * inv_beta1_sq
        alpha2 = c[2] * inv_beta2_sq
        alpha_sum = alpha0 + alpha1 + alpha2
        inv_alpha_sum = wp.float64(1.0) / wp.max(alpha_sum, wp.float64(1.0e-300))
        w0 = alpha0 * inv_alpha_sum
        w1n = alpha1 * inv_alpha_sum
        w2n = alpha2 * inv_alpha_sum

        if eno_cutoff == 1:
            psi0 = wp.float64(1.0)
            psi1 = wp.float64(1.0)
            psi2 = wp.float64(1.0)
            cutoff = wp.float64(4.0e-7)
            if w0 <= cutoff:
                psi0 = wp.float64(0.0)
            if w1n <= cutoff:
                psi1 = wp.float64(0.0)
            if w2n <= cutoff:
                psi2 = wp.float64(0.0)
            cutoff_sum = psi0 * w0 + psi1 * w1n + psi2 * w2n
            inv_cutoff_sum = wp.float64(1.0) / wp.max(cutoff_sum, wp.float64(1.0e-300))
            w0 = psi0 * w0 * inv_cutoff_sum
            w1n = psi1 * w1n * inv_cutoff_sum
            w2n = psi2 * w2n * inv_cutoff_sum

        return wp.vec3d(w0, w1n, w2n)


    @wp.func
    def smooth_weight_penalty(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, w: wp.vec3d, head: int) -> wp.float64:
        if plateau_detected(q0, q1, q2, q3, q4) == 1:
            return wp.float64(0.0)
        features = weno5_nn_features(q0, q1, q2, q3, q4)
        smooth = wp.float64(1.0) - features[3]
        c = optimal_weights(head)
        dw0 = w[0] - c[0]
        dw1 = w[1] - c[1]
        dw2 = w[2] - c[2]
        return smooth * smooth * (dw0 * dw0 + dw1 * dw1 + dw2 * dw2)


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
    ) -> wp.vec4d:
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
        value = weights[0] * s0 + weights[1] * s1 + weights[2] * s2
        penalty = smooth_weight_penalty(q0, q1, q2, q3, q4, weights, head)
        return wp.vec4d(value, penalty, weights[0], weights[1])


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
    ) -> wp.vec4d:
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
        value = weights[0] * s0 + weights[1] * s1 + weights[2] * s2
        penalty = smooth_weight_penalty(q0, q1, q2, q3, q4, weights, head)
        return wp.vec4d(value, penalty, weights[0], weights[1])


    @wp.kernel(enable_backward=False)
    def copy_periodic_boundary_kernel(src: wp.array3d(dtype=wp.float64), dst: wp.array3d(dtype=wp.float64), nx: int, ny: int, gc: int):
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


    @wp.kernel(enable_backward=False)
    def copy_transmissive_boundary_kernel(src: wp.array3d(dtype=wp.float64), dst: wp.array3d(dtype=wp.float64), nx: int, ny: int, gc: int):
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


    @wp.kernel(enable_backward=False)
    def compute_x_stage_weno_mlp_kernel(
        u: wp.array3d(dtype=wp.float64),
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
        nx: int,
        ny: int,
        reg_norm: wp.float64,
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
                left = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
                right = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
                temp_l[j, i, comp] = left[0]
                temp_r[j, i, comp] = right[0]
                wp.atomic_add(reg_loss, 0, (left[1] + right[1]) * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_y_stage_weno_mlp_kernel(
        u: wp.array3d(dtype=wp.float64),
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
        nx: int,
        ny: int,
        reg_norm: wp.float64,
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
                left = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
                right = weno5_lr_value_mlp(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
                temp_l[j, i, comp] = left[0]
                temp_r[j, i, comp] = right[0]
                wp.atomic_add(reg_loss, 0, (left[1] + right[1]) * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_x_stage_weno_mlp_characteristic_kernel(
        u: wp.array3d(dtype=wp.float64),
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
        nx: int,
        ny: int,
        reg_norm: wp.float64,
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
            l0 = weno5_lr_value_mlp(c0_l[0], c1_l[0], c2_l[0], c3_l[0], c4_l[0], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            l1 = weno5_lr_value_mlp(c0_l[1], c1_l[1], c2_l[1], c3_l[1], c4_l[1], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            l2 = weno5_lr_value_mlp(c0_l[2], c1_l[2], c2_l[2], c3_l[2], c4_l[2], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            l3 = weno5_lr_value_mlp(c0_l[3], c1_l[3], c2_l[3], c3_l[3], c4_l[3], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            ql = wh.char_to_con(wp.vec4d(l0[0], l1[0], l2[0], l3[0]), roe_l, 1, gamma)

            roe_r = wh.roe_average_state(q2, q3, gamma)
            c0_r = wh.con_to_char(q0, roe_r, 1, gamma)
            c1_r = wh.con_to_char(q1, roe_r, 1, gamma)
            c2_r = wh.con_to_char(q2, roe_r, 1, gamma)
            c3_r = wh.con_to_char(q3, roe_r, 1, gamma)
            c4_r = wh.con_to_char(q4, roe_r, 1, gamma)
            r0 = weno5_lr_value_mlp(c0_r[0], c1_r[0], c2_r[0], c3_r[0], c4_r[0], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            r1 = weno5_lr_value_mlp(c0_r[1], c1_r[1], c2_r[1], c3_r[1], c4_r[1], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            r2 = weno5_lr_value_mlp(c0_r[2], c1_r[2], c2_r[2], c3_r[2], c4_r[2], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            r3 = weno5_lr_value_mlp(c0_r[3], c1_r[3], c2_r[3], c3_r[3], c4_r[3], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            qr = wh.char_to_con(wp.vec4d(r0[0], r1[0], r2[0], r3[0]), roe_r, 1, gamma)

            for comp in range(4):
                temp_l[j, i, comp] = ql[comp]
                temp_r[j, i, comp] = qr[comp]
            wp.atomic_add(reg_loss, 0, (l0[1] + l1[1] + l2[1] + l3[1] + r0[1] + r1[1] + r2[1] + r3[1]) * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_y_stage_weno_mlp_characteristic_kernel(
        u: wp.array3d(dtype=wp.float64),
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
        nx: int,
        ny: int,
        reg_norm: wp.float64,
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
            l0 = weno5_lr_value_mlp(c0_l[0], c1_l[0], c2_l[0], c3_l[0], c4_l[0], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            l1 = weno5_lr_value_mlp(c0_l[1], c1_l[1], c2_l[1], c3_l[1], c4_l[1], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            l2 = weno5_lr_value_mlp(c0_l[2], c1_l[2], c2_l[2], c3_l[2], c4_l[2], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            l3 = weno5_lr_value_mlp(c0_l[3], c1_l[3], c2_l[3], c3_l[3], c4_l[3], 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            ql = wh.char_to_con(wp.vec4d(l0[0], l1[0], l2[0], l3[0]), roe_l, 2, gamma)

            roe_r = wh.roe_average_state(q2, q3, gamma)
            c0_r = wh.con_to_char(q0, roe_r, 2, gamma)
            c1_r = wh.con_to_char(q1, roe_r, 2, gamma)
            c2_r = wh.con_to_char(q2, roe_r, 2, gamma)
            c3_r = wh.con_to_char(q3, roe_r, 2, gamma)
            c4_r = wh.con_to_char(q4, roe_r, 2, gamma)
            r0 = weno5_lr_value_mlp(c0_r[0], c1_r[0], c2_r[0], c3_r[0], c4_r[0], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            r1 = weno5_lr_value_mlp(c0_r[1], c1_r[1], c2_r[1], c3_r[1], c4_r[1], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            r2 = weno5_lr_value_mlp(c0_r[2], c1_r[2], c2_r[2], c3_r[2], c4_r[2], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            r3 = weno5_lr_value_mlp(c0_r[3], c1_r[3], c2_r[3], c3_r[3], c4_r[3], 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
            qr = wh.char_to_con(wp.vec4d(r0[0], r1[0], r2[0], r3[0]), roe_r, 2, gamma)

            for comp in range(4):
                temp_l[j, i, comp] = ql[comp]
                temp_r[j, i, comp] = qr[comp]
            wp.atomic_add(reg_loss, 0, (l0[1] + l1[1] + l2[1] + l3[1] + r0[1] + r1[1] + r2[1] + r3[1]) * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_x_point_flux_single_mlp_kernel(
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
        tempdx_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        reg_norm: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            penalty = wp.float64(0.0)
            for comp in range(4):
                left = weno5_gauss_value_mlp(
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
                right = weno5_gauss_value_mlp(
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
                left_state[comp] = left[0]
                right_state[comp] = right[0]
                penalty = penalty + left[1] + right[1]
            f = wh.force_flux(right_state, left_state, 1, tempdx_dt, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = f[comp] * tempdx_dt
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_y_point_flux_single_mlp_kernel(
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
        tempdy_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        reg_norm: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            penalty = wp.float64(0.0)
            for comp in range(4):
                left = weno5_gauss_value_mlp(
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
                right = weno5_gauss_value_mlp(
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
                left_state[comp] = left[0]
                right_state[comp] = right[0]
                penalty = penalty + left[1] + right[1]
            f = wh.force_flux(right_state, left_state, 2, tempdy_dt, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = f[comp] * tempdy_dt
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_x_point_flux_single_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        tempdx_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                left_state[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 2, i + 1, comp],
                    temp_l[j + 3, i + 1, comp],
                    temp_l[j + 4, i + 1, comp],
                    temp_l[j + 5, i + 1, comp],
                    loca,
                )
                right_state[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j + 1, i, comp],
                    temp_r[j + 2, i, comp],
                    temp_r[j + 3, i, comp],
                    temp_r[j + 4, i, comp],
                    temp_r[j + 5, i, comp],
                    loca,
                )
            f = wh.force_flux(right_state, left_state, 1, tempdx_dt, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = f[comp] * tempdx_dt


    @wp.kernel(enable_backward=False)
    def compute_y_point_flux_single_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        tempdy_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                left_state[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 1, i + 2, comp],
                    temp_l[j + 1, i + 3, comp],
                    temp_l[j + 1, i + 4, comp],
                    temp_l[j + 1, i + 5, comp],
                    loca,
                )
                right_state[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j, i + 1, comp],
                    temp_r[j, i + 2, comp],
                    temp_r[j, i + 3, comp],
                    temp_r[j, i + 4, comp],
                    temp_r[j, i + 5, comp],
                    loca,
                )
            f = wh.force_flux(right_state, left_state, 2, tempdy_dt, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = f[comp] * tempdy_dt


    @wp.kernel(enable_backward=False)
    def compute_x_point_flux_single_mlp_evilin_kernel(
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
        tempdx_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        reg_norm: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            penalty = wp.float64(0.0)
            for comp in range(4):
                left = weno5_gauss_value_mlp(
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
                right = weno5_gauss_value_mlp(
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
                left_state[comp] = left[0]
                right_state[comp] = right[0]
                penalty = penalty + left[1] + right[1]
            f = evilin_flux(right_state, left_state, 1, tempdx_dt, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = f[comp] * tempdx_dt
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_y_point_flux_single_mlp_evilin_kernel(
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
        tempdy_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
        reg_norm: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            penalty = wp.float64(0.0)
            for comp in range(4):
                left = weno5_gauss_value_mlp(
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
                right = weno5_gauss_value_mlp(
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
                left_state[comp] = left[0]
                right_state[comp] = right[0]
                penalty = penalty + left[1] + right[1]
            f = evilin_flux(right_state, left_state, 2, tempdy_dt, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = f[comp] * tempdy_dt
            wp.atomic_add(reg_loss, 0, penalty * reg_norm)


    @wp.kernel(enable_backward=False)
    def compute_x_point_flux_single_evilin_kernel(
        flux_x: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        tempdx_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx + 1:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                left_state[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 2, i + 1, comp],
                    temp_l[j + 3, i + 1, comp],
                    temp_l[j + 4, i + 1, comp],
                    temp_l[j + 5, i + 1, comp],
                    loca,
                )
                right_state[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j + 1, i, comp],
                    temp_r[j + 2, i, comp],
                    temp_r[j + 3, i, comp],
                    temp_r[j + 4, i, comp],
                    temp_r[j + 5, i, comp],
                    loca,
                )
            f = evilin_flux(right_state, left_state, 1, tempdx_dt, gamma)
            for comp in range(4):
                flux_x[j, i, comp] = f[comp] * tempdx_dt


    @wp.kernel(enable_backward=False)
    def compute_y_point_flux_single_evilin_kernel(
        flux_y: wp.array3d(dtype=wp.float64),
        temp_l: wp.array3d(dtype=wp.float64),
        temp_r: wp.array3d(dtype=wp.float64),
        tempdy_dt: wp.float64,
        nx: int,
        ny: int,
        loca: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 1 and i < nx:
            left_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            right_state = wp.vec4d(0.0, 0.0, 0.0, 0.0)
            for comp in range(4):
                left_state[comp] = wh.weno5_gauss_lr_value(
                    temp_l[j + 1, i + 1, comp],
                    temp_l[j + 1, i + 2, comp],
                    temp_l[j + 1, i + 3, comp],
                    temp_l[j + 1, i + 4, comp],
                    temp_l[j + 1, i + 5, comp],
                    loca,
                )
                right_state[comp] = wh.weno5_gauss_lr_value(
                    temp_r[j, i + 1, comp],
                    temp_r[j, i + 2, comp],
                    temp_r[j, i + 3, comp],
                    temp_r[j, i + 4, comp],
                    temp_r[j, i + 5, comp],
                    loca,
                )
            f = evilin_flux(right_state, left_state, 2, tempdy_dt, gamma)
            for comp in range(4):
                flux_y[j, i, comp] = f[comp] * tempdy_dt


    @wp.kernel(enable_backward=False)
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
                rhs = u_stage[jj, ii, comp] - wp.float64(0.5) * (fx_r - fx_l) - wp.float64(0.5) * (fy_t - fy_b)
                if rk == 1:
                    u_out[jj, ii, comp] = rhs
                elif rk == 2:
                    u_out[jj, ii, comp] = wp.float64(0.75) * u0[jj, ii, comp] + wp.float64(0.25) * rhs
                else:
                    u_out[jj, ii, comp] = (wp.float64(1.0) / wp.float64(3.0)) * u0[jj, ii, comp] + (wp.float64(2.0) / wp.float64(3.0)) * rhs


    @wp.kernel(enable_backward=False)
    def density_mse_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            d = u[jj, ii, 0] - target[jj, ii, 0]
            wp.atomic_add(loss, 0, d * d / wp.float64(nx * ny))


    @wp.kernel(enable_backward=False)
    def conserved_relative_mse_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            for comp in range(4):
                denom = wp.max(wp.abs(target[jj, ii, comp]), wp.float64(1.0))
                d = (u[jj, ii, comp] - target[jj, ii, comp]) / denom
                wp.atomic_add(loss, 0, d * d / wp.float64(4 * nx * ny))


    @wp.kernel(enable_backward=False)
    def density_pressure_l1_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            qt = wp.vec4d(target[jj, ii, 0], target[jj, ii, 1], target[jj, ii, 2], target[jj, ii, 3])
            w = wh.con_to_pri(q, gamma)
            wt = wh.con_to_pri(qt, gamma)

            erho = smooth_abs_loss((w[0] - wt[0]) / wp.max(wp.abs(wt[0]), wp.float64(1.0)))
            ep = smooth_abs_loss((w[3] - wt[3]) / wp.max(wp.abs(wt[3]), wp.float64(1.0)))
            wp.atomic_add(loss, 0, wp.float64(0.5) * (erho + ep) / wp.float64(nx * ny))


    @wp.kernel(enable_backward=False)
    def density_pressure_lp_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
        lp_order: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            qt = wp.vec4d(target[jj, ii, 0], target[jj, ii, 1], target[jj, ii, 2], target[jj, ii, 3])
            w = wh.con_to_pri(q, gamma)
            wt = wh.con_to_pri(qt, gamma)

            erho = smooth_abs_loss((w[0] - wt[0]) / wp.max(wp.abs(wt[0]), wp.float64(1.0)))
            ep = smooth_abs_loss((w[3] - wt[3]) / wp.max(wp.abs(wt[3]), wp.float64(1.0)))
            local = wp.float64(0.5) * (wp.pow(erho, lp_order) + wp.pow(ep, lp_order))
            wp.atomic_add(loss, 0, local / wp.float64(nx * ny))


    @wp.kernel(enable_backward=False)
    def density_pressure_tv_excess_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
        tv_target_multiplier: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            qt = wp.vec4d(target[jj, ii, 0], target[jj, ii, 1], target[jj, ii, 2], target[jj, ii, 3])
            w = wh.con_to_pri(q, gamma)
            wt = wh.con_to_pri(qt, gamma)

            accum = wp.float64(0.0)
            count = wp.float64(0.0)

            if i + 1 < nx:
                qx = wp.vec4d(u[jj, ii + 1, 0], u[jj, ii + 1, 1], u[jj, ii + 1, 2], u[jj, ii + 1, 3])
                qtx = wp.vec4d(target[jj, ii + 1, 0], target[jj, ii + 1, 1], target[jj, ii + 1, 2], target[jj, ii + 1, 3])
                wx = wh.con_to_pri(qx, gamma)
                wtx = wh.con_to_pri(qtx, gamma)

                denom_rho_x = wp.max(wp.float64(0.5) * (wp.abs(wt[0]) + wp.abs(wtx[0])), wp.float64(1.0))
                denom_p_x = wp.max(wp.float64(0.5) * (wp.abs(wt[3]) + wp.abs(wtx[3])), wp.float64(1.0))
                erho_x = smooth_pos_loss(smooth_abs_loss(wx[0] - w[0]) - tv_target_multiplier * wp.abs(wtx[0] - wt[0])) / denom_rho_x
                ep_x = smooth_pos_loss(smooth_abs_loss(wx[3] - w[3]) - tv_target_multiplier * wp.abs(wtx[3] - wt[3])) / denom_p_x
                accum = accum + erho_x * erho_x + ep_x * ep_x
                count = count + wp.float64(2.0)

            if j + 1 < ny:
                qy = wp.vec4d(u[jj + 1, ii, 0], u[jj + 1, ii, 1], u[jj + 1, ii, 2], u[jj + 1, ii, 3])
                qty = wp.vec4d(target[jj + 1, ii, 0], target[jj + 1, ii, 1], target[jj + 1, ii, 2], target[jj + 1, ii, 3])
                wy = wh.con_to_pri(qy, gamma)
                wty = wh.con_to_pri(qty, gamma)

                denom_rho_y = wp.max(wp.float64(0.5) * (wp.abs(wt[0]) + wp.abs(wty[0])), wp.float64(1.0))
                denom_p_y = wp.max(wp.float64(0.5) * (wp.abs(wt[3]) + wp.abs(wty[3])), wp.float64(1.0))
                erho_y = smooth_pos_loss(smooth_abs_loss(wy[0] - w[0]) - tv_target_multiplier * wp.abs(wty[0] - wt[0])) / denom_rho_y
                ep_y = smooth_pos_loss(smooth_abs_loss(wy[3] - w[3]) - tv_target_multiplier * wp.abs(wty[3] - wt[3])) / denom_p_y
                accum = accum + erho_y * erho_y + ep_y * ep_y
                count = count + wp.float64(2.0)

            if count > wp.float64(0.0):
                wp.atomic_add(loss, 0, accum / (count * wp.float64(nx * ny)))


    @wp.kernel(enable_backward=False)
    def density_pressure_error_tv_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            qt = wp.vec4d(target[jj, ii, 0], target[jj, ii, 1], target[jj, ii, 2], target[jj, ii, 3])
            w = wh.con_to_pri(q, gamma)
            wt = wh.con_to_pri(qt, gamma)

            erho = (w[0] - wt[0]) / wp.max(wp.abs(wt[0]), wp.float64(1.0))
            ep = (w[3] - wt[3]) / wp.max(wp.abs(wt[3]), wp.float64(1.0))
            accum = wp.float64(0.0)
            count = wp.float64(0.0)

            if i + 1 < nx:
                qx = wp.vec4d(u[jj, ii + 1, 0], u[jj, ii + 1, 1], u[jj, ii + 1, 2], u[jj, ii + 1, 3])
                qtx = wp.vec4d(target[jj, ii + 1, 0], target[jj, ii + 1, 1], target[jj, ii + 1, 2], target[jj, ii + 1, 3])
                wx = wh.con_to_pri(qx, gamma)
                wtx = wh.con_to_pri(qtx, gamma)
                erho_x = (wx[0] - wtx[0]) / wp.max(wp.abs(wtx[0]), wp.float64(1.0))
                ep_x = (wx[3] - wtx[3]) / wp.max(wp.abs(wtx[3]), wp.float64(1.0))
                drho_x = erho_x - erho
                dp_x = ep_x - ep
                accum = accum + drho_x * drho_x + dp_x * dp_x
                count = count + wp.float64(2.0)

            if j + 1 < ny:
                qy = wp.vec4d(u[jj + 1, ii, 0], u[jj + 1, ii, 1], u[jj + 1, ii, 2], u[jj + 1, ii, 3])
                qty = wp.vec4d(target[jj + 1, ii, 0], target[jj + 1, ii, 1], target[jj + 1, ii, 2], target[jj + 1, ii, 3])
                wy = wh.con_to_pri(qy, gamma)
                wty = wh.con_to_pri(qty, gamma)
                erho_y = (wy[0] - wty[0]) / wp.max(wp.abs(wty[0]), wp.float64(1.0))
                ep_y = (wy[3] - wty[3]) / wp.max(wp.abs(wty[3]), wp.float64(1.0))
                drho_y = erho_y - erho
                dp_y = ep_y - ep
                accum = accum + drho_y * drho_y + dp_y * dp_y
                count = count + wp.float64(2.0)

            if count > wp.float64(0.0):
                wp.atomic_add(loss, 0, accum / (count * wp.float64(nx * ny)))


    @wp.kernel(enable_backward=False)
    def density_pressure_shock_error_tv_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        norm: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
        shock_strength: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            qt = wp.vec4d(target[jj, ii, 0], target[jj, ii, 1], target[jj, ii, 2], target[jj, ii, 3])
            w = wh.con_to_pri(q, gamma)
            wt = wh.con_to_pri(qt, gamma)

            denom_rho = wp.max(wp.abs(wt[0]), wp.float64(1.0))
            denom_p = wp.max(wp.abs(wt[3]), wp.float64(1.0))
            cell_indicator = wp.float64(0.0)
            if i + 1 < nx:
                qtxi = wp.vec4d(target[jj, ii + 1, 0], target[jj, ii + 1, 1], target[jj, ii + 1, 2], target[jj, ii + 1, 3])
                wtxi = wh.con_to_pri(qtxi, gamma)
                cell_indicator = cell_indicator + wp.abs(wtxi[0] - wt[0]) / denom_rho + wp.abs(wtxi[3] - wt[3]) / denom_p
            if i > 0:
                qtxim = wp.vec4d(target[jj, ii - 1, 0], target[jj, ii - 1, 1], target[jj, ii - 1, 2], target[jj, ii - 1, 3])
                wtxim = wh.con_to_pri(qtxim, gamma)
                cell_indicator = cell_indicator + wp.abs(wt[0] - wtxim[0]) / denom_rho + wp.abs(wt[3] - wtxim[3]) / denom_p
            if j + 1 < ny:
                qtyi = wp.vec4d(target[jj + 1, ii, 0], target[jj + 1, ii, 1], target[jj + 1, ii, 2], target[jj + 1, ii, 3])
                wtyi = wh.con_to_pri(qtyi, gamma)
                cell_indicator = cell_indicator + wp.abs(wtyi[0] - wt[0]) / denom_rho + wp.abs(wtyi[3] - wt[3]) / denom_p
            if j > 0:
                qtyim = wp.vec4d(target[jj - 1, ii, 0], target[jj - 1, ii, 1], target[jj - 1, ii, 2], target[jj - 1, ii, 3])
                wtyim = wh.con_to_pri(qtyim, gamma)
                cell_indicator = cell_indicator + wp.abs(wt[0] - wtyim[0]) / denom_rho + wp.abs(wt[3] - wtyim[3]) / denom_p

            erho = (w[0] - wt[0]) / wp.max(wp.abs(wt[0]), wp.float64(1.0))
            ep = (w[3] - wt[3]) / wp.max(wp.abs(wt[3]), wp.float64(1.0))

            if i + 1 < nx:
                qx = wp.vec4d(u[jj, ii + 1, 0], u[jj, ii + 1, 1], u[jj, ii + 1, 2], u[jj, ii + 1, 3])
                qtx = wp.vec4d(target[jj, ii + 1, 0], target[jj, ii + 1, 1], target[jj, ii + 1, 2], target[jj, ii + 1, 3])
                wx = wh.con_to_pri(qx, gamma)
                wtx = wh.con_to_pri(qtx, gamma)
                erho_x = (wx[0] - wtx[0]) / wp.max(wp.abs(wtx[0]), wp.float64(1.0))
                ep_x = (wx[3] - wtx[3]) / wp.max(wp.abs(wtx[3]), wp.float64(1.0))

                denom_rho_x = wp.max(wp.float64(0.5) * (wp.abs(wt[0]) + wp.abs(wtx[0])), wp.float64(1.0))
                denom_p_x = wp.max(wp.float64(0.5) * (wp.abs(wt[3]) + wp.abs(wtx[3])), wp.float64(1.0))
                jump_x = wp.abs(wtx[0] - wt[0]) / denom_rho_x + wp.abs(wtx[3] - wt[3]) / denom_p_x
                shock_weight_x = wp.min(wp.float64(1.0), shock_strength * wp.max(jump_x, cell_indicator))
                drho_x = erho_x - erho
                dp_x = ep_x - ep
                wp.atomic_add(loss, 0, shock_weight_x * (drho_x * drho_x + dp_x * dp_x))
                wp.atomic_add(norm, 0, shock_weight_x * wp.float64(2.0))

            if j + 1 < ny:
                qy = wp.vec4d(u[jj + 1, ii, 0], u[jj + 1, ii, 1], u[jj + 1, ii, 2], u[jj + 1, ii, 3])
                qty = wp.vec4d(target[jj + 1, ii, 0], target[jj + 1, ii, 1], target[jj + 1, ii, 2], target[jj + 1, ii, 3])
                wy = wh.con_to_pri(qy, gamma)
                wty = wh.con_to_pri(qty, gamma)
                erho_y = (wy[0] - wty[0]) / wp.max(wp.abs(wty[0]), wp.float64(1.0))
                ep_y = (wy[3] - wty[3]) / wp.max(wp.abs(wty[3]), wp.float64(1.0))

                denom_rho_y = wp.max(wp.float64(0.5) * (wp.abs(wt[0]) + wp.abs(wty[0])), wp.float64(1.0))
                denom_p_y = wp.max(wp.float64(0.5) * (wp.abs(wt[3]) + wp.abs(wty[3])), wp.float64(1.0))
                jump_y = wp.abs(wty[0] - wt[0]) / denom_rho_y + wp.abs(wty[3] - wt[3]) / denom_p_y
                shock_weight_y = wp.min(wp.float64(1.0), shock_strength * wp.max(jump_y, cell_indicator))
                drho_y = erho_y - erho
                dp_y = ep_y - ep
                wp.atomic_add(loss, 0, shock_weight_y * (drho_y * drho_y + dp_y * dp_y))
                wp.atomic_add(norm, 0, shock_weight_y * wp.float64(2.0))


    @wp.kernel(enable_backward=False)
    def density_pressure_shock_weighted_lp_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        norm: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
        lp_order: wp.float64,
        shock_strength: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            qt = wp.vec4d(target[jj, ii, 0], target[jj, ii, 1], target[jj, ii, 2], target[jj, ii, 3])
            w = wh.con_to_pri(q, gamma)
            wt = wh.con_to_pri(qt, gamma)

            denom_rho = wp.max(wp.abs(wt[0]), wp.float64(1.0))
            denom_p = wp.max(wp.abs(wt[3]), wp.float64(1.0))
            shock_indicator = wp.float64(0.0)

            if i + 1 < nx:
                qtx = wp.vec4d(target[jj, ii + 1, 0], target[jj, ii + 1, 1], target[jj, ii + 1, 2], target[jj, ii + 1, 3])
                wtx = wh.con_to_pri(qtx, gamma)
                shock_indicator = shock_indicator + wp.abs(wtx[0] - wt[0]) / denom_rho + wp.abs(wtx[3] - wt[3]) / denom_p
            if i > 0:
                qtxm = wp.vec4d(target[jj, ii - 1, 0], target[jj, ii - 1, 1], target[jj, ii - 1, 2], target[jj, ii - 1, 3])
                wtxm = wh.con_to_pri(qtxm, gamma)
                shock_indicator = shock_indicator + wp.abs(wt[0] - wtxm[0]) / denom_rho + wp.abs(wt[3] - wtxm[3]) / denom_p
            if j + 1 < ny:
                qty = wp.vec4d(target[jj + 1, ii, 0], target[jj + 1, ii, 1], target[jj + 1, ii, 2], target[jj + 1, ii, 3])
                wty = wh.con_to_pri(qty, gamma)
                shock_indicator = shock_indicator + wp.abs(wty[0] - wt[0]) / denom_rho + wp.abs(wty[3] - wt[3]) / denom_p
            if j > 0:
                qtym = wp.vec4d(target[jj - 1, ii, 0], target[jj - 1, ii, 1], target[jj - 1, ii, 2], target[jj - 1, ii, 3])
                wtym = wh.con_to_pri(qtym, gamma)
                shock_indicator = shock_indicator + wp.abs(wt[0] - wtym[0]) / denom_rho + wp.abs(wt[3] - wtym[3]) / denom_p

            erho = smooth_abs_loss((w[0] - wt[0]) / denom_rho)
            ep = smooth_abs_loss((w[3] - wt[3]) / denom_p)
            shock_weight = wp.min(wp.float64(1.0), shock_strength * shock_indicator)
            local = shock_weight * wp.float64(0.5) * (wp.pow(erho, lp_order) + wp.pow(ep, lp_order))
            wp.atomic_add(loss, 0, local)
            wp.atomic_add(norm, 0, shock_weight)


    @wp.kernel(enable_backward=False)
    def density_pressure_shock_range_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        target: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        norm: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
        shock_strength: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            qt = wp.vec4d(target[jj, ii, 0], target[jj, ii, 1], target[jj, ii, 2], target[jj, ii, 3])
            w = wh.con_to_pri(q, gamma)
            wt = wh.con_to_pri(qt, gamma)

            rho_min = wt[0]
            rho_max = wt[0]
            p_min = wt[3]
            p_max = wt[3]
            denom_rho = wp.max(wp.abs(wt[0]), wp.float64(1.0))
            denom_p = wp.max(wp.abs(wt[3]), wp.float64(1.0))
            shock_indicator = wp.float64(0.0)

            if i + 1 < nx:
                qtx = wp.vec4d(target[jj, ii + 1, 0], target[jj, ii + 1, 1], target[jj, ii + 1, 2], target[jj, ii + 1, 3])
                wtx = wh.con_to_pri(qtx, gamma)
                rho_min = wp.min(rho_min, wtx[0])
                rho_max = wp.max(rho_max, wtx[0])
                p_min = wp.min(p_min, wtx[3])
                p_max = wp.max(p_max, wtx[3])
                shock_indicator = shock_indicator + wp.abs(wtx[0] - wt[0]) / denom_rho + wp.abs(wtx[3] - wt[3]) / denom_p
            if i > 0:
                qtxm = wp.vec4d(target[jj, ii - 1, 0], target[jj, ii - 1, 1], target[jj, ii - 1, 2], target[jj, ii - 1, 3])
                wtxm = wh.con_to_pri(qtxm, gamma)
                rho_min = wp.min(rho_min, wtxm[0])
                rho_max = wp.max(rho_max, wtxm[0])
                p_min = wp.min(p_min, wtxm[3])
                p_max = wp.max(p_max, wtxm[3])
                shock_indicator = shock_indicator + wp.abs(wt[0] - wtxm[0]) / denom_rho + wp.abs(wt[3] - wtxm[3]) / denom_p
            if j + 1 < ny:
                qty = wp.vec4d(target[jj + 1, ii, 0], target[jj + 1, ii, 1], target[jj + 1, ii, 2], target[jj + 1, ii, 3])
                wty = wh.con_to_pri(qty, gamma)
                rho_min = wp.min(rho_min, wty[0])
                rho_max = wp.max(rho_max, wty[0])
                p_min = wp.min(p_min, wty[3])
                p_max = wp.max(p_max, wty[3])
                shock_indicator = shock_indicator + wp.abs(wty[0] - wt[0]) / denom_rho + wp.abs(wty[3] - wt[3]) / denom_p
            if j > 0:
                qtym = wp.vec4d(target[jj - 1, ii, 0], target[jj - 1, ii, 1], target[jj - 1, ii, 2], target[jj - 1, ii, 3])
                wtym = wh.con_to_pri(qtym, gamma)
                rho_min = wp.min(rho_min, wtym[0])
                rho_max = wp.max(rho_max, wtym[0])
                p_min = wp.min(p_min, wtym[3])
                p_max = wp.max(p_max, wtym[3])
                shock_indicator = shock_indicator + wp.abs(wt[0] - wtym[0]) / denom_rho + wp.abs(wt[3] - wtym[3]) / denom_p

            shock_weight = wp.min(wp.float64(1.0), shock_strength * shock_indicator)
            rho_span = wp.max(rho_max - rho_min, wp.float64(1.0))
            p_span = wp.max(p_max - p_min, wp.float64(1.0))
            rho_hi = smooth_pos_loss(w[0] - rho_max) / rho_span
            rho_lo = smooth_pos_loss(rho_min - w[0]) / rho_span
            p_hi = smooth_pos_loss(w[3] - p_max) / p_span
            p_lo = smooth_pos_loss(p_min - w[3]) / p_span
            local = shock_weight * wp.float64(0.5) * (rho_hi * rho_hi + rho_lo * rho_lo + p_hi * p_hi + p_lo * p_lo)
            wp.atomic_add(loss, 0, local)
            wp.atomic_add(norm, 0, shock_weight)


    @wp.kernel(enable_backward=False)
    def positivity_loss_kernel(
        u: wp.array3d(dtype=wp.float64),
        loss: wp.array(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        gamma: wp.float64,
        rho_floor: wp.float64,
        p_floor: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny and i < nx:
            jj = j + gc
            ii = i + gc
            q = wp.vec4d(u[jj, ii, 0], u[jj, ii, 1], u[jj, ii, 2], u[jj, ii, 3])
            rho = q[0]
            rho_for_pressure = wp.max(rho, rho_floor)
            kinetic = wp.float64(0.5) * (q[1] * q[1] + q[2] * q[2]) / rho_for_pressure
            pressure = (gamma - wp.float64(1.0)) * (q[3] - kinetic)
            drho = wp.max(rho_floor - rho, wp.float64(0.0))
            dp = wp.max(p_floor - pressure, wp.float64(0.0))
            wp.atomic_add(loss, 0, (drho * drho + dp * dp) / wp.float64(2 * nx * ny))


    @wp.kernel(enable_backward=False)
    def combine_loss_kernel(
        data_loss: wp.array(dtype=wp.float64),
        reg_loss: wp.array(dtype=wp.float64),
        lp_loss: wp.array(dtype=wp.float64),
        tv_loss: wp.array(dtype=wp.float64),
        errtv_loss: wp.array(dtype=wp.float64),
        shock_errtv_loss: wp.array(dtype=wp.float64),
        shock_errtv_norm: wp.array(dtype=wp.float64),
        shock_lp_loss: wp.array(dtype=wp.float64),
        shock_lp_norm: wp.array(dtype=wp.float64),
        shock_range_loss: wp.array(dtype=wp.float64),
        shock_range_norm: wp.array(dtype=wp.float64),
        pos_loss: wp.array(dtype=wp.float64),
        total_loss: wp.array(dtype=wp.float64),
        data_weight: wp.float64,
        smooth_lambda: wp.float64,
        lp_lambda: wp.float64,
        tv_lambda: wp.float64,
        errtv_lambda: wp.float64,
        shock_errtv_lambda: wp.float64,
        shock_lp_lambda: wp.float64,
        shock_range_lambda: wp.float64,
        pos_lambda: wp.float64,
        lp_order: wp.float64,
    ):
        lp_root = stable_lp_root_loss(lp_loss[0], lp_order)
        shock_errtv_mean = shock_errtv_loss[0] / wp.max(shock_errtv_norm[0], wp.float64(1.0))
        shock_lp_mean = shock_lp_loss[0] / wp.max(shock_lp_norm[0], wp.float64(1.0))
        shock_lp_root = stable_lp_root_loss(shock_lp_mean, lp_order)
        shock_range_mean = shock_range_loss[0] / wp.max(shock_range_norm[0], wp.float64(1.0))
        total_loss[0] = (
            data_weight * data_loss[0]
            + smooth_lambda * reg_loss[0]
            + lp_lambda * lp_root
            + tv_lambda * tv_loss[0]
            + errtv_lambda * errtv_loss[0]
            + shock_errtv_lambda * shock_errtv_mean
            + shock_lp_lambda * shock_lp_root
            + shock_range_lambda * shock_range_mean
            + pos_lambda * pos_loss[0]
        )


    @wp.kernel(enable_backward=False)
    def add_rollout_step_loss_kernel(
        data_loss: wp.array(dtype=wp.float64),
        lp_loss: wp.array(dtype=wp.float64),
        tv_loss: wp.array(dtype=wp.float64),
        errtv_loss: wp.array(dtype=wp.float64),
        shock_errtv_loss: wp.array(dtype=wp.float64),
        shock_errtv_norm: wp.array(dtype=wp.float64),
        shock_lp_loss: wp.array(dtype=wp.float64),
        shock_lp_norm: wp.array(dtype=wp.float64),
        shock_range_loss: wp.array(dtype=wp.float64),
        shock_range_norm: wp.array(dtype=wp.float64),
        pos_loss: wp.array(dtype=wp.float64),
        agg_data_loss: wp.array(dtype=wp.float64),
        agg_lp_loss: wp.array(dtype=wp.float64),
        agg_tv_loss: wp.array(dtype=wp.float64),
        agg_errtv_loss: wp.array(dtype=wp.float64),
        agg_shock_errtv_loss: wp.array(dtype=wp.float64),
        agg_shock_lp_loss: wp.array(dtype=wp.float64),
        agg_shock_range_loss: wp.array(dtype=wp.float64),
        agg_pos_loss: wp.array(dtype=wp.float64),
        total_loss: wp.array(dtype=wp.float64),
        rollout_weight: wp.float64,
        data_weight: wp.float64,
        lp_lambda: wp.float64,
        tv_lambda: wp.float64,
        errtv_lambda: wp.float64,
        shock_errtv_lambda: wp.float64,
        shock_lp_lambda: wp.float64,
        shock_range_lambda: wp.float64,
        pos_lambda: wp.float64,
        lp_order: wp.float64,
    ):
        lp_root = stable_lp_root_loss(lp_loss[0], lp_order)
        shock_errtv_mean = shock_errtv_loss[0] / wp.max(shock_errtv_norm[0], wp.float64(1.0))
        shock_lp_mean = shock_lp_loss[0] / wp.max(shock_lp_norm[0], wp.float64(1.0))
        shock_lp_root = stable_lp_root_loss(shock_lp_mean, lp_order)
        shock_range_mean = shock_range_loss[0] / wp.max(shock_range_norm[0], wp.float64(1.0))
        agg_data_loss[0] = agg_data_loss[0] + rollout_weight * data_loss[0]
        agg_lp_loss[0] = agg_lp_loss[0] + rollout_weight * lp_loss[0]
        agg_tv_loss[0] = agg_tv_loss[0] + rollout_weight * tv_loss[0]
        agg_errtv_loss[0] = agg_errtv_loss[0] + rollout_weight * errtv_loss[0]
        agg_shock_errtv_loss[0] = agg_shock_errtv_loss[0] + rollout_weight * shock_errtv_mean
        agg_shock_lp_loss[0] = agg_shock_lp_loss[0] + rollout_weight * shock_lp_mean
        agg_shock_range_loss[0] = agg_shock_range_loss[0] + rollout_weight * shock_range_mean
        agg_pos_loss[0] = agg_pos_loss[0] + rollout_weight * pos_loss[0]
        total_loss[0] = (
            total_loss[0]
            + rollout_weight
            * (
                data_weight * data_loss[0]
                + lp_lambda * lp_root
                + tv_lambda * tv_loss[0]
                + errtv_lambda * errtv_loss[0]
                + shock_errtv_lambda * shock_errtv_mean
                + shock_lp_lambda * shock_lp_root
                + shock_range_lambda * shock_range_mean
                + pos_lambda * pos_loss[0]
            )
        )


    @wp.kernel(enable_backward=False)
    def add_smooth_reg_loss_kernel(
        reg_loss: wp.array(dtype=wp.float64),
        total_loss: wp.array(dtype=wp.float64),
        smooth_lambda: wp.float64,
    ):
        total_loss[0] = total_loss[0] + smooth_lambda * reg_loss[0]


def zeros_like_state(params: wh.Params, device: str, requires_grad: bool = True):
    return wp.zeros(params.padded_shape, dtype=wp.float64, device=device, requires_grad=requires_grad)


def allocate_stage(params: wh.Params, device: str, prefix: str) -> dict[str, object]:
    del prefix
    shape = params.padded_shape
    return {
        "bc": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "x_l": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "x_r": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "y_l": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "y_r": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "fx1": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "fx2": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "fy1": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
        "fy2": wp.zeros(shape, dtype=wp.float64, device=device, requires_grad=True),
    }


def make_mlp_params(device: str, seed: int = 1) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    inputs = MLP_INPUTS if wp is not None else 5
    hidden1 = MLP_HIDDEN1 if wp is not None else 10
    hidden2 = MLP_HIDDEN2 if wp is not None else 6
    hidden3 = MLP_HIDDEN3 if wp is not None else 6
    w1 = rng.normal(0.0, np.sqrt(1.0 / float(inputs)), size=(1, inputs, hidden1)).astype(np.float64)
    b1 = np.zeros((1, hidden1), dtype=np.float64)
    w2 = rng.normal(0.0, np.sqrt(1.0 / float(hidden1)), size=(1, hidden1, hidden2)).astype(np.float64)
    b2 = np.zeros((1, hidden2), dtype=np.float64)
    w3 = rng.normal(0.0, np.sqrt(1.0 / float(hidden2)), size=(1, hidden2, hidden3)).astype(np.float64)
    b3 = np.zeros((1, hidden3), dtype=np.float64)
    w4 = np.zeros((1, hidden3, 3), dtype=np.float64)
    r_init = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    init_logits = np.log(r_init)
    init_logits = init_logits - np.mean(init_logits)
    b4 = (6.0 * np.arctanh(init_logits / 6.0))[None, :].astype(np.float64)

    return {
        "w1": wp.array(w1, dtype=wp.float64, device=device, requires_grad=True),
        "b1": wp.array(b1, dtype=wp.float64, device=device, requires_grad=True),
        "w2": wp.array(w2, dtype=wp.float64, device=device, requires_grad=True),
        "b2": wp.array(b2, dtype=wp.float64, device=device, requires_grad=True),
        "w3": wp.array(w3, dtype=wp.float64, device=device, requires_grad=True),
        "b3": wp.array(b3, dtype=wp.float64, device=device, requires_grad=True),
        "w4": wp.array(w4, dtype=wp.float64, device=device, requires_grad=True),
        "b4": wp.array(b4, dtype=wp.float64, device=device, requires_grad=True),
    }


def launch_diff_stage(
    u_stage: object,
    u0: object,
    u_out: object,
    stage: dict[str, object],
    params: wh.Params,
    dt: float,
    rk: int,
    device: str,
    characteristic_weno: bool,
    mlp_params: dict[str, object] | None,
    reg_loss: object | None,
    reg_norm: float,
    eno_cutoff: bool = False,
    boundary: str = "periodic",
    riemann_solver: str | int = "force",
) -> None:
    nx = params.nx
    ny = params.ny
    gc = params.ghost
    nx_total = nx + 2 * gc
    ny_total = ny + 2 * gc
    characteristic = 1 if characteristic_weno else 0
    tempdx_dt = wp.float64(dt / params.dx)
    tempdy_dt = wp.float64(dt / params.dy)
    gamma = wp.float64(params.gamma)
    eno_cutoff_i = 1 if eno_cutoff else 0
    riemann_solver_i = 1 if riemann_solver == "evilin" or riemann_solver == 1 else 0

    if boundary == "transmissive":
        wp.launch(copy_transmissive_boundary_kernel, dim=(ny_total, nx_total), inputs=[u_stage, stage["bc"], nx, ny, gc], device=device)
    else:
        wp.launch(copy_periodic_boundary_kernel, dim=(ny_total, nx_total), inputs=[u_stage, stage["bc"], nx, ny, gc], device=device)
    if mlp_params is None:
        wp.launch(wh.compute_x_stage_weno_kernel, dim=(ny + 6, nx + 2), inputs=[stage["bc"], stage["x_l"], stage["x_r"], nx, ny, characteristic, gamma], device=device)
        x_flux_kernel = compute_x_point_flux_single_evilin_kernel if riemann_solver_i == 1 else compute_x_point_flux_single_kernel
        y_flux_kernel = compute_y_point_flux_single_evilin_kernel if riemann_solver_i == 1 else compute_y_point_flux_single_kernel
        wp.launch(x_flux_kernel, dim=(ny, nx + 1), inputs=[stage["fx1"], stage["x_l"], stage["x_r"], tempdx_dt, nx, ny, 1, gamma], device=device)
        wp.launch(x_flux_kernel, dim=(ny, nx + 1), inputs=[stage["fx2"], stage["x_l"], stage["x_r"], tempdx_dt, nx, ny, 2, gamma], device=device)

        wp.launch(wh.compute_y_stage_weno_kernel, dim=(ny + 2, nx + 6), inputs=[stage["bc"], stage["y_l"], stage["y_r"], nx, ny, characteristic, gamma], device=device)
        wp.launch(y_flux_kernel, dim=(ny + 1, nx), inputs=[stage["fy1"], stage["y_l"], stage["y_r"], tempdy_dt, nx, ny, 1, gamma], device=device)
        wp.launch(y_flux_kernel, dim=(ny + 1, nx), inputs=[stage["fy2"], stage["y_l"], stage["y_r"], tempdy_dt, nx, ny, 2, gamma], device=device)
    else:
        x_stage_kernel = compute_x_stage_weno_mlp_characteristic_kernel if characteristic_weno else compute_x_stage_weno_mlp_kernel
        y_stage_kernel = compute_y_stage_weno_mlp_characteristic_kernel if characteristic_weno else compute_y_stage_weno_mlp_kernel
        x_flux_kernel = compute_x_point_flux_single_mlp_evilin_kernel if riemann_solver_i == 1 else compute_x_point_flux_single_mlp_kernel
        y_flux_kernel = compute_y_point_flux_single_mlp_evilin_kernel if riemann_solver_i == 1 else compute_y_point_flux_single_mlp_kernel
        x_stage_inputs = [
            stage["bc"],
            stage["x_l"],
            stage["x_r"],
            reg_loss,
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
            wp.float64(reg_norm),
            eno_cutoff_i,
        ]
        y_stage_inputs = [
            stage["bc"],
            stage["y_l"],
            stage["y_r"],
            reg_loss,
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
            wp.float64(reg_norm),
            eno_cutoff_i,
        ]
        if characteristic_weno:
            x_stage_inputs.append(gamma)
            y_stage_inputs.append(gamma)
        wp.launch(
            x_stage_kernel,
            dim=(ny + 6, nx + 2),
            inputs=x_stage_inputs,
            device=device,
        )
        wp.launch(
            x_flux_kernel,
            dim=(ny, nx + 1),
            inputs=[
                stage["fx1"],
                stage["x_l"],
                stage["x_r"],
                reg_loss,
                mlp_params["w1"],
                mlp_params["b1"],
                mlp_params["w2"],
                mlp_params["b2"],
                mlp_params["w3"],
                mlp_params["b3"],
                mlp_params["w4"],
                mlp_params["b4"],
                tempdx_dt,
                nx,
                ny,
                1,
                gamma,
                wp.float64(reg_norm),
                eno_cutoff_i,
            ],
            device=device,
        )
        wp.launch(
            x_flux_kernel,
            dim=(ny, nx + 1),
            inputs=[
                stage["fx2"],
                stage["x_l"],
                stage["x_r"],
                reg_loss,
                mlp_params["w1"],
                mlp_params["b1"],
                mlp_params["w2"],
                mlp_params["b2"],
                mlp_params["w3"],
                mlp_params["b3"],
                mlp_params["w4"],
                mlp_params["b4"],
                tempdx_dt,
                nx,
                ny,
                2,
                gamma,
                wp.float64(reg_norm),
                eno_cutoff_i,
            ],
            device=device,
        )

        wp.launch(
            y_stage_kernel,
            dim=(ny + 2, nx + 6),
            inputs=y_stage_inputs,
            device=device,
        )
        wp.launch(
            y_flux_kernel,
            dim=(ny + 1, nx),
            inputs=[
                stage["fy1"],
                stage["y_l"],
                stage["y_r"],
                reg_loss,
                mlp_params["w1"],
                mlp_params["b1"],
                mlp_params["w2"],
                mlp_params["b2"],
                mlp_params["w3"],
                mlp_params["b3"],
                mlp_params["w4"],
                mlp_params["b4"],
                tempdy_dt,
                nx,
                ny,
                1,
                gamma,
                wp.float64(reg_norm),
                eno_cutoff_i,
            ],
            device=device,
        )
        wp.launch(
            y_flux_kernel,
            dim=(ny + 1, nx),
            inputs=[
                stage["fy2"],
                stage["y_l"],
                stage["y_r"],
                reg_loss,
                mlp_params["w1"],
                mlp_params["b1"],
                mlp_params["w2"],
                mlp_params["b2"],
                mlp_params["w3"],
                mlp_params["b3"],
                mlp_params["w4"],
                mlp_params["b4"],
                tempdy_dt,
                nx,
                ny,
                2,
                gamma,
                wp.float64(reg_norm),
                eno_cutoff_i,
            ],
            device=device,
        )

    wp.launch(update_rk3_out_kernel, dim=(ny, nx), inputs=[stage["bc"], u0, u_out, stage["fx1"], stage["fx2"], stage["fy1"], stage["fy2"], nx, ny, gc, rk], device=device)


def launch_weno5_rk3_diff_step(
    arrays: dict[str, object],
    params: wh.Params,
    dt: float,
    device: str,
    characteristic_weno: bool,
    mlp_params: dict[str, object] | None = None,
    eno_cutoff: bool = False,
    boundary: str = "periodic",
    riemann_solver: str | int = "force",
) -> None:
    reg_norm = 1.0 / max(1.0, 3.0 * 32.0 * float(params.nx * params.ny))
    reg_loss = arrays.get("reg_loss")
    launch_diff_stage(arrays["u0"], arrays["u0"], arrays["u1"], arrays["s1"], params, dt, 1, device, characteristic_weno, mlp_params, reg_loss, reg_norm, eno_cutoff, boundary, riemann_solver)
    launch_diff_stage(arrays["u1"], arrays["u0"], arrays["u2"], arrays["s2"], params, dt, 2, device, characteristic_weno, mlp_params, reg_loss, reg_norm, eno_cutoff, boundary, riemann_solver)
    launch_diff_stage(arrays["u2"], arrays["u0"], arrays["u3"], arrays["s3"], params, dt, 3, device, characteristic_weno, mlp_params, reg_loss, reg_norm, eno_cutoff, boundary, riemann_solver)


def allocate_diff_arrays(u0_host: np.ndarray, target_host: np.ndarray, params: wh.Params, device: str) -> dict[str, object]:
    arrays = {
        "u0": wp.array(u0_host, dtype=wp.float64, device=device, requires_grad=True),
        "target": wp.array(target_host, dtype=wp.float64, device=device),
        "u1": zeros_like_state(params, device),
        "u2": zeros_like_state(params, device),
        "u3": zeros_like_state(params, device),
        "data_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "reg_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "lp_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "tv_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "errtv_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "shock_errtv_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "shock_errtv_norm": wp.zeros(1, dtype=wp.float64, device=device),
        "shock_lp_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "shock_lp_norm": wp.zeros(1, dtype=wp.float64, device=device),
        "shock_range_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "shock_range_norm": wp.zeros(1, dtype=wp.float64, device=device),
        "pos_loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "loss": wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True),
        "s1": allocate_stage(params, device, "s1"),
        "s2": allocate_stage(params, device, "s2"),
        "s3": allocate_stage(params, device, "s3"),
    }
    return arrays


def run_demo(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp.init()
    wp.set_device(args.device)

    params = wh.Params(nx=args.nx, ny=args.ny, x_length=2.0, y_length=2.0, cfl=args.cfl, t_end=args.dt)
    u0_host = make_smooth_periodic_state(params, -1.0, -1.0, 0.0)
    target_host = make_smooth_periodic_state(params, -1.0, -1.0, args.dt)
    arrays = allocate_diff_arrays(u0_host, target_host, params, args.device)
    mlp_params = None
    if args.weight_mode == "mlp":
        mlp_params = make_mlp_params(args.device, args.seed)

    with wp.Tape() as tape:
        launch_weno5_rk3_diff_step(arrays, params, args.dt, args.device, args.weno_space == "characteristic", mlp_params, args.eno_cutoff, args.boundary, args.riemann_solver)
        wp.launch(density_mse_loss_kernel, dim=(params.ny, params.nx), inputs=[arrays["u3"], arrays["target"], arrays["data_loss"], params.nx, params.ny, params.ghost], device=args.device)
        wp.launch(
            combine_loss_kernel,
            dim=1,
            inputs=[
                arrays["data_loss"],
                arrays["reg_loss"],
                arrays["lp_loss"],
                arrays["tv_loss"],
                arrays["errtv_loss"],
                arrays["shock_errtv_loss"],
                arrays["shock_errtv_norm"],
                arrays["shock_lp_loss"],
                arrays["shock_lp_norm"],
                arrays["shock_range_loss"],
                arrays["shock_range_norm"],
                arrays["pos_loss"],
                arrays["loss"],
                wp.float64(1.0),
                wp.float64(args.smooth_lambda),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(8.0),
            ],
            device=args.device,
        )
    tape.backward(loss=arrays["loss"])
    wp.synchronize()

    loss_value = float(arrays["loss"].numpy()[0])
    data_loss_value = float(arrays["data_loss"].numpy()[0])
    reg_loss_value = float(arrays["reg_loss"].numpy()[0])
    grad_u0 = tape.gradients[arrays["u0"]].numpy()
    grad_l2 = float(np.sqrt(np.sum(grad_u0 * grad_u0)))
    grad_linf = float(np.max(np.abs(grad_u0)))
    u3_host = arrays["u3"].numpy()
    stats = wh.interior_stats(u3_host, params)

    print(
        f"diff_one_step: nx={params.nx} ny={params.ny} dt={args.dt:.16e} "
        f"weno_space={args.weno_space} weight_mode={args.weight_mode} "
        f"loss={loss_value:.16e} data_loss={data_loss_value:.16e} reg_loss={reg_loss_value:.16e}"
    )
    print(
        f"state: rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
        f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] nan={int(stats['nan_count'])}"
    )
    print(f"grad_u0: l2={grad_l2:.16e} linf={grad_linf:.16e}")
    if mlp_params is not None:
        total_sq = 0.0
        max_abs = 0.0
        for name, arr in mlp_params.items():
            if arr not in tape.gradients:
                continue
            grad = tape.gradients[arr].numpy()
            total_sq += float(np.sum(grad * grad))
            max_abs = max(max_abs, float(np.max(np.abs(grad))))
        print(f"grad_mlp: l2={np.sqrt(total_sq):.16e} linf={max_abs:.16e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weno-space", choices=("conserved", "characteristic"), default="conserved")
    parser.add_argument("--weight-mode", choices=("classical", "mlp"), default="mlp")
    parser.add_argument("--smooth-lambda", type=float, default=1.0e-3)
    parser.add_argument("--eno-cutoff", action="store_true", help="Apply the WENO5-NN ENO cutoff layer during the demo forward pass.")
    parser.add_argument("--boundary", choices=("periodic", "transmissive"), default="periodic")
    parser.add_argument("--riemann-solver", choices=("force", "evilin"), default="force")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
