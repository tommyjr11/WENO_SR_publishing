#!/usr/bin/env python3
"""External-beta WENO7 helper.

This experimental path keeps the neural network out of the large WENO7/ADER4
stage kernels.  A small standalone Warp kernel first computes MLP beta values
for the normal characteristic face reconstructions.  The stage kernels then
read those beta values and otherwise use the classical WENO7/ADER4 path.

Only the ``normal`` derivative mode is implemented here.  Cross-stage/Gauss
reconstructions remain classical in this first external-beta version.
"""

from __future__ import annotations

import warp_weno7_ader4_helpers_classical_only as wh

wp = wh.wp


if wp is not None:
    MLP7_INPUTS = 6
    MLP7_HIDDEN1 = 12
    MLP7_HIDDEN2 = 8
    MLP7_HIDDEN3 = 8
    MLP7_OUTPUTS = 4
    Vec6d = wp.types.vector(length=MLP7_INPUTS, dtype=wp.float64)
    Vec8d = wp.types.vector(length=MLP7_HIDDEN2, dtype=wp.float64)
    Vec12d = wp.types.vector(length=MLP7_HIDDEN1, dtype=wp.float64)

    @wp.func
    def swish(x: wp.float64) -> wp.float64:
        return x / (wp.float64(1.0) + wp.exp(-x))


    @wp.func
    def weno7_raw_sensors(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, q5: wp.float64, q6: wp.float64) -> Vec6d:
        d01 = -wp.float64(2.0) * q0 + wp.float64(9.0) * q1 - wp.float64(18.0) * q2 + wp.float64(11.0) * q3
        d02 = -q0 + wp.float64(4.0) * q1 - wp.float64(5.0) * q2 + wp.float64(2.0) * q3
        d03 = -q0 + wp.float64(3.0) * q1 - wp.float64(3.0) * q2 + q3

        d11 = q1 - wp.float64(6.0) * q2 + wp.float64(3.0) * q3 + wp.float64(2.0) * q4
        d12 = q2 - wp.float64(2.0) * q3 + q4
        d13 = -q1 + wp.float64(3.0) * q2 - wp.float64(3.0) * q3 + q4

        d21 = -wp.float64(2.0) * q2 - wp.float64(3.0) * q3 + wp.float64(6.0) * q4 - q5
        d22 = q2 - wp.float64(2.0) * q3 + q4
        d23 = -q2 + wp.float64(3.0) * q3 - wp.float64(3.0) * q4 + q5

        d31 = -wp.float64(11.0) * q3 + wp.float64(18.0) * q4 - wp.float64(9.0) * q5 + wp.float64(2.0) * q6
        d32 = wp.float64(2.0) * q3 - wp.float64(5.0) * q4 + wp.float64(4.0) * q5 - q6
        d33 = -q3 + wp.float64(3.0) * q4 - wp.float64(3.0) * q5 + q6

        c1 = wp.float64(1.0) / wp.float64(36.0)
        c2 = wp.float64(13.0) / wp.float64(12.0)
        c3 = wp.float64(781.0) / wp.float64(720.0)
        delta0 = c1 * wp.abs(d01) + c2 * wp.abs(d02) + c3 * wp.abs(d03)
        delta1 = c1 * wp.abs(d11) + c2 * wp.abs(d12) + c3 * wp.abs(d13)
        delta2 = c1 * wp.abs(d21) + c2 * wp.abs(d22) + c3 * wp.abs(d23)
        delta3 = c1 * wp.abs(d31) + c2 * wp.abs(d32) + c3 * wp.abs(d33)

        eps = wp.float64(1.0e-15)
        dd0 = q0 - wp.float64(2.0) * q1 + q2
        dd1 = q1 - wp.float64(2.0) * q2 + q3
        dd2 = q2 - wp.float64(2.0) * q3 + q4
        dd3 = q3 - wp.float64(2.0) * q4 + q5
        dd4 = q4 - wp.float64(2.0) * q5 + q6
        g0 = wp.abs(dd0) / (wp.abs(q1 - q0) + wp.abs(q2 - q1) + eps)
        g1 = wp.abs(dd1) / (wp.abs(q2 - q1) + wp.abs(q3 - q2) + eps)
        g2 = wp.abs(dd2) / (wp.abs(q3 - q2) + wp.abs(q4 - q3) + eps)
        g3 = wp.abs(dd3) / (wp.abs(q4 - q3) + wp.abs(q5 - q4) + eps)
        g4 = wp.abs(dd4) / (wp.abs(q5 - q4) + wp.abs(q6 - q5) + eps)
        gamma_s = wp.min(wp.float64(1.0), wp.max(wp.max(wp.max(g0, g1), wp.max(g2, g3)), g4))

        out = Vec6d()
        out[0] = delta0
        out[1] = delta1
        out[2] = delta2
        out[3] = delta3
        out[4] = gamma_s
        out[5] = wp.float64(0.0)
        return out


    @wp.func
    def weno7_nn_features(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, q5: wp.float64, q6: wp.float64) -> Vec6d:
        raw = weno7_raw_sensors(q0, q1, q2, q3, q4, q5, q6)
        eps = wp.float64(1.0e-15)
        delta_max = wp.max(wp.max(raw[0], raw[1]), wp.max(raw[2], raw[3]))
        inv_delta_max = wp.float64(1.0) / wp.max(delta_max, eps)
        q_scale = wp.max(wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))), wp.max(wp.max(wp.abs(q4), wp.abs(q5)), wp.abs(q6)))
        q_scale = wp.max(q_scale, wp.float64(1.0))
        relative_scale = wp.max(delta_max / q_scale, wp.float64(1.0e-30))
        log10_relative_scale = wp.log(relative_scale) / wp.log(wp.float64(10.0))
        scale_feature = (log10_relative_scale + wp.float64(16.0)) / wp.float64(16.0)
        scale_feature = wp.min(wp.float64(1.0), wp.max(wp.float64(0.0), scale_feature))
        features = Vec6d()
        features[0] = raw[0] * inv_delta_max
        features[1] = raw[1] * inv_delta_max
        features[2] = raw[2] * inv_delta_max
        features[3] = raw[3] * inv_delta_max
        features[4] = raw[4]
        features[5] = scale_feature
        return features


    @wp.func
    def weno7_plateau_detected(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, q5: wp.float64, q6: wp.float64) -> int:
        raw = weno7_raw_sensors(q0, q1, q2, q3, q4, q5, q6)
        delta_max = wp.max(wp.max(raw[0], raw[1]), wp.max(raw[2], raw[3]))
        q_scale = wp.max(wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))), wp.max(wp.max(wp.abs(q4), wp.abs(q5)), wp.abs(q6)))
        q_scale = wp.max(q_scale, wp.float64(1.0))
        if delta_max <= wp.float64(1.0e-13) * q_scale:
            return 1
        return 0


    @wp.func
    def weno7_mlp_beta(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        q5: wp.float64,
        q6: wp.float64,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
    ) -> wp.vec4d:
        if weno7_plateau_detected(q0, q1, q2, q3, q4, q5, q6) == 1:
            return wp.vec4d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))

        x = weno7_nn_features(q0, q1, q2, q3, q4, q5, q6)
        h1 = Vec12d()
        for o in range(MLP7_HIDDEN1):
            z = b1[0, o]
            for k in range(MLP7_INPUTS):
                z = z + x[k] * w1[0, k, o]
            h1[o] = swish(z)

        h2 = Vec8d()
        for o in range(MLP7_HIDDEN2):
            z = b2[0, o]
            for k in range(MLP7_HIDDEN1):
                z = z + h1[k] * w2[0, k, o]
            h2[o] = swish(z)

        h3 = Vec8d()
        for o in range(MLP7_HIDDEN3):
            z = b3[0, o]
            for k in range(MLP7_HIDDEN2):
                z = z + h2[k] * w3[0, k, o]
            h3[o] = swish(z)

        raw0 = b4[0, 0]
        raw1 = b4[0, 1]
        raw2 = b4[0, 2]
        raw3 = b4[0, 3]
        for k in range(MLP7_HIDDEN3):
            raw0 = raw0 + h3[k] * w4[0, k, 0]
            raw1 = raw1 + h3[k] * w4[0, k, 1]
            raw2 = raw2 + h3[k] * w4[0, k, 2]
            raw3 = raw3 + h3[k] * w4[0, k, 3]

        cap = wp.float64(6.0)
        bad0 = cap * wp.tanh(raw0 / cap)
        bad1 = cap * wp.tanh(raw1 / cap)
        bad2 = cap * wp.tanh(raw2 / cap)
        bad3 = cap * wp.tanh(raw3 / cap)
        bad_max = wp.max(wp.max(bad0, bad1), wp.max(bad2, bad3))
        e0 = wp.exp(bad0 - bad_max)
        e1 = wp.exp(bad1 - bad_max)
        e2 = wp.exp(bad2 - bad_max)
        e3 = wp.exp(bad3 - bad_max)
        inv_esum = wp.float64(1.0) / wp.max(e0 + e1 + e2 + e3, wp.float64(1.0e-300))
        beta0 = wp.float64(4.0) * e0 * inv_esum
        beta1 = wp.float64(4.0) * e1 * inv_esum
        beta2 = wp.float64(4.0) * e2 * inv_esum
        beta3 = wp.float64(4.0) * e3 * inv_esum

        return wp.vec4d(beta0, beta1, beta2, beta3)


    BETA_STRIDE = 32

    @wp.func
    def _beta_index(side: int, comp: int, k: int) -> int:
        return side * 16 + comp * 4 + k

    @wp.func
    def _store_beta(beta: wp.array3d(dtype=wp.float64), j: int, i: int, side: int, comp: int, b: wp.vec4d):
        beta[j, i, _beta_index(side, comp, 0)] = b[0]
        beta[j, i, _beta_index(side, comp, 1)] = b[1]
        beta[j, i, _beta_index(side, comp, 2)] = b[2]
        beta[j, i, _beta_index(side, comp, 3)] = b[3]

    @wp.func
    def _load_beta(beta: wp.array3d(dtype=wp.float64), j: int, i: int, side: int, comp: int) -> wp.vec4d:
        return wp.vec4d(
            beta[j, i, _beta_index(side, comp, 0)],
            beta[j, i, _beta_index(side, comp, 1)],
            beta[j, i, _beta_index(side, comp, 2)],
            beta[j, i, _beta_index(side, comp, 3)],
        )

    @wp.func
    def _mirrored_mlp_beta(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        q5: wp.float64,
        q6: wp.float64,
        lr: int,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
    ) -> wp.vec4d:
        if lr == 1:
            return weno7_mlp_beta(q0, q1, q2, q3, q4, q5, q6, w1, b1, w2, b2, w3, b3, w4, b4)
        br = weno7_mlp_beta(q6, q5, q4, q3, q2, q1, q0, w1, b1, w2, b2, w3, b3, w4, b4)
        return wp.vec4d(br[3], br[2], br[1], br[0])

    @wp.func
    def _store_char_beta_for_lr(
        beta: wp.array3d(dtype=wp.float64),
        j: int,
        i: int,
        side: int,
        lr: int,
        q0: wp.vec4d,
        q1: wp.vec4d,
        q2: wp.vec4d,
        q3: wp.vec4d,
        q4: wp.vec4d,
        q5: wp.vec4d,
        q6: wp.vec4d,
        direction: int,
        gamma: wp.float64,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
    ):
        roe = wh.roe_average_state(q3, q4, gamma)
        if lr == 2:
            roe = wh.roe_average_state(q2, q3, gamma)
        c0 = wh.con_to_char(q0, roe, direction, gamma)
        c1 = wh.con_to_char(q1, roe, direction, gamma)
        c2 = wh.con_to_char(q2, roe, direction, gamma)
        c3 = wh.con_to_char(q3, roe, direction, gamma)
        c4 = wh.con_to_char(q4, roe, direction, gamma)
        c5 = wh.con_to_char(q5, roe, direction, gamma)
        c6 = wh.con_to_char(q6, roe, direction, gamma)
        _store_beta(beta, j, i, side, 0, _mirrored_mlp_beta(c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0], lr, w1, b1, w2, b2, w3, b3, w4, b4))
        _store_beta(beta, j, i, side, 1, _mirrored_mlp_beta(c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1], lr, w1, b1, w2, b2, w3, b3, w4, b4))
        _store_beta(beta, j, i, side, 2, _mirrored_mlp_beta(c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], lr, w1, b1, w2, b2, w3, b3, w4, b4))
        _store_beta(beta, j, i, side, 3, _mirrored_mlp_beta(c0[3], c1[3], c2[3], c3[3], c4[3], c5[3], c6[3], lr, w1, b1, w2, b2, w3, b3, w4, b4))

    @wp.kernel
    def compute_x_normal_mlp_beta_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_x: wp.array3d(dtype=wp.float64),
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
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 8 and i < nx + 2:
            q0 = wh.vec_from_array(u, j, i + 0)
            q1 = wh.vec_from_array(u, j, i + 1)
            q2 = wh.vec_from_array(u, j, i + 2)
            q3 = wh.vec_from_array(u, j, i + 3)
            q4 = wh.vec_from_array(u, j, i + 4)
            q5 = wh.vec_from_array(u, j, i + 5)
            q6 = wh.vec_from_array(u, j, i + 6)
            _store_char_beta_for_lr(beta_x, j, i, 0, 1, q0, q1, q2, q3, q4, q5, q6, 1, gamma, w1, b1, w2, b2, w3, b3, w4, b4)
            _store_char_beta_for_lr(beta_x, j, i, 1, 2, q0, q1, q2, q3, q4, q5, q6, 1, gamma, w1, b1, w2, b2, w3, b3, w4, b4)

    @wp.kernel
    def compute_y_normal_mlp_beta_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_y: wp.array3d(dtype=wp.float64),
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
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 8:
            q0 = wh.vec_from_array(u, j + 0, i)
            q1 = wh.vec_from_array(u, j + 1, i)
            q2 = wh.vec_from_array(u, j + 2, i)
            q3 = wh.vec_from_array(u, j + 3, i)
            q4 = wh.vec_from_array(u, j + 4, i)
            q5 = wh.vec_from_array(u, j + 5, i)
            q6 = wh.vec_from_array(u, j + 6, i)
            _store_char_beta_for_lr(beta_y, j, i, 0, 1, q0, q1, q2, q3, q4, q5, q6, 2, gamma, w1, b1, w2, b2, w3, b3, w4, b4)
            _store_char_beta_for_lr(beta_y, j, i, 1, 2, q0, q1, q2, q3, q4, q5, q6, 2, gamma, w1, b1, w2, b2, w3, b3, w4, b4)

    @wp.func
    def _weno7_lr_scalar_external_beta_value(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        q5: wp.float64,
        q6: wp.float64,
        lr: int,
        beta: wp.vec4d,
        eno_cutoff: int,
    ) -> wp.float64:
        s0 = wp.float64(0.0)
        s1 = wp.float64(0.0)
        s2 = wp.float64(0.0)
        s3 = wp.float64(0.0)
        d0 = wp.float64(0.0)
        d1 = wp.float64(0.0)
        d2 = wp.float64(0.0)
        d3 = wp.float64(0.0)
        if lr == 1:
            s0 = wh.stencil_weno7(q0, q1, q2, q3, 1, 0)
            s1 = wh.stencil_weno7(q1, q2, q3, q4, 2, 0)
            s2 = wh.stencil_weno7(q2, q3, q4, q5, 3, 0)
            s3 = wh.stencil_weno7(q3, q4, q5, q6, 4, 0)
            d0 = wp.float64(1.0) / wp.float64(35.0)
            d1 = wp.float64(12.0) / wp.float64(35.0)
            d2 = wp.float64(18.0) / wp.float64(35.0)
            d3 = wp.float64(4.0) / wp.float64(35.0)
        else:
            s0 = wh.stencil_weno7(q0, q1, q2, q3, 2, 0)
            s1 = wh.stencil_weno7(q1, q2, q3, q4, 3, 0)
            s2 = wh.stencil_weno7(q2, q3, q4, q5, 4, 0)
            s3 = wh.stencil_weno7(q3, q4, q5, q6, 5, 0)
            d0 = wp.float64(4.0) / wp.float64(35.0)
            d1 = wp.float64(18.0) / wp.float64(35.0)
            d2 = wp.float64(12.0) / wp.float64(35.0)
            d3 = wp.float64(1.0) / wp.float64(35.0)
        eps = wp.float64(1.0e-12)
        ib0 = wp.float64(1.0) / (beta[0] + eps)
        ib1 = wp.float64(1.0) / (beta[1] + eps)
        ib2 = wp.float64(1.0) / (beta[2] + eps)
        ib3 = wp.float64(1.0) / (beta[3] + eps)
        a0 = d0 * ib0 * ib0
        a1 = d1 * ib1 * ib1
        a2 = d2 * ib2 * ib2
        a3 = d3 * ib3 * ib3
        asum = a0 + a1 + a2 + a3
        ww0 = a0 / wp.max(asum, wp.float64(1.0e-300))
        ww1 = a1 / wp.max(asum, wp.float64(1.0e-300))
        ww2 = a2 / wp.max(asum, wp.float64(1.0e-300))
        ww3 = a3 / wp.max(asum, wp.float64(1.0e-300))
        if eno_cutoff == 1:
            psi0 = wp.float64(1.0)
            psi1 = wp.float64(1.0)
            psi2 = wp.float64(1.0)
            psi3 = wp.float64(1.0)
            cutoff = wp.float64(4.0e-7)
            if ww0 <= cutoff:
                psi0 = wp.float64(0.0)
            if ww1 <= cutoff:
                psi1 = wp.float64(0.0)
            if ww2 <= cutoff:
                psi2 = wp.float64(0.0)
            if ww3 <= cutoff:
                psi3 = wp.float64(0.0)
            cut_sum = psi0 * ww0 + psi1 * ww1 + psi2 * ww2 + psi3 * ww3
            inv_cut = wp.float64(1.0) / wp.max(cut_sum, wp.float64(1.0e-300))
            ww0 = psi0 * ww0 * inv_cut
            ww1 = psi1 * ww1 * inv_cut
            ww2 = psi2 * ww2 * inv_cut
            ww3 = psi3 * ww3 * inv_cut
        return ww0 * s0 + ww1 * s1 + ww2 * s2 + ww3 * s3

    @wp.func
    def _weno7_lr_vec_characteristic_external_beta_value(
        q0: wp.vec4d,
        q1: wp.vec4d,
        q2: wp.vec4d,
        q3: wp.vec4d,
        q4: wp.vec4d,
        q5: wp.vec4d,
        q6: wp.vec4d,
        lr: int,
        direction: int,
        gamma: wp.float64,
        beta: wp.array3d(dtype=wp.float64),
        j: int,
        i: int,
        eno_cutoff: int,
    ) -> wp.vec4d:
        roe = wh.roe_average_state(q3, q4, gamma)
        if lr == 2:
            roe = wh.roe_average_state(q2, q3, gamma)
        c0 = wh.con_to_char(q0, roe, direction, gamma)
        c1 = wh.con_to_char(q1, roe, direction, gamma)
        c2 = wh.con_to_char(q2, roe, direction, gamma)
        c3 = wh.con_to_char(q3, roe, direction, gamma)
        c4 = wh.con_to_char(q4, roe, direction, gamma)
        c5 = wh.con_to_char(q5, roe, direction, gamma)
        c6 = wh.con_to_char(q6, roe, direction, gamma)
        side = 0
        if lr == 2:
            side = 1
        cf = wp.vec4d(
            _weno7_lr_scalar_external_beta_value(c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0], lr, _load_beta(beta, j, i, side, 0), eno_cutoff),
            _weno7_lr_scalar_external_beta_value(c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1], lr, _load_beta(beta, j, i, side, 1), eno_cutoff),
            _weno7_lr_scalar_external_beta_value(c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], lr, _load_beta(beta, j, i, side, 2), eno_cutoff),
            _weno7_lr_scalar_external_beta_value(c0[3], c1[3], c2[3], c3[3], c4[3], c5[3], c6[3], lr, _load_beta(beta, j, i, side, 3), eno_cutoff),
        )
        return wh.char_to_con(cf, roe, direction, gamma)

    @wp.kernel
    def overwrite_x_normal_mlp_l0r0_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_x: wp.array3d(dtype=wp.float64),
        l0: wp.array3d(dtype=wp.float64),
        r0: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gamma: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 8 and i < nx + 2:
            q0 = wh.vec_from_array(u, j, i + 0)
            q1 = wh.vec_from_array(u, j, i + 1)
            q2 = wh.vec_from_array(u, j, i + 2)
            q3 = wh.vec_from_array(u, j, i + 3)
            q4 = wh.vec_from_array(u, j, i + 4)
            q5 = wh.vec_from_array(u, j, i + 5)
            q6 = wh.vec_from_array(u, j, i + 6)
            wh.write_vec(l0, j, i, _weno7_lr_vec_characteristic_external_beta_value(q0, q1, q2, q3, q4, q5, q6, 2, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(r0, j, i, _weno7_lr_vec_characteristic_external_beta_value(q0, q1, q2, q3, q4, q5, q6, 1, 1, gamma, beta_x, j, i, eno_cutoff))

    @wp.kernel
    def overwrite_y_normal_mlp_l0r0_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_y: wp.array3d(dtype=wp.float64),
        l0: wp.array3d(dtype=wp.float64),
        r0: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gamma: wp.float64,
        eno_cutoff: int,
    ):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 8:
            q0 = wh.vec_from_array(u, j + 0, i)
            q1 = wh.vec_from_array(u, j + 1, i)
            q2 = wh.vec_from_array(u, j + 2, i)
            q3 = wh.vec_from_array(u, j + 3, i)
            q4 = wh.vec_from_array(u, j + 4, i)
            q5 = wh.vec_from_array(u, j + 5, i)
            q6 = wh.vec_from_array(u, j + 6, i)
            wh.write_vec(l0, j, i, _weno7_lr_vec_characteristic_external_beta_value(q0, q1, q2, q3, q4, q5, q6, 2, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(r0, j, i, _weno7_lr_vec_characteristic_external_beta_value(q0, q1, q2, q3, q4, q5, q6, 1, 2, gamma, beta_y, j, i, eno_cutoff))

def allocate_external_beta_arrays(nx: int, ny: int, device: str) -> dict[str, object]:
    return {
        "beta_x": wp.zeros((ny + 8, nx + 2, 32), dtype=wp.float64, device=device),
        "beta_y": wp.zeros((ny + 2, nx + 8, 32), dtype=wp.float64, device=device),
    }


def launch_weno7_ader4_step_external_mlp_normal(
    arrays: dict[str, object],
    params: wh.Params,
    dt: float,
    device: str,
    boundary: str,
    riemann_solver: str,
    mlp_params: dict[str, object],
    eno_cutoff: bool,
) -> None:
    if riemann_solver not in ("evilin", "hllc"):
        raise ValueError("external WENO7 runner supports evilin or hllc")
    nx = params.nx
    ny = params.ny
    gc = params.ghost
    nx_total = nx + 2 * gc
    ny_total = ny + 2 * gc
    tempdx_dt = dt / params.dx
    tempdy_dt = dt / params.dy
    solver_kind = 1 if riemann_solver == "hllc" else 0
    eno_cutoff_i = 1 if eno_cutoff else 0

    boundary_kernel = wh.apply_periodic_boundary_kernel if boundary == "periodic" else wh.apply_boundary_kernel
    wp.launch(boundary_kernel, dim=(ny_total, nx_total), inputs=[arrays["u"], nx, ny, gc], device=device)

    wp.launch(
        wh.compute_x_stage_weno7_big_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[arrays["u"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dx), wp.float64(params.gamma)],
        device=device,
    )
    wp.launch(
        compute_x_normal_mlp_beta_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[
            arrays["u"], arrays["beta_x"],
            mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"],
            mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
            nx, ny, wp.float64(params.gamma),
        ],
        device=device,
    )
    wp.launch(
        overwrite_x_normal_mlp_l0r0_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[arrays["u"], arrays["beta_x"], arrays["l0"], arrays["r0"], nx, ny, wp.float64(params.gamma), eno_cutoff_i],
        device=device,
    )
    for loca in (1, 2):
        wp.launch(
            wh.compute_x_cross_stage_ader4_kernel,
            dim=(ny, nx + 2),
            inputs=[arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dy), wp.float64(dt), loca, wp.float64(params.gamma)],
            device=device,
        )
        wp.launch(
            wh.compute_x_flux_ader4_kernel,
            dim=(ny, nx + 1),
            inputs=[arrays["flux_x"], arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], wp.float64(tempdx_dt), nx, ny, loca, wp.float64(params.gamma), solver_kind],
            device=device,
        )

    wp.launch(
        wh.compute_y_stage_weno7_big_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[arrays["u"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dy), wp.float64(params.gamma)],
        device=device,
    )
    wp.launch(
        compute_y_normal_mlp_beta_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[
            arrays["u"], arrays["beta_y"],
            mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"],
            mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
            nx, ny, wp.float64(params.gamma),
        ],
        device=device,
    )
    wp.launch(
        overwrite_y_normal_mlp_l0r0_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[arrays["u"], arrays["beta_y"], arrays["l0"], arrays["r0"], nx, ny, wp.float64(params.gamma), eno_cutoff_i],
        device=device,
    )
    for loca in (1, 2):
        wp.launch(
            wh.compute_y_cross_stage_ader4_kernel,
            dim=(ny + 2, nx),
            inputs=[arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dx), wp.float64(dt), loca, wp.float64(params.gamma)],
            device=device,
        )
        wp.launch(
            wh.compute_y_flux_ader4_kernel,
            dim=(ny + 1, nx),
            inputs=[arrays["flux_y"], arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], wp.float64(tempdy_dt), nx, ny, loca, wp.float64(params.gamma), solver_kind],
            device=device,
        )

    wp.launch(
        wh.update_ader4_kernel,
        dim=(ny, nx),
        inputs=[arrays["u"], arrays["flux_x"], arrays["flux_y"], arrays["pri"], nx, ny, gc, wp.float64(params.gamma)],
        device=device,
    )
    wp.synchronize()
