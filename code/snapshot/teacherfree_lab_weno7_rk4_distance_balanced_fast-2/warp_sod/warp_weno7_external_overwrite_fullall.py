#!/usr/bin/env python3
"""Overwrite-only WENO7 external beta helper for full MLP-all mode.

No neural-network inference lives in this Warp module.  Beta arrays are filled
from Python/Torch, then these kernels reconstruct WENO7 values from those beta
arrays.  This keeps the MLP out of Warp CUDA code generation.
"""

from __future__ import annotations

from . import warp_weno7_ader4_helpers_classical_only as wh

wp = wh.wp


if wp is not None:
    NORMAL_BETA_STRIDE = 32
    CROSS_BETA_STRIDE = 256

    @wp.func
    def _beta_index(side: int, comp: int, k: int) -> int:
        return side * 16 + comp * 4 + k

    @wp.func
    def _cross_beta_index(group: int, base: int, side: int, comp: int, k: int) -> int:
        return group * 128 + base * 32 + side * 16 + comp * 4 + k

    @wp.func
    def _load_beta(beta: wp.array3d(dtype=wp.float64), j: int, i: int, side: int, comp: int) -> wp.vec4d:
        return wp.vec4d(
            beta[j, i, _beta_index(side, comp, 0)],
            beta[j, i, _beta_index(side, comp, 1)],
            beta[j, i, _beta_index(side, comp, 2)],
            beta[j, i, _beta_index(side, comp, 3)],
        )

    @wp.func
    def _load_cross_beta(beta: wp.array3d(dtype=wp.float64), j: int, i: int, group: int, base: int, side: int, comp: int) -> wp.vec4d:
        return wp.vec4d(
            beta[j, i, _cross_beta_index(group, base, side, comp, 0)],
            beta[j, i, _cross_beta_index(group, base, side, comp, 1)],
            beta[j, i, _cross_beta_index(group, base, side, comp, 2)],
            beta[j, i, _cross_beta_index(group, base, side, comp, 3)],
        )

    @wp.func
    def _apply_eno_cutoff(ww: wp.vec4d, eno_cutoff: int) -> wp.vec4d:
        out = ww
        if eno_cutoff == 1:
            psi0 = wp.float64(1.0)
            psi1 = wp.float64(1.0)
            psi2 = wp.float64(1.0)
            psi3 = wp.float64(1.0)
            cutoff = wp.float64(4.0e-7)
            if out[0] <= cutoff:
                psi0 = wp.float64(0.0)
            if out[1] <= cutoff:
                psi1 = wp.float64(0.0)
            if out[2] <= cutoff:
                psi2 = wp.float64(0.0)
            if out[3] <= cutoff:
                psi3 = wp.float64(0.0)
            cut_sum = psi0 * out[0] + psi1 * out[1] + psi2 * out[2] + psi3 * out[3]
            inv_cut = wp.float64(1.0) / wp.max(cut_sum, wp.float64(1.0e-300))
            out = wp.vec4d(psi0 * out[0] * inv_cut, psi1 * out[1] * inv_cut, psi2 * out[2] * inv_cut, psi3 * out[3] * inv_cut)
        return out

    @wp.func
    def _weights_from_beta(d0: wp.float64, d1: wp.float64, d2: wp.float64, d3: wp.float64, beta: wp.vec4d, eno_cutoff: int) -> wp.vec4d:
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
        ww = wp.vec4d(a0 / wp.max(asum, wp.float64(1.0e-300)), a1 / wp.max(asum, wp.float64(1.0e-300)), a2 / wp.max(asum, wp.float64(1.0e-300)), a3 / wp.max(asum, wp.float64(1.0e-300)))
        return _apply_eno_cutoff(ww, eno_cutoff)

    @wp.func
    def _weno7_lr_scalar_external_beta(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        q5: wp.float64,
        q6: wp.float64,
        lr: int,
        h: wp.float64,
        dorder: int,
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
            s0 = wh.stencil_weno7(q0, q1, q2, q3, 1, dorder)
            s1 = wh.stencil_weno7(q1, q2, q3, q4, 2, dorder)
            s2 = wh.stencil_weno7(q2, q3, q4, q5, 3, dorder)
            s3 = wh.stencil_weno7(q3, q4, q5, q6, 4, dorder)
            d0 = wp.float64(1.0) / wp.float64(35.0)
            d1 = wp.float64(12.0) / wp.float64(35.0)
            d2 = wp.float64(18.0) / wp.float64(35.0)
            d3 = wp.float64(4.0) / wp.float64(35.0)
        else:
            s0 = wh.stencil_weno7(q0, q1, q2, q3, 2, dorder)
            s1 = wh.stencil_weno7(q1, q2, q3, q4, 3, dorder)
            s2 = wh.stencil_weno7(q2, q3, q4, q5, 4, dorder)
            s3 = wh.stencil_weno7(q3, q4, q5, q6, 5, dorder)
            d0 = wp.float64(4.0) / wp.float64(35.0)
            d1 = wp.float64(18.0) / wp.float64(35.0)
            d2 = wp.float64(12.0) / wp.float64(35.0)
            d3 = wp.float64(1.0) / wp.float64(35.0)
        ww = _weights_from_beta(d0, d1, d2, d3, beta, eno_cutoff)
        return wh.scale_derivative(ww[0] * s0 + ww[1] * s1 + ww[2] * s2 + ww[3] * s3, h, dorder)

    @wp.func
    def _weno7_gauss_lr_scalar_external_beta(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        q5: wp.float64,
        q6: wp.float64,
        lr: int,
        h: wp.float64,
        dorder: int,
        beta: wp.vec4d,
        eno_cutoff: int,
    ) -> wp.float64:
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
            s0 = wh.stencil_weno7_2gauss(q0, q1, q2, q3, 1, dorder)
            s1 = wh.stencil_weno7_2gauss(q1, q2, q3, q4, 2, dorder)
            s2 = wh.stencil_weno7_2gauss(q2, q3, q4, q5, 3, dorder)
            s3 = wh.stencil_weno7_2gauss(q3, q4, q5, q6, 4, dorder)
            d0 = wp.float64(59.0)/wp.float64(880.0) - wp.float64(5.0)*root3/wp.float64(16632.0)
            d1 = wp.float64(381.0)/wp.float64(880.0) - wp.float64(587.0)*root3/wp.float64(194040.0)
            d2 = wp.float64(381.0)/wp.float64(880.0) + wp.float64(587.0)*root3/wp.float64(194040.0)
            d3 = wp.float64(59.0)/wp.float64(880.0) + wp.float64(5.0)*root3/wp.float64(16632.0)
        else:
            s0 = wh.stencil_weno7_2gauss(q0, q1, q2, q3, 5, dorder)
            s1 = wh.stencil_weno7_2gauss(q1, q2, q3, q4, 6, dorder)
            s2 = wh.stencil_weno7_2gauss(q2, q3, q4, q5, 7, dorder)
            s3 = wh.stencil_weno7_2gauss(q3, q4, q5, q6, 8, dorder)
            d0 = wp.float64(59.0)/wp.float64(880.0) + wp.float64(5.0)*root3/wp.float64(16632.0)
            d1 = wp.float64(381.0)/wp.float64(880.0) + wp.float64(587.0)*root3/wp.float64(194040.0)
            d2 = wp.float64(381.0)/wp.float64(880.0) - wp.float64(587.0)*root3/wp.float64(194040.0)
            d3 = wp.float64(59.0)/wp.float64(880.0) - wp.float64(5.0)*root3/wp.float64(16632.0)
        ww = _weights_from_beta(d0, d1, d2, d3, beta, eno_cutoff)
        return wh.scale_derivative(ww[0] * s0 + ww[1] * s1 + ww[2] * s2 + ww[3] * s3, h, dorder)

    @wp.func
    def _weno7_lr_vec_characteristic_external_beta(
        q0: wp.vec4d,
        q1: wp.vec4d,
        q2: wp.vec4d,
        q3: wp.vec4d,
        q4: wp.vec4d,
        q5: wp.vec4d,
        q6: wp.vec4d,
        lr: int,
        h: wp.float64,
        dorder: int,
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
            _weno7_lr_scalar_external_beta(c0[0], c1[0], c2[0], c3[0], c4[0], c5[0], c6[0], lr, h, dorder, _load_beta(beta, j, i, side, 0), eno_cutoff),
            _weno7_lr_scalar_external_beta(c0[1], c1[1], c2[1], c3[1], c4[1], c5[1], c6[1], lr, h, dorder, _load_beta(beta, j, i, side, 1), eno_cutoff),
            _weno7_lr_scalar_external_beta(c0[2], c1[2], c2[2], c3[2], c4[2], c5[2], c6[2], lr, h, dorder, _load_beta(beta, j, i, side, 2), eno_cutoff),
            _weno7_lr_scalar_external_beta(c0[3], c1[3], c2[3], c3[3], c4[3], c5[3], c6[3], lr, h, dorder, _load_beta(beta, j, i, side, 3), eno_cutoff),
        )
        return wh.char_to_con(cf, roe, direction, gamma)

    @wp.func
    def _weno7_gauss_vec_conservative_external_beta(
        q0: wp.vec4d,
        q1: wp.vec4d,
        q2: wp.vec4d,
        q3: wp.vec4d,
        q4: wp.vec4d,
        q5: wp.vec4d,
        q6: wp.vec4d,
        lr: int,
        h: wp.float64,
        dorder: int,
        beta: wp.array3d(dtype=wp.float64),
        j: int,
        i: int,
        group: int,
        base: int,
        eno_cutoff: int,
    ) -> wp.vec4d:
        side = 0
        if lr == 2:
            side = 1
        return wp.vec4d(
            _weno7_gauss_lr_scalar_external_beta(q0[0], q1[0], q2[0], q3[0], q4[0], q5[0], q6[0], lr, h, dorder, _load_cross_beta(beta, j, i, group, base, side, 0), eno_cutoff),
            _weno7_gauss_lr_scalar_external_beta(q0[1], q1[1], q2[1], q3[1], q4[1], q5[1], q6[1], lr, h, dorder, _load_cross_beta(beta, j, i, group, base, side, 1), eno_cutoff),
            _weno7_gauss_lr_scalar_external_beta(q0[2], q1[2], q2[2], q3[2], q4[2], q5[2], q6[2], lr, h, dorder, _load_cross_beta(beta, j, i, group, base, side, 2), eno_cutoff),
            _weno7_gauss_lr_scalar_external_beta(q0[3], q1[3], q2[3], q3[3], q4[3], q5[3], q6[3], lr, h, dorder, _load_cross_beta(beta, j, i, group, base, side, 3), eno_cutoff),
        )

    @wp.kernel
    def overwrite_x_normal_mlp_all_kernel(
        u: wp.array3d(dtype=wp.float64), beta_x: wp.array3d(dtype=wp.float64),
        l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64),
        l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64),
        l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64),
        l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64),
        nx: int, ny: int, dx: wp.float64, gamma: wp.float64, eno_cutoff: int):
        j, i = wp.tid()
        if j < ny + 8 and i < nx + 2:
            q0 = wh.vec_from_array(u, j, i + 0); q1 = wh.vec_from_array(u, j, i + 1); q2 = wh.vec_from_array(u, j, i + 2); q3 = wh.vec_from_array(u, j, i + 3); q4 = wh.vec_from_array(u, j, i + 4); q5 = wh.vec_from_array(u, j, i + 5); q6 = wh.vec_from_array(u, j, i + 6)
            wh.write_vec(l0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dx, 0, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(r0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dx, 0, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(l1, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dx, 1, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(r1, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dx, 1, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(l2, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dx, 2, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(r2, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dx, 2, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(l3, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dx, 3, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(r3, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dx, 3, 1, gamma, beta_x, j, i, eno_cutoff))

    @wp.kernel
    def overwrite_y_normal_mlp_all_kernel(
        u: wp.array3d(dtype=wp.float64), beta_y: wp.array3d(dtype=wp.float64),
        l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64),
        l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64),
        l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64),
        l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64),
        nx: int, ny: int, dy: wp.float64, gamma: wp.float64, eno_cutoff: int):
        j, i = wp.tid()
        if j < ny + 2 and i < nx + 8:
            q0 = wh.vec_from_array(u, j + 0, i); q1 = wh.vec_from_array(u, j + 1, i); q2 = wh.vec_from_array(u, j + 2, i); q3 = wh.vec_from_array(u, j + 3, i); q4 = wh.vec_from_array(u, j + 4, i); q5 = wh.vec_from_array(u, j + 5, i); q6 = wh.vec_from_array(u, j + 6, i)
            wh.write_vec(l0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dy, 0, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(r0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dy, 0, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(l1, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dy, 1, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(r1, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dy, 1, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(l2, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dy, 2, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(r2, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dy, 2, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(l3, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dy, 3, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(r3, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dy, 3, 2, gamma, beta_y, j, i, eno_cutoff))

    @wp.func
    def _cross_x_one_side_external(a0: wp.array3d(dtype=wp.float64), a1: wp.array3d(dtype=wp.float64), a2: wp.array3d(dtype=wp.float64), a3: wp.array3d(dtype=wp.float64), beta: wp.array3d(dtype=wp.float64), group: int, j: int, i: int, lr: int, dy: wp.float64, dt: wp.float64, gamma: wp.float64, loca: int, eno_cutoff: int) -> wp.vec4d:
        q00 = wh.vec_from_array(a0, j + 1, i); q01 = wh.vec_from_array(a0, j + 2, i); q02 = wh.vec_from_array(a0, j + 3, i); q03 = wh.vec_from_array(a0, j + 4, i); q04 = wh.vec_from_array(a0, j + 5, i); q05 = wh.vec_from_array(a0, j + 6, i); q06 = wh.vec_from_array(a0, j + 7, i)
        q10 = wh.vec_from_array(a1, j + 1, i); q11 = wh.vec_from_array(a1, j + 2, i); q12 = wh.vec_from_array(a1, j + 3, i); q13 = wh.vec_from_array(a1, j + 4, i); q14 = wh.vec_from_array(a1, j + 5, i); q15 = wh.vec_from_array(a1, j + 6, i); q16 = wh.vec_from_array(a1, j + 7, i)
        q20 = wh.vec_from_array(a2, j + 1, i); q21 = wh.vec_from_array(a2, j + 2, i); q22 = wh.vec_from_array(a2, j + 3, i); q23 = wh.vec_from_array(a2, j + 4, i); q24 = wh.vec_from_array(a2, j + 5, i); q25 = wh.vec_from_array(a2, j + 6, i); q26 = wh.vec_from_array(a2, j + 7, i)
        q30 = wh.vec_from_array(a3, j + 1, i); q31 = wh.vec_from_array(a3, j + 2, i); q32 = wh.vec_from_array(a3, j + 3, i); q33 = wh.vec_from_array(a3, j + 4, i); q34 = wh.vec_from_array(a3, j + 5, i); q35 = wh.vec_from_array(a3, j + 6, i); q36 = wh.vec_from_array(a3, j + 7, i)
        u = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dy, 0, beta, j, i, group, 0, eno_cutoff)
        uy = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dy, 1, beta, j, i, group, 0, eno_cutoff)
        uyy = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dy, 2, beta, j, i, group, 0, eno_cutoff)
        uyyy = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dy, 3, beta, j, i, group, 0, eno_cutoff)
        ux = _weno7_gauss_vec_conservative_external_beta(q10, q11, q12, q13, q14, q15, q16, lr, dy, 0, beta, j, i, group, 1, eno_cutoff)
        uxy = _weno7_gauss_vec_conservative_external_beta(q10, q11, q12, q13, q14, q15, q16, lr, dy, 1, beta, j, i, group, 1, eno_cutoff)
        uxyy = _weno7_gauss_vec_conservative_external_beta(q10, q11, q12, q13, q14, q15, q16, lr, dy, 2, beta, j, i, group, 1, eno_cutoff)
        uxx = _weno7_gauss_vec_conservative_external_beta(q20, q21, q22, q23, q24, q25, q26, lr, dy, 0, beta, j, i, group, 2, eno_cutoff)
        uxxy = _weno7_gauss_vec_conservative_external_beta(q20, q21, q22, q23, q24, q25, q26, lr, dy, 1, beta, j, i, group, 2, eno_cutoff)
        uxxx = _weno7_gauss_vec_conservative_external_beta(q30, q31, q32, q33, q34, q35, q36, lr, dy, 0, beta, j, i, group, 3, eno_cutoff)
        qt, qtt, qttt = wh.compute_euler_time_derivatives_2d_order3(u, ux, uy, uxx, uxy, uyy, uxxx, uxxy, uxyy, uyyy, gamma)
        return wh.after_dritq_2d(qt, qtt, qttt, u, loca, dt)

    @wp.func
    def _cross_y_one_side_external(a0: wp.array3d(dtype=wp.float64), a1: wp.array3d(dtype=wp.float64), a2: wp.array3d(dtype=wp.float64), a3: wp.array3d(dtype=wp.float64), beta: wp.array3d(dtype=wp.float64), group: int, j: int, i: int, lr: int, dx: wp.float64, dt: wp.float64, gamma: wp.float64, loca: int, eno_cutoff: int) -> wp.vec4d:
        q00 = wh.vec_from_array(a0, j, i + 1); q01 = wh.vec_from_array(a0, j, i + 2); q02 = wh.vec_from_array(a0, j, i + 3); q03 = wh.vec_from_array(a0, j, i + 4); q04 = wh.vec_from_array(a0, j, i + 5); q05 = wh.vec_from_array(a0, j, i + 6); q06 = wh.vec_from_array(a0, j, i + 7)
        q10 = wh.vec_from_array(a1, j, i + 1); q11 = wh.vec_from_array(a1, j, i + 2); q12 = wh.vec_from_array(a1, j, i + 3); q13 = wh.vec_from_array(a1, j, i + 4); q14 = wh.vec_from_array(a1, j, i + 5); q15 = wh.vec_from_array(a1, j, i + 6); q16 = wh.vec_from_array(a1, j, i + 7)
        q20 = wh.vec_from_array(a2, j, i + 1); q21 = wh.vec_from_array(a2, j, i + 2); q22 = wh.vec_from_array(a2, j, i + 3); q23 = wh.vec_from_array(a2, j, i + 4); q24 = wh.vec_from_array(a2, j, i + 5); q25 = wh.vec_from_array(a2, j, i + 6); q26 = wh.vec_from_array(a2, j, i + 7)
        q30 = wh.vec_from_array(a3, j, i + 1); q31 = wh.vec_from_array(a3, j, i + 2); q32 = wh.vec_from_array(a3, j, i + 3); q33 = wh.vec_from_array(a3, j, i + 4); q34 = wh.vec_from_array(a3, j, i + 5); q35 = wh.vec_from_array(a3, j, i + 6); q36 = wh.vec_from_array(a3, j, i + 7)
        u = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dx, 0, beta, j, i, group, 0, eno_cutoff)
        ux = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dx, 1, beta, j, i, group, 0, eno_cutoff)
        uxx = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dx, 2, beta, j, i, group, 0, eno_cutoff)
        uxxx = _weno7_gauss_vec_conservative_external_beta(q00, q01, q02, q03, q04, q05, q06, lr, dx, 3, beta, j, i, group, 0, eno_cutoff)
        uy = _weno7_gauss_vec_conservative_external_beta(q10, q11, q12, q13, q14, q15, q16, lr, dx, 0, beta, j, i, group, 1, eno_cutoff)
        uxy = _weno7_gauss_vec_conservative_external_beta(q10, q11, q12, q13, q14, q15, q16, lr, dx, 1, beta, j, i, group, 1, eno_cutoff)
        uxxy = _weno7_gauss_vec_conservative_external_beta(q10, q11, q12, q13, q14, q15, q16, lr, dx, 2, beta, j, i, group, 1, eno_cutoff)
        uyy = _weno7_gauss_vec_conservative_external_beta(q20, q21, q22, q23, q24, q25, q26, lr, dx, 0, beta, j, i, group, 2, eno_cutoff)
        uxyy = _weno7_gauss_vec_conservative_external_beta(q20, q21, q22, q23, q24, q25, q26, lr, dx, 1, beta, j, i, group, 2, eno_cutoff)
        uyyy = _weno7_gauss_vec_conservative_external_beta(q30, q31, q32, q33, q34, q35, q36, lr, dx, 0, beta, j, i, group, 3, eno_cutoff)
        qt, qtt, qttt = wh.compute_euler_time_derivatives_2d_order3(u, ux, uy, uxx, uxy, uyy, uxxx, uxxy, uxyy, uyyy, gamma)
        return wh.after_dritq_2d(qt, qtt, qttt, u, loca, dt)

    @wp.kernel
    def compute_x_cross_stage_ader4_external_beta_slot_kernel(tl: wp.array3d(dtype=wp.float64), tr: wp.array3d(dtype=wp.float64), l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64), l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64), l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64), l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64), beta: wp.array3d(dtype=wp.float64), nx: int, ny: int, dy: wp.float64, dt: wp.float64, gauss_loca: int, stage_loca: int, gamma: wp.float64, eno_cutoff: int):
        j, i = wp.tid()
        if j < ny and i < nx + 2:
            wh.write_vec(tl, j, i, _cross_x_one_side_external(l0, l1, l2, l3, beta, 0, j, i, gauss_loca, dy, dt, gamma, stage_loca, eno_cutoff))
            wh.write_vec(tr, j, i, _cross_x_one_side_external(r0, r1, r2, r3, beta, 1, j, i, gauss_loca, dy, dt, gamma, stage_loca, eno_cutoff))

    @wp.kernel
    def compute_y_cross_stage_ader4_external_beta_slot_kernel(tl: wp.array3d(dtype=wp.float64), tr: wp.array3d(dtype=wp.float64), l0: wp.array3d(dtype=wp.float64), r0: wp.array3d(dtype=wp.float64), l1: wp.array3d(dtype=wp.float64), r1: wp.array3d(dtype=wp.float64), l2: wp.array3d(dtype=wp.float64), r2: wp.array3d(dtype=wp.float64), l3: wp.array3d(dtype=wp.float64), r3: wp.array3d(dtype=wp.float64), beta: wp.array3d(dtype=wp.float64), nx: int, ny: int, dx: wp.float64, dt: wp.float64, gauss_loca: int, stage_loca: int, gamma: wp.float64, eno_cutoff: int):
        j, i = wp.tid()
        if j < ny + 2 and i < nx:
            wh.write_vec(tl, j, i, _cross_y_one_side_external(l0, l1, l2, l3, beta, 0, j, i, gauss_loca, dx, dt, gamma, stage_loca, eno_cutoff))
            wh.write_vec(tr, j, i, _cross_y_one_side_external(r0, r1, r2, r3, beta, 1, j, i, gauss_loca, dx, dt, gamma, stage_loca, eno_cutoff))


def allocate_external_beta_arrays(nx: int, ny: int, device: str) -> dict[str, object]:
    return {
        "beta_x": wp.zeros((ny + 8, nx + 2, NORMAL_BETA_STRIDE), dtype=wp.float64, device=device),
        "beta_y": wp.zeros((ny + 2, nx + 8, NORMAL_BETA_STRIDE), dtype=wp.float64, device=device),
        "beta_cross_x": wp.zeros((ny, nx + 2, CROSS_BETA_STRIDE), dtype=wp.float64, device=device),
        "beta_cross_y": wp.zeros((ny + 2, nx, CROSS_BETA_STRIDE), dtype=wp.float64, device=device),
    }
