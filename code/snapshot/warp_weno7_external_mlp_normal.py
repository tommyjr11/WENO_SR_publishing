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

import warp_weno7_ader4_helpers as wh

wp = wh.wp


if wp is not None:
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
            return wh.weno7_mlp_beta(q0, q1, q2, q3, q4, q5, q6, w1, b1, w2, b2, w3, b3, w4, b4)
        br = wh.weno7_mlp_beta(q6, q5, q4, q3, q2, q1, q0, w1, b1, w2, b2, w3, b3, w4, b4)
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
        return wh.scale_derivative(ww0 * s0 + ww1 * s1 + ww2 * s2 + ww3 * s3, h, dorder)

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

    @wp.kernel
    def compute_x_stage_weno7_external_beta_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_x: wp.array3d(dtype=wp.float64),
        l0: wp.array3d(dtype=wp.float64),
        r0: wp.array3d(dtype=wp.float64),
        l1: wp.array3d(dtype=wp.float64),
        r1: wp.array3d(dtype=wp.float64),
        l2: wp.array3d(dtype=wp.float64),
        r2: wp.array3d(dtype=wp.float64),
        l3: wp.array3d(dtype=wp.float64),
        r3: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        dx: wp.float64,
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
            wh.write_vec(l0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dx, 0, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(r0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dx, 0, 1, gamma, beta_x, j, i, eno_cutoff))
            wh.write_vec(l1, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 1, 1, gamma))
            wh.write_vec(r1, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 1, 1, gamma))
            wh.write_vec(l2, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 2, 1, gamma))
            wh.write_vec(r2, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 2, 1, gamma))
            wh.write_vec(l3, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dx, 3, 1, gamma))
            wh.write_vec(r3, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dx, 3, 1, gamma))

    @wp.kernel
    def compute_y_stage_weno7_external_beta_kernel(
        u: wp.array3d(dtype=wp.float64),
        beta_y: wp.array3d(dtype=wp.float64),
        l0: wp.array3d(dtype=wp.float64),
        r0: wp.array3d(dtype=wp.float64),
        l1: wp.array3d(dtype=wp.float64),
        r1: wp.array3d(dtype=wp.float64),
        l2: wp.array3d(dtype=wp.float64),
        r2: wp.array3d(dtype=wp.float64),
        l3: wp.array3d(dtype=wp.float64),
        r3: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        dy: wp.float64,
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
            wh.write_vec(l0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 2, dy, 0, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(r0, j, i, _weno7_lr_vec_characteristic_external_beta(q0, q1, q2, q3, q4, q5, q6, 1, dy, 0, 2, gamma, beta_y, j, i, eno_cutoff))
            wh.write_vec(l1, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 1, 2, gamma))
            wh.write_vec(r1, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 1, 2, gamma))
            wh.write_vec(l2, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 2, 2, gamma))
            wh.write_vec(r2, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 2, 2, gamma))
            wh.write_vec(l3, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 2, dy, 3, 2, gamma))
            wh.write_vec(r3, j, i, wh.weno7_lr_vec_characteristic(q0, q1, q2, q3, q4, q5, q6, 1, dy, 3, 2, gamma))


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
        compute_x_normal_mlp_beta_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[
            arrays["u"],
            arrays["beta_x"],
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
            wp.float64(params.gamma),
        ],
        device=device,
    )
    wp.launch(
        compute_x_stage_weno7_external_beta_kernel,
        dim=(ny + 8, nx + 2),
        inputs=[
            arrays["u"],
            arrays["beta_x"],
            arrays["l0"],
            arrays["r0"],
            arrays["l1"],
            arrays["r1"],
            arrays["l2"],
            arrays["r2"],
            arrays["l3"],
            arrays["r3"],
            nx,
            ny,
            wp.float64(params.dx),
            wp.float64(params.gamma),
            eno_cutoff_i,
        ],
        device=device,
    )
    for loca in (1, 2):
        wp.launch(
            wh.compute_x_cross_stage_ader4_kernel,
            dim=(ny, nx + 2),
            inputs=[
                arrays["tl1"],
                arrays["tl2"],
                arrays["tl3"],
                arrays["tr1"],
                arrays["tr2"],
                arrays["tr3"],
                arrays["l0"],
                arrays["r0"],
                arrays["l1"],
                arrays["r1"],
                arrays["l2"],
                arrays["r2"],
                arrays["l3"],
                arrays["r3"],
                nx,
                ny,
                wp.float64(params.dy),
                wp.float64(dt),
                loca,
                wp.float64(params.gamma),
            ],
            device=device,
        )
        wp.launch(
            wh.compute_x_flux_ader4_kernel,
            dim=(ny, nx + 1),
            inputs=[
                arrays["flux_x"],
                arrays["tl1"],
                arrays["tl2"],
                arrays["tl3"],
                arrays["tr1"],
                arrays["tr2"],
                arrays["tr3"],
                wp.float64(tempdx_dt),
                nx,
                ny,
                loca,
                wp.float64(params.gamma),
                solver_kind,
            ],
            device=device,
        )

    wp.launch(
        compute_y_normal_mlp_beta_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[
            arrays["u"],
            arrays["beta_y"],
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
            wp.float64(params.gamma),
        ],
        device=device,
    )
    wp.launch(
        compute_y_stage_weno7_external_beta_kernel,
        dim=(ny + 2, nx + 8),
        inputs=[
            arrays["u"],
            arrays["beta_y"],
            arrays["l0"],
            arrays["r0"],
            arrays["l1"],
            arrays["r1"],
            arrays["l2"],
            arrays["r2"],
            arrays["l3"],
            arrays["r3"],
            nx,
            ny,
            wp.float64(params.dy),
            wp.float64(params.gamma),
            eno_cutoff_i,
        ],
        device=device,
    )
    for loca in (1, 2):
        wp.launch(
            wh.compute_y_cross_stage_ader4_kernel,
            dim=(ny + 2, nx),
            inputs=[
                arrays["tl1"],
                arrays["tl2"],
                arrays["tl3"],
                arrays["tr1"],
                arrays["tr2"],
                arrays["tr3"],
                arrays["l0"],
                arrays["r0"],
                arrays["l1"],
                arrays["r1"],
                arrays["l2"],
                arrays["r2"],
                arrays["l3"],
                arrays["r3"],
                nx,
                ny,
                wp.float64(params.dx),
                wp.float64(dt),
                loca,
                wp.float64(params.gamma),
            ],
            device=device,
        )
        wp.launch(
            wh.compute_y_flux_ader4_kernel,
            dim=(ny + 1, nx),
            inputs=[
                arrays["flux_y"],
                arrays["tl1"],
                arrays["tl2"],
                arrays["tl3"],
                arrays["tr1"],
                arrays["tr2"],
                arrays["tr3"],
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
        wh.update_ader4_kernel,
        dim=(ny, nx),
        inputs=[arrays["u"], arrays["flux_x"], arrays["flux_y"], arrays["pri"], nx, ny, gc, wp.float64(params.gamma)],
        device=device,
    )
    wp.synchronize()
