from __future__ import annotations

import warp as wp

from warp_weno5_3d_rk3 import kernels as B
from weno7_external_clean import warp_weno7_ader4_helpers_classical_only as W7


wp.set_module_options({"fast_math": False, "fuse_fp": False})


Vec5d = B.Vec5d


@wp.func
def weno7_face_vec(
    q0: Vec5d,
    q1: Vec5d,
    q2: Vec5d,
    q3: Vec5d,
    q4: Vec5d,
    q5: Vec5d,
    q6: Vec5d,
    lr: int,
) -> Vec5d:
    value = Vec5d()
    for component in range(5):
        value[component] = W7.weno7_lr_scalar(
            q0[component], q1[component], q2[component], q3[component],
            q4[component], q5[component], q6[component], lr,
            wp.float64(1.0), 0,
        )
    return value


@wp.func
def weno7_gauss_vec(
    q0: Vec5d,
    q1: Vec5d,
    q2: Vec5d,
    q3: Vec5d,
    q4: Vec5d,
    q5: Vec5d,
    q6: Vec5d,
    location: int,
) -> Vec5d:
    value = Vec5d()
    for component in range(5):
        value[component] = W7.weno7_gauss_lr_scalar(
            q0[component], q1[component], q2[component], q3[component],
            q4[component], q5[component], q6[component], location,
            wp.float64(1.0), 0,
        )
    return value


@wp.func
def characteristic_weno7_x(
    q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d,
    q4: Vec5d, q5: Vec5d, q6: Vec5d, lr: int,
) -> Vec5d:
    roe_left = q2
    roe_right = q3
    if lr == 1:
        roe_left = q3
        roe_right = q4
    average = B.roe_average(roe_left, roe_right)
    left_matrix = B.eigen_left_x(average)
    right_matrix = B.eigen_right_x(average)
    c0 = B.matvec5(left_matrix, q0)
    c1 = B.matvec5(left_matrix, q1)
    c2 = B.matvec5(left_matrix, q2)
    c3 = B.matvec5(left_matrix, q3)
    c4 = B.matvec5(left_matrix, q4)
    c5 = B.matvec5(left_matrix, q5)
    c6 = B.matvec5(left_matrix, q6)
    return B.matvec5(right_matrix, weno7_face_vec(c0, c1, c2, c3, c4, c5, c6, lr))


@wp.func
def characteristic_weno7_y(
    q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d,
    q4: Vec5d, q5: Vec5d, q6: Vec5d, lr: int,
) -> Vec5d:
    roe_left = q2
    roe_right = q3
    if lr == 1:
        roe_left = q3
        roe_right = q4
    average = B.roe_average(roe_left, roe_right)
    left_matrix = B.eigen_left_y(average)
    right_matrix = B.eigen_right_y(average)
    c0 = B.matvec5(left_matrix, q0)
    c1 = B.matvec5(left_matrix, q1)
    c2 = B.matvec5(left_matrix, q2)
    c3 = B.matvec5(left_matrix, q3)
    c4 = B.matvec5(left_matrix, q4)
    c5 = B.matvec5(left_matrix, q5)
    c6 = B.matvec5(left_matrix, q6)
    return B.matvec5(right_matrix, weno7_face_vec(c0, c1, c2, c3, c4, c5, c6, lr))


@wp.func
def characteristic_weno7_z(
    q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d,
    q4: Vec5d, q5: Vec5d, q6: Vec5d, lr: int,
) -> Vec5d:
    roe_left = q2
    roe_right = q3
    if lr == 1:
        roe_left = q3
        roe_right = q4
    average = B.roe_average(roe_left, roe_right)
    left_matrix = B.eigen_left_z(average)
    right_matrix = B.eigen_right_z(average)
    c0 = B.matvec5(left_matrix, q0)
    c1 = B.matvec5(left_matrix, q1)
    c2 = B.matvec5(left_matrix, q2)
    c3 = B.matvec5(left_matrix, q3)
    c4 = B.matvec5(left_matrix, q4)
    c5 = B.matvec5(left_matrix, q5)
    c6 = B.matvec5(left_matrix, q6)
    return B.matvec5(right_matrix, weno7_face_vec(c0, c1, c2, c3, c4, c5, c6, lr))


@wp.kernel
def boundary_x_kernel(q: wp.array4d(dtype=wp.float64), nx: int):
    k, j = wp.tid()
    for component in range(5):
        left = q[k, j, 4, component]
        for offset in range(4):
            q[k, j, offset, component] = left
        right = q[k, j, nx + 3, component]
        for offset in range(4):
            q[k, j, nx + 4 + offset, component] = right


@wp.kernel
def boundary_y_kernel(q: wp.array4d(dtype=wp.float64), ny: int):
    k, i = wp.tid()
    for component in range(5):
        bottom = q[k, 4, i, component]
        for offset in range(4):
            q[k, offset, i, component] = bottom
        top = q[k, ny + 3, i, component]
        for offset in range(4):
            q[k, ny + 4 + offset, i, component] = top


@wp.kernel
def boundary_z_kernel(q: wp.array4d(dtype=wp.float64), nz: int):
    j, i = wp.tid()
    for component in range(5):
        front = q[4, j, i, component]
        for offset in range(4):
            q[offset, j, i, component] = front
        back = q[nz + 3, j, i, component]
        for offset in range(4):
            q[nz + 4 + offset, j, i, component] = back


@wp.kernel
def periodic_x_kernel(q: wp.array4d(dtype=wp.float64), nx: int):
    k, j = wp.tid()
    for component in range(5):
        for offset in range(4):
            q[k, j, offset, component] = q[k, j, nx + offset, component]
            q[k, j, nx + 4 + offset, component] = q[k, j, 4 + offset, component]


@wp.kernel
def periodic_y_kernel(q: wp.array4d(dtype=wp.float64), ny: int):
    k, i = wp.tid()
    for component in range(5):
        for offset in range(4):
            q[k, offset, i, component] = q[k, ny + offset, i, component]
            q[k, ny + 4 + offset, i, component] = q[k, 4 + offset, i, component]


@wp.kernel
def periodic_z_kernel(q: wp.array4d(dtype=wp.float64), nz: int):
    j, i = wp.tid()
    for component in range(5):
        for offset in range(4):
            q[offset, j, i, component] = q[nz + offset, j, i, component]
            q[nz + 4 + offset, j, i, component] = q[4 + offset, j, i, component]


@wp.kernel
def normal_x_kernel(
    q: wp.array4d(dtype=wp.float64),
    left: wp.array4d(dtype=wp.float64),
    right: wp.array4d(dtype=wp.float64),
):
    k, j, i = wp.tid()
    q0 = B.load_vec(q, k, j, i)
    q1 = B.load_vec(q, k, j, i + 1)
    q2 = B.load_vec(q, k, j, i + 2)
    q3 = B.load_vec(q, k, j, i + 3)
    q4 = B.load_vec(q, k, j, i + 4)
    q5 = B.load_vec(q, k, j, i + 5)
    q6 = B.load_vec(q, k, j, i + 6)
    left_value = characteristic_weno7_x(q0, q1, q2, q3, q4, q5, q6, 2)
    r0 = B.characteristic_roundtrip_x_lr2(q0, q2, q3)
    r1 = B.characteristic_roundtrip_x_lr2(q1, q2, q3)
    r2 = B.characteristic_roundtrip_x_lr2(q2, q2, q3)
    r3 = B.characteristic_roundtrip_x_lr2(q3, q2, q3)
    r4 = B.characteristic_roundtrip_x_lr2(q4, q2, q3)
    r5 = B.characteristic_roundtrip_x_lr2(q5, q2, q3)
    r6 = B.characteristic_roundtrip_x_lr2(q6, q2, q3)
    right_value = characteristic_weno7_x(r0, r1, r2, r3, r4, r5, r6, 1)
    B.store_vec(left, k, j, i, left_value)
    B.store_vec(right, k, j, i, right_value)


@wp.kernel
def normal_y_kernel(
    q: wp.array4d(dtype=wp.float64),
    left: wp.array4d(dtype=wp.float64),
    right: wp.array4d(dtype=wp.float64),
):
    k, j, i = wp.tid()
    q0 = B.load_vec(q, k, j, i)
    q1 = B.load_vec(q, k, j + 1, i)
    q2 = B.load_vec(q, k, j + 2, i)
    q3 = B.load_vec(q, k, j + 3, i)
    q4 = B.load_vec(q, k, j + 4, i)
    q5 = B.load_vec(q, k, j + 5, i)
    q6 = B.load_vec(q, k, j + 6, i)
    left_value = characteristic_weno7_y(q0, q1, q2, q3, q4, q5, q6, 2)
    r0 = B.characteristic_roundtrip_y_lr2(q0, q2, q3)
    r1 = B.characteristic_roundtrip_y_lr2(q1, q2, q3)
    r2 = B.characteristic_roundtrip_y_lr2(q2, q2, q3)
    r3 = B.characteristic_roundtrip_y_lr2(q3, q2, q3)
    r4 = B.characteristic_roundtrip_y_lr2(q4, q2, q3)
    r5 = B.characteristic_roundtrip_y_lr2(q5, q2, q3)
    r6 = B.characteristic_roundtrip_y_lr2(q6, q2, q3)
    right_value = characteristic_weno7_y(r0, r1, r2, r3, r4, r5, r6, 1)
    B.store_vec(left, k, j, i, left_value)
    B.store_vec(right, k, j, i, right_value)


@wp.kernel
def normal_z_kernel(
    q: wp.array4d(dtype=wp.float64),
    left: wp.array4d(dtype=wp.float64),
    right: wp.array4d(dtype=wp.float64),
):
    k, j, i = wp.tid()
    q0 = B.load_vec(q, k, j, i)
    q1 = B.load_vec(q, k + 1, j, i)
    q2 = B.load_vec(q, k + 2, j, i)
    q3 = B.load_vec(q, k + 3, j, i)
    q4 = B.load_vec(q, k + 4, j, i)
    q5 = B.load_vec(q, k + 5, j, i)
    q6 = B.load_vec(q, k + 6, j, i)
    left_value = characteristic_weno7_z(q0, q1, q2, q3, q4, q5, q6, 2)
    r0 = B.characteristic_roundtrip_z_lr2(q0, q2, q3)
    r1 = B.characteristic_roundtrip_z_lr2(q1, q2, q3)
    r2 = B.characteristic_roundtrip_z_lr2(q2, q2, q3)
    r3 = B.characteristic_roundtrip_z_lr2(q3, q2, q3)
    r4 = B.characteristic_roundtrip_z_lr2(q4, q2, q3)
    r5 = B.characteristic_roundtrip_z_lr2(q5, q2, q3)
    r6 = B.characteristic_roundtrip_z_lr2(q6, q2, q3)
    right_value = characteristic_weno7_z(r0, r1, r2, r3, r4, r5, r6, 1)
    B.store_vec(left, k, j, i, left_value)
    B.store_vec(right, k, j, i, right_value)


@wp.func
def transverse_x_value(source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, location: int) -> Vec5d:
    return weno7_gauss_vec(
        B.load_vec(source, k, j, i + 1), B.load_vec(source, k, j, i + 2),
        B.load_vec(source, k, j, i + 3), B.load_vec(source, k, j, i + 4),
        B.load_vec(source, k, j, i + 5), B.load_vec(source, k, j, i + 6),
        B.load_vec(source, k, j, i + 7), location,
    )


@wp.func
def transverse_y_value(source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, location: int) -> Vec5d:
    return weno7_gauss_vec(
        B.load_vec(source, k, j + 1, i), B.load_vec(source, k, j + 2, i),
        B.load_vec(source, k, j + 3, i), B.load_vec(source, k, j + 4, i),
        B.load_vec(source, k, j + 5, i), B.load_vec(source, k, j + 6, i),
        B.load_vec(source, k, j + 7, i), location,
    )


@wp.func
def transverse_z_value(source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, location: int) -> Vec5d:
    return weno7_gauss_vec(
        B.load_vec(source, k + 1, j, i), B.load_vec(source, k + 2, j, i),
        B.load_vec(source, k + 3, j, i), B.load_vec(source, k + 4, j, i),
        B.load_vec(source, k + 5, j, i), B.load_vec(source, k + 6, j, i),
        B.load_vec(source, k + 7, j, i), location,
    )


@wp.kernel
def transverse_x_kernel(
    source_left: wp.array4d(dtype=wp.float64), source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64), target_right: wp.array4d(dtype=wp.float64),
    location: int,
):
    k, j, i = wp.tid()
    B.store_vec(target_left, k, j, i, transverse_x_value(source_left, k, j, i, location))
    B.store_vec(target_right, k, j, i, transverse_x_value(source_right, k, j, i, location))


@wp.kernel
def transverse_y_kernel(
    source_left: wp.array4d(dtype=wp.float64), source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64), target_right: wp.array4d(dtype=wp.float64),
    location: int,
):
    k, j, i = wp.tid()
    B.store_vec(target_left, k, j, i, transverse_y_value(source_left, k, j, i, location))
    B.store_vec(target_right, k, j, i, transverse_y_value(source_right, k, j, i, location))


@wp.kernel
def transverse_z_kernel(
    source_left: wp.array4d(dtype=wp.float64), source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64), target_right: wp.array4d(dtype=wp.float64),
    location: int,
):
    k, j, i = wp.tid()
    B.store_vec(target_left, k, j, i, transverse_z_value(source_left, k, j, i, location))
    B.store_vec(target_right, k, j, i, transverse_z_value(source_right, k, j, i, location))


@wp.func
def numerical_flux(left_state: Vec5d, right_state: Vec5d, direction: int, dt_dh: wp.float64, reverse: int) -> Vec5d:
    interface_state = B.evilin_state(right_state, left_state, direction, dt_dh)
    if reverse == 1:
        interface_state = B.evilin_state(left_state, right_state, direction, dt_dh)
    return B.flux_from_primitive(B.conserved_to_primitive(interface_state), direction)


@wp.kernel
def flux_x_kernel(
    left_points: wp.array4d(dtype=wp.float64), right_points: wp.array4d(dtype=wp.float64),
    flux: wp.array4d(dtype=wp.float64), dt_dx: wp.float64, flag: int, reverse: int,
):
    k, j, i = wp.tid()
    value = numerical_flux(B.load_vec(left_points, k, j, i + 1), B.load_vec(right_points, k, j, i), 1, dt_dx, reverse)
    for component in range(5):
        if flag == 1:
            flux[k, j, i, component] = value[component]
        else:
            flux[k, j, i, component] = flux[k, j, i, component] + value[component]


@wp.kernel
def flux_y_kernel(
    left_points: wp.array4d(dtype=wp.float64), right_points: wp.array4d(dtype=wp.float64),
    flux: wp.array4d(dtype=wp.float64), dt_dy: wp.float64, flag: int, reverse: int,
):
    k, j, i = wp.tid()
    value = numerical_flux(B.load_vec(left_points, k, j + 1, i), B.load_vec(right_points, k, j, i), 2, dt_dy, reverse)
    for component in range(5):
        if flag == 1:
            flux[k, j, i, component] = value[component]
        else:
            flux[k, j, i, component] = flux[k, j, i, component] + value[component]


@wp.kernel
def flux_z_kernel(
    left_points: wp.array4d(dtype=wp.float64), right_points: wp.array4d(dtype=wp.float64),
    flux: wp.array4d(dtype=wp.float64), dt_dz: wp.float64, flag: int, reverse: int,
):
    k, j, i = wp.tid()
    value = numerical_flux(B.load_vec(left_points, k + 1, j, i), B.load_vec(right_points, k, j, i), 3, dt_dz, reverse)
    for component in range(5):
        if flag == 1:
            flux[k, j, i, component] = value[component]
        else:
            flux[k, j, i, component] = flux[k, j, i, component] + value[component]


@wp.kernel
def rhs_from_flux_kernel(
    rhs: wp.array4d(dtype=wp.float64),
    flux_x: wp.array4d(dtype=wp.float64), flux_y: wp.array4d(dtype=wp.float64),
    flux_z: wp.array4d(dtype=wp.float64), ghost: int,
    inv_dx: wp.float64, inv_dy: wp.float64, inv_dz: wp.float64,
):
    k, j, i = wp.tid()
    kp = k + ghost
    jp = j + ghost
    ip = i + ghost
    for component in range(5):
        rhs[kp, jp, ip, component] = (
            -wp.float64(0.25) * (flux_x[k, j, i + 1, component] - flux_x[k, j, i, component]) * inv_dx
            -wp.float64(0.25) * (flux_y[k, j + 1, i, component] - flux_y[k, j, i, component]) * inv_dy
            -wp.float64(0.25) * (flux_z[k + 1, j, i, component] - flux_z[k, j, i, component]) * inv_dz
        )


@wp.kernel
def rk_stage1_kernel(
    q0: wp.array4d(dtype=wp.float64), rhs0: wp.array4d(dtype=wp.float64),
    q1: wp.array4d(dtype=wp.float64), ghost: int, dt: wp.float64,
):
    k, j, i = wp.tid()
    kp, jp, ip = k + ghost, j + ghost, i + ghost
    for component in range(5):
        q1[kp, jp, ip, component] = q0[kp, jp, ip, component] + wp.float64(0.5) * dt * rhs0[kp, jp, ip, component]


@wp.kernel
def rk_stage2_kernel(
    q0: wp.array4d(dtype=wp.float64), rhs_t0: wp.array4d(dtype=wp.float64),
    q1: wp.array4d(dtype=wp.float64), rhs1: wp.array4d(dtype=wp.float64),
    q2: wp.array4d(dtype=wp.float64), ghost: int, dt: wp.float64,
):
    k, j, i = wp.tid()
    kp, jp, ip = k + ghost, j + ghost, i + ghost
    for component in range(5):
        q2[kp, jp, ip, component] = (
            wp.float64(649.0) / wp.float64(1600.0) * q0[kp, jp, ip, component]
            - wp.float64(10890423.0) / wp.float64(25193600.0) * dt * rhs_t0[kp, jp, ip, component]
            + wp.float64(951.0) / wp.float64(1600.0) * q1[kp, jp, ip, component]
            + wp.float64(5000.0) / wp.float64(7873.0) * dt * rhs1[kp, jp, ip, component]
        )


@wp.kernel
def rk_stage3_kernel(
    q0: wp.array4d(dtype=wp.float64), rhs_t0: wp.array4d(dtype=wp.float64),
    q1: wp.array4d(dtype=wp.float64), rhs_t1: wp.array4d(dtype=wp.float64),
    q2: wp.array4d(dtype=wp.float64), rhs2: wp.array4d(dtype=wp.float64),
    q3: wp.array4d(dtype=wp.float64), ghost: int, dt: wp.float64,
):
    k, j, i = wp.tid()
    kp, jp, ip = k + ghost, j + ghost, i + ghost
    for component in range(5):
        q3[kp, jp, ip, component] = (
            wp.float64(53989.0) / wp.float64(2500000.0) * q0[kp, jp, ip, component]
            - wp.float64(102261.0) / wp.float64(5000000.0) * dt * rhs_t0[kp, jp, ip, component]
            + wp.float64(4806213.0) / wp.float64(20000000.0) * q1[kp, jp, ip, component]
            - wp.float64(5121.0) / wp.float64(20000.0) * dt * rhs_t1[kp, jp, ip, component]
            + wp.float64(23619.0) / wp.float64(32000.0) * q2[kp, jp, ip, component]
            + wp.float64(7873.0) / wp.float64(10000.0) * dt * rhs2[kp, jp, ip, component]
        )


@wp.kernel
def rk_final_kernel(
    q0: wp.array4d(dtype=wp.float64), rhs0: wp.array4d(dtype=wp.float64),
    q1: wp.array4d(dtype=wp.float64), rhs1: wp.array4d(dtype=wp.float64),
    q2: wp.array4d(dtype=wp.float64), q3: wp.array4d(dtype=wp.float64),
    rhs3: wp.array4d(dtype=wp.float64), primitive: wp.array4d(dtype=wp.float64),
    ghost: int, dt: wp.float64,
):
    k, j, i = wp.tid()
    kp, jp, ip = k + ghost, j + ghost, i + ghost
    value = Vec5d()
    for component in range(5):
        value[component] = (
            wp.float64(1.0) / wp.float64(5.0) * q0[kp, jp, ip, component]
            + wp.float64(1.0) / wp.float64(10.0) * dt * rhs0[kp, jp, ip, component]
            + wp.float64(6127.0) / wp.float64(30000.0) * q1[kp, jp, ip, component]
            + wp.float64(1.0) / wp.float64(6.0) * dt * rhs1[kp, jp, ip, component]
            + wp.float64(7873.0) / wp.float64(30000.0) * q2[kp, jp, ip, component]
            + wp.float64(1.0) / wp.float64(3.0) * q3[kp, jp, ip, component]
            + wp.float64(1.0) / wp.float64(6.0) * dt * rhs3[kp, jp, ip, component]
        )
        q0[kp, jp, ip, component] = value[component]
    B.store_vec(primitive, kp, jp, ip, B.conserved_to_primitive(value))


@wp.kernel
def scalar_reconstruction_probe(stencils: wp.array2d(dtype=wp.float64), output: wp.array2d(dtype=wp.float64)):
    row = wp.tid()
    q0 = stencils[row, 0]
    q1 = stencils[row, 1]
    q2 = stencils[row, 2]
    q3 = stencils[row, 3]
    q4 = stencils[row, 4]
    q5 = stencils[row, 5]
    q6 = stencils[row, 6]
    output[row, 0] = W7.weno7_lr_scalar(q0, q1, q2, q3, q4, q5, q6, 1, wp.float64(1.0), 0)
    output[row, 1] = W7.weno7_lr_scalar(q0, q1, q2, q3, q4, q5, q6, 2, wp.float64(1.0), 0)
    output[row, 2] = W7.weno7_gauss_lr_scalar(q0, q1, q2, q3, q4, q5, q6, 1, wp.float64(1.0), 0)
    output[row, 3] = W7.weno7_gauss_lr_scalar(q0, q1, q2, q3, q4, q5, q6, 2, wp.float64(1.0), 0)
