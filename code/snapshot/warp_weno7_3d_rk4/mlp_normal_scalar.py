from __future__ import annotations

import warp as wp

from warp_weno5_3d_rk3 import kernels as B

from . import mlp_kernels as M


wp.set_module_options({"fast_math": False, "fuse_fp": False})

Mat75d = wp.types.matrix(shape=(7, 5), dtype=wp.float64)


@wp.func
def load_direction_stencil(
    q: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, direction: int,
) -> Mat75d:
    stencil = Mat75d()
    q0 = B.load_vec(q, k, j, i)
    q1 = B.load_vec(q, k, j, i + 1)
    q2 = B.load_vec(q, k, j, i + 2)
    q3 = B.load_vec(q, k, j, i + 3)
    q4 = B.load_vec(q, k, j, i + 4)
    q5 = B.load_vec(q, k, j, i + 5)
    q6 = B.load_vec(q, k, j, i + 6)
    if direction == 2:
        q1 = B.load_vec(q, k, j + 1, i)
        q2 = B.load_vec(q, k, j + 2, i)
        q3 = B.load_vec(q, k, j + 3, i)
        q4 = B.load_vec(q, k, j + 4, i)
        q5 = B.load_vec(q, k, j + 5, i)
        q6 = B.load_vec(q, k, j + 6, i)
    elif direction == 3:
        q1 = B.load_vec(q, k + 1, j, i)
        q2 = B.load_vec(q, k + 2, j, i)
        q3 = B.load_vec(q, k + 3, j, i)
        q4 = B.load_vec(q, k + 4, j, i)
        q5 = B.load_vec(q, k + 5, j, i)
        q6 = B.load_vec(q, k + 6, j, i)
    for component in range(5):
        stencil[0, component] = q0[component]
        stencil[1, component] = q1[component]
        stencil[2, component] = q2[component]
        stencil[3, component] = q3[component]
        stencil[4, component] = q4[component]
        stencil[5, component] = q5[component]
        stencil[6, component] = q6[component]
    return stencil


@wp.func
def stencil_row(stencil: Mat75d, point: int) -> B.Vec5d:
    value = B.Vec5d()
    for component in range(5):
        value[component] = stencil[point, component]
    return value


@wp.func
def left_matrix_for_direction(average: B.Vec5d, direction: int) -> B.Mat55d:
    matrix = B.eigen_left_z(average)
    if direction == 1:
        matrix = B.eigen_left_x(average)
    elif direction == 2:
        matrix = B.eigen_left_y(average)
    return matrix


@wp.func
def right_matrix_for_direction(average: B.Vec5d, direction: int) -> B.Mat55d:
    matrix = B.eigen_right_z(average)
    if direction == 1:
        matrix = B.eigen_right_x(average)
    elif direction == 2:
        matrix = B.eigen_right_y(average)
    return matrix


@wp.func
def roundtrip_for_direction(
    value: B.Vec5d, left: B.Vec5d, right: B.Vec5d, direction: int,
) -> B.Vec5d:
    result = B.characteristic_roundtrip_z_lr2(value, left, right)
    if direction == 1:
        result = B.characteristic_roundtrip_x_lr2(value, left, right)
    elif direction == 2:
        result = B.characteristic_roundtrip_y_lr2(value, left, right)
    return result


@wp.func
def matrix_row_dot(matrix: B.Mat55d, row: int, value: B.Vec5d) -> wp.float64:
    total = wp.float64(0.0)
    for column in range(5):
        total = total + matrix[row, column] * value[column]
    return total


@wp.func
def prepare_stencil(stencil_in: Mat75d, lr: int, direction: int) -> Mat75d:
    stencil = stencil_in
    if lr == 1:
        roe_left = stencil_row(stencil_in, 2)
        roe_right = stencil_row(stencil_in, 3)
        for point in range(7):
            value = roundtrip_for_direction(
                stencil_row(stencil_in, point), roe_left, roe_right, direction
            )
            for component in range(5):
                stencil[point, component] = value[component]
    return stencil


@wp.kernel
def reconstruct_characteristic_mlp_kernel(
    q: wp.array4d(dtype=wp.float64),
    characteristic: wp.array4d(dtype=wp.float64),
    lr: int,
    direction: int,
    w1: wp.array3d(dtype=wp.float64),
    b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64),
    b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64),
    b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64),
    b4: wp.array2d(dtype=wp.float64),
    eno_cutoff: int,
):
    k, j, i, component = wp.tid()
    stencil = prepare_stencil(load_direction_stencil(q, k, j, i, direction), lr, direction)
    roe_left = stencil_row(stencil, 2)
    roe_right = stencil_row(stencil, 3)
    if lr == 1:
        roe_left = stencil_row(stencil, 3)
        roe_right = stencil_row(stencil, 4)
    left_matrix = left_matrix_for_direction(B.roe_average(roe_left, roe_right), direction)
    c0 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 0))
    c1 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 1))
    c2 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 2))
    c3 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 3))
    c4 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 4))
    c5 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 5))
    c6 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 6))
    characteristic[k, j, i, component] = M.face_scalar(
        c0, c1, c2, c3, c4, c5, c6, lr,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )


@wp.kernel
def characteristic_to_conserved_kernel(
    q: wp.array4d(dtype=wp.float64),
    characteristic: wp.array4d(dtype=wp.float64),
    target: wp.array4d(dtype=wp.float64),
    lr: int,
    direction: int,
):
    k, j, i, component = wp.tid()
    stencil = prepare_stencil(load_direction_stencil(q, k, j, i, direction), lr, direction)
    roe_left = stencil_row(stencil, 2)
    roe_right = stencil_row(stencil, 3)
    if lr == 1:
        roe_left = stencil_row(stencil, 3)
        roe_right = stencil_row(stencil, 4)
    right_matrix = right_matrix_for_direction(B.roe_average(roe_left, roe_right), direction)
    value = B.load_vec(characteristic, k, j, i)
    target[k, j, i, component] = matrix_row_dot(right_matrix, component, value)
