from __future__ import annotations

import warp as wp

from . import kernels as K
from . import mlp_kernels as M


wp.set_module_options({"fast_math": False, "fuse_fp": False})


@wp.func
def load_direction_stencil(
    q: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, direction: int,
) -> K.Mat55d:
    stencil = K.Mat55d()
    q0 = K.load_vec(q, k, j, i)
    q1 = K.load_vec(q, k, j, i + 1)
    q2 = K.load_vec(q, k, j, i + 2)
    q3 = K.load_vec(q, k, j, i + 3)
    q4 = K.load_vec(q, k, j, i + 4)
    if direction == 2:
        q1 = K.load_vec(q, k, j + 1, i)
        q2 = K.load_vec(q, k, j + 2, i)
        q3 = K.load_vec(q, k, j + 3, i)
        q4 = K.load_vec(q, k, j + 4, i)
    elif direction == 3:
        q1 = K.load_vec(q, k + 1, j, i)
        q2 = K.load_vec(q, k + 2, j, i)
        q3 = K.load_vec(q, k + 3, j, i)
        q4 = K.load_vec(q, k + 4, j, i)
    for component in range(5):
        stencil[0, component] = q0[component]
        stencil[1, component] = q1[component]
        stencil[2, component] = q2[component]
        stencil[3, component] = q3[component]
        stencil[4, component] = q4[component]
    return stencil


@wp.func
def stencil_row(stencil: K.Mat55d, point: int) -> K.Vec5d:
    value = K.Vec5d()
    for component in range(5):
        value[component] = stencil[point, component]
    return value


@wp.func
def left_matrix_for_direction(average: K.Vec5d, direction: int) -> K.Mat55d:
    matrix = K.eigen_left_z(average)
    if direction == 1:
        matrix = K.eigen_left_x(average)
    elif direction == 2:
        matrix = K.eigen_left_y(average)
    return matrix


@wp.func
def right_matrix_for_direction(average: K.Vec5d, direction: int) -> K.Mat55d:
    matrix = K.eigen_right_z(average)
    if direction == 1:
        matrix = K.eigen_right_x(average)
    elif direction == 2:
        matrix = K.eigen_right_y(average)
    return matrix


@wp.func
def roundtrip_for_direction(value: K.Vec5d, left: K.Vec5d, right: K.Vec5d, direction: int) -> K.Vec5d:
    result = K.characteristic_roundtrip_z_lr2(value, left, right)
    if direction == 1:
        result = K.characteristic_roundtrip_x_lr2(value, left, right)
    elif direction == 2:
        result = K.characteristic_roundtrip_y_lr2(value, left, right)
    return result


@wp.func
def matrix_row_dot(matrix: K.Mat55d, row: int, value: K.Vec5d) -> wp.float64:
    total = wp.float64(0.0)
    for column in range(5):
        total = total + matrix[row, column] * value[column]
    return total


@wp.func
def prepare_stencil(stencil_in: K.Mat55d, lr: int, direction: int) -> K.Mat55d:
    stencil = stencil_in
    if lr == 1:
        roe_left = stencil_row(stencil_in, 1)
        roe_right = stencil_row(stencil_in, 2)
        for point in range(5):
            value = roundtrip_for_direction(stencil_row(stencil_in, point), roe_left, roe_right, direction)
            for component in range(5):
                stencil[point, component] = value[component]
    return stencil


@wp.kernel
def reconstruct_characteristic_mlp_kernel(
    q: wp.array4d(dtype=wp.float64),
    characteristic: wp.array4d(dtype=wp.float64),
    lr: int,
    direction: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
):
    k, j, i, component = wp.tid()
    stencil = prepare_stencil(load_direction_stencil(q, k, j, i, direction), lr, direction)
    roe_left = stencil_row(stencil, 1)
    roe_right = stencil_row(stencil, 2)
    if lr == 1:
        roe_left = stencil_row(stencil, 2)
        roe_right = stencil_row(stencil, 3)
    left_matrix = left_matrix_for_direction(K.roe_average(roe_left, roe_right), direction)
    c0 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 0))
    c1 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 1))
    c2 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 2))
    c3 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 3))
    c4 = matrix_row_dot(left_matrix, component, stencil_row(stencil, 4))
    characteristic[k, j, i, component] = M.face_scalar(
        c0, c1, c2, c3, c4, lr,
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
    roe_left = stencil_row(stencil, 1)
    roe_right = stencil_row(stencil, 2)
    if lr == 1:
        roe_left = stencil_row(stencil, 2)
        roe_right = stencil_row(stencil, 3)
    right_matrix = right_matrix_for_direction(K.roe_average(roe_left, roe_right), direction)
    value = K.load_vec(characteristic, k, j, i)
    target[k, j, i, component] = matrix_row_dot(right_matrix, component, value)
