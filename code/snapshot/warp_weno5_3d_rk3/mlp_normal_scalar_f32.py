from __future__ import annotations

import warp as wp

from . import kernels as K
from . import mlp_kernels_f32 as M
from . import mlp_normal_scalar as Base


wp.set_module_options({"fast_math": False, "fuse_fp": False})


@wp.kernel
def reconstruct_characteristic_mlp_f32_kernel(
    q: wp.array4d(dtype=wp.float64),
    characteristic: wp.array4d(dtype=wp.float64),
    lr: int,
    direction: int,
    w1: wp.array3d(dtype=wp.float32), b1: wp.array2d(dtype=wp.float32),
    w2: wp.array3d(dtype=wp.float32), b2: wp.array2d(dtype=wp.float32),
    w3: wp.array3d(dtype=wp.float32), b3: wp.array2d(dtype=wp.float32),
    w4: wp.array3d(dtype=wp.float32), b4: wp.array2d(dtype=wp.float32), eno_cutoff: int,
):
    k, j, i, component = wp.tid()
    stencil = Base.prepare_stencil(Base.load_direction_stencil(q, k, j, i, direction), lr, direction)
    roe_left = Base.stencil_row(stencil, 1)
    roe_right = Base.stencil_row(stencil, 2)
    if lr == 1:
        roe_left = Base.stencil_row(stencil, 2)
        roe_right = Base.stencil_row(stencil, 3)
    left_matrix = Base.left_matrix_for_direction(K.roe_average(roe_left, roe_right), direction)
    c0 = Base.matrix_row_dot(left_matrix, component, Base.stencil_row(stencil, 0))
    c1 = Base.matrix_row_dot(left_matrix, component, Base.stencil_row(stencil, 1))
    c2 = Base.matrix_row_dot(left_matrix, component, Base.stencil_row(stencil, 2))
    c3 = Base.matrix_row_dot(left_matrix, component, Base.stencil_row(stencil, 3))
    c4 = Base.matrix_row_dot(left_matrix, component, Base.stencil_row(stencil, 4))
    characteristic[k, j, i, component] = M.face_scalar(
        c0, c1, c2, c3, c4, lr,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )
