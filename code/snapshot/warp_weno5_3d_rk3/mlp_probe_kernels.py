from __future__ import annotations

import warp as wp

from . import mlp_kernels as M


wp.set_module_options({"fast_math": False, "fuse_fp": False})


@wp.kernel
def scalar_reconstruction_probe_kernel(
    stencils: wp.array2d(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64),
):
    index = wp.tid()
    q0 = stencils[index, 0]
    q1 = stencils[index, 1]
    q2 = stencils[index, 2]
    q3 = stencils[index, 3]
    q4 = stencils[index, 4]
    output[index, 0] = M.face_scalar(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, 0)
    output[index, 1] = M.face_scalar(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, 0)
    output[index, 2] = M.gauss_scalar(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, 0)
    output[index, 3] = M.gauss_scalar(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, 0)

