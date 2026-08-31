from __future__ import annotations

import warp as wp

from . import mlp_kernels as M


wp.set_module_options({"fast_math": False, "fuse_fp": False})


@wp.kernel
def transverse_mlp_kernel(
    source: wp.array4d(dtype=wp.float64),
    target: wp.array4d(dtype=wp.float64),
    location: int,
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
    q0 = source[k, j, i + 1, component]
    q1 = source[k, j, i + 2, component]
    q2 = source[k, j, i + 3, component]
    q3 = source[k, j, i + 4, component]
    q4 = source[k, j, i + 5, component]
    q5 = source[k, j, i + 6, component]
    q6 = source[k, j, i + 7, component]
    if direction == 2:
        q0 = source[k, j + 1, i, component]
        q1 = source[k, j + 2, i, component]
        q2 = source[k, j + 3, i, component]
        q3 = source[k, j + 4, i, component]
        q4 = source[k, j + 5, i, component]
        q5 = source[k, j + 6, i, component]
        q6 = source[k, j + 7, i, component]
    elif direction == 3:
        q0 = source[k + 1, j, i, component]
        q1 = source[k + 2, j, i, component]
        q2 = source[k + 3, j, i, component]
        q3 = source[k + 4, j, i, component]
        q4 = source[k + 5, j, i, component]
        q5 = source[k + 6, j, i, component]
        q6 = source[k + 7, j, i, component]
    target[k, j, i, component] = M.gauss_scalar(
        q0, q1, q2, q3, q4, q5, q6, location,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )
