from __future__ import annotations

import warp as wp

from weno7_external_clean import warp_weno7_ader4_helpers_classical_only as W7


wp.set_module_options({"fast_math": False, "fuse_fp": False})

MLP_INPUTS = 6
MLP_HIDDEN1 = 24
MLP_HIDDEN2 = 16
MLP_HIDDEN3 = 16

Features6d = wp.types.vector(length=MLP_INPUTS, dtype=wp.float64)
Hidden24d = wp.types.vector(length=MLP_HIDDEN1, dtype=wp.float64)
Hidden16d = wp.types.vector(length=MLP_HIDDEN2, dtype=wp.float64)


@wp.func
def swish(value: wp.float64) -> wp.float64:
    return value / (wp.float64(1.0) + wp.exp(-value))


@wp.func
def raw_sensors(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    q5: wp.float64,
    q6: wp.float64,
) -> Features6d:
    f00 = -wp.float64(2.0) * q0 + wp.float64(9.0) * q1 - wp.float64(18.0) * q2 + wp.float64(11.0) * q3
    f01 = -q0 + wp.float64(4.0) * q1 - wp.float64(5.0) * q2 + wp.float64(2.0) * q3
    f02 = -q0 + wp.float64(3.0) * q1 - wp.float64(3.0) * q2 + q3

    f10 = q1 - wp.float64(6.0) * q2 + wp.float64(3.0) * q3 + wp.float64(2.0) * q4
    f11 = q2 - wp.float64(2.0) * q3 + q4
    f12 = -q1 + wp.float64(3.0) * q2 - wp.float64(3.0) * q3 + q4

    f20 = -wp.float64(2.0) * q2 - wp.float64(3.0) * q3 + wp.float64(6.0) * q4 - q5
    f21 = q2 - wp.float64(2.0) * q3 + q4
    f22 = -q2 + wp.float64(3.0) * q3 - wp.float64(3.0) * q4 + q5

    f30 = -wp.float64(11.0) * q3 + wp.float64(18.0) * q4 - wp.float64(9.0) * q5 + wp.float64(2.0) * q6
    f31 = wp.float64(2.0) * q3 - wp.float64(5.0) * q4 + wp.float64(4.0) * q5 - q6
    f32 = -q3 + wp.float64(3.0) * q4 - wp.float64(3.0) * q5 + q6

    c0 = wp.float64(1.0) / wp.float64(36.0)
    c1 = wp.float64(13.0) / wp.float64(12.0)
    c2 = wp.float64(781.0) / wp.float64(720.0)
    delta0 = c0 * wp.abs(f00) + c1 * wp.abs(f01) + c2 * wp.abs(f02)
    delta1 = c0 * wp.abs(f10) + c1 * wp.abs(f11) + c2 * wp.abs(f12)
    delta2 = c0 * wp.abs(f20) + c1 * wp.abs(f21) + c2 * wp.abs(f22)
    delta3 = c0 * wp.abs(f30) + c1 * wp.abs(f31) + c2 * wp.abs(f32)

    eps = wp.float64(1.0e-15)
    d20 = q0 - wp.float64(2.0) * q1 + q2
    d21 = q1 - wp.float64(2.0) * q2 + q3
    d22 = q2 - wp.float64(2.0) * q3 + q4
    d23 = q3 - wp.float64(2.0) * q4 + q5
    d24 = q4 - wp.float64(2.0) * q5 + q6
    gamma0 = wp.abs(d20) / (wp.abs(q1 - q0) + wp.abs(q2 - q1) + eps)
    gamma1 = wp.abs(d21) / (wp.abs(q2 - q1) + wp.abs(q3 - q2) + eps)
    gamma2 = wp.abs(d22) / (wp.abs(q3 - q2) + wp.abs(q4 - q3) + eps)
    gamma3 = wp.abs(d23) / (wp.abs(q4 - q3) + wp.abs(q5 - q4) + eps)
    gamma4 = wp.abs(d24) / (wp.abs(q5 - q4) + wp.abs(q6 - q5) + eps)
    gamma_s = wp.min(
        wp.float64(1.0),
        wp.max(wp.max(wp.max(gamma0, gamma1), wp.max(gamma2, gamma3)), gamma4),
    )

    result = Features6d()
    result[0] = delta0
    result[1] = delta1
    result[2] = delta2
    result[3] = delta3
    result[4] = gamma_s
    result[5] = wp.float64(0.0)
    return result


@wp.func
def nn_features(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    q5: wp.float64,
    q6: wp.float64,
) -> Features6d:
    raw = raw_sensors(q0, q1, q2, q3, q4, q5, q6)
    delta_max = wp.max(wp.max(raw[0], raw[1]), wp.max(raw[2], raw[3]))
    inverse_delta = wp.float64(1.0) / wp.max(delta_max, wp.float64(1.0e-15))
    q_scale = wp.max(
        wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))),
        wp.max(wp.max(wp.abs(q4), wp.abs(q5)), wp.abs(q6)),
    )
    q_scale = wp.max(q_scale, wp.float64(1.0))
    relative_scale = wp.max(delta_max / q_scale, wp.float64(1.0e-30))
    log10_scale = wp.log(relative_scale) / wp.log(wp.float64(10.0))
    scale_feature = (log10_scale + wp.float64(16.0)) / wp.float64(16.0)
    scale_feature = wp.min(wp.float64(1.0), wp.max(wp.float64(0.0), scale_feature))
    features = Features6d()
    features[0] = raw[0] * inverse_delta
    features[1] = raw[1] * inverse_delta
    features[2] = raw[2] * inverse_delta
    features[3] = raw[3] * inverse_delta
    features[4] = raw[4]
    features[5] = scale_feature
    return features


@wp.func
def plateau_detected(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    q5: wp.float64,
    q6: wp.float64,
) -> int:
    raw = raw_sensors(q0, q1, q2, q3, q4, q5, q6)
    delta_max = wp.max(wp.max(raw[0], raw[1]), wp.max(raw[2], raw[3]))
    q_scale = wp.max(
        wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))),
        wp.max(wp.max(wp.abs(q4), wp.abs(q5)), wp.abs(q6)),
    )
    q_scale = wp.max(q_scale, wp.float64(1.0))
    if delta_max <= wp.float64(1.0e-13) * q_scale:
        return 1
    return 0


@wp.func
def raw_badness_ratios(
    features: Features6d,
    w1: wp.array3d(dtype=wp.float64),
    b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64),
    b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64),
    b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64),
    b4: wp.array2d(dtype=wp.float64),
) -> wp.vec4d:
    hidden1 = Hidden24d()
    for output in range(MLP_HIDDEN1):
        value = b1[0, output]
        for feature in range(MLP_INPUTS):
            value = value + features[feature] * w1[0, feature, output]
        hidden1[output] = swish(value)
    hidden2 = Hidden16d()
    for output in range(MLP_HIDDEN2):
        value = b2[0, output]
        for feature in range(MLP_HIDDEN1):
            value = value + hidden1[feature] * w2[0, feature, output]
        hidden2[output] = swish(value)
    hidden3 = Hidden16d()
    for output in range(MLP_HIDDEN3):
        value = b3[0, output]
        for feature in range(MLP_HIDDEN2):
            value = value + hidden2[feature] * w3[0, feature, output]
        hidden3[output] = swish(value)

    raw0 = b4[0, 0]
    raw1 = b4[0, 1]
    raw2 = b4[0, 2]
    raw3 = b4[0, 3]
    for feature in range(MLP_HIDDEN3):
        raw0 = raw0 + hidden3[feature] * w4[0, feature, 0]
        raw1 = raw1 + hidden3[feature] * w4[0, feature, 1]
        raw2 = raw2 + hidden3[feature] * w4[0, feature, 2]
        raw3 = raw3 + hidden3[feature] * w4[0, feature, 3]

    cap = wp.float64(6.0)
    bad0 = cap * wp.tanh(raw0 / cap)
    bad1 = cap * wp.tanh(raw1 / cap)
    bad2 = cap * wp.tanh(raw2 / cap)
    bad3 = cap * wp.tanh(raw3 / cap)
    maximum = wp.max(wp.max(bad0, bad1), wp.max(bad2, bad3))
    exp0 = wp.exp(bad0 - maximum)
    exp1 = wp.exp(bad1 - maximum)
    exp2 = wp.exp(bad2 - maximum)
    exp3 = wp.exp(bad3 - maximum)
    inverse = wp.float64(1.0) / wp.max(exp0 + exp1 + exp2 + exp3, wp.float64(1.0e-300))
    return wp.vec4d(exp0 * inverse, exp1 * inverse, exp2 * inverse, exp3 * inverse)


@wp.func
def reflection_symmetric_ratios(
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
    direct = raw_badness_ratios(nn_features(q0, q1, q2, q3, q4, q5, q6), w1, b1, w2, b2, w3, b3, w4, b4)
    reflected = raw_badness_ratios(nn_features(q6, q5, q4, q3, q2, q1, q0), w1, b1, w2, b2, w3, b3, w4, b4)
    return wp.vec4d(
        wp.float64(0.5) * (direct[0] + reflected[3]),
        wp.float64(0.5) * (direct[1] + reflected[2]),
        wp.float64(0.5) * (direct[2] + reflected[1]),
        wp.float64(0.5) * (direct[3] + reflected[0]),
    )


@wp.func
def reconstruction_weights(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    q5: wp.float64,
    q6: wp.float64,
    lr: int,
    gauss: int,
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
    linear = W7.weno7_optimal_weights(lr, gauss)
    if plateau_detected(q0, q1, q2, q3, q4, q5, q6) == 1:
        return linear
    ratio = reflection_symmetric_ratios(q0, q1, q2, q3, q4, q5, q6, w1, b1, w2, b2, w3, b3, w4, b4)
    inverse0 = wp.float64(1.0) / (wp.float64(4.0) * ratio[0] + wp.float64(1.0e-12))
    inverse1 = wp.float64(1.0) / (wp.float64(4.0) * ratio[1] + wp.float64(1.0e-12))
    inverse2 = wp.float64(1.0) / (wp.float64(4.0) * ratio[2] + wp.float64(1.0e-12))
    inverse3 = wp.float64(1.0) / (wp.float64(4.0) * ratio[3] + wp.float64(1.0e-12))
    alpha0 = linear[0] * inverse0 * inverse0
    alpha1 = linear[1] * inverse1 * inverse1
    alpha2 = linear[2] * inverse2 * inverse2
    alpha3 = linear[3] * inverse3 * inverse3
    inverse_sum = wp.float64(1.0) / wp.max(alpha0 + alpha1 + alpha2 + alpha3, wp.float64(1.0e-300))
    weights = wp.vec4d(
        alpha0 * inverse_sum,
        alpha1 * inverse_sum,
        alpha2 * inverse_sum,
        alpha3 * inverse_sum,
    )
    if eno_cutoff == 1:
        keep0 = wp.float64(1.0)
        keep1 = wp.float64(1.0)
        keep2 = wp.float64(1.0)
        keep3 = wp.float64(1.0)
        cutoff = wp.float64(4.0e-7)
        if weights[0] <= cutoff:
            keep0 = wp.float64(0.0)
        if weights[1] <= cutoff:
            keep1 = wp.float64(0.0)
        if weights[2] <= cutoff:
            keep2 = wp.float64(0.0)
        if weights[3] <= cutoff:
            keep3 = wp.float64(0.0)
        kept_sum = keep0 * weights[0] + keep1 * weights[1] + keep2 * weights[2] + keep3 * weights[3]
        inverse_kept = wp.float64(1.0) / wp.max(kept_sum, wp.float64(1.0e-300))
        weights = wp.vec4d(
            keep0 * weights[0] * inverse_kept,
            keep1 * weights[1] * inverse_kept,
            keep2 * weights[2] * inverse_kept,
            keep3 * weights[3] * inverse_kept,
        )
    return weights


@wp.func
def face_scalar(
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
    eno_cutoff: int,
) -> wp.float64:
    location0 = 1
    location1 = 2
    location2 = 3
    location3 = 4
    if lr == 2:
        location0 = 2
        location1 = 3
        location2 = 4
        location3 = 5
    weights = reconstruction_weights(
        q0, q1, q2, q3, q4, q5, q6, lr, 0,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )
    s0 = W7.stencil_weno7(q0, q1, q2, q3, location0, 0)
    s1 = W7.stencil_weno7(q1, q2, q3, q4, location1, 0)
    s2 = W7.stencil_weno7(q2, q3, q4, q5, location2, 0)
    s3 = W7.stencil_weno7(q3, q4, q5, q6, location3, 0)
    return weights[0] * s0 + weights[1] * s1 + weights[2] * s2 + weights[3] * s3


@wp.func
def gauss_scalar(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    q5: wp.float64,
    q6: wp.float64,
    location: int,
    w1: wp.array3d(dtype=wp.float64),
    b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64),
    b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64),
    b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64),
    b4: wp.array2d(dtype=wp.float64),
    eno_cutoff: int,
) -> wp.float64:
    stencil0 = 1
    stencil1 = 2
    stencil2 = 3
    stencil3 = 4
    if location == 2:
        stencil0 = 5
        stencil1 = 6
        stencil2 = 7
        stencil3 = 8
    weights = reconstruction_weights(
        q0, q1, q2, q3, q4, q5, q6, location, 1,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )
    s0 = W7.stencil_weno7_2gauss(q0, q1, q2, q3, stencil0, 0)
    s1 = W7.stencil_weno7_2gauss(q1, q2, q3, q4, stencil1, 0)
    s2 = W7.stencil_weno7_2gauss(q2, q3, q4, q5, stencil2, 0)
    s3 = W7.stencil_weno7_2gauss(q3, q4, q5, q6, stencil3, 0)
    return weights[0] * s0 + weights[1] * s1 + weights[2] * s2 + weights[3] * s3


@wp.kernel
def weights_probe_kernel(
    stencils: wp.array2d(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
    lr: int,
    gauss: int,
    w1: wp.array3d(dtype=wp.float64),
    b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64),
    b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64),
    b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64),
    b4: wp.array2d(dtype=wp.float64),
):
    index = wp.tid()
    q0 = stencils[index, 0]
    q1 = stencils[index, 1]
    q2 = stencils[index, 2]
    q3 = stencils[index, 3]
    q4 = stencils[index, 4]
    q5 = stencils[index, 5]
    q6 = stencils[index, 6]
    weights = reconstruction_weights(
        q0, q1, q2, q3, q4, q5, q6, lr, gauss,
        w1, b1, w2, b2, w3, b3, w4, b4, 0,
    )
    output[index, 0] = weights[0]
    output[index, 1] = weights[1]
    output[index, 2] = weights[2]
    output[index, 3] = weights[3]
