from __future__ import annotations

import warp as wp

from . import kernels as K


wp.set_module_options({"fast_math": False, "fuse_fp": False})

MLP_INPUTS = 5
MLP_HIDDEN1 = 10
MLP_HIDDEN2 = 6
MLP_HIDDEN3 = 6

Features5f = wp.types.vector(length=MLP_INPUTS, dtype=wp.float32)
Hidden10f = wp.types.vector(length=MLP_HIDDEN1, dtype=wp.float32)
Hidden6f = wp.types.vector(length=MLP_HIDDEN2, dtype=wp.float32)

HEAD_NORMAL_LR1 = wp.constant(0)
HEAD_NORMAL_LR2 = wp.constant(1)
HEAD_GAUSS_LR1 = wp.constant(2)
HEAD_GAUSS_LR2 = wp.constant(3)


@wp.func
def swish(value: wp.float32) -> wp.float32:
    return value / (wp.float32(1.0) + wp.exp(-value))


@wp.func
def raw_sensors(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
) -> wp.vec4d:
    d20 = q0 - wp.float64(2.0) * q1 + q2
    d21 = q1 - wp.float64(2.0) * q2 + q3
    d22 = q2 - wp.float64(2.0) * q3 + q4
    delta0 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d20) + wp.float64(0.25) * wp.abs(
        q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2
    )
    delta1 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d21) + wp.float64(0.25) * wp.abs(q1 - q3)
    delta2 = (wp.float64(13.0) / wp.float64(12.0)) * wp.abs(d22) + wp.float64(0.25) * wp.abs(
        wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4
    )
    eps = wp.float64(1.0e-15)
    gamma0 = wp.abs(d20) / (wp.abs(q1 - q0) + wp.abs(q2 - q1) + eps)
    gamma1 = wp.abs(d21) / (wp.abs(q2 - q1) + wp.abs(q3 - q2) + eps)
    gamma2 = wp.abs(d22) / (wp.abs(q3 - q2) + wp.abs(q4 - q3) + eps)
    gamma_s = wp.min(wp.float64(1.0), wp.max(wp.max(gamma0, gamma1), gamma2))
    return wp.vec4d(delta0, delta1, delta2, gamma_s)


@wp.func
def nn_features(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
) -> Features5f:
    raw = raw_sensors(q0, q1, q2, q3, q4)
    delta_max = wp.max(wp.max(raw[0], raw[1]), raw[2])
    inv_delta_max = wp.float64(1.0) / wp.max(delta_max, wp.float64(1.0e-15))
    q_scale = wp.max(
        wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))),
        wp.abs(q4),
    )
    q_scale = wp.max(q_scale, wp.float64(1.0))
    relative_scale = wp.max(delta_max / q_scale, wp.float64(1.0e-30))
    log10_scale = wp.log(relative_scale) / wp.log(wp.float64(10.0))
    scale_feature = (log10_scale + wp.float64(4.0)) / wp.float64(4.0)
    scale_feature = wp.min(wp.float64(1.0), wp.max(wp.float64(0.0), scale_feature))
    result = Features5f()
    result[0] = wp.float32(raw[0] * inv_delta_max)
    result[1] = wp.float32(raw[1] * inv_delta_max)
    result[2] = wp.float32(raw[2] * inv_delta_max)
    result[3] = wp.float32(raw[3])
    result[4] = wp.float32(scale_feature)
    return result


@wp.func
def plateau_detected(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
) -> int:
    raw = raw_sensors(q0, q1, q2, q3, q4)
    delta_max = wp.max(wp.max(raw[0], raw[1]), raw[2])
    q_scale = wp.max(
        wp.max(wp.max(wp.abs(q0), wp.abs(q1)), wp.max(wp.abs(q2), wp.abs(q3))),
        wp.abs(q4),
    )
    q_scale = wp.max(q_scale, wp.float64(1.0))
    if delta_max <= wp.float64(1.0e-13) * q_scale:
        return 1
    return 0


@wp.func
def optimal_weights(head: int) -> wp.vec3d:
    root3 = wp.sqrt(wp.float64(3.0))
    if head == HEAD_NORMAL_LR1:
        return wp.vec3d(wp.float64(0.1), wp.float64(0.6), wp.float64(0.3))
    if head == HEAD_NORMAL_LR2:
        return wp.vec3d(wp.float64(0.3), wp.float64(0.6), wp.float64(0.1))
    if head == HEAD_GAUSS_LR1:
        return wp.vec3d(
            (wp.float64(210.0) + root3) / wp.float64(1080.0),
            wp.float64(11.0) / wp.float64(18.0),
            (wp.float64(210.0) - root3) / wp.float64(1080.0),
        )
    return wp.vec3d(
        (wp.float64(210.0) - root3) / wp.float64(1080.0),
        wp.float64(11.0) / wp.float64(18.0),
        (wp.float64(210.0) + root3) / wp.float64(1080.0),
    )


@wp.func
def badness_ratios(
    features: Features5f,
    w1: wp.array3d(dtype=wp.float32),
    b1: wp.array2d(dtype=wp.float32),
    w2: wp.array3d(dtype=wp.float32),
    b2: wp.array2d(dtype=wp.float32),
    w3: wp.array3d(dtype=wp.float32),
    b3: wp.array2d(dtype=wp.float32),
    w4: wp.array3d(dtype=wp.float32),
    b4: wp.array2d(dtype=wp.float32),
) -> wp.vec3f:
    hidden1 = Hidden10f()
    for output in range(MLP_HIDDEN1):
        value = b1[0, output]
        for feature in range(MLP_INPUTS):
            value = value + features[feature] * w1[0, feature, output]
        hidden1[output] = swish(value)
    hidden2 = Hidden6f()
    for output in range(MLP_HIDDEN2):
        value = b2[0, output]
        for feature in range(MLP_HIDDEN1):
            value = value + hidden1[feature] * w2[0, feature, output]
        hidden2[output] = swish(value)
    hidden3 = Hidden6f()
    for output in range(MLP_HIDDEN3):
        value = b3[0, output]
        for feature in range(MLP_HIDDEN2):
            value = value + hidden2[feature] * w3[0, feature, output]
        hidden3[output] = swish(value)
    raw0 = b4[0, 0]
    raw1 = b4[0, 1]
    raw2 = b4[0, 2]
    for feature in range(MLP_HIDDEN3):
        raw0 = raw0 + hidden3[feature] * w4[0, feature, 0]
        raw1 = raw1 + hidden3[feature] * w4[0, feature, 1]
        raw2 = raw2 + hidden3[feature] * w4[0, feature, 2]
    bound = wp.float32(6.0)
    value0 = bound * wp.tanh(raw0 / bound)
    value1 = bound * wp.tanh(raw1 / bound)
    value2 = bound * wp.tanh(raw2 / bound)
    maximum = wp.max(wp.max(value0, value1), value2)
    exp0 = wp.exp(value0 - maximum)
    exp1 = wp.exp(value1 - maximum)
    exp2 = wp.exp(value2 - maximum)
    inverse = wp.float32(1.0) / (exp0 + exp1 + exp2)
    return wp.vec3f(exp0 * inverse, exp1 * inverse, exp2 * inverse)


@wp.func
def reflection_symmetric_weights(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    head: int,
    w1: wp.array3d(dtype=wp.float32),
    b1: wp.array2d(dtype=wp.float32),
    w2: wp.array3d(dtype=wp.float32),
    b2: wp.array2d(dtype=wp.float32),
    w3: wp.array3d(dtype=wp.float32),
    b3: wp.array2d(dtype=wp.float32),
    w4: wp.array3d(dtype=wp.float32),
    b4: wp.array2d(dtype=wp.float32),
    eno_cutoff: int,
) -> wp.vec3d:
    linear = optimal_weights(head)
    if plateau_detected(q0, q1, q2, q3, q4) == 1:
        return linear
    direct = badness_ratios(nn_features(q0, q1, q2, q3, q4), w1, b1, w2, b2, w3, b3, w4, b4)
    reflected = badness_ratios(nn_features(q4, q3, q2, q1, q0), w1, b1, w2, b2, w3, b3, w4, b4)
    ratio0 = wp.float64(0.5) * (wp.float64(direct[0]) + wp.float64(reflected[2]))
    ratio1 = wp.float64(0.5) * (wp.float64(direct[1]) + wp.float64(reflected[1]))
    ratio2 = wp.float64(0.5) * (wp.float64(direct[2]) + wp.float64(reflected[0]))
    inverse0 = wp.float64(1.0) / (wp.float64(3.0) * ratio0 + wp.float64(1.0e-12))
    inverse1 = wp.float64(1.0) / (wp.float64(3.0) * ratio1 + wp.float64(1.0e-12))
    inverse2 = wp.float64(1.0) / (wp.float64(3.0) * ratio2 + wp.float64(1.0e-12))
    alpha0 = linear[0] * inverse0 * inverse0
    alpha1 = linear[1] * inverse1 * inverse1
    alpha2 = linear[2] * inverse2 * inverse2
    inverse_sum = wp.float64(1.0) / wp.max(alpha0 + alpha1 + alpha2, wp.float64(1.0e-300))
    weight0 = alpha0 * inverse_sum
    weight1 = alpha1 * inverse_sum
    weight2 = alpha2 * inverse_sum
    if eno_cutoff == 1:
        keep0 = wp.float64(1.0)
        keep1 = wp.float64(1.0)
        keep2 = wp.float64(1.0)
        cutoff = wp.float64(4.0e-7)
        if weight0 <= cutoff:
            keep0 = wp.float64(0.0)
        if weight1 <= cutoff:
            keep1 = wp.float64(0.0)
        if weight2 <= cutoff:
            keep2 = wp.float64(0.0)
        inverse_kept = wp.float64(1.0) / wp.max(
            keep0 * weight0 + keep1 * weight1 + keep2 * weight2,
            wp.float64(1.0e-300),
        )
        weight0 = keep0 * weight0 * inverse_kept
        weight1 = keep1 * weight1 * inverse_kept
        weight2 = keep2 * weight2 * inverse_kept
    return wp.vec3d(weight0, weight1, weight2)


@wp.func
def face_scalar(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    lr: int,
    w1: wp.array3d(dtype=wp.float32),
    b1: wp.array2d(dtype=wp.float32),
    w2: wp.array3d(dtype=wp.float32),
    b2: wp.array2d(dtype=wp.float32),
    w3: wp.array3d(dtype=wp.float32),
    b3: wp.array2d(dtype=wp.float32),
    w4: wp.array3d(dtype=wp.float32),
    b4: wp.array2d(dtype=wp.float32),
    eno_cutoff: int,
) -> wp.float64:
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    head = HEAD_NORMAL_LR1
    if lr == 1:
        s0 = (wp.float64(1.0) / wp.float64(3.0)) * q0 - (wp.float64(7.0) / wp.float64(6.0)) * q1 + (wp.float64(11.0) / wp.float64(6.0)) * q2
        s1 = (-wp.float64(1.0) / wp.float64(6.0)) * q1 + (wp.float64(5.0) / wp.float64(6.0)) * q2 + (wp.float64(1.0) / wp.float64(3.0)) * q3
        s2 = (wp.float64(1.0) / wp.float64(3.0)) * q2 + (wp.float64(5.0) / wp.float64(6.0)) * q3 - (wp.float64(1.0) / wp.float64(6.0)) * q4
    else:
        head = HEAD_NORMAL_LR2
        s0 = (-wp.float64(1.0) / wp.float64(6.0)) * q0 + (wp.float64(5.0) / wp.float64(6.0)) * q1 + (wp.float64(1.0) / wp.float64(3.0)) * q2
        s1 = (wp.float64(1.0) / wp.float64(3.0)) * q1 + (wp.float64(5.0) / wp.float64(6.0)) * q2 - (wp.float64(1.0) / wp.float64(6.0)) * q3
        s2 = (wp.float64(11.0) / wp.float64(6.0)) * q2 - (wp.float64(7.0) / wp.float64(6.0)) * q3 + (wp.float64(1.0) / wp.float64(3.0)) * q4
    weights = reflection_symmetric_weights(q0, q1, q2, q3, q4, head, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    return weights[0] * s0 + weights[1] * s1 + weights[2] * s2


@wp.func
def gauss_scalar(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    lr: int,
    w1: wp.array3d(dtype=wp.float32),
    b1: wp.array2d(dtype=wp.float32),
    w2: wp.array3d(dtype=wp.float32),
    b2: wp.array2d(dtype=wp.float32),
    w3: wp.array3d(dtype=wp.float32),
    b3: wp.array2d(dtype=wp.float32),
    w4: wp.array3d(dtype=wp.float32),
    b4: wp.array2d(dtype=wp.float32),
    eno_cutoff: int,
) -> wp.float64:
    head = HEAD_GAUSS_LR1
    location0 = 1
    location1 = 2
    location2 = 3
    if lr == 2:
        head = HEAD_GAUSS_LR2
        location0 = 4
        location1 = 5
        location2 = 6
    weights = reflection_symmetric_weights(q0, q1, q2, q3, q4, head, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    s0 = K.gauss_candidate(q0, q1, q2, location0)
    s1 = K.gauss_candidate(q1, q2, q3, location1)
    s2 = K.gauss_candidate(q2, q3, q4, location2)
    return weights[0] * s0 + weights[1] * s1 + weights[2] * s2
