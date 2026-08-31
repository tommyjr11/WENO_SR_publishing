from __future__ import annotations

import warp as wp

from . import kernels as K


wp.set_module_options({"fast_math": False, "fuse_fp": False})

MLP_INPUTS = 5
MLP_HIDDEN1 = 10
MLP_HIDDEN2 = 6
MLP_HIDDEN3 = 6

Features5d = wp.types.vector(length=MLP_INPUTS, dtype=wp.float64)
Hidden10d = wp.types.vector(length=MLP_HIDDEN1, dtype=wp.float64)
Hidden6d = wp.types.vector(length=MLP_HIDDEN2, dtype=wp.float64)

HEAD_NORMAL_LR1 = wp.constant(0)
HEAD_NORMAL_LR2 = wp.constant(1)
HEAD_GAUSS_LR1 = wp.constant(2)
HEAD_GAUSS_LR2 = wp.constant(3)


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
) -> Features5d:
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
    # V20 remaps the legacy feature before the MLP.  Its deployed effective
    # feature is clip((log10(relative_scale)+4)/4, 0, 1).
    scale_feature = (log10_scale + wp.float64(4.0)) / wp.float64(4.0)
    scale_feature = wp.min(wp.float64(1.0), wp.max(wp.float64(0.0), scale_feature))
    result = Features5d()
    result[0] = raw[0] * inv_delta_max
    result[1] = raw[1] * inv_delta_max
    result[2] = raw[2] * inv_delta_max
    result[3] = raw[3]
    result[4] = scale_feature
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
    features: Features5d,
    w1: wp.array3d(dtype=wp.float64),
    b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64),
    b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64),
    b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64),
    b4: wp.array2d(dtype=wp.float64),
) -> wp.vec3d:
    hidden1 = Hidden10d()
    for output in range(MLP_HIDDEN1):
        value = b1[0, output]
        for feature in range(MLP_INPUTS):
            value = value + features[feature] * w1[0, feature, output]
        hidden1[output] = swish(value)
    hidden2 = Hidden6d()
    for output in range(MLP_HIDDEN2):
        value = b2[0, output]
        for feature in range(MLP_HIDDEN1):
            value = value + hidden1[feature] * w2[0, feature, output]
        hidden2[output] = swish(value)
    hidden3 = Hidden6d()
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
    bound = wp.float64(6.0)
    value0 = bound * wp.tanh(raw0 / bound)
    value1 = bound * wp.tanh(raw1 / bound)
    value2 = bound * wp.tanh(raw2 / bound)
    maximum = wp.max(wp.max(value0, value1), value2)
    exp0 = wp.exp(value0 - maximum)
    exp1 = wp.exp(value1 - maximum)
    exp2 = wp.exp(value2 - maximum)
    inverse = wp.float64(1.0) / (exp0 + exp1 + exp2)
    return wp.vec3d(exp0 * inverse, exp1 * inverse, exp2 * inverse)


@wp.func
def reflection_symmetric_weights(
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    q3: wp.float64,
    q4: wp.float64,
    head: int,
    w1: wp.array3d(dtype=wp.float64),
    b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64),
    b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64),
    b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64),
    b4: wp.array2d(dtype=wp.float64),
    eno_cutoff: int,
) -> wp.vec3d:
    linear = optimal_weights(head)
    if plateau_detected(q0, q1, q2, q3, q4) == 1:
        return linear
    direct = badness_ratios(nn_features(q0, q1, q2, q3, q4), w1, b1, w2, b2, w3, b3, w4, b4)
    reflected = badness_ratios(nn_features(q4, q3, q2, q1, q0), w1, b1, w2, b2, w3, b3, w4, b4)
    ratio0 = wp.float64(0.5) * (direct[0] + reflected[2])
    ratio1 = wp.float64(0.5) * (direct[1] + reflected[1])
    ratio2 = wp.float64(0.5) * (direct[2] + reflected[0])
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
def face_vec(
    q0: K.Vec5d,
    q1: K.Vec5d,
    q2: K.Vec5d,
    q3: K.Vec5d,
    q4: K.Vec5d,
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
) -> K.Vec5d:
    result = K.Vec5d()
    for component in range(5):
        result[component] = face_scalar(
            q0[component], q1[component], q2[component], q3[component], q4[component], lr,
            w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
        )
    return result


@wp.func
def characteristic_face_x(
    q0: K.Vec5d, q1: K.Vec5d, q2: K.Vec5d, q3: K.Vec5d, q4: K.Vec5d, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
) -> K.Vec5d:
    roe_left = q1
    roe_right = q2
    if lr == 1:
        roe_left = q2
        roe_right = q3
    average = K.roe_average(roe_left, roe_right)
    left_matrix = K.eigen_left_x(average)
    right_matrix = K.eigen_right_x(average)
    characteristic = face_vec(
        K.matvec5(left_matrix, q0), K.matvec5(left_matrix, q1), K.matvec5(left_matrix, q2),
        K.matvec5(left_matrix, q3), K.matvec5(left_matrix, q4), lr,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )
    return K.matvec5(right_matrix, characteristic)


@wp.func
def characteristic_face_y(
    q0: K.Vec5d, q1: K.Vec5d, q2: K.Vec5d, q3: K.Vec5d, q4: K.Vec5d, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
) -> K.Vec5d:
    roe_left = q1
    roe_right = q2
    if lr == 1:
        roe_left = q2
        roe_right = q3
    average = K.roe_average(roe_left, roe_right)
    left_matrix = K.eigen_left_y(average)
    right_matrix = K.eigen_right_y(average)
    characteristic = face_vec(
        K.matvec5(left_matrix, q0), K.matvec5(left_matrix, q1), K.matvec5(left_matrix, q2),
        K.matvec5(left_matrix, q3), K.matvec5(left_matrix, q4), lr,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )
    return K.matvec5(right_matrix, characteristic)


@wp.func
def characteristic_face_z(
    q0: K.Vec5d, q1: K.Vec5d, q2: K.Vec5d, q3: K.Vec5d, q4: K.Vec5d, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
) -> K.Vec5d:
    roe_left = q1
    roe_right = q2
    if lr == 1:
        roe_left = q2
        roe_right = q3
    average = K.roe_average(roe_left, roe_right)
    left_matrix = K.eigen_left_z(average)
    right_matrix = K.eigen_right_z(average)
    characteristic = face_vec(
        K.matvec5(left_matrix, q0), K.matvec5(left_matrix, q1), K.matvec5(left_matrix, q2),
        K.matvec5(left_matrix, q3), K.matvec5(left_matrix, q4), lr,
        w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
    )
    return K.matvec5(right_matrix, characteristic)


@wp.func
def gauss_scalar(
    q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
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


@wp.func
def gauss_vec(
    q0: K.Vec5d, q1: K.Vec5d, q2: K.Vec5d, q3: K.Vec5d, q4: K.Vec5d, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
) -> K.Vec5d:
    result = K.Vec5d()
    for component in range(5):
        result[component] = gauss_scalar(
            q0[component], q1[component], q2[component], q3[component], q4[component], lr,
            w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff,
        )
    return result


@wp.kernel
def normal_x_mlp_kernel(
    q: wp.array4d(dtype=wp.float64), left: wp.array4d(dtype=wp.float64), right: wp.array4d(dtype=wp.float64),
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
):
    k, j, i = wp.tid()
    q0 = K.load_vec(q, k, j, i); q1 = K.load_vec(q, k, j, i + 1); q2 = K.load_vec(q, k, j, i + 2)
    q3 = K.load_vec(q, k, j, i + 3); q4 = K.load_vec(q, k, j, i + 4)
    left_value = characteristic_face_x(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    r0 = K.characteristic_roundtrip_x_lr2(q0, q1, q2); r1 = K.characteristic_roundtrip_x_lr2(q1, q1, q2)
    r2 = K.characteristic_roundtrip_x_lr2(q2, q1, q2); r3 = K.characteristic_roundtrip_x_lr2(q3, q1, q2)
    r4 = K.characteristic_roundtrip_x_lr2(q4, q1, q2)
    right_value = characteristic_face_x(r0, r1, r2, r3, r4, 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    K.store_vec(left, k, j, i, left_value); K.store_vec(right, k, j, i, right_value)


@wp.kernel
def normal_y_mlp_kernel(
    q: wp.array4d(dtype=wp.float64), left: wp.array4d(dtype=wp.float64), right: wp.array4d(dtype=wp.float64),
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
):
    k, j, i = wp.tid()
    q0 = K.load_vec(q, k, j, i); q1 = K.load_vec(q, k, j + 1, i); q2 = K.load_vec(q, k, j + 2, i)
    q3 = K.load_vec(q, k, j + 3, i); q4 = K.load_vec(q, k, j + 4, i)
    left_value = characteristic_face_y(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    r0 = K.characteristic_roundtrip_y_lr2(q0, q1, q2); r1 = K.characteristic_roundtrip_y_lr2(q1, q1, q2)
    r2 = K.characteristic_roundtrip_y_lr2(q2, q1, q2); r3 = K.characteristic_roundtrip_y_lr2(q3, q1, q2)
    r4 = K.characteristic_roundtrip_y_lr2(q4, q1, q2)
    right_value = characteristic_face_y(r0, r1, r2, r3, r4, 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    K.store_vec(left, k, j, i, left_value); K.store_vec(right, k, j, i, right_value)


@wp.kernel
def normal_z_mlp_kernel(
    q: wp.array4d(dtype=wp.float64), left: wp.array4d(dtype=wp.float64), right: wp.array4d(dtype=wp.float64),
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
):
    k, j, i = wp.tid()
    q0 = K.load_vec(q, k, j, i); q1 = K.load_vec(q, k + 1, j, i); q2 = K.load_vec(q, k + 2, j, i)
    q3 = K.load_vec(q, k + 3, j, i); q4 = K.load_vec(q, k + 4, j, i)
    left_value = characteristic_face_z(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    r0 = K.characteristic_roundtrip_z_lr2(q0, q1, q2); r1 = K.characteristic_roundtrip_z_lr2(q1, q1, q2)
    r2 = K.characteristic_roundtrip_z_lr2(q2, q1, q2); r3 = K.characteristic_roundtrip_z_lr2(q3, q1, q2)
    r4 = K.characteristic_roundtrip_z_lr2(q4, q1, q2)
    right_value = characteristic_face_z(r0, r1, r2, r3, r4, 1, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)
    K.store_vec(left, k, j, i, left_value); K.store_vec(right, k, j, i, right_value)


@wp.func
def transverse_y_value(
    source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
) -> K.Vec5d:
    return gauss_vec(K.load_vec(source, k, j + 1, i), K.load_vec(source, k, j + 2, i), K.load_vec(source, k, j + 3, i), K.load_vec(source, k, j + 4, i), K.load_vec(source, k, j + 5, i), lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)


@wp.func
def transverse_x_value(
    source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
) -> K.Vec5d:
    return gauss_vec(K.load_vec(source, k, j, i + 1), K.load_vec(source, k, j, i + 2), K.load_vec(source, k, j, i + 3), K.load_vec(source, k, j, i + 4), K.load_vec(source, k, j, i + 5), lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)


@wp.func
def transverse_z_value(
    source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
) -> K.Vec5d:
    return gauss_vec(K.load_vec(source, k + 1, j, i), K.load_vec(source, k + 2, j, i), K.load_vec(source, k + 3, j, i), K.load_vec(source, k + 4, j, i), K.load_vec(source, k + 5, j, i), lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff)


@wp.kernel
def transverse_y_mlp_kernel(
    source_left: wp.array4d(dtype=wp.float64), source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64), target_right: wp.array4d(dtype=wp.float64), lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
):
    k, j, i = wp.tid()
    K.store_vec(target_left, k, j, i, transverse_y_value(source_left, k, j, i, lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff))
    K.store_vec(target_right, k, j, i, transverse_y_value(source_right, k, j, i, lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff))


@wp.kernel
def transverse_x_mlp_kernel(
    source_left: wp.array4d(dtype=wp.float64), source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64), target_right: wp.array4d(dtype=wp.float64), lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
):
    k, j, i = wp.tid()
    K.store_vec(target_left, k, j, i, transverse_x_value(source_left, k, j, i, lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff))
    K.store_vec(target_right, k, j, i, transverse_x_value(source_right, k, j, i, lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff))


@wp.kernel
def transverse_z_mlp_kernel(
    source_left: wp.array4d(dtype=wp.float64), source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64), target_right: wp.array4d(dtype=wp.float64), lr: int,
    w1: wp.array3d(dtype=wp.float64), b1: wp.array2d(dtype=wp.float64),
    w2: wp.array3d(dtype=wp.float64), b2: wp.array2d(dtype=wp.float64),
    w3: wp.array3d(dtype=wp.float64), b3: wp.array2d(dtype=wp.float64),
    w4: wp.array3d(dtype=wp.float64), b4: wp.array2d(dtype=wp.float64), eno_cutoff: int,
):
    k, j, i = wp.tid()
    K.store_vec(target_left, k, j, i, transverse_z_value(source_left, k, j, i, lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff))
    K.store_vec(target_right, k, j, i, transverse_z_value(source_right, k, j, i, lr, w1, b1, w2, b2, w3, b3, w4, b4, eno_cutoff))


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
    q0 = stencils[index, 0]; q1 = stencils[index, 1]; q2 = stencils[index, 2]
    q3 = stencils[index, 3]; q4 = stencils[index, 4]
    output[index, 0] = face_scalar(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, 0)
    output[index, 1] = face_scalar(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, 0)
    output[index, 2] = gauss_scalar(q0, q1, q2, q3, q4, 1, w1, b1, w2, b2, w3, b3, w4, b4, 0)
    output[index, 3] = gauss_scalar(q0, q1, q2, q3, q4, 2, w1, b1, w2, b2, w3, b3, w4, b4, 0)
