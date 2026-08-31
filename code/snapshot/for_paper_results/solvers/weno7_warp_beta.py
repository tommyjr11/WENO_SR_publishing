"""Warp beta provider for the selected reflection-symmetric WENO7-SR model.

Only MLP inference is moved from Torch to Warp. The state, characteristic
projection, WENO reconstruction, HLLC flux, and Shu SSP-RK4 update remain the
trusted FP64 implementations used by the paper solvers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from teacherfree_lab_weno7_rk4_distance_balanced_fast.warp_sod import (
    point_rk4 as base,
)
from teacherfree_lab_weno7_rk4_distance_balanced_fast.warp_sod import (
    warp_weno7_ader4_helpers_classical_only as wh,
)


wp = wh.wp
Params = base.Params

INPUTS = 6
HIDDEN1 = 24
HIDDEN2 = 16
HIDDEN3 = 16
OUTPUTS = 4


if wp is not None:
    Vec6d = wp.types.vector(length=INPUTS, dtype=wp.float64)
    Vec16d = wp.types.vector(length=HIDDEN2, dtype=wp.float64)
    Vec24d = wp.types.vector(length=HIDDEN1, dtype=wp.float64)

    @wp.func
    def _swish(value: wp.float64) -> wp.float64:
        return value / (wp.float64(1.0) + wp.exp(-value))


    @wp.func
    def _raw_sensors(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        q5: wp.float64,
        q6: wp.float64,
    ) -> Vec6d:
        d01 = -wp.float64(2.0) * q0 + wp.float64(9.0) * q1 - wp.float64(18.0) * q2 + wp.float64(11.0) * q3
        d02 = -q0 + wp.float64(4.0) * q1 - wp.float64(5.0) * q2 + wp.float64(2.0) * q3
        d03 = -q0 + wp.float64(3.0) * q1 - wp.float64(3.0) * q2 + q3

        d11 = q1 - wp.float64(6.0) * q2 + wp.float64(3.0) * q3 + wp.float64(2.0) * q4
        d12 = q2 - wp.float64(2.0) * q3 + q4
        d13 = -q1 + wp.float64(3.0) * q2 - wp.float64(3.0) * q3 + q4

        d21 = -wp.float64(2.0) * q2 - wp.float64(3.0) * q3 + wp.float64(6.0) * q4 - q5
        d22 = q2 - wp.float64(2.0) * q3 + q4
        d23 = -q2 + wp.float64(3.0) * q3 - wp.float64(3.0) * q4 + q5

        d31 = -wp.float64(11.0) * q3 + wp.float64(18.0) * q4 - wp.float64(9.0) * q5 + wp.float64(2.0) * q6
        d32 = wp.float64(2.0) * q3 - wp.float64(5.0) * q4 + wp.float64(4.0) * q5 - q6
        d33 = -q3 + wp.float64(3.0) * q4 - wp.float64(3.0) * q5 + q6

        c1 = wp.float64(1.0) / wp.float64(36.0)
        c2 = wp.float64(13.0) / wp.float64(12.0)
        c3 = wp.float64(781.0) / wp.float64(720.0)
        delta0 = c1 * wp.abs(d01) + c2 * wp.abs(d02) + c3 * wp.abs(d03)
        delta1 = c1 * wp.abs(d11) + c2 * wp.abs(d12) + c3 * wp.abs(d13)
        delta2 = c1 * wp.abs(d21) + c2 * wp.abs(d22) + c3 * wp.abs(d23)
        delta3 = c1 * wp.abs(d31) + c2 * wp.abs(d32) + c3 * wp.abs(d33)

        eps = wp.float64(1.0e-15)
        dd0 = q0 - wp.float64(2.0) * q1 + q2
        dd1 = q1 - wp.float64(2.0) * q2 + q3
        dd2 = q2 - wp.float64(2.0) * q3 + q4
        dd3 = q3 - wp.float64(2.0) * q4 + q5
        dd4 = q4 - wp.float64(2.0) * q5 + q6
        g0 = wp.abs(dd0) / (wp.abs(q1 - q0) + wp.abs(q2 - q1) + eps)
        g1 = wp.abs(dd1) / (wp.abs(q2 - q1) + wp.abs(q3 - q2) + eps)
        g2 = wp.abs(dd2) / (wp.abs(q3 - q2) + wp.abs(q4 - q3) + eps)
        g3 = wp.abs(dd3) / (wp.abs(q4 - q3) + wp.abs(q5 - q4) + eps)
        g4 = wp.abs(dd4) / (wp.abs(q5 - q4) + wp.abs(q6 - q5) + eps)
        gamma_s = wp.min(
            wp.float64(1.0),
            wp.max(wp.max(wp.max(g0, g1), wp.max(g2, g3)), g4),
        )

        values = Vec6d()
        values[0] = delta0
        values[1] = delta1
        values[2] = delta2
        values[3] = delta3
        values[4] = gamma_s
        values[5] = wp.float64(0.0)
        return values


    @wp.func
    def _features(
        q0: wp.float64,
        q1: wp.float64,
        q2: wp.float64,
        q3: wp.float64,
        q4: wp.float64,
        q5: wp.float64,
        q6: wp.float64,
    ) -> Vec6d:
        raw = _raw_sensors(q0, q1, q2, q3, q4, q5, q6)
        delta_max = wp.max(wp.max(raw[0], raw[1]), wp.max(raw[2], raw[3]))
        inverse_delta = wp.float64(1.0) / wp.max(delta_max, wp.float64(1.0e-15))
        q_scale = wp.max(
            wp.max(
                wp.max(wp.abs(q0), wp.abs(q1)),
                wp.max(wp.abs(q2), wp.abs(q3)),
            ),
            wp.max(wp.max(wp.abs(q4), wp.abs(q5)), wp.abs(q6)),
        )
        q_scale = wp.max(q_scale, wp.float64(1.0))
        relative_scale = wp.max(delta_max / q_scale, wp.float64(1.0e-30))
        scale_feature = (
            wp.log(relative_scale) / wp.log(wp.float64(10.0))
            + wp.float64(16.0)
        ) / wp.float64(16.0)
        scale_feature = wp.min(
            wp.float64(1.0), wp.max(wp.float64(0.0), scale_feature)
        )

        values = Vec6d()
        values[0] = raw[0] * inverse_delta
        values[1] = raw[1] * inverse_delta
        values[2] = raw[2] * inverse_delta
        values[3] = raw[3] * inverse_delta
        values[4] = raw[4]
        values[5] = scale_feature
        return values


    @wp.func
    def _raw_ratio(
        features: Vec6d,
        w1: wp.array3d(dtype=wp.float64),
        b1: wp.array2d(dtype=wp.float64),
        w2: wp.array3d(dtype=wp.float64),
        b2: wp.array2d(dtype=wp.float64),
        w3: wp.array3d(dtype=wp.float64),
        b3: wp.array2d(dtype=wp.float64),
        w4: wp.array3d(dtype=wp.float64),
        b4: wp.array2d(dtype=wp.float64),
    ) -> wp.vec4d:
        hidden1 = Vec24d()
        for output in range(HIDDEN1):
            value = b1[0, output]
            for source in range(INPUTS):
                value = value + features[source] * w1[0, source, output]
            hidden1[output] = _swish(value)

        hidden2 = Vec16d()
        for output in range(HIDDEN2):
            value = b2[0, output]
            for source in range(HIDDEN1):
                value = value + hidden1[source] * w2[0, source, output]
            hidden2[output] = _swish(value)

        hidden3 = Vec16d()
        for output in range(HIDDEN3):
            value = b3[0, output]
            for source in range(HIDDEN2):
                value = value + hidden2[source] * w3[0, source, output]
            hidden3[output] = _swish(value)

        raw0 = b4[0, 0]
        raw1 = b4[0, 1]
        raw2 = b4[0, 2]
        raw3 = b4[0, 3]
        for source in range(HIDDEN3):
            raw0 = raw0 + hidden3[source] * w4[0, source, 0]
            raw1 = raw1 + hidden3[source] * w4[0, source, 1]
            raw2 = raw2 + hidden3[source] * w4[0, source, 2]
            raw3 = raw3 + hidden3[source] * w4[0, source, 3]

        cap = wp.float64(6.0)
        bad0 = cap * wp.tanh(raw0 / cap)
        bad1 = cap * wp.tanh(raw1 / cap)
        bad2 = cap * wp.tanh(raw2 / cap)
        bad3 = cap * wp.tanh(raw3 / cap)
        bad_max = wp.max(wp.max(bad0, bad1), wp.max(bad2, bad3))
        exp0 = wp.exp(bad0 - bad_max)
        exp1 = wp.exp(bad1 - bad_max)
        exp2 = wp.exp(bad2 - bad_max)
        exp3 = wp.exp(bad3 - bad_max)
        inverse_sum = wp.float64(1.0) / wp.max(
            exp0 + exp1 + exp2 + exp3, wp.float64(1.0e-300)
        )
        return wp.vec4d(
            exp0 * inverse_sum,
            exp1 * inverse_sum,
            exp2 * inverse_sum,
            exp3 * inverse_sum,
        )


    @wp.func
    def _reflection_symmetric_beta(
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
        raw = _raw_sensors(q0, q1, q2, q3, q4, q5, q6)
        delta_max = wp.max(wp.max(raw[0], raw[1]), wp.max(raw[2], raw[3]))
        q_scale = wp.max(
            wp.max(
                wp.max(wp.abs(q0), wp.abs(q1)),
                wp.max(wp.abs(q2), wp.abs(q3)),
            ),
            wp.max(wp.max(wp.abs(q4), wp.abs(q5)), wp.abs(q6)),
        )
        q_scale = wp.max(q_scale, wp.float64(1.0))
        if delta_max <= wp.float64(1.0e-13) * q_scale:
            return wp.vec4d(
                wp.float64(1.0),
                wp.float64(1.0),
                wp.float64(1.0),
                wp.float64(1.0),
            )

        direct_features = _features(q0, q1, q2, q3, q4, q5, q6)
        reflected_features = Vec6d()
        reflected_features[0] = direct_features[3]
        reflected_features[1] = direct_features[2]
        reflected_features[2] = direct_features[1]
        reflected_features[3] = direct_features[0]
        reflected_features[4] = direct_features[4]
        reflected_features[5] = direct_features[5]
        direct = _raw_ratio(
            direct_features, w1, b1, w2, b2, w3, b3, w4, b4
        )
        reflected = _raw_ratio(
            reflected_features, w1, b1, w2, b2, w3, b3, w4, b4
        )
        scale = wp.float64(2.0)
        return wp.vec4d(
            scale * (direct[0] + reflected[3]),
            scale * (direct[1] + reflected[2]),
            scale * (direct[2] + reflected[1]),
            scale * (direct[3] + reflected[0]),
        )


    @wp.func
    def _oriented_beta(
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
            return _reflection_symmetric_beta(
                q0, q1, q2, q3, q4, q5, q6,
                w1, b1, w2, b2, w3, b3, w4, b4,
            )
        reversed_beta = _reflection_symmetric_beta(
            q6, q5, q4, q3, q2, q1, q0,
            w1, b1, w2, b2, w3, b3, w4, b4,
        )
        return wp.vec4d(
            reversed_beta[3],
            reversed_beta[2],
            reversed_beta[1],
            reversed_beta[0],
        )


    @wp.func
    def _store_normal(
        beta: wp.array3d(dtype=wp.float64),
        j: int,
        i: int,
        side: int,
        component: int,
        values: wp.vec4d,
    ):
        start = side * 16 + component * 4
        beta[j, i, start + 0] = values[0]
        beta[j, i, start + 1] = values[1]
        beta[j, i, start + 2] = values[2]
        beta[j, i, start + 3] = values[3]


    @wp.kernel(enable_backward=False)
    def fill_normal_x_kernel(
        state: wp.array3d(dtype=wp.float64),
        beta: wp.array3d(dtype=wp.float64),
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
        j, i, lane = wp.tid()
        if j < ny + 8 and i < nx + 2 and lane < 8:
            side = 0
            component = lane
            if lane >= 4:
                side = 1
                component = lane - 4
            lr = side + 1
            q0 = wh.vec_from_array(state, j, i + 0)
            q1 = wh.vec_from_array(state, j, i + 1)
            q2 = wh.vec_from_array(state, j, i + 2)
            q3 = wh.vec_from_array(state, j, i + 3)
            q4 = wh.vec_from_array(state, j, i + 4)
            q5 = wh.vec_from_array(state, j, i + 5)
            q6 = wh.vec_from_array(state, j, i + 6)
            roe = wh.roe_average_state(q3, q4, gamma)
            if lr == 2:
                roe = wh.roe_average_state(q2, q3, gamma)
            c0 = wh.con_to_char(q0, roe, 1, gamma)
            c1 = wh.con_to_char(q1, roe, 1, gamma)
            c2 = wh.con_to_char(q2, roe, 1, gamma)
            c3 = wh.con_to_char(q3, roe, 1, gamma)
            c4 = wh.con_to_char(q4, roe, 1, gamma)
            c5 = wh.con_to_char(q5, roe, 1, gamma)
            c6 = wh.con_to_char(q6, roe, 1, gamma)
            values = _oriented_beta(
                c0[component],
                c1[component],
                c2[component],
                c3[component],
                c4[component],
                c5[component],
                c6[component],
                lr,
                w1, b1, w2, b2, w3, b3, w4, b4,
            )
            _store_normal(beta, j, i, side, component, values)


    @wp.kernel(enable_backward=False)
    def fill_normal_y_kernel(
        state: wp.array3d(dtype=wp.float64),
        beta: wp.array3d(dtype=wp.float64),
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
        j, i, lane = wp.tid()
        if j < ny + 2 and i < nx + 8 and lane < 8:
            side = 0
            component = lane
            if lane >= 4:
                side = 1
                component = lane - 4
            lr = side + 1
            q0 = wh.vec_from_array(state, j + 0, i)
            q1 = wh.vec_from_array(state, j + 1, i)
            q2 = wh.vec_from_array(state, j + 2, i)
            q3 = wh.vec_from_array(state, j + 3, i)
            q4 = wh.vec_from_array(state, j + 4, i)
            q5 = wh.vec_from_array(state, j + 5, i)
            q6 = wh.vec_from_array(state, j + 6, i)
            roe = wh.roe_average_state(q3, q4, gamma)
            if lr == 2:
                roe = wh.roe_average_state(q2, q3, gamma)
            c0 = wh.con_to_char(q0, roe, 2, gamma)
            c1 = wh.con_to_char(q1, roe, 2, gamma)
            c2 = wh.con_to_char(q2, roe, 2, gamma)
            c3 = wh.con_to_char(q3, roe, 2, gamma)
            c4 = wh.con_to_char(q4, roe, 2, gamma)
            c5 = wh.con_to_char(q5, roe, 2, gamma)
            c6 = wh.con_to_char(q6, roe, 2, gamma)
            values = _oriented_beta(
                c0[component],
                c1[component],
                c2[component],
                c3[component],
                c4[component],
                c5[component],
                c6[component],
                lr,
                w1, b1, w2, b2, w3, b3, w4, b4,
            )
            _store_normal(beta, j, i, side, component, values)


    @wp.func
    def _store_cross(
        beta: wp.array3d(dtype=wp.float64),
        j: int,
        i: int,
        group: int,
        side: int,
        component: int,
        values: wp.vec4d,
    ):
        start = group * 128 + side * 16 + component * 4
        beta[j, i, start + 0] = values[0]
        beta[j, i, start + 1] = values[1]
        beta[j, i, start + 2] = values[2]
        beta[j, i, start + 3] = values[3]


    @wp.kernel(enable_backward=False)
    def fill_cross_x_kernel(
        right: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        beta: wp.array3d(dtype=wp.float64),
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
    ):
        j, i, lane = wp.tid()
        if j < ny and i < nx + 2 and lane < 16:
            group = 0
            local_lane = lane
            if lane >= 8:
                group = 1
                local_lane = lane - 8
            side = 0
            component = local_lane
            if local_lane >= 4:
                side = 1
                component = local_lane - 4
            lr = side + 1

            q0 = wh.vec_from_array(right, j + 1, i)
            q1 = wh.vec_from_array(right, j + 2, i)
            q2 = wh.vec_from_array(right, j + 3, i)
            q3 = wh.vec_from_array(right, j + 4, i)
            q4 = wh.vec_from_array(right, j + 5, i)
            q5 = wh.vec_from_array(right, j + 6, i)
            q6 = wh.vec_from_array(right, j + 7, i)
            if group == 1:
                q0 = wh.vec_from_array(left, j + 1, i)
                q1 = wh.vec_from_array(left, j + 2, i)
                q2 = wh.vec_from_array(left, j + 3, i)
                q3 = wh.vec_from_array(left, j + 4, i)
                q4 = wh.vec_from_array(left, j + 5, i)
                q5 = wh.vec_from_array(left, j + 6, i)
                q6 = wh.vec_from_array(left, j + 7, i)
            values = _oriented_beta(
                q0[component],
                q1[component],
                q2[component],
                q3[component],
                q4[component],
                q5[component],
                q6[component],
                lr,
                w1, b1, w2, b2, w3, b3, w4, b4,
            )
            _store_cross(beta, j, i, group, side, component, values)


    @wp.kernel(enable_backward=False)
    def fill_cross_y_kernel(
        right: wp.array3d(dtype=wp.float64),
        left: wp.array3d(dtype=wp.float64),
        beta: wp.array3d(dtype=wp.float64),
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
    ):
        j, i, lane = wp.tid()
        if j < ny + 2 and i < nx and lane < 16:
            group = 0
            local_lane = lane
            if lane >= 8:
                group = 1
                local_lane = lane - 8
            side = 0
            component = local_lane
            if local_lane >= 4:
                side = 1
                component = local_lane - 4
            lr = side + 1

            q0 = wh.vec_from_array(right, j, i + 1)
            q1 = wh.vec_from_array(right, j, i + 2)
            q2 = wh.vec_from_array(right, j, i + 3)
            q3 = wh.vec_from_array(right, j, i + 4)
            q4 = wh.vec_from_array(right, j, i + 5)
            q5 = wh.vec_from_array(right, j, i + 6)
            q6 = wh.vec_from_array(right, j, i + 7)
            if group == 1:
                q0 = wh.vec_from_array(left, j, i + 1)
                q1 = wh.vec_from_array(left, j, i + 2)
                q2 = wh.vec_from_array(left, j, i + 3)
                q3 = wh.vec_from_array(left, j, i + 4)
                q4 = wh.vec_from_array(left, j, i + 5)
                q5 = wh.vec_from_array(left, j, i + 6)
                q6 = wh.vec_from_array(left, j, i + 7)
            values = _oriented_beta(
                q0[component],
                q1[component],
                q2[component],
                q3[component],
                q4[component],
                q5[component],
                q6[component],
                lr,
                w1, b1, w2, b2, w3, b3, w4, b4,
            )
            _store_cross(beta, j, i, group, side, component, values)


class WarpWeno7PointBeta:
    """Supply the trusted point-RK4 beta layouts using FP64 Warp kernels."""

    def __init__(
        self, model_path: Path, device: str, gamma: float = 1.4
    ) -> None:
        wh.require_warp()
        wp.init()
        wp.set_device(device)
        data = np.load(Path(model_path), allow_pickle=True)
        expected = {
            "w1": (1, INPUTS, HIDDEN1),
            "b1": (1, HIDDEN1),
            "w2": (1, HIDDEN1, HIDDEN2),
            "b2": (1, HIDDEN2),
            "w3": (1, HIDDEN2, HIDDEN3),
            "b3": (1, HIDDEN3),
            "w4": (1, HIDDEN3, OUTPUTS),
            "b4": (1, OUTPUTS),
        }
        wrong = {
            name: data[name].shape
            for name, shape in expected.items()
            if name not in data.files or data[name].shape != shape
        }
        if wrong:
            raise ValueError(
                "Warp WENO7 provider requires 6->24->16->16->4, "
                f"got {wrong}"
            )
        self.device = device
        self.gamma = float(gamma)
        self.parameters = {
            name: wp.array(
                np.ascontiguousarray(data[name], dtype=np.float64),
                dtype=wp.float64,
                device=device,
                requires_grad=False,
            )
            for name in expected
        }

    def _inputs(self) -> list[object]:
        return [
            self.parameters["w1"],
            self.parameters["b1"],
            self.parameters["w2"],
            self.parameters["b2"],
            self.parameters["w3"],
            self.parameters["b3"],
            self.parameters["w4"],
            self.parameters["b4"],
        ]

    def fill_normal_x_for(
        self, arrays: dict[str, object], params: Params, src_name: str
    ) -> None:
        wp.launch(
            fill_normal_x_kernel,
            dim=(params.ny + 8, params.nx + 2, 8),
            inputs=[
                arrays[src_name],
                arrays["beta_x"],
                *self._inputs(),
                params.nx,
                params.ny,
                wp.float64(params.gamma),
            ],
            device=self.device,
        )

    def fill_normal_y_for(
        self, arrays: dict[str, object], params: Params, src_name: str
    ) -> None:
        wp.launch(
            fill_normal_y_kernel,
            dim=(params.ny + 2, params.nx + 8, 8),
            inputs=[
                arrays[src_name],
                arrays["beta_y"],
                *self._inputs(),
                params.nx,
                params.ny,
                wp.float64(params.gamma),
            ],
            device=self.device,
        )

    def fill_cross_x_point(
        self, arrays: dict[str, object], params: Params
    ) -> None:
        wp.launch(
            fill_cross_x_kernel,
            dim=(params.ny, params.nx + 2, 16),
            inputs=[
                arrays["x_r"],
                arrays["x_l"],
                arrays["beta_cross_x"],
                *self._inputs(),
                params.nx,
                params.ny,
            ],
            device=self.device,
        )

    def fill_cross_y_point(
        self, arrays: dict[str, object], params: Params
    ) -> None:
        wp.launch(
            fill_cross_y_kernel,
            dim=(params.ny + 2, params.nx, 16),
            inputs=[
                arrays["y_r"],
                arrays["y_l"],
                arrays["beta_cross_y"],
                *self._inputs(),
                params.nx,
                params.ny,
            ],
            device=self.device,
        )
