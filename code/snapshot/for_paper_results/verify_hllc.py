#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import warp_weno5_helpers as wh5
import weno5_rk3_diff as core5
from weno7_external_clean import warp_weno7_ader4_helpers_classical_only as wh7
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32 import weno5_rk3_diff_mlp_f32 as core5mixed

from for_paper_results import config
from for_paper_results.solvers import weno5_hllc


wp = wh5.wp


if wp is not None:

    @wp.kernel
    def compare_hllc_kernel(
        left: wp.array2d(dtype=wp.float64),
        right: wp.array2d(dtype=wp.float64),
        ours: wp.array2d(dtype=wp.float64),
        reference: wp.array2d(dtype=wp.float64),
        direction: int,
        gamma: wp.float64,
    ):
        i = wp.tid()
        ql = wp.vec4d(left[i, 0], left[i, 1], left[i, 2], left[i, 3])
        qr = wp.vec4d(right[i, 0], right[i, 1], right[i, 2], right[i, 3])
        fo = weno5_hllc.hllc_flux(ql, qr, direction, gamma)
        fr = wh7.hllc_flux(ql, qr, direction, gamma)
        for c in range(4):
            ours[i, c] = fo[c]
            reference[i, c] = fr[c]


    @wp.kernel
    def invalid_hllc_kernel(output: wp.array2d(dtype=wp.float64)):
        bad = wp.vec4d(1.0, 0.0, 0.0, -1.0)
        good = wp.vec4d(1.0, 0.0, 0.0, 2.5)
        ours = weno5_hllc.hllc_flux(bad, good, 1, wp.float64(1.4))
        reference = wh7.hllc_flux(bad, good, 1, wp.float64(1.4))
        for c in range(4):
            output[0, c] = ours[c]
            output[1, c] = reference[c]


def conserved(rho: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.empty((rho.size, 4), dtype=np.float64)
    out[:, 0] = rho
    out[:, 1] = rho * u
    out[:, 2] = rho * v
    out[:, 3] = p / 0.4 + 0.5 * rho * (u * u + v * v)
    return out


def hllc_direct_numpy(left: np.ndarray, right: np.ndarray, direction: int,
                      gamma: float, tiny: float = 1.0e-16) -> np.ndarray:
    """Independent host reference using direct min/max wave-speed estimates."""
    out = np.empty_like(left)
    normal = 1 if direction == 1 else 2
    tangent = 2 if direction == 1 else 1
    for i, (ul, ur) in enumerate(zip(left, right, strict=True)):
        rho_l_raw, rho_r_raw = ul[0], ur[0]
        u_l, v_l = ul[1] / rho_l_raw, ul[2] / rho_l_raw
        u_r, v_r = ur[1] / rho_r_raw, ur[2] / rho_r_raw
        p_l_raw = (gamma - 1.0) * (ul[3] - 0.5 * rho_l_raw * (u_l * u_l + v_l * v_l))
        p_r_raw = (gamma - 1.0) * (ur[3] - 0.5 * rho_r_raw * (u_r * u_r + v_r * v_r))
        wl = np.array([rho_l_raw, u_l, v_l, p_l_raw])
        wr = np.array([rho_r_raw, u_r, v_r, p_r_raw])
        rho_l, rho_r = max(rho_l_raw, tiny), max(rho_r_raw, tiny)
        p_l, p_r = max(p_l_raw, tiny), max(p_r_raw, tiny)

        un_l, un_r = wl[normal], wr[normal]
        a_l = np.sqrt(gamma * p_l / rho_l)
        a_r = np.sqrt(gamma * p_r / rho_r)
        s_l = min(un_l - a_l, un_r - a_r)
        s_r = max(un_l + a_l, un_r + a_r)

        def signed_floor(value: float) -> float:
            if abs(value) >= tiny:
                return value
            return -tiny if value < 0.0 else tiny

        def flux(w: np.ndarray, state: np.ndarray) -> np.ndarray:
            un = w[normal]
            result = np.empty(4)
            result[0] = w[0] * un
            result[1] = w[0] * w[1] * un
            result[2] = w[0] * w[2] * un
            result[normal] += w[3]
            result[3] = un * (state[3] + w[3])
            return result

        f_l, f_r = flux(wl, ul), flux(wr, ur)
        if 0.0 <= s_l:
            out[i] = f_l
            continue
        if 0.0 >= s_r:
            out[i] = f_r
            continue

        denominator = signed_floor(rho_l * (s_l - un_l) - rho_r * (s_r - un_r))
        s_star = (
            p_r - p_l + rho_l * un_l * (s_l - un_l)
            - rho_r * un_r * (s_r - un_r)
        ) / denominator
        state, w, f, s = (ul, wl, f_l, s_l) if s_star >= 0.0 else (ur, wr, f_r, s_r)
        un = w[normal]
        rho = max(w[0], tiny)
        pressure = max(w[3], tiny)
        factor = rho * (s - un) / signed_floor(s - s_star)
        star = np.empty(4)
        star[0] = factor
        star[normal] = factor * s_star
        star[tangent] = factor * w[tangent]
        star[3] = factor * (
            state[3] / rho
            + (s_star - un) * (s_star + pressure / (rho * signed_floor(s - un)))
        )
        out[i] = f + s * (star - state)
    return out


def test_hllc(device: str) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(20260711)
    n = 512
    left = conserved(
        rng.uniform(0.1, 3.0, n), rng.uniform(-2.0, 2.0, n),
        rng.uniform(-2.0, 2.0, n), rng.uniform(0.05, 5.0, n),
    )
    right = conserved(
        rng.uniform(0.1, 3.0, n), rng.uniform(-2.0, 2.0, n),
        rng.uniform(-2.0, 2.0, n), rng.uniform(0.05, 5.0, n),
    )
    maxima: dict[str, float] = {}
    host_maxima: dict[str, float] = {}
    host_scaled_maxima: dict[str, float] = {}
    for direction in (1, 2):
        ql = wp.array(left, dtype=wp.float64, device=device)
        qr = wp.array(right, dtype=wp.float64, device=device)
        ours = wp.zeros((n, 4), dtype=wp.float64, device=device)
        ref = wp.zeros((n, 4), dtype=wp.float64, device=device)
        wp.launch(compare_hllc_kernel, dim=n,
                  inputs=[ql, qr, ours, ref, direction, wp.float64(1.4)], device=device)
        ours_host = ours.numpy()
        delta = np.max(np.abs(ours_host - ref.numpy()))
        maxima[f"direction_{direction}"] = float(delta)
        if delta != 0.0:
            raise AssertionError(f"WENO5 HLLC differs from trusted WENO7 HLLC: dir={direction}, max={delta}")
        host_reference = hllc_direct_numpy(left, right, direction, 1.4)
        host_difference = np.abs(ours_host - host_reference)
        host_delta = np.max(host_difference)
        host_scaled_delta = np.max(host_difference / np.maximum(np.abs(host_reference), 1.0))
        host_maxima[f"direction_{direction}"] = float(host_delta)
        host_scaled_maxima[f"direction_{direction}"] = float(host_scaled_delta)
        if host_scaled_delta > 1.0e-12:
            raise AssertionError(
                f"Warp HLLC differs from direct min/max host reference: "
                f"dir={direction}, abs={host_delta}, scaled={host_scaled_delta}"
            )
    return maxima, host_maxima, host_scaled_maxima


def test_tiny_regularization_matches(device: str) -> str:
    output = wp.zeros((2, 4), dtype=wp.float64, device=device)
    wp.launch(invalid_hllc_kernel, dim=1, inputs=[output], device=device)
    values = output.numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"tiny-regularized HLLC returned non-finite flux: {values}")
    if not np.array_equal(values[0], values[1]):
        raise AssertionError(f"WENO5/WENO7 tiny regularization differs: {values}")
    return "finite and bitwise-matched pass"


def expected_periodic(src: np.ndarray, nx: int, ny: int, gc: int) -> np.ndarray:
    interior = src[gc : gc + ny, gc : gc + nx, :]
    return np.pad(interior, ((gc, gc), (gc, gc), (0, 0)), mode="wrap")


def test_periodic_kernel(kernel, device: str) -> None:
    nx, ny, gc = 7, 5, 3
    src = np.full((ny + 2 * gc, nx + 2 * gc, 4), -999.0, dtype=np.float64)
    vals = np.arange(ny * nx * 4, dtype=np.float64).reshape(ny, nx, 4)
    src[gc : gc + ny, gc : gc + nx, :] = vals
    dst = wp.zeros_like(wp.array(src, dtype=wp.float64, device=device))
    src_wp = wp.array(src, dtype=wp.float64, device=device)
    wp.launch(kernel, dim=(ny + 2 * gc, nx + 2 * gc),
              inputs=[src_wp, dst, nx, ny, gc], device=device)
    actual = dst.numpy()
    expected = expected_periodic(src, nx, ny, gc)
    if not np.array_equal(actual, expected):
        raise AssertionError(f"periodic ghost mismatch, max={np.max(np.abs(actual - expected))}")


def main() -> None:
    if wp is None:
        raise RuntimeError("Warp is unavailable")
    device = "cuda"
    wp.init()
    wp.set_device(device)
    hllc, hllc_host, hllc_host_scaled = test_hllc(device)
    tiny_regularization = test_tiny_regularization_matches(device)
    test_periodic_kernel(core5.copy_periodic_boundary_kernel, device)
    test_periodic_kernel(core5mixed.copy_periodic_boundary_kernel, device)
    result = {
        "device": device,
        "hllc_max_abs_diff_vs_weno7": hllc,
        "hllc_max_abs_diff_vs_direct_numpy": hllc_host,
        "hllc_max_scaled_diff_vs_direct_numpy": hllc_host_scaled,
        "hllc_tiny": 1.0e-16,
        "tiny_regularization_hllc": tiny_regularization,
        "periodic_fp64": "bitwise pass",
        "periodic_mixed": "bitwise pass",
    }
    path = config.PACKAGE / "verification_hllc.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
