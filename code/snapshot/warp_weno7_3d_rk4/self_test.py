from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from warp_weno5_3d_rk3 import kernels as B

from . import kernels as K
from .binary_io import read_step, sha256, write_step
from .config import ShockBubbleConfig
from .solver import Weno7Rk4Solver


EXPECTED_INITIAL_SHA256 = "39bcd77c4cfa80761e4c4af530f07495f5417f7c1b0e3d637ebf0d96290c23ae"


def _conserved(primitive: np.ndarray) -> np.ndarray:
    rho, u, v, w, pressure = primitive
    result = np.empty(5, dtype=np.float64)
    result[0] = rho
    result[1] = rho * u
    result[2] = rho * v
    result[3] = rho * w
    result[4] = pressure / (1.4 - 1.0) + 0.5 * rho * (u * u + v * v + w * w)
    return result


def test_binary_roundtrip() -> None:
    values = np.arange(4 * 5 * 6 * 5, dtype=np.float64).reshape(4, 5, 6, 5) / 17.0
    with tempfile.TemporaryDirectory(prefix="weno7_3d_binary_") as directory:
        path = Path(directory) / "step.bin"
        write_step(path, 0.125, values)
        time, loaded = read_step(path)
    assert time == 0.125
    assert np.array_equal(values.view(np.uint64), loaded.view(np.uint64))


def test_scalar_reconstruction(device: str) -> dict[str, float]:
    stencils = np.array(
        [
            [2.0] * 7,
            [-0.9, -0.25, 0.1, 0.7, 1.4, 1.55, 1.8],
            [1.0, 1.0, 1.0, 0.2, -0.4, -0.4, -0.4],
            [0.17, -0.63, 1.2, 0.09, -0.31, 0.82, 0.41],
        ],
        dtype=np.float64,
    )
    probes = np.concatenate([stencils, stencils[:, ::-1]], axis=0)
    source = wp.array(probes, dtype=wp.float64, device=device)
    output = wp.empty((probes.shape[0], 4), dtype=wp.float64, device=device)
    wp.launch(K.scalar_reconstruction_probe, dim=probes.shape[0], inputs=[source, output], device=device)
    wp.synchronize_device(wp.get_device(device))
    values = output.numpy()

    assert np.array_equal(values[0].view(np.uint64), np.full(4, 2.0).view(np.uint64))
    n = stencils.shape[0]
    face_defect = np.max(np.abs(values[:n, 0] - values[n:, 1]))
    gauss_defect = np.max(np.abs(values[:n, 2] - values[n:, 3]))
    assert face_defect <= 4.0e-15, face_defect
    assert gauss_defect <= 4.0e-15, gauss_defect
    assert np.all(np.isfinite(values))
    return {"face_reflection_max_abs": float(face_defect), "gauss_reflection_max_abs": float(gauss_defect)}


def test_transmissive_boundary(device: str) -> None:
    config = replace(ShockBubbleConfig(), nx=7, ny=7, nz=7, t_end=0.0)
    solver = Weno7Rk4Solver(config, device=device)
    g = config.ghost
    interior = np.empty((config.nz, config.ny, config.nx, 5), dtype=np.float64)
    k, j, i, c = np.indices(interior.shape)
    interior[...] = 10000.0 * k + 100.0 * j + i + c / 16.0
    host = np.full(config.padded_shape, -1.0, dtype=np.float64)
    host[g : g + config.nz, g : g + config.ny, g : g + config.nx, :] = interior
    solver.q.assign(host)
    solver.apply_boundary(solver.q)
    expected = np.pad(interior, ((g, g), (g, g), (g, g), (0, 0)), mode="edge")
    actual = solver.q.numpy()
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_periodic_boundary(device: str) -> None:
    config = replace(ShockBubbleConfig(), nx=7, ny=8, nz=9, t_end=0.0)
    solver = Weno7Rk4Solver(config, device=device, boundary="periodic")
    g = config.ghost
    interior = np.empty((config.nz, config.ny, config.nx, 5), dtype=np.float64)
    k, j, i, c = np.indices(interior.shape)
    interior[...] = 10000.0 * k + 100.0 * j + i + c / 16.0
    host = np.full(config.padded_shape, -1.0, dtype=np.float64)
    host[g : g + config.nz, g : g + config.ny, g : g + config.nx, :] = interior
    solver.q.assign(host)
    solver.apply_boundary(solver.q)
    expected = np.pad(interior, ((g, g), (g, g), (g, g), (0, 0)), mode="wrap")
    actual = solver.q.numpy()
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_uniform_flow(device: str) -> float:
    config = replace(ShockBubbleConfig(), nx=8, ny=8, nz=8, t_end=1.0e-8, cfl=0.05)
    solver = Weno7Rk4Solver(config, device=device)
    state = _conserved(np.array([1.2, 0.2, -0.1, 0.05, 1.0], dtype=np.float64))
    host = np.empty(config.padded_shape, dtype=np.float64)
    host[...] = state
    solver.q.assign(host)
    solver._launch(B.conserved_to_primitive_kernel, config.padded_shape[:3], [solver.q, solver.primitive])
    before = solver.q.numpy().copy()
    assert solver.advance()
    g = config.ghost
    physical = np.s_[g : g + config.nz, g : g + config.ny, g : g + config.nx, :]
    after = solver.q.numpy()
    scale = np.maximum(np.abs(before[physical]), 1.0)
    relative = float(np.max(np.abs(after[physical] - before[physical]) / scale))
    assert relative <= 5.0e-15, relative
    diagnostics = solver.diagnostics()
    assert diagnostics["nan_count"] == 0
    assert diagnostics["rho_min"] > 0.0
    assert diagnostics["p_min"] > 0.0
    return relative


def test_initialization_hash(device: str) -> str:
    config = ShockBubbleConfig()
    solver = Weno7Rk4Solver(config, device=device)
    solver.initialize()
    with tempfile.TemporaryDirectory(prefix="weno7_3d_initial_") as directory:
        path = Path(directory) / "step_0000.bin"
        write_step(path, 0.0, solver.primitive_host())
        digest = sha256(path)
    assert digest == EXPECTED_INITIAL_SHA256, (digest, EXPECTED_INITIAL_SHA256)
    return digest


def _smooth_2d_state(nx: int, ny: int, ghost: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(nx, dtype=np.float64) + 0.5) / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    rho = 1.0 + 0.03 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    u = 0.2 + 0.015 * np.cos(2.0 * np.pi * yy)
    v = -0.1 + 0.01 * np.sin(2.0 * np.pi * xx)
    pressure = 1.0 + 0.02 * np.cos(2.0 * np.pi * (xx + yy))
    energy = pressure / 0.4 + 0.5 * rho * (u * u + v * v)

    state2 = np.zeros((ny + 2 * ghost, nx + 2 * ghost, 4), dtype=np.float64)
    state2[ghost : ghost + ny, ghost : ghost + nx, 0] = rho
    state2[ghost : ghost + ny, ghost : ghost + nx, 1] = rho * u
    state2[ghost : ghost + ny, ghost : ghost + nx, 2] = rho * v
    state2[ghost : ghost + ny, ghost : ghost + nx, 3] = energy

    physical3 = np.zeros((ny, nx, 5), dtype=np.float64)
    physical3[..., 0] = rho
    physical3[..., 1] = rho * u
    physical3[..., 2] = rho * v
    physical3[..., 4] = energy
    return state2, physical3


def test_two_dimensional_reduction(device: str) -> dict[str, float]:
    from weno7_point_rk4_shu import point_rk4 as P2

    nx = 8
    ny = 8
    nz = 7
    ghost = 4
    dt = 1.0e-4
    state2, physical3 = _smooth_2d_state(nx, ny, ghost)

    params2 = P2.Params(nx=nx, ny=ny, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, ghost=ghost)
    arrays2 = P2.allocate_arrays(state2, params2, device)
    P2.compute_rhs(
        arrays2,
        params2,
        "u",
        "rhs0",
        dt,
        device,
        solver_kind=0,
        characteristic=True,
        reverse_upwind=False,
        boundary="outflow",
    )
    wp.synchronize_device(wp.get_device(device))
    rhs2 = arrays2["rhs0"].numpy()[ghost : ghost + ny, ghost : ghost + nx, :]

    config3 = replace(
        ShockBubbleConfig(),
        nx=nx,
        ny=ny,
        nz=nz,
        x_start=0.0,
        x_end=1.0,
        y_start=0.0,
        y_end=1.0,
        z_start=0.0,
        z_end=1.0,
        t_end=0.0,
    )
    solver3 = Weno7Rk4Solver(config3, device=device)
    host3 = np.zeros(config3.padded_shape, dtype=np.float64)
    for k in range(nz):
        host3[ghost + k, ghost : ghost + ny, ghost : ghost + nx, :] = physical3
    solver3.q.assign(host3)
    solver3.compute_rhs(solver3.q, solver3.rhs0, dt, reverse=False)
    wp.synchronize_device(solver3.device)
    rhs3 = solver3.rhs0.numpy()[ghost : ghost + nz, ghost : ghost + ny, ghost : ghost + nx, :]

    mapped3 = rhs3[..., (0, 1, 2, 4)]
    spread_z = float(np.max(np.abs(mapped3 - mapped3[:1])))
    max_abs = float(np.max(np.abs(mapped3[0] - rhs2)))
    denominator = max(float(np.max(np.abs(rhs2))), 1.0)
    max_relative = max_abs / denominator
    assert spread_z <= 2.0e-12, spread_z
    assert max_relative <= 2.0e-11, (max_abs, max_relative)
    assert float(np.max(np.abs(rhs3[..., 3]))) <= 2.0e-12
    return {"max_abs": max_abs, "max_relative": max_relative, "z_spread": spread_z}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-tests for the Warp 3-D WENO7-JS--RK4 port")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-2d", action="store_true", help="skip comparison with the trusted 2-D WENO7 operator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    test_binary_roundtrip()
    reconstruction = test_scalar_reconstruction(args.device)
    test_transmissive_boundary(args.device)
    test_periodic_boundary(args.device)
    uniform_defect = test_uniform_flow(args.device)
    initial_sha = test_initialization_hash(args.device)
    report: dict[str, object] = {
        "binary_roundtrip": "pass",
        "scalar_reconstruction": {"status": "pass", **reconstruction},
        "transmissive_boundary": "pass",
        "periodic_boundary": "pass",
        "uniform_flow": {"status": "pass", "max_scaled_defect": uniform_defect},
        "initialization": {"status": "pass", "sha256": initial_sha},
    }
    if not args.skip_2d:
        report["two_dimensional_reduction"] = {"status": "pass", **test_two_dimensional_reduction(args.device)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
