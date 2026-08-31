from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from . import kernels as K
from .binary_io import read_step, sha256, write_step
from .config import ShockBubbleConfig
from .solver import Weno5Rk3Solver, run_solver


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
    with tempfile.TemporaryDirectory(prefix="weno3d_binary_") as directory:
        path = Path(directory) / "step.bin"
        write_step(path, 0.125, values)
        time, loaded = read_step(path)
    assert time == 0.125
    assert np.array_equal(values.view(np.uint64), loaded.view(np.uint64))


def test_transmissive_boundary(device: str) -> None:
    config = replace(ShockBubbleConfig(), nx=6, ny=6, nz=6, t_end=0.0)
    solver = Weno5Rk3Solver(config, device=device)
    g = config.ghost
    interior = np.empty((config.nz, config.ny, config.nx, 5), dtype=np.float64)
    k, j, i, c = np.indices(interior.shape)
    interior[...] = 10000.0 * k + 100.0 * j + i + c / 16.0
    host = np.full(config.padded_shape, -1.0, dtype=np.float64)
    host[g : g + config.nz, g : g + config.ny, g : g + config.nx, :] = interior
    solver.q.assign(host)
    solver.apply_boundary()
    expected = np.pad(interior, ((g, g), (g, g), (g, g), (0, 0)), mode="edge")
    actual = solver.q.numpy()
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_uniform_flow(device: str) -> None:
    config = replace(ShockBubbleConfig(), nx=8, ny=7, nz=6, t_end=1.0e-7)
    solver = Weno5Rk3Solver(config, device=device)
    state = _conserved(np.array([1.2, 0.2, -0.1, 0.05, 1.0], dtype=np.float64))
    host = np.empty(config.padded_shape, dtype=np.float64)
    host[...] = state
    solver.q.assign(host)
    solver._launch(K.conserved_to_primitive_kernel, config.padded_shape[:3], [solver.q, solver.primitive])
    before = solver.q.numpy().copy()
    assert solver.advance()
    g = config.ghost
    physical = np.s_[g : g + config.nz, g : g + config.ny, g : g + config.nx, :]
    after = solver.q.numpy()
    relative = np.max(np.abs((after[physical] - before[physical]) / before[physical]))
    # SSPRK3's final 1/3 + 2/3 convex combination can round a constant by one ULP.
    assert relative <= 2.0 * np.finfo(np.float64).eps, relative
    diagnostics = solver.diagnostics()
    assert diagnostics["nan_count"] == 0
    assert diagnostics["rho_min"] > 0.0
    assert diagnostics["p_min"] > 0.0


def test_initialization_hash(device: str) -> str:
    config = ShockBubbleConfig()
    solver = Weno5Rk3Solver(config, device=device)
    solver.initialize()
    with tempfile.TemporaryDirectory(prefix="weno3d_initial_") as directory:
        path = Path(directory) / "step_0000.bin"
        write_step(path, 0.0, solver.primitive_host())
        digest = sha256(path)
    assert digest == EXPECTED_INITIAL_SHA256, (digest, EXPECTED_INITIAL_SHA256)
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-tests for the exact Warp 3-D WENO5--RK3 port")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--full", action="store_true", help="also run the complete 140-step reference comparison")
    parser.add_argument("--reference", type=Path, default=Path("ADER_TR_Project/data/step_0140.bin"))
    parser.add_argument("--out-dir", type=Path, default=Path("warp_weno5_3d_rk3/runs/self_test_full"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    test_binary_roundtrip()
    test_transmissive_boundary(args.device)
    test_uniform_flow(args.device)
    initial_sha = test_initialization_hash(args.device)
    report: dict[str, object] = {
        "binary_roundtrip": "pass",
        "transmissive_boundary": "pass",
        "uniform_flow": "pass",
        "initialization": {"status": "pass", "sha256": initial_sha},
    }
    if args.full:
        full = run_solver(ShockBubbleConfig(), args.device, args.out_dir, reference=args.reference)
        comparison = full.get("comparison")
        if not isinstance(comparison, dict) or not comparison["file_bitwise_identical"]:
            raise AssertionError("the full Warp result is not bitwise identical to the C++ reference")
        report["full_reference"] = {
            "status": "pass",
            "step": full["step"],
            "sha256": comparison["candidate_sha256"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
