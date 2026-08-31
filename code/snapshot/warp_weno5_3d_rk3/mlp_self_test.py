from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import warp as wp

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v12_reflection_sym.v12_model import load_checkpoint

from . import kernels as K
from . import mlp_probe_kernels as MP
from .config import ShockBubbleConfig
from .mlp import load_mlp_parameters
from .solver import Weno5Rk3Solver


DEFAULT_MODEL = Path(
    "teacherfree_lab_weno5_v20_distance_balanced/runs/"
    "apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/"
    "model_step_012250.npz"
)


def probe_stencils() -> np.ndarray:
    rng = np.random.default_rng(20260803)
    random = rng.normal(size=(24, 5))
    return np.ascontiguousarray(
        np.vstack(
            (
                np.zeros((1, 5)),
                np.ones((1, 5)),
                np.array([[0.0, 0.0, 1.0, 1.0, 1.0]]),
                np.array([[1.0, 1.0, 1.0, 0.0, 0.0]]),
                np.array([[0.0, 0.1, 0.4, 0.9, 1.6]]),
                np.array([[1.6, 0.9, 0.4, 0.1, 0.0]]),
                np.array([[1.0, 1.0 + 1.0e-7, 1.0 - 2.0e-7, 1.0e0 + 1.0e-7, 1.0]]),
                random,
            )
        ),
        dtype=np.float64,
    )


@torch.no_grad()
def torch_reconstruction(model_path: Path, stencils: np.ndarray) -> np.ndarray:
    model = load_checkpoint(model_path, "cpu").eval()
    q = torch.as_tensor(stencils, dtype=torch.float64)
    ratios = model(W.weno5_features(q))
    columns = []
    for lr in (1, 2, 3, 4):
        weights = W.omega_from_ratio(ratios, lr)
        plateau = W.plateau_mask(q).reshape(-1, 1)
        linear = W.optimal_d(lr, "cpu").reshape(1, 3).expand_as(weights)
        weights = torch.where(plateau, linear, weights)
        columns.append(torch.sum(weights * W.candidate_values(q, lr), dim=1))
    return torch.stack(columns, dim=1).cpu().numpy()


def warp_reconstruction(model_path: Path, stencils: np.ndarray, device: str) -> np.ndarray:
    parameters = load_mlp_parameters(model_path, device)
    stencil_array = wp.array(stencils, dtype=wp.float64, device=device)
    output = wp.empty((stencils.shape[0], 4), dtype=wp.float64, device=device)
    wp.launch(
        MP.scalar_reconstruction_probe_kernel,
        dim=stencils.shape[0],
        inputs=[stencil_array, output] + parameters.kernel_inputs(),
        device=device,
    )
    wp.synchronize_device(device)
    return output.numpy()


def test_scalar_parity(model_path: Path, device: str) -> dict[str, float]:
    stencils = probe_stencils()
    expected = torch_reconstruction(model_path, stencils)
    actual = warp_reconstruction(model_path, stencils, device)
    error = np.abs(actual - expected)
    maximum = float(np.max(error))
    relative = float(np.max(error / np.maximum(np.abs(expected), 1.0)))
    if maximum > 2.0e-13 or relative > 2.0e-13:
        index = np.unravel_index(int(np.argmax(error)), error.shape)
        raise AssertionError(
            f"Warp/Torch MLP reconstruction mismatch at {index}: "
            f"warp={actual[index]:.17e}, torch={expected[index]:.17e}, error={maximum:.3e}"
        )
    return {"max_absolute_error": maximum, "max_scaled_error": relative}


def test_mlp_uniform_flow(model_path: Path, device: str) -> dict[str, float | int]:
    config = replace(ShockBubbleConfig(), nx=8, ny=7, nz=6, t_end=1.0e-7)
    solver = Weno5Rk3Solver(config, device=device, model=model_path)
    primitive = np.array([1.2, 0.2, -0.1, 0.05, 1.0], dtype=np.float64)
    conserved = np.array(
        [
            primitive[0],
            primitive[0] * primitive[1],
            primitive[0] * primitive[2],
            primitive[0] * primitive[3],
            primitive[4] / 0.4
            + 0.5 * primitive[0] * np.sum(np.square(primitive[1:4])),
        ],
        dtype=np.float64,
    )
    host = np.empty(config.padded_shape, dtype=np.float64)
    host[...] = conserved
    solver.q.assign(host)
    solver._launch(K.conserved_to_primitive_kernel, config.padded_shape[:3], [solver.q, solver.primitive])
    before = solver.q.numpy().copy()
    solver.advance()
    g = config.ghost
    physical = np.s_[g : g + config.nz, g : g + config.ny, g : g + config.nx, :]
    error = float(np.max(np.abs(solver.q.numpy()[physical] - before[physical])))
    diagnostics = solver.diagnostics()
    if error > 2.0e-15 or diagnostics["nan_count"] != 0 or diagnostics["rho_min"] <= 0.0 or diagnostics["p_min"] <= 0.0:
        raise AssertionError({"max_absolute_error": error, **diagnostics})
    return {"max_absolute_error": error, **diagnostics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WENO5-SR 3-D Warp deployment self-test")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    report = {
        "model": str(args.model.resolve()),
        "scalar_four_head_parity": test_scalar_parity(args.model, args.device),
        "three_dimensional_uniform_flow": test_mlp_uniform_flow(args.model, args.device),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
