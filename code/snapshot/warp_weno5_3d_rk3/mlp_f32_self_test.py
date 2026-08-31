from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import warp as wp

from teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast import weno5_core as W

from . import kernels as K
from . import mlp_probe_kernels_f32 as MP
from .config import ShockBubbleConfig
from .mlp_f32 import PARAMETER_NAMES, load_mlp_float32_parameters
from .mlp_self_test import probe_stencils
from .solver import Weno5Rk3Solver


DEFAULT_MODEL = Path(
    "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/runs/"
    "apost_weno5_v20_mlp_f32_fast_200k/checkpoints/model_step_016500.npz"
)


def _torch_parameters(model_path: Path) -> dict[str, torch.Tensor]:
    with np.load(model_path, allow_pickle=False) as archive:
        return {
            name: torch.as_tensor(archive[name][0], dtype=torch.float32)
            for name in PARAMETER_NAMES
        }


def _raw_forward(features: torch.Tensor, parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    old_scale = features[..., 4]
    log10_relative = 16.0 * old_scale - 16.0
    new_scale = torch.clamp((log10_relative + 4.0) / 4.0, 0.0, 1.0)
    hidden = torch.cat((features[..., :4], new_scale.unsqueeze(-1)), dim=-1).float()
    hidden = hidden @ parameters["w1"] + parameters["b1"]
    hidden = hidden * torch.sigmoid(hidden)
    hidden = hidden @ parameters["w2"] + parameters["b2"]
    hidden = hidden * torch.sigmoid(hidden)
    hidden = hidden @ parameters["w3"] + parameters["b3"]
    hidden = hidden * torch.sigmoid(hidden)
    raw = hidden @ parameters["w4"] + parameters["b4"]
    return torch.softmax(6.0 * torch.tanh(raw / 6.0), dim=-1)


@torch.no_grad()
def torch_reconstruction(model_path: Path, stencils: np.ndarray) -> np.ndarray:
    parameters = _torch_parameters(model_path)
    q = torch.as_tensor(stencils, dtype=torch.float64)
    features = W.weno5_features(q)
    direct = _raw_forward(features, parameters).double()
    reflected_features = torch.stack(
        (features[..., 2], features[..., 1], features[..., 0], features[..., 3], features[..., 4]),
        dim=-1,
    )
    reflected_raw = _raw_forward(reflected_features, parameters).double()
    reflected = torch.stack(
        (reflected_raw[..., 2], reflected_raw[..., 1], reflected_raw[..., 0]),
        dim=-1,
    )
    ratios = 0.5 * (direct + reflected)
    columns = []
    for lr in (1, 2, 3, 4):
        weights = W.omega_from_ratio(ratios, lr)
        plateau = W.plateau_mask(q).reshape(-1, 1)
        linear = W.optimal_d(lr, "cpu").reshape(1, 3).expand_as(weights)
        weights = torch.where(plateau, linear, weights)
        columns.append(torch.sum(weights * W.candidate_values(q, lr), dim=1))
    return torch.stack(columns, dim=1).cpu().numpy()


def warp_reconstruction(model_path: Path, stencils: np.ndarray, device: str) -> np.ndarray:
    parameters = load_mlp_float32_parameters(model_path, device)
    stencil_array = wp.array(stencils, dtype=wp.float64, device=device)
    output = wp.empty((stencils.shape[0], 4), dtype=wp.float64, device=device)
    wp.launch(
        MP.scalar_reconstruction_probe_f32_kernel,
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
    if maximum > 2.0e-6 or relative > 2.0e-6:
        index = np.unravel_index(int(np.argmax(error)), error.shape)
        raise AssertionError(
            f"Warp/Torch mixed-FP32 reconstruction mismatch at {index}: "
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
    parser = argparse.ArgumentParser(description="WENO5 mixed-FP32 3-D Warp deployment self-test")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    report = {
        "model": str(args.model.resolve()),
        "parameter_dtype": "float32",
        "solver_state_dtype": "float64",
        "scalar_four_head_parity": test_scalar_parity(args.model, args.device),
        "three_dimensional_uniform_flow": test_mlp_uniform_flow(args.model, args.device),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
