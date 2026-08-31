from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from teacherfree_lab_weno7_rk4_distance_balanced_fast import weno7_core as W

from . import mlp_kernels as M
from .mlp import load_mlp_parameters


DEFAULT_MODEL = Path(
    "teacherfree_lab_weno7_rk4_distance_balanced_fast/runs/"
    "apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/"
    "model_step_016750.npz"
)


def reference_weights_and_candidates(
    model_path: Path, stencils: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = W.load_checkpoint(model_path, "cpu").eval()
    q = torch.from_numpy(stencils)
    with torch.no_grad():
        ratio = model(W.weno7_features(q))
        plateau = W.plateau_mask(q).unsqueeze(-1)
        weights = []
        candidates = []
        # Warp Gauss location 1 is the +sqrt(3)/6 head (training head 4),
        # while location 2 is the -sqrt(3)/6 head (training head 3).
        for head in (1, 2, 4, 3):
            linear = W.optimal_d(head, q.device, q.dtype).expand(len(q), 4)
            omega = torch.where(plateau, linear, W.omega_from_ratio(ratio, head))
            weights.append(omega)
            candidates.append(W.candidate_values(q, head))
        return torch.stack(weights, dim=1).numpy(), torch.stack(candidates, dim=1).numpy()


def run(model_path: Path, device_name: str) -> dict[str, object]:
    rng = np.random.default_rng(20260803)
    stencils = np.vstack(
        (
            np.ones((1, 7)),
            np.linspace(-1.0, 1.0, 7)[None, :],
            np.array(((0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),)),
            rng.normal(size=(61, 7)),
        )
    ).astype(np.float64)

    wp.init()
    device = wp.get_device(device_name)
    parameters = load_mlp_parameters(model_path, device)
    stencil_array = wp.array(stencils, dtype=wp.float64, device=device)
    output = wp.empty((len(stencils), 4), dtype=wp.float64, device=device)
    warp_heads = []
    for lr, gauss in ((1, 0), (2, 0), (1, 1), (2, 1)):
        wp.launch(
            M.weights_probe_kernel,
            dim=len(stencils),
            inputs=[stencil_array, output, lr, gauss] + parameters.kernel_inputs(),
            device=device,
        )
        wp.synchronize_device(device)
        warp_heads.append(output.numpy().copy())
    warp_weights = np.stack(warp_heads, axis=1)
    torch_weights, candidates = reference_weights_and_candidates(model_path, stencils)
    weight_difference = np.abs(warp_weights - torch_weights)
    warp_reconstruction = np.sum(warp_weights * candidates, axis=-1)
    torch_reconstruction = np.sum(torch_weights * candidates, axis=-1)
    reconstruction_difference = np.abs(warp_reconstruction - torch_reconstruction)
    worst = np.unravel_index(int(np.argmax(weight_difference)), weight_difference.shape)
    return {
        "device": str(device),
        "model": parameters.manifest(),
        "stencil_count": len(stencils),
        "head_order": ["normal_lr1", "normal_lr2", "gauss_plus", "gauss_minus"],
        "finite": bool(np.isfinite(warp_weights).all()),
        "weight_max_abs": float(np.max(weight_difference)),
        "weight_mean_abs": float(np.mean(weight_difference)),
        "per_head_weight_max_abs": [
            float(value) for value in np.max(weight_difference, axis=(0, 2))
        ],
        "reconstruction_max_abs": float(np.max(reconstruction_difference)),
        "per_head_reconstruction_max_abs": [
            float(value) for value in np.max(reconstruction_difference, axis=0)
        ],
        "worst_weight_index": [int(value) for value in worst],
        "worst_warp_weight": float(warp_weights[worst]),
        "worst_torch_weight": float(torch_weights[worst]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the four Warp WENO7 MLP heads with the Torch training core")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.model.resolve(), args.device)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="ascii")
    print(text)


if __name__ == "__main__":
    main()
