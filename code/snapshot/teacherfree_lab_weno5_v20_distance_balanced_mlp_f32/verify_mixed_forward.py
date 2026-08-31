#!/usr/bin/env python3
"""Compare Torch and Warp mixed-precision WENO weights on random stencils."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v20_distance_balanced_mlp_f32.v20_mlp_f32_model import (
    load_checkpoint,
)
from teacherfree_lab_weno5_v20_distance_balanced_mlp_f32.warp_mlp_f32 import (
    weno5_rk3_diff_v20_mlp_f32 as D,
)


@wp.kernel(enable_backward=False)
def evaluate_weights(
    q: wp.array2d(dtype=wp.float64),
    w1: wp.array3d(dtype=wp.float32),
    b1: wp.array2d(dtype=wp.float32),
    w2: wp.array3d(dtype=wp.float32),
    b2: wp.array2d(dtype=wp.float32),
    w3: wp.array3d(dtype=wp.float32),
    b3: wp.array2d(dtype=wp.float32),
    w4: wp.array3d(dtype=wp.float32),
    b4: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float64),
):
    i = wp.tid()
    value = D.mlp_weights(
        q[i, 0],
        q[i, 1],
        q[i, 2],
        q[i, 3],
        q[i, 4],
        0,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        w4,
        b4,
        0,
    )
    out[i, 0] = value[0]
    out[i, 1] = value[1]
    out[i, 2] = value[2]


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    q_np = rng.normal(size=(args.samples, 5)).astype(np.float64)
    model = load_checkpoint(args.checkpoint, "cpu").eval()
    q_torch = torch.from_numpy(q_np)
    with torch.no_grad():
        ratios = model(W.weno5_features(q_torch))
        expected = W.omega_from_ratio(ratios, 1).numpy()

    required = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    with np.load(args.checkpoint, allow_pickle=True) as data:
        params = {
            name: wp.array(
                data[name],
                dtype=wp.float32,
                device=args.device,
            )
            for name in required
        }
    q_warp = wp.array(q_np, dtype=wp.float64, device=args.device)
    output = wp.zeros(
        (args.samples, 3),
        dtype=wp.float64,
        device=args.device,
    )
    wp.launch(
        evaluate_weights,
        dim=args.samples,
        inputs=[
            q_warp,
            params["w1"],
            params["b1"],
            params["w2"],
            params["b2"],
            params["w3"],
            params["b3"],
            params["w4"],
            params["b4"],
            output,
        ],
        device=args.device,
    )
    wp.synchronize()
    maximum = float(np.max(np.abs(output.numpy() - expected)))
    print(f"torch_warp_weight_max_abs_diff={maximum:.9e}")
    if maximum > args.tolerance:
        raise RuntimeError(
            f"Torch/Warp mixed-forward mismatch {maximum:.3e} exceeds "
            f"{args.tolerance:.3e}"
        )
    print("torch_warp_mixed_forward=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--tolerance", type=float, default=5.0e-7)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
