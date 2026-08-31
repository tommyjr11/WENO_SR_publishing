#!/usr/bin/env python3
"""Pure Torch WENO5 MLP core matching the deployed 5->10->6->6->3 runner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT3 = 3.0 ** 0.5
LR_VALUES = (1, 2, 3, 4)
MLP_INPUTS = 5
MLP_HIDDEN1 = 10
MLP_HIDDEN2 = 6
MLP_HIDDEN3 = 6
MLP_OUTPUTS = 3
BADNESS_RATIO_SCALE = 3.0
BADNESS_RATIO_POWER = 2.0
ARCH_TAG = "shared_direct_beta_ratio_5_10_6_6_3"


class SharedBadnessMLP(torch.nn.Module):
    """The exact small WENO5 MLP shape consumed by the Warp WENO5 runner.

    Solver tensors stay float64; only the MLP weights and hidden activations are
    float32. The output is cast back to the input dtype before WENO normalization.
    """

    def __init__(self, seed: int = 41) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self.w1 = torch.nn.Parameter(
            torch.randn((MLP_INPUTS, MLP_HIDDEN1), generator=gen, dtype=torch.float32)
            * np.sqrt(1.0 / MLP_INPUTS)
        )
        self.b1 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN1,), dtype=torch.float32))
        self.w2 = torch.nn.Parameter(
            torch.randn((MLP_HIDDEN1, MLP_HIDDEN2), generator=gen, dtype=torch.float32)
            * np.sqrt(1.0 / MLP_HIDDEN1)
        )
        self.b2 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN2,), dtype=torch.float32))
        self.w3 = torch.nn.Parameter(
            torch.randn((MLP_HIDDEN2, MLP_HIDDEN3), generator=gen, dtype=torch.float32)
            * np.sqrt(1.0 / MLP_HIDDEN2)
        )
        self.b3 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN3,), dtype=torch.float32))
        self.w4 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN3, MLP_OUTPUTS), dtype=torch.float32))
        self.b4 = torch.nn.Parameter(torch.zeros((MLP_OUTPUTS,), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out_dtype = features.dtype
        features = features.to(torch.float32)
        h = features @ self.w1 + self.b1
        h = h * torch.sigmoid(h)
        h = h @ self.w2 + self.b2
        h = h * torch.sigmoid(h)
        h = h @ self.w3 + self.b3
        h = h * torch.sigmoid(h)
        raw = h @ self.w4 + self.b4
        badness = 6.0 * torch.tanh(raw / 6.0)
        return torch.softmax(badness, dim=-1).to(out_dtype)


def expected_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "w1": (1, 5, 10),
        "b1": (1, 10),
        "w2": (1, 10, 6),
        "b2": (1, 6),
        "w3": (1, 6, 6),
        "b3": (1, 6),
        "w4": (1, 6, 3),
        "b4": (1, 3),
    }


def checkpoint_payload(model: SharedBadnessMLP, meta: dict[str, object]) -> dict[str, np.ndarray]:
    payload = {
        "w1": model.w1.detach().cpu().numpy()[None, :, :].astype(np.float32),
        "b1": model.b1.detach().cpu().numpy()[None, :].astype(np.float32),
        "w2": model.w2.detach().cpu().numpy()[None, :, :].astype(np.float32),
        "b2": model.b2.detach().cpu().numpy()[None, :].astype(np.float32),
        "w3": model.w3.detach().cpu().numpy()[None, :, :].astype(np.float32),
        "b3": model.b3.detach().cpu().numpy()[None, :].astype(np.float32),
        "w4": model.w4.detach().cpu().numpy()[None, :, :].astype(np.float32),
        "b4": model.b4.detach().cpu().numpy()[None, :].astype(np.float32),
    }
    full_meta = {
        "mlp_architecture": f"{ARCH_TAG}_power2_normdelta_scale",
        "mlp_features": (
            "[delta0/max_delta, delta1/max_delta, delta2/max_delta, gamma_s, "
            "clipped((log10(max_delta/q_scale)+16)/16)]"
        ),
        "mlp_weight_formula": (
            "shared hidden MLP direct beta-like badness logits: 5->10->6->6->3; "
            "r=softmax(6*tanh(raw/6)); beta=3*r; omega=normalize(d/(beta+1e-12)^2)"
        ),
        "precision": "mlp_float32_state_float64",
    }
    full_meta.update(meta)
    payload["meta_json"] = np.array(json.dumps(full_meta, sort_keys=True), dtype=np.str_)
    return payload


def save_checkpoint(path: Path, model: SharedBadnessMLP, meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **checkpoint_payload(model, meta))


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> SharedBadnessMLP:
    data = np.load(path, allow_pickle=True)
    shapes = expected_shapes()
    missing = [name for name in shapes if name not in data.files]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    wrong = {name: data[name].shape for name, shape in shapes.items() if data[name].shape != shape}
    if wrong:
        raise ValueError(f"{path} has incompatible WENO5 MLP shapes: {wrong}")
    model = SharedBadnessMLP(seed=0).to(device)
    with torch.no_grad():
        model.w1.copy_(torch.as_tensor(data["w1"][0], device=device, dtype=torch.float32))
        model.b1.copy_(torch.as_tensor(data["b1"][0], device=device, dtype=torch.float32))
        model.w2.copy_(torch.as_tensor(data["w2"][0], device=device, dtype=torch.float32))
        model.b2.copy_(torch.as_tensor(data["b2"][0], device=device, dtype=torch.float32))
        model.w3.copy_(torch.as_tensor(data["w3"][0], device=device, dtype=torch.float32))
        model.b3.copy_(torch.as_tensor(data["b3"][0], device=device, dtype=torch.float32))
        model.w4.copy_(torch.as_tensor(data["w4"][0], device=device, dtype=torch.float32))
        model.b4.copy_(torch.as_tensor(data["b4"][0], device=device, dtype=torch.float32))
    return model


def torch_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def weno5_gamma_s(q: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3, q4 = [q[:, i] for i in range(5)]
    eps = 1.0e-15
    d20 = q0 - 2.0 * q1 + q2
    d21 = q1 - 2.0 * q2 + q3
    d22 = q2 - 2.0 * q3 + q4
    g0 = torch.abs(d20) / (torch.abs(q1 - q0) + torch.abs(q2 - q1) + eps)
    g1 = torch.abs(d21) / (torch.abs(q2 - q1) + torch.abs(q3 - q2) + eps)
    g2 = torch.abs(d22) / (torch.abs(q3 - q2) + torch.abs(q4 - q3) + eps)
    return torch.clamp(torch.maximum(torch.maximum(g0, g1), g2), 0.0, 1.0)


def weno5_delta_max(q: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3, q4 = [q[:, i] for i in range(5)]
    d20 = q0 - 2.0 * q1 + q2
    d21 = q1 - 2.0 * q2 + q3
    d22 = q2 - 2.0 * q3 + q4
    delta0 = (13.0 / 12.0) * torch.abs(d20) + 0.25 * torch.abs(q0 - 4.0 * q1 + 3.0 * q2)
    delta1 = (13.0 / 12.0) * torch.abs(d21) + 0.25 * torch.abs(q1 - q3)
    delta2 = (13.0 / 12.0) * torch.abs(d22) + 0.25 * torch.abs(3.0 * q2 - 4.0 * q3 + q4)
    return torch.maximum(torch.maximum(delta0, delta1), delta2)


def weno5_features(q: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3, q4 = [q[:, i] for i in range(5)]
    d20 = q0 - 2.0 * q1 + q2
    d21 = q1 - 2.0 * q2 + q3
    d22 = q2 - 2.0 * q3 + q4
    delta0 = (13.0 / 12.0) * torch.abs(d20) + 0.25 * torch.abs(q0 - 4.0 * q1 + 3.0 * q2)
    delta1 = (13.0 / 12.0) * torch.abs(d21) + 0.25 * torch.abs(q1 - q3)
    delta2 = (13.0 / 12.0) * torch.abs(d22) + 0.25 * torch.abs(3.0 * q2 - 4.0 * q3 + q4)
    delta_max = torch.maximum(torch.maximum(delta0, delta1), delta2)
    inv_delta_max = 1.0 / torch.clamp(delta_max, min=1.0e-15)
    gamma_s = weno5_gamma_s(q)
    q_scale = torch.clamp(torch.max(torch.abs(q), dim=1).values, min=1.0)
    relative_scale = torch.clamp(delta_max / q_scale, min=1.0e-30)
    scale_feature = torch.clamp((torch.log10(relative_scale) + 16.0) / 16.0, 0.0, 1.0)
    return torch.stack(
        (delta0 * inv_delta_max, delta1 * inv_delta_max, delta2 * inv_delta_max, gamma_s, scale_feature),
        dim=1,
    )


def plateau_mask(q: torch.Tensor) -> torch.Tensor:
    q_scale = torch.clamp(torch.max(torch.abs(q), dim=1).values, min=1.0)
    return weno5_delta_max(q) <= 1.0e-13 * q_scale


def candidate_values(q: torch.Tensor, lr: int) -> torch.Tensor:
    q0, q1, q2, q3, q4 = [q[:, i] for i in range(5)]
    if lr == 1:
        s0 = (1.0 / 3.0) * q0 - (7.0 / 6.0) * q1 + (11.0 / 6.0) * q2
        s1 = -(1.0 / 6.0) * q1 + (5.0 / 6.0) * q2 + (1.0 / 3.0) * q3
        s2 = (1.0 / 3.0) * q2 + (5.0 / 6.0) * q3 - (1.0 / 6.0) * q4
    elif lr == 2:
        s0 = -(1.0 / 6.0) * q0 + (5.0 / 6.0) * q1 + (1.0 / 3.0) * q2
        s1 = (1.0 / 3.0) * q1 + (5.0 / 6.0) * q2 - (1.0 / 6.0) * q3
        s2 = (11.0 / 6.0) * q2 - (7.0 / 6.0) * q3 + (1.0 / 3.0) * q4
    elif lr == 3:
        s0 = (-ROOT3 / 12.0) * q0 + (ROOT3 / 3.0) * q1 + (1.0 - ROOT3 / 4.0) * q2
        s1 = (ROOT3 / 12.0) * q1 + q2 - (ROOT3 / 12.0) * q3
        s2 = (1.0 + ROOT3 / 4.0) * q2 - (ROOT3 / 3.0) * q3 + (ROOT3 / 12.0) * q4
    elif lr == 4:
        s0 = (ROOT3 / 12.0) * q0 - (ROOT3 / 3.0) * q1 + (1.0 + ROOT3 / 4.0) * q2
        s1 = (-ROOT3 / 12.0) * q1 + q2 + (ROOT3 / 12.0) * q3
        s2 = (1.0 - ROOT3 / 4.0) * q2 + (ROOT3 / 3.0) * q3 - (ROOT3 / 12.0) * q4
    else:
        raise ValueError(f"invalid lr={lr}")
    return torch.stack((s0, s1, s2), dim=1)


def optimal_d(lr: int, device: torch.device | str) -> torch.Tensor:
    if lr == 1:
        values = (0.1, 0.6, 0.3)
    elif lr == 2:
        values = (0.3, 0.6, 0.1)
    elif lr == 3:
        values = ((210.0 + ROOT3) / 1080.0, 11.0 / 18.0, (210.0 - ROOT3) / 1080.0)
    elif lr == 4:
        values = ((210.0 - ROOT3) / 1080.0, 11.0 / 18.0, (210.0 + ROOT3) / 1080.0)
    else:
        raise ValueError(f"invalid lr={lr}")
    return torch.as_tensor(values, device=device, dtype=torch.float64)


def omega_from_ratio(r: torch.Tensor, lr: int) -> torch.Tensor:
    d = optimal_d(lr, r.device).reshape(1, 3)
    beta = BADNESS_RATIO_SCALE * r
    alpha = d / torch.pow(beta + 1.0e-12, BADNESS_RATIO_POWER)
    return alpha / torch.sum(alpha, dim=1, keepdim=True)


def classical_beta(q: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3, q4 = [q[:, i] for i in range(5)]
    b0 = (13.0 / 12.0) * torch.square(q0 - 2.0 * q1 + q2) + 0.25 * torch.square(q0 - 4.0 * q1 + 3.0 * q2)
    b1 = (13.0 / 12.0) * torch.square(q1 - 2.0 * q2 + q3) + 0.25 * torch.square(q1 - q3)
    b2 = (13.0 / 12.0) * torch.square(q2 - 2.0 * q3 + q4) + 0.25 * torch.square(3.0 * q2 - 4.0 * q3 + q4)
    return torch.stack((b0, b1, b2), dim=1)


def classical_omega(q: torch.Tensor, lr: int, eps: float = 1.0e-6) -> torch.Tensor:
    beta = classical_beta(q)
    d = optimal_d(lr, q.device).reshape(1, 3)
    alpha = d / torch.square(beta + eps)
    return alpha / torch.sum(alpha, dim=1, keepdim=True)


# reconstruction target x-offsets from the center cell, in cell units:
# lr=1 -> +1/2 (i+1/2 left state), lr=2 -> -1/2 (i-1/2 right state),
# lr=3/4 -> the two interior Gauss points -/+ sqrt(3)/6.
LR_TARGET_X = {1: 0.5, 2: -0.5, 3: -ROOT3 / 6.0, 4: ROOT3 / 6.0}


def check_weno5_coefficients() -> None:
    """First-principles self-check of candidate stencils and linear weights.

    Exact cell averages of monomials x^k over cells [j-1/2, j+1/2],
    j = -2..2, are pushed through the reconstruction:
      * each 3-cell substencil must reproduce point values of any degree<=2
        polynomial exactly (3rd-order candidates);
      * the d-weighted combination must reproduce degree<=4 polynomials
        exactly at all four target points (5th-order linear scheme).
    Any wrong coefficient in candidate_values / optimal_d fails the assert."""
    device = torch.device("cpu")
    centers = torch.arange(-2.0, 3.0, dtype=torch.float64)
    # cell average of x^k over [c-1/2, c+1/2] = (F(c+1/2)-F(c-1/2)), F=x^(k+1)/(k+1)
    avgs = []
    for k in range(5):
        hi = torch.pow(centers + 0.5, k + 1) / (k + 1)
        lo = torch.pow(centers - 0.5, k + 1) / (k + 1)
        avgs.append((hi - lo).reshape(1, 5))
    for lr in LR_VALUES:
        xt = LR_TARGET_X[lr]
        d = optimal_d(lr, device).reshape(1, 3)
        assert abs(float(d.sum()) - 1.0) < 1.0e-14, f"lr={lr}: sum(d) != 1"
        for k in range(5):
            cand = candidate_values(avgs[k], lr)          # (1,3)
            exact = xt ** k
            if k <= 2:  # every substencil must be exact for degree <= 2
                err = torch.max(torch.abs(cand - exact))
                assert float(err) < 1.0e-12, \
                    f"lr={lr} deg={k}: candidate error {float(err):.3e}"
            full = float(torch.sum(d * cand))             # 5th-order combo
            assert abs(full - exact) < 1.0e-12, \
                f"lr={lr} deg={k}: linear-scheme error {abs(full - exact):.3e}"
    print("check_weno5_coefficients: all candidate stencils 3rd-order exact, "
          "d-combinations 5th-order exact at x=+1/2,-1/2,-s3/6,+s3/6", flush=True)
