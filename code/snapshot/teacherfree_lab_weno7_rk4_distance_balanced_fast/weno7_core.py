#!/usr/bin/env python3
"""Self-contained WENO7 reconstruction and reflection-equivariant MLP."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

R = 4
FULL_STENCIL = 7
INPUTS = 6
DEFAULT_HIDDEN = (24, 16, 16)
OUTPUTS = 4
LR_VALUES = (1, 2, 3, 4)
LR_OFFSETS = (0.5, -0.5, -math.sqrt(3.0) / 6.0, math.sqrt(3.0) / 6.0)
BADNESS_RATIO_SCALE = 4.0
BADNESS_RATIO_POWER = 2.0
ENO_CUTOFF = 4.0e-7

_S = math.sqrt(3.0)

CANDIDATE_COEFFS_NP: dict[int, np.ndarray] = {
    1: np.array(
        [
            [-1.0 / 4.0, 13.0 / 12.0, -23.0 / 12.0, 25.0 / 12.0],
            [1.0 / 12.0, -5.0 / 12.0, 13.0 / 12.0, 1.0 / 4.0],
            [-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0],
            [1.0 / 4.0, 13.0 / 12.0, -5.0 / 12.0, 1.0 / 12.0],
        ],
        dtype=np.float64,
    ),
    2: np.array(
        [
            [1.0 / 12.0, -5.0 / 12.0, 13.0 / 12.0, 1.0 / 4.0],
            [-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0],
            [1.0 / 4.0, 13.0 / 12.0, -5.0 / 12.0, 1.0 / 12.0],
            [25.0 / 12.0, -23.0 / 12.0, 13.0 / 12.0, -1.0 / 4.0],
        ],
        dtype=np.float64,
    ),
    3: np.array(
        [
            [
                11.0 * _S / 216.0,
                -17.0 * _S / 72.0,
                35.0 * _S / 72.0,
                (216.0 - 65.0 * _S) / 216.0,
            ],
            [
                -7.0 * _S / 216.0,
                13.0 * _S / 72.0,
                (72.0 - 7.0 * _S) / 72.0,
                -11.0 * _S / 216.0,
            ],
            [
                11.0 * _S / 216.0,
                (7.0 * _S + 72.0) / 72.0,
                -13.0 * _S / 72.0,
                7.0 * _S / 216.0,
            ],
            [
                (65.0 * _S + 216.0) / 216.0,
                -35.0 * _S / 72.0,
                17.0 * _S / 72.0,
                -11.0 * _S / 216.0,
            ],
        ],
        dtype=np.float64,
    ),
    4: np.array(
        [
            [
                -11.0 * _S / 216.0,
                17.0 * _S / 72.0,
                -35.0 * _S / 72.0,
                (65.0 * _S + 216.0) / 216.0,
            ],
            [
                7.0 * _S / 216.0,
                -13.0 * _S / 72.0,
                (7.0 * _S + 72.0) / 72.0,
                11.0 * _S / 216.0,
            ],
            [
                -11.0 * _S / 216.0,
                (72.0 - 7.0 * _S) / 72.0,
                13.0 * _S / 72.0,
                -7.0 * _S / 216.0,
            ],
            [
                (216.0 - 65.0 * _S) / 216.0,
                35.0 * _S / 72.0,
                -17.0 * _S / 72.0,
                11.0 * _S / 216.0,
            ],
        ],
        dtype=np.float64,
    ),
}

OPTIMAL_D_NP: dict[int, np.ndarray] = {
    1: np.array((1.0, 12.0, 18.0, 4.0), dtype=np.float64) / 35.0,
    2: np.array((4.0, 18.0, 12.0, 1.0), dtype=np.float64) / 35.0,
    3: np.array(
        (
            59.0 / 880.0 + 5.0 * _S / 16632.0,
            381.0 / 880.0 + 587.0 * _S / 194040.0,
            381.0 / 880.0 - 587.0 * _S / 194040.0,
            59.0 / 880.0 - 5.0 * _S / 16632.0,
        ),
        dtype=np.float64,
    ),
    4: np.array(
        (
            59.0 / 880.0 - 5.0 * _S / 16632.0,
            381.0 / 880.0 - 587.0 * _S / 194040.0,
            381.0 / 880.0 + 587.0 * _S / 194040.0,
            59.0 / 880.0 + 5.0 * _S / 16632.0,
        ),
        dtype=np.float64,
    ),
}


def torch_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is false"
        )
    return torch.device(name)


def _delta_forms(q: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3, q4, q5, q6 = q.unbind(dim=-1)
    forms = (
        (
            -2.0 * q0 + 9.0 * q1 - 18.0 * q2 + 11.0 * q3,
            -q0 + 4.0 * q1 - 5.0 * q2 + 2.0 * q3,
            -q0 + 3.0 * q1 - 3.0 * q2 + q3,
        ),
        (
            q1 - 6.0 * q2 + 3.0 * q3 + 2.0 * q4,
            q2 - 2.0 * q3 + q4,
            -q1 + 3.0 * q2 - 3.0 * q3 + q4,
        ),
        (
            -2.0 * q2 - 3.0 * q3 + 6.0 * q4 - q5,
            q2 - 2.0 * q3 + q4,
            -q2 + 3.0 * q3 - 3.0 * q4 + q5,
        ),
        (
            -11.0 * q3 + 18.0 * q4 - 9.0 * q5 + 2.0 * q6,
            2.0 * q3 - 5.0 * q4 + 4.0 * q5 - q6,
            -q3 + 3.0 * q4 - 3.0 * q5 + q6,
        ),
    )
    return torch.stack(
        tuple(torch.stack(item, dim=-1) for item in forms), dim=-2
    )


def weno7_gamma_s(q: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-15
    d2 = q[..., :-2] - 2.0 * q[..., 1:-1] + q[..., 2:]
    denom = (
        torch.abs(q[..., 1:-1] - q[..., :-2])
        + torch.abs(q[..., 2:] - q[..., 1:-1])
        + eps
    )
    return torch.clamp(
        torch.max(torch.abs(d2) / denom, dim=-1).values, 0.0, 1.0
    )


def delta_values(q: torch.Tensor) -> torch.Tensor:
    forms = torch.abs(_delta_forms(q))
    coeff = torch.as_tensor(
        (1.0 / 36.0, 13.0 / 12.0, 781.0 / 720.0),
        device=q.device,
        dtype=q.dtype,
    )
    return torch.sum(forms * coeff, dim=-1)


def classical_beta(q: torch.Tensor) -> torch.Tensor:
    forms = _delta_forms(q)
    coeff = torch.as_tensor(
        (1.0 / 36.0, 13.0 / 12.0, 781.0 / 720.0),
        device=q.device,
        dtype=q.dtype,
    )
    return torch.sum(forms.square() * coeff, dim=-1)


def weno7_features(q: torch.Tensor) -> torch.Tensor:
    delta = delta_values(q)
    delta_max = torch.max(delta, dim=-1).values
    norm_delta = delta / torch.clamp(delta_max.unsqueeze(-1), min=1.0e-15)
    q_scale = torch.clamp(torch.max(torch.abs(q), dim=-1).values, min=1.0)
    relative_scale = torch.clamp(delta_max / q_scale, min=1.0e-30)
    scale_feature = torch.clamp(
        (torch.log10(relative_scale) + 16.0) / 16.0, 0.0, 1.0
    )
    return torch.cat(
        (
            norm_delta,
            weno7_gamma_s(q).unsqueeze(-1),
            scale_feature.unsqueeze(-1),
        ),
        dim=-1,
    )


def plateau_mask(q: torch.Tensor) -> torch.Tensor:
    delta_max = torch.max(delta_values(q), dim=-1).values
    q_scale = torch.clamp(torch.max(torch.abs(q), dim=-1).values, min=1.0)
    return delta_max <= 1.0e-13 * q_scale


def candidate_values(q: torch.Tensor, lr: int) -> torch.Tensor:
    if lr not in LR_VALUES:
        raise ValueError(f"invalid WENO7 reconstruction head lr={lr}")
    coeff = torch.as_tensor(
        CANDIDATE_COEFFS_NP[lr], device=q.device, dtype=q.dtype
    )
    values = []
    for k in range(R):
        values.append(torch.sum(q[..., k : k + R] * coeff[k], dim=-1))
    return torch.stack(values, dim=-1)


def optimal_d(lr: int, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if lr not in LR_VALUES:
        raise ValueError(f"invalid WENO7 reconstruction head lr={lr}")
    return torch.as_tensor(OPTIMAL_D_NP[lr], device=device, dtype=dtype)


def omega_from_ratio(ratio: torch.Tensor, lr: int) -> torch.Tensor:
    d = optimal_d(lr, ratio.device, ratio.dtype)
    beta = BADNESS_RATIO_SCALE * ratio
    alpha = d / torch.pow(beta + 1.0e-12, BADNESS_RATIO_POWER)
    return alpha / torch.sum(alpha, dim=-1, keepdim=True)


def classical_omega(q: torch.Tensor, lr: int, eps: float = 1.0e-6) -> torch.Tensor:
    beta = classical_beta(q)
    d = optimal_d(lr, q.device, q.dtype)
    alpha = d / torch.square(beta + eps)
    return alpha / torch.sum(alpha, dim=-1, keepdim=True)


def apply_eno_cutoff(omega: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return omega
    kept = torch.where(omega > ENO_CUTOFF, omega, torch.zeros_like(omega))
    denom = torch.sum(kept, dim=-1, keepdim=True)
    return torch.where(denom > 0.0, kept / denom, omega)


class ReflectionSymmetricBadnessMLP(torch.nn.Module):
    """Shared badness MLP with reflection equivariance by construction."""

    def __init__(
        self,
        hidden: tuple[int, int, int] = DEFAULT_HIDDEN,
        seed: int = 0,
    ) -> None:
        super().__init__()
        h1, h2, h3 = (int(value) for value in hidden)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        def random_weight(shape: tuple[int, int], fan_in: int) -> torch.Tensor:
            return (
                torch.randn(shape, generator=gen, dtype=torch.float64)
                * math.sqrt(1.0 / float(fan_in))
            )

        self.w1 = torch.nn.Parameter(random_weight((INPUTS, h1), INPUTS))
        self.b1 = torch.nn.Parameter(torch.zeros(h1, dtype=torch.float64))
        self.w2 = torch.nn.Parameter(random_weight((h1, h2), h1))
        self.b2 = torch.nn.Parameter(torch.zeros(h2, dtype=torch.float64))
        self.w3 = torch.nn.Parameter(random_weight((h2, h3), h2))
        self.b3 = torch.nn.Parameter(torch.zeros(h3, dtype=torch.float64))
        self.w4 = torch.nn.Parameter(torch.zeros((h3, OUTPUTS), dtype=torch.float64))
        self.b4 = torch.nn.Parameter(torch.zeros(OUTPUTS, dtype=torch.float64))

    @staticmethod
    def reflect_features(features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                torch.flip(features[..., :4], dims=(-1,)),
                features[..., 4:],
            ),
            dim=-1,
        )

    @staticmethod
    def reflect_ratios(ratios: torch.Tensor) -> torch.Tensor:
        return torch.flip(ratios, dims=(-1,))

    @staticmethod
    def _swish(value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)

    def raw_forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self._swish(features @ self.w1 + self.b1)
        hidden = self._swish(hidden @ self.w2 + self.b2)
        hidden = self._swish(hidden @ self.w3 + self.b3)
        raw = hidden @ self.w4 + self.b4
        badness = 6.0 * torch.tanh(raw / 6.0)
        return torch.softmax(badness, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        direct = self.raw_forward(features)
        reflected = self.raw_forward(self.reflect_features(features))
        return 0.5 * (direct + self.reflect_ratios(reflected))


@torch.no_grad()
def reflection_defect(
    model: ReflectionSymmetricBadnessMLP, features: torch.Tensor
) -> float:
    lhs = model(model.reflect_features(features))
    rhs = model.reflect_ratios(model(features))
    return float(torch.max(torch.abs(lhs - rhs)))


def checkpoint_shapes(hidden: tuple[int, int, int]) -> dict[str, tuple[int, ...]]:
    h1, h2, h3 = hidden
    return {
        "w1": (1, INPUTS, h1),
        "b1": (1, h1),
        "w2": (1, h1, h2),
        "b2": (1, h2),
        "w3": (1, h2, h3),
        "b3": (1, h3),
        "w4": (1, h3, OUTPUTS),
        "b4": (1, OUTPUTS),
    }


def save_checkpoint(
    path: Path,
    model: ReflectionSymmetricBadnessMLP,
    meta: dict[str, Any],
) -> None:
    hidden = (
        int(model.w1.shape[1]),
        int(model.w2.shape[1]),
        int(model.w3.shape[1]),
    )
    payload: dict[str, np.ndarray] = {}
    for name in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"):
        payload[name] = getattr(model, name).detach().cpu().numpy()[None, ...]
    metadata = dict(meta)
    metadata.update(
        {
            "mlp_architecture": (
                f"reflection_sym_direct_beta_ratio_6_"
                f"{hidden[0]}_{hidden[1]}_{hidden[2]}_4"
            ),
            "hidden": list(hidden),
            "precision": "float64",
            "reflection_formula": "0.5*(M(x)+P4*M(P6*x))",
            "deployment_requires_reflection_symmetrization": True,
        }
    )
    payload["meta_json"] = np.array(
        json.dumps(metadata, sort_keys=True), dtype=np.str_
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> ReflectionSymmetricBadnessMLP:
    data = np.load(path, allow_pickle=True)
    required = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"{path} is missing arrays: {missing}")
    if data["w1"].ndim != 3 or data["w1"].shape[1] != INPUTS:
        raise ValueError(f"{path} has incompatible input shape {data['w1'].shape}")
    if data["w4"].ndim != 3 or data["w4"].shape[2] != OUTPUTS:
        raise ValueError(f"{path} has incompatible output shape {data['w4'].shape}")
    hidden = (
        int(data["w1"].shape[2]),
        int(data["w2"].shape[2]),
        int(data["w3"].shape[2]),
    )
    expected = checkpoint_shapes(hidden)
    wrong = {
        name: data[name].shape
        for name, shape in expected.items()
        if data[name].shape != shape
    }
    if wrong:
        raise ValueError(f"{path} has incompatible WENO7 MLP shapes: {wrong}")
    model = ReflectionSymmetricBadnessMLP(hidden=hidden, seed=0).to(device)
    with torch.no_grad():
        for name in required:
            getattr(model, name).copy_(
                torch.as_tensor(
                    data[name][0], device=device, dtype=torch.float64
                )
            )
    return model


def _cell_average_monomial(cell: int, degree: int) -> float:
    if degree == 0:
        return 1.0
    upper = (cell + 0.5) ** (degree + 1)
    lower = (cell - 0.5) ** (degree + 1)
    return (upper - lower) / float(degree + 1)


def check_weno7_coefficients(tol: float = 3.0e-12) -> None:
    """Fail loudly if any candidate, optimal weight, or reflection is wrong."""
    for lr in LR_VALUES:
        if abs(float(np.sum(OPTIMAL_D_NP[lr])) - 1.0) > tol:
            raise AssertionError(f"optimal weights do not sum to one for lr={lr}")
        offset = LR_OFFSETS[lr - 1]
        for degree in range(4):
            q = np.array(
                [_cell_average_monomial(cell, degree) for cell in range(-3, 4)]
            )
            candidate = np.array(
                [
                    np.dot(
                        CANDIDATE_COEFFS_NP[lr][k],
                        q[k : k + R],
                    )
                    for k in range(R)
                ]
            )
            exact = offset**degree
            if np.max(np.abs(candidate - exact)) > tol:
                raise AssertionError(
                    f"candidate polynomial check failed lr={lr} degree={degree}"
                )
        for degree in range(7):
            q = np.array(
                [_cell_average_monomial(cell, degree) for cell in range(-3, 4)]
            )
            candidate = np.array(
                [
                    np.dot(
                        CANDIDATE_COEFFS_NP[lr][k],
                        q[k : k + R],
                    )
                    for k in range(R)
                ]
            )
            combined = float(np.dot(OPTIMAL_D_NP[lr], candidate))
            if abs(combined - offset**degree) > 20.0 * tol:
                raise AssertionError(
                    f"linear WENO7 check failed lr={lr} degree={degree}"
                )
    if not np.allclose(
        CANDIDATE_COEFFS_NP[2],
        np.flip(CANDIDATE_COEFFS_NP[1], axis=(0, 1)),
        rtol=0.0,
        atol=tol,
    ):
        raise AssertionError("lr=1/lr=2 candidate reflection failed")
    if not np.allclose(
        CANDIDATE_COEFFS_NP[4],
        np.flip(CANDIDATE_COEFFS_NP[3], axis=(0, 1)),
        rtol=0.0,
        atol=tol,
    ):
        raise AssertionError("lr=3/lr=4 candidate reflection failed")
    if not np.allclose(
        OPTIMAL_D_NP[2], OPTIMAL_D_NP[1][::-1], rtol=0.0, atol=tol
    ):
        raise AssertionError("lr=1/lr=2 optimal-weight reflection failed")
    if not np.allclose(
        OPTIMAL_D_NP[4], OPTIMAL_D_NP[3][::-1], rtol=0.0, atol=tol
    ):
        raise AssertionError("lr=3/lr=4 optimal-weight reflection failed")

