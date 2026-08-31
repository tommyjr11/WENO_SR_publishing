#!/usr/bin/env python3
"""Offline FVM WENO5-NN pretraining on Nogueira-style canonical stencils.

The checkpoint format intentionally matches the shared direct beta-ratio
5->10->6->6->3 MLP consumed by weno5_rk3_diff.py and run_weno5_circle_mlp_compare.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import warp_weno5_helpers as wh

try:
    from train_weno5_mlp import run_circle_sod_validation, write_circle_validation_outputs
except ModuleNotFoundError:
    run_circle_sod_validation = None
    write_circle_validation_outputs = None


ROOT3 = 3.0**0.5
KIND_NAMES = (
    "discontinuity",
    "sine",
    "sawtooth",
    "tanh",
    "cubic",
    "quadratic",
    "linear",
    "constant",
)
NOGUEIRA_COUNTS = np.array([655861, 460355, 72135, 82837, 52700, 63935, 2893, 393], dtype=np.float64)
NOGUEIRA_VALIDATION_FRACTION = 0.2
MLP_INPUTS = 5
MLP_HIDDEN1 = 10
MLP_HIDDEN2 = 6
MLP_HIDDEN3 = 6
MLP_OUTPUTS = 3
BADNESS_RATIO_SCALE = 3.0
BADNESS_RATIO_POWER = 2.0
LR_VALUES = (1, 2, 3, 4)
GAMMA_FILTERS = {
    "discontinuity": (0.99, 1.0),
    "sawtooth": (0.9, 1.0),
    "sine": (0.0, 0.9),
    "cubic": (0.0, 0.7),
    "quadratic": (0.0, 0.7),
}


@dataclass
class Batch:
    q: torch.Tensor
    targets: torch.Tensor
    kind_ids: torch.Tensor
    gamma_s: torch.Tensor


class SharedBadnessMLP(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self.w1 = torch.nn.Parameter(torch.randn((MLP_INPUTS, MLP_HIDDEN1), generator=gen, dtype=torch.float64) * np.sqrt(1.0 / MLP_INPUTS))
        self.b1 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN1,), dtype=torch.float64))
        self.w2 = torch.nn.Parameter(torch.randn((MLP_HIDDEN1, MLP_HIDDEN2), generator=gen, dtype=torch.float64) * np.sqrt(1.0 / MLP_HIDDEN1))
        self.b2 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN2,), dtype=torch.float64))
        self.w3 = torch.nn.Parameter(torch.randn((MLP_HIDDEN2, MLP_HIDDEN3), generator=gen, dtype=torch.float64) * np.sqrt(1.0 / MLP_HIDDEN2))
        self.b3 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN3,), dtype=torch.float64))
        self.w4 = torch.nn.Parameter(torch.zeros((MLP_HIDDEN3, MLP_OUTPUTS), dtype=torch.float64))
        self.b4 = torch.nn.Parameter(torch.zeros((MLP_OUTPUTS,), dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = features @ self.w1 + self.b1
        h = h * torch.sigmoid(h)
        h = h @ self.w2 + self.b2
        h = h * torch.sigmoid(h)
        h = h @ self.w3 + self.b3
        h = h * torch.sigmoid(h)
        raw = h @ self.w4 + self.b4
        badness = 6.0 * torch.tanh(raw / 6.0)
        return torch.softmax(badness, dim=-1)


def torch_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def rand_uniform(shape: tuple[int, ...], low: float, high: float, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, device=device, generator=generator, dtype=torch.float64)


def cell_centers(
    batch_size: int,
    dx: float,
    device: torch.device,
    generator: torch.Generator,
    oscillatory: bool = False,
    paper_grid_centers: bool = False,
) -> torch.Tensor:
    margin = 2.5 * dx
    if paper_grid_centers:
        low = margin if oscillatory else -1.0 + margin
        high = 1.0 - margin
        n_grid = max(1, int(np.floor((high - low) / dx + 1.0e-12)) + 1)
        idx = torch.randint(n_grid, (batch_size,), device=device, generator=generator)
        return low + idx.to(torch.float64) * dx
    if oscillatory:
        return rand_uniform((batch_size,), margin, 1.0 - margin, device, generator)
    return rand_uniform((batch_size,), -1.0 + margin, 1.0 - margin, device, generator)


def canonical_values(kind: str, x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    if kind == "discontinuity":
        return torch.where(x < params["x0"], params["ul"], params["ur"])
    if kind == "sine":
        return torch.sin(params["k"] * np.pi * x + params["phase"])
    if kind == "sawtooth":
        return params["slope"] * x + params["delta"] * (x >= params["x0"]).to(torch.float64)
    if kind == "tanh":
        return torch.tanh(params["k"] * (x - params["x0"]))
    if kind == "cubic":
        return params["a0"] + params["a1"] * x + params["a2"] * x * x + params["a3"] * x * x * x
    if kind == "quadratic":
        return params["a0"] + params["a1"] * x + params["a2"] * x * x
    if kind == "linear":
        return params["a0"] + params["a1"] * x
    if kind == "constant":
        return torch.zeros_like(x) + params["a0"] + params["eps"]
    raise ValueError(f"unknown canonical kind {kind}")


def stable_log_cosh(x: torch.Tensor) -> torch.Tensor:
    ax = torch.abs(x)
    return ax + torch.log1p(torch.exp(-2.0 * ax)) - np.log(2.0)


def canonical_antiderivative(kind: str, x: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    if kind == "discontinuity":
        return params["ul"] * x + (params["ur"] - params["ul"]) * torch.clamp(x - params["x0"], min=0.0)
    if kind == "sine":
        a = params["k"] * np.pi
        return -torch.cos(a * x + params["phase"]) / a
    if kind == "sawtooth":
        return 0.5 * params["slope"] * x * x + params["delta"] * torch.clamp(x - params["x0"], min=0.0)
    if kind == "tanh":
        z = params["k"] * (x - params["x0"])
        return stable_log_cosh(z) / params["k"]
    if kind == "cubic":
        return (
            params["a0"] * x
            + 0.5 * params["a1"] * x * x
            + (params["a2"] / 3.0) * x * x * x
            + 0.25 * params["a3"] * x * x * x * x
        )
    if kind == "quadratic":
        return params["a0"] * x + 0.5 * params["a1"] * x * x + (params["a2"] / 3.0) * x * x * x
    if kind == "linear":
        return params["a0"] * x + 0.5 * params["a1"] * x * x
    if kind == "constant":
        return (params["a0"] + params["eps"]) * x
    raise ValueError(f"unknown canonical kind {kind}")


def canonical_cell_average(kind: str, left: torch.Tensor, right: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (canonical_antiderivative(kind, right, params) - canonical_antiderivative(kind, left, params)) / (right - left)


def make_kind_params(kind: str, n: int, x_center: torch.Tensor, dx: float, device: torch.device, generator: torch.Generator) -> dict[str, torch.Tensor]:
    def col(v: torch.Tensor) -> torch.Tensor:
        return v.reshape(n, 1)

    if kind == "discontinuity":
        ul = rand_uniform((n,), -40.0, 80.0, device, generator)
        ur = rand_uniform((n,), -40.0, 80.0, device, generator)
        too_close = torch.abs(ur - ul) < 1.0e-8
        ur = torch.where(too_close, ur + 1.0, ur)
        x0 = torch.full((n,), 0.5, device=device, dtype=torch.float64)
        return {"ul": col(ul), "ur": col(ur), "x0": col(x0)}
    if kind == "sine":
        k = rand_uniform((n,), 2.0, 27.0, device, generator)
        phase = torch.zeros((n,), device=device, dtype=torch.float64)
        return {"k": col(k), "phase": col(phase)}
    if kind == "sawtooth":
        sign = torch.where(torch.rand((n,), device=device, generator=generator) < 0.5, -1.0, 1.0).to(torch.float64)
        delta = rand_uniform((n,), 0.5, 2.0, device, generator)
        x0 = torch.full((n,), 0.5, device=device, dtype=torch.float64)
        return {"slope": col(sign), "delta": col(delta), "x0": col(x0)}
    if kind == "tanh":
        k = rand_uniform((n,), 5.0, 30.0, device, generator)
        x0 = torch.zeros((n,), device=device, dtype=torch.float64)
        return {"k": col(k), "x0": col(x0)}
    if kind == "cubic":
        return {f"a{i}": col(rand_uniform((n,), -1.0, 1.0, device, generator)) for i in range(4)}
    if kind == "quadratic":
        return {f"a{i}": col(rand_uniform((n,), -1.0, 1.0, device, generator)) for i in range(3)}
    if kind == "linear":
        return {f"a{i}": col(rand_uniform((n,), -1.0, 1.0, device, generator)) for i in range(2)}
    if kind == "constant":
        a0 = rand_uniform((n,), -1.0, 1.0, device, generator)
        eps = rand_uniform((n,), 0.0, 1.0e-13, device, generator)
        return {"a0": col(a0), "eps": col(eps)}
    raise ValueError(f"unknown canonical kind {kind}")


def make_kind_batch_unfiltered(
    kind: str,
    n: int,
    dx: float,
    device: torch.device,
    generator: torch.Generator,
    paper_grid_centers: bool,
    cell_average_mode: str,
    mirror_augmentation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_center = cell_centers(n, dx, device, generator, oscillatory=(kind == "sine"), paper_grid_centers=paper_grid_centers)
    params = make_kind_params(kind, n, x_center, dx, device, generator)

    offsets = torch.arange(-2, 3, device=device, dtype=torch.float64)
    centers = x_center[:, None] + offsets[None, :] * dx
    if cell_average_mode == "exact":
        q = canonical_cell_average(kind, centers - 0.5 * dx, centers + 0.5 * dx, params)
    elif cell_average_mode == "gauss15":
        xi = torch.as_tensor(wh.GAUSS15_XI, device=device, dtype=torch.float64)
        weights = torch.as_tensor(wh.GAUSS15_W, device=device, dtype=torch.float64)
        quad_x = centers[:, :, None] + 0.5 * dx * xi[None, None, :]
        values = canonical_values(kind, quad_x.reshape(n, -1), params).reshape(n, 5, -1)
        q = 0.5 * torch.sum(values * weights[None, None, :], dim=-1)
    else:
        raise ValueError(f"invalid cell_average_mode={cell_average_mode}")

    points = torch.stack(
        (
            x_center + 0.5 * dx,
            x_center - 0.5 * dx,
            x_center - ROOT3 * dx / 6.0,
            x_center + ROOT3 * dx / 6.0,
        ),
        dim=1,
    )
    target_points = points
    if kind in ("discontinuity", "sawtooth"):
        # FVM left/right interface reconstructions need one-sided traces.
        # At an exact jump face, LR=1 is the left trace and LR=2 is the right trace.
        trace_eps = 1.0e-12 * dx
        target_points = points.clone()
        target_points[:, 0] -= trace_eps
        target_points[:, 1] += trace_eps
    targets = canonical_values(kind, target_points, params)
    if mirror_augmentation:
        mirror = torch.rand((n,), device=device, generator=generator) < 0.5
        if bool(mirror.any().item()):
            q_m = torch.flip(q[mirror], dims=(1,))
            targets_m = targets[mirror][:, [1, 0, 3, 2]]
            q = q.clone()
            targets = targets.clone()
            q[mirror] = q_m
            targets[mirror] = targets_m
    return q, targets, weno5_gamma_s(q)


def make_kind_batch(
    kind: str,
    n: int,
    dx: float,
    device: torch.device,
    generator: torch.Generator,
    apply_gamma_filter: bool = True,
    paper_grid_centers: bool = False,
    cell_average_mode: str = "exact",
    mirror_augmentation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (not apply_gamma_filter) or kind not in GAMMA_FILTERS:
        return make_kind_batch_unfiltered(kind, n, dx, device, generator, paper_grid_centers, cell_average_mode, mirror_augmentation)

    lo, hi = GAMMA_FILTERS[kind]
    q_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    gamma_parts: list[torch.Tensor] = []
    collected = 0
    attempts = 0
    while collected < n and attempts < 64:
        request = max(2 * (n - collected), n)
        q, targets, gamma_s = make_kind_batch_unfiltered(kind, request, dx, device, generator, paper_grid_centers, cell_average_mode, mirror_augmentation)
        mask = (gamma_s >= lo) & (gamma_s <= hi)
        if int(mask.sum().item()) > 0:
            q_keep = q[mask]
            targets_keep = targets[mask]
            gamma_keep = gamma_s[mask]
            take = min(n - collected, int(q_keep.shape[0]))
            q_parts.append(q_keep[:take])
            target_parts.append(targets_keep[:take])
            gamma_parts.append(gamma_keep[:take])
            collected += take
        attempts += 1

    if collected < n:
        q, targets, gamma_s = make_kind_batch_unfiltered(kind, n - collected, dx, device, generator, paper_grid_centers, cell_average_mode, mirror_augmentation)
        q_parts.append(q)
        target_parts.append(targets)
        gamma_parts.append(gamma_s)

    return torch.cat(q_parts, dim=0), torch.cat(target_parts, dim=0), torch.cat(gamma_parts, dim=0)


def make_batch(
    batch_size: int,
    dx: float,
    kind_probs: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
    apply_gamma_filter: bool,
    paper_grid_centers: bool,
    cell_average_mode: str,
    mirror_augmentation: bool,
) -> Batch:
    kind_ids = torch.multinomial(kind_probs, batch_size, replacement=True, generator=generator)
    q = torch.empty((batch_size, 5), device=device, dtype=torch.float64)
    targets = torch.empty((batch_size, 4), device=device, dtype=torch.float64)
    gamma_s = torch.empty((batch_size,), device=device, dtype=torch.float64)
    for kind_id, kind in enumerate(KIND_NAMES):
        mask = kind_ids == kind_id
        n = int(mask.sum().item())
        if n == 0:
            continue
        q_k, targets_k, gamma_k = make_kind_batch(kind, n, dx, device, generator, apply_gamma_filter, paper_grid_centers, cell_average_mode, mirror_augmentation)
        q[mask] = q_k
        targets[mask] = targets_k
        gamma_s[mask] = gamma_k
    return Batch(q=q, targets=targets, kind_ids=kind_ids, gamma_s=gamma_s)


def make_validation_batches(samples_per_kind: int, dx: float, device: torch.device, seed: int, apply_gamma_filter: bool, paper_grid_centers: bool, cell_average_mode: str, mirror_augmentation: bool) -> list[Batch]:
    batches: list[Batch] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for kind_id, kind in enumerate(KIND_NAMES):
        q, targets, gamma_s = make_kind_batch(kind, samples_per_kind, dx, device, gen, apply_gamma_filter, paper_grid_centers, cell_average_mode, mirror_augmentation)
        kind_ids = torch.full((samples_per_kind,), kind_id, device=device, dtype=torch.long)
        batches.append(Batch(q=q, targets=targets, kind_ids=kind_ids, gamma_s=gamma_s))
    return batches


def make_validation_batches_by_counts(counts: np.ndarray, dx: float, device: torch.device, seed: int, apply_gamma_filter: bool, paper_grid_centers: bool, cell_average_mode: str, mirror_augmentation: bool) -> list[Batch]:
    batches: list[Batch] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for kind_id, kind in enumerate(KIND_NAMES):
        n = max(1, int(counts[kind_id]))
        q, targets, gamma_s = make_kind_batch(kind, n, dx, device, gen, apply_gamma_filter, paper_grid_centers, cell_average_mode, mirror_augmentation)
        kind_ids = torch.full((n,), kind_id, device=device, dtype=torch.long)
        batches.append(Batch(q=q, targets=targets, kind_ids=kind_ids, gamma_s=gamma_s))
    return batches


def canonical_kind_probabilities(discontinuity_multiplier: float) -> np.ndarray:
    counts = NOGUEIRA_COUNTS.copy()
    counts[KIND_NAMES.index("discontinuity")] *= discontinuity_multiplier
    return counts / counts.sum()


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
    delta0_norm = delta0 * inv_delta_max
    delta1_norm = delta1 * inv_delta_max
    delta2_norm = delta2 * inv_delta_max
    gamma_s = weno5_gamma_s(q)
    q_scale = torch.clamp(torch.max(torch.abs(q), dim=1).values, min=1.0)
    relative_scale = torch.clamp(delta_max / q_scale, min=1.0e-30)
    scale_feature = torch.clamp((torch.log10(relative_scale) + 16.0) / 16.0, 0.0, 1.0)
    return torch.stack((delta0_norm, delta1_norm, delta2_norm, gamma_s, scale_feature), dim=1)


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
        raise ValueError(f"invalid LR={lr}")
    return torch.stack((s0, s1, s2), dim=1)


def optimal_d(lr: int, device: torch.device) -> torch.Tensor:
    if lr == 1:
        values = (0.1, 0.6, 0.3)
    elif lr == 2:
        values = (0.3, 0.6, 0.1)
    elif lr == 3:
        values = ((210.0 + ROOT3) / 1080.0, 11.0 / 18.0, (210.0 - ROOT3) / 1080.0)
    elif lr == 4:
        values = ((210.0 - ROOT3) / 1080.0, 11.0 / 18.0, (210.0 + ROOT3) / 1080.0)
    else:
        raise ValueError(f"invalid LR={lr}")
    return torch.as_tensor(values, device=device, dtype=torch.float64)


def omega_from_ratio(r: torch.Tensor, lr: int) -> torch.Tensor:
    d = optimal_d(lr, r.device).reshape(1, 3)
    beta = BADNESS_RATIO_SCALE * r
    alpha = d / torch.pow(beta + 1.0e-12, BADNESS_RATIO_POWER)
    return alpha / torch.sum(alpha, dim=1, keepdim=True)


def reconstruction_predictions(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    preds = []
    for lr in LR_VALUES:
        omega = omega_from_ratio(r, lr)
        stencil = candidate_values(q, lr)
        preds.append(torch.sum(omega * stencil, dim=1))
    return torch.stack(preds, dim=1)


def weight_l2(model: SharedBadnessMLP) -> torch.Tensor:
    return torch.sum(torch.square(model.w1)) + torch.sum(torch.square(model.w2)) + torch.sum(torch.square(model.w3)) + torch.sum(torch.square(model.w4))


def loss_terms(
    model: SharedBadnessMLP,
    batch: Batch,
    smooth_anchor_lambda: float,
    reconstruction_gamma_alpha: float,
    weight_l2_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = weno5_features(batch.q)
    r = model(features)
    preds = reconstruction_predictions(batch.q, r)
    sq = torch.square(preds - batch.targets)
    if reconstruction_gamma_alpha == 0.0:
        recon_weight = torch.ones_like(features[:, 3])
    else:
        recon_weight = torch.pow(torch.clamp(features[:, 3], min=0.0), reconstruction_gamma_alpha)
    raw_recon = torch.mean(sq)
    recon = torch.mean(recon_weight.reshape(-1, 1) * sq)
    smooth = torch.square(1.0 - features[:, 3]).reshape(-1, 1)
    r_target = torch.full_like(r, 1.0 / 3.0)
    anchor = torch.mean(smooth * torch.square(r - r_target))
    l2 = weight_l2(model)
    loss = recon + smooth_anchor_lambda * anchor + weight_l2_lambda * l2
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(recon.detach().cpu()),
            "raw_reconstruction_loss": float(raw_recon.detach().cpu()),
            "smooth_anchor_loss": float(anchor.detach().cpu()),
            "weight_l2_loss": float(l2.detach().cpu()),
            "gamma_mean": float(features[:, 3].mean().detach().cpu()),
            "reconstruction_weight_mean": float(recon_weight.mean().detach().cpu()),
            "r0_mean": float(r[:, 0].mean().detach().cpu()),
            "r1_mean": float(r[:, 1].mean().detach().cpu()),
            "r2_mean": float(r[:, 2].mean().detach().cpu()),
        }
    return loss, stats


def evaluate_local(
    model: SharedBadnessMLP,
    batches: list[Batch],
    smooth_anchor_lambda: float,
    reconstruction_gamma_alpha: float,
    weight_l2_lambda: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    total_weight = 0
    sums: dict[str, float] = {}
    by_kind: list[dict[str, object]] = []
    with torch.no_grad():
        for kind_id, batch in enumerate(batches):
            _, stats = loss_terms(model, batch, smooth_anchor_lambda, reconstruction_gamma_alpha, weight_l2_lambda)
            n = int(batch.q.shape[0])
            total_weight += n
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + value * n
            row = {"kind": KIND_NAMES[kind_id], "n": n}
            row.update(stats)
            by_kind.append(row)
    overall = {key: value / max(total_weight, 1) for key, value in sums.items()}
    return overall, by_kind


def save_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_gamma_histogram(path: Path, batches: list[Batch]) -> None:
    gamma = torch.cat([batch.gamma_s.detach().cpu() for batch in batches]).numpy()
    bins = np.linspace(0.0, 1.0, 11)
    hist, edges = np.histogram(gamma, bins=bins)
    rows = []
    total = max(int(hist.sum()), 1)
    for i, count in enumerate(hist):
        rows.append({"gamma_min": edges[i], "gamma_max": edges[i + 1], "count": int(count), "fraction": float(count / total)})
    save_csv(path, rows, ["gamma_min", "gamma_max", "count", "fraction"])


def checkpoint_payload(model: SharedBadnessMLP, meta: dict[str, object]) -> dict[str, np.ndarray]:
    payload = {
        "w1": model.w1.detach().cpu().numpy()[None, :, :],
        "b1": model.b1.detach().cpu().numpy()[None, :],
        "w2": model.w2.detach().cpu().numpy()[None, :, :],
        "b2": model.b2.detach().cpu().numpy()[None, :],
        "w3": model.w3.detach().cpu().numpy()[None, :, :],
        "b3": model.b3.detach().cpu().numpy()[None, :],
        "w4": model.w4.detach().cpu().numpy()[None, :, :],
        "b4": model.b4.detach().cpu().numpy()[None, :],
    }
    payload["meta_json"] = np.array(json.dumps(meta, sort_keys=True), dtype=np.str_)
    return payload


def load_checkpoint_into_model(model: SharedBadnessMLP, model_path: Path, device: torch.device) -> None:
    data = np.load(model_path, allow_pickle=True)
    expected_shapes = {
        "w1": (1, MLP_INPUTS, MLP_HIDDEN1),
        "b1": (1, MLP_HIDDEN1),
        "w2": (1, MLP_HIDDEN1, MLP_HIDDEN2),
        "b2": (1, MLP_HIDDEN2),
        "w3": (1, MLP_HIDDEN2, MLP_HIDDEN3),
        "b3": (1, MLP_HIDDEN3),
        "w4": (1, MLP_HIDDEN3, MLP_OUTPUTS),
        "b4": (1, MLP_OUTPUTS),
    }
    missing = [name for name in expected_shapes if name not in data.files]
    if missing:
        raise ValueError(f"--init-model {model_path} is missing arrays: {missing}")
    wrong = {name: data[name].shape for name, shape in expected_shapes.items() if data[name].shape != shape}
    if wrong:
        raise ValueError(
            f"--init-model {model_path} has incompatible MLP shapes: {wrong}; "
            "expected shared_direct_beta_ratio_5_10_6_6_3 checkpoints."
        )
    if "meta_json" in data.files:
        meta = str(data["meta_json"])
        if "5_10_6_6_3_power2_normdelta_scale" not in meta:
            print(
                "warning: --init-model metadata does not match current "
                "5_10_6_6_3_power2_normdelta_scale formula; old power/rawdelta/larger checkpoints "
                "will load by shape but are not mathematically equivalent."
            )
    with torch.no_grad():
        model.w1.copy_(torch.as_tensor(data["w1"][0], device=device, dtype=torch.float64))
        model.b1.copy_(torch.as_tensor(data["b1"][0], device=device, dtype=torch.float64))
        model.w2.copy_(torch.as_tensor(data["w2"][0], device=device, dtype=torch.float64))
        model.b2.copy_(torch.as_tensor(data["b2"][0], device=device, dtype=torch.float64))
        model.w3.copy_(torch.as_tensor(data["w3"][0], device=device, dtype=torch.float64))
        model.b3.copy_(torch.as_tensor(data["b3"][0], device=device, dtype=torch.float64))
        model.w4.copy_(torch.as_tensor(data["w4"][0], device=device, dtype=torch.float64))
        model.b4.copy_(torch.as_tensor(data["b4"][0], device=device, dtype=torch.float64))


def save_checkpoint(path: Path, model: SharedBadnessMLP, args: argparse.Namespace, step: int, extra: dict[str, object] | None = None) -> None:
    args_meta = vars(args).copy()
    args_meta["out_dir"] = str(args.out_dir)
    args_meta["init_model"] = str(args.init_model) if args.init_model else ""
    meta = {
        "raw_step": step,
        "accepted": step,
        "mlp_architecture": "shared_direct_beta_ratio_5_10_6_6_3_power2_normdelta_scale",
        "mlp_features": "[delta0/max_delta, delta1/max_delta, delta2/max_delta, gamma_s, clipped((log10(max_delta/q_scale)+16)/16)]",
        "mlp_weight_formula": "offline FVM canonical pretrain; 5->10->6->6->3 shared r=softmax(6*tanh(raw/6)); beta=3*r; per-LR omega=normalize(d_lr/(beta+1e-12)^2)",
        "offline_pretrain": True,
        "dx": args.dx,
        "paper_grid_centers": args.paper_grid_centers,
        "cell_average_mode": args.cell_average_mode,
        "mirror_augmentation": args.mirror_augmentation,
        "validation_mode": args.validation_mode,
        "paper_validation_fraction": args.paper_validation_fraction,
        "validation_sample_scale": args.validation_sample_scale,
        "smooth_anchor_lambda": args.smooth_anchor_lambda,
        "reconstruction_gamma_alpha": args.reconstruction_gamma_alpha,
        "weight_l2_lambda": args.weight_l2_lambda,
        "discontinuity_multiplier": args.discontinuity_multiplier,
        "canonical_kinds": KIND_NAMES,
        "nogueira_counts": NOGUEIRA_COUNTS.tolist(),
        "nogueira_total_samples": int(np.sum(NOGUEIRA_COUNTS)),
        "nogueira_train_samples_80pct": int(round(float(np.sum(NOGUEIRA_COUNTS)) * (1.0 - args.paper_validation_fraction))),
        "init_model": str(args.init_model) if args.init_model else "",
        "step_offset": args.step_offset,
        "args": args_meta,
    }
    if extra:
        meta.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **checkpoint_payload(model, meta))


def model_to_wp_params(model: SharedBadnessMLP, device: str) -> dict[str, object]:
    payload = checkpoint_payload(model, {"temporary": True})
    wp = wh.wp
    return {
        name: wp.array(payload[name], dtype=wp.float64, device=device, requires_grad=False)
        for name in ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    }


def make_sod_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        circle_validation_nx=args.sod_nx,
        circle_validation_ny=args.sod_ny,
        circle_validation_cfl=args.sod_cfl,
        circle_validation_t_end=args.sod_t_end,
        circle_validation_reference_mode="exact-sod",
        circle_validation_axis=args.sod_axis,
        circle_validation_eno_cutoff=args.sod_eno_cutoff,
        out_dir=args.out_dir,
        device=args.sod_device if args.sod_device else args.device,
        weno_space="characteristic",
    )


def plot_offline_history(out_dir: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    steps = np.array([r["step"] for r in records], dtype=np.float64)
    loss = np.array([r["train_loss"] for r in records], dtype=np.float64)
    val = np.array([r["local_val_loss"] for r in records], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(steps, np.maximum(loss, 1.0e-300), label="train")
    ax.semilogy(steps, np.maximum(val, 1.0e-300), label="local validation")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "offline_loss_latest.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    device = torch_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reconstruction_gamma_alpha < 0.0:
        raise ValueError("--reconstruction-gamma-alpha must be non-negative.")
    if args.weight_l2_lambda < 0.0:
        raise ValueError("--weight-l2-lambda must be non-negative.")
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    kind_probs_np = canonical_kind_probabilities(args.discontinuity_multiplier)
    kind_probs = torch.as_tensor(kind_probs_np, device=device, dtype=torch.float64)
    paper_total_samples = float(np.sum(NOGUEIRA_COUNTS))
    paper_train_samples = paper_total_samples * (1.0 - args.paper_validation_fraction)
    if args.validation_mode == "paper20":
        validation_counts = np.maximum(
            1,
            np.rint(NOGUEIRA_COUNTS * args.paper_validation_fraction * args.validation_sample_scale).astype(np.int64),
        )
        validation_batches = make_validation_batches_by_counts(
            validation_counts,
            args.dx,
            device,
            args.seed + 1000,
            args.gamma_filter,
            args.paper_grid_centers,
            args.cell_average_mode,
            args.mirror_augmentation,
        )
        validation_label = "paper20_scaled"
        validation_desc = " ".join(f"{kind}={int(n)}" for kind, n in zip(KIND_NAMES, validation_counts))
    else:
        val_n = args.validation_samples_per_kind if args.validation_samples_per_kind > 0 else min(args.samples_per_kind, 4096)
        validation_batches = make_validation_batches(
            val_n,
            args.dx,
            device,
            args.seed + 1000,
            args.gamma_filter,
            args.paper_grid_centers,
            args.cell_average_mode,
            args.mirror_augmentation,
        )
        validation_label = "uniform_per_kind"
        validation_desc = f"{val_n}/kind"
    save_gamma_histogram(args.out_dir / "gamma_histogram.csv", validation_batches)

    model = SharedBadnessMLP(args.seed).to(device)
    if args.init_model is not None:
        load_checkpoint_into_model(model, args.init_model, device)
        print(f"loaded_init_model: {args.init_model}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, object]] = []
    local_records: list[dict[str, object]] = []
    sod_records: list[dict[str, object]] = []
    best_sod_linf = float("inf")
    best_sod_l2 = float("inf")
    best_local_loss = float("inf")

    print(
        f"offline_pretrain_start: steps={args.steps} batch={args.batch_size} dx={args.dx} "
        f"lr={args.lr:.3e} smooth_anchor_lambda={args.smooth_anchor_lambda:.3e} "
        f"reconstruction_gamma_alpha={args.reconstruction_gamma_alpha:.3e} "
        f"weight_l2_lambda={args.weight_l2_lambda:.3e} "
        f"validation_mode={validation_label} validation={validation_desc} gamma_filter={args.gamma_filter} "
        f"paper_grid_centers={args.paper_grid_centers} cell_average_mode={args.cell_average_mode} "
        f"mirror_augmentation={args.mirror_augmentation} "
        f"paper_train_samples_80pct={paper_train_samples:.0f} "
        f"discontinuity_multiplier={args.discontinuity_multiplier:.3g} device={args.device}"
    )
    print(
        f"paper_scale_reference: architecture=5->10->6->6->3, "
        f"domain=[-1,1] except sine=[0,1], dx=0.01 in paper; "
        f"this_run_dx={args.dx}, equivalent_epoch = step*batch_size/{paper_train_samples:.0f}"
    )
    print("kind_probs:", " ".join(f"{k}={p:.4f}" for k, p in zip(KIND_NAMES, kind_probs_np)))

    wp_initialized = False
    for local_step in range(1, args.steps + 1):
        step = args.step_offset + local_step
        model.train()
        batch = make_batch(
            args.batch_size,
            args.dx,
            kind_probs,
            device,
            gen,
            args.gamma_filter,
            args.paper_grid_centers,
            args.cell_average_mode,
            args.mirror_augmentation,
        )
        loss, stats = loss_terms(model, batch, args.smooth_anchor_lambda, args.reconstruction_gamma_alpha, args.weight_l2_lambda)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().cpu())
        optimizer.step()

        if step == 1 or step % args.log_interval == 0:
            paper_epoch = step * args.batch_size / max(paper_train_samples, 1.0)
            print(
                f"step={step} loss={stats['loss']:.6e} recon={stats['reconstruction_loss']:.6e} "
                f"raw_recon={stats['raw_reconstruction_loss']:.6e} "
                f"anchor={stats['smooth_anchor_loss']:.6e} gamma={stats['gamma_mean']:.4f} "
                f"gamma_w={stats['reconstruction_weight_mean']:.4f} "
                f"r=[{stats['r0_mean']:.4f},{stats['r1_mean']:.4f},{stats['r2_mean']:.4f}] "
                f"paper_epoch={paper_epoch:.4f} grad={grad_norm:.3e}"
            )

        if args.checkpoint_interval > 0 and (step % args.checkpoint_interval == 0 or local_step == args.steps):
            save_checkpoint(args.out_dir / "checkpoints" / f"model_step_{step:06d}.npz", model, args, step, {"checkpoint_interval": args.checkpoint_interval})

        if (args.eval_step_one and local_step == 1) or (args.eval_interval > 0 and step % args.eval_interval == 0) or local_step == args.steps:
            local_overall, by_kind = evaluate_local(model, validation_batches, args.smooth_anchor_lambda, args.reconstruction_gamma_alpha, args.weight_l2_lambda)
            best_local_loss = min(best_local_loss, local_overall["loss"])
            for row in by_kind:
                row = {"step": step, **row}
                local_records.append(row)
            record = {
                "step": step,
                "paper_epoch_equiv": step * args.batch_size / max(paper_train_samples, 1.0),
                "train_loss": stats["loss"],
                "train_reconstruction_loss": stats["reconstruction_loss"],
                "train_raw_reconstruction_loss": stats["raw_reconstruction_loss"],
                "train_smooth_anchor_loss": stats["smooth_anchor_loss"],
                "train_weight_l2_loss": stats["weight_l2_loss"],
                "train_reconstruction_weight_mean": stats["reconstruction_weight_mean"],
                "local_val_loss": local_overall["loss"],
                "local_val_reconstruction_loss": local_overall["reconstruction_loss"],
                "local_val_raw_reconstruction_loss": local_overall["raw_reconstruction_loss"],
                "local_val_smooth_anchor_loss": local_overall["smooth_anchor_loss"],
                "local_val_weight_l2_loss": local_overall["weight_l2_loss"],
                "local_val_reconstruction_weight_mean": local_overall["reconstruction_weight_mean"],
                "grad_norm": grad_norm,
            }
            history.append(record)
            save_csv(
                args.out_dir / "offline_history.csv",
                history,
                [
                    "step",
                    "paper_epoch_equiv",
                    "train_loss",
                    "train_reconstruction_loss",
                    "train_raw_reconstruction_loss",
                    "train_smooth_anchor_loss",
                    "train_weight_l2_loss",
                    "train_reconstruction_weight_mean",
                    "local_val_loss",
                    "local_val_reconstruction_loss",
                    "local_val_raw_reconstruction_loss",
                    "local_val_smooth_anchor_loss",
                    "local_val_weight_l2_loss",
                    "local_val_reconstruction_weight_mean",
                    "grad_norm",
                ],
            )
            save_csv(
                args.out_dir / "local_validation_by_kind.csv",
                local_records,
                [
                    "step",
                    "kind",
                    "n",
                    "loss",
                    "reconstruction_loss",
                    "raw_reconstruction_loss",
                    "smooth_anchor_loss",
                    "weight_l2_loss",
                    "gamma_mean",
                    "reconstruction_weight_mean",
                    "r0_mean",
                    "r1_mean",
                    "r2_mean",
                ],
            )
            plot_offline_history(args.out_dir, history)
            save_checkpoint(args.out_dir / "model_latest.npz", model, args, step, {"local_val_loss": local_overall["loss"]})
            if local_overall["loss"] <= best_local_loss:
                save_checkpoint(args.out_dir / "model_best_local.npz", model, args, step, {"local_val_loss": local_overall["loss"]})

            if args.sod_eval and args.sod_nx > 0:
                if run_circle_sod_validation is None or write_circle_validation_outputs is None:
                    raise RuntimeError(
                        "Sod validation requires train_weno5_mlp.py in the current Python path. "
                        "Restore train_weno5_mlp.py or run with --no-sod-eval."
                    )
                if not wp_initialized:
                    wh.require_warp()
                    wh.wp.init()
                    wh.wp.set_device(args.sod_device if args.sod_device else args.device)
                    wp_initialized = True
                sod_device = args.sod_device if args.sod_device else args.device
                sod_record = run_circle_sod_validation(step, step, model_to_wp_params(model, sod_device), make_sod_args(args), None)
                sod_records.append(sod_record)
                write_circle_validation_outputs(args.out_dir, sod_records)
                m_linf = float(sod_record.get("mlp_vs_reference_linf", float("nan")))
                m_l2 = float(sod_record.get("mlp_vs_reference_l2", float("nan")))
                if np.isfinite(m_l2) and m_l2 < best_sod_l2:
                    best_sod_l2 = m_l2
                    save_checkpoint(args.out_dir / "model_best_sod_l2.npz", model, args, step, {"best_sod_l2": best_sod_l2})
                if np.isfinite(m_linf) and m_linf < best_sod_linf:
                    best_sod_linf = m_linf
                    save_checkpoint(args.out_dir / "model_best_sod_linf.npz", model, args, step, {"best_sod_linf": best_sod_linf})
                print(
                    f"sod_validation step={step} MLP_ref_L2={float(sod_record.get('mlp_vs_reference_l2', float('nan'))):.6e} "
                    f"MLP_ref_Linf={m_linf:.6e} classic_ref_Linf={float(sod_record.get('classical_vs_reference_linf', float('nan'))):.6e} "
                    f"best_sod_l2={best_sod_l2:.6e} best_sod_linf={best_sod_linf:.6e}"
                )

    final_step = args.step_offset + args.steps
    save_checkpoint(args.out_dir / "model_latest.npz", model, args, final_step, {"best_sod_l2": best_sod_l2, "best_sod_linf": best_sod_linf, "best_local_loss": best_local_loss})
    print(f"done: history={args.out_dir / 'offline_history.csv'} model={args.out_dir / 'model_latest.npz'} best_sod={args.out_dir / 'model_best_sod_linf.npz'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--samples-per-kind", type=int, default=4096, help="Requested canonical samples per kind; validation is capped unless --validation-samples-per-kind is set.")
    parser.add_argument("--validation-samples-per-kind", type=int, default=0, help="Fixed local validation samples per kind; defaults to min(samples_per_kind, 4096).")
    parser.add_argument("--validation-mode", choices=("uniform", "paper20"), default="uniform", help="uniform keeps equal-per-kind local validation; paper20 uses Nogueira's 20% split proportions.")
    parser.add_argument("--paper-validation-fraction", type=float, default=NOGUEIRA_VALIDATION_FRACTION, help="Validation fraction used for paper20 local validation and equivalent-epoch reporting.")
    parser.add_argument("--validation-sample-scale", type=float, default=1.0, help="Scale paper20 validation counts; 1.0 is the full paper 20% split, 0.25 is a faster quarter-sized check.")
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--paper-grid-centers", action=argparse.BooleanOptionalAction, default=False, help="Sample stencil centers on the fixed dx grid used by the paper instead of continuous random centers.")
    parser.add_argument("--cell-average-mode", choices=("exact", "gauss15"), default="exact", help="How to form FVM cell averages for canonical functions. exact uses analytic antiderivatives; gauss15 keeps the old quadrature path.")
    parser.add_argument("--mirror-augmentation", action=argparse.BooleanOptionalAction, default=False, help="Randomly mirror half of canonical stencils and swap LR targets 1<->2 and 3<->4 to enforce left/right symmetry. Disabled by default for paper-original data generation.")
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--smooth-anchor-lambda", type=float, default=1.0e-1)
    parser.add_argument("--weight-l2-lambda", type=float, default=1.0e-9, help="Paper-style beta_W L2 regularization on MLP weight matrices.")
    parser.add_argument(
        "--reconstruction-gamma-alpha",
        type=float,
        default=0.2,
        help="Paper-style reconstruction weighting exponent: reconstruction MSE is multiplied by gamma_s^alpha. Use 0.0 to recover the old unweighted reconstruction loss.",
    )
    parser.add_argument("--discontinuity-multiplier", type=float, default=1.0, help="Multiply the Nogueira discontinuity sampling probability before renormalizing.")
    parser.add_argument("--gamma-filter", action=argparse.BooleanOptionalAction, default=True, help="Apply Nogueira Table-2-style gamma_s filters to canonical samples.")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Save numbered offline checkpoints every N steps under out_dir/checkpoints; 0 disables.")
    parser.add_argument("--eval-step-one", action=argparse.BooleanOptionalAction, default=False, help="Run local/Sod validation at step 1. Default is off so paper-scale offline runs validate only every --eval-interval steps.")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sod-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sod-nx", type=int, default=100)
    parser.add_argument("--sod-ny", type=int, default=8)
    parser.add_argument("--sod-t-end", type=float, default=0.25)
    parser.add_argument("--sod-cfl", type=float, default=0.4)
    parser.add_argument("--sod-axis", choices=("x", "y"), default="x")
    parser.add_argument("--sod-eno-cutoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sod-device", choices=("cuda", "cpu"), default=None, help="Device for Sod validation; defaults to --device. Use cpu to avoid CUDA Warp validation JIT during offline GPU training.")
    parser.add_argument("--init-model", type=Path, default=None, help="Initialize offline pretraining from a compatible model_latest/model_best .npz checkpoint. Optimizer state starts fresh.")
    parser.add_argument("--step-offset", type=int, default=0, help="Add this offset to reported/saved step numbers when continuing a previous offline run.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("plots/weno5_offline_pretrain"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
