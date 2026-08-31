#!/usr/bin/env python3
"""Offline FVM WENO7-NN pretraining on Nogueira-style canonical stencils.

This is the WENO7 analogue of pretrain_weno5_offline.py.  It trains the
compact shared badness-ratio network

    6 -> 12 -> 6 -> 6 -> 4

using normalized substencil badness features, gamma_s, and a scale feature.
The checkpoint format keeps the same w1,b1,...,w4,b4,npz convention so the
model can later be wired into the Warp WENO7/ADER4 solver.
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

from compute_soomth_WENO import get_decomposition
from pretrain_weno5_offline import (
    GAMMA_FILTERS,
    KIND_NAMES,
    NOGUEIRA_COUNTS,
    NOGUEIRA_VALIDATION_FRACTION,
    ROOT3,
    canonical_cell_average,
    canonical_values,
    make_kind_params as make_base_kind_params,
    rand_uniform,
)


WENO_ORDER = 7
R = 4
FULL_STENCIL = 2 * R - 1
MLP_INPUTS = R + 2
MLP_HIDDEN1 = 12
MLP_HIDDEN2 = 8
MLP_HIDDEN3 = 8
MLP_OUTPUTS = R
BADNESS_RATIO_SCALE = float(R)
BADNESS_RATIO_POWER = 2.0
LR_VALUES = (1, 2, 3, 4)
POSITIVE_TARGET_LR_VALUES = (1, 4)
NEGATIVE_TARGET_LR_VALUES = (2, 3)


def lr_values_for_target_side(target_side: str) -> tuple[int, ...]:
    if target_side == "all":
        return LR_VALUES
    if target_side == "positive":
        return POSITIVE_TARGET_LR_VALUES
    if target_side == "negative":
        return NEGATIVE_TARGET_LR_VALUES
    raise ValueError(f"unknown training target side {target_side}")


def lr_indices(lr_values: tuple[int, ...]) -> list[int]:
    return [LR_VALUES.index(lr) for lr in lr_values]


@dataclass
class Batch:
    q: torch.Tensor
    targets: torch.Tensor
    kind_ids: torch.Tensor
    gamma_s: torch.Tensor


def _linear_form_coeffs(expr, symbols: tuple[object, ...]) -> np.ndarray:
    return np.array([float(expr.coeff(sym)) for sym in symbols], dtype=np.float64)


def _build_delta_mode_arrays() -> tuple[np.ndarray, np.ndarray]:
    data = get_decomposition(WENO_ORDER)
    q_symbols = tuple(data["q"])
    d_coeffs = np.zeros((R, R - 1, FULL_STENCIL), dtype=np.float64)
    c_coeffs = np.zeros((R, R - 1), dtype=np.float64)
    for k, modes in enumerate(data["items"]):
        for m, mode in enumerate(modes):
            d_coeffs[k, m, :] = _linear_form_coeffs(mode["D"], q_symbols)
            c_coeffs[k, m] = float(mode["coeff"])
    return d_coeffs, c_coeffs


DELTA_D_COEFFS_NP, DELTA_C_COEFFS_NP = _build_delta_mode_arrays()


def _weno7_candidate_coefficients() -> dict[int, np.ndarray]:
    s = ROOT3
    loc = {
        1: np.array([-1.0 / 4.0, 13.0 / 12.0, -23.0 / 12.0, 25.0 / 12.0], dtype=np.float64),
        2: np.array([1.0 / 12.0, -5.0 / 12.0, 13.0 / 12.0, 1.0 / 4.0], dtype=np.float64),
        3: np.array([-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0], dtype=np.float64),
        4: np.array([1.0 / 4.0, 13.0 / 12.0, -5.0 / 12.0, 1.0 / 12.0], dtype=np.float64),
        5: np.array([25.0 / 12.0, -23.0 / 12.0, 13.0 / 12.0, -1.0 / 4.0], dtype=np.float64),
    }
    gauss = {
        1: np.array([-11.0 * s / 216.0, 17.0 * s / 72.0, -35.0 * s / 72.0, (65.0 * s + 216.0) / 216.0], dtype=np.float64),
        2: np.array([7.0 * s / 216.0, -13.0 * s / 72.0, (7.0 * s + 72.0) / 72.0, 11.0 * s / 216.0], dtype=np.float64),
        3: np.array([-11.0 * s / 216.0, (72.0 - 7.0 * s) / 72.0, 13.0 * s / 72.0, -7.0 * s / 216.0], dtype=np.float64),
        4: np.array([(216.0 - 65.0 * s) / 216.0, 35.0 * s / 72.0, -17.0 * s / 72.0, 11.0 * s / 216.0], dtype=np.float64),
        5: np.array([11.0 * s / 216.0, -17.0 * s / 72.0, 35.0 * s / 72.0, (216.0 - 65.0 * s) / 216.0], dtype=np.float64),
        6: np.array([-7.0 * s / 216.0, 13.0 * s / 72.0, (72.0 - 7.0 * s) / 72.0, -11.0 * s / 216.0], dtype=np.float64),
        7: np.array([11.0 * s / 216.0, (7.0 * s + 72.0) / 72.0, -13.0 * s / 72.0, 7.0 * s / 216.0], dtype=np.float64),
        8: np.array([(65.0 * s + 216.0) / 216.0, -35.0 * s / 72.0, 17.0 * s / 72.0, -11.0 * s / 216.0], dtype=np.float64),
    }
    return {
        1: np.stack([loc[1], loc[2], loc[3], loc[4]], axis=0),
        2: np.stack([loc[2], loc[3], loc[4], loc[5]], axis=0),
        # LR=3 is x_c - sqrt(3)/6 dx, which is the helper's gauss lr=2.
        3: np.stack([gauss[5], gauss[6], gauss[7], gauss[8]], axis=0),
        # LR=4 is x_c + sqrt(3)/6 dx, which is the helper's gauss lr=1.
        4: np.stack([gauss[1], gauss[2], gauss[3], gauss[4]], axis=0),
    }


def _weno7_linear_weights() -> dict[int, np.ndarray]:
    s = ROOT3
    return {
        1: np.array([1.0 / 35.0, 12.0 / 35.0, 18.0 / 35.0, 4.0 / 35.0], dtype=np.float64),
        2: np.array([4.0 / 35.0, 18.0 / 35.0, 12.0 / 35.0, 1.0 / 35.0], dtype=np.float64),
        3: np.array(
            [
                59.0 / 880.0 + 5.0 * s / 16632.0,
                381.0 / 880.0 + 587.0 * s / 194040.0,
                381.0 / 880.0 - 587.0 * s / 194040.0,
                59.0 / 880.0 - 5.0 * s / 16632.0,
            ],
            dtype=np.float64,
        ),
        4: np.array(
            [
                59.0 / 880.0 - 5.0 * s / 16632.0,
                381.0 / 880.0 - 587.0 * s / 194040.0,
                381.0 / 880.0 + 587.0 * s / 194040.0,
                59.0 / 880.0 + 5.0 * s / 16632.0,
            ],
            dtype=np.float64,
        ),
    }


CANDIDATE_COEFFS_NP = _weno7_candidate_coefficients()
OPTIMAL_D_NP = _weno7_linear_weights()


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


def cell_centers_weno7(
    batch_size: int,
    dx: float,
    device: torch.device,
    generator: torch.Generator,
    oscillatory: bool = False,
    paper_grid_centers: bool = False,
) -> torch.Tensor:
    margin = 3.5 * dx
    if paper_grid_centers:
        low = margin if oscillatory else -1.0 + margin
        high = 1.0 - margin
        n_grid = max(1, int(np.floor((high - low) / dx + 1.0e-12)) + 1)
        idx = torch.randint(n_grid, (batch_size,), device=device, generator=generator)
        return low + idx.to(torch.float64) * dx
    if oscillatory:
        return rand_uniform((batch_size,), margin, 1.0 - margin, device, generator)
    return rand_uniform((batch_size,), -1.0 + margin, 1.0 - margin, device, generator)


def make_kind_params(
    kind: str,
    n: int,
    x_center: torch.Tensor,
    dx: float,
    device: torch.device,
    generator: torch.Generator,
    discontinuity_x0_mode: str,
    discontinuity_cutcell_prob: float,
    discontinuity_value_min: float,
    discontinuity_value_max: float,
) -> dict[str, torch.Tensor]:
    if kind != "discontinuity":
        return make_base_kind_params(kind, n, x_center, dx, device, generator)

    def col(v: torch.Tensor) -> torch.Tensor:
        return v.reshape(n, 1)

    ul = rand_uniform((n,), discontinuity_value_min, discontinuity_value_max, device, generator)
    ur = rand_uniform((n,), discontinuity_value_min, discontinuity_value_max, device, generator)
    too_close = torch.abs(ur - ul) < 1.0e-8
    ur = torch.where(too_close, ur + 1.0, ur)
    x0_fixed = torch.full((n,), 0.5, device=device, dtype=torch.float64)
    if discontinuity_x0_mode == "fixed":
        x0 = x0_fixed
    elif discontinuity_x0_mode in ("cutcell", "mixed"):
        # Put the jump strictly inside one of the seven FVM cells.  This creates
        # partial cell averages such as low-low-low-low-mixed-high-high, which
        # fixed edge-aligned jumps do not generate.
        cell_offset = torch.randint(-3, 4, (n,), device=device, generator=generator).to(torch.float64)
        in_cell_fraction = rand_uniform((n,), 0.1, 0.9, device, generator)
        x0_cutcell = x_center + (cell_offset + in_cell_fraction - 0.5) * dx
        if discontinuity_x0_mode == "cutcell":
            x0 = x0_cutcell
        else:
            use_cutcell = torch.rand((n,), device=device, generator=generator) < discontinuity_cutcell_prob
            x0 = torch.where(use_cutcell, x0_cutcell, x0_fixed)
    else:
        raise ValueError(f"unknown discontinuity_x0_mode {discontinuity_x0_mode}")
    return {"ul": col(ul), "ur": col(ur), "x0": col(x0)}


def make_kind_batch_unfiltered(
    kind: str,
    n: int,
    dx: float,
    device: torch.device,
    generator: torch.Generator,
    paper_grid_centers: bool,
    cell_average_mode: str,
    mirror_augmentation: bool,
    discontinuity_x0_mode: str,
    discontinuity_cutcell_prob: float,
    discontinuity_value_min: float,
    discontinuity_value_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_center = cell_centers_weno7(n, dx, device, generator, oscillatory=(kind == "sine"), paper_grid_centers=paper_grid_centers)
    params = make_kind_params(
        kind,
        n,
        x_center,
        dx,
        device,
        generator,
        discontinuity_x0_mode,
        discontinuity_cutcell_prob,
        discontinuity_value_min,
        discontinuity_value_max,
    )
    offsets = torch.arange(-3, 4, device=device, dtype=torch.float64)
    centers = x_center[:, None] + offsets[None, :] * dx
    if cell_average_mode != "exact":
        raise ValueError("WENO7 offline currently supports --cell-average-mode exact only.")
    q = canonical_cell_average(kind, centers - 0.5 * dx, centers + 0.5 * dx, params)
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
    return q, targets, weno7_gamma_s(q)


def make_kind_batch(
    kind: str,
    n: int,
    dx: float,
    device: torch.device,
    generator: torch.Generator,
    apply_gamma_filter: bool = True,
    paper_grid_centers: bool = False,
    cell_average_mode: str = "exact",
    mirror_augmentation: bool = False,
    discontinuity_x0_mode: str = "fixed",
    discontinuity_cutcell_prob: float = 0.2,
    discontinuity_value_min: float = -40.0,
    discontinuity_value_max: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (not apply_gamma_filter) or kind not in GAMMA_FILTERS:
        return make_kind_batch_unfiltered(
            kind,
            n,
            dx,
            device,
            generator,
            paper_grid_centers,
            cell_average_mode,
            mirror_augmentation,
            discontinuity_x0_mode,
            discontinuity_cutcell_prob,
            discontinuity_value_min,
            discontinuity_value_max,
        )
    lo, hi = GAMMA_FILTERS[kind]
    q_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    gamma_parts: list[torch.Tensor] = []
    collected = 0
    attempts = 0
    while collected < n and attempts < 64:
        request = max(2 * (n - collected), n)
        q, targets, gamma_s = make_kind_batch_unfiltered(
            kind,
            request,
            dx,
            device,
            generator,
            paper_grid_centers,
            cell_average_mode,
            mirror_augmentation,
            discontinuity_x0_mode,
            discontinuity_cutcell_prob,
            discontinuity_value_min,
            discontinuity_value_max,
        )
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
        q, targets, gamma_s = make_kind_batch_unfiltered(
            kind,
            n - collected,
            dx,
            device,
            generator,
            paper_grid_centers,
            cell_average_mode,
            mirror_augmentation,
            discontinuity_x0_mode,
            discontinuity_cutcell_prob,
            discontinuity_value_min,
            discontinuity_value_max,
        )
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
    discontinuity_x0_mode: str,
    discontinuity_cutcell_prob: float,
    discontinuity_value_min: float,
    discontinuity_value_max: float,
) -> Batch:
    kind_ids = torch.multinomial(kind_probs, batch_size, replacement=True, generator=generator)
    q = torch.empty((batch_size, FULL_STENCIL), device=device, dtype=torch.float64)
    targets = torch.empty((batch_size, len(LR_VALUES)), device=device, dtype=torch.float64)
    gamma_s = torch.empty((batch_size,), device=device, dtype=torch.float64)
    for kind_id, kind in enumerate(KIND_NAMES):
        mask = kind_ids == kind_id
        n = int(mask.sum().item())
        if n == 0:
            continue
        q_k, targets_k, gamma_k = make_kind_batch(
            kind,
            n,
            dx,
            device,
            generator,
            apply_gamma_filter,
            paper_grid_centers,
            cell_average_mode,
            mirror_augmentation,
            discontinuity_x0_mode,
            discontinuity_cutcell_prob,
            discontinuity_value_min,
            discontinuity_value_max,
        )
        q[mask] = q_k
        targets[mask] = targets_k
        gamma_s[mask] = gamma_k
    return Batch(q=q, targets=targets, kind_ids=kind_ids, gamma_s=gamma_s)


def make_validation_batches(
    samples_per_kind: int,
    dx: float,
    device: torch.device,
    seed: int,
    apply_gamma_filter: bool,
    paper_grid_centers: bool,
    cell_average_mode: str,
    mirror_augmentation: bool,
    discontinuity_x0_mode: str,
    discontinuity_cutcell_prob: float,
    discontinuity_value_min: float,
    discontinuity_value_max: float,
) -> list[Batch]:
    batches: list[Batch] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for kind_id, kind in enumerate(KIND_NAMES):
        q, targets, gamma_s = make_kind_batch(
            kind,
            samples_per_kind,
            dx,
            device,
            gen,
            apply_gamma_filter,
            paper_grid_centers,
            cell_average_mode,
            mirror_augmentation,
            discontinuity_x0_mode,
            discontinuity_cutcell_prob,
            discontinuity_value_min,
            discontinuity_value_max,
        )
        kind_ids = torch.full((samples_per_kind,), kind_id, device=device, dtype=torch.long)
        batches.append(Batch(q=q, targets=targets, kind_ids=kind_ids, gamma_s=gamma_s))
    return batches


def make_validation_batches_by_counts(
    counts: np.ndarray,
    dx: float,
    device: torch.device,
    seed: int,
    apply_gamma_filter: bool,
    paper_grid_centers: bool,
    cell_average_mode: str,
    mirror_augmentation: bool,
    discontinuity_x0_mode: str,
    discontinuity_cutcell_prob: float,
    discontinuity_value_min: float,
    discontinuity_value_max: float,
) -> list[Batch]:
    batches: list[Batch] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for kind_id, kind in enumerate(KIND_NAMES):
        n = max(1, int(counts[kind_id]))
        q, targets, gamma_s = make_kind_batch(
            kind,
            n,
            dx,
            device,
            gen,
            apply_gamma_filter,
            paper_grid_centers,
            cell_average_mode,
            mirror_augmentation,
            discontinuity_x0_mode,
            discontinuity_cutcell_prob,
            discontinuity_value_min,
            discontinuity_value_max,
        )
        kind_ids = torch.full((n,), kind_id, device=device, dtype=torch.long)
        batches.append(Batch(q=q, targets=targets, kind_ids=kind_ids, gamma_s=gamma_s))
    return batches


def canonical_kind_probabilities(discontinuity_multiplier: float) -> np.ndarray:
    counts = NOGUEIRA_COUNTS.copy()
    counts[KIND_NAMES.index("discontinuity")] *= discontinuity_multiplier
    return counts / counts.sum()


def weno7_gamma_s(q: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-15
    d2 = q[:, :-2] - 2.0 * q[:, 1:-1] + q[:, 2:]
    denom = torch.abs(q[:, 1:-1] - q[:, :-2]) + torch.abs(q[:, 2:] - q[:, 1:-1]) + eps
    gamma = torch.max(torch.abs(d2) / denom, dim=1).values
    return torch.clamp(gamma, 0.0, 1.0)


def weno7_features(q: torch.Tensor) -> torch.Tensor:
    d_coeffs = torch.as_tensor(DELTA_D_COEFFS_NP, device=q.device, dtype=torch.float64)
    c_coeffs = torch.as_tensor(DELTA_C_COEFFS_NP, device=q.device, dtype=torch.float64)
    forms = torch.einsum("bn,kmn->bkm", q, d_coeffs)
    delta = torch.sum(c_coeffs.reshape(1, R, R - 1) * torch.abs(forms), dim=2)
    delta_max = torch.max(delta, dim=1).values
    norm_delta = delta / torch.clamp(delta_max.reshape(-1, 1), min=1.0e-15)
    gamma_s = weno7_gamma_s(q)
    q_scale = torch.clamp(torch.max(torch.abs(q), dim=1).values, min=1.0)
    relative_scale = torch.clamp(delta_max / q_scale, min=1.0e-30)
    scale_feature = torch.clamp((torch.log10(relative_scale) + 16.0) / 16.0, 0.0, 1.0)
    return torch.cat((norm_delta, gamma_s.reshape(-1, 1), scale_feature.reshape(-1, 1)), dim=1)


def candidate_values(q: torch.Tensor, lr: int) -> torch.Tensor:
    coeffs = torch.as_tensor(CANDIDATE_COEFFS_NP[lr], device=q.device, dtype=torch.float64)
    vals = []
    for k in range(R):
        vals.append(torch.sum(q[:, k : k + R] * coeffs[k].reshape(1, R), dim=1))
    return torch.stack(vals, dim=1)


def optimal_d(lr: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(OPTIMAL_D_NP[lr], device=device, dtype=torch.float64)


def omega_from_ratio(r: torch.Tensor, lr: int) -> torch.Tensor:
    d = optimal_d(lr, r.device).reshape(1, R)
    beta = BADNESS_RATIO_SCALE * r
    alpha = d / torch.pow(beta + 1.0e-12, BADNESS_RATIO_POWER)
    return alpha / torch.sum(alpha, dim=1, keepdim=True)


def omega_from_ratio_with_cutoff(r: torch.Tensor, lr: int, eno_cutoff: bool) -> torch.Tensor:
    omega = omega_from_ratio(r, lr)
    if not eno_cutoff:
        return omega
    mask = omega >= 4.0e-7
    clipped = torch.where(mask, omega, torch.zeros_like(omega))
    denom = torch.sum(clipped, dim=1, keepdim=True)
    return torch.where(denom > 0.0, clipped / denom, omega)


def ratio_for_lr(
    model: SharedBadnessMLP,
    q: torch.Tensor,
    r: torch.Tensor,
    lr: int,
    mirror_positive_targets: bool,
) -> torch.Tensor:
    if mirror_positive_targets and lr in POSITIVE_TARGET_LR_VALUES:
        flipped = torch.flip(q, dims=(1,))
        return torch.flip(model(weno7_features(flipped)), dims=(1,))
    return r


def nonlocal_omega_penalty(
    model: SharedBadnessMLP,
    q: torch.Tensor,
    r: torch.Tensor,
    gamma_s: torch.Tensor,
    tau: float,
    gamma_power: float,
    mirror_positive_targets: bool,
    lr_values: tuple[int, ...],
) -> torch.Tensor:
    gamma = torch.clamp(gamma_s, min=0.0)
    if gamma_power == 0.0:
        gamma_weight = torch.ones_like(gamma)
    else:
        gamma_weight = torch.pow(gamma, gamma_power)
    penalty_terms = []
    for lr in lr_values:
        r_lr = ratio_for_lr(model, q, r, lr, mirror_positive_targets)
        omega = omega_from_ratio(r_lr, lr)
        excess = torch.relu(omega - tau)
        penalty = (
            excess[:, 0] * excess[:, 2]
            + excess[:, 0] * excess[:, 3]
            + excess[:, 1] * excess[:, 3]
        )
        penalty_terms.append(penalty)
    stacked = torch.stack(penalty_terms, dim=1)
    return torch.mean(gamma_weight.reshape(-1, 1) * stacked)


def reconstruction_predictions(
    model: SharedBadnessMLP,
    q: torch.Tensor,
    r: torch.Tensor,
    mirror_positive_targets: bool,
    lr_values: tuple[int, ...],
) -> torch.Tensor:
    preds = []
    for lr in lr_values:
        r_lr = ratio_for_lr(model, q, r, lr, mirror_positive_targets)
        omega = omega_from_ratio(r_lr, lr)
        stencil = candidate_values(q, lr)
        preds.append(torch.sum(omega * stencil, dim=1))
    return torch.stack(preds, dim=1)


def reconstruction_prediction_lr(q: torch.Tensor, r: torch.Tensor, lr: int, eno_cutoff: bool = False) -> torch.Tensor:
    omega = omega_from_ratio_with_cutoff(r, lr, eno_cutoff)
    stencil = candidate_values(q, lr)
    return torch.sum(omega * stencil, dim=1)


def weight_l2(model: SharedBadnessMLP) -> torch.Tensor:
    return torch.sum(torch.square(model.w1)) + torch.sum(torch.square(model.w2)) + torch.sum(torch.square(model.w3)) + torch.sum(torch.square(model.w4))


def reconstruction_error_scale(q: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return torch.ones((q.shape[0],), device=q.device, dtype=torch.float64)
    if mode == "stencil-range":
        q_range = torch.max(q, dim=1).values - torch.min(q, dim=1).values
        return torch.clamp(q_range, min=1.0)
    raise ValueError(f"unknown reconstruction error normalization mode {mode}")


def loss_terms(
    model: SharedBadnessMLP,
    batch: Batch,
    smooth_anchor_lambda: float,
    reconstruction_gamma_alpha: float,
    weight_l2_lambda: float,
    reconstruction_error_normalization: str,
    nonlocal_omega_lambda: float,
    nonlocal_omega_tau: float,
    nonlocal_omega_gamma_power: float,
    mirror_positive_targets: bool,
    training_target_side: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    active_lrs = lr_values_for_target_side(training_target_side)
    target_indices = lr_indices(active_lrs)
    features = weno7_features(batch.q)
    r = model(features)
    preds = reconstruction_predictions(model, batch.q, r, mirror_positive_targets, active_lrs)
    err = preds - batch.targets[:, target_indices]
    raw_sq = torch.square(err)
    err_scale = reconstruction_error_scale(batch.q, reconstruction_error_normalization)
    sq = torch.square(err / err_scale.reshape(-1, 1))
    if reconstruction_gamma_alpha == 0.0:
        recon_weight = torch.ones_like(features[:, R])
    else:
        recon_weight = torch.pow(torch.clamp(features[:, R], min=0.0), reconstruction_gamma_alpha)
    raw_recon = torch.mean(raw_sq)
    recon = torch.mean(recon_weight.reshape(-1, 1) * sq)
    smooth = torch.square(1.0 - features[:, R]).reshape(-1, 1)
    r_target = torch.full_like(r, 1.0 / R)
    anchor = torch.mean(smooth * torch.square(r - r_target))
    nonlocal_penalty = nonlocal_omega_penalty(
        model,
        batch.q,
        r,
        features[:, R],
        nonlocal_omega_tau,
        nonlocal_omega_gamma_power,
        mirror_positive_targets,
        active_lrs,
    )
    l2 = weight_l2(model)
    loss = recon + smooth_anchor_lambda * anchor + nonlocal_omega_lambda * nonlocal_penalty + weight_l2_lambda * l2
    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(recon.detach().cpu()),
            "raw_reconstruction_loss": float(raw_recon.detach().cpu()),
            "smooth_anchor_loss": float(anchor.detach().cpu()),
            "nonlocal_omega_loss": float(nonlocal_penalty.detach().cpu()),
            "nonlocal_omega_weighted": float((nonlocal_omega_lambda * nonlocal_penalty).detach().cpu()),
            "weight_l2_loss": float(l2.detach().cpu()),
            "gamma_mean": float(features[:, R].mean().detach().cpu()),
            "reconstruction_weight_mean": float(recon_weight.mean().detach().cpu()),
            "reconstruction_error_scale_mean": float(err_scale.mean().detach().cpu()),
            "training_target_count": float(len(active_lrs)),
        }
        for k in range(R):
            stats[f"r{k}_mean"] = float(r[:, k].mean().detach().cpu())
    return loss, stats


def evaluate_local(
    model: SharedBadnessMLP,
    batches: list[Batch],
    smooth_anchor_lambda: float,
    reconstruction_gamma_alpha: float,
    weight_l2_lambda: float,
    reconstruction_error_normalization: str,
    nonlocal_omega_lambda: float,
    nonlocal_omega_tau: float,
    nonlocal_omega_gamma_power: float,
    mirror_positive_targets: bool,
    training_target_side: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    total_weight = 0
    sums: dict[str, float] = {}
    by_kind: list[dict[str, object]] = []
    with torch.no_grad():
        for kind_id, batch in enumerate(batches):
            _, stats = loss_terms(
                model,
                batch,
                smooth_anchor_lambda,
                reconstruction_gamma_alpha,
                weight_l2_lambda,
                reconstruction_error_normalization,
                nonlocal_omega_lambda,
                nonlocal_omega_tau,
                nonlocal_omega_gamma_power,
                mirror_positive_targets,
                training_target_side,
            )
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
    total = max(int(hist.sum()), 1)
    rows = [
        {
            "gamma_min": edges[i],
            "gamma_max": edges[i + 1],
            "count": int(count),
            "fraction": float(count / total),
        }
        for i, count in enumerate(hist)
    ]
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
            "expected WENO7 shared_direct_beta_ratio_6_12_8_8_4 checkpoints."
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
        "mlp_architecture": "shared_direct_beta_ratio_6_12_8_8_4_power2_normdelta_scale",
        "mlp_features": "[delta0/max_delta, delta1/max_delta, delta2/max_delta, delta3/max_delta, gamma_s, clipped((log10(max_delta/q_scale)+16)/16)]",
        "mlp_weight_formula": "offline FVM canonical pretrain; 6->12->8->8->4 shared r=softmax(6*tanh(raw/6)); beta=4*r; per-LR omega=normalize(d_lr/(beta+1e-12)^2)",
        "offline_pretrain": True,
        "weno_order": WENO_ORDER,
        "r": R,
        "dx": args.dx,
        "paper_grid_centers": args.paper_grid_centers,
        "cell_average_mode": args.cell_average_mode,
        "mirror_augmentation": args.mirror_augmentation,
        "validation_mode": args.validation_mode,
        "paper_validation_fraction": args.paper_validation_fraction,
        "validation_sample_scale": args.validation_sample_scale,
        "smooth_anchor_lambda": args.smooth_anchor_lambda,
        "reconstruction_gamma_alpha": args.reconstruction_gamma_alpha,
        "reconstruction_error_normalization": args.reconstruction_error_normalization,
        "nonlocal_omega_lambda": args.nonlocal_omega_lambda,
        "nonlocal_omega_tau": args.nonlocal_omega_tau,
        "nonlocal_omega_gamma_power": args.nonlocal_omega_gamma_power,
        "mirror_positive_targets": args.mirror_positive_targets,
        "training_target_side": args.training_target_side,
        "weight_l2_lambda": args.weight_l2_lambda,
        "discontinuity_multiplier": args.discontinuity_multiplier,
        "discontinuity_x0_mode": args.discontinuity_x0_mode,
        "discontinuity_cutcell_prob": args.discontinuity_cutcell_prob,
        "discontinuity_value_min": args.discontinuity_value_min,
        "discontinuity_value_max": args.discontinuity_value_max,
        "canonical_kinds": KIND_NAMES,
        "nogueira_counts": NOGUEIRA_COUNTS.tolist(),
        "nogueira_total_samples": int(np.sum(NOGUEIRA_COUNTS)),
        "nogueira_train_samples": int(round(float(np.sum(NOGUEIRA_COUNTS)) * (1.0 - args.paper_validation_fraction))),
        "init_model": str(args.init_model) if args.init_model else "",
        "step_offset": args.step_offset,
        "args": args_meta,
    }
    if extra:
        meta.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **checkpoint_payload(model, meta))


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


def primitive_to_conserved_1d(rho: np.ndarray, u: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    out = np.empty((rho.size, 3), dtype=np.float64)
    out[:, 0] = rho
    out[:, 1] = rho * u
    out[:, 2] = p / (gamma - 1.0) + 0.5 * rho * u * u
    return out


def conserved_to_primitive_1d(u: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = u[:, 0]
    vel = u[:, 1] / rho
    p = (gamma - 1.0) * (u[:, 2] - 0.5 * rho * vel * vel)
    return rho, vel, p


def euler_flux_1d(u: np.ndarray, gamma: float) -> np.ndarray:
    rho, vel, p = conserved_to_primitive_1d(u, gamma)
    flux = np.empty_like(u)
    flux[:, 0] = rho * vel
    flux[:, 1] = rho * vel * vel + p
    flux[:, 2] = vel * (u[:, 2] + p)
    return flux


def exact_sod_primitive(x: np.ndarray, t: float, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from train_weno5_mlp import sample_riemann_primitive

    left = (1.0, 0.0, 1.0)
    right = (0.125, 0.0, 0.1)
    rho = np.empty_like(x)
    vel = np.empty_like(x)
    p = np.empty_like(x)
    if t <= 0.0:
        mask = x < 0.0
        rho[mask], vel[mask], p[mask] = left
        rho[~mask], vel[~mask], p[~mask] = right
        return rho, vel, p
    for idx, coord in np.ndenumerate(x):
        rho[idx], vel[idx], p[idx] = sample_riemann_primitive(float(coord / t), left, right, gamma)
    return rho, vel, p


def exact_sod_cell_average_1d(xc: np.ndarray, dx: float, t: float, gamma: float) -> np.ndarray:
    xi, wi = np.polynomial.legendre.leggauss(15)
    state = np.zeros((xc.size, 3), dtype=np.float64)
    for node, weight in zip(xi.astype(np.float64), wi.astype(np.float64)):
        xq = xc + 0.5 * dx * float(node)
        rho, vel, p = exact_sod_primitive(xq, t, gamma)
        state += 0.5 * float(weight) * primitive_to_conserved_1d(rho, vel, p, gamma)
    return state


def apply_transmissive_boundary_1d(u: np.ndarray, nx: int, gc: int) -> None:
    u[:gc, :] = u[gc : gc + 1, :]
    u[gc + nx :, :] = u[gc + nx - 1 : gc + nx, :]


def max_wave_speed_1d(u: np.ndarray, gamma: float, nx: int, gc: int) -> float:
    interior = u[gc : gc + nx, :]
    rho, vel, p = conserved_to_primitive_1d(interior, gamma)
    if np.any(rho <= 0.0) or np.any(p <= 0.0):
        return float("nan")
    return float(np.max(np.abs(vel) + np.sqrt(gamma * p / rho)))


def reconstruct_weno7_primitive_1d(
    u: np.ndarray,
    nx: int,
    gc: int,
    gamma: float,
    model: SharedBadnessMLP | None,
    eno_cutoff: bool,
) -> tuple[np.ndarray, np.ndarray]:
    apply_transmissive_boundary_1d(u, nx, gc)
    rho, vel, p = conserved_to_primitive_1d(u, gamma)
    prim = np.stack((rho, vel, p), axis=1)
    left = np.empty((nx + 1, 3), dtype=np.float64)
    right = np.empty((nx + 1, 3), dtype=np.float64)
    device = next(model.parameters()).device if model is not None else torch.device("cpu")

    with torch.no_grad():
        for comp in range(3):
            q_left = np.empty((nx + 1, FULL_STENCIL), dtype=np.float64)
            q_right = np.empty((nx + 1, FULL_STENCIL), dtype=np.float64)
            for face in range(nx + 1):
                left_center = gc + face - 1
                right_center = gc + face
                q_left[face, :] = prim[left_center - 3 : left_center + 4, comp]
                q_right[face, :] = prim[right_center - 3 : right_center + 4, comp]
            ql = torch.as_tensor(q_left, device=device, dtype=torch.float64)
            qr = torch.as_tensor(q_right, device=device, dtype=torch.float64)
            if model is None:
                wl = optimal_d(1, ql.device).reshape(1, R)
                wr = optimal_d(2, qr.device).reshape(1, R)
                left[:, comp] = torch.sum(wl * candidate_values(ql, 1), dim=1).cpu().numpy()
                right[:, comp] = torch.sum(wr * candidate_values(qr, 2), dim=1).cpu().numpy()
            else:
                rl = model(weno7_features(ql))
                rr = model(weno7_features(qr))
                left[:, comp] = reconstruction_prediction_lr(ql, rl, 1, eno_cutoff).cpu().numpy()
                right[:, comp] = reconstruction_prediction_lr(qr, rr, 2, eno_cutoff).cpu().numpy()
    left[:, 0] = np.maximum(left[:, 0], 1.0e-12)
    right[:, 0] = np.maximum(right[:, 0], 1.0e-12)
    left[:, 2] = np.maximum(left[:, 2], 1.0e-12)
    right[:, 2] = np.maximum(right[:, 2], 1.0e-12)
    return left, right


def rusanov_flux_from_primitive(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    ul = primitive_to_conserved_1d(left[:, 0], left[:, 1], left[:, 2], gamma)
    ur = primitive_to_conserved_1d(right[:, 0], right[:, 1], right[:, 2], gamma)
    fl = euler_flux_1d(ul, gamma)
    fr = euler_flux_1d(ur, gamma)
    al = np.sqrt(gamma * left[:, 2] / left[:, 0])
    ar = np.sqrt(gamma * right[:, 2] / right[:, 0])
    speed = np.maximum(np.abs(left[:, 1]) + al, np.abs(right[:, 1]) + ar)
    return 0.5 * (fl + fr) - 0.5 * speed[:, None] * (ur - ul)


def rhs_weno7_sod_1d(
    u: np.ndarray,
    nx: int,
    gc: int,
    dx: float,
    gamma: float,
    model: SharedBadnessMLP | None,
    eno_cutoff: bool,
) -> tuple[np.ndarray, bool]:
    work = u.copy()
    left, right = reconstruct_weno7_primitive_1d(work, nx, gc, gamma, model, eno_cutoff)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return np.zeros_like(u), True
    flux = rusanov_flux_from_primitive(left, right, gamma)
    dudt = np.zeros_like(u)
    dudt[gc : gc + nx, :] = -(flux[1:, :] - flux[:-1, :]) / dx
    return dudt, False


def advance_weno7_sod_1d(
    u0: np.ndarray,
    nx: int,
    dx: float,
    cfl: float,
    t_end: float,
    gamma: float,
    model: SharedBadnessMLP | None,
    eno_cutoff: bool,
) -> tuple[np.ndarray, float, int, list[float], bool]:
    gc = R
    u = u0.copy()
    t = 0.0
    steps = 0
    dt_values: list[float] = []
    failed = False
    while t < t_end - 1.0e-14:
        apply_transmissive_boundary_1d(u, nx, gc)
        speed = max_wave_speed_1d(u, gamma, nx, gc)
        if not np.isfinite(speed) or speed <= 0.0:
            failed = True
            break
        dt = min(cfl * dx / speed, t_end - t)

        k1, bad = rhs_weno7_sod_1d(u, nx, gc, dx, gamma, model, eno_cutoff)
        if bad:
            failed = True
            break
        u1 = u.copy()
        u1[gc : gc + nx, :] = u[gc : gc + nx, :] + dt * k1[gc : gc + nx, :]

        k2, bad = rhs_weno7_sod_1d(u1, nx, gc, dx, gamma, model, eno_cutoff)
        if bad:
            failed = True
            break
        u2 = u.copy()
        u2[gc : gc + nx, :] = 0.75 * u[gc : gc + nx, :] + 0.25 * (u1[gc : gc + nx, :] + dt * k2[gc : gc + nx, :])

        k3, bad = rhs_weno7_sod_1d(u2, nx, gc, dx, gamma, model, eno_cutoff)
        if bad:
            failed = True
            break
        unew = u.copy()
        unew[gc : gc + nx, :] = (1.0 / 3.0) * u[gc : gc + nx, :] + (2.0 / 3.0) * (u2[gc : gc + nx, :] + dt * k3[gc : gc + nx, :])
        apply_transmissive_boundary_1d(unew, nx, gc)

        rho, _vel, p = conserved_to_primitive_1d(unew[gc : gc + nx, :], gamma)
        if (not np.all(np.isfinite(unew))) or np.any(rho <= 0.0) or np.any(p <= 0.0):
            failed = True
            u = unew
            break
        u = unew
        t += dt
        steps += 1
        dt_values.append(float(dt))
    return u, t, steps, dt_values, failed


def density_error_1d(pred: np.ndarray, reference: np.ndarray, nx: int, gc: int) -> tuple[float, float, float]:
    diff = pred[gc : gc + nx, 0] - reference[gc : gc + nx, 0]
    ad = np.abs(diff)
    return float(np.mean(ad)), float(np.sqrt(np.mean(diff * diff))), float(np.max(ad))


def run_weno7_sod_validation(step: int, model: SharedBadnessMLP, args: argparse.Namespace) -> dict[str, object]:
    import weno7_ader4_warp as weno7_formal

    gamma = 1.4
    nx = args.sod_nx
    gc = R
    x_length = 2.0
    dx = x_length / float(nx)
    ny = args.sod_ny
    y_length = max(float(ny) * dx, dx)
    centers = -1.0 + (np.arange(nx + 2 * gc, dtype=np.float64) - gc + 0.5) * dx
    reference = np.zeros((nx + 2 * gc, 3), dtype=np.float64)
    reference[:, :] = exact_sod_cell_average_1d(centers, dx, args.sod_t_end, gamma)
    apply_transmissive_boundary_1d(reference, nx, gc)

    model_was_training = model.training
    model.eval()
    eval_model_path = args.out_dir / "sod_validation_models" / f"model_step_{step:06d}.npz"
    save_checkpoint(eval_model_path, model, args, step, {"sod_validation_eval_model": True})
    if model_was_training:
        model.train()

    device = args.sod_device or args.device
    params = weno7_formal.wh.Params(
        nx=nx,
        ny=ny,
        x_length=x_length,
        y_length=y_length,
        cfl=args.sod_cfl,
        t_end=args.sod_t_end,
    )

    classical_host, classical_summary = weno7_formal.run_sod_case(
        params,
        device,
        axis=args.sod_axis,
        init_quadrature=args.sod_init_quadrature,
        model_path=None,
        eno_cutoff=False,
        riemann_solver="evilin",
        mlp_derivative_mode="all",
        max_steps=1_000_000,
        report_interval=0,
        label=f"sod_classical_step_{step:06d}",
    )
    mlp_host, mlp_summary = weno7_formal.run_sod_case(
        params,
        device,
        axis=args.sod_axis,
        init_quadrature=args.sod_init_quadrature,
        model_path=eval_model_path,
        eno_cutoff=args.sod_eno_cutoff,
        riemann_solver="evilin",
        mlp_derivative_mode=args.sod_mlp_derivative_mode,
        max_steps=1_000_000,
        report_interval=0,
        label=f"sod_mlp_step_{step:06d}",
    )
    classical_t = float(classical_summary["t"])
    classical_steps = int(classical_summary["steps"])
    classical_failed = bool(classical_summary["failed"])
    mlp_t = float(mlp_summary["t"])
    mlp_steps = int(mlp_summary["steps"])
    mlp_failed = bool(mlp_summary["failed"])

    def host_to_1d(host: np.ndarray) -> np.ndarray:
        avg = host[gc : gc + ny, :, :].mean(axis=0)
        out = np.zeros((nx + 2 * gc, 3), dtype=np.float64)
        out[:, 0] = avg[:, 0]
        out[:, 1] = avg[:, 1]
        out[:, 2] = avg[:, 3]
        return out

    classical = host_to_1d(classical_host)
    mlp = host_to_1d(mlp_host)

    c_l1, c_l2, c_linf = density_error_1d(classical, reference, nx, gc)
    if mlp_failed:
        m_l1 = m_l2 = m_linf = float("nan")
    else:
        m_l1, m_l2, m_linf = density_error_1d(mlp, reference, nx, gc)
    rho_m, vel_m, p_m = conserved_to_primitive_1d(mlp[gc : gc + nx, :], gamma)
    rho_c, vel_c, p_c = conserved_to_primitive_1d(classical[gc : gc + nx, :], gamma)
    record = {
        "step": step,
        "sod_nx": nx,
        "t_end": args.sod_t_end,
        "cfl": args.sod_cfl,
        "failed": float(classical_failed or mlp_failed),
        "classical_t": classical_t,
        "classical_steps": classical_steps,
        "mlp_t": mlp_t,
        "mlp_steps": mlp_steps,
        "dt_min": float(mlp_summary["dt_min"]),
        "dt_max": float(mlp_summary["dt_max"]),
        "classical_vs_reference_l1": c_l1,
        "classical_vs_reference_l2": c_l2,
        "classical_vs_reference_linf": c_linf,
        "mlp_vs_reference_l1": m_l1,
        "mlp_vs_reference_l2": m_l2,
        "mlp_vs_reference_linf": m_linf,
        "gain_vs_reference_l1": c_l1 - m_l1 if np.isfinite(m_l1) else float("nan"),
        "gain_vs_reference_l2": c_l2 - m_l2 if np.isfinite(m_l2) else float("nan"),
        "gain_vs_reference_linf": c_linf - m_linf if np.isfinite(m_linf) else float("nan"),
        "classical_rho_min": float(np.min(rho_c)),
        "classical_rho_max": float(np.max(rho_c)),
        "classical_p_min": float(np.min(p_c)),
        "mlp_rho_min": float(np.min(rho_m)),
        "mlp_rho_max": float(np.max(rho_m)),
        "mlp_p_min": float(np.min(p_m)),
    }
    write_weno7_sod_validation_plot(args.out_dir, step, centers[gc : gc + nx], classical, mlp, reference, nx, gc, record)
    return record


WENO7_SOD_FIELDS = [
    "step",
    "sod_nx",
    "t_end",
    "cfl",
    "failed",
    "classical_t",
    "classical_steps",
    "mlp_t",
    "mlp_steps",
    "dt_min",
    "dt_max",
    "classical_vs_reference_l1",
    "classical_vs_reference_l2",
    "classical_vs_reference_linf",
    "mlp_vs_reference_l1",
    "mlp_vs_reference_l2",
    "mlp_vs_reference_linf",
    "gain_vs_reference_l1",
    "gain_vs_reference_l2",
    "gain_vs_reference_linf",
    "classical_rho_min",
    "classical_rho_max",
    "classical_p_min",
    "mlp_rho_min",
    "mlp_rho_max",
    "mlp_p_min",
]


def write_weno7_sod_outputs(out_dir: Path, records: list[dict[str, object]]) -> None:
    save_csv(out_dir / "sod_validation_metrics.csv", records, WENO7_SOD_FIELDS)
    if not records:
        return
    steps = np.array([r["step"] for r in records], dtype=np.float64)
    c_l2 = np.array([r["classical_vs_reference_l2"] for r in records], dtype=np.float64)
    m_l2 = np.array([r["mlp_vs_reference_l2"] for r in records], dtype=np.float64)
    c_linf = np.array([r["classical_vs_reference_linf"] for r in records], dtype=np.float64)
    m_linf = np.array([r["mlp_vs_reference_linf"] for r in records], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    axes[0].plot(steps, c_l2, "o-", ms=3.0, label="classical")
    axes[0].plot(steps, m_l2, "o-", ms=3.0, label="mlp")
    axes[0].set_ylabel("density L2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(steps, c_linf, "o-", ms=3.0, label="classical")
    axes[1].plot(steps, m_linf, "o-", ms=3.0, label="mlp")
    axes[1].set_xlabel("offline step")
    axes[1].set_ylabel("density Linf")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sod_validation_trends.png", dpi=180)
    plt.close(fig)


def write_weno7_sod_validation_plot(
    out_dir: Path,
    step: int,
    x: np.ndarray,
    classical: np.ndarray,
    mlp: np.ndarray,
    reference: np.ndarray,
    nx: int,
    gc: int,
    record: dict[str, object],
) -> None:
    step_dir = out_dir / "sod_validation" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    rho_ref = reference[gc : gc + nx, 0]
    rho_c = classical[gc : gc + nx, 0]
    rho_m = mlp[gc : gc + nx, 0]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(x, rho_ref, "k-", lw=2.0, label="exact")
    ax.plot(x, rho_c, "--", lw=1.5, label="classical WENO7")
    ax.plot(x, rho_m, "-", lw=1.4, label="MLP WENO7")
    ax.set_xlabel("x")
    ax.set_ylabel("rho")
    ax.set_title(
        f"WENO7 1D Sod validation step {step}: "
        f"L2 classic={record['classical_vs_reference_l2']:.3e}, MLP={record['mlp_vs_reference_l2']:.3e}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(step_dir / "density_profile.png", dpi=180)
    plt.close(fig)
    np.savez(
        step_dir / "sod_profile.npz",
        x=x,
        classical=classical[gc : gc + nx, :],
        mlp=mlp[gc : gc + nx, :],
        reference=reference[gc : gc + nx, :],
        record=np.array(record, dtype=object),
    )


WENO7_EXPLOSION_FIELDS = [
    "step",
    "explosion_nx",
    "explosion_ny",
    "t_end",
    "cfl",
    "failed",
    "mlp_t",
    "mlp_steps",
    "dt_min",
    "dt_max",
    "dt_mean",
    "mlp_mass",
    "mlp_rho_min",
    "mlp_rho_max",
    "mlp_p_min",
    "mlp_p_max",
    "mlp_nan_count",
    "mlp_rho_neg",
    "mlp_p_neg",
    "classical_t",
    "classical_steps",
    "classical_mass",
    "classical_rho_min",
    "classical_rho_max",
    "classical_p_min",
    "classical_p_max",
    "classical_nan_count",
]


def write_weno7_explosion_centerline_plot(
    step_dir: Path,
    u: np.ndarray,
    params: object,
    title: str,
) -> None:
    import run_weno7_explosion_compare as explosion

    pri = explosion.primitive_interior(u, params)
    rho = pri[..., 0]
    p = pri[..., 3]
    x = (np.arange(params.nx, dtype=np.float64) + 0.5) * params.dx
    y = (np.arange(params.ny, dtype=np.float64) + 0.5) * params.dy
    ix = int(np.argmin(np.abs(x - 0.5 * params.x_length)))
    iy = int(np.argmin(np.abs(y - 0.5 * params.y_length)))

    rho_x = rho[iy, :]
    rho_y = rho[:, ix]
    p_x = p[iy, :]
    p_y = p[:, ix]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=False)
    axes[0].plot(x, rho_x, color="#1f77b4", lw=1.8, label="rho along center y")
    axes[0].plot(y, rho_y, color="#d62728", lw=1.8, ls="--", label="rho along center x")
    axes[0].set_ylabel("density")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(frameon=False)
    axes[1].plot(x, p_x, color="#1f77b4", lw=1.8, label="p along center y")
    axes[1].plot(y, p_y, color="#d62728", lw=1.8, ls="--", label="p along center x")
    axes[1].set_xlabel("coordinate")
    axes[1].set_ylabel("pressure")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(step_dir / "mlp_centerline_rho_p_slices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    np.savetxt(
        step_dir / "mlp_centerline_slices.csv",
        np.column_stack([x, rho_x, p_x, y, rho_y, p_y]),
        delimiter=",",
        header="x,rho_y_center,p_y_center,y,rho_x_center,p_x_center",
        comments="",
    )


def run_weno7_explosion_validation(step: int, model: SharedBadnessMLP, args: argparse.Namespace) -> dict[str, object]:
    import run_weno7_explosion_compare as explosion

    model_was_training = model.training
    model.eval()
    eval_model_path = args.out_dir / "explosion_validation_models" / f"model_step_{step:06d}.npz"
    save_checkpoint(eval_model_path, model, args, step, {"explosion_validation_eval_model": True})
    if model_was_training:
        model.train()

    device = args.explosion_device or args.device
    explosion.wh.require_warp()
    explosion.wp.init()
    explosion.wp.set_device(device)
    params = explosion.wh.Params(
        nx=args.explosion_nx,
        ny=args.explosion_ny,
        x_length=args.explosion_x_length,
        y_length=args.explosion_y_length,
        cfl=args.explosion_cfl,
        t_end=args.explosion_t_end,
    )
    run_args = argparse.Namespace(
        device=device,
        t_end=args.explosion_t_end,
        max_steps=args.explosion_max_steps,
        boundary=args.explosion_boundary,
        riemann_solver=args.explosion_riemann_solver,
        eno_cutoff=args.explosion_eno_cutoff,
        mlp_derivative_mode=args.explosion_mlp_derivative_mode,
        report_interval=args.explosion_report_interval,
    )
    initial = explosion.make_explosion_state(params, args.explosion_init_quadrature, args.explosion_radius)
    mlp_params = explosion.weno7.load_mlp_params(eval_model_path, device)
    mlp, mlp_summary = explosion.run_solution(initial, params, run_args, mlp_params, "explosion_mlp")

    classical_summary: dict[str, float] = {}
    classical_t = classical_steps = classical_mass = classical_rho_min = classical_rho_max = classical_p_min = classical_p_max = classical_nan_count = float("nan")
    if args.explosion_run_classical:
        classical, classical_summary = explosion.run_solution(initial, params, run_args, None, "explosion_classical")
        classical_t = classical_summary["t"]
        classical_steps = classical_summary["steps"]
        classical_mass = classical_summary["mass"]
        classical_rho_min = classical_summary["rho_min"]
        classical_rho_max = classical_summary["rho_max"]
        classical_p_min = classical_summary["p_min"]
        classical_p_max = classical_summary["p_max"]
        classical_nan_count = classical_summary["nan_count"]

    failed = bool(mlp_summary["nan_count"] > 0 or mlp_summary["rho_neg"] > 0 or mlp_summary["p_neg"] > 0)
    step_dir = args.out_dir / "explosion_validation" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        step_dir / "explosion_results.npz",
        initial=initial,
        mlp=mlp,
        mlp_summary=np.array(mlp_summary, dtype=object),
        classical_summary=np.array(classical_summary, dtype=object),
    )
    explosion.plot_density_pressure(
        mlp,
        params,
        step_dir / "mlp_explosion_pressure_density.png",
        f"WENO7 MLP explosion step {step}, t={mlp_summary['t']:.3f}",
    )
    write_weno7_explosion_centerline_plot(
        step_dir,
        mlp,
        params,
        f"WENO7 MLP explosion centerline step {step}, t={mlp_summary['t']:.3f}",
    )
    if args.explosion_run_classical:
        explosion.plot_density_pressure(
            classical,
            params,
            step_dir / "classical_explosion_pressure_density.png",
            f"WENO7 classical explosion step {step}, t={classical_summary['t']:.3f}",
        )

    return {
        "step": step,
        "explosion_nx": args.explosion_nx,
        "explosion_ny": args.explosion_ny,
        "t_end": args.explosion_t_end,
        "cfl": args.explosion_cfl,
        "failed": float(failed),
        "mlp_t": mlp_summary["t"],
        "mlp_steps": mlp_summary["steps"],
        "dt_min": mlp_summary["dt_min"],
        "dt_max": mlp_summary["dt_max"],
        "dt_mean": mlp_summary["dt_mean"],
        "mlp_mass": mlp_summary["mass"],
        "mlp_rho_min": mlp_summary["rho_min"],
        "mlp_rho_max": mlp_summary["rho_max"],
        "mlp_p_min": mlp_summary["p_min"],
        "mlp_p_max": mlp_summary["p_max"],
        "mlp_nan_count": mlp_summary["nan_count"],
        "mlp_rho_neg": mlp_summary["rho_neg"],
        "mlp_p_neg": mlp_summary["p_neg"],
        "classical_t": classical_t,
        "classical_steps": classical_steps,
        "classical_mass": classical_mass,
        "classical_rho_min": classical_rho_min,
        "classical_rho_max": classical_rho_max,
        "classical_p_min": classical_p_min,
        "classical_p_max": classical_p_max,
        "classical_nan_count": classical_nan_count,
    }


def write_weno7_explosion_outputs(out_dir: Path, records: list[dict[str, object]]) -> None:
    save_csv(out_dir / "explosion_validation_metrics.csv", records, WENO7_EXPLOSION_FIELDS)
    if not records:
        return
    steps = np.array([r["step"] for r in records], dtype=np.float64)
    p_min = np.array([r["mlp_p_min"] for r in records], dtype=np.float64)
    rho_min = np.array([r["mlp_rho_min"] for r in records], dtype=np.float64)
    p_max = np.array([r["mlp_p_max"] for r in records], dtype=np.float64)
    failed = np.array([r["failed"] for r in records], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.0), sharex=True)
    axes[0].plot(steps, p_min, "o-", ms=3.0)
    axes[0].set_ylabel("MLP p_min")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, rho_min, "o-", ms=3.0)
    axes[1].set_ylabel("MLP rho_min")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, p_max, "o-", ms=3.0, label="p_max")
    axes[2].plot(steps, failed, "x--", ms=4.0, label="failed")
    axes[2].set_xlabel("offline step")
    axes[2].set_ylabel("p_max / failed")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "explosion_validation_trends.png", dpi=180)
    plt.close(fig)


WENO7_QUADRANT_RK4_FIELDS = [
    "step",
    "quadrant_nx",
    "quadrant_ny",
    "t_end",
    "cfl",
    "failed",
    "mlp_t",
    "mlp_steps",
    "dt_min",
    "dt_max",
    "dt_mean",
    "mlp_mass",
    "mlp_rho_min",
    "mlp_rho_max",
    "mlp_p_min",
    "mlp_p_max",
    "mlp_nan_count",
    "mlp_rho_neg",
    "mlp_p_neg",
]


def run_weno7_quadrant_rk4_validation(step: int, model: SharedBadnessMLP, args: argparse.Namespace) -> dict[str, object]:
    from weno7_point_rk4_shu import point_rk4 as rk4
    from weno7_point_rk4_shu import point_rk4_mlp as rk4_mlp
    from weno7_point_rk4_shu import run_quadrant_point_rk4 as quadrant

    step_dir = args.out_dir / "quadrant_rk4_validation" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()
    eval_model_path = args.out_dir / "quadrant_rk4_validation_models" / f"model_step_{step:06d}.npz"
    save_checkpoint(eval_model_path, model, args, step, {"quadrant_rk4_validation_eval_model": True})
    if model_was_training:
        model.train()

    device = args.quadrant_rk4_device or args.device
    params = rk4.Params(
        nx=args.quadrant_rk4_nx,
        ny=args.quadrant_rk4_ny,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        cfl=args.quadrant_rk4_cfl,
        t_end=args.quadrant_rk4_t_end,
    )

    record: dict[str, object] = {
        "step": step,
        "quadrant_nx": args.quadrant_rk4_nx,
        "quadrant_ny": args.quadrant_rk4_ny,
        "t_end": args.quadrant_rk4_t_end,
        "cfl": args.quadrant_rk4_cfl,
        "failed": 1.0,
        "mlp_t": float("nan"),
        "mlp_steps": float("nan"),
        "dt_min": float("nan"),
        "dt_max": float("nan"),
        "dt_mean": float("nan"),
        "mlp_mass": float("nan"),
        "mlp_rho_min": float("nan"),
        "mlp_rho_max": float("nan"),
        "mlp_p_min": float("nan"),
        "mlp_p_max": float("nan"),
        "mlp_nan_count": float("nan"),
        "mlp_rho_neg": float("nan"),
        "mlp_p_neg": float("nan"),
    }

    try:
        rk4.wh.require_warp()
        rk4.wp.init()
        rk4.wp.set_device(device)
        initial = quadrant.make_quadrant_state(params, args.quadrant_rk4_case, args.quadrant_rk4_init_quadrature)
        beta_model = rk4_mlp.TorchWeno7PointBeta(eval_model_path, device, params.gamma)
        mlp, mlp_summary = rk4_mlp.run_from_initial_mlp(
            initial.copy(),
            params,
            device=device,
            riemann_solver=args.quadrant_rk4_riemann_solver,
            beta_model=beta_model,
            report_interval=args.quadrant_rk4_report_interval,
            max_steps=args.quadrant_rk4_max_steps,
            boundary=args.quadrant_rk4_boundary,
            eno_cutoff=args.quadrant_rk4_eno_cutoff,
        )

        failed = bool(
            float(mlp_summary["nan_count"]) > 0.0
            or float(mlp_summary["rho_neg"]) > 0.0
            or float(mlp_summary["p_neg"]) > 0.0
            or float(mlp_summary["t"]) < args.quadrant_rk4_t_end - 1.0e-12
        )
        record.update(
            {
                "failed": float(failed),
                "mlp_t": float(mlp_summary["t"]),
                "mlp_steps": int(mlp_summary["steps"]),
                "dt_min": float(mlp_summary["dt_min"]),
                "dt_max": float(mlp_summary["dt_max"]),
                "dt_mean": float(mlp_summary["dt_mean"]),
                "mlp_mass": float(mlp_summary["mass"]),
                "mlp_rho_min": float(mlp_summary["rho_min"]),
                "mlp_rho_max": float(mlp_summary["rho_max"]),
                "mlp_p_min": float(mlp_summary["p_min"]),
                "mlp_p_max": float(mlp_summary["p_max"]),
                "mlp_nan_count": float(mlp_summary["nan_count"]),
                "mlp_rho_neg": float(mlp_summary["rho_neg"]),
                "mlp_p_neg": float(mlp_summary["p_neg"]),
            }
        )
        np.savez(
            step_dir / "quadrant_rk4_mlp_results.npz",
            initial=initial,
            mlp=mlp,
            mlp_summary=np.array(mlp_summary, dtype=object),
            record=np.array(record, dtype=object),
        )
        try:
            quadrant.plot_qstyle(
                mlp,
                params,
                step_dir / "mlp_point_rk4_pressure_rho_quiver_rho016_171_step005.png",
                (
                    f"WENO7 MLP point-RK4 {args.quadrant_rk4_case} "
                    f"{args.quadrant_rk4_riemann_solver} {params.nx}x{params.ny} "
                    f"step {step}, t={float(mlp_summary['t']):.3f}"
                ),
            )
        except Exception as plot_exc:  # Keep training alive even if a failed run cannot be contoured.
            (step_dir / "plot_error.txt").write_text(repr(plot_exc), encoding="utf-8")
    except Exception as exc:
        (step_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
    return record


def write_weno7_quadrant_rk4_outputs(out_dir: Path, records: list[dict[str, object]]) -> None:
    save_csv(out_dir / "quadrant_rk4_validation_metrics.csv", records, WENO7_QUADRANT_RK4_FIELDS)
    if not records:
        return
    steps = np.array([r["step"] for r in records], dtype=np.float64)
    t_reached = np.array([r["mlp_t"] for r in records], dtype=np.float64)
    p_min = np.array([r["mlp_p_min"] for r in records], dtype=np.float64)
    rho_min = np.array([r["mlp_rho_min"] for r in records], dtype=np.float64)
    failed = np.array([r["failed"] for r in records], dtype=np.float64)
    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.4), sharex=True)
    axes[0].plot(steps, t_reached, "o-", ms=3.0)
    axes[0].set_ylabel("reached t")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, p_min, "o-", ms=3.0)
    axes[1].set_ylabel("MLP p_min")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, rho_min, "o-", ms=3.0)
    axes[2].set_ylabel("MLP rho_min")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(steps, failed, "x--", ms=4.0)
    axes[3].set_xlabel("offline step")
    axes[3].set_ylabel("failed")
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "quadrant_rk4_validation_trends.png", dpi=180)
    plt.close(fig)


def _poly_cell_averages(power: int) -> np.ndarray:
    centers = np.arange(-3, 4, dtype=np.float64)
    left = centers - 0.5
    right = centers + 0.5
    return (right ** (power + 1) - left ** (power + 1)) / float(power + 1)


def check_weno7_coefficients() -> None:
    targets = {1: 0.5, 2: -0.5, 3: -ROOT3 / 6.0, 4: ROOT3 / 6.0}
    for power in range(7):
        q = torch.as_tensor(_poly_cell_averages(power).reshape(1, FULL_STENCIL), dtype=torch.float64)
        for lr in LR_VALUES:
            d = optimal_d(lr, q.device).reshape(1, R)
            pred = torch.sum(d * candidate_values(q, lr), dim=1).item()
            exact = targets[lr] ** power
            if abs(pred - exact) > 2.0e-11:
                raise RuntimeError(
                    f"WENO7 linear coefficient check failed: power={power} lr={lr} "
                    f"pred={pred:.17e} exact={exact:.17e}"
                )


def run(args: argparse.Namespace) -> None:
    if args.cell_average_mode != "exact":
        raise ValueError("WENO7 offline currently supports --cell-average-mode exact only.")
    if args.reconstruction_gamma_alpha < 0.0:
        raise ValueError("--reconstruction-gamma-alpha must be non-negative.")
    if args.weight_l2_lambda < 0.0:
        raise ValueError("--weight-l2-lambda must be non-negative.")
    if not (0.0 <= args.discontinuity_cutcell_prob <= 1.0):
        raise ValueError("--discontinuity-cutcell-prob must be in [0, 1].")
    if not args.discontinuity_value_min < args.discontinuity_value_max:
        raise ValueError("--discontinuity-value-min must be smaller than --discontinuity-value-max.")
    check_weno7_coefficients()

    device = torch_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
            args.discontinuity_x0_mode,
            args.discontinuity_cutcell_prob,
            args.discontinuity_value_min,
            args.discontinuity_value_max,
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
            args.discontinuity_x0_mode,
            args.discontinuity_cutcell_prob,
            args.discontinuity_value_min,
            args.discontinuity_value_max,
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
    explosion_records: list[dict[str, object]] = []
    quadrant_rk4_records: list[dict[str, object]] = []
    best_local_loss = float("inf")
    best_sod_l2 = float("inf")
    best_sod_linf = float("inf")
    best_explosion_p_min = -float("inf")

    print(
        f"offline_pretrain_start: weno_order={WENO_ORDER} steps={args.steps} batch={args.batch_size} dx={args.dx} "
        f"lr={args.lr:.3e} smooth_anchor_lambda={args.smooth_anchor_lambda:.3e} "
        f"reconstruction_gamma_alpha={args.reconstruction_gamma_alpha:.3e} "
        f"reconstruction_error_normalization={args.reconstruction_error_normalization} "
        f"nonlocal_omega_lambda={args.nonlocal_omega_lambda:.3e} "
        f"nonlocal_omega_tau={args.nonlocal_omega_tau:.3e} "
        f"nonlocal_omega_gamma_power={args.nonlocal_omega_gamma_power:.3e} "
        f"mirror_positive_targets={args.mirror_positive_targets} "
        f"training_target_side={args.training_target_side} "
        f"weight_l2_lambda={args.weight_l2_lambda:.3e} "
        f"validation_mode={validation_label} validation={validation_desc} gamma_filter={args.gamma_filter} "
        f"paper_grid_centers={args.paper_grid_centers} cell_average_mode={args.cell_average_mode} "
        f"mirror_augmentation={args.mirror_augmentation} paper_train_samples={paper_train_samples:.0f} "
        f"discontinuity_multiplier={args.discontinuity_multiplier:.3g} "
        f"discontinuity_x0_mode={args.discontinuity_x0_mode} "
        f"discontinuity_cutcell_prob={args.discontinuity_cutcell_prob:.3f} "
        f"discontinuity_value_range=[{args.discontinuity_value_min:.3g},{args.discontinuity_value_max:.3g}] "
        f"sod_eval={args.sod_eval} "
        f"sod_nx={args.sod_nx} sod_t_end={args.sod_t_end} sod_cfl={args.sod_cfl} "
        f"sod_init_quadrature={args.sod_init_quadrature} "
        f"sod_mlp_derivative_mode={args.sod_mlp_derivative_mode} sod_eno_cutoff={args.sod_eno_cutoff} "
        f"explosion_eval={args.explosion_eval} explosion_nx={args.explosion_nx} "
        f"explosion_t_end={args.explosion_t_end} explosion_cfl={args.explosion_cfl} "
        f"quadrant_rk4_eval={args.quadrant_rk4_eval} "
        f"quadrant_rk4_interval={args.quadrant_rk4_interval} "
        f"quadrant_rk4_nx={args.quadrant_rk4_nx} quadrant_rk4_t_end={args.quadrant_rk4_t_end} "
        f"quadrant_rk4_cfl={args.quadrant_rk4_cfl} device={args.device}"
    )
    print(
        f"paper_scale_reference: architecture=6->12->8->8->4, "
        f"domain=[-1,1] except sine=[0,1], dx=0.01 in paper; "
        f"this_run_dx={args.dx}, equivalent_epoch = step*batch_size/{paper_train_samples:.0f}"
    )
    print("kind_probs:", " ".join(f"{k}={p:.4f}" for k, p in zip(KIND_NAMES, kind_probs_np)))

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
            args.discontinuity_x0_mode,
            args.discontinuity_cutcell_prob,
            args.discontinuity_value_min,
            args.discontinuity_value_max,
        )
        loss, stats = loss_terms(
            model,
            batch,
            args.smooth_anchor_lambda,
            args.reconstruction_gamma_alpha,
            args.weight_l2_lambda,
            args.reconstruction_error_normalization,
            args.nonlocal_omega_lambda,
            args.nonlocal_omega_tau,
            args.nonlocal_omega_gamma_power,
            args.mirror_positive_targets,
            args.training_target_side,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().cpu())
        optimizer.step()

        if step == 1 or step % args.log_interval == 0:
            paper_epoch = step * args.batch_size / max(paper_train_samples, 1.0)
            print(
                f"step={step} loss={stats['loss']:.6e} recon={stats['reconstruction_loss']:.6e} "
                f"raw_recon={stats['raw_reconstruction_loss']:.6e} "
                f"anchor={stats['smooth_anchor_loss']:.6e} "
                f"nonlocal={stats['nonlocal_omega_loss']:.6e} "
                f"nonlocal_w={stats['nonlocal_omega_weighted']:.6e} "
                f"target_n={stats['training_target_count']:.0f} "
                f"gamma={stats['gamma_mean']:.4f} "
                f"gamma_w={stats['reconstruction_weight_mean']:.4f} "
                f"r=[{stats['r0_mean']:.4f},{stats['r1_mean']:.4f},{stats['r2_mean']:.4f},{stats['r3_mean']:.4f}] "
                f"paper_epoch={paper_epoch:.4f} grad={grad_norm:.3e}"
            )

        if args.checkpoint_interval > 0 and (step % args.checkpoint_interval == 0 or local_step == args.steps):
            save_checkpoint(args.out_dir / "checkpoints" / f"model_step_{step:06d}.npz", model, args, step, {"checkpoint_interval": args.checkpoint_interval})

        if (args.eval_step_one and local_step == 1) or (args.eval_interval > 0 and step % args.eval_interval == 0) or local_step == args.steps:
            local_overall, by_kind = evaluate_local(
                model,
                validation_batches,
                args.smooth_anchor_lambda,
                args.reconstruction_gamma_alpha,
                args.weight_l2_lambda,
                args.reconstruction_error_normalization,
                args.nonlocal_omega_lambda,
                args.nonlocal_omega_tau,
                args.nonlocal_omega_gamma_power,
                args.mirror_positive_targets,
                args.training_target_side,
            )
            improved = local_overall["loss"] <= best_local_loss
            best_local_loss = min(best_local_loss, local_overall["loss"])
            for row in by_kind:
                local_records.append({"step": step, **row})
            record = {
                "step": step,
                "paper_epoch_equiv": step * args.batch_size / max(paper_train_samples, 1.0),
                "train_loss": stats["loss"],
                "train_reconstruction_loss": stats["reconstruction_loss"],
                "train_raw_reconstruction_loss": stats["raw_reconstruction_loss"],
                "train_smooth_anchor_loss": stats["smooth_anchor_loss"],
                "train_nonlocal_omega_loss": stats["nonlocal_omega_loss"],
                "train_nonlocal_omega_weighted": stats["nonlocal_omega_weighted"],
                "train_target_count": stats["training_target_count"],
                "train_weight_l2_loss": stats["weight_l2_loss"],
                "train_reconstruction_weight_mean": stats["reconstruction_weight_mean"],
                "train_reconstruction_error_scale_mean": stats["reconstruction_error_scale_mean"],
                "local_val_loss": local_overall["loss"],
                "local_val_reconstruction_loss": local_overall["reconstruction_loss"],
                "local_val_raw_reconstruction_loss": local_overall["raw_reconstruction_loss"],
                "local_val_smooth_anchor_loss": local_overall["smooth_anchor_loss"],
                "local_val_nonlocal_omega_loss": local_overall["nonlocal_omega_loss"],
                "local_val_nonlocal_omega_weighted": local_overall["nonlocal_omega_weighted"],
                "local_val_target_count": local_overall["training_target_count"],
                "local_val_weight_l2_loss": local_overall["weight_l2_loss"],
                "local_val_reconstruction_weight_mean": local_overall["reconstruction_weight_mean"],
                "local_val_reconstruction_error_scale_mean": local_overall["reconstruction_error_scale_mean"],
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
                    "train_nonlocal_omega_loss",
                    "train_nonlocal_omega_weighted",
                    "train_target_count",
                    "train_weight_l2_loss",
                    "train_reconstruction_weight_mean",
                    "train_reconstruction_error_scale_mean",
                    "local_val_loss",
                    "local_val_reconstruction_loss",
                    "local_val_raw_reconstruction_loss",
                    "local_val_smooth_anchor_loss",
                    "local_val_nonlocal_omega_loss",
                    "local_val_nonlocal_omega_weighted",
                    "local_val_target_count",
                    "local_val_weight_l2_loss",
                    "local_val_reconstruction_weight_mean",
                    "local_val_reconstruction_error_scale_mean",
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
                    "nonlocal_omega_loss",
                    "nonlocal_omega_weighted",
                    "training_target_count",
                    "weight_l2_loss",
                    "gamma_mean",
                    "reconstruction_weight_mean",
                    "reconstruction_error_scale_mean",
                    "r0_mean",
                    "r1_mean",
                    "r2_mean",
                    "r3_mean",
                ],
            )
            plot_offline_history(args.out_dir, history)
            save_checkpoint(args.out_dir / "model_latest.npz", model, args, step, {"local_val_loss": local_overall["loss"]})
            if improved:
                save_checkpoint(args.out_dir / "model_best_local.npz", model, args, step, {"local_val_loss": local_overall["loss"]})
            print(
                f"local_validation step={step} local_loss={local_overall['loss']:.6e} "
                f"local_recon={local_overall['reconstruction_loss']:.6e} "
                f"best_local={best_local_loss:.6e}"
            )

            if args.sod_eval and args.sod_nx > 0:
                sod_record = run_weno7_sod_validation(step, model, args)
                sod_records.append(sod_record)
                write_weno7_sod_outputs(args.out_dir, sod_records)
                m_l2 = float(sod_record["mlp_vs_reference_l2"])
                m_linf = float(sod_record["mlp_vs_reference_linf"])
                if np.isfinite(m_l2) and m_l2 < best_sod_l2:
                    best_sod_l2 = m_l2
                    save_checkpoint(args.out_dir / "model_best_sod_l2.npz", model, args, step, {"best_sod_l2": best_sod_l2, "sod_validation": sod_record})
                if np.isfinite(m_linf) and m_linf < best_sod_linf:
                    best_sod_linf = m_linf
                    save_checkpoint(args.out_dir / "model_best_sod_linf.npz", model, args, step, {"best_sod_linf": best_sod_linf, "sod_validation": sod_record})
                print(
                    f"sod_validation step={step} MLP_ref_L2={m_l2:.6e} MLP_ref_Linf={m_linf:.6e} "
                    f"classic_ref_L2={float(sod_record['classical_vs_reference_l2']):.6e} "
                    f"classic_ref_Linf={float(sod_record['classical_vs_reference_linf']):.6e} "
                    f"gain_L2={float(sod_record['gain_vs_reference_l2']):.6e} "
                    f"best_sod_l2={best_sod_l2:.6e} best_sod_linf={best_sod_linf:.6e}"
                )

            if args.explosion_eval and args.explosion_nx > 0 and args.explosion_ny > 0:
                explosion_record = run_weno7_explosion_validation(step, model, args)
                explosion_records.append(explosion_record)
                write_weno7_explosion_outputs(args.out_dir, explosion_records)
                p_min = float(explosion_record["mlp_p_min"])
                failed = bool(float(explosion_record["failed"]) > 0.0)
                if (not failed) and np.isfinite(p_min) and p_min > best_explosion_p_min:
                    best_explosion_p_min = p_min
                    save_checkpoint(
                        args.out_dir / "model_best_explosion_pmin.npz",
                        model,
                        args,
                        step,
                        {"best_explosion_p_min": best_explosion_p_min, "explosion_validation": explosion_record},
                    )
                print(
                    f"explosion_validation step={step} failed={int(failed)} "
                    f"MLP_p_min={p_min:.6e} MLP_rho_min={float(explosion_record['mlp_rho_min']):.6e} "
                    f"MLP_p_max={float(explosion_record['mlp_p_max']):.6e} "
                    f"best_explosion_p_min={best_explosion_p_min:.6e}"
                )

        if (
            args.quadrant_rk4_eval
            and args.quadrant_rk4_interval > 0
            and args.quadrant_rk4_nx > 0
            and args.quadrant_rk4_ny > 0
            and (step % args.quadrant_rk4_interval == 0 or local_step == args.steps)
        ):
            quadrant_record = run_weno7_quadrant_rk4_validation(step, model, args)
            quadrant_rk4_records.append(quadrant_record)
            write_weno7_quadrant_rk4_outputs(args.out_dir, quadrant_rk4_records)
            failed = bool(float(quadrant_record["failed"]) > 0.0)
            print(
                f"quadrant_rk4_validation step={step} failed={int(failed)} "
                f"t={float(quadrant_record['mlp_t']):.6e} "
                f"steps={int(float(quadrant_record['mlp_steps'])) if np.isfinite(float(quadrant_record['mlp_steps'])) else -1} "
                f"p_min={float(quadrant_record['mlp_p_min']):.6e} "
                f"rho_min={float(quadrant_record['mlp_rho_min']):.6e} "
                f"nan={float(quadrant_record['mlp_nan_count']):.0f}"
            )

    final_step = args.step_offset + args.steps
    save_checkpoint(
        args.out_dir / "model_latest.npz",
        model,
        args,
        final_step,
        {
            "best_local_loss": best_local_loss,
            "best_sod_l2": best_sod_l2,
            "best_sod_linf": best_sod_linf,
            "best_explosion_p_min": best_explosion_p_min,
            "quadrant_rk4_records": len(quadrant_rk4_records),
        },
    )
    print(
        f"done: history={args.out_dir / 'offline_history.csv'} "
        f"model={args.out_dir / 'model_latest.npz'} best_local={args.out_dir / 'model_best_local.npz'} "
        f"best_sod={args.out_dir / 'model_best_sod_l2.npz'} "
        f"best_explosion={args.out_dir / 'model_best_explosion_pmin.npz'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--samples-per-kind", type=int, default=4096, help="Requested canonical samples per kind; validation is capped unless --validation-samples-per-kind is set.")
    parser.add_argument("--validation-samples-per-kind", type=int, default=0, help="Fixed local validation samples per kind; defaults to min(samples_per_kind, 4096).")
    parser.add_argument("--validation-mode", choices=("uniform", "paper20"), default="uniform")
    parser.add_argument("--paper-validation-fraction", type=float, default=NOGUEIRA_VALIDATION_FRACTION)
    parser.add_argument("--validation-sample-scale", type=float, default=1.0)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--paper-grid-centers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cell-average-mode", choices=("exact",), default="exact")
    parser.add_argument("--mirror-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--smooth-anchor-lambda", type=float, default=1.0e-1)
    parser.add_argument("--weight-l2-lambda", type=float, default=1.0e-9)
    parser.add_argument("--reconstruction-gamma-alpha", type=float, default=0.2)
    parser.add_argument(
        "--nonlocal-omega-lambda",
        type=float,
        default=0.0,
        help=(
            "Penalty strength for non-adjacent WENO7 nonlinear weights. "
            "The penalty is gamma_s^p times hinge products for pairs (0,2), (0,3), and (1,3)."
        ),
    )
    parser.add_argument(
        "--nonlocal-omega-tau",
        type=float,
        default=0.05,
        help="Hinge threshold for --nonlocal-omega-lambda.",
    )
    parser.add_argument(
        "--nonlocal-omega-gamma-power",
        type=float,
        default=1.0,
        help="Power p in gamma_s^p for the nonlocal omega penalty.",
    )
    parser.add_argument(
        "--mirror-positive-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy diagnostic option: mirror positive target points LR=1 and LR=4 "
            "during offline loss evaluation. Leave this off for positive-only training "
            "with runtime negative-side mirroring."
        ),
    )
    parser.add_argument(
        "--training-target-side",
        choices=("all", "positive", "negative"),
        default="all",
        help=(
            "Which target points enter the offline reconstruction loss. "
            "positive uses only i+1/2 and i+sqrt(3)/6; negative uses only the mirrored negative-side targets."
        ),
    )
    parser.add_argument(
        "--reconstruction-error-normalization",
        choices=("none", "stencil-range"),
        default="none",
        help=(
            "Normalize reconstruction errors before squaring. "
            "stencil-range divides each sample by max(max(q)-min(q), 1), "
            "while raw_reconstruction_loss remains unnormalized for diagnostics."
        ),
    )
    parser.add_argument("--discontinuity-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--discontinuity-x0-mode",
        choices=("fixed", "cutcell", "mixed"),
        default="fixed",
        help=(
            "fixed keeps the Nogueira-style discontinuity at x0=0.5; "
            "cutcell samples x0 strictly inside one of the seven FVM cells to create partial-cell averages; "
            "mixed uses fixed jumps except for a random fraction controlled by --discontinuity-cutcell-prob."
        ),
    )
    parser.add_argument(
        "--discontinuity-cutcell-prob",
        type=float,
        default=0.2,
        help="Fraction of discontinuity samples using random cut-cell x0 when --discontinuity-x0-mode mixed.",
    )
    parser.add_argument(
        "--discontinuity-value-min",
        type=float,
        default=-40.0,
        help="Lower bound for canonical discontinuity left/right states.",
    )
    parser.add_argument(
        "--discontinuity-value-max",
        type=float,
        default=80.0,
        help="Upper bound for canonical discontinuity left/right states.",
    )
    parser.add_argument("--gamma-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--eval-step-one", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--sod-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sod-nx", type=int, default=100)
    parser.add_argument("--sod-ny", type=int, default=8)
    parser.add_argument("--sod-t-end", type=float, default=0.25)
    parser.add_argument("--sod-cfl", type=float, default=0.4)
    parser.add_argument("--sod-axis", choices=("x", "y"), default="x")
    parser.add_argument("--sod-init-quadrature", type=int, default=15)
    parser.add_argument("--sod-eno-cutoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sod-mlp-derivative-mode", choices=("all", "classical", "normal"), default="all")
    parser.add_argument("--sod-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--explosion-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explosion-nx", type=int, default=100)
    parser.add_argument("--explosion-ny", type=int, default=100)
    parser.add_argument("--explosion-x-length", type=float, default=2.0)
    parser.add_argument("--explosion-y-length", type=float, default=2.0)
    parser.add_argument("--explosion-radius", type=float, default=0.4)
    parser.add_argument("--explosion-t-end", type=float, default=0.25)
    parser.add_argument("--explosion-cfl", type=float, default=0.4)
    parser.add_argument("--explosion-init-quadrature", type=int, default=15)
    parser.add_argument("--explosion-riemann-solver", choices=("evilin", "hllc"), default="evilin")
    parser.add_argument("--explosion-boundary", choices=("outflow", "periodic"), default="outflow")
    parser.add_argument("--explosion-eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explosion-mlp-derivative-mode", choices=("all", "classical", "normal"), default="normal")
    parser.add_argument("--explosion-run-classical", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explosion-report-interval", type=int, default=0)
    parser.add_argument("--explosion-max-steps", type=int, default=10_000_000)
    parser.add_argument("--explosion-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--quadrant-rk4-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quadrant-rk4-interval", type=int, default=100000)
    parser.add_argument("--quadrant-rk4-nx", type=int, default=400)
    parser.add_argument("--quadrant-rk4-ny", type=int, default=400)
    parser.add_argument("--quadrant-rk4-t-end", type=float, default=0.5)
    parser.add_argument("--quadrant-rk4-cfl", type=float, default=0.3)
    parser.add_argument("--quadrant-rk4-case", choices=("case12", "case6"), default="case12")
    parser.add_argument("--quadrant-rk4-init-quadrature", type=int, default=15)
    parser.add_argument("--quadrant-rk4-riemann-solver", choices=("evilin", "hllc"), default="evilin")
    parser.add_argument("--quadrant-rk4-boundary", choices=("outflow", "periodic"), default="outflow")
    parser.add_argument("--quadrant-rk4-eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quadrant-rk4-report-interval", type=int, default=0)
    parser.add_argument("--quadrant-rk4-max-steps", type=int, default=10_000_000)
    parser.add_argument("--quadrant-rk4-device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("plots/WENO7_MLP/weno7_offline_pretrain"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
