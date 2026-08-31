#!/usr/bin/env python3
"""Autoregressive exact-FVM and classical non-regression losses for V19."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v4_fvm_e2e.apost_advect_fvm import (
    LR_OFFSETS,
    _scaled_smooth_l1,
    stencils,
)
from teacherfree_lab_weno5_v5_fvm_e2e.v5_losses import (
    flat_region_mask,
    second_difference,
)
from teacherfree_lab_weno5_v12_reflection_sym.v12_losses import (
    _state_error_per_sample,
    exact_cell_average_signed,
    exact_point_targets_signed,
    interface_state_signed,
    ssprk3_step_signed,
)

torch.set_default_dtype(torch.float64)


def _tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.numel())))
    return torch.topk(flat, count, sorted=False).values.mean()


def _family_mean(values: torch.Tensor, family_id: torch.Tensor) -> torch.Tensor:
    """Average samples within each present family, then average families."""
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    means = []
    for family in torch.unique(family_id, sorted=True):
        means.append(per_sample[family_id == family].mean())
    return torch.stack(means).mean()


def _robust_violation(
    values: torch.Tensor,
    family_id: torch.Tensor,
    cvar_fraction: float,
) -> torch.Tensor:
    return 0.25 * _family_mean(values, family_id) + 0.75 * _tail_mean(
        values, cvar_fraction
    )


def _periodic_local_rms(error: torch.Tensor, window: int) -> torch.Tensor:
    if window < 1 or window > error.shape[1]:
        raise ValueError("local window must lie between one and the grid size")
    left = (window - 1) // 2
    right = window - 1 - left
    padded = F.pad(error.square().unsqueeze(1), (left, right), mode="circular")
    return torch.sqrt(
        F.avg_pool1d(padded, window, stride=1).squeeze(1) + 1.0e-30
    )


def _reconstruct_all_heads(model, exact_state: torch.Tensor) -> torch.Tensor:
    batch, n = exact_state.shape
    q = stencils(exact_state)
    ratios = model(W.weno5_features(q))
    plateau = W.plateau_mask(q).reshape(-1, 1)
    values = []
    for lr in W.LR_VALUES:
        d = W.optimal_d(lr, q.device).reshape(1, 3).expand(q.shape[0], 3)
        omega = torch.where(plateau, d, W.omega_from_ratio(ratios, lr))
        values.append(
            torch.sum(omega * W.candidate_values(q, lr), dim=1).reshape(batch, n)
        )
    return torch.stack(values, dim=1)


def autoregressive_trajectory_loss(
    model,
    profiles,
    n: int,
    n_steps: int,
    cfl: float,
    velocities: torch.Tensor,
    *,
    state_lambda: float,
    face_path_lambda: float,
    exact_recon_lambda: float,
    flat_d2_lambda: float,
    flat_tolerance: float,
    tv_lambda: float,
    global_guard_lambda: float,
    local_guard_lambda: float,
    local_window: int,
    cvar_fraction: float,
    guard_tolerance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Advance model and WENO5-JS trajectories and supervise every full RK step.

    The model state remains in the graph for the entire trajectory.  The
    classical path is detached and only defines a non-regression baseline.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    dx = 1.0 / float(n)
    dt = float(cfl) * dx
    family_id = profiles.family_id

    with torch.no_grad():
        state = exact_cell_average_signed(profiles, n, 0.0, velocities)
        classical = state.clone()
        qscale = torch.clamp(torch.max(torch.abs(state), dim=1).values, min=1.0)
        value_range = torch.clamp(
            torch.max(state, dim=1).values - torch.min(state, dim=1).values,
            min=1.0e-6 * qscale,
        )

    state_sum = torch.zeros((), device=state.device)
    face_sum = torch.zeros((), device=state.device)
    recon_sum = torch.zeros((), device=state.device)
    flat_sum = torch.zeros((), device=state.device)
    tv_sum = torch.zeros((), device=state.device)
    global_guard_sum = torch.zeros((), device=state.device)
    local_guard_sum = torch.zeros((), device=state.device)
    model_error_last = torch.zeros((), device=state.device)
    classical_error_last = torch.zeros((), device=state.device)

    for step in range(n_steps):
        time_now = float(step) * dt
        with torch.no_grad():
            exact_now = exact_cell_average_signed(
                profiles, n, time_now, velocities
            )
            exact_points = exact_point_targets_signed(
                profiles, n, time_now, LR_OFFSETS, velocities
            )

        if face_path_lambda > 0.0:
            face_sum = face_sum + _scaled_smooth_l1(
                interface_state_signed(model, state, velocities),
                exact_points[:, 0, :],
                qscale,
            )
        if exact_recon_lambda > 0.0:
            recon_sum = recon_sum + _scaled_smooth_l1(
                _reconstruct_all_heads(model, exact_now), exact_points, qscale
            )

        state = ssprk3_step_signed(model, state, dt, 1.0 / dx, velocities)
        with torch.no_grad():
            classical = ssprk3_step_signed(
                "classical", classical, dt, 1.0 / dx, velocities
            )
            exact_next = exact_cell_average_signed(
                profiles, n, float(step + 1) * dt, velocities
            )

        model_per_sample = _state_error_per_sample(
            state, exact_next, qscale, value_range
        )
        with torch.no_grad():
            classical_per_sample = _state_error_per_sample(
                classical, exact_next, qscale, value_range
            )
        model_error_last = model_per_sample.mean()
        classical_error_last = classical_per_sample.mean()
        if state_lambda > 0.0:
            state_sum = state_sum + _family_mean(model_per_sample, family_id)

        if global_guard_lambda > 0.0:
            violation = torch.relu(
                model_per_sample - classical_per_sample - guard_tolerance
            )
            global_guard_sum = global_guard_sum + _robust_violation(
                violation, family_id, cvar_fraction
            )
        if local_guard_lambda > 0.0:
            scale = value_range.reshape(-1, 1)
            model_local = _periodic_local_rms((state - exact_next) / scale, local_window)
            with torch.no_grad():
                classical_local = _periodic_local_rms(
                    (classical - exact_next) / scale, local_window
                )
            local_violation = torch.relu(
                model_local - classical_local - guard_tolerance
            )
            local_guard_sum = local_guard_sum + _robust_violation(
                local_violation, family_id, cvar_fraction
            )

        if flat_d2_lambda > 0.0:
            mask = flat_region_mask(
                exact_next, value_range, flat_tolerance
            ).to(state.dtype)
            count = torch.clamp(torch.sum(mask, dim=1), min=1.0)
            d2_error = torch.abs(
                second_difference(state) - second_difference(exact_next)
            )
            flat_per_sample = torch.sum(mask * d2_error, dim=1) / (
                count * value_range
            )
            flat_sum = flat_sum + _family_mean(flat_per_sample, family_id)

        if tv_lambda > 0.0:
            tv_model = torch.sum(
                torch.abs(state - torch.roll(state, 1, dims=1)), dim=1
            )
            with torch.no_grad():
                tv_exact = torch.sum(
                    torch.abs(exact_next - torch.roll(exact_next, 1, dims=1)),
                    dim=1,
                )
            tv_violation = torch.relu((tv_model - tv_exact) / value_range)
            tv_sum = tv_sum + _family_mean(tv_violation, family_id)

    inv = 1.0 / float(n_steps)
    state_loss = state_sum * inv
    face_loss = face_sum * inv
    recon_loss = recon_sum * inv
    flat_loss = flat_sum * inv
    tv_loss = tv_sum * inv
    global_guard = global_guard_sum * inv
    local_guard = local_guard_sum * inv
    total = (
        state_lambda * state_loss
        + face_path_lambda * face_loss
        + exact_recon_lambda * recon_loss
        + flat_d2_lambda * flat_loss
        + tv_lambda * tv_loss
        + global_guard_lambda * global_guard
        + local_guard_lambda * local_guard
    )
    return total, {
        "trajectory": float(state_loss.detach()),
        "face_path": float(face_loss.detach()),
        "exact_recon": float(recon_loss.detach()),
        "flat_d2": float(flat_loss.detach()),
        "tv_excess": float(tv_loss.detach()),
        "global_js_guard": float(global_guard.detach()),
        "local_js_guard": float(local_guard.detach()),
        "final_model_error": float(model_error_last.detach()),
        "final_classical_error": float(classical_error_last.detach()),
        "final_vs_classical": float(
            (model_error_last / torch.clamp(classical_error_last, min=1.0e-30)).detach()
        ),
    }
