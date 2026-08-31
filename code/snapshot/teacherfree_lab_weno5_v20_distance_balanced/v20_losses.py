#!/usr/bin/env python3
"""Checkpointed V19 losses for very long WENO5 autoregressive trajectories."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from teacherfree_lab_weno5_v4_fvm_e2e.apost_advect_fvm import (
    LR_OFFSETS,
    _scaled_smooth_l1,
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
from teacherfree_lab_weno5_v19_autoregressive_js.v19_losses import (
    _family_mean,
    _periodic_local_rms,
    _reconstruct_all_heads,
    _robust_violation,
)

torch.set_default_dtype(torch.float64)


def checkpointed_autoregressive_trajectory_loss(
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
    """Supervise every full SSPRK3 step while checkpointing model activations."""
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
            def face_term(
                current: torch.Tensor,
                target: torch.Tensor = exact_points[:, 0, :],
                scale: torch.Tensor = qscale,
            ) -> torch.Tensor:
                return _scaled_smooth_l1(
                    interface_state_signed(model, current, velocities),
                    target,
                    scale,
                )

            face_sum = face_sum + checkpoint(
                face_term,
                state,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        if exact_recon_lambda > 0.0:
            def recon_term(
                exact_state: torch.Tensor,
                target: torch.Tensor = exact_points,
                scale: torch.Tensor = qscale,
            ) -> torch.Tensor:
                return _scaled_smooth_l1(
                    _reconstruct_all_heads(model, exact_state),
                    target,
                    scale,
                )

            recon_sum = recon_sum + checkpoint(
                recon_term,
                exact_now,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        def advance(
            current: torch.Tensor,
            local_dt: float = dt,
        ) -> torch.Tensor:
            return ssprk3_step_signed(
                model, current, local_dt, 1.0 / dx, velocities
            )

        state = checkpoint(
            advance,
            state,
            use_reentrant=False,
            preserve_rng_state=False,
        )
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
            model_local = _periodic_local_rms(
                (state - exact_next) / scale, local_window
            )
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
            (
                model_error_last
                / torch.clamp(classical_error_last, min=1.0e-30)
            ).detach()
        ),
    }
