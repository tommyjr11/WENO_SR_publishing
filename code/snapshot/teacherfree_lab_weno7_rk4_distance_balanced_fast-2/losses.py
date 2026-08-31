#!/usr/bin/env python3
"""Long-trajectory exact-FVM losses for WENO7 with Shu RK4."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

import rk4_advection as A
import weno7_core as W

torch.set_default_dtype(torch.float64)


def exact_cell_average_signed(
    profiles,
    n: int,
    time: float,
    velocities: torch.Tensor,
) -> torch.Tensor:
    positive = profiles.cell_average(n, time)
    negative = profiles.cell_average(n, -time)
    return A.select_by_velocity(positive, negative, velocities)


def exact_point_targets_signed(
    profiles,
    n: int,
    time: float,
    offsets: tuple[float, ...],
    velocities: torch.Tensor,
) -> torch.Tensor:
    positive = profiles.point_targets(n, time, offsets)
    negative = profiles.point_targets(n, -time, offsets)
    return A.select_by_velocity(positive, negative, velocities)


def scaled_smooth_l1(
    prediction: torch.Tensor,
    exact: torch.Tensor,
    qscale: torch.Tensor,
) -> torch.Tensor:
    scale = qscale
    while scale.ndim < prediction.ndim:
        scale = scale.unsqueeze(-1)
    error = (prediction - exact) / scale
    return torch.mean(
        torch.sqrt(torch.square(error) + 1.0e-12) - 1.0e-6
    )


def state_error_per_sample(
    prediction: torch.Tensor,
    exact: torch.Tensor,
    qscale: torch.Tensor,
    value_range: torch.Tensor,
) -> torch.Tensor:
    absolute = (prediction - exact) / qscale.reshape(-1, 1)
    relative = (prediction - exact) / value_range.reshape(-1, 1)
    l4_squared = torch.sqrt(
        torch.mean(torch.pow(torch.abs(absolute), 4), dim=1) + 1.0e-30
    )
    relative_l2 = torch.mean(torch.square(relative), dim=1)
    return l4_squared + 0.05 * relative_l2


def second_difference(state: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(state, shifts=-1, dims=1)
        - 2.0 * state
        + torch.roll(state, shifts=1, dims=1)
    )


def flat_region_mask(
    exact: torch.Tensor,
    value_range: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    left = torch.abs(exact - torch.roll(exact, shifts=1, dims=1))
    right = torch.abs(torch.roll(exact, shifts=-1, dims=1) - exact)
    flat = torch.maximum(left, right) <= tolerance * value_range.reshape(-1, 1)
    return (
        flat
        & torch.roll(flat, shifts=1, dims=1)
        & torch.roll(flat, shifts=-1, dims=1)
    )


def family_mean(values: torch.Tensor, family_id: torch.Tensor) -> torch.Tensor:
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    means = []
    for family in torch.unique(family_id, sorted=True):
        means.append(per_sample[family_id == family].mean())
    return torch.stack(means).mean()


def tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.numel())))
    return torch.topk(flat, count, sorted=False).values.mean()


def robust_violation(
    values: torch.Tensor,
    family_id: torch.Tensor,
    cvar_fraction: float,
) -> torch.Tensor:
    return 0.25 * family_mean(values, family_id) + 0.75 * tail_mean(
        values, cvar_fraction
    )


def periodic_local_rms(error: torch.Tensor, window: int) -> torch.Tensor:
    if window < 1 or window > error.shape[1]:
        raise ValueError("local window must lie inside the grid")
    left = (window - 1) // 2
    right = window - 1 - left
    padded = functional.pad(
        error.square().unsqueeze(1), (left, right), mode="circular"
    )
    return torch.sqrt(
        functional.avg_pool1d(padded, window, stride=1).squeeze(1) + 1.0e-30
    )


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
    """Supervise every complete Shu-RK4 step, never an internal RK stage."""
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
            torch.max(state, dim=1).values
            - torch.min(state, dim=1).values,
            min=1.0e-6 * qscale,
        )

    state_sum = torch.zeros((), device=state.device)
    face_sum = torch.zeros((), device=state.device)
    reconstruction_sum = torch.zeros((), device=state.device)
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
                profiles, n, time_now, W.LR_OFFSETS, velocities
            )

        if face_path_lambda > 0.0:

            def face_term(
                current: torch.Tensor,
                target: torch.Tensor = exact_points[:, 0, :],
                scale: torch.Tensor = qscale,
            ) -> torch.Tensor:
                return scaled_smooth_l1(
                    A.interface_state_signed(model, current, velocities),
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

            def reconstruction_term(
                exact_state: torch.Tensor,
                target: torch.Tensor = exact_points,
                scale: torch.Tensor = qscale,
            ) -> torch.Tensor:
                return scaled_smooth_l1(
                    A.all_head_reconstruction(model, exact_state),
                    target,
                    scale,
                )

            reconstruction_sum = reconstruction_sum + checkpoint(
                reconstruction_term,
                exact_now,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        def advance(current: torch.Tensor, local_dt: float = dt) -> torch.Tensor:
            return A.shu_rk4_step_signed(
                model, current, local_dt, 1.0 / dx, velocities
            )

        state = checkpoint(
            advance,
            state,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        with torch.no_grad():
            classical = A.shu_rk4_step_signed(
                "classical", classical, dt, 1.0 / dx, velocities
            )
            exact_next = exact_cell_average_signed(
                profiles, n, float(step + 1) * dt, velocities
            )

        model_per_sample = state_error_per_sample(
            state, exact_next, qscale, value_range
        )
        with torch.no_grad():
            classical_per_sample = state_error_per_sample(
                classical, exact_next, qscale, value_range
            )
        model_error_last = model_per_sample.mean()
        classical_error_last = classical_per_sample.mean()

        if state_lambda > 0.0:
            state_sum = state_sum + family_mean(model_per_sample, family_id)

        if global_guard_lambda > 0.0:
            violation = torch.relu(
                model_per_sample - classical_per_sample - guard_tolerance
            )
            global_guard_sum = global_guard_sum + robust_violation(
                violation, family_id, cvar_fraction
            )

        if local_guard_lambda > 0.0:
            scale = value_range.reshape(-1, 1)
            model_local = periodic_local_rms(
                (state - exact_next) / scale, local_window
            )
            with torch.no_grad():
                classical_local = periodic_local_rms(
                    (classical - exact_next) / scale, local_window
                )
            local_violation = torch.relu(
                model_local - classical_local - guard_tolerance
            )
            local_guard_sum = local_guard_sum + robust_violation(
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
            per_sample = torch.sum(mask * d2_error, dim=1) / (
                count * value_range
            )
            flat_sum = flat_sum + family_mean(per_sample, family_id)

        if tv_lambda > 0.0:
            tv_model = torch.sum(
                torch.abs(state - torch.roll(state, shifts=1, dims=1)), dim=1
            )
            with torch.no_grad():
                tv_exact = torch.sum(
                    torch.abs(
                        exact_next
                        - torch.roll(exact_next, shifts=1, dims=1)
                    ),
                    dim=1,
                )
            tv_violation = torch.relu(
                (tv_model - tv_exact) / value_range
            )
            tv_sum = tv_sum + family_mean(tv_violation, family_id)

    inverse_steps = 1.0 / float(n_steps)
    state_loss = state_sum * inverse_steps
    face_loss = face_sum * inverse_steps
    reconstruction_loss = reconstruction_sum * inverse_steps
    flat_loss = flat_sum * inverse_steps
    tv_loss = tv_sum * inverse_steps
    global_guard = global_guard_sum * inverse_steps
    local_guard = local_guard_sum * inverse_steps
    total = (
        state_lambda * state_loss
        + face_path_lambda * face_loss
        + exact_recon_lambda * reconstruction_loss
        + flat_d2_lambda * flat_loss
        + tv_lambda * tv_loss
        + global_guard_lambda * global_guard
        + local_guard_lambda * local_guard
    )
    return total, {
        "trajectory": float(state_loss.detach()),
        "face_path": float(face_loss.detach()),
        "exact_recon": float(reconstruction_loss.detach()),
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


@torch.no_grad()
def final_state_error_signed(
    model,
    profiles,
    n: int,
    distance_cells: float,
    cfl: float,
    velocities: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    raw_steps = float(distance_cells) / float(cfl)
    n_steps = int(round(raw_steps))
    if abs(raw_steps - n_steps) > 1.0e-12:
        raise ValueError("evaluation distance must contain full RK steps")
    dx = 1.0 / float(n)
    initial = exact_cell_average_signed(profiles, n, 0.0, velocities)
    state = initial
    for _ in range(n_steps):
        state = A.shu_rk4_step_signed(
            model, state, cfl * dx, 1.0 / dx, velocities
        )
    exact = exact_cell_average_signed(
        profiles, n, distance_cells * dx, velocities
    )
    qscale = torch.clamp(
        torch.max(torch.abs(initial), dim=1).values, min=1.0
    )
    value_range = torch.clamp(
        torch.max(initial, dim=1).values
        - torch.min(initial, dim=1).values,
        min=1.0e-6 * qscale,
    )
    per_sample = state_error_per_sample(
        state, exact, qscale, value_range
    )
    return float(torch.mean(per_sample)), per_sample

