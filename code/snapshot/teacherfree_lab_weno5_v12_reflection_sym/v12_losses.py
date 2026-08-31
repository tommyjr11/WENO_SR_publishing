#!/usr/bin/env python3
"""Bidirectional exact-FVM rollout losses for reflection-symmetric WENO5 V12."""
from __future__ import annotations

import math

import numpy as np
import torch

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v4_fvm_e2e.apost_advect_fvm import (
    LR_OFFSETS,
    _fundamental_and_residual,
    _scaled_smooth_l1,
    reconstruct_q,
    stencils,
)
from teacherfree_lab_weno5_v4_fvm_e2e.fvm_profiles import make_profiles
from teacherfree_lab_weno5_v5_fvm_e2e.v5_losses import (
    flat_region_mask,
    second_difference,
)

torch.set_default_dtype(torch.float64)


def balanced_velocities(batch: int, device: torch.device) -> torch.Tensor:
    """Return an exactly balanced a=+1/-1 batch."""
    if batch < 2 or batch % 2 != 0:
        raise ValueError("bidirectional batches must be positive and even")
    velocities = torch.ones(batch, device=device, dtype=torch.float64)
    velocities[batch // 2 :] = -1.0
    return velocities


def _select_by_velocity(
    positive: torch.Tensor,
    negative: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    mask = velocities > 0.0
    while mask.ndim < positive.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, positive, negative)


def exact_cell_average_signed(
    profiles,
    n: int,
    time: float,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Exact cell averages of q_t + a q_x = 0 for a in {-1,+1}."""
    positive = profiles.cell_average(n, time)
    negative = profiles.cell_average(n, -time)
    return _select_by_velocity(positive, negative, velocities)


def exact_point_targets_signed(
    profiles,
    n: int,
    time: float,
    offsets: tuple[float, ...],
    velocities: torch.Tensor,
) -> torch.Tensor:
    positive = profiles.point_targets(n, time, offsets)
    negative = profiles.point_targets(n, -time, offsets)
    return _select_by_velocity(positive, negative, velocities)


def _reconstruct_lr_from_ratios(
    ratios: torch.Tensor,
    q: torch.Tensor,
    lr: int,
) -> torch.Tensor:
    d = W.optimal_d(lr, q.device).reshape(1, 3).expand(q.shape[0], 3)
    omega = W.omega_from_ratio(ratios, lr)
    omega = torch.where(W.plateau_mask(q).reshape(-1, 1), d, omega)
    return torch.sum(omega * W.candidate_values(q, lr), dim=1)


def interface_state_signed(
    model,
    state: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Return the upwind state at interface i+1/2 for each batch sample.

    Positive speed uses lr=1 centered on cell i. Negative speed uses lr=2
    centered on cell i+1, hence the one-cell left roll after reconstruction.
    """
    batch, n = state.shape
    q = stencils(state)
    if model is None or isinstance(model, str):
        positive = reconstruct_q(model, q, 1).reshape(batch, n)
        negative_at_left_face = reconstruct_q(model, q, 2).reshape(batch, n)
    else:
        ratios = model(W.weno5_features(q))
        positive = _reconstruct_lr_from_ratios(ratios, q, 1).reshape(batch, n)
        negative_at_left_face = _reconstruct_lr_from_ratios(
            ratios, q, 2
        ).reshape(batch, n)
    negative = torch.roll(negative_at_left_face, -1, dims=1)
    return _select_by_velocity(positive, negative, velocities)


def rhs_signed(
    model,
    state: torch.Tensor,
    dxinv: float,
    velocities: torch.Tensor,
) -> torch.Tensor:
    face_state = interface_state_signed(model, state, velocities)
    flux = velocities.reshape(-1, 1) * face_state
    return -(flux - torch.roll(flux, 1, dims=1)) * dxinv


def ssprk3_step_signed(
    model,
    state: torch.Tensor,
    dt: float,
    dxinv: float,
    velocities: torch.Tensor,
) -> torch.Tensor:
    u1 = state + dt * rhs_signed(model, state, dxinv, velocities)
    u2 = 0.75 * state + 0.25 * (
        u1 + dt * rhs_signed(model, u1, dxinv, velocities)
    )
    return state / 3.0 + (2.0 / 3.0) * (
        u2 + dt * rhs_signed(model, u2, dxinv, velocities)
    )


def _state_error_per_sample(
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


def cfl_schedule(distance_cells: float, cfl: float) -> tuple[float, ...]:
    """Return full CFL steps plus one shortened step at the exact distance."""
    if distance_cells <= 0.0 or cfl <= 0.0:
        raise ValueError("distance_cells and cfl must be positive")
    full_steps = int(math.floor(distance_cells / cfl))
    remainder = distance_cells - full_steps * cfl
    tolerance = 2.0e-13 * max(1.0, distance_cells)
    if remainder < tolerance:
        remainder = 0.0
    elif cfl - remainder < tolerance:
        full_steps += 1
        remainder = 0.0
    schedule = [float(cfl)] * full_steps
    if remainder > 0.0:
        schedule.append(float(remainder))
    if not schedule:
        schedule.append(float(distance_cells))
    if abs(sum(schedule) - distance_cells) > 5.0e-12 * max(1.0, distance_cells):
        raise RuntimeError("CFL schedule does not reach the requested distance")
    return tuple(schedule)


def rollout_distance(
    model,
    initial: torch.Tensor,
    distance_cells: float,
    cfl: float,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Advance exactly ``distance_cells`` cell widths with a shortened last step."""
    n = initial.shape[1]
    dx = 1.0 / float(n)
    state = initial
    for step_cfl in cfl_schedule(distance_cells, cfl):
        state = ssprk3_step_signed(
            model, state, step_cfl * dx, 1.0 / dx, velocities
        )
    return state


def trajectory_loss_equal_distance(
    model,
    profiles,
    n: int,
    distance_cells: float,
    cfl: float,
    face_path_lambda: float,
    exact_recon_lambda: float,
    flat_d2_lambda: float,
    flat_tolerance: float,
    tv_bg_lambda: float,
    velocities: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise complete SSPRK3 steps over an identical physical displacement."""
    dx = 1.0 / float(n)
    schedule = cfl_schedule(distance_cells, cfl)
    with torch.no_grad():
        state = profiles.cell_average(n, 0.0)
        qscale = torch.clamp(torch.max(torch.abs(state), dim=1).values, min=1.0)
        value_range = torch.clamp(
            torch.max(state, dim=1).values - torch.min(state, dim=1).values,
            min=1.0e-6 * qscale,
        )

    state_loss = torch.zeros((), device=state.device)
    face_loss = torch.zeros((), device=state.device)
    recon_loss = torch.zeros((), device=state.device)
    flat_loss = torch.zeros((), device=state.device)
    tv_excess = torch.zeros((), device=state.device)
    travelled = 0.0

    for step_cfl in schedule:
        weight = step_cfl / distance_cells
        time_now = travelled / float(n)
        with torch.no_grad():
            exact_now = exact_cell_average_signed(
                profiles, n, time_now, velocities
            )
            exact_points = exact_point_targets_signed(
                profiles, n, time_now, LR_OFFSETS, velocities
            )

        if face_path_lambda > 0.0:
            face_loss = face_loss + weight * _scaled_smooth_l1(
                interface_state_signed(model, state, velocities),
                exact_points[:, 0, :],
                qscale,
            )

        if exact_recon_lambda > 0.0:
            q_exact = stencils(exact_now)
            ratios = model(W.weno5_features(q_exact))
            plateau = W.plateau_mask(q_exact).reshape(-1, 1)
            values = []
            for lr in W.LR_VALUES:
                d = W.optimal_d(lr, q_exact.device).reshape(1, 3).expand(
                    q_exact.shape[0], 3
                )
                omega = torch.where(plateau, d, W.omega_from_ratio(ratios, lr))
                value = torch.sum(omega * W.candidate_values(q_exact, lr), dim=1)
                values.append(value.reshape(state.shape[0], n))
            recon_loss = recon_loss + weight * _scaled_smooth_l1(
                torch.stack(values, dim=1), exact_points, qscale
            )

        state = ssprk3_step_signed(
            model, state, step_cfl * dx, 1.0 / dx, velocities
        )
        travelled += step_cfl
        with torch.no_grad():
            exact_next = exact_cell_average_signed(
                profiles, n, travelled / float(n), velocities
            )
        state_error = _state_error_per_sample(
            state, exact_next, qscale, value_range
        )
        state_loss = state_loss + weight * torch.mean(state_error)

        if flat_d2_lambda > 0.0:
            mask = flat_region_mask(exact_next, value_range, flat_tolerance).to(
                state.dtype
            )
            count = torch.clamp(torch.sum(mask, dim=1), min=1.0)
            d2_error = torch.abs(
                second_difference(state) - second_difference(exact_next)
            )
            per_sample = torch.sum(mask * d2_error, dim=1) / (
                count * value_range
            )
            flat_loss = flat_loss + weight * torch.mean(per_sample)

        if tv_bg_lambda > 0.0:
            tv_model = torch.sum(
                torch.abs(state - torch.roll(state, 1, dims=1)), dim=1
            )
            tv_exact = torch.sum(
                torch.abs(exact_next - torch.roll(exact_next, 1, dims=1)), dim=1
            )
            tv_excess = tv_excess + weight * torch.mean(
                torch.relu(tv_model - tv_exact) / value_range
            )

    total = (
        state_loss
        + face_path_lambda * face_loss
        + exact_recon_lambda * recon_loss
        + flat_d2_lambda * flat_loss
        + tv_bg_lambda * tv_excess
    )
    positive = velocities > 0.0
    negative = velocities < 0.0
    final_error = _state_error_per_sample(state, exact_next, qscale, value_range)
    return total, {
        "trajectory": float(state_loss.detach()),
        "trajectory_plus": float(torch.mean(final_error[positive]).detach()),
        "trajectory_minus": float(torch.mean(final_error[negative]).detach()),
        "face_path": float(face_loss.detach()),
        "exact_recon": float(recon_loss.detach()),
        "flat_d2": float(flat_loss.detach()),
        "tv_excess": float(tv_excess.detach()),
        "rk_steps": float(len(schedule)),
    }


def semidiscrete_rhs_loss_signed(
    model,
    profiles,
    n: int,
    time: float,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Match the exact FVM spatial operator for both advection directions."""
    dx = 1.0 / float(n)
    with torch.no_grad():
        state = exact_cell_average_signed(profiles, n, time, velocities)
        exact_face = exact_point_targets_signed(
            profiles, n, time, (0.5,), velocities
        )[:, 0, :]
        exact_rhs = -velocities.reshape(-1, 1) * (
            exact_face - torch.roll(exact_face, 1, dims=1)
        ) / dx
        qscale = torch.clamp(torch.max(torch.abs(state), dim=1).values, min=1.0)
        value_range = torch.clamp(
            torch.max(state, dim=1).values - torch.min(state, dim=1).values,
            min=1.0e-6 * qscale,
        )
    numerical_rhs = rhs_signed(model, state, 1.0 / dx, velocities)
    error = (numerical_rhs - exact_rhs) * dx / value_range.reshape(-1, 1)
    return torch.mean(torch.sqrt(torch.square(error) + 1.0e-12) - 1.0e-6)


def robust_tail(values: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    """A saturating mean/CVaR aggregate that keeps rare violations visible."""
    flat = values.reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.numel())))
    tail = torch.topk(flat, count, sorted=False).values
    return 0.25 * torch.mean(flat) + 0.75 * torch.mean(tail)


def make_exact_mode_profiles(
    batch: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
):
    """Sample resolved modes plus a small under-resolved safety subset."""
    if batch < 4:
        raise ValueError("mode batch must be at least four")
    profile = make_profiles(batch, n, device, gen, kind="smooth")
    profile.sine_amp.zero_()
    profile.gaussian_amp.zero_()
    profile.triangle_amp.zero_()
    profile.ellipse_amp.zero_()
    profile.box_amp.zero_()

    resolved_count = max(1, int(round(0.75 * batch)))
    resolved_count = min(resolved_count, batch - 1)
    resolved_options = torch.as_tensor(
        (6.0, 8.0, 10.0, 12.0, 16.0, 24.0), device=device
    )
    safety_options = torch.as_tensor((4.0, 5.0), device=device)
    requested = torch.empty(batch, device=device, dtype=torch.float64)
    resolved_pick = torch.randint(
        resolved_options.numel(),
        (resolved_count,),
        device=device,
        generator=gen,
    )
    requested[:resolved_count] = resolved_options[resolved_pick]
    safety_pick = torch.randint(
        safety_options.numel(),
        (batch - resolved_count,),
        device=device,
        generator=gen,
    )
    requested[resolved_count:] = safety_options[safety_pick]
    wave_number = torch.clamp(torch.round(float(n) / requested), min=1.0)
    actual_cpw = float(n) / wave_number

    base_scale = torch.clamp(torch.abs(profile.base[:, 0]), min=1.0)
    log_amplitude = np.log10(2.0e-2) + (
        np.log10(3.0e-1) - np.log10(2.0e-2)
    ) * torch.rand(batch, device=device, generator=gen)
    amplitude = torch.pow(10.0, log_amplitude) * base_scale
    sign = torch.where(
        torch.rand(batch, device=device, generator=gen) < 0.5, -1.0, 1.0
    )
    profile.sine_amp[:, 0] = amplitude * sign
    profile.sine_k[:, 0] = wave_number
    profile.sine_phase[:, 0] = 2.0 * np.pi * torch.rand(
        batch, device=device, generator=gen
    )
    return profile, actual_cpw


def _mode_vector(value: torch.Tensor, wave_number: torch.Tensor) -> torch.Tensor:
    n = value.shape[1]
    x = (torch.arange(n, device=value.device, dtype=torch.float64) + 0.5) / n
    angle = 2.0 * np.pi * wave_number.reshape(-1, 1) * x.reshape(1, n)
    centered = value - torch.mean(value, dim=1, keepdim=True)
    return torch.stack(
        (
            2.0 * torch.mean(centered * torch.cos(angle), dim=1),
            2.0 * torch.mean(centered * torch.sin(angle), dim=1),
        ),
        dim=1,
    )


def exact_mode_rollout_loss(
    model,
    profiles,
    cpw: torch.Tensor,
    n: int,
    distance_cells: float,
    cfl: float,
    growth_weight: float = 1.0,
    growth_tolerance: float = 1.0e-4,
    cvar_fraction: float = 0.25,
    velocities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match exact translated cell averages without using linear-d as a target."""
    wave_number = profiles.sine_k[:, 0]
    if velocities is None:
        velocities = balanced_velocities(profiles.batch, profiles.device)
    if velocities.shape != (profiles.batch,):
        raise ValueError("velocities must have one entry per mode profile")
    with torch.no_grad():
        initial = profiles.cell_average(n, 0.0)
        initial_vector = _mode_vector(initial, wave_number)
        initial_amplitude = torch.linalg.vector_norm(initial_vector, dim=1)
        signal_energy = torch.clamp(
            torch.mean(
                torch.square(initial - torch.mean(initial, dim=1, keepdim=True)),
                dim=1,
            ),
            min=1.0e-30,
        )
    resolved = cpw >= 5.5
    schedule = cfl_schedule(distance_cells, cfl)
    learned = initial
    travelled = 0.0
    path_error = torch.zeros(initial.shape[0], device=initial.device)
    for step_cfl in schedule:
        learned = ssprk3_step_signed(
            model, learned, step_cfl / float(n), float(n), velocities
        )
        travelled += step_cfl
        with torch.no_grad():
            exact_step = exact_cell_average_signed(
                profiles, n, travelled / float(n), velocities
            )
        normalized_error = torch.mean(
            torch.square(learned - exact_step), dim=1
        ) / signal_energy
        path_error = path_error + (step_cfl / distance_cells) * normalized_error

    with torch.no_grad():
        exact = exact_cell_average_signed(
            profiles, n, distance_cells / float(n), velocities
        )
        exact_vector = _mode_vector(exact, wave_number)
        exact_amplitude = torch.linalg.vector_norm(exact_vector, dim=1)

    learned_vector = _mode_vector(learned, wave_number)
    learned_amplitude = torch.linalg.vector_norm(learned_vector, dim=1)
    gain_model = learned_amplitude / torch.clamp(initial_amplitude, min=1.0e-20)
    gain_exact = exact_amplitude / torch.clamp(initial_amplitude, min=1.0e-20)
    gain_abs_error = torch.abs(gain_model - gain_exact)
    learned_unit = learned_vector / torch.clamp(
        learned_amplitude.reshape(-1, 1), min=1.0e-20
    )
    exact_unit = exact_vector / torch.clamp(
        exact_amplitude.reshape(-1, 1), min=1.0e-20
    )
    phase_error = torch.linalg.vector_norm(learned_unit - exact_unit, dim=1)
    _, residual_model = _fundamental_and_residual(learned, wave_number)
    energy_scale = torch.clamp(torch.square(initial_amplitude), min=1.0e-30)
    harmonic_error = residual_model / energy_scale
    complex_error = torch.sum(
        torch.square(learned_vector - exact_vector), dim=1
    ) / energy_scale

    rhs_model = rhs_signed(model, initial, float(n), velocities)
    rhs_model_vector = _mode_vector(rhs_model, wave_number)
    norm2 = torch.clamp(torch.sum(torch.square(initial_vector), dim=1), min=1.0e-30)
    angular_frequency = 2.0 * np.pi * wave_number
    growth_ratio = torch.sum(rhs_model_vector * initial_vector, dim=1) / (
        norm2 * angular_frequency
    )
    growth_values = torch.square(
        torch.relu(growth_ratio - growth_tolerance)
    )
    exact_loss = robust_tail(path_error[resolved], cvar_fraction)
    growth_loss = robust_tail(growth_values, cvar_fraction)
    total = exact_loss + growth_weight * growth_loss
    damping = -growth_ratio
    positive = velocities > 0.0
    negative = velocities < 0.0
    return total, {
        "mode": float(total.detach()),
        "mode_exact": float(exact_loss.detach()),
        "mode_growth": float(growth_loss.detach()),
        "mode_complex_error": float(torch.mean(complex_error[resolved]).detach()),
        "mode_gain_abs_error": float(
            torch.mean(gain_abs_error[resolved]).detach()
        ),
        "mode_phase_error": float(torch.mean(phase_error[resolved]).detach()),
        "mode_harmonic_error": float(
            torch.mean(harmonic_error[resolved]).detach()
        ),
        "mode_gain_mean": float(torch.mean(gain_model[resolved]).detach()),
        "mode_damping_min": float(torch.min(damping).detach()),
        "mode_damping_mean": float(torch.mean(damping).detach()),
        "mode_damping_min_plus": float(torch.min(damping[positive]).detach()),
        "mode_damping_min_minus": float(torch.min(damping[negative]).detach()),
        "mode_exact_plus": float(torch.mean(path_error[positive & resolved]).detach()),
        "mode_exact_minus": float(torch.mean(path_error[negative & resolved]).detach()),
        "mode_resolved_fraction": float(torch.mean(resolved.to(torch.float64))),
        "mode_rk_steps": float(len(schedule)),
    }


@torch.no_grad()
def final_state_error_signed(
    model,
    profiles,
    n: int,
    distance_cells: float,
    cfl: float,
    velocities: torch.Tensor,
) -> tuple[float, float, float]:
    initial = profiles.cell_average(n, 0.0)
    prediction = rollout_distance(
        model, initial, distance_cells, cfl, velocities
    )
    exact = exact_cell_average_signed(
        profiles, n, distance_cells / float(n), velocities
    )
    qscale = torch.clamp(torch.max(torch.abs(initial), dim=1).values, min=1.0)
    value_range = torch.clamp(
        torch.max(initial, dim=1).values - torch.min(initial, dim=1).values,
        min=1.0e-6 * qscale,
    )
    errors = _state_error_per_sample(prediction, exact, qscale, value_range)
    positive = velocities > 0.0
    negative = velocities < 0.0
    return (
        float(torch.mean(errors)),
        float(torch.mean(errors[positive])),
        float(torch.mean(errors[negative])),
    )


@torch.no_grad()
def fixed_symbol_audit(
    model,
    n: int,
    device: torch.device,
    amplitude: float = 0.1,
) -> dict[str, float]:
    """Report damping in both directions and their reflection defect."""
    rows: dict[str, float] = {}
    for requested_cpw in (6.0, 8.0, 12.0):
        wave_number = max(1, int(round(n / requested_cpw)))
        cpw = float(n) / wave_number
        x = (torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n
        angle = 2.0 * np.pi * wave_number * x
        average_factor = float(np.sinc(wave_number / n))
        one_state = 1.0 + amplitude * average_factor * torch.sin(angle)
        state = torch.stack((one_state, one_state), dim=0)
        velocities = torch.as_tensor((1.0, -1.0), device=device)
        numerical_rhs = rhs_signed(model, state, float(n), velocities)
        mode_k = torch.as_tensor(
            (wave_number, wave_number), device=device, dtype=torch.float64
        )
        state_vector = _mode_vector(state, mode_k)
        rhs_vector = _mode_vector(numerical_rhs, mode_k)
        norm2 = torch.sum(torch.square(state_vector), dim=1)
        growth = torch.sum(rhs_vector * state_vector, dim=1) / norm2
        damping = -growth / (2.0 * np.pi * wave_number)
        tag = f"cpw{int(requested_cpw):02d}"
        rows[f"damping_{tag}_plus"] = float(damping[0])
        rows[f"damping_{tag}_minus"] = float(damping[1])
        rows[f"damping_{tag}"] = float(torch.mean(damping))
        rows[f"damping_sym_defect_{tag}"] = float(
            torch.abs(damping[0] - damping[1])
        )
        rows[f"actual_cpw{int(requested_cpw):02d}"] = cpw
    rows["damping_min"] = min(
        value for key, value in rows.items() if key.startswith("damping_cpw")
    )
    rows["damping_sym_defect_max"] = max(
        value
        for key, value in rows.items()
        if key.startswith("damping_sym_defect_cpw")
    )
    return rows
