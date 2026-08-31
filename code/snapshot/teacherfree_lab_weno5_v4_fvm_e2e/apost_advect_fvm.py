#!/usr/bin/env python3
"""Differentiable WENO5/SSPRK3 rollout against analytic FVM trajectories."""
from __future__ import annotations

import numpy as np
import torch

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v4_fvm_e2e.fvm_profiles import ProfileBatch

torch.set_default_dtype(torch.float64)

LR_OFFSETS = tuple(W.LR_TARGET_X[lr] for lr in W.LR_VALUES)


def stencils(u: torch.Tensor) -> torch.Tensor:
    bsz, n = u.shape
    up = torch.cat([u[:, -2:], u, u[:, :2]], dim=1)
    return up.unfold(1, 5, 1).reshape(bsz * n, 5)


def reconstruction_omega(model, q: torch.Tensor, lr: int) -> torch.Tensor:
    d = W.optimal_d(lr, q.device).reshape(1, 3).expand(q.shape[0], 3)
    if model is None:
        omega = d
    elif isinstance(model, str) and model == "classical":
        omega = W.classical_omega(q, lr)
    else:
        omega = W.omega_from_ratio(model(W.weno5_features(q)), lr)
    return torch.where(W.plateau_mask(q).reshape(-1, 1), d, omega)


def reconstruct_q(model, q: torch.Tensor, lr: int) -> torch.Tensor:
    omega = reconstruction_omega(model, q, lr)
    return torch.sum(omega * W.candidate_values(q, lr), dim=1)


def reconstruct_iplus(model, u: torch.Tensor) -> torch.Tensor:
    bsz, n = u.shape
    return reconstruct_q(model, stencils(u), 1).reshape(bsz, n)


def rhs(model, u: torch.Tensor, dxinv: float) -> torch.Tensor:
    flux = reconstruct_iplus(model, u)
    return -(flux - torch.roll(flux, 1, dims=1)) * dxinv


def ssprk3_step(model, u: torch.Tensor, dt: float, dxinv: float) -> torch.Tensor:
    u1 = u + dt * rhs(model, u, dxinv)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(model, u1, dxinv))
    return u / 3.0 + (2.0 / 3.0) * (u2 + dt * rhs(model, u2, dxinv))


def rollout(model, u0: torch.Tensor, n_steps: int, cfl: float) -> torch.Tensor:
    n = u0.shape[1]
    dx = 1.0 / float(n)
    u = u0
    for _ in range(n_steps):
        u = ssprk3_step(model, u, cfl * dx, 1.0 / dx)
    return u


def _state_error(pred: torch.Tensor, exact: torch.Tensor, qscale: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
    e_abs = (pred - exact) / qscale.reshape(-1, 1)
    e_rel = (pred - exact) / rng.reshape(-1, 1)
    l4_squared = torch.sqrt(torch.mean(torch.pow(torch.abs(e_abs), 4), dim=1) + 1.0e-30)
    relative_l2 = torch.mean(torch.square(e_rel), dim=1)
    return torch.mean(l4_squared + 0.05 * relative_l2)


def _scaled_smooth_l1(pred: torch.Tensor, exact: torch.Tensor, qscale: torch.Tensor) -> torch.Tensor:
    scale = qscale
    while scale.ndim < pred.ndim:
        scale = scale.unsqueeze(-1)
    z = (pred - exact) / scale
    return torch.mean(torch.sqrt(torch.square(z) + 1.0e-12) - 1.0e-6)


def trajectory_loss(
    model,
    profiles: ProfileBatch,
    n: int,
    n_steps: int,
    cfl: float,
    face_path_lambda: float,
    exact_recon_lambda: float,
    tv_bg_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise every complete RK step, not the internal SSPRK stages."""
    dx = 1.0 / float(n)
    dt = cfl * dx
    with torch.no_grad():
        u = profiles.cell_average(n, 0.0)
        u0 = u
        qscale = torch.clamp(torch.max(torch.abs(u0), dim=1).values, min=1.0)
        rng = torch.clamp(torch.max(u0, dim=1).values - torch.min(u0, dim=1).values, min=1.0e-6 * qscale)

    state_sum = torch.zeros((), device=u.device)
    path_face_sum = torch.zeros((), device=u.device)
    exact_recon_sum = torch.zeros((), device=u.device)
    tv_sum = torch.zeros((), device=u.device)

    for step in range(n_steps):
        t = float(step) * dt
        with torch.no_grad():
            exact_now = profiles.cell_average(n, t)
            exact_points = profiles.point_targets(n, t, LR_OFFSETS)

        if face_path_lambda > 0.0:
            face = reconstruct_iplus(model, u)
            path_face_sum = path_face_sum + _scaled_smooth_l1(face, exact_points[:, 0, :], qscale)

        if exact_recon_lambda > 0.0:
            q_exact = stencils(exact_now)
            if model is None or isinstance(model, str):
                reconstructed = [reconstruct_q(model, q_exact, lr).reshape(profiles.batch, n) for lr in W.LR_VALUES]
            else:
                ratios = model(W.weno5_features(q_exact))
                plateau = W.plateau_mask(q_exact).reshape(-1, 1)
                reconstructed = []
                for lr in W.LR_VALUES:
                    d = W.optimal_d(lr, q_exact.device).reshape(1, 3).expand(q_exact.shape[0], 3)
                    omega = W.omega_from_ratio(ratios, lr)
                    omega = torch.where(plateau, d, omega)
                    val = torch.sum(omega * W.candidate_values(q_exact, lr), dim=1)
                    reconstructed.append(val.reshape(profiles.batch, n))
            recon = torch.stack(reconstructed, dim=1)
            exact_recon_sum = exact_recon_sum + _scaled_smooth_l1(recon, exact_points, qscale)

        u = ssprk3_step(model, u, dt, 1.0 / dx)
        with torch.no_grad():
            exact_next = profiles.cell_average(n, float(step + 1) * dt)
        state_sum = state_sum + _state_error(u, exact_next, qscale, rng)

        if tv_bg_lambda > 0.0:
            tv_num = torch.sum(torch.abs(u - torch.roll(u, 1, dims=1)), dim=1)
            tv_exact = torch.sum(torch.abs(exact_next - torch.roll(exact_next, 1, dims=1)), dim=1)
            tv_sum = tv_sum + torch.mean(torch.relu(tv_num - tv_exact) / rng)

    inv_steps = 1.0 / float(n_steps)
    state_loss = state_sum * inv_steps
    path_face_loss = path_face_sum * inv_steps
    exact_recon_loss = exact_recon_sum * inv_steps
    tv_excess = tv_sum * inv_steps
    total = (
        state_loss
        + face_path_lambda * path_face_loss
        + exact_recon_lambda * exact_recon_loss
        + tv_bg_lambda * tv_excess
    )
    stats = {
        "trajectory": float(state_loss.detach()),
        "face_path": float(path_face_loss.detach()),
        "exact_recon": float(exact_recon_loss.detach()),
        "tv_excess": float(tv_excess.detach()),
    }
    return total, stats


def _fundamental_and_residual(u: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = u.shape[1]
    x = (torch.arange(n, device=u.device, dtype=torch.float64) + 0.5) / float(n)
    angle = 2.0 * np.pi * k.reshape(-1, 1) * x.reshape(1, n)
    centered = u - torch.mean(u, dim=1, keepdim=True)
    sin_basis = torch.sin(angle)
    cos_basis = torch.cos(angle)
    a = 2.0 * torch.mean(centered * sin_basis, dim=1)
    b = 2.0 * torch.mean(centered * cos_basis, dim=1)
    fundamental = a.reshape(-1, 1) * sin_basis + b.reshape(-1, 1) * cos_basis
    amplitude = torch.sqrt(torch.square(a) + torch.square(b) + 1.0e-30)
    residual_energy = torch.mean(torch.square(centered - fundamental), dim=1)
    return amplitude, residual_energy


def shortwave_stability_loss(
    model,
    profiles: ProfileBatch,
    n: int,
    n_steps: int,
    cfl: float,
    gain_tol: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Prevent learned 4/6/8-cell modes from being less damped than linear d."""
    with torch.no_grad():
        u0 = profiles.cell_average(n, 0.0)
        linear = rollout(None, u0, n_steps, cfl)
        amp0, _ = _fundamental_and_residual(u0, profiles.sine_k[:, 0])
        amp_d, residual_d = _fundamental_and_residual(linear, profiles.sine_k[:, 0])
        gain_d = amp_d / torch.clamp(amp0, min=1.0e-14)

    learned = rollout(model, u0, n_steps, cfl)
    amp_m, residual_m = _fundamental_and_residual(learned, profiles.sine_k[:, 0])
    gain_m = amp_m / torch.clamp(amp0, min=1.0e-14)
    scale_energy = torch.clamp(torch.square(amp0), min=1.0e-20)
    harmonic_m = residual_m / scale_energy
    harmonic_d = residual_d / scale_energy

    gain_pen = torch.mean(torch.square(torch.relu(gain_m - gain_d - gain_tol)))
    growth_pen = torch.mean(torch.square(torch.relu(gain_m - 1.0)))
    harmonic_pen = torch.mean(torch.relu(harmonic_m - harmonic_d - 1.0e-6))
    loss = gain_pen + growth_pen + 0.1 * harmonic_pen
    stats = {
        "shortwave": float(loss.detach()),
        "short_gain_model": float(torch.mean(gain_m).detach()),
        "short_gain_linear": float(torch.mean(gain_d).detach()),
        "short_harmonic": float(torch.mean(harmonic_m).detach()),
    }
    return loss, stats


@torch.no_grad()
def final_state_error(model, profiles: ProfileBatch, n: int, n_steps: int, cfl: float) -> float:
    u0 = profiles.cell_average(n, 0.0)
    pred = rollout(model, u0, n_steps, cfl)
    exact = profiles.cell_average(n, n_steps * cfl / float(n))
    qscale = torch.clamp(torch.max(torch.abs(u0), dim=1).values, min=1.0)
    rng = torch.clamp(torch.max(u0, dim=1).values - torch.min(u0, dim=1).values, min=1.0e-6 * qscale)
    return float(_state_error(pred, exact, qscale, rng))
