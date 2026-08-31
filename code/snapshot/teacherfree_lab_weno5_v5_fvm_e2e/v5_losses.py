#!/usr/bin/env python3
"""Full-step losses for the stability-focused WENO5-v5 experiment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v4_fvm_e2e.apost_advect_fvm import (
    LR_OFFSETS,
    _fundamental_and_residual,
    _scaled_smooth_l1,
    _state_error,
    reconstruct_iplus,
    rollout,
    ssprk3_step,
    stencils,
)
from teacherfree_lab_weno5_v4_fvm_e2e.fvm_profiles import ProfileBatch, make_profiles

torch.set_default_dtype(torch.float64)

_GL15_X, _GL15_W = np.polynomial.legendre.leggauss(15)


def second_difference(u: torch.Tensor) -> torch.Tensor:
    return torch.roll(u, -1, dims=1) - 2.0 * u + torch.roll(u, 1, dims=1)


def flat_region_mask(exact: torch.Tensor, rng: torch.Tensor, tolerance: float) -> torch.Tensor:
    left = torch.abs(exact - torch.roll(exact, 1, dims=1))
    right = torch.abs(torch.roll(exact, -1, dims=1) - exact)
    flat = torch.maximum(left, right) <= tolerance * rng.reshape(-1, 1)
    return flat & torch.roll(flat, 1, dims=1) & torch.roll(flat, -1, dims=1)


def trajectory_loss_v5(
    model,
    profiles: ProfileBatch,
    n: int,
    n_steps: int,
    cfl: float,
    face_path_lambda: float,
    exact_recon_lambda: float,
    flat_d2_lambda: float,
    flat_tolerance: float,
    tv_bg_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Analytic state/point supervision at every complete SSPRK3 step."""
    dx = 1.0 / float(n)
    dt = cfl * dx
    with torch.no_grad():
        u = profiles.cell_average(n, 0.0)
        qscale = torch.clamp(torch.max(torch.abs(u), dim=1).values, min=1.0)
        rng = torch.clamp(
            torch.max(u, dim=1).values - torch.min(u, dim=1).values,
            min=1.0e-6 * qscale,
        )

    state_sum = torch.zeros((), device=u.device)
    face_sum = torch.zeros((), device=u.device)
    recon_sum = torch.zeros((), device=u.device)
    flat_sum = torch.zeros((), device=u.device)
    tv_sum = torch.zeros((), device=u.device)

    for step in range(n_steps):
        t = float(step) * dt
        with torch.no_grad():
            exact_now = profiles.cell_average(n, t)
            exact_points = profiles.point_targets(n, t, LR_OFFSETS)

        if face_path_lambda > 0.0:
            face_sum = face_sum + _scaled_smooth_l1(
                reconstruct_iplus(model, u), exact_points[:, 0, :], qscale
            )

        if exact_recon_lambda > 0.0:
            q_exact = stencils(exact_now)
            ratios = model(W.weno5_features(q_exact))
            plateau = W.plateau_mask(q_exact).reshape(-1, 1)
            values = []
            for lr in W.LR_VALUES:
                d = W.optimal_d(lr, q_exact.device).reshape(1, 3).expand(q_exact.shape[0], 3)
                omega = torch.where(plateau, d, W.omega_from_ratio(ratios, lr))
                value = torch.sum(omega * W.candidate_values(q_exact, lr), dim=1)
                values.append(value.reshape(profiles.batch, n))
            recon_sum = recon_sum + _scaled_smooth_l1(
                torch.stack(values, dim=1), exact_points, qscale
            )

        u = ssprk3_step(model, u, dt, 1.0 / dx)
        with torch.no_grad():
            exact_next = profiles.cell_average(n, float(step + 1) * dt)
        state_sum = state_sum + _state_error(u, exact_next, qscale, rng)

        if flat_d2_lambda > 0.0:
            mask = flat_region_mask(exact_next, rng, flat_tolerance).to(u.dtype)
            count = torch.clamp(torch.sum(mask, dim=1), min=1.0)
            d2_error = torch.abs(second_difference(u) - second_difference(exact_next))
            per_sample = torch.sum(mask * d2_error, dim=1) / (count * rng)
            flat_sum = flat_sum + torch.mean(per_sample)

        if tv_bg_lambda > 0.0:
            tv_num = torch.sum(torch.abs(u - torch.roll(u, 1, dims=1)), dim=1)
            tv_exact = torch.sum(torch.abs(exact_next - torch.roll(exact_next, 1, dims=1)), dim=1)
            tv_sum = tv_sum + torch.mean(torch.relu(tv_num - tv_exact) / rng)

    inv = 1.0 / float(n_steps)
    state_loss = state_sum * inv
    face_loss = face_sum * inv
    recon_loss = recon_sum * inv
    flat_loss = flat_sum * inv
    tv_excess = tv_sum * inv
    total = (
        state_loss
        + face_path_lambda * face_loss
        + exact_recon_lambda * recon_loss
        + flat_d2_lambda * flat_loss
        + tv_bg_lambda * tv_excess
    )
    return total, {
        "trajectory": float(state_loss.detach()),
        "face_path": float(face_loss.detach()),
        "exact_recon": float(recon_loss.detach()),
        "flat_d2": float(flat_loss.detach()),
        "tv_excess": float(tv_excess.detach()),
    }


def _uniform(
    gen: torch.Generator,
    device: torch.device,
    lo: float,
    hi: float,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen, dtype=torch.float64)


def make_shortwave_profiles_v5(
    batch: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[ProfileBatch, torch.Tensor]:
    profile = make_profiles(batch, n, device, gen, kind="smooth")
    profile.sine_amp.zero_()
    cpw_options = torch.as_tensor((4.0, 6.0, 8.0, 12.0), device=device)
    pick = torch.randint(0, len(cpw_options), (batch,), device=device, generator=gen)
    k = torch.clamp(torch.round(float(n) / cpw_options[pick]), min=1.0)
    actual_cpw = float(n) / k
    scale = torch.clamp(torch.abs(profile.base[:, 0]), min=1.0)
    amplitude = torch.pow(10.0, _uniform(gen, device, -5.0, np.log10(0.3), (batch,))) * scale
    sign = torch.where(torch.rand(batch, device=device, generator=gen) < 0.5, -1.0, 1.0)
    profile.sine_amp[:, 0] = amplitude * sign
    profile.sine_k[:, 0] = k
    profile.sine_phase[:, 0] = _uniform(gen, device, 0.0, 2.0 * np.pi, (batch,))
    profile.gaussian_amp.zero_()
    profile.triangle_amp.zero_()
    profile.ellipse_amp.zero_()
    profile.box_amp.zero_()
    return profile, actual_cpw


def shortwave_corridor_loss(
    model,
    profiles: ProfileBatch,
    cpw: torch.Tensor,
    n: int,
    n_steps: int,
    cfl: float,
    lower_margin: float,
    gain_tolerance: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        u0 = profiles.cell_average(n, 0.0)
        linear = rollout(None, u0, n_steps, cfl)
        amp0, _ = _fundamental_and_residual(u0, profiles.sine_k[:, 0])
        amp_linear, residual_linear = _fundamental_and_residual(linear, profiles.sine_k[:, 0])
        gain_linear = amp_linear / torch.clamp(amp0, min=1.0e-14)

    learned = rollout(model, u0, n_steps, cfl)
    amp_model, residual_model = _fundamental_and_residual(learned, profiles.sine_k[:, 0])
    gain_model = amp_model / torch.clamp(amp0, min=1.0e-14)
    upper = torch.mean(torch.square(torch.relu(gain_model - gain_linear - gain_tolerance)))

    resolved = (cpw >= 7.5).to(torch.float64)
    lower_bound = gain_linear * (1.0 - lower_margin)
    lower_values = torch.square(torch.relu(lower_bound - gain_model)) * resolved
    lower = torch.sum(lower_values) / torch.clamp(torch.sum(resolved), min=1.0)

    scale_energy = torch.clamp(torch.square(amp0), min=1.0e-20)
    harmonic_model = residual_model / scale_energy
    harmonic_linear = residual_linear / scale_energy
    harmonic = torch.mean(torch.relu(harmonic_model - harmonic_linear - 1.0e-6))
    loss = upper + lower + 0.1 * harmonic
    return loss, {
        "shortwave": float(loss.detach()),
        "short_upper": float(upper.detach()),
        "short_lower": float(lower.detach()),
        "short_gain_model": float(torch.mean(gain_model).detach()),
        "short_gain_linear": float(torch.mean(gain_linear).detach()),
        "short_harmonic": float(torch.mean(harmonic_model).detach()),
    }


@dataclass
class JumpWavePacketBatch:
    base: torch.Tensor
    box_amp: torch.Tensor
    box_left: torch.Tensor
    box_width: torch.Tensor
    packet_amp: torch.Tensor
    packet_center: torch.Tensor
    packet_sigma: torch.Tensor
    packet_k: torch.Tensor
    packet_phase: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.base.device

    def _box_average(self, n: int, t: float) -> torch.Tensor:
        batch = self.base.shape[0]
        dx = 1.0 / n
        tmod = float(t) % 1.0
        edges = torch.arange(n + 1, device=self.device, dtype=torch.float64) / n - tmod
        lo, hi = edges[:-1].reshape(1, n), edges[1:].reshape(1, n)
        left = self.box_left
        right = left + self.box_width
        overlap = torch.zeros((batch, n), device=self.device)
        for image in (-2.0, -1.0, 0.0, 1.0, 2.0):
            overlap = overlap + torch.clamp(
                torch.minimum(hi, right + image) - torch.maximum(lo, left + image), min=0.0
            )
        return self.box_amp * overlap / dx

    def cell_average(self, n: int, t: float) -> torch.Tensor:
        nodes = torch.as_tensor(_GL15_X, device=self.device)
        weights = torch.as_tensor(_GL15_W, device=self.device)
        centers = (torch.arange(n, device=self.device, dtype=torch.float64) + 0.5) / n
        x = centers.reshape(1, n, 1) + 0.5 / n * nodes.reshape(1, 1, -1) - t
        distance = torch.remainder(x - self.packet_center.reshape(-1, 1, 1) + 0.5, 1.0) - 0.5
        packet = self.packet_amp.reshape(-1, 1, 1) * torch.exp(
            -0.5 * torch.square(distance / self.packet_sigma.reshape(-1, 1, 1))
        )
        packet = packet * torch.sin(
            2.0 * np.pi * self.packet_k.reshape(-1, 1, 1) * x
            + self.packet_phase.reshape(-1, 1, 1)
        )
        smooth = self.base + 0.5 * torch.sum(packet * weights.reshape(1, 1, -1), dim=-1)
        return smooth + self._box_average(n, t)


def make_jump_wavepackets(
    batch: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> JumpWavePacketBatch:
    base = _uniform(gen, device, -2.0, 4.0, (batch, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    box_amp = torch.pow(10.0, _uniform(gen, device, -2.0, 0.0, (batch, 1))) * scale
    box_amp = box_amp * torch.where(torch.rand((batch, 1), device=device, generator=gen) < 0.5, -1.0, 1.0)
    box_left = _uniform(gen, device, 0.1, 0.55, (batch, 1))
    box_width = _uniform(gen, device, 0.15, 0.35, (batch, 1))
    packet_center = torch.remainder(box_left - _uniform(gen, device, 3.0 / n, 10.0 / n, (batch, 1)), 1.0)
    packet_sigma = _uniform(gen, device, 2.0 / n, 5.0 / n, (batch, 1))
    cpw = torch.as_tensor((4.0, 6.0, 8.0, 12.0), device=device)
    pick = torch.randint(0, len(cpw), (batch,), device=device, generator=gen)
    packet_k = torch.clamp(torch.round(float(n) / cpw[pick]), min=1.0).reshape(-1, 1)
    packet_amp = _uniform(gen, device, 0.01, 0.15, (batch, 1)) * torch.abs(box_amp)
    packet_phase = _uniform(gen, device, 0.0, 2.0 * np.pi, (batch, 1))
    return JumpWavePacketBatch(
        base, box_amp, box_left, box_width, packet_amp,
        packet_center, packet_sigma, packet_k, packet_phase,
    )


def jump_shedding_loss(
    model,
    batch: JumpWavePacketBatch,
    n: int,
    n_steps: int,
    cfl: float,
    flat_tolerance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        u0 = batch.cell_average(n, 0.0)
        exact = batch.cell_average(n, n_steps * cfl / n)
        linear = rollout(None, u0, n_steps, cfl)
        qscale = torch.clamp(torch.max(torch.abs(u0), dim=1).values, min=1.0)
        rng = torch.clamp(
            torch.max(u0, dim=1).values - torch.min(u0, dim=1).values,
            min=1.0e-6 * qscale,
        )
        mask = flat_region_mask(exact, rng, flat_tolerance).to(torch.float64)
        count = torch.clamp(torch.sum(mask, dim=1), min=1.0)

    learned = rollout(model, u0, n_steps, cfl)
    d2_exact = second_difference(exact)
    error_model = torch.square(second_difference(learned) - d2_exact)
    error_linear = torch.square(second_difference(linear) - d2_exact)
    energy_model = torch.sum(mask * error_model, dim=1) / (count * torch.square(rng))
    energy_linear = torch.sum(mask * error_linear, dim=1) / (count * torch.square(rng))
    flat_penalty = torch.mean(torch.relu(energy_model - energy_linear - 1.0e-8))

    tv_model = torch.sum(torch.abs(learned - torch.roll(learned, 1, dims=1)), dim=1)
    tv_linear = torch.sum(torch.abs(linear - torch.roll(linear, 1, dims=1)), dim=1)
    tv_penalty = torch.mean(torch.relu(tv_model - tv_linear) / rng)
    loss = flat_penalty + 0.1 * tv_penalty
    return loss, {
        "jump_shedding": float(loss.detach()),
        "jump_flat_pen": float(flat_penalty.detach()),
        "jump_tv_pen": float(tv_penalty.detach()),
    }
