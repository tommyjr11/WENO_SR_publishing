#!/usr/bin/env python3
"""Differentiable scalar positive-speed WENO5/SSPRK3 advection rollout."""
from __future__ import annotations

import numpy as np
import torch

from teacherfree_lab_weno5 import weno5_core as W

torch.set_default_dtype(torch.float64)

IC_KINDS = ("smooth", "smeared", "sharp", "ramp", "multi")
IC_DEFAULT_PROBS = (0.20, 0.15, 0.35, 0.15, 0.15)


def reconstruction_omega(model, q: torch.Tensor, lr: int) -> torch.Tensor:
    d = W.optimal_d(lr, q.device).reshape(1, 3).expand(q.shape[0], 3)
    if model is None:
        om = d
    elif isinstance(model, str) and model == "classical":
        om = W.classical_omega(q, lr)
    else:
        om = W.omega_from_ratio(model(W.weno5_features(q)), lr)
    return torch.where(W.plateau_mask(q).reshape(-1, 1), d, om)


def reconstruct_iplus(model, u: torch.Tensor) -> torch.Tensor:
    """Cell averages u(B,N) -> left state at every i+1/2 for positive speed."""
    bsz, n = u.shape
    up = torch.cat([u[:, -2:], u, u[:, :2]], dim=1)
    q = up.unfold(1, 5, 1).reshape(bsz * n, 5)
    om = reconstruction_omega(model, q, 1)
    val = torch.sum(om * W.candidate_values(q, 1), dim=1)
    return val.reshape(bsz, n)


def rhs(model, u: torch.Tensor, dxinv: float) -> torch.Tensor:
    flux = reconstruct_iplus(model, u)
    return -(flux - torch.roll(flux, 1, dims=1)) * dxinv


def ssprk3(model, u: torch.Tensor, dt: float, dxinv: float) -> torch.Tensor:
    u1 = u + dt * rhs(model, u, dxinv)
    u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(model, u1, dxinv))
    return u / 3.0 + (2.0 / 3.0) * (u2 + dt * rhs(model, u2, dxinv))


def rollout(model, u0: torch.Tensor, n_steps: int, cfl: float) -> torch.Tensor:
    n = u0.shape[1]
    dx = 1.0 / float(n)
    u = u0
    for _ in range(n_steps):
        u = ssprk3(model, u, cfl * dx, 1.0 / dx)
    return u


def make_ic(
    batch: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
    kind: str | None = None,
    probs: tuple[float, ...] | None = None,
) -> torch.Tensor:
    def rand(lo: float, hi: float, size: tuple[int, int] = (batch, 1)) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(size, device=device, generator=gen)

    x = (torch.arange(n, device=device, dtype=torch.float64) + 0.5).reshape(1, n) / float(n)
    base = rand(-2.0, 4.0)
    scale = torch.clamp(torch.abs(base), min=1.0)

    def rand_amp() -> torch.Tensor:
        amp = torch.pow(10.0, rand(-3.0, 0.0)) * scale
        sign = torch.where(torch.rand((batch, 1), device=device, generator=gen) < 0.5, -1.0, 1.0)
        return amp * sign

    def smooth_ic() -> torch.Tensor:
        out = torch.zeros((batch, n), device=device, dtype=torch.float64)
        for _ in range(3):
            k = torch.randint(1, 5, (batch, 1), device=device, generator=gen).to(torch.float64)
            out = out + rand(-1.0, 1.0) * torch.sin(2.0 * np.pi * k * x + rand(0.0, 2.0 * np.pi))
        return base + torch.pow(10.0, rand(-3.0, 0.0)) * scale * out / 3.0

    def step_pair(w_log_lo: float, w_log_hi: float) -> torch.Tensor:
        amp = rand_amp()
        c1 = rand(0.15, 0.45)
        c2 = c1 + rand(0.2, 0.4)
        width = torch.pow(10.0, rand(w_log_lo, w_log_hi)) / float(n)
        prof = torch.sigmoid((x - c1) / width) - torch.sigmoid((x - c2) / width)
        m = torch.randint(6, max(7, n // 3), (batch, 1), device=device, generator=gen).to(torch.float64)
        ripple = rand(0.02, 0.3) * torch.abs(amp) * torch.sin(2.0 * np.pi * m * x + rand(0.0, 2.0 * np.pi))
        has_r = (torch.rand((batch, 1), device=device, generator=gen) < 0.7).to(torch.float64)
        return base + amp * prof + has_r * ripple

    def ramp_ic() -> torch.Tensor:
        amp = rand_amp()
        c1 = rand(0.2, 0.4)
        width = torch.pow(10.0, rand(np.log10(0.25), np.log10(3.0))) / float(n)
        front = torch.sigmoid((x - c1) / width)
        c2 = rand(0.7, 0.85)
        w_ret = rand(8.0, 14.0) / float(n)
        ret = torch.sigmoid((x - c2) / w_ret)
        m = torch.randint(6, max(7, n // 3), (batch, 1), device=device, generator=gen).to(torch.float64)
        lam_decay = rand(0.03, 0.15)
        behind = torch.sigmoid((x - c1) / (0.5 / float(n)))
        envelope = behind * torch.exp(-torch.clamp(x - c1, min=0.0) / lam_decay)
        ripple = rand(0.02, 0.35) * torch.abs(amp) * torch.sin(2.0 * np.pi * m * x + rand(0.0, 2.0 * np.pi))
        has_r = (torch.rand((batch, 1), device=device, generator=gen) < 0.85).to(torch.float64)
        return base + amp * (front - ret) + has_r * envelope * ripple

    def multi_ic() -> torch.Tensor:
        out = base.expand(batch, n).clone()
        for _ in range(3):
            active = (torch.rand((batch, 1), device=device, generator=gen) < 0.75).to(torch.float64)
            amp = rand_amp()
            c1 = rand(0.05, 0.55)
            c2 = c1 + rand(0.1, 0.35)
            width = torch.pow(10.0, rand(np.log10(0.05), np.log10(1.5))) / float(n)
            out = out + active * amp * (torch.sigmoid((x - c1) / width) - torch.sigmoid((x - c2) / width))
        return out

    makers = {
        "smooth": smooth_ic,
        "smeared": lambda: step_pair(np.log10(0.25), np.log10(3.0)),
        "sharp": lambda: step_pair(np.log10(0.05), np.log10(0.15)),
        "ramp": ramp_ic,
        "multi": multi_ic,
    }
    if kind is not None:
        return makers[kind]()

    fams = [makers[name]() for name in IC_KINDS]
    pr = torch.as_tensor(IC_DEFAULT_PROBS if probs is None else probs, device=device, dtype=torch.float64)
    sel = torch.multinomial(pr.expand(batch, len(IC_KINDS)), 1, generator=gen)
    out = fams[0]
    for i in range(1, len(IC_KINDS)):
        out = torch.where(sel == i, fams[i], out)
    return out


def rollout_loss(
    model,
    u0: torch.Tensor,
    n_steps: int,
    cfl: float,
    tv_lambda: float = 0.0,
    err_power: float = 2.0,
    tv_floor: float = 0.0,
    tv_bg_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    shift = cfl * n_steps
    assert abs(shift - round(shift)) < 1.0e-12, "n_steps*cfl must be an integer shift"
    u_t = rollout(model, u0, n_steps, cfl)
    exact = torch.roll(u0, int(round(shift)), dims=1)
    rng = torch.clamp(torch.max(u0, dim=1).values - torch.min(u0, dim=1).values, min=1.0e-13)
    e = (u_t - exact) / rng.reshape(-1, 1)
    if err_power == 2.0:
        err = torch.mean(e * e, dim=1)
    else:
        err = torch.pow(torch.mean(torch.abs(e) ** err_power, dim=1) + 1.0e-300, 2.0 / err_power)
    loss = err.mean()
    stats = {"evolve_err": float(loss.detach())}
    if tv_lambda > 0.0 or tv_bg_lambda > 0.0:
        tv = lambda u: torch.sum(torch.abs(u - torch.roll(u, 1, dims=1)), dim=1)
        excess = torch.relu(tv(u_t) - tv(u0)) / rng
        pen = torch.relu(excess.mean() - tv_floor)
        loss = loss + tv_lambda * pen + tv_bg_lambda * excess.mean()
        stats["tv_excess"] = float(excess.mean().detach())
        stats["tv_pen"] = float(pen.detach())
    return loss, stats

