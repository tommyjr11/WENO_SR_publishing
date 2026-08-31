#!/usr/bin/env python3
"""Analytic periodic profiles evaluated as true finite-volume cell averages."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

PROFILE_NAMES = ("smooth", "gaussian", "square", "triangle", "ellipse", "multi", "composite")
PROFILE_DEFAULT_PROBS = (0.20, 0.10, 0.20, 0.10, 0.10, 0.15, 0.15)

_GL15_X_NP, _GL15_W_NP = np.polynomial.legendre.leggauss(15)


def _uniform(
    gen: torch.Generator,
    device: torch.device,
    lo: float,
    hi: float,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen, dtype=torch.float64)


def _signed_log_amp(
    gen: torch.Generator,
    device: torch.device,
    scale: torch.Tensor,
    shape: tuple[int, ...],
    lo: float = -3.0,
    hi: float = 0.0,
) -> torch.Tensor:
    amp = torch.pow(10.0, _uniform(gen, device, lo, hi, shape)) * scale
    sign = torch.where(
        torch.rand(shape, device=device, generator=gen) < 0.5,
        -torch.ones((), device=device),
        torch.ones((), device=device),
    )
    return amp * sign


@dataclass
class ProfileBatch:
    """A batch of periodic analytic profiles on [0,1)."""

    base: torch.Tensor
    sine_amp: torch.Tensor
    sine_k: torch.Tensor
    sine_phase: torch.Tensor
    gaussian_amp: torch.Tensor
    gaussian_center: torch.Tensor
    gaussian_sigma: torch.Tensor
    triangle_amp: torch.Tensor
    triangle_center: torch.Tensor
    triangle_halfwidth: torch.Tensor
    ellipse_amp: torch.Tensor
    ellipse_center: torch.Tensor
    ellipse_halfwidth: torch.Tensor
    box_amp: torch.Tensor
    box_left: torch.Tensor
    box_width: torch.Tensor
    family_id: torch.Tensor

    @property
    def batch(self) -> int:
        return int(self.base.shape[0])

    @property
    def device(self) -> torch.device:
        return self.base.device

    def _component_view(self, value: torch.Tensor, ndim: int) -> torch.Tensor:
        return value.reshape(value.shape + (1,) * ndim)

    def smooth_point_value(self, x: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Evaluate every continuous component; x may have any trailing shape."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float64)
        ndim = x.ndim
        y = x.reshape((1,) + x.shape) - torch.as_tensor(t, device=self.device, dtype=torch.float64)
        out = self.base.reshape((self.batch,) + (1,) * ndim).expand((self.batch,) + x.shape)

        yk = y.unsqueeze(1)
        amp = self._component_view(self.sine_amp, ndim)
        kval = self._component_view(self.sine_k, ndim)
        phase = self._component_view(self.sine_phase, ndim)
        out = out + torch.sum(amp * torch.sin(2.0 * np.pi * kval * yk + phase), dim=1)

        def periodic_distance(center: torch.Tensor) -> torch.Tensor:
            c = self._component_view(center, ndim)
            return torch.remainder(yk - c + 0.5, 1.0) - 0.5

        dg = periodic_distance(self.gaussian_center)
        sg = self._component_view(self.gaussian_sigma, ndim)
        ag = self._component_view(self.gaussian_amp, ndim)
        out = out + torch.sum(ag * torch.exp(-0.5 * torch.square(dg / sg)), dim=1)

        dt = periodic_distance(self.triangle_center)
        wt = self._component_view(self.triangle_halfwidth, ndim)
        at = self._component_view(self.triangle_amp, ndim)
        out = out + torch.sum(at * torch.clamp(1.0 - torch.abs(dt) / wt, min=0.0), dim=1)

        de = periodic_distance(self.ellipse_center)
        we = self._component_view(self.ellipse_halfwidth, ndim)
        ae = self._component_view(self.ellipse_amp, ndim)
        ellipse = torch.sqrt(torch.clamp(1.0 - torch.square(de / we), min=0.0))
        out = out + torch.sum(ae * ellipse, dim=1)
        return out

    def point_value(self, x: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Evaluate the periodic profile point value at time t for u_t+u_x=0."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float64)
        ndim = x.ndim
        out = self.smooth_point_value(x, t)
        y = x.reshape((1,) + x.shape) - torch.as_tensor(t, device=self.device, dtype=torch.float64)
        y = y.unsqueeze(1)
        left = self._component_view(self.box_left, ndim)
        width = self._component_view(self.box_width, ndim)
        amp = self._component_view(self.box_amp, ndim)
        inside = torch.remainder(y - left, 1.0) < width
        return out + torch.sum(amp * inside.to(torch.float64), dim=1)

    def _box_cell_average(self, n: int, t: float | torch.Tensor) -> torch.Tensor:
        """Exact overlap of periodic top hats with every finite-volume cell."""
        dx = 1.0 / float(n)
        tmod = torch.remainder(torch.as_tensor(t, device=self.device, dtype=torch.float64), 1.0)
        edges = torch.arange(n + 1, device=self.device, dtype=torch.float64) / float(n) - tmod
        lo = edges[:-1].reshape(1, 1, n)
        hi = edges[1:].reshape(1, 1, n)
        left = self.box_left.unsqueeze(-1)
        right = left + self.box_width.unsqueeze(-1)
        overlap = torch.zeros((self.batch, self.box_amp.shape[1], n), device=self.device)
        for image in (-2.0, -1.0, 0.0, 1.0, 2.0):
            li = left + image
            ri = right + image
            overlap = overlap + torch.clamp(torch.minimum(hi, ri) - torch.maximum(lo, li), min=0.0)
        return torch.sum(self.box_amp.unsqueeze(-1) * overlap / dx, dim=1)

    def cell_average(self, n: int, t: float | torch.Tensor) -> torch.Tensor:
        """Return exact/15-point-Gauss finite-volume averages, never center samples."""
        dx = 1.0 / float(n)
        nodes = torch.as_tensor(_GL15_X_NP, device=self.device, dtype=torch.float64)
        weights = torch.as_tensor(_GL15_W_NP, device=self.device, dtype=torch.float64)
        centers = (torch.arange(n, device=self.device, dtype=torch.float64) + 0.5) / float(n)
        points = centers.reshape(n, 1) + 0.5 * dx * nodes.reshape(1, -1)
        smooth = self.smooth_point_value(points, t)
        smooth_avg = 0.5 * torch.sum(smooth * weights.reshape(1, 1, -1), dim=-1)
        return smooth_avg + self._box_cell_average(n, t)

    def point_targets(self, n: int, t: float | torch.Tensor, offsets: tuple[float, ...]) -> torch.Tensor:
        """Point targets at offsets measured in cell widths from each cell center."""
        centers = (torch.arange(n, device=self.device, dtype=torch.float64) + 0.5) / float(n)
        off = torch.as_tensor(offsets, device=self.device, dtype=torch.float64)
        points = centers.reshape(1, n) + off.reshape(-1, 1) / float(n)
        return self.point_value(points, t)


def make_profiles(
    batch: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
    kind: str | None = None,
    probs: tuple[float, ...] | None = None,
) -> ProfileBatch:
    """Generate randomized resolved profiles; under-resolved waves are excluded."""
    if kind is not None and kind not in PROFILE_NAMES:
        raise ValueError(f"unknown profile kind {kind!r}")
    if kind is None:
        p = torch.as_tensor(PROFILE_DEFAULT_PROBS if probs is None else probs, device=device)
        family = torch.multinomial(p.expand(batch, -1), 1, generator=gen).reshape(-1)
    else:
        family = torch.full((batch,), PROFILE_NAMES.index(kind), device=device, dtype=torch.long)

    base = _uniform(gen, device, -2.0, 4.0, (batch, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    masks = [(family == i).to(torch.float64).reshape(batch, 1) for i in range(len(PROFILE_NAMES))]

    sine_amp = _signed_log_amp(gen, device, scale, (batch, 3)) / 3.0
    sine_k = torch.randint(1, max(2, n // 10) + 1, (batch, 3), device=device, generator=gen).to(torch.float64)
    sine_phase = _uniform(gen, device, 0.0, 2.0 * np.pi, (batch, 3))
    sine_mask = masks[0] + 0.35 * masks[6]
    sine_amp = sine_amp * sine_mask

    gaussian_amp = _signed_log_amp(gen, device, scale, (batch, 2))
    gaussian_center = _uniform(gen, device, 0.0, 1.0, (batch, 2))
    gaussian_sigma = _uniform(gen, device, 1.5 / n, 8.0 / n, (batch, 2))
    gaussian_mask = masks[1] + 0.5 * masks[6]
    gaussian_amp = gaussian_amp * gaussian_mask

    triangle_amp = _signed_log_amp(gen, device, scale, (batch, 2))
    triangle_center = _uniform(gen, device, 0.0, 1.0, (batch, 2))
    triangle_halfwidth = _uniform(gen, device, 3.0 / n, min(18.0 / n, 0.24), (batch, 2))
    triangle_mask = masks[3] + 0.5 * masks[6]
    triangle_amp = triangle_amp * triangle_mask

    ellipse_amp = _signed_log_amp(gen, device, scale, (batch, 1))
    ellipse_center = _uniform(gen, device, 0.0, 1.0, (batch, 1))
    ellipse_halfwidth = _uniform(gen, device, 4.0 / n, min(20.0 / n, 0.24), (batch, 1))
    ellipse_amp = ellipse_amp * (masks[4] + masks[6])

    box_amp = _signed_log_amp(gen, device, scale, (batch, 3))
    box_left = _uniform(gen, device, 0.0, 1.0, (batch, 3))
    box_width = _uniform(gen, device, 3.0 / n, min(32.0 / n, 0.40), (batch, 3))
    multi_active = (
        torch.rand((batch, 3), device=device, generator=gen) < torch.tensor((1.0, 0.85, 0.55), device=device)
    ).to(torch.float64)
    box_mask = masks[2] * torch.tensor((1.0, 0.0, 0.0), device=device).reshape(1, 3)
    box_mask = box_mask + masks[5] * multi_active
    box_mask = box_mask + masks[6] * torch.tensor((1.0, 0.0, 0.0), device=device).reshape(1, 3)
    box_amp = box_amp * box_mask

    return ProfileBatch(
        base=base,
        sine_amp=sine_amp,
        sine_k=sine_k,
        sine_phase=sine_phase,
        gaussian_amp=gaussian_amp,
        gaussian_center=gaussian_center,
        gaussian_sigma=gaussian_sigma,
        triangle_amp=triangle_amp,
        triangle_center=triangle_center,
        triangle_halfwidth=triangle_halfwidth,
        ellipse_amp=ellipse_amp,
        ellipse_center=ellipse_center,
        ellipse_halfwidth=ellipse_halfwidth,
        box_amp=box_amp,
        box_left=box_left,
        box_width=box_width,
        family_id=family,
    )


def make_shortwave_profiles(
    batch: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> ProfileBatch:
    """Generate pure periodic 4/6/8-cell waves for the stability audit loss."""
    profile = make_profiles(batch, n, device, gen, kind="smooth")
    profile.sine_amp.zero_()
    cpw_options = torch.as_tensor((4.0, 6.0, 8.0), device=device)
    pick = torch.randint(0, 3, (batch,), device=device, generator=gen)
    k = torch.clamp(torch.round(float(n) / cpw_options[pick]), min=1.0)
    scale = torch.clamp(torch.abs(profile.base[:, 0]), min=1.0)
    amp = torch.pow(10.0, _uniform(gen, device, -5.0, np.log10(0.3), (batch,))) * scale
    sign = torch.where(torch.rand(batch, device=device, generator=gen) < 0.5, -1.0, 1.0)
    profile.sine_amp[:, 0] = amp * sign
    profile.sine_k[:, 0] = k
    profile.sine_phase[:, 0] = _uniform(gen, device, 0.0, 2.0 * np.pi, (batch,))
    profile.gaussian_amp.zero_()
    profile.triangle_amp.zero_()
    profile.ellipse_amp.zero_()
    profile.box_amp.zero_()
    return profile
