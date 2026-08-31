#!/usr/bin/env python3
"""Random analytic profiles evaluated as finite-volume cell averages."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

PROFILE_NAMES = (
    "smooth",
    "gaussian",
    "square",
    "triangle",
    "ellipse",
    "multi",
    "composite",
)
PROFILE_DEFAULT_PROBS = (0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10)
_GL15_X, _GL15_W = np.polynomial.legendre.leggauss(15)


def _uniform(
    gen: torch.Generator,
    device: torch.device,
    lo: float,
    hi: float,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(
        shape, device=device, generator=gen, dtype=torch.float64
    )


def _signed_log_amp(
    gen: torch.Generator,
    device: torch.device,
    scale: torch.Tensor,
    shape: tuple[int, ...],
    lo: float = -3.0,
    hi: float = 0.0,
) -> torch.Tensor:
    magnitude = torch.pow(10.0, _uniform(gen, device, lo, hi, shape)) * scale
    sign = torch.where(
        torch.rand(shape, device=device, generator=gen) < 0.5,
        -torch.ones((), device=device),
        torch.ones((), device=device),
    )
    return magnitude * sign


@dataclass
class ProfileBatch:
    """A batch of periodic profiles on [0,1)."""

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

    @staticmethod
    def _view(value: torch.Tensor, ndim: int) -> torch.Tensor:
        return value.reshape(value.shape + (1,) * ndim)

    def smooth_point_value(
        self, x: torch.Tensor, time: float | torch.Tensor
    ) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float64)
        ndim = x.ndim
        y = x.reshape((1,) + x.shape) - torch.as_tensor(
            time, device=self.device, dtype=torch.float64
        )
        output = self.base.reshape((self.batch,) + (1,) * ndim).expand(
            (self.batch,) + x.shape
        )

        yk = y.unsqueeze(1)
        output = output + torch.sum(
            self._view(self.sine_amp, ndim)
            * torch.sin(
                2.0
                * np.pi
                * self._view(self.sine_k, ndim)
                * yk
                + self._view(self.sine_phase, ndim)
            ),
            dim=1,
        )

        def periodic_distance(center: torch.Tensor) -> torch.Tensor:
            value = self._view(center, ndim)
            return torch.remainder(yk - value + 0.5, 1.0) - 0.5

        distance = periodic_distance(self.gaussian_center)
        output = output + torch.sum(
            self._view(self.gaussian_amp, ndim)
            * torch.exp(
                -0.5
                * torch.square(
                    distance / self._view(self.gaussian_sigma, ndim)
                )
            ),
            dim=1,
        )

        distance = periodic_distance(self.triangle_center)
        output = output + torch.sum(
            self._view(self.triangle_amp, ndim)
            * torch.clamp(
                1.0
                - torch.abs(distance)
                / self._view(self.triangle_halfwidth, ndim),
                min=0.0,
            ),
            dim=1,
        )

        distance = periodic_distance(self.ellipse_center)
        ellipse = torch.sqrt(
            torch.clamp(
                1.0
                - torch.square(
                    distance / self._view(self.ellipse_halfwidth, ndim)
                ),
                min=0.0,
            )
        )
        output = output + torch.sum(
            self._view(self.ellipse_amp, ndim) * ellipse, dim=1
        )
        return output

    def point_value(
        self, x: torch.Tensor, time: float | torch.Tensor
    ) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device, dtype=torch.float64)
        ndim = x.ndim
        output = self.smooth_point_value(x, time)
        y = x.reshape((1,) + x.shape) - torch.as_tensor(
            time, device=self.device, dtype=torch.float64
        )
        y = y.unsqueeze(1)
        inside = (
            torch.remainder(y - self._view(self.box_left, ndim), 1.0)
            < self._view(self.box_width, ndim)
        )
        return output + torch.sum(
            self._view(self.box_amp, ndim) * inside.to(torch.float64), dim=1
        )

    def _box_cell_average(
        self, n: int, time: float | torch.Tensor
    ) -> torch.Tensor:
        dx = 1.0 / float(n)
        time_mod = torch.remainder(
            torch.as_tensor(time, device=self.device, dtype=torch.float64), 1.0
        )
        edges = (
            torch.arange(n + 1, device=self.device, dtype=torch.float64)
            / float(n)
            - time_mod
        )
        lo = edges[:-1].reshape(1, 1, n)
        hi = edges[1:].reshape(1, 1, n)
        left = self.box_left.unsqueeze(-1)
        right = left + self.box_width.unsqueeze(-1)
        overlap = torch.zeros(
            (self.batch, self.box_amp.shape[1], n),
            device=self.device,
            dtype=torch.float64,
        )
        for image in (-2.0, -1.0, 0.0, 1.0, 2.0):
            overlap = overlap + torch.clamp(
                torch.minimum(hi, right + image)
                - torch.maximum(lo, left + image),
                min=0.0,
            )
        return torch.sum(self.box_amp.unsqueeze(-1) * overlap / dx, dim=1)

    def _box_cell_average_times(
        self, n: int, times: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized exact box averages with shape ``(batch,time,cell)``."""
        dx = 1.0 / float(n)
        times = torch.as_tensor(
            times, device=self.device, dtype=torch.float64
        ).reshape(-1)
        time_mod = torch.remainder(times, 1.0)
        edges = (
            torch.arange(n + 1, device=self.device, dtype=torch.float64)
            / float(n)
        )
        shifted = edges.reshape(1, n + 1) - time_mod.reshape(-1, 1)
        lo = shifted[:, :-1].reshape(1, 1, times.numel(), n)
        hi = shifted[:, 1:].reshape(1, 1, times.numel(), n)
        left = self.box_left.unsqueeze(-1).unsqueeze(-1)
        right = left + self.box_width.unsqueeze(-1).unsqueeze(-1)
        overlap = torch.zeros(
            (
                self.batch,
                self.box_amp.shape[1],
                times.numel(),
                n,
            ),
            device=self.device,
            dtype=torch.float64,
        )
        for image in (-2.0, -1.0, 0.0, 1.0, 2.0):
            overlap = overlap + torch.clamp(
                torch.minimum(hi, right + image)
                - torch.maximum(lo, left + image),
                min=0.0,
            )
        return torch.sum(
            self.box_amp.unsqueeze(-1).unsqueeze(-1) * overlap / dx,
            dim=1,
        )

    def cell_average(
        self, n: int, time: float | torch.Tensor
    ) -> torch.Tensor:
        """Use GL15 for continuous pieces and exact overlap for top hats."""
        dx = 1.0 / float(n)
        nodes = torch.as_tensor(_GL15_X, device=self.device)
        weights = torch.as_tensor(_GL15_W, device=self.device)
        centers = (
            torch.arange(n, device=self.device, dtype=torch.float64) + 0.5
        ) / float(n)
        points = centers.reshape(n, 1) + 0.5 * dx * nodes.reshape(1, -1)
        smooth = self.smooth_point_value(points, time)
        smooth_average = 0.5 * torch.sum(
            smooth * weights.reshape(1, 1, -1), dim=-1
        )
        return smooth_average + self._box_cell_average(n, time)

    def cell_average_times(
        self, n: int, times: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized finite-volume averages with shape ``(batch,time,cell)``."""
        times = torch.as_tensor(
            times, device=self.device, dtype=torch.float64
        ).reshape(-1)
        dx = 1.0 / float(n)
        nodes = torch.as_tensor(
            _GL15_X, device=self.device, dtype=torch.float64
        )
        weights = torch.as_tensor(
            _GL15_W, device=self.device, dtype=torch.float64
        )
        centers = (
            torch.arange(n, device=self.device, dtype=torch.float64) + 0.5
        ) / float(n)
        points = (
            centers.reshape(1, n, 1)
            + 0.5 * dx * nodes.reshape(1, 1, -1)
        ).expand(times.numel(), n, nodes.numel())
        smooth = self.smooth_point_value(
            points, times.reshape(-1, 1, 1)
        )
        smooth_average = 0.5 * torch.sum(
            smooth * weights.reshape(1, 1, 1, -1), dim=-1
        )
        return smooth_average + self._box_cell_average_times(n, times)

    def point_targets(
        self,
        n: int,
        time: float | torch.Tensor,
        offsets: tuple[float, ...],
    ) -> torch.Tensor:
        centers = (
            torch.arange(n, device=self.device, dtype=torch.float64) + 0.5
        ) / float(n)
        offset = torch.as_tensor(offsets, device=self.device)
        points = centers.reshape(1, n) + offset.reshape(-1, 1) / float(n)
        return self.point_value(points, time)

    def point_targets_times(
        self,
        n: int,
        times: torch.Tensor,
        offsets: tuple[float, ...],
    ) -> torch.Tensor:
        """Vectorized point targets with shape ``(batch,time,head,cell)``."""
        times = torch.as_tensor(
            times, device=self.device, dtype=torch.float64
        ).reshape(-1)
        centers = (
            torch.arange(n, device=self.device, dtype=torch.float64) + 0.5
        ) / float(n)
        offset = torch.as_tensor(
            offsets, device=self.device, dtype=torch.float64
        )
        points = (
            centers.reshape(1, 1, n)
            + offset.reshape(1, -1, 1) / float(n)
        ).expand(times.numel(), offset.numel(), n)
        return self.point_value(points, times.reshape(-1, 1, 1))


def make_profiles(
    batch: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
    kind: str | None = None,
    probs: tuple[float, ...] | None = None,
) -> ProfileBatch:
    """Generate randomized training profiles; the fixed GSTE test is absent."""
    if kind is not None and kind not in PROFILE_NAMES:
        raise ValueError(f"unknown profile family {kind!r}")
    if kind is None:
        probabilities = torch.as_tensor(
            PROFILE_DEFAULT_PROBS if probs is None else probs,
            device=device,
            dtype=torch.float64,
        )
        family = torch.multinomial(
            probabilities.expand(batch, -1), 1, generator=gen
        ).reshape(-1)
    else:
        family = torch.full(
            (batch,),
            PROFILE_NAMES.index(kind),
            device=device,
            dtype=torch.long,
        )

    base = _uniform(gen, device, -2.0, 4.0, (batch, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    masks = [
        (family == index).to(torch.float64).reshape(batch, 1)
        for index in range(len(PROFILE_NAMES))
    ]

    sine_amp = _signed_log_amp(gen, device, scale, (batch, 3)) / 3.0
    sine_k = torch.randint(
        1,
        max(2, n // 10) + 1,
        (batch, 3),
        device=device,
        generator=gen,
    ).to(torch.float64)
    sine_phase = _uniform(gen, device, 0.0, 2.0 * np.pi, (batch, 3))
    sine_amp = sine_amp * (masks[0] + 0.35 * masks[6])

    gaussian_amp = _signed_log_amp(gen, device, scale, (batch, 2))
    gaussian_center = _uniform(gen, device, 0.0, 1.0, (batch, 2))
    gaussian_sigma = _uniform(gen, device, 1.5 / n, 8.0 / n, (batch, 2))
    gaussian_amp = gaussian_amp * (masks[1] + 0.5 * masks[6])

    triangle_amp = _signed_log_amp(gen, device, scale, (batch, 2))
    triangle_center = _uniform(gen, device, 0.0, 1.0, (batch, 2))
    triangle_halfwidth = _uniform(
        gen, device, 3.0 / n, min(18.0 / n, 0.24), (batch, 2)
    )
    triangle_amp = triangle_amp * (masks[3] + 0.5 * masks[6])

    ellipse_amp = _signed_log_amp(gen, device, scale, (batch, 1))
    ellipse_center = _uniform(gen, device, 0.0, 1.0, (batch, 1))
    ellipse_halfwidth = _uniform(
        gen, device, 4.0 / n, min(20.0 / n, 0.24), (batch, 1)
    )
    ellipse_amp = ellipse_amp * (masks[4] + masks[6])

    box_amp = _signed_log_amp(gen, device, scale, (batch, 3))
    box_left = _uniform(gen, device, 0.0, 1.0, (batch, 3))
    box_width = _uniform(
        gen, device, 3.0 / n, min(32.0 / n, 0.40), (batch, 3)
    )
    active = (
        torch.rand((batch, 3), device=device, generator=gen)
        < torch.tensor((1.0, 0.85, 0.55), device=device)
    ).to(torch.float64)
    box_mask = masks[2] * torch.tensor(
        (1.0, 0.0, 0.0), device=device
    ).reshape(1, 3)
    box_mask = box_mask + masks[5] * active
    box_mask = box_mask + masks[6] * torch.tensor(
        (1.0, 0.0, 0.0), device=device
    ).reshape(1, 3)
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
