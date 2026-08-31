#!/usr/bin/env python3
"""Standalone GSTE initial data, cell averages, and SSPRK3 integration."""
from __future__ import annotations

import math

import numpy as np
import torch

X_MIN = -1.0
X_MAX = 1.0
LENGTH = X_MAX - X_MIN


def wrap(x: np.ndarray) -> np.ndarray:
    return X_MIN + np.mod(x - X_MIN, LENGTH)


def gste_point(x: np.ndarray) -> np.ndarray:
    x = wrap(np.asarray(x, dtype=np.float64))
    delta = 0.005
    beta = math.log(2.0) / (36.0 * delta * delta)
    gaussian_center = -0.7
    ellipse_center = 0.5
    alpha = 10.0

    def gaussian(center: float) -> np.ndarray:
        return np.exp(-beta * np.square(x - center))

    def ellipse(center: float) -> np.ndarray:
        return np.sqrt(
            np.maximum(1.0 - np.square(alpha * (x - center)), 0.0)
        )

    output = np.zeros_like(x)
    mask = (-0.8 < x) & (x < -0.6)
    output[mask] = (
        gaussian(gaussian_center - delta)[mask]
        + 4.0 * gaussian(gaussian_center)[mask]
        + gaussian(gaussian_center + delta)[mask]
    ) / 6.0
    mask = (-0.4 < x) & (x < -0.2)
    output[mask] = 1.0
    mask = (0.0 < x) & (x < 0.2)
    output[mask] = 1.0 - np.abs(10.0 * (x[mask] - 0.1))
    mask = (0.4 < x) & (x < 0.6)
    output[mask] = (
        ellipse(ellipse_center - delta)[mask]
        + 4.0 * ellipse(ellipse_center)[mask]
        + ellipse(ellipse_center + delta)[mask]
    ) / 6.0
    return output


def cell_averages(
    nx: int, time: float, quadrature: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    dx = LENGTH / float(nx)
    centers = X_MIN + (np.arange(nx) + 0.5) * dx
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    points = centers[:, None] + 0.5 * dx * nodes[None, :] - time
    averages = 0.5 * np.sum(
        weights[None, :] * gste_point(points), axis=1
    )
    return centers, averages


def integrate_ssprk3(
    model,
    stepper,
    initial: np.ndarray,
    cfl_limit: float,
    t_end: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    nx = int(initial.size)
    dx = LENGTH / float(nx)
    steps = int(math.ceil(t_end / (cfl_limit * dx)))
    dt = t_end / float(steps)
    actual_cfl = dt / dx
    state = torch.as_tensor(
        initial, device=device, dtype=torch.float64
    ).reshape(1, nx)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for _ in range(steps):
            state = stepper(model, state, dt, 1.0 / dx)
            if not bool(torch.all(torch.isfinite(state))):
                break
    result = state[0].detach().cpu().numpy()
    return result, {
        "steps": steps,
        "dt": dt,
        "cfl": actual_cfl,
        "t": t_end,
        "finite": bool(np.all(np.isfinite(result))),
    }
