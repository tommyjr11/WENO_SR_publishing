#!/usr/bin/env python3
"""Lightweight, deterministic GSTE monitor for long WENO5 training runs."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from teacherfree_lab_weno5 import apost_advect as adv5


X_MIN = -1.0
X_MAX = 1.0
LENGTH = X_MAX - X_MIN


def _wrap(x: np.ndarray) -> np.ndarray:
    return X_MIN + np.mod(x - X_MIN, LENGTH)


def _gste_point(x: np.ndarray) -> np.ndarray:
    x = _wrap(np.asarray(x, dtype=np.float64))
    delta = 0.005
    beta = math.log(2.0) / (36.0 * delta * delta)
    z = -0.7
    a = 0.5
    alpha = 10.0

    def gaussian(center: float) -> np.ndarray:
        return np.exp(-beta * np.square(x - center))

    def ellipse(center: float) -> np.ndarray:
        return np.sqrt(np.maximum(1.0 - np.square(alpha * (x - center)), 0.0))

    out = np.zeros_like(x)
    mask = (-0.8 < x) & (x < -0.6)
    out[mask] = (
        gaussian(z - delta)[mask] + 4.0 * gaussian(z)[mask] + gaussian(z + delta)[mask]
    ) / 6.0
    mask = (-0.4 < x) & (x < -0.2)
    out[mask] = 1.0
    mask = (0.0 < x) & (x < 0.2)
    out[mask] = 1.0 - np.abs(10.0 * (x[mask] - 0.1))
    mask = (0.4 < x) & (x < 0.6)
    out[mask] = (
        ellipse(a - delta)[mask] + 4.0 * ellipse(a)[mask] + ellipse(a + delta)[mask]
    ) / 6.0
    return out


def _cell_averages(nx: int, time: float, quadrature: int) -> tuple[np.ndarray, np.ndarray]:
    dx = LENGTH / nx
    centers = X_MIN + (np.arange(nx) + 0.5) * dx
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    samples = centers[:, None] + 0.5 * dx * nodes[None, :] - time
    averages = 0.5 * np.sum(weights[None, :] * _gste_point(samples), axis=1)
    return centers, averages


def _metrics(final: np.ndarray, exact: np.ndarray) -> dict[str, float | bool]:
    diff = final - exact
    return {
        "l1": float(np.mean(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(np.square(diff)))),
        "linf": float(np.max(np.abs(diff))),
        "tv": float(np.sum(np.abs(final - np.roll(final, 1)))),
        "min": float(np.min(final)),
        "max": float(np.max(final)),
        "complete": bool(np.all(np.isfinite(final))),
    }


@dataclass
class GsteMonitor:
    nx: int
    t_end: float
    cfl_limit: float
    quadrature: int
    device: torch.device

    def __post_init__(self) -> None:
        self.x, self.u0 = _cell_averages(self.nx, 0.0, self.quadrature)
        _, self.exact = _cell_averages(self.nx, self.t_end, self.quadrature)
        self.dx = LENGTH / self.nx
        self.steps = int(math.ceil(self.t_end / (self.cfl_limit * self.dx)))
        self.dt = self.t_end / self.steps
        self.actual_cfl = self.dt / self.dx
        self.classical = self._run("classical")

    def _run(self, model) -> dict[str, float | bool]:
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        u = torch.as_tensor(self.u0, device=self.device, dtype=torch.float64).reshape(1, self.nx)
        with torch.no_grad():
            for _ in range(self.steps):
                u = adv5.ssprk3(model, u, self.dt, 1.0 / self.dx)
        final = u[0].detach().cpu().numpy()
        if hasattr(model, "train"):
            model.train(was_training)
        return _metrics(final, self.exact)

    def evaluate(self, model, step: int) -> dict[str, float | bool | int]:
        learned = self._run(model)
        cls_l1 = float(self.classical["l1"])
        cls_l2 = float(self.classical["l2"])
        return {
            "step": step,
            "l1": learned["l1"],
            "l2": learned["l2"],
            "linf": learned["linf"],
            "tv": learned["tv"],
            "min": learned["min"],
            "max": learned["max"],
            "complete": learned["complete"],
            "gain_l1_vs_js": 1.0 - float(learned["l1"]) / cls_l1,
            "gain_l2_vs_js": 1.0 - float(learned["l2"]) / cls_l2,
            "js_l1": cls_l1,
            "js_l2": cls_l2,
            "js_tv": self.classical["tv"],
            "steps": self.steps,
            "cfl": self.actual_cfl,
            "t": self.t_end,
        }
