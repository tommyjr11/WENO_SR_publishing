"""Scalar finite-volume WENO-Z reconstructions used by the GSTE audit."""
from __future__ import annotations

import torch

from teacherfree_lab_weno5 import weno5_core as W5
from teacherfree_lab_weno7_rk4_distance_balanced_fast import weno7_core as W7


P_Z5 = 2
P_Z7 = 3


def omega5(q: torch.Tensor, lr: int, h: float) -> torch.Tensor:
    beta = W5.classical_beta(q)
    tau = torch.abs(beta[..., 0] - beta[..., 2]).unsqueeze(-1)
    d = W5.optimal_d(lr, q.device).to(q.dtype)
    ratio = tau / (beta + h**2)
    alpha = d * (1.0 + ratio**P_Z5)
    return alpha / torch.sum(alpha, dim=-1, keepdim=True)


def omega7(q: torch.Tensor, lr: int, h: float) -> torch.Tensor:
    beta = W7.classical_beta(q)
    tau = torch.abs(
        -beta[..., 0] - 3.0 * beta[..., 1]
        + 3.0 * beta[..., 2] + beta[..., 3]
    ).unsqueeze(-1)
    d = W7.optimal_d(lr, q.device, q.dtype)
    ratio = tau / (beta + h**3)
    alpha = d * (1.0 + ratio**P_Z7)
    return alpha / torch.sum(alpha, dim=-1, keepdim=True)


def weno5_faces(state: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n = state.shape
    periodic = torch.cat((state[:, -2:], state, state[:, :2]), dim=1)
    q = periodic.unfold(1, 5, 1).reshape(batch * n, 5)
    left = torch.sum(omega5(q, 1, h) * W5.candidate_values(q, 1), dim=-1)
    right_left_face = torch.sum(
        omega5(q, 2, h) * W5.candidate_values(q, 2), dim=-1
    )
    return left.reshape(batch, n), torch.roll(
        right_left_face.reshape(batch, n), shifts=-1, dims=1
    )


def weno7_faces(state: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n = state.shape
    periodic = torch.cat((state[:, -3:], state, state[:, :3]), dim=1)
    q = periodic.unfold(1, 7, 1).reshape(batch * n, 7)
    left = torch.sum(omega7(q, 1, h) * W7.candidate_values(q, 1), dim=-1)
    right_left_face = torch.sum(
        omega7(q, 2, h) * W7.candidate_values(q, 2), dim=-1
    )
    return left.reshape(batch, n), torch.roll(
        right_left_face.reshape(batch, n), shifts=-1, dims=1
    )


def rhs5(state: torch.Tensor, dxinv: float, reverse: bool = False) -> torch.Tensor:
    left, right = weno5_faces(state, 1.0 / dxinv)
    face = right if reverse else left
    return -(face - torch.roll(face, shifts=1, dims=1)) * dxinv


def rhs7(state: torch.Tensor, dxinv: float, reverse: bool = False) -> torch.Tensor:
    left, right = weno7_faces(state, 1.0 / dxinv)
    face = right if reverse else left
    return -(face - torch.roll(face, shifts=1, dims=1)) * dxinv


def ssprk3_step(state: torch.Tensor, dt: float, dxinv: float) -> torch.Tensor:
    u1 = state + dt * rhs5(state, dxinv)
    u2 = 0.75 * state + 0.25 * (u1 + dt * rhs5(u1, dxinv))
    return state / 3.0 + (2.0 / 3.0) * (u2 + dt * rhs5(u2, dxinv))


def shu_rk4_step(state: torch.Tensor, dt: float, dxinv: float) -> torch.Tensor:
    l0 = rhs7(state, dxinv)
    lt0 = rhs7(state, dxinv, reverse=True)
    u1 = state + 0.5 * dt * l0
    l1 = rhs7(u1, dxinv)
    lt1 = rhs7(u1, dxinv, reverse=True)
    u2 = (
        (649.0 / 1600.0) * state
        - (10890423.0 / 25193600.0) * dt * lt0
        + (951.0 / 1600.0) * u1
        + (5000.0 / 7873.0) * dt * l1
    )
    l2 = rhs7(u2, dxinv)
    u3 = (
        (53989.0 / 2500000.0) * state
        - (102261.0 / 5000000.0) * dt * lt0
        + (4806213.0 / 20000000.0) * u1
        - (5121.0 / 20000.0) * dt * lt1
        + (23619.0 / 32000.0) * u2
        + (7873.0 / 10000.0) * dt * l2
    )
    l3 = rhs7(u3, dxinv)
    return (
        state / 5.0
        + dt * l0 / 10.0
        + (6127.0 / 30000.0) * u1
        + dt * l1 / 6.0
        + (7873.0 / 30000.0) * u2
        + u3 / 3.0
        + dt * l3 / 6.0
    )


def self_test() -> None:
    q5 = torch.ones((3, 5), dtype=torch.float64)
    q7 = torch.ones((3, 7), dtype=torch.float64)
    for lr in (1, 2, 3, 4):
        torch.testing.assert_close(
            omega5(q5, lr, 0.01), W5.optimal_d(lr, q5.device).expand(3, 3)
        )
        torch.testing.assert_close(
            omega7(q7, lr, 0.01), W7.optimal_d(lr, q7.device).expand(3, 4)
        )
    sample5 = torch.tensor([[0.1, 0.2, 0.45, 0.9, 1.4]], dtype=torch.float64)
    sample7 = torch.tensor(
        [[0.0, 0.1, 0.25, 0.5, 0.95, 1.4, 1.8]], dtype=torch.float64
    )
    for omega, q in ((omega5, sample5), (omega7, sample7)):
        weights = omega(q, 1, 0.01)
        if not torch.all(torch.isfinite(weights)):
            raise AssertionError("non-finite WENO-Z weights")
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
