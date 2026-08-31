#!/usr/bin/env python3
"""Differentiable scalar FV advection with WENO7 and Shu's TVD RK4."""
from __future__ import annotations

import torch

from . import weno7_core as W

torch.set_default_dtype(torch.float64)


def balanced_velocities(batch: int, device: torch.device) -> torch.Tensor:
    if batch < 2 or batch % 2:
        raise ValueError("bidirectional batches must be positive and even")
    velocity = torch.ones(batch, device=device, dtype=torch.float64)
    velocity[batch // 2 :] = -1.0
    return velocity


def select_by_velocity(
    positive: torch.Tensor,
    negative: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    mask = velocities > 0.0
    while mask.ndim < positive.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, positive, negative)


def stencils(state: torch.Tensor) -> torch.Tensor:
    batch, n = state.shape
    periodic = torch.cat((state[:, -3:], state, state[:, :3]), dim=1)
    return periodic.unfold(1, W.FULL_STENCIL, 1).reshape(
        batch * n, W.FULL_STENCIL
    )


def reconstruct_q(
    model,
    q: torch.Tensor,
    lr: int,
    *,
    eno_cutoff: bool = False,
) -> torch.Tensor:
    d = W.optimal_d(lr, q.device, q.dtype).expand(q.shape[0], W.R)
    if model is None:
        omega = d
    elif isinstance(model, str) and model == "classical":
        omega = W.classical_omega(q, lr)
    else:
        ratio = model(W.weno7_features(q))
        omega = W.omega_from_ratio(ratio, lr)
        omega = torch.where(W.plateau_mask(q).unsqueeze(-1), d, omega)
    omega = W.apply_eno_cutoff(omega, eno_cutoff)
    return torch.sum(omega * W.candidate_values(q, lr), dim=-1)


def all_head_reconstruction(model, state: torch.Tensor) -> torch.Tensor:
    batch, n = state.shape
    q = stencils(state)
    if model is None or isinstance(model, str):
        values = [
            reconstruct_q(model, q, lr).reshape(batch, n)
            for lr in W.LR_VALUES
        ]
    else:
        ratio = model(W.weno7_features(q))
        plateau = W.plateau_mask(q).unsqueeze(-1)
        values = []
        for lr in W.LR_VALUES:
            d = W.optimal_d(lr, q.device, q.dtype).expand(q.shape[0], W.R)
            omega = torch.where(plateau, d, W.omega_from_ratio(ratio, lr))
            values.append(
                torch.sum(
                    omega * W.candidate_values(q, lr), dim=-1
                ).reshape(batch, n)
            )
    return torch.stack(values, dim=1)


def physical_face_states(
    model,
    state: torch.Tensor,
    *,
    eno_cutoff: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left/right states at physical face i+1/2."""
    batch, n = state.shape
    q = stencils(state)
    left = reconstruct_q(
        model, q, 1, eno_cutoff=eno_cutoff
    ).reshape(batch, n)
    right_at_left_face = reconstruct_q(
        model, q, 2, eno_cutoff=eno_cutoff
    ).reshape(batch, n)
    right = torch.roll(right_at_left_face, shifts=-1, dims=1)
    return left, right


def interface_state_signed(
    model,
    state: torch.Tensor,
    velocities: torch.Tensor,
    *,
    reverse_upwind: bool = False,
    eno_cutoff: bool = False,
) -> torch.Tensor:
    left, right = physical_face_states(
        model, state, eno_cutoff=eno_cutoff
    )
    if reverse_upwind:
        return select_by_velocity(right, left, velocities)
    return select_by_velocity(left, right, velocities)


def rhs_signed(
    model,
    state: torch.Tensor,
    dxinv: float,
    velocities: torch.Tensor,
    *,
    reverse_upwind: bool = False,
    eno_cutoff: bool = False,
) -> torch.Tensor:
    face_state = interface_state_signed(
        model,
        state,
        velocities,
        reverse_upwind=reverse_upwind,
        eno_cutoff=eno_cutoff,
    )
    flux = velocities.reshape(-1, 1) * face_state
    return -(flux - torch.roll(flux, shifts=1, dims=1)) * dxinv


def shu_rk4_step_signed(
    model,
    state: torch.Tensor,
    dt: float,
    dxinv: float,
    velocities: torch.Tensor,
    *,
    eno_cutoff: bool = False,
) -> torch.Tensor:
    """Exact coefficient sequence used by weno7_point_rk4_shu."""
    l0 = rhs_signed(
        model, state, dxinv, velocities, eno_cutoff=eno_cutoff
    )
    lt0 = rhs_signed(
        model,
        state,
        dxinv,
        velocities,
        reverse_upwind=True,
        eno_cutoff=eno_cutoff,
    )
    u1 = state + 0.5 * dt * l0

    l1 = rhs_signed(model, u1, dxinv, velocities, eno_cutoff=eno_cutoff)
    lt1 = rhs_signed(
        model,
        u1,
        dxinv,
        velocities,
        reverse_upwind=True,
        eno_cutoff=eno_cutoff,
    )
    u2 = (
        (649.0 / 1600.0) * state
        - (10890423.0 / 25193600.0) * dt * lt0
        + (951.0 / 1600.0) * u1
        + (5000.0 / 7873.0) * dt * l1
    )

    l2 = rhs_signed(model, u2, dxinv, velocities, eno_cutoff=eno_cutoff)
    u3 = (
        (53989.0 / 2500000.0) * state
        - (102261.0 / 5000000.0) * dt * lt0
        + (4806213.0 / 20000000.0) * u1
        - (5121.0 / 20000.0) * dt * lt1
        + (23619.0 / 32000.0) * u2
        + (7873.0 / 10000.0) * dt * l2
    )

    l3 = rhs_signed(model, u3, dxinv, velocities, eno_cutoff=eno_cutoff)
    return (
        (1.0 / 5.0) * state
        + (1.0 / 10.0) * dt * l0
        + (6127.0 / 30000.0) * u1
        + (1.0 / 6.0) * dt * l1
        + (7873.0 / 30000.0) * u2
        + (1.0 / 3.0) * u3
        + (1.0 / 6.0) * dt * l3
    )


def rollout_signed(
    model,
    initial: torch.Tensor,
    n_steps: int,
    cfl: float,
    velocities: torch.Tensor,
    *,
    domain_length: float = 1.0,
    eno_cutoff: bool = False,
) -> torch.Tensor:
    n = initial.shape[1]
    dx = float(domain_length) / float(n)
    state = initial
    for _ in range(n_steps):
        state = shu_rk4_step_signed(
            model,
            state,
            float(cfl) * dx,
            1.0 / dx,
            velocities,
            eno_cutoff=eno_cutoff,
        )
    return state


def check_shu_rk4_order() -> float:
    """Check the algebraic RK formula on y'=y where L_tilde=L."""

    def one_step(value: torch.Tensor, dt: float) -> torch.Tensor:
        l0 = value
        lt0 = value
        u1 = value + 0.5 * dt * l0
        l1 = u1
        lt1 = u1
        u2 = (
            (649.0 / 1600.0) * value
            - (10890423.0 / 25193600.0) * dt * lt0
            + (951.0 / 1600.0) * u1
            + (5000.0 / 7873.0) * dt * l1
        )
        l2 = u2
        u3 = (
            (53989.0 / 2500000.0) * value
            - (102261.0 / 5000000.0) * dt * lt0
            + (4806213.0 / 20000000.0) * u1
            - (5121.0 / 20000.0) * dt * lt1
            + (23619.0 / 32000.0) * u2
            + (7873.0 / 10000.0) * dt * l2
        )
        return (
            (1.0 / 5.0) * value
            + (1.0 / 10.0) * dt * l0
            + (6127.0 / 30000.0) * u1
            + (1.0 / 6.0) * dt * l1
            + (7873.0 / 30000.0) * u2
            + (1.0 / 3.0) * u3
            + (1.0 / 6.0) * dt * u3
        )

    errors = []
    for dt in (0.2, 0.1, 0.05):
        value = one_step(torch.ones((), dtype=torch.float64), dt)
        errors.append(abs(float(value) - float(torch.exp(torch.tensor(dt)))))
    order = float(torch.log2(torch.tensor(errors[1] / errors[2])))
    if order < 3.8:
        raise AssertionError(f"Shu RK4 order check failed: order={order:.6f}")
    return order
