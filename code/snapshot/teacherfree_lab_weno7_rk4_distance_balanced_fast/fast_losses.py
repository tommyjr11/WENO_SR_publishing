#!/usr/bin/env python3
"""Execution-optimized form of the original WENO7 distance-balanced loss.

The mathematical objective is unchanged.  This module only reduces launch and
Python overhead by precomputing exact/classical trajectories, reusing the
primary trajectory for the short edge guard, and checkpointing several full
RK4 steps at a time.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

import rk4_advection as A
import weno7_core as W

torch.set_default_dtype(torch.float64)

PRIMARY_COMPONENTS = (
    "trajectory",
    "face_path",
    "exact_recon",
    "flat_d2",
    "tv_excess",
    "global_js_guard",
    "local_js_guard",
)


def _signed_select(
    positive: torch.Tensor,
    negative: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    mask = velocities > 0.0
    while mask.ndim < positive.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, positive, negative)


@torch.no_grad()
def precompute_exact_trajectory(
    profiles,
    n: int,
    n_steps: int,
    dt: float,
    velocities: torch.Tensor,
    *,
    target_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact FV states and four reconstruction targets for every step."""
    if n_steps < 1 or target_chunk < 1:
        raise ValueError("n_steps and target_chunk must be positive")

    # Python multiplication reproduces the scalar implementation's time values.
    times = torch.as_tensor(
        [float(step) * float(dt) for step in range(n_steps + 1)],
        device=velocities.device,
        dtype=torch.float64,
    )
    state_blocks: list[torch.Tensor] = []
    point_blocks: list[torch.Tensor] = []
    for begin in range(0, n_steps + 1, target_chunk):
        block_times = times[begin : begin + target_chunk]
        positive = profiles.cell_average_times(n, block_times)
        negative = profiles.cell_average_times(n, -block_times)
        state_blocks.append(
            _signed_select(positive, negative, velocities)
            .permute(1, 0, 2)
            .contiguous()
        )
    for begin in range(0, n_steps, target_chunk):
        block_times = times[begin : min(begin + target_chunk, n_steps)]
        positive = profiles.point_targets_times(
            n, block_times, W.LR_OFFSETS
        )
        negative = profiles.point_targets_times(
            n, -block_times, W.LR_OFFSETS
        )
        point_blocks.append(
            _signed_select(positive, negative, velocities)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
    return torch.cat(state_blocks, dim=0), torch.cat(point_blocks, dim=0)


@torch.no_grad()
def precompute_classical_trajectory(
    initial: torch.Tensor,
    n_steps: int,
    dt: float,
    dxinv: float,
    velocities: torch.Tensor,
    step_operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Run the identical WENO7-JS/Shu-RK4 baseline once and retain its states."""
    state = initial
    states = [state]
    for _ in range(n_steps):
        if step_operator is None:
            state = A.shu_rk4_step_signed(
                "classical", state, dt, dxinv, velocities
            )
        else:
            state = step_operator(state, velocities)
        states.append(state)
    return torch.stack(states, dim=0)


def scaled_smooth_l1(
    prediction: torch.Tensor,
    exact: torch.Tensor,
    qscale: torch.Tensor,
) -> torch.Tensor:
    scale = qscale
    while scale.ndim < prediction.ndim:
        scale = scale.unsqueeze(-1)
    error = (prediction - exact) / scale
    return torch.mean(
        torch.sqrt(torch.square(error) + 1.0e-12) - 1.0e-6
    )


def state_error_per_sample(
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


def second_difference(state: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(state, shifts=-1, dims=1)
        - 2.0 * state
        + torch.roll(state, shifts=1, dims=1)
    )


def flat_region_mask(
    exact: torch.Tensor,
    value_range: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    left = torch.abs(exact - torch.roll(exact, shifts=1, dims=1))
    right = torch.abs(torch.roll(exact, shifts=-1, dims=1) - exact)
    flat = torch.maximum(left, right) <= tolerance * value_range.reshape(-1, 1)
    return (
        flat
        & torch.roll(flat, shifts=1, dims=1)
        & torch.roll(flat, shifts=-1, dims=1)
    )


@torch.no_grad()
def family_sample_weights(family_id: torch.Tensor) -> torch.Tensor:
    """Weights exactly equivalent to averaging present family means."""
    _, inverse, counts = torch.unique(
        family_id, sorted=True, return_inverse=True, return_counts=True
    )
    family_count = counts.numel()
    return 1.0 / (
        float(family_count) * counts[inverse].to(torch.float64)
    )


def weighted_family_mean(
    values: torch.Tensor, sample_weights: torch.Tensor
) -> torch.Tensor:
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    return torch.sum(per_sample * sample_weights)


def tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.numel())))
    return torch.topk(flat, count, sorted=False).values.mean()


def robust_violation(
    values: torch.Tensor,
    sample_weights: torch.Tensor,
    cvar_fraction: float,
) -> torch.Tensor:
    return 0.25 * weighted_family_mean(values, sample_weights) + 0.75 * (
        tail_mean(values, cvar_fraction)
    )


def periodic_local_rms(error: torch.Tensor, window: int) -> torch.Tensor:
    if window < 1 or window > error.shape[1]:
        raise ValueError("local window must lie inside the grid")
    left = (window - 1) // 2
    right = window - 1 - left
    padded = functional.pad(
        error.square().unsqueeze(1), (left, right), mode="circular"
    )
    return torch.sqrt(
        functional.avg_pool1d(padded, window, stride=1).squeeze(1) + 1.0e-30
    )


class TrajectoryStep(torch.nn.Module):
    """One complete RK4 step and all original per-step loss components."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        dt: float,
        dxinv: float,
        flat_tolerance: float,
        local_window: int,
        cvar_fraction: float,
        guard_tolerance: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.dt = float(dt)
        self.dxinv = float(dxinv)
        self.flat_tolerance = float(flat_tolerance)
        self.local_window = int(local_window)
        self.cvar_fraction = float(cvar_fraction)
        self.guard_tolerance = float(guard_tolerance)

    def forward(
        self,
        state: torch.Tensor,
        exact_now: torch.Tensor,
        exact_next: torch.Tensor,
        exact_points: torch.Tensor,
        classical_next: torch.Tensor,
        qscale: torch.Tensor,
        value_range: torch.Tensor,
        sample_weights: torch.Tensor,
        velocities: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        face_loss = scaled_smooth_l1(
            A.interface_state_signed(self.model, state, velocities),
            exact_points[:, 0, :],
            qscale,
        )
        reconstruction_loss = scaled_smooth_l1(
            A.all_head_reconstruction(self.model, exact_now),
            exact_points,
            qscale,
        )
        updated = A.shu_rk4_step_signed(
            self.model, state, self.dt, self.dxinv, velocities
        )

        model_per_sample = state_error_per_sample(
            updated, exact_next, qscale, value_range
        )
        classical_per_sample = state_error_per_sample(
            classical_next, exact_next, qscale, value_range
        )
        trajectory_loss = weighted_family_mean(
            model_per_sample, sample_weights
        )

        global_violation = torch.relu(
            model_per_sample
            - classical_per_sample
            - self.guard_tolerance
        )
        global_guard = robust_violation(
            global_violation, sample_weights, self.cvar_fraction
        )

        scale = value_range.reshape(-1, 1)
        model_local = periodic_local_rms(
            (updated - exact_next) / scale, self.local_window
        )
        classical_local = periodic_local_rms(
            (classical_next - exact_next) / scale, self.local_window
        )
        local_violation = torch.relu(
            model_local - classical_local - self.guard_tolerance
        )
        local_guard = robust_violation(
            local_violation, sample_weights, self.cvar_fraction
        )

        mask = flat_region_mask(
            exact_next, value_range, self.flat_tolerance
        ).to(updated.dtype)
        count = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        d2_error = torch.abs(
            second_difference(updated) - second_difference(exact_next)
        )
        flat_per_sample = torch.sum(mask * d2_error, dim=1) / (
            count * value_range
        )
        flat_loss = weighted_family_mean(flat_per_sample, sample_weights)

        tv_model = torch.sum(
            torch.abs(updated - torch.roll(updated, shifts=1, dims=1)),
            dim=1,
        )
        tv_exact = torch.sum(
            torch.abs(
                exact_next - torch.roll(exact_next, shifts=1, dims=1)
            ),
            dim=1,
        )
        tv_loss = weighted_family_mean(
            torch.relu((tv_model - tv_exact) / value_range),
            sample_weights,
        )
        return (
            updated,
            trajectory_loss,
            face_loss,
            reconstruction_loss,
            flat_loss,
            tv_loss,
            global_guard,
            local_guard,
            model_per_sample.mean(),
            classical_per_sample.mean(),
        )


class ClassicalStep(torch.nn.Module):
    """Identical classical RK4 step exposed as a compilable module."""

    def __init__(self, *, dt: float, dxinv: float) -> None:
        super().__init__()
        self.dt = float(dt)
        self.dxinv = float(dxinv)

    def forward(
        self, state: torch.Tensor, velocities: torch.Tensor
    ) -> torch.Tensor:
        return A.shu_rk4_step_signed(
            "classical", state, self.dt, self.dxinv, velocities
        )


class ShapeTracedCallable:
    """Cache one portable TorchScript trace for each fixed tensor shape."""

    def __init__(self, module: torch.nn.Module, label: str) -> None:
        self.module = module
        self.label = label
        self.cache: dict[tuple[object, ...], torch.jit.ScriptModule] = {}

    @staticmethod
    def _key(arguments: tuple[torch.Tensor, ...]) -> tuple[object, ...]:
        return tuple(
            (
                tuple(argument.shape),
                str(argument.dtype),
                argument.device.type,
                argument.device.index,
            )
            for argument in arguments
        )

    def ensure_traced(
        self, *arguments: torch.Tensor
    ) -> torch.jit.ScriptModule:
        key = self._key(arguments)
        traced = self.cache.get(key)
        if traced is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
                traced = torch.jit.trace(
                    self.module,
                    arguments,
                    check_trace=False,
                    strict=False,
                )
            self.cache[key] = traced
            print(
                f"JIT_TRACE_READY label={self.label} "
                f"primary_shape={tuple(arguments[0].shape)}",
                flush=True,
            )
        return traced

    def __call__(self, *arguments: torch.Tensor):
        traced = self.ensure_traced(*arguments)
        return traced(*arguments)


def _chunk_boundaries(
    n_steps: int, chunk_steps: int, edge_steps: int
) -> tuple[int, ...]:
    boundaries = set(range(chunk_steps, n_steps, chunk_steps))
    if 0 < edge_steps < n_steps:
        boundaries.add(edge_steps)
    boundaries.add(n_steps)
    return tuple(sorted(boundaries))


def fast_autoregressive_trajectory_loss(
    step_operator: Callable[..., tuple[torch.Tensor, ...]],
    classical_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    profiles,
    n: int,
    n_steps: int,
    cfl: float,
    velocities: torch.Tensor,
    *,
    state_lambda: float,
    face_path_lambda: float,
    exact_recon_lambda: float,
    flat_d2_lambda: float,
    tv_lambda: float,
    global_guard_lambda: float,
    local_guard_lambda: float,
    edge_steps: int,
    edge_lambda: float,
    chunk_steps: int,
    target_chunk: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the original objective with lower execution overhead."""
    if n_steps < 1 or chunk_steps < 1 or target_chunk < 1:
        raise ValueError(
            "n_steps, chunk_steps, and target_chunk must be positive"
        )
    edge_steps = min(max(int(edge_steps), 1), n_steps)
    dx = 1.0 / float(n)
    dt = float(cfl) * dx
    dxinv = 1.0 / dx

    exact_states, exact_points = precompute_exact_trajectory(
        profiles,
        n,
        n_steps,
        dt,
        velocities,
        target_chunk=target_chunk,
    )
    classical_states = precompute_classical_trajectory(
        exact_states[0],
        n_steps,
        dt,
        dxinv,
        velocities,
        classical_step,
    )
    initial = exact_states[0]
    qscale = torch.clamp(
        torch.max(torch.abs(initial), dim=1).values, min=1.0
    )
    value_range = torch.clamp(
        torch.max(initial, dim=1).values
        - torch.min(initial, dim=1).values,
        min=1.0e-6 * qscale,
    )
    sample_weights = family_sample_weights(profiles.family_id)
    if isinstance(step_operator, ShapeTracedCallable):
        step_operator.ensure_traced(
            initial,
            exact_states[0],
            exact_states[1],
            exact_points[0],
            classical_states[1],
            qscale,
            value_range,
            sample_weights,
            velocities,
        )

    component_sums = [
        torch.zeros((), device=initial.device, dtype=torch.float64)
        for _ in PRIMARY_COMPONENTS
    ]
    edge_global_sum = torch.zeros(
        (), device=initial.device, dtype=torch.float64
    )
    edge_local_sum = torch.zeros(
        (), device=initial.device, dtype=torch.float64
    )
    model_error_last = torch.zeros((), device=initial.device)
    classical_error_last = torch.zeros((), device=initial.device)
    edge_model_error = torch.zeros((), device=initial.device)
    edge_classical_error = torch.zeros((), device=initial.device)
    use_jit_checkpoint = isinstance(step_operator, ShapeTracedCallable)
    state = (
        initial.detach().requires_grad_(True)
        if use_jit_checkpoint
        else initial
    )
    begin = 0

    for end in _chunk_boundaries(n_steps, chunk_steps, edge_steps):
        exact_chunk = exact_states[begin : end + 1]
        point_chunk = exact_points[begin:end]
        classical_chunk = classical_states[begin + 1 : end + 1]
        edge_flags = (
            torch.arange(begin, end, device=initial.device) < edge_steps
        ).to(torch.float64)

        def advance_chunk(
            current: torch.Tensor,
            local_exact: torch.Tensor,
            local_points: torch.Tensor,
            local_classical: torch.Tensor,
            local_edge_flags: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            sums = [
                torch.zeros((), device=current.device, dtype=current.dtype)
                for _ in PRIMARY_COMPONENTS
            ]
            edge_global = torch.zeros(
                (), device=current.device, dtype=current.dtype
            )
            edge_local = torch.zeros(
                (), device=current.device, dtype=current.dtype
            )
            local_model_last = torch.zeros(
                (), device=current.device, dtype=current.dtype
            )
            local_classical_last = torch.zeros(
                (), device=current.device, dtype=current.dtype
            )
            local_state = current
            for local_index in range(local_points.shape[0]):
                outputs = step_operator(
                    local_state,
                    local_exact[local_index],
                    local_exact[local_index + 1],
                    local_points[local_index],
                    local_classical[local_index],
                    qscale,
                    value_range,
                    sample_weights,
                    velocities,
                )
                local_state = outputs[0]
                for component_index in range(len(PRIMARY_COMPONENTS)):
                    sums[component_index] = (
                        sums[component_index] + outputs[1 + component_index]
                    )
                edge_global = (
                    edge_global + local_edge_flags[local_index] * outputs[6]
                )
                edge_local = (
                    edge_local + local_edge_flags[local_index] * outputs[7]
                )
                local_model_last = outputs[8]
                local_classical_last = outputs[9]
            return (
                local_state,
                *sums,
                edge_global,
                edge_local,
                local_model_last,
                local_classical_last,
            )

        outputs = checkpoint(
            advance_chunk,
            state,
            exact_chunk,
            point_chunk,
            classical_chunk,
            edge_flags,
            use_reentrant=use_jit_checkpoint,
            preserve_rng_state=False,
        )
        state = outputs[0]
        for index in range(len(PRIMARY_COMPONENTS)):
            component_sums[index] = (
                component_sums[index] + outputs[1 + index]
            )
        edge_global_sum = edge_global_sum + outputs[8]
        edge_local_sum = edge_local_sum + outputs[9]
        model_error_last = outputs[10]
        classical_error_last = outputs[11]
        if end == edge_steps:
            edge_model_error = model_error_last
            edge_classical_error = classical_error_last
        begin = end

    primary = [value / float(n_steps) for value in component_sums]
    edge_global = edge_global_sum / float(edge_steps)
    edge_local = edge_local_sum / float(edge_steps)
    total = (
        state_lambda * primary[0]
        + face_path_lambda * primary[1]
        + exact_recon_lambda * primary[2]
        + flat_d2_lambda * primary[3]
        + tv_lambda * primary[4]
        + global_guard_lambda * primary[5]
        + local_guard_lambda * primary[6]
        + edge_lambda
        * (
            global_guard_lambda * edge_global
            + local_guard_lambda * edge_local
        )
    )
    primary_total = (
        state_lambda * primary[0]
        + face_path_lambda * primary[1]
        + exact_recon_lambda * primary[2]
        + flat_d2_lambda * primary[3]
        + tv_lambda * primary[4]
        + global_guard_lambda * primary[5]
        + local_guard_lambda * primary[6]
    )
    edge_total = (
        global_guard_lambda * edge_global
        + local_guard_lambda * edge_local
    )
    stats = {
        name: value.detach()
        for name, value in zip(PRIMARY_COMPONENTS, primary)
    }
    stats.update(
        {
            "primary_loss": primary_total.detach(),
            "edge_loss": edge_total.detach(),
            "edge_global_js_guard": edge_global.detach(),
            "edge_local_js_guard": edge_local.detach(),
            "final_model_error": model_error_last.detach(),
            "final_classical_error": classical_error_last.detach(),
            "final_vs_classical": (
                model_error_last
                / torch.clamp(classical_error_last, min=1.0e-30)
            ).detach(),
            "edge_final_model_error": edge_model_error.detach(),
            "edge_final_classical_error": edge_classical_error.detach(),
            "edge_final_vs_classical": (
                edge_model_error
                / torch.clamp(edge_classical_error, min=1.0e-30)
            ).detach(),
        }
    )
    return total, stats
