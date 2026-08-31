#!/usr/bin/env python3
"""Trusted 2D characteristic WENO7/Shu-RK4 Sod evaluation."""
from __future__ import annotations

import numpy as np

import sod_exact
from warp_sod import point_rk4 as base
from warp_sod import point_rk4_mlp as mlp_solver

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)


def make_sod_params(
    nx: int = 100,
    ny: int = 10,
    t_end: float = 0.25,
    cfl: float = 0.4,
) -> base.Params:
    return base.Params(
        nx=nx,
        ny=ny,
        x_min=-0.5,
        x_max=0.5,
        y_min=0.0,
        y_max=ny * (1.0 / nx),
        cfl=cfl,
        t_end=t_end,
    )


def make_sod_u0(params: base.Params) -> np.ndarray:
    ghost = params.ghost
    _, ii = np.indices(
        (params.ny + 2 * ghost, params.nx + 2 * ghost)
    )
    x = params.x_min + (ii - ghost + 0.5) * params.dx
    density = np.where(x < 0.0, SOD_LEFT[0], SOD_RIGHT[0])
    velocity = np.where(x < 0.0, SOD_LEFT[1], SOD_RIGHT[1])
    pressure = np.where(x < 0.0, SOD_LEFT[2], SOD_RIGHT[2])
    transverse_velocity = np.zeros_like(density)
    return base.conserved_from_primitive(
        density,
        velocity,
        transverse_velocity,
        pressure,
        params.gamma,
    )


def reference_density(params: base.Params, time: float) -> np.ndarray:
    centers = params.x_min + (
        np.arange(params.nx) + 0.5
    ) * params.dx
    return sod_exact.density_cell_average(
        centers, params.dx, time, params.gamma, quadrature=15
    )


def interior_density(final: np.ndarray, params: base.Params) -> np.ndarray:
    ghost = params.ghost
    density = final[
        ghost : ghost + params.ny,
        ghost : ghost + params.nx,
        0,
    ]
    return density.mean(axis=0)


def density_metrics(
    density: np.ndarray, reference: np.ndarray
) -> dict[str, float | bool]:
    difference = density - reference
    finite = bool(np.all(np.isfinite(density)))
    return {
        "l1": (
            float(np.mean(np.abs(difference))) if finite else float("nan")
        ),
        "l2": (
            float(np.sqrt(np.mean(np.square(difference))))
            if finite
            else float("nan")
        ),
        "linf": (
            float(np.max(np.abs(difference))) if finite else float("nan")
        ),
        "finite": finite,
    }


def eval_classical(
    params: base.Params,
    device: str,
    *,
    solver: str,
    report_interval: int = 0,
) -> dict[str, object]:
    final, summary = base.run_from_initial(
        make_sod_u0(params),
        params,
        device=device,
        riemann_solver=solver,
        characteristic=True,
        boundary="outflow",
        report_interval=report_interval,
    )
    density = interior_density(final, params)
    metric = density_metrics(
        density, reference_density(params, float(summary["t"]))
    )
    return {**metric, "summary": summary, "density": density}


def eval_mlp(
    params: base.Params,
    device: str,
    beta_model,
    *,
    solver: str,
    eno_cutoff: bool = False,
    report_interval: int = 0,
) -> dict[str, object]:
    final, summary = mlp_solver.run_from_initial_mlp(
        make_sod_u0(params),
        params,
        device=device,
        riemann_solver=solver,
        beta_model=beta_model,
        boundary="outflow",
        eno_cutoff=eno_cutoff,
        report_interval=report_interval,
    )
    density = interior_density(final, params)
    metric = density_metrics(
        density, reference_density(params, float(summary["t"]))
    )
    return {**metric, "summary": summary, "density": density}

