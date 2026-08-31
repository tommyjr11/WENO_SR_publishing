#!/usr/bin/env python3
"""Exact ideal-gas Riemann solution used by the standalone Sod validator."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def pressure_function(
    pressure_star: float,
    density: float,
    pressure: float,
    sound: float,
    gamma: float,
) -> tuple[float, float]:
    if pressure_star > pressure:
        coefficient_a = 2.0 / ((gamma + 1.0) * density)
        coefficient_b = (gamma - 1.0) / (gamma + 1.0) * pressure
        root = np.sqrt(coefficient_a / (pressure_star + coefficient_b))
        value = (pressure_star - pressure) * root
        derivative = root * (
            1.0
            - 0.5
            * (pressure_star - pressure)
            / (pressure_star + coefficient_b)
        )
        return float(value), float(derivative)
    exponent = (gamma - 1.0) / (2.0 * gamma)
    ratio = max(pressure_star / pressure, 1.0e-300)
    value = (
        2.0 * sound / (gamma - 1.0) * (ratio**exponent - 1.0)
    )
    derivative = (
        1.0
        / (density * sound)
        * ratio ** (-(gamma + 1.0) / (2.0 * gamma))
    )
    return float(value), float(derivative)


@lru_cache(maxsize=64)
def solve_star(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    gamma: float,
) -> tuple[float, float]:
    density_l, velocity_l, pressure_l = left
    density_r, velocity_r, pressure_r = right
    sound_l = np.sqrt(gamma * pressure_l / density_l)
    sound_r = np.sqrt(gamma * pressure_r / density_r)
    guess = 0.5 * (pressure_l + pressure_r) - 0.125 * (
        velocity_r - velocity_l
    ) * (density_l + density_r) * (sound_l + sound_r)
    pressure_star = max(float(guess), 1.0e-12)
    for _ in range(80):
        value_l, derivative_l = pressure_function(
            pressure_star, density_l, pressure_l, sound_l, gamma
        )
        value_r, derivative_r = pressure_function(
            pressure_star, density_r, pressure_r, sound_r, gamma
        )
        residual = value_l + value_r + velocity_r - velocity_l
        new_pressure = pressure_star - residual / (
            derivative_l + derivative_r
        )
        if new_pressure <= 0.0 or not np.isfinite(new_pressure):
            new_pressure = 0.5 * pressure_star
        if abs(new_pressure - pressure_star) <= 1.0e-12 * (
            0.5 * (new_pressure + pressure_star) + 1.0e-12
        ):
            pressure_star = new_pressure
            break
        pressure_star = new_pressure
    value_l, _ = pressure_function(
        pressure_star, density_l, pressure_l, sound_l, gamma
    )
    value_r, _ = pressure_function(
        pressure_star, density_r, pressure_r, sound_r, gamma
    )
    velocity_star = 0.5 * (
        velocity_l + velocity_r + value_r - value_l
    )
    return float(max(pressure_star, 1.0e-14)), float(velocity_star)


def sample_primitive(
    similarity: float,
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    gamma: float,
) -> tuple[float, float, float]:
    density_l, velocity_l, pressure_l = left
    density_r, velocity_r, pressure_r = right
    pressure_star, velocity_star = solve_star(left, right, gamma)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    sound_l = np.sqrt(gamma * pressure_l / density_l)
    sound_r = np.sqrt(gamma * pressure_r / density_r)

    if similarity <= velocity_star:
        if pressure_star > pressure_l:
            shock_speed = velocity_l - sound_l * np.sqrt(
                gp1 / (2.0 * gamma) * pressure_star / pressure_l
                + gm1 / (2.0 * gamma)
            )
            if similarity <= shock_speed:
                return density_l, velocity_l, pressure_l
            ratio = pressure_star / pressure_l
            density_star = density_l * (
                (ratio + gm1 / gp1) / (gm1 / gp1 * ratio + 1.0)
            )
            return float(density_star), velocity_star, pressure_star
        sound_star = sound_l * (pressure_star / pressure_l) ** (
            gm1 / (2.0 * gamma)
        )
        head_speed = velocity_l - sound_l
        tail_speed = velocity_star - sound_star
        if similarity <= head_speed:
            return density_l, velocity_l, pressure_l
        if similarity >= tail_speed:
            density_star = density_l * (
                pressure_star / pressure_l
            ) ** (1.0 / gamma)
            return float(density_star), velocity_star, pressure_star
        velocity = 2.0 / gp1 * (
            sound_l + 0.5 * gm1 * velocity_l + similarity
        )
        sound = 2.0 / gp1 * (
            sound_l + 0.5 * gm1 * (velocity_l - similarity)
        )
        density = density_l * (sound / sound_l) ** (2.0 / gm1)
        pressure = pressure_l * (sound / sound_l) ** (
            2.0 * gamma / gm1
        )
        return float(density), float(velocity), float(pressure)

    if pressure_star > pressure_r:
        shock_speed = velocity_r + sound_r * np.sqrt(
            gp1 / (2.0 * gamma) * pressure_star / pressure_r
            + gm1 / (2.0 * gamma)
        )
        if similarity >= shock_speed:
            return density_r, velocity_r, pressure_r
        ratio = pressure_star / pressure_r
        density_star = density_r * (
            (ratio + gm1 / gp1) / (gm1 / gp1 * ratio + 1.0)
        )
        return float(density_star), velocity_star, pressure_star
    sound_star = sound_r * (pressure_star / pressure_r) ** (
        gm1 / (2.0 * gamma)
    )
    head_speed = velocity_r + sound_r
    tail_speed = velocity_star + sound_star
    if similarity >= head_speed:
        return density_r, velocity_r, pressure_r
    if similarity <= tail_speed:
        density_star = density_r * (
            pressure_star / pressure_r
        ) ** (1.0 / gamma)
        return float(density_star), velocity_star, pressure_star
    velocity = 2.0 / gp1 * (
        -sound_r + 0.5 * gm1 * velocity_r + similarity
    )
    sound = 2.0 / gp1 * (
        sound_r - 0.5 * gm1 * (velocity_r - similarity)
    )
    density = density_r * (sound / sound_r) ** (2.0 / gm1)
    pressure = pressure_r * (sound / sound_r) ** (
        2.0 * gamma / gm1
    )
    return float(density), float(velocity), float(pressure)


def exact_primitive(
    x: np.ndarray,
    time: float,
    gamma: float = 1.4,
    left: tuple[float, float, float] = (1.0, 0.0, 1.0),
    right: tuple[float, float, float] = (0.125, 0.0, 0.1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    density = np.empty_like(x)
    velocity = np.empty_like(x)
    pressure = np.empty_like(x)
    if time <= 0.0:
        mask = x < 0.0
        density[mask], velocity[mask], pressure[mask] = left
        density[~mask], velocity[~mask], pressure[~mask] = right
        return density, velocity, pressure
    for index, coordinate in np.ndenumerate(x):
        density[index], velocity[index], pressure[index] = sample_primitive(
            float(coordinate / time), left, right, gamma
        )
    return density, velocity, pressure


def density_cell_average(
    centers: np.ndarray,
    dx: float,
    time: float,
    gamma: float = 1.4,
    quadrature: int = 15,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    output = np.zeros_like(centers, dtype=np.float64)
    for node, weight in zip(nodes, weights):
        density, _, _ = exact_primitive(
            centers + 0.5 * dx * float(node), time, gamma
        )
        output += 0.5 * float(weight) * density
    return output

