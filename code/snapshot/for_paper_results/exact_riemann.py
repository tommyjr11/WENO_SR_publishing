"""Exact self-similar solution of the one-dimensional Euler Riemann problem."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


Primitive = tuple[float, float, float]


def _pressure_function(
    pressure_star: float,
    rho: float,
    pressure: float,
    sound_speed: float,
    gamma: float,
) -> tuple[float, float]:
    if pressure_star > pressure:
        coeff_a = 2.0 / ((gamma + 1.0) * rho)
        coeff_b = (gamma - 1.0) / (gamma + 1.0) * pressure
        root = np.sqrt(coeff_a / (pressure_star + coeff_b))
        value = (pressure_star - pressure) * root
        derivative = root * (
            1.0 - 0.5 * (pressure_star - pressure) / (pressure_star + coeff_b)
        )
        return float(value), float(derivative)

    exponent = (gamma - 1.0) / (2.0 * gamma)
    ratio = pressure_star / pressure
    value = 2.0 * sound_speed / (gamma - 1.0) * (ratio**exponent - 1.0)
    derivative = (1.0 / (rho * sound_speed)) * ratio ** (
        -(gamma + 1.0) / (2.0 * gamma)
    )
    return float(value), float(derivative)


@lru_cache(maxsize=64)
def solve_star(left: Primitive, right: Primitive, gamma: float) -> tuple[float, float]:
    rho_l, velocity_l, pressure_l = left
    rho_r, velocity_r, pressure_r = right
    sound_l = np.sqrt(gamma * pressure_l / rho_l)
    sound_r = np.sqrt(gamma * pressure_r / rho_r)
    guess = 0.5 * (pressure_l + pressure_r) - 0.125 * (
        velocity_r - velocity_l
    ) * (rho_l + rho_r) * (sound_l + sound_r)
    pressure_star = max(float(guess), np.finfo(np.float64).tiny)

    for _ in range(100):
        value_l, derivative_l = _pressure_function(
            pressure_star, rho_l, pressure_l, sound_l, gamma
        )
        value_r, derivative_r = _pressure_function(
            pressure_star, rho_r, pressure_r, sound_r, gamma
        )
        residual = value_l + value_r + velocity_r - velocity_l
        updated = pressure_star - residual / (derivative_l + derivative_r)
        if updated <= 0.0 or not np.isfinite(updated):
            updated = 0.5 * pressure_star
        scale = 0.5 * (updated + pressure_star)
        if abs(updated - pressure_star) <= 1.0e-13 * scale:
            pressure_star = updated
            break
        pressure_star = updated
    else:
        raise RuntimeError("exact Riemann pressure solve did not converge")

    value_l, _ = _pressure_function(
        pressure_star, rho_l, pressure_l, sound_l, gamma
    )
    value_r, _ = _pressure_function(
        pressure_star, rho_r, pressure_r, sound_r, gamma
    )
    velocity_star = 0.5 * (
        velocity_l + velocity_r + value_r - value_l
    )
    return float(pressure_star), float(velocity_star)


def sample_similarity(
    similarity: float,
    left: Primitive,
    right: Primitive,
    gamma: float,
) -> Primitive:
    rho_l, velocity_l, pressure_l = left
    rho_r, velocity_r, pressure_r = right
    pressure_star, velocity_star = solve_star(left, right, gamma)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    sound_l = np.sqrt(gamma * pressure_l / rho_l)
    sound_r = np.sqrt(gamma * pressure_r / rho_r)

    if similarity <= velocity_star:
        if pressure_star > pressure_l:
            shock_speed = velocity_l - sound_l * np.sqrt(
                gp1 / (2.0 * gamma) * (pressure_star / pressure_l)
                + gm1 / (2.0 * gamma)
            )
            if similarity <= shock_speed:
                return left
            ratio = pressure_star / pressure_l
            rho_star = rho_l * (
                (ratio + gm1 / gp1) / ((gm1 / gp1) * ratio + 1.0)
            )
            return float(rho_star), velocity_star, pressure_star

        sound_star = sound_l * (pressure_star / pressure_l) ** (
            gm1 / (2.0 * gamma)
        )
        head_speed = velocity_l - sound_l
        tail_speed = velocity_star - sound_star
        if similarity <= head_speed:
            return left
        if similarity >= tail_speed:
            rho_star = rho_l * (pressure_star / pressure_l) ** (1.0 / gamma)
            return float(rho_star), velocity_star, pressure_star
        velocity = 2.0 / gp1 * (
            sound_l + 0.5 * gm1 * velocity_l + similarity
        )
        sound = 2.0 / gp1 * (
            sound_l + 0.5 * gm1 * (velocity_l - similarity)
        )
        rho = rho_l * (sound / sound_l) ** (2.0 / gm1)
        pressure = pressure_l * (sound / sound_l) ** (2.0 * gamma / gm1)
        return float(rho), float(velocity), float(pressure)

    if pressure_star > pressure_r:
        shock_speed = velocity_r + sound_r * np.sqrt(
            gp1 / (2.0 * gamma) * (pressure_star / pressure_r)
            + gm1 / (2.0 * gamma)
        )
        if similarity >= shock_speed:
            return right
        ratio = pressure_star / pressure_r
        rho_star = rho_r * (
            (ratio + gm1 / gp1) / ((gm1 / gp1) * ratio + 1.0)
        )
        return float(rho_star), velocity_star, pressure_star

    sound_star = sound_r * (pressure_star / pressure_r) ** (
        gm1 / (2.0 * gamma)
    )
    head_speed = velocity_r + sound_r
    tail_speed = velocity_star + sound_star
    if similarity >= head_speed:
        return right
    if similarity <= tail_speed:
        rho_star = rho_r * (pressure_star / pressure_r) ** (1.0 / gamma)
        return float(rho_star), velocity_star, pressure_star
    velocity = 2.0 / gp1 * (
        -sound_r + 0.5 * gm1 * velocity_r + similarity
    )
    sound = 2.0 / gp1 * (
        sound_r - 0.5 * gm1 * (velocity_r - similarity)
    )
    rho = rho_r * (sound / sound_r) ** (2.0 / gm1)
    pressure = pressure_r * (sound / sound_r) ** (2.0 * gamma / gm1)
    return float(rho), float(velocity), float(pressure)


def wave_speeds(left: Primitive, right: Primitive, gamma: float) -> tuple[float, ...]:
    rho_l, velocity_l, pressure_l = left
    rho_r, velocity_r, pressure_r = right
    pressure_star, velocity_star = solve_star(left, right, gamma)
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    sound_l = np.sqrt(gamma * pressure_l / rho_l)
    sound_r = np.sqrt(gamma * pressure_r / rho_r)
    speeds: list[float] = []

    if pressure_star > pressure_l:
        speeds.append(
            velocity_l
            - sound_l
            * np.sqrt(
                gp1 / (2.0 * gamma) * (pressure_star / pressure_l)
                + gm1 / (2.0 * gamma)
            )
        )
    else:
        sound_star_l = sound_l * (pressure_star / pressure_l) ** (
            gm1 / (2.0 * gamma)
        )
        speeds.extend((velocity_l - sound_l, velocity_star - sound_star_l))

    speeds.append(velocity_star)
    if pressure_star > pressure_r:
        speeds.append(
            velocity_r
            + sound_r
            * np.sqrt(
                gp1 / (2.0 * gamma) * (pressure_star / pressure_r)
                + gm1 / (2.0 * gamma)
            )
        )
    else:
        sound_star_r = sound_r * (pressure_star / pressure_r) ** (
            gm1 / (2.0 * gamma)
        )
        speeds.extend((velocity_star + sound_star_r, velocity_r + sound_r))
    return tuple(sorted(float(speed) for speed in speeds))


def sample_points(
    x: np.ndarray,
    time: float,
    discontinuity: float,
    left: Primitive,
    right: Primitive,
    gamma: float = 1.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if time <= 0.0:
        mask = x < discontinuity
        rho = np.where(mask, left[0], right[0]).astype(np.float64)
        velocity = np.where(mask, left[1], right[1]).astype(np.float64)
        pressure = np.where(mask, left[2], right[2]).astype(np.float64)
        return rho, velocity, pressure
    sampled = np.array(
        [
            sample_similarity((float(coord) - discontinuity) / time, left, right, gamma)
            for coord in np.asarray(x)
        ],
        dtype=np.float64,
    )
    return sampled[:, 0], sampled[:, 1], sampled[:, 2]


def _conserved(primitive: Primitive, gamma: float) -> np.ndarray:
    rho, velocity, pressure = primitive
    return np.array(
        [
            rho,
            rho * velocity,
            0.0,
            pressure / (gamma - 1.0) + 0.5 * rho * velocity * velocity,
        ],
        dtype=np.float64,
    )


def cell_average_conserved(
    centers: np.ndarray,
    width: float,
    time: float,
    discontinuity: float,
    left: Primitive,
    right: Primitive,
    gamma: float = 1.4,
    quadrature: int = 20,
) -> np.ndarray:
    """Integrate each exact cell average, splitting at every wave boundary."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    if time > 0.0:
        boundaries = np.array(
            [discontinuity + time * speed for speed in wave_speeds(left, right, gamma)],
            dtype=np.float64,
        )
    else:
        boundaries = np.array([discontinuity], dtype=np.float64)

    averages = np.zeros((len(centers), 4), dtype=np.float64)
    for cell_index, center in enumerate(np.asarray(centers, dtype=np.float64)):
        lower = center - 0.5 * width
        upper = center + 0.5 * width
        internal = boundaries[(boundaries > lower) & (boundaries < upper)]
        edges = np.concatenate(([lower], internal, [upper]))
        integral = np.zeros(4, dtype=np.float64)
        for segment_lower, segment_upper in zip(edges[:-1], edges[1:]):
            midpoint = 0.5 * (segment_lower + segment_upper)
            half_width = 0.5 * (segment_upper - segment_lower)
            for node, weight in zip(nodes, weights):
                coord = midpoint + half_width * float(node)
                if time <= 0.0:
                    primitive = left if coord < discontinuity else right
                else:
                    primitive = sample_similarity(
                        (coord - discontinuity) / time, left, right, gamma
                    )
                integral += half_width * float(weight) * _conserved(primitive, gamma)
        averages[cell_index] = integral / width
    return averages
