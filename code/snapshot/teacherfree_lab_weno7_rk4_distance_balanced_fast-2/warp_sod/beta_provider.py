#!/usr/bin/env python3
"""Torch beta provider for reflection-symmetric WENO7 checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch

import weno7_core as W


class TorchWeno7Beta:
    """Evaluate the compact MLP in Torch and supply beta arrays to Warp."""

    def __init__(
        self, model_path: Path, torch_device: str, gamma: float = 1.4
    ) -> None:
        self.device = torch.device(torch_device)
        self.gamma = float(gamma)
        self.model = W.load_checkpoint(Path(model_path), self.device)
        self.model.eval()

    def _sync_after_copy(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _mlp_beta(self, q: torch.Tensor) -> torch.Tensor:
        ratio = self.model(W.weno7_features(q))
        beta = W.BADNESS_RATIO_SCALE * ratio
        return torch.where(
            W.plateau_mask(q).unsqueeze(-1), torch.ones_like(beta), beta
        )

    def _roe_average(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        gamma = self.gamma
        tiny = 1.0e-15
        gm1 = gamma - 1.0
        rho_l = torch.clamp(left[..., 0], min=tiny)
        u_l = left[..., 1] / rho_l
        v_l = left[..., 2] / rho_l
        e_l = left[..., 3]
        p_l = torch.clamp(
            gm1 * (e_l - 0.5 * rho_l * (u_l * u_l + v_l * v_l)),
            min=tiny,
        )
        h_l = (e_l + p_l) / rho_l
        rho_r = torch.clamp(right[..., 0], min=tiny)
        u_r = right[..., 1] / rho_r
        v_r = right[..., 2] / rho_r
        e_r = right[..., 3]
        p_r = torch.clamp(
            gm1 * (e_r - 0.5 * rho_r * (u_r * u_r + v_r * v_r)),
            min=tiny,
        )
        h_r = (e_r + p_r) / rho_r
        sqrt_l = torch.sqrt(rho_l)
        sqrt_r = torch.sqrt(rho_r)
        inverse = 1.0 / (sqrt_l + sqrt_r)
        u = (sqrt_l * u_l + sqrt_r * u_r) * inverse
        v = (sqrt_l * v_l + sqrt_r * v_r) * inverse
        h = (sqrt_l * h_l + sqrt_r * h_r) * inverse
        rho = sqrt_l * sqrt_r
        speed_squared = u * u + v * v
        energy = (
            rho * h + 0.5 * gm1 * rho * speed_squared
        ) / gamma
        return torch.stack(
            (rho, rho * u, rho * v, energy), dim=-1
        )

    def _jac_values(
        self, roe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = self.gamma
        tiny = 1.0e-15
        rho = torch.clamp(roe[..., 0], min=tiny)
        u = roe[..., 1] / rho
        v = roe[..., 2] / rho
        gm1 = gamma - 1.0
        speed_squared = u * u + v * v
        pressure = torch.clamp(
            gm1 * (roe[..., 3] - 0.5 * rho * speed_squared), min=tiny
        )
        sound = torch.sqrt(gamma * pressure / rho)
        enthalpy = 0.5 * speed_squared + sound * sound / gm1
        return u, v, sound, enthalpy

    def _con_to_char(
        self, q: torch.Tensor, roe: torch.Tensor, direction: int
    ) -> torch.Tensor:
        gamma = self.gamma
        u, v, sound, enthalpy = self._jac_values(roe)
        gm1 = gamma - 1.0
        sound_squared = sound * sound
        scale = gm1 / (2.0 * sound_squared)
        if direction == 1:
            c0 = (
                (
                    enthalpy
                    + sound * (u - sound) / gm1
                )
                * q[..., 0]
                - (u + sound / gm1) * q[..., 1]
                - v * q[..., 2]
                + q[..., 3]
            ) * scale
            c1 = (
                (-2.0 * enthalpy + 4.0 * sound_squared / gm1)
                * q[..., 0]
                + 2.0 * u * q[..., 1]
                + 2.0 * v * q[..., 2]
                - 2.0 * q[..., 3]
            ) * scale
            c2 = (
                -2.0 * v * sound_squared / gm1 * q[..., 0]
                + 2.0 * sound_squared / gm1 * q[..., 2]
            ) * scale
            c3 = (
                (
                    enthalpy
                    - sound * (u + sound) / gm1
                )
                * q[..., 0]
                + (-u + sound / gm1) * q[..., 1]
                - v * q[..., 2]
                + q[..., 3]
            ) * scale
        else:
            tangent = u
            normal = v
            c0 = (
                (
                    enthalpy
                    + sound * (normal - sound) / gm1
                )
                * q[..., 0]
                - tangent * q[..., 1]
                - (normal + sound / gm1) * q[..., 2]
                + q[..., 3]
            ) * scale
            c1 = (
                (-2.0 * enthalpy + 4.0 * sound_squared / gm1)
                * q[..., 0]
                + 2.0 * tangent * q[..., 1]
                + 2.0 * normal * q[..., 2]
                - 2.0 * q[..., 3]
            ) * scale
            c2 = (
                -2.0 * tangent * sound_squared / gm1 * q[..., 0]
                + 2.0 * sound_squared / gm1 * q[..., 1]
            ) * scale
            c3 = (
                (
                    enthalpy
                    - sound * (normal + sound) / gm1
                )
                * q[..., 0]
                - tangent * q[..., 1]
                + (-normal + sound / gm1) * q[..., 2]
                + q[..., 3]
            ) * scale
        return torch.stack((c0, c1, c2, c3), dim=-1)

    def _compute_beta(
        self, q: list[torch.Tensor], direction: int
    ) -> torch.Tensor:
        shape = q[0].shape[:-1]
        beta = torch.empty(
            shape + (32,), dtype=torch.float64, device=self.device
        )
        for side, lr in ((0, 1), (1, 2)):
            roe = (
                self._roe_average(q[3], q[4])
                if lr == 1
                else self._roe_average(q[2], q[3])
            )
            characteristic = [
                self._con_to_char(value, roe, direction) for value in q
            ]
            for component in range(4):
                values = torch.stack(
                    [value[..., component] for value in characteristic],
                    dim=-1,
                )
                if lr == 1:
                    output = self._mlp_beta(values)
                else:
                    output = torch.flip(
                        self._mlp_beta(torch.flip(values, dims=(-1,))),
                        dims=(-1,),
                    )
                start = side * 16 + component * 4
                beta[..., start : start + 4] = output
        return beta

    def _compute_beta_conservative(
        self, q: list[torch.Tensor]
    ) -> torch.Tensor:
        shape = q[0].shape[:-1]
        beta = torch.empty(
            shape + (32,), dtype=torch.float64, device=self.device
        )
        for side, lr in ((0, 1), (1, 2)):
            for component in range(4):
                values = torch.stack(
                    [value[..., component] for value in q], dim=-1
                )
                if lr == 1:
                    output = self._mlp_beta(values)
                else:
                    output = torch.flip(
                        self._mlp_beta(torch.flip(values, dims=(-1,))),
                        dims=(-1,),
                    )
                start = side * 16 + component * 4
                beta[..., start : start + 4] = output
        return beta

