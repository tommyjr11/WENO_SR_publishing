from __future__ import annotations

import warp as wp

from warp_weno5_3d_rk3 import solver as base

from . import weno5_3d_kernels_z as K


# The trusted solver owns allocation, EVILIN, boundaries, and SSPRK3.  Only
# its classical reconstruction launches are specialized here so each WENO-Z
# evaluation receives the physical spacing of its reconstruction direction.
base.K = K


class Weno5ZRk3Solver(base.Weno5Rk3Solver):
    def _x_fluxes(self, dt_dx: float) -> None:
        c = self.config
        pz, py, _, _ = c.padded_shape
        self._launch(
            K.normal_x_kernel,
            (pz, py, c.nx + 2),
            [self.q, self.normal_left, self.normal_right, wp.float64(c.dx)],
        )
        for flag in range(1, 5):
            first_lr = 1 if flag in (1, 3) else 2
            second_lr = 1 if flag in (1, 2) else 2
            self._launch(
                K.transverse_y_kernel,
                (pz, c.ny, c.nx + 2),
                [
                    self.normal_left, self.normal_right,
                    self.stage_left, self.stage_right,
                    first_lr, wp.float64(c.dy),
                ],
            )
            self._launch(
                K.transverse_z_kernel,
                (c.nz, c.ny, c.nx + 2),
                [
                    self.stage_left, self.stage_right,
                    self.point_left, self.point_right,
                    second_lr, wp.float64(c.dz),
                ],
            )
            self._launch(
                K.flux_x_kernel,
                (c.nz, c.ny, c.nx + 1),
                [self.point_left, self.point_right, self.flux_x, wp.float64(dt_dx), flag],
            )

    def _y_fluxes(self, dt_dy: float) -> None:
        c = self.config
        pz, _, px, _ = c.padded_shape
        self._launch(
            K.normal_y_kernel,
            (pz, c.ny + 2, px),
            [self.q, self.normal_left, self.normal_right, wp.float64(c.dy)],
        )
        for flag in range(1, 5):
            first_lr = 1 if flag in (1, 3) else 2
            second_lr = 1 if flag in (1, 2) else 2
            self._launch(
                K.transverse_x_kernel,
                (pz, c.ny + 2, c.nx),
                [
                    self.normal_left, self.normal_right,
                    self.stage_left, self.stage_right,
                    first_lr, wp.float64(c.dx),
                ],
            )
            self._launch(
                K.transverse_z_kernel,
                (c.nz, c.ny + 2, c.nx),
                [
                    self.stage_left, self.stage_right,
                    self.point_left, self.point_right,
                    second_lr, wp.float64(c.dz),
                ],
            )
            self._launch(
                K.flux_y_kernel,
                (c.nz, c.ny + 1, c.nx),
                [self.point_left, self.point_right, self.flux_y, wp.float64(dt_dy), flag],
            )

    def _z_fluxes(self, dt_dz: float) -> None:
        c = self.config
        _, py, px, _ = c.padded_shape
        self._launch(
            K.normal_z_kernel,
            (c.nz + 2, py, px),
            [self.q, self.normal_left, self.normal_right, wp.float64(c.dz)],
        )
        for flag in range(1, 5):
            first_lr = 1 if flag in (1, 3) else 2
            second_lr = 1 if flag in (1, 2) else 2
            self._launch(
                K.transverse_y_kernel,
                (c.nz + 2, c.ny, px),
                [
                    self.normal_left, self.normal_right,
                    self.stage_left, self.stage_right,
                    first_lr, wp.float64(c.dy),
                ],
            )
            self._launch(
                K.transverse_x_kernel,
                (c.nz + 2, c.ny, c.nx),
                [
                    self.stage_left, self.stage_right,
                    self.point_left, self.point_right,
                    second_lr, wp.float64(c.dx),
                ],
            )
            self._launch(
                K.flux_z_kernel,
                (c.nz + 1, c.ny, c.nx),
                [self.point_left, self.point_right, self.flux_z, wp.float64(dt_dz), flag],
            )
