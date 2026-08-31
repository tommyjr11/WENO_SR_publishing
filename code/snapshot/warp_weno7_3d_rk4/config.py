from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShockBubbleConfig:
    """Configuration for the Ma=3 three-dimensional shock--bubble problem."""

    nx: int = 224
    ny: int = 88
    nz: int = 88
    ghost: int = 4

    x_start: float = 0.0
    x_end: float = 0.225
    y_start: float = 0.0
    y_end: float = 0.089
    z_start: float = 0.0
    z_end: float = 0.089

    gamma: float = 1.4
    cfl: float = 0.25
    t_start: float = 0.0
    t_end: float = 2.0e-5

    @property
    def dx(self) -> float:
        return (self.x_end - self.x_start) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_end - self.y_start) / self.ny

    @property
    def dz(self) -> float:
        return (self.z_end - self.z_start) / self.nz

    @property
    def padded_shape(self) -> tuple[int, int, int, int]:
        g = self.ghost
        return (self.nz + 2 * g, self.ny + 2 * g, self.nx + 2 * g, 5)

    def validate(self) -> None:
        if min(self.nx, self.ny, self.nz) < 7:
            raise ValueError("WENO7 requires at least seven cells per direction")
        if self.ghost != 4:
            raise ValueError("the WENO7 port requires four ghost cells")
        if self.gamma != 1.4:
            raise ValueError("the source implementation hard-codes gamma=1.4")
        if self.cfl <= 0.0:
            raise ValueError("CFL must be positive")
        if self.t_end < self.t_start:
            raise ValueError("t_end must not precede t_start")

