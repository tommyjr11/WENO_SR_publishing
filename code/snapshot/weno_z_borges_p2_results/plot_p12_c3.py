#!/usr/bin/env python3
"""Plot the isolated WENO7-Z p=1/p=2 C.3 audit results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results.common import primitive
from for_paper_results.make_riemann_figures import DENSITY_CONTOUR_LEVELS


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_beta_h3_p12_audit/riemann/c3/N400"
OUT = ROOT / "raw_beta_h3_p12_audit/figures/c3"


def load(power: int) -> np.ndarray:
    with np.load(RAW / f"weno7_z_p{power}.npz") as data:
        state = np.asarray(data["state"], dtype=np.float64)
    if state.shape != (400, 400, 4):
        raise ValueError(f"unexpected C.3 state shape: {state.shape}")
    return primitive(state)


def main() -> None:
    fields = {power: load(power) for power in (1, 2)}
    p_min = min(float(field[..., 3].min()) for field in fields.values())
    p_max = max(float(field[..., 3].max()) for field in fields.values())
    pressure_levels = np.linspace(p_min, p_max, 241)
    density_levels = DENSITY_CONTOUR_LEVELS["c3"]
    coordinates = (np.arange(400) + 0.5) / 400.0
    xx, yy = np.meshgrid(coordinates, coordinates)
    velocity_max = max(
        float(np.max(np.hypot(field[..., 1], field[..., 2])))
        for field in fields.values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.35, 4.25), constrained_layout=True)
    image = None
    for axis, power in zip(axes, (1, 2)):
        field = fields[power]
        image = axis.contourf(
            xx,
            yy,
            field[..., 3],
            levels=pressure_levels,
            cmap="turbo",
            extend="both",
        )
        axis.contour(
            xx,
            yy,
            field[..., 0],
            levels=density_levels,
            colors="black",
            linewidths=0.30,
        )
        stride = 20
        axis.quiver(
            xx[::stride, ::stride],
            yy[::stride, ::stride],
            field[::stride, ::stride, 1],
            field[::stride, ::stride, 2],
            color="white",
            alpha=0.82,
            pivot="mid",
            scale=max(1.0, 32.0 * velocity_max),
            width=0.0024,
            headwidth=3.2,
            headlength=4.2,
        )
        axis.set_title(rf"WENO7-Z-RK4, $p_Z={power}$")
        axis.set_aspect("equal")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.tick_params(labelsize=8)

    if image is not None:
        fig.colorbar(image, ax=axes, label=r"Pressure $p$", shrink=0.92)
    fig.suptitle(
        r"Two-dimensional Riemann configuration C.3, $400^2$, $t=0.5$"
        "\n"
        r"black lines: equally spaced density contours; white arrows: velocity",
        fontsize=11,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", {"dpi": 350}), ("pdf", {})):
        fig.savefig(OUT / f"c3_weno7_z_p1_p2.{suffix}", bbox_inches="tight", **options)
    plt.close(fig)
    print(OUT / "c3_weno7_z_p1_p2.png")


if __name__ == "__main__":
    main()
