from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .binary_io import read_step
from .config import ShockBubbleConfig
from .plot_midplane import mock_schlieren, save_figure


def plot_comparison(
    js_path: str | Path,
    sr_path: str | Path,
    out_stem: str | Path,
) -> tuple[Path, Path]:
    js_time, js_primitive = read_step(js_path)
    sr_time, sr_primitive = read_step(sr_path)
    if js_primitive.shape != sr_primitive.shape:
        raise ValueError(f"shape mismatch: JS={js_primitive.shape}, SR={sr_primitive.shape}")
    if not np.isclose(js_time, sr_time, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"time mismatch: JS={js_time:.17g}, SR={sr_time:.17g}")

    nz, ny, nx, components = js_primitive.shape
    if components != 5:
        raise ValueError(f"expected five primitive components, got {components}")
    config = ShockBubbleConfig(nx=nx, ny=ny, nz=nz)
    extent = (config.x_start, config.x_end, config.y_start, config.y_end)
    midplane = nz // 2
    js_rho = js_primitive[midplane, ..., 0]
    sr_rho = sr_primitive[midplane, ..., 0]
    js_schlieren = mock_schlieren(js_rho, config.dx, config.dy)
    sr_schlieren = mock_schlieren(sr_rho, config.dx, config.dy)

    density_min = float(min(js_rho.min(), sr_rho.min()))
    density_max = float(max(js_rho.max(), sr_rho.max()))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.3), constrained_layout=True)
    density_images = []
    for axis, field, title in zip(
        axes[0],
        (js_rho, sr_rho),
        ("WENO7-JS--RK4", "WENO7-SR--RK4 (FP64)"),
        strict=True,
    ):
        density_images.append(
            axis.imshow(
                field,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap="turbo",
                vmin=density_min,
                vmax=density_max,
                aspect="equal",
            )
        )
        axis.set_title(title)

    schlieren_images = []
    for axis, field in zip(axes[1], (js_schlieren, sr_schlieren), strict=True):
        schlieren_images.append(
            axis.imshow(
                field,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
            )
        )

    for row in axes:
        for axis in row:
            axis.set_xlabel(r"$x$")
            axis.set_ylabel(r"$y$")
    density_colorbar = fig.colorbar(
        density_images[0], ax=axes[0, :], label=r"Density $\rho$", shrink=0.88, pad=0.015
    )
    density_colorbar.ax.tick_params(labelsize=9)
    schlieren_colorbar = fig.colorbar(
        schlieren_images[0], ax=axes[1, :], label="Mock-schlieren intensity", shrink=0.88, pad=0.015
    )
    schlieren_colorbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        rf"Three-dimensional Ma=3 shock--bubble, central $z$ plane, "
        rf"$N_x\times N_y\times N_z={nx}\times{ny}\times{nz}$, $t={js_time:.1e}$"
    )

    out_stem = Path(out_stem)
    save_figure(fig, out_stem)
    return out_stem.with_suffix(".png"), out_stem.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare central planes from WENO7-JS and WENO7-SR")
    parser.add_argument("--js", type=Path, required=True)
    parser.add_argument("--sr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in plot_comparison(args.js, args.sr, args.out):
        print(output)


if __name__ == "__main__":
    main()
