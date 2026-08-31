from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .binary_io import read_step
from .config import ShockBubbleConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot central z-plane density for JS, FP64 MLP, and mixed-FP32 MLP"
    )
    parser.add_argument("--js", type=Path, required=True)
    parser.add_argument("--fp64", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [read_step(path) for path in (args.js, args.fp64, args.fp32)]
    times = [record[0] for record in records]
    fields = [record[1] for record in records]
    if len({field.shape for field in fields}) != 1:
        raise ValueError(f"shape mismatch: {[field.shape for field in fields]}")
    if len(set(times)) != 1:
        raise ValueError(f"time mismatch: {times}")

    k = fields[0].shape[0] // 2
    density = [field[k, ..., 0] for field in fields]
    differences = [
        np.abs(density[1] - density[0]),
        np.abs(density[2] - density[0]),
        np.abs(density[2] - density[1]),
    ]
    density_min = float(min(np.min(field) for field in density))
    density_max = float(max(np.max(field) for field in density))
    difference_max = float(max(np.max(field) for field in differences))
    config = ShockBubbleConfig(
        nx=fields[0].shape[2],
        ny=fields[0].shape[1],
        nz=fields[0].shape[0],
    )
    extent = (config.x_start, config.x_end, config.y_start, config.y_end)

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.2), constrained_layout=True)
    titles = ("WENO5-JS", "WENO5-SR FP64", "WENO5-SR MLP-FP32")
    density_images = []
    for axis, field, title in zip(axes[0], density, titles, strict=True):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
            vmin=density_min,
            vmax=density_max,
            aspect="equal",
        )
        density_images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(
        density_images[0],
        ax=axes[0].tolist(),
        label=r"Density $\rho$",
        shrink=0.90,
    )

    difference_titles = (
        r"$|\rho_{\mathrm{FP64}}-\rho_{\mathrm{JS}}|$",
        r"$|\rho_{\mathrm{FP32}}-\rho_{\mathrm{JS}}|$",
        r"$|\rho_{\mathrm{FP32}}-\rho_{\mathrm{FP64}}|$",
    )
    difference_images = []
    for axis, field, title in zip(axes[1], differences, difference_titles, strict=True):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=difference_max,
            aspect="equal",
        )
        difference_images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(
        difference_images[0],
        ax=axes[1].tolist(),
        label="Absolute density difference",
        shrink=0.90,
    )
    fig.suptitle(
        f"Ma=3 shock-bubble, 224x88x88, CFL=0.25, central z-plane, t={times[0]:.1e}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
