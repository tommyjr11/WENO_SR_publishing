from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .binary_io import read_step
from .config import ShockBubbleConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a central z-plane from two 3-D WENO5 result files")
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_mlp, mlp = read_step(args.mlp)
    t_js, classical = read_step(args.classical)
    if mlp.shape != classical.shape:
        raise ValueError(f"shape mismatch: {mlp.shape} != {classical.shape}")
    if t_mlp != t_js:
        raise ValueError(f"time mismatch: {t_mlp} != {t_js}")

    k = mlp.shape[0] // 2
    rho_mlp = mlp[k, ..., 0]
    rho_js = classical[k, ..., 0]
    error = np.abs(rho_mlp - rho_js)
    vmin = float(min(np.min(rho_mlp), np.min(rho_js)))
    vmax = float(max(np.max(rho_mlp), np.max(rho_js)))
    config = ShockBubbleConfig(nx=mlp.shape[2], ny=mlp.shape[1], nz=mlp.shape[0])
    extent = (config.x_start, config.x_end, config.y_start, config.y_end)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.45), constrained_layout=True)
    images = []
    for axis, field, title in zip(
        axes[:2],
        (rho_js, rho_mlp),
        ("WENO5-JS", "WENO5-SR V20@12,250"),
        strict=True,
    ):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(images[0], ax=axes[:2], label=r"Density $\rho$", shrink=0.88)

    error_image = axes[2].imshow(
        error,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="magma",
        aspect="equal",
    )
    axes[2].set_title(r"$|\rho_{\mathrm{SR}}-\rho_{\mathrm{JS}}|$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(error_image, ax=axes[2], label="Absolute density difference", shrink=0.88)
    fig.suptitle(f"Ma=3 shock-bubble, central z-plane, t={t_mlp:.1e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
