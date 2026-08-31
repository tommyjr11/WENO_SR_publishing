from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .binary_io import read_step
from .config import ShockBubbleConfig


def mock_schlieren(rho: np.ndarray, dx: float, dy: float) -> np.ndarray:
    grad_y, grad_x = np.gradient(rho, dy, dx, edge_order=2)
    gradient = np.hypot(grad_x, grad_y)
    return np.exp(-5.0 * gradient / (2000.0 * np.maximum(rho, 1.0e-16)))


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_file(
    path: str | Path,
    out_dir: str | Path,
    scheme_label: str = "WENO7-JS--RK4",
) -> list[Path]:
    path = Path(path)
    out_dir = Path(out_dir)
    time, primitive = read_step(path)
    nz, ny, nx, _ = primitive.shape
    config = ShockBubbleConfig(nx=nx, ny=ny, nz=nz)
    rho = primitive[nz // 2, ..., 0]
    schlieren = mock_schlieren(rho, config.dx, config.dy)
    extent = (config.x_start, config.x_end, config.y_start, config.y_end)
    z_value = config.z_start + (nz // 2 + 0.5) * config.dz

    outputs: list[Path] = []
    for field, cmap, label, stem_name in (
        (rho, "turbo", r"Density $\rho$", "midplane_density"),
        (schlieren, "gray", "Mock-schlieren intensity", "midplane_mock_schlieren"),
    ):
        fig, axis = plt.subplots(figsize=(8.2, 3.7), constrained_layout=True)
        image = axis.imshow(field, origin="lower", extent=extent, interpolation="nearest", cmap=cmap, aspect="equal")
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(rf"{scheme_label}, $z={z_value:.5f}$, $t={time:.1e}$")
        fig.colorbar(image, ax=axis, label=label, shrink=0.9)
        stem = out_dir / stem_name
        save_figure(fig, stem)
        outputs.append(stem.with_suffix(".png"))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.45), constrained_layout=True)
    density_image = axes[0].imshow(rho, origin="lower", extent=extent, interpolation="nearest", cmap="turbo", aspect="equal")
    schlieren_image = axes[1].imshow(schlieren, origin="lower", extent=extent, interpolation="nearest", cmap="gray", aspect="equal")
    for axis, title in zip(axes, (r"Density $\rho$", "Mock schlieren"), strict=True):
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_title(title)
    fig.colorbar(density_image, ax=axes[0], label=r"Density $\rho$", shrink=0.88)
    fig.colorbar(schlieren_image, ax=axes[1], label="Intensity", shrink=0.88)
    fig.suptitle(
        rf"Three-dimensional Ma=3 shock--bubble, {scheme_label}, central $z$ plane, $t={time:.1e}$"
    )
    combined = out_dir / "midplane_density_schlieren"
    save_figure(fig, combined)
    outputs.append(combined.with_suffix(".png"))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the central z-plane of a 3-D WENO7 result")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scheme-label", default="WENO7-JS--RK4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in plot_file(args.input, args.out_dir, scheme_label=args.scheme_label):
        print(output)


if __name__ == "__main__":
    main()
