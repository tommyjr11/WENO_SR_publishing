#!/usr/bin/env python3
"""Isolated multi-method adaptation of data_shock_bubble_3D_4th/draw_cut.py."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from warp_weno5_3d_rk3.binary_io import read_step

from .render_shockbubble_3d import (
    METHOD_IDS,
    METHOD_LABELS,
    PHYS_X_MAX,
    PHYS_X_MIN,
    PHYS_Y_MAX,
    PHYS_Y_MIN,
    PHYS_Z_MAX,
    PHYS_Z_MIN,
    RHO_THRESHOLD,
    WINDOW_SIZE,
    method_sources,
    nonblack_bounds,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "figures/shockbubble_3d/N224x88x88/cutaway_renders"
REFERENCE_RENDER = Path(
    "/home/ruijie/data_shock_bubble_3D_4th/frames_cutaway_dual/cut_dual_007048.png"
)
X_MIN_CROP, X_MAX_CROP = 0.04, 0.16


def render_cutaway(rho: np.ndarray, output: Path) -> float:
    nz, ny, nx_full = rho.shape
    dx = (PHYS_X_MAX - PHYS_X_MIN) / nx_full
    dy = (PHYS_Y_MAX - PHYS_Y_MIN) / ny
    dz = (PHYS_Z_MAX - PHYS_Z_MIN) / nz
    i_start = max(0, int((X_MIN_CROP - PHYS_X_MIN) / dx))
    i_end = min(nx_full, int((X_MAX_CROP - PHYS_X_MIN) / dx))
    rho_crop = np.asarray(rho[:, :, i_start:i_end], dtype=np.float32)

    gz, gy, gx = np.gradient(rho_crop, dz, dy, dx)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)
    grad_blue = np.where(rho_crop < RHO_THRESHOLD, grad_mag, 0.0).astype(np.float32)
    grad_orange = np.where(rho_crop >= RHO_THRESHOLD, grad_mag, 0.0).astype(np.float32)
    contrast = float(
        max(np.percentile(grad_blue[::5], 98), np.percentile(grad_orange[::5], 98))
    )

    grid = pv.ImageData()
    grid.dimensions = grad_blue.shape[::-1]
    grid.origin = (PHYS_X_MIN + i_start * dx, PHYS_Y_MIN, PHYS_Z_MIN)
    grid.spacing = (dx, dy, dz)
    grid.point_data["grad_blue"] = grad_blue.ravel(order="C")
    grid.point_data["grad_orange"] = grad_orange.ravel(order="C")
    dim_x, dim_y, dim_z = grid.dimensions
    grid_half = grid.extract_subset((0, dim_x - 1, 0, dim_y - 1, 0, dim_z // 2))

    opacity_blue = [0.00, 0.0, 0.12, 0.0, 0.15, 0.15, 0.70, 0.355, 1.00, 0.1]
    opacity_orange = [
        0.00, 0.0, 0.12, 0.0, 0.15, 0.0030, 0.70, 0.00305, 1.00, 0.002
    ]
    average_spacing = (dx + dy + dz) / 3.0

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("black")
    plotter.add_volume(
        grid_half,
        scalars="grad_blue",
        cmap="Blues",
        clim=(0.0, contrast),
        opacity=opacity_blue,
        shade=False,
        ambient=1.4,
        diffuse=0.8,
        specular=0.2,
        opacity_unit_distance=average_spacing * 2.0,
        show_scalar_bar=False,
    )
    plotter.add_volume(
        grid_half,
        scalars="grad_orange",
        cmap="Oranges",
        clim=(0.0, contrast),
        opacity=opacity_orange,
        shade=False,
        ambient=1.4,
        diffuse=0.8,
        specular=0.2,
        opacity_unit_distance=average_spacing,
        show_scalar_bar=False,
    )
    plotter.camera_position = [
        (0.087288, 0.0445, 0.2845),
        (0.087288, 0.0445, 0.0445),
        (0.0, 1.0, 0.0),
    ]
    plotter.camera.zoom(1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.render()
    plotter.screenshot(str(output))
    plotter.close()
    return contrast


def compose(output_dir: Path) -> Path:
    ids = (*METHOD_IDS, "reference")
    images = [plt.imread(output_dir / f"{method_id}.png") for method_id in ids]
    y0, y1, x0, x1 = nonblack_bounds(images)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
    })
    fig = plt.figure(figsize=(12.8, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 1, 0.10))
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(4)]
    for axis, method_id, image in zip(axes, ids, images):
        axis.imshow(image[y0:y1, x0:x1])
        axis.set_title(METHOD_LABELS[method_id], pad=5)
        axis.set_facecolor("black")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#333333")

    key_axis = fig.add_subplot(grid[:, 4])
    ramp = np.linspace(0.0, 1.0, 256)
    blue = matplotlib.colormaps["Blues"](ramp)[:, :3]
    orange = matplotlib.colormaps["Oranges"](ramp)[:, :3]
    key_axis.imshow(
        np.stack((blue, orange), axis=1),
        origin="lower",
        aspect="auto",
        extent=(0.0, 2.0, 0.0, 1.0),
    )
    key_axis.set_xticks((0.5, 1.5), (r"$\rho<4.5$", r"$\rho\geq4.5$"), rotation=90)
    key_axis.set_ylabel(r"Panel-normalized gradient magnitude $|\nabla\rho|$")
    key_axis.yaxis.set_label_position("right")
    key_axis.yaxis.tick_right()
    key_axis.tick_params(direction="in", length=3)
    fig.suptitle(
        r"Three-dimensional Ma=3 shock--bubble: half-domain cutaway at $t=10^{-4}$",
        fontsize=12.5,
    )
    output = output_dir.parent / "cutaway_all_methods_with_reference"
    fig.savefig(output.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=(*METHOD_IDS, "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_copy = OUTPUT_DIR / "reference.png"
    if args.force or not reference_copy.is_file():
        if not REFERENCE_RENDER.is_file():
            raise FileNotFoundError(REFERENCE_RENDER)
        shutil.copy2(REFERENCE_RENDER, reference_copy)

    contrast_rows = []
    if not args.compose_only:
        sources = method_sources()
        selected = METHOD_IDS if args.method == "all" else (args.method,)
        for method_id in selected:
            output = OUTPUT_DIR / f"{method_id}.png"
            if output.is_file() and not args.force:
                print(f"skip existing {output}")
                continue
            time, primitive = read_step(sources[method_id])
            if primitive.shape != (88, 88, 224, 5):
                raise ValueError(f"{method_id}: unexpected shape {primitive.shape}")
            print(f"render {method_id}: t={time:.8e}, source={sources[method_id]}")
            contrast = render_cutaway(primitive[..., 0].astype(np.float32), output)
            contrast_rows.append((method_id, contrast))
            print(f"{output} contrast={contrast:.8e}")

    if contrast_rows:
        table = OUTPUT_DIR / "auto_contrast.csv"
        previous = {}
        if table.is_file():
            with table.open(newline="", encoding="ascii") as stream:
                previous = {row["method"]: row["gradient_p98"] for row in csv.DictReader(stream)}
        previous.update({method: f"{contrast:.16e}" for method, contrast in contrast_rows})
        with table.open("w", newline="", encoding="ascii") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "gradient_p98"))
            for method_id in METHOD_IDS:
                if method_id in previous:
                    writer.writerow((method_id, previous[method_id]))

    if args.method == "all" or args.compose_only:
        print(compose(OUTPUT_DIR))


if __name__ == "__main__":
    main()
