#!/usr/bin/env python3
"""Isolated multi-method adaptation of data_shock_bubble_3D_4th/draw_time.py."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from warp_weno5_3d_rk3.binary_io import read_step

from .plot_shockbubble_3d import METHODS, latest_binary


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUTPUT_DIR = ROOT / "figures/shockbubble_3d/N224x88x88/volume_renders"
REFERENCE_RENDER = Path(
    "/home/ruijie/data_shock_bubble_3D_4th/renders/step_7048_x0.0000_0.1250.png"
)

PHYS_X_MIN, PHYS_X_MAX = 0.0, 0.225
PHYS_Y_MIN, PHYS_Y_MAX = 0.0, 0.089
PHYS_Z_MIN, PHYS_Z_MAX = 0.0, 0.089
RHO_THRESHOLD = 4.5
WINDOW_SIZE = (1840, 1200)
X_MIN_CROP, X_MAX_CROP = 0.0, 0.125
GRAD_MAX = 1500.0

METHOD_IDS = (
    "weno5_js",
    "weno5_z",
    "weno5_sr_f64",
    "weno5_sr_f32",
    "weno7_js",
    "weno7_z",
    "weno7_sr_f64",
)
METHOD_LABELS = {
    "weno5_js": r"WENO5-JS-RK3, $224\times88^2$",
    "weno5_z": r"WENO5-Z-RK3, $224\times88^2$",
    "weno5_sr_f64": r"WENO5-SR-RK3, $224\times88^2$",
    "weno5_sr_f32": r"WENO5-SR-FP32-RK3, $224\times88^2$",
    "weno7_js": r"WENO7-JS-RK4, $224\times88^2$",
    "weno7_z": r"WENO7-Z-RK4, $224\times88^2$",
    "weno7_sr_f64": r"WENO7-SR-RK4, $224\times88^2$",
    "reference": r"TR1W reference, $1120\times440^2$",
}


def method_sources() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for method_id, (_, path, manifest_path) in zip(METHOD_IDS, METHODS):
        if manifest_path is not None:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            if not manifest.get("complete", False):
                raise RuntimeError(f"{method_id}: incomplete run")
        binary = latest_binary(path)
        if binary is None:
            raise FileNotFoundError(f"{method_id}: no step binary under {path}")
        sources[method_id] = binary
    return sources


def render_density(rho: np.ndarray, output: Path, grad_max: float) -> None:
    nz, ny, nx_full = rho.shape
    dx = (PHYS_X_MAX - PHYS_X_MIN) / nx_full
    dy = (PHYS_Y_MAX - PHYS_Y_MIN) / ny
    dz = (PHYS_Z_MAX - PHYS_Z_MIN) / nz
    i_start = max(0, int((X_MIN_CROP - PHYS_X_MIN) / dx))
    i_end = min(nx_full, int((X_MAX_CROP - PHYS_X_MIN) / dx))
    if i_start >= i_end:
        raise ValueError("invalid x crop")

    rho_crop = np.asarray(rho[:, :, i_start:i_end], dtype=np.float32)
    gz, gy, gx = np.gradient(rho_crop, dz, dy, dx)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)
    grad_blue = np.where(rho_crop < RHO_THRESHOLD, grad_mag, 0.0).astype(np.float32)
    grad_orange = np.where(rho_crop >= RHO_THRESHOLD, grad_mag, 0.0).astype(np.float32)

    grid = pv.ImageData()
    grid.dimensions = grad_blue.shape[::-1]
    grid.origin = (PHYS_X_MIN + i_start * dx, PHYS_Y_MIN, PHYS_Z_MIN)
    grid.spacing = (dx, dy, dz)
    grid.point_data["grad_blue"] = grad_blue.ravel(order="C")
    grid.point_data["grad_orange"] = grad_orange.ravel(order="C")

    opacity_blue = [0.00, 0.0, 0.12, 0.0, 0.15, 0.35, 0.70, 0.555, 1.00, 0.25]
    opacity_orange = [
        0.00, 0.0, 0.12, 0.0, 0.15, 0.0015, 0.70, 0.00155, 1.00, 0.001
    ]
    average_spacing = (dx + dy + dz) / 3.0

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("black")
    plotter.add_volume(
        grid,
        scalars="grad_blue",
        cmap="Blues",
        clim=(0.0, grad_max),
        opacity=opacity_blue,
        shade=False,
        ambient=0.9,
        diffuse=0.9,
        specular=0.2,
        specular_power=20,
        opacity_unit_distance=average_spacing * 0.2,
        show_scalar_bar=False,
    )
    plotter.add_volume(
        grid,
        scalars="grad_orange",
        cmap="Oranges",
        clim=(0.0, grad_max),
        opacity=opacity_orange,
        shade=False,
        ambient=0.9,
        diffuse=0.9,
        specular=0.2,
        specular_power=20,
        opacity_unit_distance=average_spacing * 2.0,
        show_scalar_bar=False,
    )
    plotter.camera_position = [
        (0.087288, -0.30, 0.0445),
        (0.087288, 0.0445, 0.0445),
        (0.0, 1.0, 1.0),
    ]
    plotter.camera.zoom(1.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.render()
    plotter.screenshot(str(output))
    plotter.close()


def nonblack_bounds(images: list[np.ndarray]) -> tuple[int, int, int, int]:
    height = min(image.shape[0] for image in images)
    width = min(image.shape[1] for image in images)
    union = np.zeros((height, width), dtype=bool)
    for image in images:
        rgb = image[:height, :width, :3]
        union |= np.max(rgb, axis=2) > 0.025
    rows, columns = np.nonzero(union)
    if rows.size == 0:
        return 0, height, 0, width
    pad_y = max(8, int(0.035 * (rows.max() - rows.min() + 1)))
    pad_x = max(8, int(0.035 * (columns.max() - columns.min() + 1)))
    return (
        max(0, int(rows.min()) - pad_y),
        min(height, int(rows.max()) + pad_y + 1),
        max(0, int(columns.min()) - pad_x),
        min(width, int(columns.max()) + pad_x + 1),
    )


def compose(output_dir: Path, grad_max: float) -> Path:
    ids = (*METHOD_IDS, "reference")
    paths = [output_dir / f"{method_id}.png" for method_id in ids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing renders: {missing}")
    images = [plt.imread(path) for path in paths]
    y0, y1, x0, x1 = nonblack_bounds(images)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
    })
    fig = plt.figure(figsize=(13.0, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 1, 0.12))
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
    dual_key = np.stack((blue, orange), axis=1)
    key_axis.imshow(
        dual_key,
        origin="lower",
        aspect="auto",
        extent=(0.0, 2.0, 0.0, grad_max),
    )
    key_axis.set_xticks((0.5, 1.5), (r"$\rho<4.5$", r"$\rho\geq4.5$"), rotation=90)
    key_axis.set_ylabel(r"Clipped gradient magnitude $|\nabla\rho|$")
    key_axis.yaxis.set_label_position("right")
    key_axis.yaxis.tick_right()
    key_axis.tick_params(direction="in", length=3)
    fig.suptitle(
        r"Three-dimensional Ma=3 shock--bubble interaction at $t=10^{-4}$",
        fontsize=12.5,
    )
    output = output_dir.parent / "volume_render_all_methods_with_reference"
    fig.savefig(output.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=(*METHOD_IDS, "all"), default="all")
    parser.add_argument("--grad-max", type=float, default=GRAD_MAX)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_copy = OUTPUT_DIR / "reference.png"
    if args.force or not reference_copy.is_file():
        if not REFERENCE_RENDER.is_file():
            raise FileNotFoundError(REFERENCE_RENDER)
        shutil.copy2(REFERENCE_RENDER, reference_copy)

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
            render_density(primitive[..., 0].astype(np.float32), output, args.grad_max)
            print(output)

    if args.method == "all" or args.compose_only:
        print(compose(OUTPUT_DIR, args.grad_max))


if __name__ == "__main__":
    main()
