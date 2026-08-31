from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .binary_io import read_step
from .draw_density_line_x_reference import read_step_rho


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88"

X_EXTENT = (0.0, 0.225)
Y_EXTENT = (0.0, 0.089)
Z_EXTENT = (0.0, 0.089)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three 224x88x88 Warp results with the TR1W 1120x440x440 reference"
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/home/ruijie/data_shock_bubble_3D_4th/data_3D"),
    )
    parser.add_argument("--reference-step", type=int, default=7048)
    parser.add_argument("--js", type=Path, default=RUN_DIR / "weno5_js/step_0823.bin")
    parser.add_argument(
        "--fp64",
        type=Path,
        default=RUN_DIR / "weno5_sr_f64_v20_step012250/step_0863.bin",
    )
    parser.add_argument(
        "--fp32",
        type=Path,
        default=RUN_DIR / "weno5_sr_f32_v20_step016500/step_0863.bin",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RUN_DIR / "midplane_density_four_variants_reference.png",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=RUN_DIR / "density_l1_vs_tr1w1120.csv",
    )
    return parser.parse_args()


def block_average(reference: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    nz, ny, nx = reference.shape
    target_nz, target_ny, target_nx = target_shape
    if nz % target_nz or ny % target_ny or nx % target_nx:
        raise ValueError(f"reference shape {reference.shape} is not divisible by {target_shape}")
    rz, ry, rx = nz // target_nz, ny // target_ny, nx // target_nx
    reshaped = reference.reshape(target_nz, rz, target_ny, ry, target_nx, rx)
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float64)


def load_density(path: Path) -> tuple[float, np.ndarray]:
    time, primitive = read_step(path)
    density = np.array(primitive[..., 0], dtype=np.float64, copy=True)
    del primitive
    return time, density


def l1_metrics(
    density: np.ndarray,
    reference: np.ndarray,
    plane_index: int,
) -> dict[str, float]:
    error = np.abs(density - reference)
    plane_error = error[plane_index]
    reference_plane = reference[plane_index]

    mean_l1 = float(np.mean(error, dtype=np.float64))
    relative_l1 = float(np.sum(error, dtype=np.float64) / np.sum(np.abs(reference), dtype=np.float64))
    plane_mean_l1 = float(np.mean(plane_error, dtype=np.float64))
    plane_relative_l1 = float(
        np.sum(plane_error, dtype=np.float64)
        / np.sum(np.abs(reference_plane), dtype=np.float64)
    )
    return {
        "volume_mean_l1": mean_l1,
        "volume_integrated_l1": mean_l1
        * (X_EXTENT[1] - X_EXTENT[0])
        * (Y_EXTENT[1] - Y_EXTENT[0])
        * (Z_EXTENT[1] - Z_EXTENT[0]),
        "volume_relative_l1": relative_l1,
        "midplane_mean_l1": plane_mean_l1,
        "midplane_integrated_l1": plane_mean_l1
        * (X_EXTENT[1] - X_EXTENT[0])
        * (Y_EXTENT[1] - Y_EXTENT[0]),
        "midplane_relative_l1": plane_relative_l1,
        "max_abs_error": float(np.max(error)),
    }


def save_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    print(f"Reading TR1W reference step {args.reference_step} from {args.reference_dir}")
    reference_density, reference_time = read_step_rho(args.reference_step, args.reference_dir)
    if reference_density.shape != (440, 440, 1120):
        raise ValueError(f"unexpected reference shape: {reference_density.shape}")

    method_specs = (
        ("WENO5-JS", args.js, "#6F6F6F"),
        ("WENO5-SR FP64", args.fp64, "#0072B2"),
        ("WENO5-SR MLP-FP32", args.fp32, "#D55E00"),
    )
    times: list[float] = []
    densities: list[np.ndarray] = []
    for label, path, _ in method_specs:
        time, density = load_density(path)
        if density.shape != (88, 88, 224):
            raise ValueError(f"{label}: unexpected shape {density.shape}")
        times.append(time)
        densities.append(density)

    if max(times) - min(times) > 1.0e-15:
        raise ValueError(f"candidate time mismatch: {times}")

    averaged_reference = block_average(reference_density, densities[0].shape)
    coarse_k = densities[0].shape[0] // 2
    refinement = reference_density.shape[0] // densities[0].shape[0]
    fine_k = coarse_k * refinement + refinement // 2
    coarse_z = Z_EXTENT[0] + (coarse_k + 0.5) * (
        (Z_EXTENT[1] - Z_EXTENT[0]) / densities[0].shape[0]
    )

    rows: list[dict[str, object]] = []
    metrics_by_label: dict[str, dict[str, float]] = {}
    for (label, path, _), density, time in zip(method_specs, densities, times, strict=True):
        metrics = l1_metrics(density, averaged_reference, coarse_k)
        metrics_by_label[label] = metrics
        rows.append(
            {
                "method": label,
                "grid": "224x88x88",
                "candidate_time": f"{time:.16e}",
                "reference": "TR1W-4 1120x440x440 averaged 5x5x5",
                "reference_time": f"{reference_time:.16e}",
                "volume_mean_l1": f"{metrics['volume_mean_l1']:.10e}",
                "volume_integrated_l1": f"{metrics['volume_integrated_l1']:.10e}",
                "volume_relative_l1": f"{metrics['volume_relative_l1']:.10e}",
                "midplane_mean_l1": f"{metrics['midplane_mean_l1']:.10e}",
                "midplane_integrated_l1": f"{metrics['midplane_integrated_l1']:.10e}",
                "midplane_relative_l1": f"{metrics['midplane_relative_l1']:.10e}",
                "max_abs_error": f"{metrics['max_abs_error']:.10e}",
                "source": str(path),
            }
        )
    save_metrics(args.metrics, rows)

    reference_plane = reference_density[fine_k]
    coarse_reference_plane = averaged_reference[coarse_k]
    density_planes = [density[coarse_k] for density in densities]
    error_planes = [np.abs(plane - coarse_reference_plane) for plane in density_planes]

    density_fields = [reference_plane, *density_planes]
    density_min = float(min(np.min(field) for field in density_fields))
    density_max = float(max(np.max(field) for field in density_fields))
    error_max = float(max(np.max(field) for field in error_planes))

    fig = plt.figure(figsize=(12.8, 4.75), dpi=300)
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.045),
        left=0.055,
        right=0.965,
        bottom=0.115,
        top=0.835,
        wspace=0.16,
        hspace=0.22,
    )
    top_axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
    bar_axis = fig.add_subplot(grid[1, 0])
    error_axes = [fig.add_subplot(grid[1, column]) for column in range(1, 4)]
    density_cax = fig.add_subplot(grid[0, 4])
    error_cax = fig.add_subplot(grid[1, 4])

    extent = (*X_EXTENT, *Y_EXTENT)
    density_titles = (
        r"(a) TR1W reference, $1120\times440\times440$",
        r"(b) WENO5-JS, $224\times88\times88$",
        r"(c) WENO5-SR FP64, $224\times88\times88$",
        r"(d) WENO5-SR MLP-FP32, $224\times88\times88$",
    )
    density_image = None
    for index, (axis, field, title) in enumerate(
        zip(top_axes, density_fields, density_titles, strict=True)
    ):
        density_image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="viridis",
            vmin=density_min,
            vmax=density_max,
            aspect="equal",
            rasterized=True,
        )
        axis.set_title(title, fontsize=9.2, pad=5)
        axis.set_xlabel(r"$x$ (m)")
        if index == 0:
            axis.set_ylabel(r"$y$ (m)")
        else:
            axis.tick_params(labelleft=False)
    fig.colorbar(density_image, cax=density_cax, label=r"Density $\rho$")

    labels = [spec[0] for spec in method_specs]
    colors = [spec[2] for spec in method_specs]
    relative_percent = [100.0 * metrics_by_label[label]["volume_relative_l1"] for label in labels]
    y_positions = np.arange(len(labels))
    bars = bar_axis.barh(y_positions, relative_percent, color=colors, height=0.58)
    bar_axis.set_yticks(y_positions, ("JS", "SR FP64", "SR MLP-FP32"))
    bar_axis.invert_yaxis()
    bar_axis.set_xlabel(r"Volume relative $L_1$ (\%)")
    bar_axis.set_title(r"(e) Density $L_1$ vs. averaged reference", fontsize=9.2, pad=5)
    bar_axis.grid(axis="x", linestyle=(0, (2, 2)), linewidth=0.45, alpha=0.55)
    bar_axis.set_axisbelow(True)
    for bar, value in zip(bars, relative_percent, strict=True):
        bar_axis.text(
            value + 0.04 * max(relative_percent),
            bar.get_y() + 0.5 * bar.get_height(),
            f"{value:.3f}%",
            va="center",
            ha="left",
            fontsize=8.2,
        )
    bar_axis.set_xlim(0.0, 1.22 * max(relative_percent))

    error_titles = (
        r"(f) $|\rho_{\mathrm{JS}}-\bar{\rho}_{\mathrm{ref}}|$",
        r"(g) $|\rho_{\mathrm{FP64}}-\bar{\rho}_{\mathrm{ref}}|$",
        r"(h) $|\rho_{\mathrm{FP32}}-\bar{\rho}_{\mathrm{ref}}|$",
    )
    error_image = None
    for index, (axis, field, title) in enumerate(
        zip(error_axes, error_planes, error_titles, strict=True)
    ):
        error_image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
            aspect="equal",
            rasterized=True,
        )
        axis.set_title(title, fontsize=9.2, pad=5)
        axis.set_xlabel(r"$x$ (m)")
        if index == 0:
            axis.set_ylabel(r"$y$ (m)")
        else:
            axis.tick_params(labelleft=False)
    fig.colorbar(error_image, cax=error_cax, label=r"Absolute density error")

    fig.suptitle(
        (
            r"Ma = 3 shock-bubble interaction, central coarse-grid plane "
            rf"$z={coarse_z:.6f}$ m, $t={times[0]:.1e}$"
        ),
        fontsize=12.0,
        y=0.965,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"reference_time={reference_time:.16e}")
    print(f"candidate_time={times[0]:.16e}")
    print(f"coarse_k={coarse_k} fine_k={fine_k} z={coarse_z:.10e}")
    for row in rows:
        print(
            f"{row['method']}: volume_mean_L1={row['volume_mean_l1']} "
            f"volume_relative_L1={row['volume_relative_l1']} "
            f"midplane_mean_L1={row['midplane_mean_l1']}"
        )
    print(args.out)
    print(args.metrics)


if __name__ == "__main__":
    main()
