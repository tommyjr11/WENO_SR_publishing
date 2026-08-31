#!/usr/bin/env python3
"""Create shock--bubble mock-schlieren and reference line-cut figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results import config


METHODS = config.EULER_METHODS


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def scientific_math(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def load_result(path: Path) -> dict[str, object]:
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return {
            "state": np.asarray(data["state"], dtype=np.float64),
            "x": np.asarray(data["x"], dtype=np.float64),
            "y": np.asarray(data["y"], dtype=np.float64),
            "metadata": metadata,
        }


def mock_schlieren(rho: np.ndarray, dx: float, dy: float) -> np.ndarray:
    grad_y, grad_x = np.gradient(rho, dy, dx, edge_order=2)
    grad = np.hypot(grad_x, grad_y)
    return np.exp(-5.0 * grad / (2000.0 * np.maximum(rho, 1.0e-16)))


def interpolate_horizontal(
    field: np.ndarray, x: np.ndarray, y: np.ndarray, y_value: float
) -> tuple[np.ndarray, np.ndarray]:
    upper = int(np.searchsorted(y, y_value))
    if upper <= 0:
        return x, field[0].copy()
    if upper >= y.size:
        return x, field[-1].copy()
    lower = upper - 1
    fraction = (y_value - y[lower]) / (y[upper] - y[lower])
    return x, (1.0 - fraction) * field[lower] + fraction * field[upper]


def interpolate_vertical(
    field: np.ndarray, x: np.ndarray, y: np.ndarray, x_value: float
) -> tuple[np.ndarray, np.ndarray]:
    upper = int(np.searchsorted(x, x_value))
    if upper <= 0:
        return y, field[:, 0].copy()
    if upper >= x.size:
        return y, field[:, -1].copy()
    lower = upper - 1
    fraction = (x_value - x[lower]) / (x[upper] - x[lower])
    return y, (1.0 - fraction) * field[:, lower] + fraction * field[:, upper]


def conservative_restrict_axis(
    field: np.ndarray,
    source_edges: np.ndarray,
    target_edges: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Restrict cell averages by exact one-dimensional overlap weights."""
    moved = np.moveaxis(np.asarray(field, dtype=np.float64), axis, -1)
    output = np.zeros(moved.shape[:-1] + (target_edges.size - 1,), dtype=np.float64)
    source_cell = 0
    tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, abs(float(source_edges[-1])), abs(float(target_edges[-1])),
    )
    for target_cell in range(target_edges.size - 1):
        left = float(target_edges[target_cell])
        right = float(target_edges[target_cell + 1])
        width = right - left
        while (
            source_cell + 1 < source_edges.size - 1
            and source_edges[source_cell + 1] <= left + tolerance
        ):
            source_cell += 1
        current = source_cell
        while current < source_edges.size - 1 and source_edges[current] < right - tolerance:
            overlap = min(right, float(source_edges[current + 1])) - max(
                left, float(source_edges[current]),
            )
            if overlap > 0.0:
                output[..., target_cell] += moved[..., current] * (overlap / width)
            current += 1
    return np.moveaxis(output, -1, axis)


def restrict_reference_density(
    reference: dict[str, object], target: dict[str, object],
) -> np.ndarray:
    """Conservatively project a fine FV density field onto the target FV mesh."""
    reference_metadata = reference["metadata"]
    target_metadata = target["metadata"]
    reference_rho = reference["state"][..., 0]
    reference_x_edges = np.linspace(
        0.0, float(reference_metadata["x_length"]),
        int(reference_metadata["nx"]) + 1,
    )
    reference_y_edges = np.linspace(
        0.0, float(reference_metadata["y_length"]),
        int(reference_metadata["ny"]) + 1,
    )
    target_x_edges = np.linspace(
        0.0, float(target_metadata["x_length"]),
        int(target_metadata["nx"]) + 1,
    )
    target_y_edges = np.linspace(
        0.0, float(target_metadata["y_length"]),
        int(target_metadata["ny"]) + 1,
    )
    restricted_x = conservative_restrict_axis(
        reference_rho, reference_x_edges, target_x_edges, axis=1,
    )
    return conservative_restrict_axis(
        restricted_x, reference_y_edges, target_y_edges, axis=0,
    )


def error_metrics(numerical: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = numerical - reference
    return {
        "l1": float(np.mean(np.abs(delta))),
        "l2": float(np.sqrt(np.mean(delta * delta))),
        "linf": float(np.max(np.abs(delta))),
    }


def make_schlieren_grid(
    results: dict[str, dict[str, object]],
    fields: dict[str, np.ndarray],
    figure_dir: Path,
    *,
    stem: str,
    title: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    reference: dict[str, object] | None = None,
    reference_label: str | None = None,
) -> None:
    keys = [key for key in METHODS if key in results]
    if not keys:
        return
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
    })
    panels = [
        (config.METHODS[key].label, results[key], fields[key])
        for key in keys
    ]
    if reference is not None:
        reference_metadata = reference["metadata"]
        label = reference_label or (
            "Reference\n"
            f"${int(reference_metadata['nx'])}\\times"
            f"{int(reference_metadata['ny'])}$"
        )
        panels.append((
            label,
            reference,
            mock_schlieren(
                reference["state"][..., 0],
                float(reference_metadata["dx"]),
                float(reference_metadata["dy"]),
            ),
        ))
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10.2, 2.25 * nrows + 0.55),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    image = None
    for ax, (label, result, field) in zip(axes, panels):
        metadata = result["metadata"]
        image = ax.imshow(
            field,
            origin="lower",
            extent=(0.0, float(metadata["x_length"]), 0.0, float(metadata["y_length"])),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(label)
        ax.set_ylabel("$y$")
        ax.set_xlabel("$x$")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    assert image is not None
    colorbar = fig.colorbar(
        image, ax=[ax for ax in axes[:len(panels)]],
        label="Mock-schlieren intensity", shrink=0.92,
    )
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=11)
    save_figure(fig, figure_dir / stem)


def make_schlieren(
    results: dict[str, dict[str, object]], figure_dir: Path,
    vortex_window: tuple[float, float, float, float],
    reference: dict[str, object] | None = None,
    reference_label: str | None = None,
) -> None:
    keys = [key for key in METHODS if key in results]
    if not keys:
        return
    metadata = results[keys[0]]["metadata"]
    fields = {
        key: mock_schlieren(
            results[key]["state"][..., 0],
            float(results[key]["metadata"]["dx"]),
            float(results[key]["metadata"]["dy"]),
        )
        for key in keys
    }
    common = (
        "Shock--helium-bubble interaction at "
        f"$t={scientific_math(float(metadata['t']))}$: "
        f"${int(metadata['nx'])}\\times{int(metadata['ny'])}$, "
        f"CFL $={float(metadata['cfl']):.3g}$, characteristic HLLC"
    )
    make_schlieren_grid(
        results,
        fields,
        figure_dir,
        stem="density_mock_schlieren_compare_light",
        title=common,
        x_limits=(0.0, float(metadata["x_length"])),
        y_limits=(0.0, float(metadata["y_length"])),
        reference=reference,
        reference_label=reference_label,
    )
    x0, x1, y0, y1 = vortex_window
    make_schlieren_grid(
        results,
        fields,
        figure_dir,
        stem="density_mock_schlieren_vortex_zoom",
        title=common + " (vortex region)",
        x_limits=(x0, x1),
        y_limits=(y0, y1),
        reference=reference,
        reference_label=reference_label,
    )


def make_linecuts(
    results: dict[str, dict[str, object]], reference: dict[str, object],
    figure_dir: Path, table_path: Path, reference_method: str,
) -> None:
    keys = [key for key in METHODS if key in results]
    target = results[keys[0]]
    ref_rho = restrict_reference_density(reference, target)
    ref_x = target["x"]
    ref_y = target["y"]
    reference_nx = int(reference["metadata"]["nx"])
    target_nx = int(target["metadata"]["nx"])
    reference_label = config.METHODS[reference_method].label
    reference_slug = reference_method.replace("_", "")
    cuts = (
        (0.085, "Line 1"),
        (0.097, "Line 2"),
        (0.115, "Line 3"),
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 4.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    error_figure, error_axes = plt.subplots(
        1, 3, figsize=(12.6, 4.0), sharex=True,
        constrained_layout=True,
    )
    rows: list[dict[str, object]] = []
    for ax, error_ax, (value, line_name) in zip(
        np.asarray(axes).ravel(), np.asarray(error_axes).ravel(), cuts,
    ):
        ref_coord, ref_profile = interpolate_vertical(
            ref_rho, ref_x, ref_y, value,
        )
        ax.plot(
            ref_coord,
            ref_profile,
            color="black",
            lw=1.55,
            label=(
                f"{reference_label} $N_x={reference_nx}$, "
                f"FV-restricted to $N_x={target_nx}$"
            ),
        )

        for key in keys:
            result = results[key]
            rho = result["state"][..., 0]
            coord, profile = interpolate_vertical(
                rho, result["x"], result["y"], value,
            )
            ref_on_grid = np.interp(coord, ref_coord, ref_profile)
            metrics = error_metrics(profile, ref_on_grid)
            rows.append({
                "method": key,
                "cut": "vertical",
                "location": value,
                "samples": coord.size,
                **metrics,
            })
            method = config.METHODS[key]
            ax.plot(
                coord,
                profile,
                color=method.color,
                linestyle=method.linestyle,
                lw=1.05,
                label=method.label,
            )
            error_ax.semilogy(
                coord,
                np.maximum(np.abs(profile - ref_on_grid), 1.0e-12),
                color=method.color,
                linestyle=method.linestyle,
                lw=1.05,
                label=method.label,
            )
        ax.set_title(rf"{line_name}: $x={value:.3f}$")
        ax.set_xlabel("$y$")
        ax.set_ylabel("Density $\\rho$")
        ax.grid(True, alpha=0.22)
        error_ax.set_title(rf"{line_name}: $x={value:.3f}$")
        error_ax.set_xlabel("$y$")
        error_ax.set_ylabel(r"$|\rho-\rho_{\mathrm{ref}}|$")
        error_ax.grid(True, which="both", alpha=0.22)

    handles, labels = np.asarray(axes).ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=3, frameon=False,
    )
    fig.suptitle(
        "Shock--bubble vertical density cuts against the "
        f"${reference_nx}\\times{int(reference['metadata']['ny'])}$ "
        f"{reference_label} reference"
    )
    save_figure(
        fig,
        figure_dir / f"density_linecuts_vs_{reference_slug}_N{reference_nx}",
    )

    error_handles, error_labels = np.asarray(error_axes).ravel()[0].get_legend_handles_labels()
    error_figure.legend(
        error_handles, error_labels, loc="outside lower center", ncol=3,
        frameon=False,
    )
    error_figure.suptitle(
        "Absolute errors on shock--bubble vertical density cuts after "
        "conservative FV restriction"
    )
    save_figure(
        error_figure,
        figure_dir
        / f"density_linecut_absolute_errors_vs_{reference_slug}_N{reference_nx}",
    )

    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("method", "cut", "location", "samples", "l1", "l2", "linf")
        )
        writer.writeheader()
        writer.writerows(rows)


def make_reference_figures(
    reference: dict[str, object], figure_dir: Path,
    vortex_window: tuple[float, float, float, float], reference_method: str,
) -> None:
    state = reference["state"]
    metadata = reference["metadata"]
    rho = state[..., 0]
    x_length = float(metadata["x_length"])
    y_length = float(metadata["y_length"])
    nx = int(metadata["nx"])
    ny = int(metadata["ny"])
    reference_label = config.METHODS[reference_method].label
    title = (
        f"{reference_label} reference: shock--helium-bubble interaction at "
        f"$t={scientific_math(float(metadata['t']))}$, "
        f"${nx}\\times{ny}$, CFL $={float(metadata['cfl']):.3g}$"
    )

    def draw_field(
        field: np.ndarray, *, cmap: str, colorbar_label: str,
        stem: str, limits: tuple[float, float, float, float],
        vmin: float | None = None, vmax: float | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
        image = ax.imshow(
            field,
            origin="lower",
            extent=(0.0, x_length, 0.0, y_length),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        x0, x1, y0, y1 = limits
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label=colorbar_label, shrink=0.9)
        save_figure(fig, figure_dir / stem)

    full_domain = (0.0, x_length, 0.0, y_length)
    draw_field(
        rho,
        cmap="viridis",
        colorbar_label="Density $\\rho$",
        stem=f"reference_N{nx}_density",
        limits=full_domain,
    )
    schlieren = mock_schlieren(
        rho, float(metadata["dx"]), float(metadata["dy"]),
    )
    draw_field(
        schlieren,
        cmap="gray",
        colorbar_label="Mock-schlieren intensity",
        stem=f"reference_N{nx}_mock_schlieren",
        limits=full_domain,
        vmin=0.0,
        vmax=1.0,
    )
    draw_field(
        schlieren,
        cmap="gray",
        colorbar_label="Mock-schlieren intensity",
        stem=f"reference_N{nx}_mock_schlieren_vortex_zoom",
        limits=vortex_window,
        vmin=0.0,
        vmax=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-dir", type=Path,
        default=config.RAW / "shockbubble_t0006" / "N1000x396",
    )
    parser.add_argument(
        "--reference-dir", type=Path,
        default=config.RAW / "shockbubble_t0006" / "reference_N2000x791",
    )
    parser.add_argument(
        "--reference-method", choices=METHODS, default="weno5_js",
    )
    parser.add_argument(
        "--figure-dir", type=Path,
        default=config.FIGURES / "shockbubble_t0006",
    )
    parser.add_argument(
        "--table-path", type=Path,
        default=config.TABLES / "shockbubble_t0006_linecut_errors.csv",
    )
    parser.add_argument("--vortex-x-min", type=float, default=0.075)
    parser.add_argument("--vortex-x-max", type=float, default=0.155)
    parser.add_argument("--vortex-y-min", type=float, default=0.0)
    parser.add_argument("--vortex-y-max", type=float, default=0.089)
    args = parser.parse_args()
    results = {
        key: load_result(args.main_dir / f"{key}.npz")
        for key in METHODS
        if (args.main_dir / f"{key}.npz").is_file()
    }
    if not results:
        raise FileNotFoundError(f"no completed shock-bubble results in {args.main_dir}")
    reference_path = args.reference_dir / f"{args.reference_method}.npz"
    reference = load_result(reference_path) if reference_path.is_file() else None
    reference_label = None
    if reference is not None:
        reference_metadata = reference["metadata"]
        reference_label = (
            f"{config.METHODS[args.reference_method].label} reference\n"
            f"${int(reference_metadata['nx'])}\\times"
            f"{int(reference_metadata['ny'])}$"
        )
    make_schlieren(
        results,
        args.figure_dir,
        (
            args.vortex_x_min,
            args.vortex_x_max,
            args.vortex_y_min,
            args.vortex_y_max,
        ),
        reference=reference,
        reference_label=reference_label,
    )
    if reference_path.is_file():
        assert reference is not None
        make_reference_figures(
            reference,
            args.figure_dir,
            (
                args.vortex_x_min,
                args.vortex_x_max,
                args.vortex_y_min,
                args.vortex_y_max,
            ),
            args.reference_method,
        )
        make_linecuts(
            results, reference, args.figure_dir, args.table_path,
            args.reference_method,
        )
    else:
        print(f"reference not ready; skipped line cuts: {reference_path}", flush=True)
    print(f"shockbubble_figures={args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
