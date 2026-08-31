#!/usr/bin/env python3
"""Create uncluttered, pairwise shock--bubble line-cut figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from for_paper_results import config
from for_paper_results.make_shockbubble_figures import (
    interpolate_horizontal,
    interpolate_vertical,
    load_result,
    mock_schlieren,
    restrict_reference_density,
    save_figure,
)


METHODS = (
    "weno5_js",
    "weno5_sr_f64",
    "weno5_sr_f32",
    "weno7_js",
    "weno7_sr_f64",
)


def profile_at(
    field: np.ndarray,
    result: dict[str, object],
    direction: str,
    value: float,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "x":
        return interpolate_vertical(field, result["x"], result["y"], value)
    if direction == "y":
        return interpolate_horizontal(field, result["x"], result["y"], value)
    raise ValueError(f"unsupported cut direction: {direction}")


def crop_profile(
    coordinate: np.ndarray,
    profile: np.ndarray,
    coordinate_min: float,
    coordinate_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (coordinate >= coordinate_min) & (coordinate <= coordinate_max)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"cut window [{coordinate_min}, {coordinate_max}] contains fewer "
            "than two cells"
        )
    return coordinate[mask], profile[mask]


def cut_annotation(direction: str, value: float, digits: int = 5) -> str:
    return rf"${direction}={value:.{digits}f}$"


def line_metrics(profile: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = profile - reference
    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.sqrt(np.mean(error * error))),
        "linf": float(np.max(np.abs(error))),
        "tv": float(np.sum(np.abs(np.diff(profile)))),
    }


def create_figures(
    main_dir: Path,
    reference_path: Path,
    output_dir: Path,
    table_path: Path,
    cuts: tuple[tuple[str, float, float, float], ...],
    case_label: str,
) -> None:
    results = {key: load_result(main_dir / f"{key}.npz") for key in METHODS}
    target = results["weno5_js"]
    reference = load_result(reference_path)
    reference_rho = restrict_reference_density(reference, target)
    reference_nx = int(reference["metadata"]["nx"])
    reference_ny = int(reference["metadata"]["ny"])
    target_nx = int(target["metadata"]["nx"])
    target_ny = int(target["metadata"]["ny"])

    styles = {
        "reference": ("#111111", "-", 1.55),
        "weno5_js": ("#666666", "--", 1.15),
        "weno5_sr_f64": ("#0072B2", "-", 1.25),
        "weno5_sr_f32": ("#009E73", "-.", 1.25),
        "weno7_js": ("#D55E00", "--", 1.15),
        "weno7_sr_f64": ("#CC79A7", "-", 1.25),
    }
    labels = {
        "reference": f"WENO7-JS-RK4 {reference_nx} reference",
        **{key: config.METHODS[key].label for key in METHODS},
    }
    groups = (
        ("WENO5 reconstructions", ("weno5_js", "weno5_sr_f64", "weno5_sr_f32")),
        ("WENO7 reconstructions", ("weno7_js", "weno7_sr_f64")),
    )

    profile_figure, profile_axes = plt.subplots(
        2, 3, figsize=(12.6, 7.1), sharex="col", sharey="col",
        constrained_layout=True,
    )
    error_figure, error_axes = plt.subplots(
        2, 3, figsize=(12.6, 7.1), sharex="col",
        constrained_layout=True,
    )
    rows: list[dict[str, object]] = []
    panel_letters = "abcdef"

    for row_index, (family_label, family_methods) in enumerate(groups):
        for column_index, cut in enumerate(cuts):
            direction, value, coordinate_min, coordinate_max = cut
            profile_axis = profile_axes[row_index, column_index]
            error_axis = error_axes[row_index, column_index]
            coordinate, reference_profile = profile_at(
                reference_rho, target, direction, value,
            )
            coordinate, reference_profile = crop_profile(
                coordinate, reference_profile, coordinate_min, coordinate_max,
            )
            color, linestyle, linewidth = styles["reference"]
            profile_axis.plot(
                coordinate, reference_profile, color=color, linestyle=linestyle,
                linewidth=linewidth, zorder=2,
            )
            reference_tv = float(np.sum(np.abs(np.diff(reference_profile))))

            baseline_key = "weno5_js" if row_index == 0 else "weno7_js"
            baseline_coordinate, baseline_profile = profile_at(
                results[baseline_key]["state"][..., 0],
                results[baseline_key], direction, value,
            )
            _, baseline_profile = crop_profile(
                baseline_coordinate, baseline_profile,
                coordinate_min, coordinate_max,
            )
            baseline_l1 = line_metrics(baseline_profile, reference_profile)["l1"]

            for key in family_methods:
                result = results[key]
                result_coordinate, profile = profile_at(
                    result["state"][..., 0], result, direction, value,
                )
                _, profile = crop_profile(
                    result_coordinate, profile, coordinate_min, coordinate_max,
                )
                metrics = line_metrics(profile, reference_profile)
                rows.append({
                    "case": case_label,
                    "direction": direction,
                    "location": value,
                    "coordinate_min": coordinate_min,
                    "coordinate_max": coordinate_max,
                    "family": family_label,
                    "method": key,
                    **metrics,
                    "l1_over_classical": metrics["l1"] / baseline_l1,
                    "tv_over_reference": metrics["tv"] / max(reference_tv, 1.0e-15),
                })
                color, linestyle, linewidth = styles[key]
                profile_axis.plot(
                    coordinate, profile, color=color, linestyle=linestyle,
                    linewidth=linewidth, zorder=3,
                )
                error_axis.semilogy(
                    coordinate,
                    np.maximum(np.abs(profile - reference_profile), 1.0e-12),
                    color=color, linestyle=linestyle, linewidth=linewidth,
                )

            panel = panel_letters[3 * row_index + column_index]
            annotation = cut_annotation(direction, value, digits=4)
            profile_axis.set_title(rf"({panel}) {annotation}")
            error_axis.set_title(rf"({panel}) {annotation}")
            coordinate_label = "$y$" if direction == "x" else "$x$"
            profile_axis.set_xlabel(coordinate_label)
            error_axis.set_xlabel(coordinate_label)
            profile_axis.grid(True, alpha=0.18)
            error_axis.grid(True, which="both", alpha=0.18)
            if column_index == 0:
                profile_axis.set_ylabel(f"{family_label}\nDensity $\\rho$")
                error_axis.set_ylabel(
                    f"{family_label}\n$|\\rho-\\rho_{{\\mathrm{{ref}}}}|$"
                )

    legend_order = ("reference",) + METHODS
    handles = [
        Line2D(
            [0], [0], color=styles[key][0], linestyle=styles[key][1],
            linewidth=styles[key][2], label=labels[key],
        )
        for key in legend_order
    ]
    title = f"{case_label}: selected density profiles at $N_x={target_nx}$"
    profile_figure.suptitle(title)
    error_figure.suptitle(title + " (absolute error)")
    profile_figure.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False,
    )
    error_figure.legend(
        handles=handles[1:], loc="outside lower center", ncol=3, frameon=False,
    )
    save_figure(profile_figure, output_dir / "selected_density_linecuts")
    save_figure(error_figure, output_dir / "selected_density_linecut_errors")

    combined_figure = plt.figure(figsize=(11.2, 7.6), constrained_layout=True)
    combined_grid = combined_figure.add_gridspec(
        2, 2, height_ratios=(0.82, 1.35),
    )
    combined_axes = (
        combined_figure.add_subplot(combined_grid[0, 0]),
        combined_figure.add_subplot(combined_grid[0, 1]),
        combined_figure.add_subplot(combined_grid[1, :]),
    )
    combined_error_figure = plt.figure(
        figsize=(11.2, 7.6), constrained_layout=True,
    )
    combined_error_grid = combined_error_figure.add_gridspec(
        2, 2, height_ratios=(0.82, 1.35),
    )
    combined_error_axes = (
        combined_error_figure.add_subplot(combined_error_grid[0, 0]),
        combined_error_figure.add_subplot(combined_error_grid[0, 1]),
        combined_error_figure.add_subplot(combined_error_grid[1, :]),
    )
    for column_index, cut in enumerate(cuts):
        direction, value, coordinate_min, coordinate_max = cut
        axis = combined_axes[column_index]
        error_axis = combined_error_axes[column_index]
        coordinate, reference_profile = profile_at(
            reference_rho, target, direction, value,
        )
        coordinate, reference_profile = crop_profile(
            coordinate, reference_profile, coordinate_min, coordinate_max,
        )
        color, linestyle, linewidth = styles["reference"]
        axis.plot(
            coordinate, reference_profile, color=color, linestyle=linestyle,
            linewidth=linewidth, zorder=2,
        )
        for key in METHODS:
            result = results[key]
            result_coordinate, profile = profile_at(
                result["state"][..., 0], result, direction, value,
            )
            _, profile = crop_profile(
                result_coordinate, profile, coordinate_min, coordinate_max,
            )
            color, linestyle, linewidth = styles[key]
            axis.plot(
                coordinate, profile, color=color, linestyle=linestyle,
                linewidth=linewidth, zorder=3,
            )
            error_axis.semilogy(
                coordinate,
                np.maximum(np.abs(profile - reference_profile), 1.0e-12),
                color=color, linestyle=linestyle, linewidth=linewidth,
            )
        panel = "abc"[column_index]
        annotation = cut_annotation(direction, value)
        axis.set_title(rf"({panel}) {annotation}")
        error_axis.set_title(rf"({panel}) {annotation}")
        coordinate_label = "$y$" if direction == "x" else "$x$"
        axis.set_xlabel(coordinate_label)
        error_axis.set_xlabel(coordinate_label)
        axis.grid(True, alpha=0.18)
        error_axis.grid(True, which="both", alpha=0.18)
        if column_index in (0, 2):
            axis.set_ylabel("Density $\\rho$")
            error_axis.set_ylabel(r"$|\rho-\rho_{\mathrm{ref}}|$")
    combined_figure.suptitle(title)
    combined_error_figure.suptitle(title + " (absolute error)")
    combined_figure.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False,
    )
    combined_error_figure.legend(
        handles=handles[1:], loc="outside lower center", ncol=3, frameon=False,
    )
    save_figure(
        combined_figure, output_dir / "selected_density_linecuts_combined",
    )
    save_figure(
        combined_error_figure,
        output_dir / "selected_density_linecut_errors_combined",
    )

    reference_state = reference["state"]
    metadata = reference["metadata"]
    dx = float(metadata["x_length"]) / int(metadata["nx"])
    dy = float(metadata["y_length"]) / int(metadata["ny"])
    schlieren = mock_schlieren(reference_state[..., 0], dx, dy)
    overview, axis = plt.subplots(figsize=(7.1, 3.4), constrained_layout=True)
    image = axis.imshow(
        schlieren,
        origin="lower",
        extent=(0.0, float(metadata["x_length"]), 0.0, float(metadata["y_length"])),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    colors = ("#0072B2", "#D55E00", "#009E73")
    for index, (cut, cut_color) in enumerate(zip(cuts, colors), start=1):
        direction, value, _, _ = cut
        if direction == "x":
            axis.axvline(value, color=cut_color, linewidth=1.25, linestyle="--")
            axis.text(
                value + 0.001, 0.003,
                f"Line {index}: $x={value:.4f}$",
                color=cut_color, fontsize=7.5, rotation=90,
                ha="left", va="bottom",
            )
        else:
            axis.axhline(value, color=cut_color, linewidth=1.25, linestyle="--")
            axis.text(
                0.124, value + 0.001,
                f"Line {index}: $y={value:.4f}$",
                color=cut_color, fontsize=7.5, ha="right", va="bottom",
            )
    x_length = float(metadata["x_length"])
    x_markers = [value for direction, value, _, _ in cuts if direction == "x"]
    horizontal_windows = [
        (coordinate_min, coordinate_max)
        for direction, _, coordinate_min, coordinate_max in cuts
        if direction == "y"
    ]
    visible_x = x_markers + [edge for window in horizontal_windows for edge in window]
    axis.set_xlim(
        max(0.0, min(visible_x) - 0.003),
        min(x_length, max(visible_x) + 0.005),
    )
    axis.set_ylim(0.0, float(metadata["y_length"]))
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")
    axis.set_title(f"{case_label}: selected line cuts", fontsize=11)
    axis.tick_params(labelsize=9)
    colorbar = overview.colorbar(image, ax=axis)
    colorbar.set_label("Mock-schlieren intensity", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    save_figure(overview, output_dir / "selected_cut_locations")

    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nLocal density L1 against conservatively restricted reference")
    print(f"reference=WENO7-JS-RK4 {reference_nx}x{reference_ny}")
    print(f"comparison_grid={target_nx}x{target_ny}")
    for direction, value, coordinate_min, coordinate_max in cuts:
        print(
            f"\ncut {direction}={value:.7f} "
            f"window=[{coordinate_min:.6f}, {coordinate_max:.6f}]"
        )
        cut_rows = {
            str(row["method"]): row
            for row in rows
            if row["direction"] == direction
            and abs(float(row["location"]) - value) < 1.0e-12
        }
        for key in METHODS:
            row = cut_rows[key]
            print(
                f"  {key:<15} L1={float(row['l1']):.10e} "
                f"L1/classical={float(row['l1_over_classical']):.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table-path", type=Path, required=True)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--line-x", type=float, nargs="+")
    parser.add_argument("--line-y", type=float)
    parser.add_argument(
        "--cut",
        action="append",
        nargs=4,
        metavar=("DIRECTION", "VALUE", "MIN", "MAX"),
        help=(
            "Explicit cut as direction/value/profile-min/profile-max. "
            "Repeat exactly three times; cannot be combined with --line-x/--line-y."
        ),
    )
    args = parser.parse_args()
    if args.cut is not None:
        if args.line_x is not None or args.line_y is not None:
            parser.error("--cut cannot be combined with --line-x or --line-y")
        if len(args.cut) != 3:
            parser.error("--cut must be supplied exactly three times")
        explicit_cuts = []
        for direction, value, coordinate_min, coordinate_max in args.cut:
            if direction not in ("x", "y"):
                parser.error("cut DIRECTION must be x or y")
            explicit_cuts.append(
                (direction, float(value), float(coordinate_min), float(coordinate_max))
            )
        cuts = tuple(explicit_cuts)
    elif args.line_x is None:
        parser.error("supply either --line-x or three explicit --cut arguments")
    elif args.line_y is None:
        if len(args.line_x) != 3:
            parser.error("--line-x requires three values when --line-y is omitted")
        cuts = tuple(
            ("x", value, 0.0, 0.089)
            for value in args.line_x
        )
    else:
        if len(args.line_x) != 2:
            parser.error("--line-x requires two values when --line-y is supplied")
        cuts = (
            ("x", args.line_x[0], 0.005, 0.084),
            ("x", args.line_x[1], 0.012, 0.077),
            ("y", args.line_y, 0.060, 0.125),
        )
    create_figures(
        args.main_dir,
        args.reference,
        args.output_dir,
        args.table_path,
        cuts,
        args.case_label,
    )
    print(f"figures={args.output_dir}")
    print(f"table={args.table_path}")


if __name__ == "__main__":
    main()
