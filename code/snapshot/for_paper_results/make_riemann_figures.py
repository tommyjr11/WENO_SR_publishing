#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results import config
from for_paper_results.common import primitive


CASES = {
    "c3": {
        "label": "C.3 (q400)",
        "t_end": 0.50,
        "raw_case": "q400",
    },
    "c4": {"label": "C.4", "t_end": 0.25},
    "c5": {"label": "C.5", "t_end": 0.23},
    "c6": {"label": "C.6", "t_end": 0.30},
}

# Each configuration uses one fixed, equally spaced density sequence for all
# five methods and its high-resolution reference.  The offsets avoid placing a
# contour exactly on an initial constant state, where roundoff can create
# visually distracting fragments.
DENSITY_CONTOUR_LEVELS = {
    # Preserve the original q400 presentation used throughout the project.
    "c3": np.arange(0.16, 1.71 + 1.0e-12, 0.05),
    "c4": np.linspace(0.64, 1.92, 19),
    "c5": np.arange(1.10, 3.90 + 1.0e-12, 0.14),
    "c6": np.arange(0.54, 2.94 + 1.0e-12, 0.10),
}


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_fields(case: str) -> dict[str, np.ndarray]:
    raw_case = CASES[case].get("raw_case", case)
    raw = config.RAW / raw_case / "N400"
    fields: dict[str, np.ndarray] = {}
    for key in config.EULER_METHODS:
        metadata = json.loads((raw / f"{key}.json").read_text(encoding="utf-8"))
        if not metadata.get("complete", False):
            raise RuntimeError(f"{case}/{key} did not pass the validation gates")
        state = np.asarray(np.load(raw / f"{key}.npz")["state"])
        if state.shape != (400, 400, 4) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"{case}/{key} has an invalid state array")
        fields[key] = primitive(state)
    return fields


def make_case(
    case: str,
    density_levels_override: np.ndarray | None = None,
    output_tag: str | None = None,
) -> Path:
    definition = CASES[case]
    fields = load_fields(case)
    x = (np.arange(400, dtype=np.float64) + 0.5) / 400.0
    if case == "c4":
        x = x - 0.5
    xx, yy = np.meshgrid(x, x)

    pressure_min = min(float(np.min(value[..., 3])) for value in fields.values())
    pressure_max = max(float(np.max(value[..., 3])) for value in fields.values())
    pressure_levels = np.linspace(pressure_min, pressure_max, 241)
    density_min = min(float(np.min(value[..., 0])) for value in fields.values())
    density_max = max(float(np.max(value[..., 0])) for value in fields.values())
    density_levels = (
        density_levels_override
        if density_levels_override is not None
        else DENSITY_CONTOUR_LEVELS.get(
            case, np.linspace(density_min, density_max, 14)[1:-1]
        )
    )
    if density_levels[0] <= density_min or density_levels[-1] >= density_max:
        raise RuntimeError(
            f"density contour levels for {case} must lie inside "
            f"({density_min:.6g}, {density_max:.6g})"
        )
    velocity_max = max(
        float(np.max(np.hypot(value[..., 1], value[..., 2])))
        for value in fields.values()
    )
    quiver_scale = max(1.0, 32.0 * velocity_max)
    stride = 20
    axis_min, axis_max = (-0.5, 0.5) if case == "c4" else (0.0, 1.0)

    fig = plt.figure(figsize=(11.2, 7.6))
    grid = fig.add_gridspec(
        2, 6, left=0.065, right=0.885, bottom=0.075, top=0.89,
        wspace=0.62, hspace=0.42,
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    image = None
    for ax, key in zip(axes, config.EULER_METHODS):
        value = fields[key]
        image = ax.contourf(
            xx, yy, value[..., 3], levels=pressure_levels,
            cmap="turbo", extend="both",
        )
        ax.contour(
            xx, yy, value[..., 0], levels=density_levels,
            colors="black",
            linewidths=0.30 if case == "c3" else 0.36,
            alpha=1.0 if case == "c3" else 0.78,
        )
        if case != "c4":
            ax.quiver(
                xx[::stride, ::stride], yy[::stride, ::stride],
                value[::stride, ::stride, 1], value[::stride, ::stride, 2],
                color="white", alpha=0.82, pivot="mid", scale=quiver_scale,
                width=0.0024, headwidth=3.2, headlength=4.2,
            )
        ax.set_title(config.METHODS[key].label, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        if case == "c4":
            ax.set_xticks(np.linspace(-0.5, 0.5, 5))
            ax.set_yticks(np.linspace(-0.5, 0.5, 5))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.tick_params(labelsize=8)
    if image is None:
        raise RuntimeError(f"no fields plotted for {case}")
    colorbar_axis = fig.add_axes((0.915, 0.15, 0.024, 0.68))
    colorbar = fig.colorbar(image, cax=colorbar_axis, label="Pressure $p$")
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Two-dimensional Riemann configuration {definition['label']} "
        f"at $t={definition['t_end']:.2f}$",
        fontsize=12, y=0.965,
    )
    suffix = f"_{output_tag}" if output_tag else ""
    output_stem = config.FIGURES / "riemann" / f"riemann_{case}{suffix}"
    save(fig, output_stem)
    return output_stem.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--density-min", type=float)
    parser.add_argument("--density-max", type=float)
    parser.add_argument("--density-count", type=int)
    parser.add_argument("--output-tag")
    args = parser.parse_args()

    density_args = (args.density_min, args.density_max, args.density_count)
    if any(value is not None for value in density_args) and not all(
        value is not None for value in density_args
    ):
        parser.error(
            "--density-min, --density-max, and --density-count must be used together"
        )
    if args.density_count is not None:
        if args.case == "all":
            parser.error("custom density levels require one explicit --case")
        if args.density_count < 2:
            parser.error("--density-count must be at least 2")
        if args.density_min >= args.density_max:
            parser.error("--density-min must be smaller than --density-max")
    if args.output_tag and (
        Path(args.output_tag).name != args.output_tag
        or not args.output_tag.replace("-", "").replace("_", "").isalnum()
    ):
        parser.error("--output-tag may contain only letters, numbers, '-' and '_'")

    density_levels = None
    if args.density_count is not None:
        density_levels = np.linspace(
            args.density_min, args.density_max, args.density_count
        )
        print(
            f"density contours: min={density_levels[0]:.8g} "
            f"max={density_levels[-1]:.8g} count={density_levels.size} "
            f"spacing={density_levels[1] - density_levels[0]:.8g}"
        )
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        print(make_case(case, density_levels, args.output_tag))


if __name__ == "__main__":
    main()
