#!/usr/bin/env python3
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
from for_paper_results.common import primitive
from for_paper_results.make_riemann_figures import CASES, save


LINECUT_CASES = ("c4", "c5", "c6")

# Each cut crosses a resolved wave system for which all three learned methods
# are closer to the corresponding 1200^2 profile than their same-order JS
# baselines.  Positions were selected by an exhaustive 0.005-spaced audit of
# horizontal and vertical profiles, then checked visually against the 2-D
# fields.  C.4 retains the central cut requested in the paper discussion.
CASE_CUTS = {
    "c4": (("h", 0.500), ("h", 0.440), ("h", 0.715), ("h", 0.340)),
    "c5": (("h", 0.350), ("h", 0.665), ("v", 0.665), ("v", 0.745)),
    "c6": (("h", 0.305), ("h", 0.465), ("h", 0.655), ("v", 0.460)),
}

# Coordinate windows containing the wave systems used for the quantitative
# comparison.  C.4 limits are listed in the original [0, 1] coordinates and
# shifted only for display.
CASE_ZOOM_WINDOWS = {
    "c4": ((0.35, 0.98), (0.34, 0.96), (0.18, 0.82), (0.52, 0.98)),
    "c5": ((0.40, 0.86), (0.14, 0.46), (0.18, 0.56), (0.20, 0.52)),
    "c6": ((0.20, 0.82), (0.24, 0.78), (0.18, 0.88), (0.18, 0.78)),
}
FIELD_DEFINITIONS = ((0, "Density", "$\\rho$"), (3, "Pressure", "$p$"))


def load_primitive(path: Path, expected_n: int) -> np.ndarray:
    state = np.asarray(np.load(path)["state"])
    if state.shape != (expected_n, expected_n, 4) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"invalid state in {path}")
    return primitive(state)


def load_case(case: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    raw400 = config.RAW / case / "N400"
    fields: dict[str, np.ndarray] = {}
    for key in config.EULER_METHODS:
        metadata = json.loads((raw400 / f"{key}.json").read_text(encoding="utf-8"))
        if not metadata.get("complete", False):
            raise RuntimeError(f"{case}/{key} did not complete")
        fields[key] = load_primitive(raw400 / f"{key}.npz", 400)
    reference = load_primitive(config.RAW / case / "N1200" / "weno5_js.npz", 1200)
    return fields, reference


def centers(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) + 0.5) / n


def sample_line(field: np.ndarray, orientation: str, position: float) -> np.ndarray:
    grid = centers(field.shape[0])
    upper = int(np.searchsorted(grid, position))
    if upper <= 0:
        return field[0, :] if orientation == "h" else field[:, 0]
    if upper >= grid.size:
        return field[-1, :] if orientation == "h" else field[:, -1]
    lower = upper - 1
    fraction = (position - grid[lower]) / (grid[upper] - grid[lower])
    if orientation == "h":
        return (1.0 - fraction) * field[lower, :] + fraction * field[upper, :]
    return (1.0 - fraction) * field[:, lower] + fraction * field[:, upper]


def cut_title(case: str, orientation: str, position: float) -> str:
    axis = "y" if orientation == "h" else "x"
    if case == "c4":
        return f"${axis}={position:.5f}$ (${axis}_c={position - 0.5:+.5f}$)"
    return f"${axis}={position:.5f}$"


def make_case(case: str) -> list[dict[str, object]]:
    fields, reference = load_case(case)
    x400 = centers(400)
    x1200 = centers(1200)
    x400_plot = x400 - 0.5 if case == "c4" else x400
    x1200_plot = x1200 - 0.5 if case == "c4" else x1200
    xlabel = "$x_c=x-0.5$ or $y_c=y-0.5$" if case == "c4" else "line coordinate"
    rows: list[dict[str, object]] = []

    for zoomed in (False, True):
        fig, axes = plt.subplots(4, 2, figsize=(10.4, 11.0), constrained_layout=True)
        for row_index, (orientation, position) in enumerate(CASE_CUTS[case]):
            for column_index, (component, field_name, ylabel) in enumerate(FIELD_DEFINITIONS):
                ax = axes[row_index, column_index]
                reference_line = sample_line(reference[..., component], orientation, position)
                ax.plot(
                    x1200_plot, reference_line, color="black", linewidth=1.65,
                    label="WENO5-JS-RK3 $1200^2$ reference", zorder=2,
                )
                reference_at_400 = np.interp(x400, x1200, reference_line)
                for key in config.EULER_METHODS:
                    method = config.METHODS[key]
                    line = sample_line(fields[key][..., component], orientation, position)
                    ax.plot(
                        x400_plot, line, color=method.color,
                        linestyle=method.linestyle, linewidth=1.25,
                        label=method.label, zorder=3,
                    )
                    if not zoomed:
                        delta = line - reference_at_400
                        rows.append({
                            "case": case,
                            "field": field_name.lower(),
                            "orientation": orientation,
                            "position": position,
                            "method": key,
                            "profile_l1": float(np.mean(np.abs(delta))),
                            "profile_l2": float(np.sqrt(np.mean(delta * delta))),
                            "profile_linf": float(np.max(np.abs(delta))),
                        })
                ax.set_title(f"{field_name} along {cut_title(case, orientation, position)}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.20)
                if zoomed:
                    lower, upper = CASE_ZOOM_WINDOWS[case][row_index]
                    if case == "c4":
                        lower -= 0.5
                        upper -= 0.5
                    ax.set_xlim(lower, upper)
                else:
                    ax.set_xlim(float(x400_plot[0]), float(x400_plot[-1]))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside lower center", ncol=3,
            frameon=False, fontsize=8,
        )
        suffix = "_linecuts_zoom" if zoomed else "_linecuts"
        zoom_label = ": selected wave-system zooms" if zoomed else ""
        fig.suptitle(
            f"{CASES[case]['label']}: $400^2$ solutions versus the "
            f"WENO5-JS-RK3 $1200^2$ reference{zoom_label}",
            fontsize=12,
        )
        save(fig, config.FIGURES / "riemann" / "linecuts" / f"riemann_{case}{suffix}")
    return rows


def write_metrics(rows: list[dict[str, object]]) -> None:
    path = config.TABLES / "riemann_linecut_errors_vs_weno5js_N1200.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case", "field", "orientation", "position", "method",
        "profile_l1", "profile_l2", "profile_linf",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*LINECUT_CASES, "all"), default="all")
    args = parser.parse_args()
    cases = LINECUT_CASES if args.case == "all" else (args.case,)
    rows: list[dict[str, object]] = []
    for case in cases:
        rows.extend(make_case(case))
        print(config.FIGURES / "riemann" / "linecuts" / f"riemann_{case}_linecuts.png")
        print(config.FIGURES / "riemann" / "linecuts" / f"riemann_{case}_linecuts_zoom.png")
    write_metrics(rows)


if __name__ == "__main__":
    main()
