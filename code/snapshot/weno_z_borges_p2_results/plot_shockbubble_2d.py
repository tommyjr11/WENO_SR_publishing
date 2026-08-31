#!/usr/bin/env python3
"""Plot 2-D shock-bubble fields and selected line cuts with WENO-Z."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results.make_shockbubble_figures import (
    interpolate_horizontal,
    interpolate_vertical,
    load_result,
    mock_schlieren,
    restrict_reference_density,
)


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
METHODS = (
    "weno5_js", "weno5_z_p2", "weno5_sr_f64", "weno5_sr_f32",
    "weno7_js", "weno7_z_p3", "weno7_sr_f64",
)
FAMILIES = (
    ("WENO5 reconstructions", METHODS[:4]),
    ("WENO7 reconstructions", METHODS[4:]),
)
LABELS = {
    "weno5_js": "WENO5-JS-RK3",
    "weno5_z_p2": "WENO5-Z-RK3",
    "weno5_sr_f64": "WENO5-SR-RK3",
    "weno5_sr_f32": "WENO5-SR-FP32-RK3",
    "weno7_js": "WENO7-JS-RK4",
    "weno7_z_p3": "WENO7-Z-RK4",
    "weno7_sr_f64": "WENO7-SR-RK4",
}
STYLES = {
    "reference": ("#111111", "-", 1.65),
    "weno5_js": ("#666666", "--", 1.05),
    "weno5_z_p2": ("#56B4E9", ":", 1.40),
    "weno5_sr_f64": ("#0072B2", "-", 1.20),
    "weno5_sr_f32": ("#009E73", "-.", 1.20),
    "weno7_js": ("#D55E00", "--", 1.05),
    "weno7_z_p3": ("#E69F00", ":", 1.40),
    "weno7_sr_f64": ("#CC79A7", "-", 1.20),
}
CASES = {
    "ma122": {
        "label": "Shock--bubble interaction, Ma=1.22",
        "old": REPO / "shockbubble_t0006_cfl04_server/results/raw/shockbubble_t0006_cfl04/N1000x396",
        "reference": REPO / "shockbubble_t0006_cfl04_server/results/raw/shockbubble_t0006_cfl04/reference_weno7_N2000x791/weno7_js.npz",
        "cuts": (("y", 0.019, 0.115, 0.128),
                 ("y", 0.04525, 0.100, 0.145),
                 ("x", 0.1256625, 0.005, 0.084)),
    },
    "ma30": {
        "label": "Shock--bubble interaction, Ma=3.0",
        "old": REPO / "shockbubble_ma3_t0001_cfl04_server/results/raw/shockbubble_ma3_t0001_cfl04/N1000x396",
        "reference": REPO / "shockbubble_ma3_t0001_cfl04_server/results/raw/shockbubble_ma3_t0001_cfl04/reference_weno7_N2000x791/weno7_js.npz",
        "cuts": (("x", 0.0675, 0.005, 0.084),
                 ("x", 0.1099125, 0.012, 0.077),
                 ("y", 0.02056439393939394, 0.060, 0.125)),
    },
}


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_case(case: str):
    definition = CASES[case]
    new = ROOT / f"raw/shockbubble_2d/{case}/N1000x396"
    results: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = {}
    for method in METHODS:
        base = new if "_z_" in method else definition["old"]
        path = base / f"{method}.npz"
        if not path.is_file():
            failures[method] = "result unavailable"
            continue
        item = load_result(path)
        metadata = item["metadata"]
        if not metadata.get("complete", True):
            failures[method] = f"failed at step {metadata.get('steps', '?')}"
            continue
        state = item["state"]
        if state.shape != (396, 1000, 4) or not np.all(np.isfinite(state)):
            failures[method] = "invalid state"
            continue
        results[method] = item
    reference = load_result(definition["reference"])
    return results, failures, reference


def field_panel(case: str, results, failures, reference, field_name: str) -> None:
    panels = [(method, results.get(method)) for method in METHODS]
    panels.append(("reference", reference))
    density = {
        key: item["state"][..., 0]
        for key, item in panels if item is not None
    }
    if field_name == "density":
        fields = density
        vmin = min(float(value.min()) for value in fields.values())
        vmax = max(float(value.max()) for value in fields.values())
        cmap = "turbo"
        label = r"Density $\rho$"
    else:
        fields = {}
        for key, item in panels:
            if item is None:
                continue
            metadata = item["metadata"]
            fields[key] = mock_schlieren(
                item["state"][..., 0], float(metadata["dx"]), float(metadata["dy"])
            )
        vmin, vmax, cmap, label = 0.0, 1.0, "gray", "Mock-schlieren intensity"

    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.5), constrained_layout=True,
                             sharex=True, sharey=True)
    image = None
    for axis, (key, item) in zip(axes.flat, panels):
        title = (
            f"WENO7-JS-RK4\n{int(reference['metadata']['nx'])} reference"
            if key == "reference" else LABELS[key]
        )
        axis.set_title(title, fontsize=9.2)
        if item is not None:
            metadata = item["metadata"]
            image = axis.imshow(
                fields[key], origin="lower",
                extent=(0.0, float(metadata["x_length"]), 0.0, float(metadata["y_length"])),
                cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal",
            )
        else:
            axis.set_facecolor("#f6f6f6")
            axis.text(0.5, 0.54, "Run failed", ha="center", va="center",
                      transform=axis.transAxes, weight="bold")
            axis.text(0.5, 0.43, failures[key], ha="center", va="center",
                      transform=axis.transAxes, fontsize=8)
        axis.set_xlim(0.0, 0.225)
        axis.set_ylim(0.0, 0.089)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=label, shrink=0.90)
    fig.suptitle(CASES[case]["label"])
    save(fig, ROOT / f"figures/shockbubble_2d/{case}/{field_name}_with_weno_z")


def extract_profile(field, result, direction: str, value: float):
    if direction == "x":
        return interpolate_vertical(field, result["x"], result["y"], value)
    return interpolate_horizontal(field, result["x"], result["y"], value)


def crop(coordinate, values, lower: float, upper: float):
    mask = (coordinate >= lower) & (coordinate <= upper)
    return coordinate[mask], values[mask]


def linecuts(case: str, results, failures, reference) -> None:
    target = results["weno5_js"]
    restricted_reference = restrict_reference_density(reference, target)
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True,
                             sharex="col", sharey="col")
    for row_index, (family, family_methods) in enumerate(FAMILIES):
        for column, (direction, value, lower, upper) in enumerate(CASES[case]["cuts"]):
            axis = axes[row_index, column]
            coordinate, ref_values = extract_profile(
                restricted_reference, target, direction, value,
            )
            coordinate, ref_values = crop(coordinate, ref_values, lower, upper)
            axis.plot(coordinate, ref_values, color=STYLES["reference"][0],
                      linestyle=STYLES["reference"][1], linewidth=STYLES["reference"][2],
                      label="WENO7-JS 2000 reference")
            for method in family_methods:
                if method not in results:
                    rows.append({
                        "case": case, "direction": direction, "location": value,
                        "coordinate_min": lower, "coordinate_max": upper,
                        "method": method, "status": failures[method],
                        "l1": np.nan, "l2": np.nan, "linf": np.nan,
                    })
                    continue
                result = results[method]
                method_coordinate, values = extract_profile(
                    result["state"][..., 0], result, direction, value,
                )
                _, values = crop(method_coordinate, values, lower, upper)
                error = values - ref_values
                rows.append({
                    "case": case, "direction": direction, "location": value,
                    "coordinate_min": lower, "coordinate_max": upper,
                    "method": method, "status": "complete",
                    "l1": float(np.mean(np.abs(error))),
                    "l2": float(np.sqrt(np.mean(error * error))),
                    "linf": float(np.max(np.abs(error))),
                })
                color, linestyle, linewidth = STYLES[method]
                axis.plot(coordinate, values, color=color, linestyle=linestyle,
                          linewidth=linewidth, label=LABELS[method])
            failed_family = [method for method in family_methods if method in failures]
            if failed_family:
                axis.text(
                    0.02, 0.96,
                    ", ".join(f"{LABELS[method]} failed" for method in failed_family),
                    transform=axis.transAxes, ha="left", va="top",
                    fontsize=7.2, color="#A33A2B",
                )
            axis.set_title(rf"${direction}={value:.5f}$")
            axis.set_xlabel(r"$y$" if direction == "x" else r"$x$")
            axis.grid(alpha=0.18)
            if column == 0:
                axis.set_ylabel(f"{family}\nDensity $\\rho$")
    handles, labels = [], []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, fontsize=8)
    fig.suptitle(CASES[case]["label"] + ": selected density line cuts")
    save(fig, ROOT / f"figures/shockbubble_2d/{case}/selected_density_linecuts_with_weno_z")
    table = ROOT / f"tables/shockbubble_2d_{case}_linecuts_with_weno_z.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    args = parser.parse_args()
    selected = CASES if args.case == "all" else (args.case,)
    for case in selected:
        results, failures, reference = load_case(case)
        field_panel(case, results, failures, reference, "density")
        field_panel(case, results, failures, reference, "mock_schlieren")
        linecuts(case, results, failures, reference)
        print(ROOT / f"figures/shockbubble_2d/{case}")


if __name__ == "__main__":
    main()
