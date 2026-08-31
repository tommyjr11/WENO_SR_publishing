#!/usr/bin/env python3
"""Add WENO7-Z to the established C.3--C.6 field and line-cut figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results.common import primitive
from for_paper_results.make_riemann_figures import DENSITY_CONTOUR_LEVELS


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
METHODS = (
    "weno5_js", "weno5_sr_f64", "weno5_sr_f32",
    "weno7_js", "weno7_z_p3", "weno7_sr_f64",
)
LABELS = {
    "weno5_js": "WENO5-JS-RK3",
    "weno5_sr_f64": "WENO5-SR-RK3",
    "weno5_sr_f32": "WENO5-SR-FP32-RK3",
    "weno7_js": "WENO7-JS-RK4",
    "weno7_z_p3": "WENO7-Z-RK4",
    "weno7_sr_f64": "WENO7-SR-RK4",
}
STYLES = {
    "reference": ("#111111", "-", 1.65),
    "weno5_js": ("#666666", "--", 1.05),
    "weno5_sr_f64": ("#0072B2", "-", 1.15),
    "weno5_sr_f32": ("#009E73", "-.", 1.15),
    "weno7_js": ("#D55E00", "--", 1.05),
    "weno7_z_p3": ("#E69F00", ":", 1.35),
    "weno7_sr_f64": ("#CC79A7", "-", 1.20),
}
CASES = {
    "c3": {"label": "C.3", "t": 0.50, "raw": "q400"},
    "c4": {"label": "C.4", "t": 0.25, "raw": "c4"},
    "c5": {"label": "C.5", "t": 0.23, "raw": "c5"},
    "c6": {"label": "C.6", "t": 0.30, "raw": "c6"},
}
CUTS = {
    "c3": (("y", 0.42), ("y", 0.50), ("x", 0.43), ("x", 0.62)),
    "c4": (("y", 0.50), ("y", 0.44), ("y", 0.715), ("y", 0.34)),
    "c5": (("y", 0.35), ("y", 0.665), ("x", 0.665), ("x", 0.745)),
    "c6": (("y", 0.305), ("y", 0.465), ("y", 0.655), ("x", 0.460)),
}


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def source(case: str, method: str) -> tuple[Path, Path | None]:
    if method == "weno7_z_p3":
        base = ROOT / f"raw/riemann/{case}/N400"
        return base / f"{method}.npz", base / f"{method}.json"
    raw_case = CASES[case]["raw"]
    base = REPO / f"for_paper_results/raw/{raw_case}/N400"
    return base / f"{method}.npz", base / f"{method}.json"


def load_case(case: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    fields: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    for method in METHODS:
        npz_path, json_path = source(case, method)
        metadata = {}
        if json_path is not None and json_path.is_file():
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if not npz_path.is_file():
            failures[method] = "result unavailable"
            continue
        if metadata and not metadata.get("complete", False):
            failures[method] = f"failed at step {metadata.get('steps', '?')}"
            continue
        with np.load(npz_path) as data:
            state = np.asarray(data["state"], dtype=np.float64)
        if state.shape != (400, 400, 4) or not np.all(np.isfinite(state)):
            failures[method] = "invalid state"
            continue
        fields[method] = primitive(state)
    return fields, failures


def load_reference(case: str) -> np.ndarray:
    raw_case = CASES[case]["raw"]
    method = "weno7_js" if case == "c3" else "weno5_js"
    path = REPO / f"for_paper_results/raw/{raw_case}/reference1200/{method}.npz"
    if not path.is_file():
        path = REPO / f"for_paper_results/raw/{raw_case}/N1200/{method}.npz"
    with np.load(path) as data:
        state = np.asarray(data["state"], dtype=np.float64)
    if state.shape != (1200, 1200, 4):
        raise ValueError(f"unexpected reference shape for {case}: {state.shape}")
    return state[..., 0].reshape(400, 3, 400, 3).mean(axis=(1, 3))


def make_field_figure(
    case: str,
    fields: dict[str, np.ndarray],
    failures: dict[str, str],
    *,
    hybrid_pdf: bool = False,
) -> None:
    complete = list(fields.values())
    p_min = min(float(item[..., 3].min()) for item in complete)
    p_max = max(float(item[..., 3].max()) for item in complete)
    pressure_levels = np.linspace(p_min, p_max, 241)
    density_levels = DENSITY_CONTOUR_LEVELS[case]
    coordinates = (np.arange(400) + 0.5) / 400.0
    if case == "c4":
        coordinates = coordinates - 0.5
    xx, yy = np.meshgrid(coordinates, coordinates)
    limits = (-0.5, 0.5) if case == "c4" else (0.0, 1.0)
    velocity_max = max(
        float(np.max(np.hypot(value[..., 1], value[..., 2])))
        for value in complete
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.0), constrained_layout=True)
    image = None
    for axis, method in zip(axes.flat, METHODS):
        if hybrid_pdf:
            # Keep labels, contours, and vectors as PDF primitives while
            # rasterizing only the dense 241-level pressure fill.
            axis.set_rasterization_zorder(0.0)
        axis.set_title(LABELS[method], fontsize=9.5)
        if method in fields:
            value = fields[method]
            image = axis.contourf(
                xx, yy, value[..., 3], levels=pressure_levels,
                cmap="turbo", extend="both", zorder=-1.0 if hybrid_pdf else 1.0,
            )
            axis.contour(
                xx, yy, value[..., 0], levels=density_levels,
                colors="black", linewidths=0.30 if case == "c3" else 0.36,
                alpha=1.0 if case == "c3" else 0.78, zorder=1.0,
            )
            if case != "c4":
                stride = 20
                axis.quiver(
                    xx[::stride, ::stride], yy[::stride, ::stride],
                    value[::stride, ::stride, 1], value[::stride, ::stride, 2],
                    color="white", alpha=0.82, pivot="mid",
                    scale=max(1.0, 32.0 * velocity_max), width=0.0024,
                    headwidth=3.2, headlength=4.2, zorder=2.0,
                )
        else:
            axis.set_facecolor("#f6f6f6")
            axis.text(0.5, 0.54, "Run failed", ha="center", va="center",
                      transform=axis.transAxes, weight="bold")
            axis.text(0.5, 0.44, failures[method], ha="center", va="center",
                      transform=axis.transAxes, fontsize=8)
        axis.set_aspect("equal")
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.tick_params(labelsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=r"Pressure $p$", shrink=0.90)
    fig.suptitle(
        f"Two-dimensional Riemann configuration {CASES[case]['label']} "
        f"at $t={CASES[case]['t']:.2f}$"
    )
    stem = ROOT / f"figures/riemann/{case}/fields_with_weno7_z"
    if hybrid_pdf:
        stem = stem.with_name(stem.name + "_hybrid")
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".pdf"), dpi=350, bbox_inches="tight")
        plt.close(fig)
    else:
        save(fig, stem)


def profile(field: np.ndarray, direction: str, value: float) -> tuple[np.ndarray, np.ndarray, float]:
    coordinates = (np.arange(400) + 0.5) / 400.0
    index = int(np.argmin(np.abs(coordinates - value)))
    if direction == "y":
        return coordinates, field[index, :], float(coordinates[index])
    return coordinates, field[:, index], float(coordinates[index])


def make_linecuts(case: str, fields: dict[str, np.ndarray], failures: dict[str, str]) -> None:
    reference = load_reference(case)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)
    rows: list[dict[str, object]] = []
    for axis, (direction, requested) in zip(axes.flat, CUTS[case]):
        coordinate, ref_profile, actual = profile(reference, direction, requested)
        axis.plot(coordinate, ref_profile, color=STYLES["reference"][0],
                  linestyle=STYLES["reference"][1], linewidth=STYLES["reference"][2],
                  label="1200 reference")
        for method in METHODS:
            if method not in fields:
                rows.append({
                    "case": case, "direction": direction, "requested": requested,
                    "actual": actual, "method": method, "status": failures[method],
                    "l1": np.nan, "l2": np.nan, "linf": np.nan,
                })
                continue
            _, values, _ = profile(fields[method][..., 0], direction, requested)
            error = values - ref_profile
            rows.append({
                "case": case, "direction": direction, "requested": requested,
                "actual": actual, "method": method, "status": "complete",
                "l1": float(np.mean(np.abs(error))),
                "l2": float(np.sqrt(np.mean(error * error))),
                "linf": float(np.max(np.abs(error))),
            })
            color, linestyle, linewidth = STYLES[method]
            axis.plot(coordinate, values, color=color, linestyle=linestyle,
                      linewidth=linewidth, label=LABELS[method])
        axis.set_title(rf"${direction}={actual:.4f}$")
        axis.set_xlabel(r"$x$" if direction == "y" else r"$y$")
        axis.set_ylabel(r"Density $\rho$")
        axis.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"{CASES[case]['label']}: density line cuts against the 1200-cell reference")
    out = ROOT / f"figures/riemann/{case}"
    save(fig, out / "density_linecuts_with_weno7_z")
    table = ROOT / f"tables/riemann_{case}_linecuts_with_weno7_z.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid-fields-only",
        action="store_true",
        help="write compact hybrid field PDFs without replacing established figures",
    )
    args = parser.parse_args()
    for case in CASES:
        fields, failures = load_case(case)
        make_field_figure(case, fields, failures, hybrid_pdf=args.hybrid_fields_only)
        if not args.hybrid_fields_only:
            make_linecuts(case, fields, failures)
        print(ROOT / f"figures/riemann/{case}")


if __name__ == "__main__":
    main()
