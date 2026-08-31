#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

from for_paper_results import config


BASELINE_DIR = config.ROOT / "plots/WENO5_MLP/weno_double_reflective_1200"
FORMAL_METHODS = (
    ("weno5_js", config.METHODS["weno5_js"].label),
    ("weno5_sr_f64", config.METHODS["weno5_sr_f64"].label),
    ("weno5_sr_f32", config.METHODS["weno5_sr_f32"].label),
    ("weno7_js", config.METHODS["weno7_js"].label),
    ("weno7_sr_f64", config.METHODS["weno7_sr_f64"].label),
)
METHODS = (
    ("weno5_js", "WENO5-JS-RK3", "#4D4D4D"),
    ("weno5_sr_f64", config.METHODS["weno5_sr_f64"].label, config.METHODS["weno5_sr_f64"].color),
    ("weno5_sr_f32", config.METHODS["weno5_sr_f32"].label, config.METHODS["weno5_sr_f32"].color),
    ("weno7_js", config.METHODS["weno7_js"].label, config.METHODS["weno7_js"].color),
    ("weno7_sr_f64", config.METHODS["weno7_sr_f64"].label, config.METHODS["weno7_sr_f64"].color),
)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_completed_state(key: str) -> np.ndarray:
    meta_path = config.RAW / "double_mach" / f"{key}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        not meta.get("complete")
        or not meta.get("complete_time")
        or abs(float(meta["t"]) - 0.2) > 1.0e-14
        or int(meta.get("nan_count", -1)) != 0
        or float(meta.get("rho_min", -1.0)) <= 0.0
        or float(meta.get("p_min", -1.0)) <= 0.0
    ):
        raise RuntimeError(f"the selected {key} double-Mach result is incomplete or invalid")
    state = np.asarray(np.load(config.RAW / "double_mach" / f"{key}.npz")["state"])
    if state.shape != (300, 1200, 4) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"the selected {key} double-Mach state is invalid")
    return state


def make_formal_comparison() -> None:
    js_state = np.load(config.DOUBLE_MACH_WENO5_JS_STATE)[3:303, 3:1203, :]
    states = {
        "weno5_js": js_state,
        "weno5_sr_f64": load_completed_state("weno5_sr_f64"),
        "weno5_sr_f32": load_completed_state("weno5_sr_f32"),
        "weno7_js": load_completed_state("weno7_js"),
        "weno7_sr_f64": load_completed_state("weno7_sr_f64"),
    }
    if {state.shape for state in states.values()} != {(300, 1200, 4)}:
        raise RuntimeError("unexpected formal double-Mach state shape")

    densities = {key: state[..., 0] for key, state in states.items()}
    rho_min = min(float(np.min(rho)) for rho in densities.values())
    rho_max = max(float(np.max(rho)) for rho in densities.values())
    zoom = (2.05, 2.85, 0.0, 0.55)

    fig, axes = plt.subplots(
        5, 2, figsize=(12.4, 12.2), constrained_layout=True,
        gridspec_kw={"width_ratios": (3.4, 1.55)},
    )
    image = None
    for row, (key, label) in enumerate(FORMAL_METHODS):
        rho = densities[key]
        full_ax, zoom_ax = axes[row]
        image = full_ax.imshow(
            rho, origin="lower", extent=(0.0, 4.0, 0.0, 1.0),
            cmap="turbo", vmin=rho_min, vmax=rho_max,
            interpolation="nearest", aspect="equal",
        )
        full_ax.add_patch(
            Rectangle(
                (zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                fill=False, edgecolor="#102A83", linewidth=1.35,
            )
        )
        zoom_ax.imshow(
            rho, origin="lower", extent=(0.0, 4.0, 0.0, 1.0),
            cmap="turbo", vmin=rho_min, vmax=rho_max,
            interpolation="nearest", aspect="equal",
        )
        full_ax.set_title(label, fontsize=10)
        full_ax.set_xlim(0.0, 4.0)
        full_ax.set_ylim(0.0, 1.0)
        zoom_ax.set_xlim(zoom[0], zoom[1])
        zoom_ax.set_ylim(zoom[2], zoom[3])
        if row == 0:
            zoom_ax.set_title("Vortex-region enlargement", fontsize=10)
        for ax in (full_ax, zoom_ax):
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.tick_params(labelsize=8)
        for y_value, target_y in ((zoom[2], 0.0), (zoom[3], 1.0)):
            fig.add_artist(
                ConnectionPatch(
                    xyA=(zoom[1], y_value), coordsA=full_ax.transData,
                    xyB=(0.0, target_y), coordsB=zoom_ax.transAxes,
                    color="#102A83", linestyle="--", linewidth=0.65,
                    alpha=0.75,
                )
            )
    fig.colorbar(
        image, ax=axes.ravel().tolist(), label="Density $\\rho$",
        shrink=0.92, pad=0.02,
    )
    save(fig, config.FIGURES / "double_mach/double_mach_selected_models")


def load_states() -> dict[str, np.ndarray]:
    states = {
        "weno5_js": np.load(BASELINE_DIR / "weno5_classical.npy")[3:303, 3:1203, :],
    }
    for key in ("weno5_sr_f64", "weno5_sr_f32", "weno7_js", "weno7_sr_f64"):
        states[key] = np.load(config.RAW / "double_mach" / f"{key}.npz")["state"]
    shapes = {state.shape for state in states.values()}
    if shapes != {(300, 1200, 4)}:
        raise RuntimeError(f"unexpected double-Mach state shapes: {shapes}")
    return states


def write_table(rows: list[dict[str, str]]) -> None:
    row_map = {row["method"]: row for row in rows}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Density differences of the selected WENO-SR solutions from the classical baselines for the double-Mach-reflection problem at $t=0.2$.}",
        "\\label{tab:double_mach_differences}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & $L_1$ vs. WENO5-JS & $L_2$ vs. WENO5-JS & $L_1$ vs. WENO7-JS & $L_2$ vs. WENO7-JS \\\\",
        "\\midrule",
    ]
    for key in ("weno5_sr_f64", "weno5_sr_f32", "weno7_sr_f64"):
        row = row_map[key]
        lines.append(
            f"{config.METHODS[key].label} & {float(row['rho_l1_vs_weno5_js']):.3e} & "
            f"{float(row['rho_l2_vs_weno5_js']):.3e} & "
            f"{float(row['rho_l1_vs_weno7_js']):.3e} & "
            f"{float(row['rho_l2_vs_weno7_js']):.3e} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (config.TABLES / "double_mach_differences.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def main() -> None:
    make_formal_comparison()
    summary_path = config.RAW / "double_mach/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    failed = [row["method"] for row in summary["rows"] if not row["complete"]]
    if failed:
        print(
            "double-Mach figures skipped because validation failed for: "
            + ", ".join(failed)
        )
        return
    states = load_states()
    densities = {key: state[..., 0] for key, state in states.items()}
    x = (np.arange(1200) + 0.5) * (4.0 / 1200.0)
    y = (np.arange(300) + 0.5) / 300.0
    xx, yy = np.meshgrid(x, y)
    rho_min = min(float(np.min(rho)) for rho in densities.values())
    rho_max = max(float(np.max(rho)) for rho in densities.values())

    fig, axes = plt.subplots(5, 1, figsize=(12.2, 12.4), constrained_layout=True)
    image = None
    for ax, (key, label, _) in zip(axes, METHODS):
        image = ax.imshow(
            densities[key], origin="lower", extent=(0.0, 4.0, 0.0, 1.0),
            cmap="turbo", vmin=rho_min, vmax=rho_max,
            interpolation="nearest", aspect="equal",
        )
        ax.set_title(label)
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    fig.colorbar(image, ax=axes, label="Density $\\rho$", shrink=0.92)
    save(fig, config.FIGURES / "double_mach/double_mach_density")

    reference = densities["weno7_js"]
    errors = {
        key: np.abs(densities[key] - reference)
        for key in ("weno5_sr_f64", "weno5_sr_f32", "weno7_sr_f64")
    }
    error_max = max(float(np.max(error)) for error in errors.values())
    error_levels = np.linspace(0.0, error_max, 120)
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 8.2), constrained_layout=True)
    image = None
    for ax, key in zip(axes, errors):
        image = ax.contourf(xx, yy, errors[key], levels=error_levels, cmap="magma", extend="max")
        ax.set_title(f"{config.METHODS[key].label}: $|\\rho-\\rho_{{\\mathrm{{WENO7-JS}}}}|$")
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    fig.colorbar(image, ax=axes, label="Absolute density difference", shrink=0.92)
    save(fig, config.FIGURES / "double_mach/double_mach_rho_errors_vs_weno7")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.3), constrained_layout=True)
    for ax, target_y in zip(axes, (0.2, 0.4, 0.6)):
        index = int(np.argmin(np.abs(y - target_y)))
        for key, label, color in METHODS:
            ax.plot(x, densities[key][index], label=label, color=color, lw=1.0)
        ax.set_title(f"$y={y[index]:.3f}$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("Density $\\rho$")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, config.FIGURES / "double_mach/double_mach_linecuts")

    with (config.RAW / "double_mach/metrics.csv").open(newline="", encoding="utf-8") as stream:
        write_table(list(csv.DictReader(stream)))
    print(config.FIGURES / "double_mach")


if __name__ == "__main__":
    main()
