#!/usr/bin/env python3
"""Plot the Double-Mach comparison with order-matched WENO-Z results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
METHODS = (
    "weno5_js", "weno5_z_p2", "weno5_sr_f64", "weno5_sr_f32",
    "weno7_js", "weno7_z_p3", "weno7_sr_f64",
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
    "weno5_js": ("#666666", "--", 1.05),
    "weno5_z_p2": ("#56B4E9", ":", 1.35),
    "weno5_sr_f64": ("#0072B2", "-", 1.15),
    "weno5_sr_f32": ("#009E73", "-.", 1.15),
    "weno7_js": ("#D55E00", "--", 1.05),
    "weno7_z_p3": ("#E69F00", ":", 1.35),
    "weno7_sr_f64": ("#CC79A7", "-", 1.20),
}

FULL_EXTENT = (0.0, 4.0, 0.0, 1.0)
VORTEX_ZOOM = (2.05, 2.85, 0.0, 0.55)
METHOD_GROUPS = (
    (
        "weno5",
        ("weno5_js", "weno5_z_p2", "weno5_sr_f64", "weno5_sr_f32"),
        "WENO5 family",
    ),
    (
        "weno7",
        ("weno7_js", "weno7_z_p3", "weno7_sr_f64"),
        "WENO7 family",
    ),
)


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_states() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    states = {
        "weno5_js": np.asarray(
            np.load(REPO / "plots/WENO5_MLP/weno_double_reflective_1200/weno5_classical.npy")[
                3:303, 3:1203, :
            ], dtype=np.float64,
        ),
    }
    failures: dict[str, str] = {}
    old = REPO / "for_paper_results/raw/double_mach"
    new = ROOT / "raw/double_mach/N1200x300"
    for method in METHODS[1:]:
        base = new if "_z_" in method else old
        meta_path = base / f"{method}.json"
        npz_path = base / f"{method}.npz"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not metadata.get("complete", False):
            failures[method] = f"failed at step {metadata.get('steps', '?')}"
            continue
        with np.load(npz_path) as data:
            state = np.asarray(data["state"], dtype=np.float64)
        if state.shape != (300, 1200, 4) or not np.all(np.isfinite(state)):
            failures[method] = "invalid state"
            continue
        states[method] = state
    return states, failures


def field_panel(states: dict[str, np.ndarray], failures: dict[str, str], *, zoom: bool) -> None:
    rho = {key: state[..., 0] for key, state in states.items()}
    vmin = min(float(value.min()) for value in rho.values())
    vmax = max(float(value.max()) for value in rho.values())
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 5.8), constrained_layout=True,
                             sharex=True, sharey=True)
    image = None
    for axis, method in zip(axes.flat, METHODS):
        axis.set_title(LABELS[method], fontsize=9.4)
        if method in rho:
            image = axis.imshow(
                rho[method], origin="lower", extent=(0.0, 4.0, 0.0, 1.0),
                cmap="turbo", vmin=vmin, vmax=vmax,
                interpolation="nearest", aspect="equal",
            )
        else:
            axis.set_facecolor("#f6f6f6")
            axis.text(0.5, 0.54, "Run failed", ha="center", va="center",
                      transform=axis.transAxes, weight="bold")
            axis.text(0.5, 0.43, failures[method], ha="center", va="center",
                      transform=axis.transAxes, fontsize=8)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        if zoom:
            axis.set_xlim(2.05, 2.85)
            axis.set_ylim(0.0, 0.55)
        else:
            axis.set_xlim(0.0, 4.0)
            axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
    axes.flat[-1].axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=r"Density $\rho$", shrink=0.90)
    fig.suptitle(
        "Double-Mach reflection at $t=0.2$" + (": vortex-region enlargement" if zoom else "")
    )
    suffix = "density_vortex_zoom_with_weno_z" if zoom else "density_with_weno_z"
    save(fig, ROOT / f"figures/double_mach/{suffix}")


def full_and_contour_zoom_panels(
    states: dict[str, np.ndarray], failures: dict[str, str]
) -> None:
    """Plot each method as a full field beside a common contour enlargement."""
    rho = {key: state[..., 0] for key, state in states.items()}
    vmin = min(float(value.min()) for value in rho.values())
    vmax = max(float(value.max()) for value in rho.values())
    contour_levels = np.linspace(vmin, vmax, 31)[1:-1]
    x = (np.arange(1200) + 0.5) * (4.0 / 1200.0)
    y = (np.arange(300) + 0.5) / 300.0

    for tag, methods, group_title in METHOD_GROUPS:
        nrows = len(methods)
        fig, axes = plt.subplots(
            nrows,
            2,
            figsize=(12.4, 2.48 * nrows + 0.65),
            constrained_layout=True,
            gridspec_kw={"width_ratios": (3.35, 1.72)},
        )
        if nrows == 1:
            axes = np.asarray([axes])
        image = None
        for row, method in enumerate(methods):
            full_ax, zoom_ax = axes[row]
            if method not in rho:
                for axis in (full_ax, zoom_ax):
                    axis.set_facecolor("#f6f6f6")
                    axis.text(
                        0.5,
                        0.54,
                        "Run failed",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        weight="bold",
                    )
                    axis.text(
                        0.5,
                        0.43,
                        failures[method],
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=8,
                    )
                continue

            density = rho[method]
            image = full_ax.imshow(
                density,
                origin="lower",
                extent=FULL_EXTENT,
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
                rasterized=True,
            )
            full_ax.add_patch(
                Rectangle(
                    (VORTEX_ZOOM[0], VORTEX_ZOOM[2]),
                    VORTEX_ZOOM[1] - VORTEX_ZOOM[0],
                    VORTEX_ZOOM[3] - VORTEX_ZOOM[2],
                    fill=False,
                    edgecolor="#111111",
                    linewidth=1.15,
                )
            )
            zoom_ax.imshow(
                density,
                origin="lower",
                extent=FULL_EXTENT,
                cmap="turbo",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
                rasterized=True,
            )
            zoom_ax.contour(
                x,
                y,
                density,
                levels=contour_levels,
                colors="#161616",
                linewidths=0.38,
                alpha=0.93,
                antialiased=True,
            )

            full_ax.set_title(LABELS[method], fontsize=10.0, pad=4)
            if row == 0:
                zoom_ax.set_title("Vortex-region enlargement", fontsize=10.0, pad=4)
            full_ax.set_xlim(FULL_EXTENT[0], FULL_EXTENT[1])
            full_ax.set_ylim(FULL_EXTENT[2], FULL_EXTENT[3])
            zoom_ax.set_xlim(VORTEX_ZOOM[0], VORTEX_ZOOM[1])
            zoom_ax.set_ylim(VORTEX_ZOOM[2], VORTEX_ZOOM[3])
            for axis in (full_ax, zoom_ax):
                axis.set_xlabel(r"$x$")
                axis.set_ylabel(r"$y$")
                axis.tick_params(labelsize=8)
                axis.set_aspect("equal", adjustable="box")

            for y_value, target_y in (
                (VORTEX_ZOOM[2], 0.0),
                (VORTEX_ZOOM[3], 1.0),
            ):
                fig.add_artist(
                    ConnectionPatch(
                        xyA=(VORTEX_ZOOM[1], y_value),
                        coordsA=full_ax.transData,
                        xyB=(0.0, target_y),
                        coordsB=zoom_ax.transAxes,
                        color="#333333",
                        linestyle=(0, (3, 2)),
                        linewidth=0.58,
                        alpha=0.72,
                    )
                )

        if image is not None:
            fig.colorbar(
                image,
                ax=axes.ravel().tolist(),
                label=r"Density $\rho$",
                shrink=0.94,
                pad=0.018,
            )
        fig.suptitle(
            rf"Double-Mach reflection at $t=0.2$: {group_title}", fontsize=12.2
        )
        save(
            fig,
            ROOT
            / "figures/double_mach"
            / f"double_mach_{tag}_full_and_contour_zoom",
        )


def linecuts(states: dict[str, np.ndarray], failures: dict[str, str]) -> None:
    x = (np.arange(1200) + 0.5) * (4.0 / 1200.0)
    y = (np.arange(300) + 0.5) / 300.0
    reference = states["weno7_js"][..., 0]
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), constrained_layout=True)
    for axis, requested in zip(axes, (0.2, 0.4, 0.6)):
        index = int(np.argmin(np.abs(y - requested)))
        ref_profile = reference[index]
        for method in METHODS:
            if method not in states:
                rows.append({
                    "requested_y": requested, "actual_y": float(y[index]),
                    "method": method, "status": failures[method],
                    "l1_vs_weno7_js": np.nan, "l2_vs_weno7_js": np.nan,
                    "linf_vs_weno7_js": np.nan,
                })
                continue
            values = states[method][index, :, 0]
            error = values - ref_profile
            rows.append({
                "requested_y": requested, "actual_y": float(y[index]),
                "method": method, "status": "complete",
                "l1_vs_weno7_js": float(np.mean(np.abs(error))),
                "l2_vs_weno7_js": float(np.sqrt(np.mean(error * error))),
                "linf_vs_weno7_js": float(np.max(np.abs(error))),
            })
            color, linestyle, linewidth = STYLES[method]
            axis.plot(x, values, color=color, linestyle=linestyle,
                      linewidth=linewidth, label=LABELS[method])
        axis.set_title(rf"$y={y[index]:.4f}$")
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"Density $\rho$")
        axis.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, fontsize=8)
    fig.suptitle("Double-Mach reflection: density line cuts")
    save(fig, ROOT / "figures/double_mach/density_linecuts_with_weno_z")
    table = ROOT / "tables/double_mach_linecuts_with_weno_z.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    states, failures = load_states()
    field_panel(states, failures, zoom=False)
    field_panel(states, failures, zoom=True)
    full_and_contour_zoom_panels(states, failures)
    linecuts(states, failures)
    print(ROOT / "figures/double_mach")


if __name__ == "__main__":
    main()
