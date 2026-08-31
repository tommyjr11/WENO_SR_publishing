#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results import config
from for_paper_results.common import primitive


def levels_for_case(case: str) -> tuple[np.ndarray, str]:
    if case == "case6":
        return np.arange(0.54, 1.70 + 1.0e-12, 0.04), "rho054_170_step004"
    return np.arange(0.16, 1.71 + 1.0e-12, 0.05), "rho016_171_step005"


def plot_qstyle(state: np.ndarray, case: str, title: str, out_path: Path) -> None:
    pri = primitive(state)
    rho, vx, vy, pressure = (pri[..., index] for index in range(4))
    ny, nx = rho.shape
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    rho_levels, _ = levels_for_case(case)
    skip_x = max(1, nx // 30)
    skip_y = max(1, ny // 30)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    image = ax.contourf(xx, yy, pressure, levels=300, cmap="jet")
    ax.contour(xx, yy, rho, levels=rho_levels, colors="k", linewidths=0.3)
    ax.quiver(
        xx[::skip_y, ::skip_x], yy[::skip_y, ::skip_x],
        vx[::skip_y, ::skip_x], vy[::skip_y, ::skip_x],
        color="white", scale=40, width=0.002,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(image, ax=ax, label="Pressure $p$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_failure(case: str, title: str, metadata: dict, out_path: Path) -> None:
    failure_t = metadata.get("t")
    try:
        time_text = f"{float(failure_t):.8g}" if np.isfinite(float(failure_t)) else "not finite"
    except (TypeError, ValueError):
        time_text = "not available"
    fig, ax = plt.subplots(figsize=(6.2, 6.0), facecolor="white")
    ax.set_facecolor("white")
    ax.text(
        0.5, 0.55, "FAILED VALIDATION", ha="center", va="center",
        fontsize=17, fontweight="bold", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.45,
        f"t={time_text}\nnan_count={metadata.get('nan_count')}",
        ha="center", va="center", fontsize=11, transform=ax.transAxes,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for case in ("case6", "q400"):
        raw = config.RAW / case
        out = config.FIGURES / case / "qstyle"
        _, suffix = levels_for_case(case)
        for key in config.EULER_METHODS:
            path = out / f"{key}_pressure_rho_quiver_{suffix}.png"
            metadata = json.loads((raw / "N400" / f"{key}.json").read_text())
            title = f"{config.METHODS[key].label}, $400^2$"
            if metadata["complete"]:
                state = np.load(raw / "N400" / f"{key}.npz")["state"]
                plot_qstyle(state, case, title, path)
            else:
                plot_failure(case, title, metadata, path)
        reference = np.load(raw / "reference1200/weno7_js.npz")["state"]
        plot_qstyle(
            reference, case, "WENO7-JS-RK4 reference, $1200^2$",
            out / f"reference_weno7_js_1200_pressure_rho_quiver_{suffix}.png",
        )
        print(out)


if __name__ == "__main__":
    main()
