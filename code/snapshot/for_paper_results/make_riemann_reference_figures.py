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
from for_paper_results.make_riemann_figures import (
    CASES,
    DENSITY_CONTOUR_LEVELS,
    save,
)


REFERENCE_CASES = ("c4", "c5", "c6")


def load_reference(case: str) -> np.ndarray:
    raw = config.RAW / case / "N1200"
    metadata_path = raw / "weno5_js.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("complete", False):
            raise RuntimeError(f"{case}/N1200/weno5_js did not complete")
    state = np.asarray(np.load(raw / "weno5_js.npz")["state"])
    if state.shape != (1200, 1200, 4) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"{case}/N1200/weno5_js has an invalid state")
    return primitive(state)


def make_case(case: str) -> None:
    value = load_reference(case)
    centered = case == "c4"
    extent = (-0.5, 0.5, -0.5, 0.5) if centered else (0.0, 1.0, 0.0, 1.0)
    x = (np.arange(1200, dtype=np.float64) + 0.5) / 1200.0
    if centered:
        x = x - 0.5

    fig, ax = plt.subplots(figsize=(5.35, 4.75), constrained_layout=True)
    image = ax.imshow(
        value[..., 3], origin="lower", extent=extent, cmap="turbo",
        vmin=float(np.min(value[..., 3])), vmax=float(np.max(value[..., 3])),
        interpolation="nearest", rasterized=True,
    )
    ax.contour(
        x, x, value[..., 0], levels=DENSITY_CONTOUR_LEVELS[case],
        colors="black", linewidths=0.34, alpha=0.80,
    )
    if case != "c4":
        stride = 60
        xx, yy = np.meshgrid(x[::stride], x[::stride])
        speed_max = float(np.max(np.hypot(value[..., 1], value[..., 2])))
        ax.quiver(
            xx, yy, value[::stride, ::stride, 1], value[::stride, ::stride, 2],
            color="white", alpha=0.82, pivot="mid",
            scale=max(1.0, 32.0 * speed_max), width=0.0024,
            headwidth=3.2, headlength=4.2,
        )
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if centered:
        ax.set_xticks(np.linspace(-0.5, 0.5, 5))
        ax.set_yticks(np.linspace(-0.5, 0.5, 5))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(
        f"{CASES[case]['label']}: WENO5-JS-RK3, $1200^2$, "
        f"$t={CASES[case]['t_end']:.2f}$"
    )
    fig.colorbar(image, ax=ax, label="Pressure $p$", shrink=0.88)
    save(
        fig,
        config.FIGURES / "riemann" / "reference1200"
        / f"riemann_{case}_weno5_js_N1200",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*REFERENCE_CASES, "all"), default="all")
    args = parser.parse_args()
    cases = REFERENCE_CASES if args.case == "all" else (args.case,)
    for case in cases:
        make_case(case)
        print(
            config.FIGURES / "riemann" / "reference1200"
            / f"riemann_{case}_weno5_js_N1200.png"
        )


if __name__ == "__main__":
    main()
