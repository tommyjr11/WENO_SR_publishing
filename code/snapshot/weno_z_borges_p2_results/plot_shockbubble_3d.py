#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from warp_weno5_3d_rk3.binary_io import read_step
from warp_weno5_3d_rk3.config import ShockBubbleConfig
from warp_weno7_3d_rk4.plot_midplane import mock_schlieren


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
NATIVE = ROOT / "raw/shockbubble_3d/N224x88x88"
REFERENCE_DATA = Path("/home/ruijie/data_shock_bubble_3D_4th/data_3D")
REFERENCE_STEP = 7048

METHODS = (
    (
        "WENO5-JS-RK3",
        REPO / "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88/weno5_js/step_0823.bin",
        None,
    ),
    (
        "WENO5-Z-RK3",
        NATIVE / "weno5_z_p2",
        NATIVE / "weno5_z_p2/run_manifest.json",
    ),
    (
        "WENO5-SR-RK3",
        REPO / "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88/weno5_sr_f64_v20_step012250/step_0863.bin",
        None,
    ),
    (
        "WENO5-SR-FP32-RK3",
        REPO / "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88/weno5_sr_f32_v20_step016500/step_0863.bin",
        None,
    ),
    (
        "WENO7-JS-RK4",
        REPO / "warp_weno7_3d_rk4/runs/ma3_js_rk4_t0001_cfl025_N224x88x88/step_0841.bin",
        None,
    ),
    (
        "WENO7-Z-RK4",
        NATIVE / "weno7_z_p3",
        NATIVE / "weno7_z_p3/run_manifest.json",
    ),
    (
        "WENO7-SR-RK4",
        REPO / "warp_weno7_3d_rk4/runs/ma3_sr_f64_step016750_rk4_t0001_cfl025_N224x88x88/step_0871.bin",
        None,
    ),
)


def latest_binary(path: Path) -> Path | None:
    if path.is_file():
        return path
    candidates = sorted(path.glob("step_*.bin"))
    return candidates[-1] if candidates else None


def load_record(label: str, path: Path, manifest_path: Path | None) -> dict[str, object]:
    manifest = None
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        if not manifest.get("complete", False):
            return {
                "label": label,
                "complete": False,
                "failure": manifest.get("failure") or "incomplete run",
                "step": manifest.get("step"),
            }
    binary = latest_binary(path)
    if binary is None:
        return {"label": label, "complete": False, "failure": "result unavailable", "step": None}
    time, primitive = read_step(binary)
    if primitive.shape != (88, 88, 224, 5):
        raise ValueError(f"{label}: unexpected shape {primitive.shape}")
    density_midplane = primitive[primitive.shape[0] // 2, ..., 0].copy()
    return {
        "label": label,
        "complete": True,
        "time": time,
        "density_midplane": density_midplane,
        "dx": 0.225 / primitive.shape[2],
        "dy": 0.089 / primitive.shape[1],
        "binary": binary,
    }


def load_reference_midplane() -> dict[str, object]:
    files = sorted(REFERENCE_DATA.glob(f"step_{REFERENCE_STEP:04d}_*.bin"))
    if not files:
        return {
            "label": "TR1W reference\n$1120\\times440^2$",
            "complete": False,
            "failure": "reference unavailable",
            "step": REFERENCE_STEP,
        }

    pieces = []
    time = None
    nz = ny = None
    for path in files:
        with path.open("rb") as stream:
            header = np.fromfile(stream, dtype=np.uint32, count=3)
            if header.size != 3:
                raise ValueError(f"{path}: incomplete header")
            local_nz, local_ny, local_nx = (int(value) for value in header)
            local_time = np.fromfile(stream, dtype=np.float64, count=1)
            if local_time.size != 1:
                raise ValueError(f"{path}: missing time")
            if time is None:
                time = float(local_time[0])
                nz, ny = local_nz, local_ny
            elif local_nz != nz or local_ny != ny or float(local_time[0]) != time:
                raise ValueError(f"{path}: inconsistent MPI reference metadata")

            plane_values = local_ny * local_nx * 5
            plane_offset = 20 + (local_nz // 2) * plane_values * 8
            stream.seek(plane_offset)
            raw = np.fromfile(stream, dtype=np.float64, count=plane_values)
            if raw.size != plane_values:
                raise ValueError(f"{path}: incomplete central z plane")
            pieces.append(raw.reshape(local_ny, local_nx, 5)[..., 0])

    density_midplane = np.concatenate(pieces, axis=1)
    return {
        "label": "TR1W reference\n$1120\\times440^2$",
        "complete": True,
        "time": time,
        "density_midplane": density_midplane,
        "dx": 0.225 / density_midplane.shape[1],
        "dy": 0.089 / density_midplane.shape[0],
        "binary": REFERENCE_DATA,
    }


def failed_panel(axis: plt.Axes, record: dict[str, object]) -> None:
    axis.set_facecolor("#fafafa")
    axis.text(0.5, 0.56, "Run failed", ha="center", va="center", transform=axis.transAxes,
              fontsize=11, weight="bold")
    detail = str(record.get("failure", "incomplete run"))
    step = record.get("step")
    if step is not None:
        detail = f"{detail}\nstep {step}"
    axis.text(0.5, 0.42, detail, ha="center", va="center", transform=axis.transAxes, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])


def save_panel(records: list[dict[str, object]], field_name: str, output_name: str) -> Path:
    config = ShockBubbleConfig(nx=224, ny=88, nz=88)
    extent = (config.x_start, config.x_end, config.y_start, config.y_end)
    valid = [record for record in records if record["complete"]]
    if not valid:
        raise RuntimeError("no complete three-dimensional result to plot")

    fields = []
    for record in valid:
        density = record["density_midplane"]
        fields.append(
            density if field_name == "density" else mock_schlieren(
                density, float(record["dx"]), float(record["dy"])
            )
        )
    if field_name == "density":
        vmin = float(min(np.min(field) for field in fields))
        vmax = float(max(np.max(field) for field in fields))
        cmap = "turbo"
        colorbar_label = r"Density $\rho$"
    else:
        vmin, vmax = 0.0, 1.0
        cmap = "gray"
        colorbar_label = "Mock-schlieren intensity"

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, axes = plt.subplots(2, 4, figsize=(12.3, 5.5), constrained_layout=True)
    axes_flat = list(axes.flat)
    image = None
    field_index = 0
    for axis, record in zip(axes_flat, records):
        axis.set_title(record["label"])
        if not record["complete"]:
            failed_panel(axis, record)
            continue
        field = fields[field_index]
        field_index += 1
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
    for axis in axes_flat[len(records):]:
        axis.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes_flat, label=colorbar_label, shrink=0.88, pad=0.015)
    fig.suptitle(
        r"Three-dimensional Ma=3 shock--bubble, central $z$ plane, $t=10^{-4}$"
    )
    out = ROOT / f"figures/shockbubble_3d/N224x88x88/{output_name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def main() -> None:
    records = [load_record(*method) for method in METHODS]
    records.append(load_reference_midplane())
    print(save_panel(records, "density", "density_all_methods_with_weno_z"))
    print(save_panel(records, "schlieren", "mock_schlieren_all_methods_with_weno_z"))


if __name__ == "__main__":
    main()
