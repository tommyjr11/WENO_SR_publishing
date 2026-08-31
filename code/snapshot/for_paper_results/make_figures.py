#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from for_paper_results import config
from for_paper_results.common import conservative_block_average, primitive


plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
})


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require_complete(rows: list[dict], methods: tuple[str, ...]) -> None:
    found = {row["method"] for row in rows if str(row.get("complete", "")).lower() in ("true", "1")}
    missing = set(methods) - found
    if missing:
        raise RuntimeError(f"validated results are missing for {sorted(missing)}")


def latex_table(path: Path, caption: str, label: str, columns: list[str],
                headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        "\\toprule", " & ".join(headers) + " \\\\", "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def gste_point(x: np.ndarray) -> np.ndarray:
    x = -1.0 + np.mod(np.asarray(x, dtype=np.float64) + 1.0, 2.0)
    delta = 0.005
    beta = np.log(2.0) / (36.0 * delta * delta)

    def gaussian(center: float) -> np.ndarray:
        return np.exp(-beta * np.square(x - center))

    def ellipse(center: float) -> np.ndarray:
        return np.sqrt(np.maximum(1.0 - np.square(10.0 * (x - center)), 0.0))

    profile = np.zeros_like(x)
    mask = (-0.8 < x) & (x < -0.6)
    profile[mask] = (
        gaussian(-0.705)[mask]
        + 4.0 * gaussian(-0.7)[mask]
        + gaussian(-0.695)[mask]
    ) / 6.0
    mask = (-0.4 < x) & (x < -0.2)
    profile[mask] = 1.0
    mask = (0.0 < x) & (x < 0.2)
    profile[mask] = 1.0 - np.abs(10.0 * (x[mask] - 0.1))
    mask = (0.4 < x) & (x < 0.6)
    profile[mask] = (
        ellipse(0.495)[mask]
        + 4.0 * ellipse(0.5)[mask]
        + ellipse(0.505)[mask]
    ) / 6.0
    return profile


def make_gste() -> None:
    raw = config.RAW / "gste"
    rows = read_rows(raw / "metrics.csv")
    methods = config.EULER_METHODS
    require_complete(rows, methods)
    row_map = {row["method"]: row for row in rows}
    numerical = {}
    x = None
    for key in methods:
        result = np.load(raw / f"{key}.npz")
        numerical[key] = np.asarray(result["final"])
        current_x = np.asarray(result["x"])
        if x is None:
            x = current_x
        elif not np.array_equal(x, current_x):
            raise RuntimeError(f"inconsistent GSTE grid for {key}")
    assert x is not None
    dense_x = np.linspace(-1.0, 1.0, 20001)
    dense_exact = gste_point(dense_x - 10.0)
    fig, ax = plt.subplots(figsize=(8.0, 3.6), constrained_layout=True)
    ax.plot(dense_x, dense_exact, "k-", lw=1.8, label="Exact")
    for key in methods:
        method = config.METHODS[key]
        ax.plot(
            x,
            numerical[key],
            color=method.color,
            linestyle=method.linestyle,
            marker="o",
            markevery=2,
            markersize=1.8,
            markerfacecolor="white" if "sr" in key else "none",
            markeredgewidth=0.4,
            label=method.label,
            alpha=0.94,
        )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.2)
    ax.legend(
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25),
        frameon=False,
    )
    save(fig, config.FIGURES / "gste/gste_selected_models")

    windows = (
        ("Gaussian", -0.83, -0.57),
        ("Square", -0.43, -0.17),
        ("Triangle", -0.03, 0.23),
        ("Semi-ellipse", 0.37, 0.63),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7))
    for ax, (title, left, right) in zip(axes.flat, windows):
        exact_mask = (dense_x >= left) & (dense_x <= right)
        point_mask = (x >= left) & (x <= right)
        ax.plot(
            dense_x[exact_mask], dense_exact[exact_mask],
            "k-", lw=1.8, label="Exact",
        )
        for key in methods:
            method = config.METHODS[key]
            ax.plot(
                x[point_mask], numerical[key][point_mask],
                color=method.color, linestyle=method.linestyle,
                marker="o", markersize=3.0,
                markerfacecolor="white" if "sr" in key else "none",
                markeredgewidth=0.55, label=method.label,
            )
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u$")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 0.985), frameon=False,
    )
    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.08, top=0.84,
        hspace=0.36, wspace=0.20,
    )
    save(fig, config.FIGURES / "gste/gste_selected_models_components")

    latex_table(
        config.TABLES / "gste_errors.tex",
        "Errors for GSTE advection on 200 control volumes at $t=10$. "
        "All schemes use CFL $0.6$; WENO5 uses SSPRK3 and WENO7 uses "
        "fourth-order SSP Runge--Kutta time integration.",
        "tab:gste_errors", ["method", "l1", "l2", "tv"],
        ["Method", "$L_1$", "$L_2$", "TV"],
        [[config.METHODS[key].label, f"{float(row_map[key]['l1']):.3e}",
          f"{float(row_map[key]['l2']):.3e}",
          f"{float(row_map[key]['tv']):.4f}"] for key in methods],
    )


def make_sod() -> None:
    raw = config.RAW / "sod/N51_t020"
    rows = read_rows(raw / "metrics.csv")
    require_complete(rows, config.EULER_METHODS)
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    exact_drawn = False
    for key in config.EULER_METHODS:
        data = np.load(raw / f"{key}.npz")
        if not exact_drawn:
            ax.plot(data["x"], data["exact_rho"], "k-", lw=2.1, label="Exact")
            exact_drawn = True
        method = config.METHODS[key]
        ax.plot(data["x"], data["rho"], color=method.color,
                linestyle=method.linestyle, label=method.label)
    ax.set_xlabel("$x$")
    ax.set_ylabel("Density $\\rho$")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, frameon=False)
    save(fig, config.FIGURES / "sod/sod_density")
    row_map = {row["method"]: row for row in rows}
    latex_table(
        config.TABLES / "sod_errors.tex",
        "Density errors for the one-dimensional Sod problem with $N=51$ at $t=0.2$.",
        "tab:sod_errors", ["method", "l1", "l2"],
        ["Method", "$L_1(\\rho)$", "$L_2(\\rho)$"],
        [[config.METHODS[key].label, f"{float(row_map[key]['rho_l1']):.3e}",
          f"{float(row_map[key]['rho_l2']):.3e}"] for key in config.EULER_METHODS],
    )


def make_vortex() -> None:
    raw = config.RAW / "vortex"
    rows = read_rows(raw / "metrics.csv")
    for key in config.EULER_METHODS:
        subset = [row for row in rows if row["method"] == key]
        if len(subset) != 4 or not all(row["complete"].lower() in ("true", "1") for row in subset):
            raise RuntimeError(f"incomplete vortex series for {key}")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for key in config.EULER_METHODS:
        subset = sorted((row for row in rows if row["method"] == key), key=lambda row: int(row["N"]))
        n = np.array([int(row["N"]) for row in subset])
        method = config.METHODS[key]
        for ax, metric, ylabel in zip(axes, ("rho_l1", "rho_l2"), ("$L_1(\\rho)$", "$L_2(\\rho)$")):
            ax.loglog(n, [float(row[metric]) for row in subset], marker="o",
                      color=method.color, linestyle=method.linestyle, label=method.label)
            ax.set_xlabel("$N$")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, config.FIGURES / "vortex/vortex_convergence")
    table_rows: list[list[str]] = []
    for key in config.EULER_METHODS:
        for row in sorted((row for row in rows if row["method"] == key), key=lambda row: int(row["N"])):
            table_rows.append([
                config.METHODS[key].label, row["N"], f"{float(row['rho_l1']):.3e}",
                "--" if not np.isfinite(float(row["rho_l1_order"])) else f"{float(row['rho_l1_order']):.2f}",
                f"{float(row['rho_l2']):.3e}",
                "--" if not np.isfinite(float(row["rho_l2_order"])) else f"{float(row['rho_l2_order']):.2f}",
            ])
    latex_table(
        config.TABLES / "vortex_convergence.tex",
        "Density errors and observed orders for the isentropic vortex at $t=2$.",
        "tab:vortex_convergence", ["method", "N", "l1", "o1", "l2", "o2"],
        ["Method", "$N$", "$L_1$", "order", "$L_2$", "order"], table_rows,
    )


def quadrant_fields(case: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, dict]]:
    raw = config.RAW / case
    reference = np.load(raw / "reference1200/weno7_js.npz")["state"]
    reference400 = conservative_block_average(reference, 400)
    fields: dict[str, np.ndarray] = {}
    failed: dict[str, dict] = {}
    for key in config.EULER_METHODS:
        metadata = json.loads((raw / "N400" / f"{key}.json").read_text())
        if metadata["complete"]:
            fields[key] = np.load(raw / "N400" / f"{key}.npz")["state"]
        else:
            failed[key] = metadata
    return fields, reference400, failed


def make_quadrant(case: str) -> None:
    fields, reference, failed = quadrant_fields(case)
    if not fields:
        raise RuntimeError(f"no complete methods for {case}")
    pri = {key: primitive(value) for key, value in fields.items()}
    ref_pri = primitive(reference)
    pmin = min(float(np.min(value[..., 3])) for value in pri.values())
    pmax = max(float(np.max(value[..., 3])) for value in pri.values())
    levels_p = np.linspace(pmin, pmax, 181)
    rho_levels = np.arange(0.54, 1.70 + 1e-12, 0.04) if case == "case6" else np.arange(0.16, 1.71 + 1e-12, 0.05)
    x = (np.arange(400) + 0.5) / 400
    xx, yy = np.meshgrid(x, x)
    fig, axes = plt.subplots(
        1, len(fields), figsize=(3.05 * len(fields), 3.2), constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, fields):
        value = pri[key]
        image = ax.contourf(xx, yy, value[..., 3], levels=levels_p, cmap="viridis", extend="both")
        ax.contour(xx, yy, value[..., 0], levels=rho_levels, colors="k", linewidths=0.25)
        ax.set_title(config.METHODS[key].label)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.colorbar(image, ax=axes, label="Pressure $p$", shrink=0.82)
    save(fig, config.FIGURES / case / f"{case}_solutions")

    errors = {key: np.abs(value[..., 0] - ref_pri[..., 0]) for key, value in pri.items()}
    vmax = max(float(np.max(value)) for value in errors.values())
    levels_e = np.linspace(0.0, vmax, 181)
    fig, axes = plt.subplots(
        1, len(fields), figsize=(3.05 * len(fields), 3.2), constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, fields):
        image = ax.contourf(xx, yy, errors[key], levels=levels_e, cmap="magma", extend="max")
        ax.set_title(config.METHODS[key].label)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.colorbar(image, ax=axes, label="$|\\rho-\\rho_{ref}|$", shrink=0.82)
    save(fig, config.FIGURES / case / f"{case}_rho_errors")

    if case == "case6":
        cuts = [("$y=0.42$", "h", 0.42), ("$y=0.50$", "h", 0.50),
                ("$x=0.43$", "v", 0.43), ("$x=0.62$", "v", 0.62)]
    else:
        cuts = [("$y=0.50$", "h", 0.50), ("$x=0.50$", "v", 0.50),
                ("$y=x$", "d", 0.0), ("$y=1-x$", "a", 0.0)]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), constrained_layout=True)
    for ax, (title, orient, position) in zip(axes.ravel(), cuts):
        idx = int(np.argmin(np.abs(x - position))) if orient in ("h", "v") else 0
        def cut(array: np.ndarray) -> np.ndarray:
            if orient == "h": return array[idx, :]
            if orient == "v": return array[:, idx]
            if orient == "d": return np.diag(array)
            return np.diag(np.fliplr(array))
        ax.plot(x, cut(ref_pri[..., 0]), "k-", lw=2.0, label="WENO7-JS-RK4 $1200^2$")
        for key in fields:
            method = config.METHODS[key]
            ax.plot(x, cut(pri[key][..., 0]), color=method.color,
                    linestyle=method.linestyle, label=method.label)
        ax.set_title(title)
        ax.set_xlabel("coordinate")
        ax.set_ylabel("$\\rho$")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=6, ncol=2)
    save(fig, config.FIGURES / case / f"{case}_linecuts")

    rows: list[list[str]] = []
    metrics: list[dict] = []
    for key in fields:
        delta = pri[key][..., 0] - ref_pri[..., 0]
        row = {
            "method": key,
            "complete": True,
            "rho_l1": float(np.mean(np.abs(delta))),
            "rho_l2": float(np.sqrt(np.mean(delta * delta))),
            "rho_linf": float(np.max(np.abs(delta))),
        }
        if case == "q400":
            sym = pri[key][..., 0] - pri[key][..., 0].T
            row["sym_l2"] = float(np.sqrt(np.mean(sym * sym)))
        metrics.append(row)
        table_row = [config.METHODS[key].label, f"{row['rho_l1']:.3e}",
                     f"{row['rho_l2']:.3e}", f"{row['rho_linf']:.3e}"]
        if case == "q400":
            table_row.append(f"{row['sym_l2']:.3e}")
        rows.append(table_row)
    for key, metadata in failed.items():
        metrics.append({
            "method": key,
            "complete": False,
            "failure_t": metadata.get("t"),
            "nan_count": metadata.get("nan_count"),
        })
    (config.RAW / case / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    columns = ["method", "l1", "l2", "linf"]
    headers = ["Method", "$L_1(\\rho)$", "$L_2(\\rho)$", "$L_\\infty(\\rho)$"]
    caption = f"Density errors for {case} against the conservatively coarsened $1200^2$ reference."
    if case == "q400":
        ref_delta = ref_pri[..., 0] - ref_pri[..., 0].T
        ref_sym_l2 = float(np.sqrt(np.mean(ref_delta * ref_delta)))
        diagnostics = {"reference_sym_rho_l2": ref_sym_l2,
                       "reference_sym_rho_linf": float(np.max(np.abs(ref_delta)))}
        (config.RAW / case / "reference_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
        columns.append("sym")
        headers.append("$E_{\\rm sym,2}$")
        caption += (
            f" The coarsened reference has $E_{{\\rm sym,2}}={ref_sym_l2:.3e}$, "
            "so reference-based q400 rankings must be interpreted with caution."
        )
        if failed:
            failed_labels = ", ".join(config.METHODS[key].label for key in failed)
            caption += f" Failed validation and omitted: {failed_labels}."
    latex_table(
        config.TABLES / f"{case}_errors.tex",
        caption, f"tab:{case}_errors", columns, headers, rows,
    )


def main() -> None:
    config.ensure_output_dirs()
    tasks = [
        ("gste", make_gste), ("sod", make_sod), ("vortex", make_vortex),
    ]
    for name, task in tasks:
        try:
            task()
            print(f"generated {name}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"TODO {name}: {exc}")


if __name__ == "__main__":
    main()
