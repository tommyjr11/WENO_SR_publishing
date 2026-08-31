#!/usr/bin/env python3
"""Build the immutable WENO-SR review snapshot from the local research tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY.parent
SNAPSHOT = REPOSITORY / "code" / "snapshot"

CODE_SUFFIXES = {
    ".py",
    ".sh",
    ".sbatch",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "runs",
    "raw",
    "figures",
    "tables",
    "logs",
    "diagnostics",
    "audits",
    "probe",
    "beta_normalized_audit",
    "raw_beta_tiny_epsilon_audit",
    "weno7-fast-more",
}

# Older version directories are retained only as the dependency closure of the
# final WENO5 FP64 entry point, not as alternate experiments or training runs.
TREE_ALLOWLISTS = {
    "teacherfree_lab_weno5_v4_fvm_e2e": {
        "__init__.py", "apost_advect_fvm.py", "fvm_profiles.py", "train_weno5_v4.py",
    },
    "teacherfree_lab_weno5_v5_fvm_e2e": {
        "__init__.py", "v5_losses.py",
    },
    "teacherfree_lab_weno5_v6_long": {
        "__init__.py", "gste_monitor.py", "train_weno5_v6.py",
    },
    "teacherfree_lab_weno5_v9_exact_fvm": {
        "__init__.py", "v9_model.py",
    },
    "teacherfree_lab_weno5_v12_reflection_sym": {
        "__init__.py", "v12_losses.py", "v12_model.py",
        "warp_v12/__init__.py", "warp_v12/run_weno5_circle_mlp_compare_v12.py",
        "warp_v12/weno5_rk3_diff_v12.py",
    },
    "teacherfree_lab_weno5_v19_autoregressive_js": {
        "__init__.py", "v19_losses.py",
    },
}

# These are the final training packages and the exact dependency/runtime trees
# used by the paper. Historical directories are copied only when imported by a
# selected training or inference entry point.
CODE_TREES = (
    "teacherfree_lab_weno5",
    "teacherfree_lab_weno5_v4_fvm_e2e",
    "teacherfree_lab_weno5_v5_fvm_e2e",
    "teacherfree_lab_weno5_v6_long",
    "teacherfree_lab_weno5_v9_exact_fvm",
    "teacherfree_lab_weno5_v12_reflection_sym",
    "teacherfree_lab_weno5_v19_autoregressive_js",
    "teacherfree_lab_weno5_v20_distance_balanced",
    "teacherfree_lab_weno5_mlp_f32",
    "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32",
    "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast",
    "teacherfree_lab_weno7_rk4_distance_balanced_fast",
    "teacherfree_lab_weno7_rk4_distance_balanced_fast-2",
    "weno7_external_clean",
    "weno7_point_rk4_shu",
    "for_paper_results",
    "weno_z_borges_p2_results",
    "warp_weno5_3d_rk3",
    "warp_weno7_3d_rk4",
)

ROOT_CODE_FILES = (
    "compute_soomth_WENO.py",
    "pretrain_weno5_offline.py",
    "pretrain_weno7_offline.py",
    "warp_weno5_helpers.py",
    "weno5_rk3_diff.py",
    "weno5_rk3_forward.py",
    "weno5_rk3_warp.py",
    "run_weno5_circle_mlp_compare.py",
    "run_weno5_smooth_periodic.py",
    "run_weno5_quadrant_mlp_compare.py",
    "run_weno5_quadrant_mlp_only.py",
    "run_double_mach_compare.py",
    "warp_weno7_ader4_helpers.py",
    "weno7_ader4_warp.py",
    "warp_weno7_external_mlp_normal.py",
)

# The Mach-3 planar shock--bubble benchmark used a dedicated, otherwise
# byte-for-byte-compatible runner with the Mach number and post-shock state
# changed from the Mach-1.22 publication runner.  Preserve that actual source
# path instead of reconstructing the initial condition in a release wrapper.
EXTRA_CODE_FILES = (
    "shockbubble_ma3_t0001_cfl04_server/for_paper_results/"
    "run_weno5_shockbubble.py",
)

MODELS = {
    "weno5_sr_fp64_step012250.npz": (
        "teacherfree_lab_weno5_v20_distance_balanced/runs/"
        "apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/"
        "model_step_012250.npz"
    ),
    "weno5_sr_fp32_step016500.npz": (
        "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/runs/"
        "apost_weno5_v20_mlp_f32_fast_200k/checkpoints/"
        "model_step_016500.npz"
    ),
    "weno7_sr_fp64_step016750.npz": (
        "teacherfree_lab_weno7_rk4_distance_balanced_fast-2/runs/"
        "apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/"
        "model_step_016750.npz"
    ),
}

MODEL_ALIASES = {
    (
        "teacherfree_lab_weno5_v20_distance_balanced/runs/"
        "apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/"
        "model_step_012250.npz"
    ): "weno5_sr_fp64_step012250.npz",
    (
        "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/runs/"
        "apost_weno5_v20_mlp_f32_fast_200k/checkpoints/"
        "model_step_016500.npz"
    ): "weno5_sr_fp32_step016500.npz",
    (
        "teacherfree_lab_weno7_rk4_distance_balanced_fast/runs/"
        "apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/"
        "model_step_016750.npz"
    ): "weno7_sr_fp64_step016750.npz",
}

EXPERIMENT_ASSETS = {
    "01_gste_long_time_advection": {
        "figures": {
            "gste_all_methods.pdf": (
                "weno_z_borges_p2_results/figures/gste/N200_t10_cfl06/"
                "gste_all_methods_with_weno_z_rminus1.pdf"
            ),
        },
        "tables": {
            "gste_l1.tex": "for_paper_results/tables/gste_l1_weno_z.tex",
            "gste_metrics.csv": (
                "weno_z_borges_p2_results/tables/gste/N200_t10_cfl06/"
                "gste_errors_with_weno_z_rminus1.csv"
            ),
        },
    },
    "02_sod_shock_tube": {
        "figures": {
            "sod_density.pdf": (
                "weno_z_borges_p2_results/figures/riemann_1d/sod/"
                "N100_t020_cfl08/sod_density_points_with_weno_z_rminus1.pdf"
            ),
        },
        "tables": {
            "sod_l1.tex": "for_paper_results/tables/sod_n100_l1_weno_z.tex",
            "sod_metrics.csv": (
                "weno_z_borges_p2_results/tables/riemann_1d/sod/"
                "N100_t020_cfl08/sod_density_errors_vs_exact_fv.csv"
            ),
        },
    },
    "03_lax_shock_tube": {
        "figures": {
            "lax_primitive.pdf": (
                "weno_z_borges_p2_results/figures/riemann_1d/lax/"
                "lax_primitive_compare_with_weno_z_rminus1.pdf"
            ),
        },
        "tables": {
            "lax_l1.tex": "for_paper_results/tables/lax_l1_weno_z.tex",
            "lax_metrics.csv": (
                "weno_z_borges_p2_results/tables/riemann_1d/lax/"
                "lax_errors_vs_exact_fv.csv"
            ),
        },
    },
    "04_titarev_toro": {
        "figures": {
            "titarev_toro_N1001.pdf": (
                "weno_z_borges_p2_results/figures/titarev_toro_cfl08/"
                "N1001x10/titarev_density_with_weno_z_rminus1.pdf"
            ),
            "titarev_toro_N2000.pdf": (
                "weno_z_borges_p2_results/figures/titarev_toro_cfl08/"
                "N2000x10/titarev_density_with_weno_z_rminus1.pdf"
            ),
        },
        "tables": {
            "titarev_toro_l1.tex": "for_paper_results/tables/titarev_toro_l1_weno_z.tex",
            "titarev_toro_N1001_metrics.csv": (
                "weno_z_borges_p2_results/tables/titarev_toro_cfl08/"
                "N1001x10_density_errors.csv"
            ),
            "titarev_toro_N2000_metrics.csv": (
                "weno_z_borges_p2_results/tables/titarev_toro_cfl08/"
                "N2000x10_density_errors.csv"
            ),
        },
    },
    "05_isentropic_vortex": {
        "figures": {
            "vortex_convergence.pdf": (
                "weno_z_borges_p2_results/figures/vortex_cfl04/"
                "vortex_convergence_with_weno_z.pdf"
            ),
        },
        "tables": {
            "vortex_convergence.tex": (
                "weno_z_borges_p2_results/tables/vortex_cfl04/"
                "vortex_convergence_with_weno_z.tex"
            ),
            "vortex_convergence.md": (
                "weno_z_borges_p2_results/tables/vortex_cfl04/"
                "vortex_convergence_with_weno_z.md"
            ),
        },
    },
    "06_riemann_c3": {
        "figures": {
            "riemann_c3.pdf": (
                "weno_z_borges_p2_results/figures/riemann/c3/"
                "fields_with_weno7_z_hybrid.pdf"
            ),
        },
        "tables": {
            "linecuts.csv": "weno_z_borges_p2_results/tables/riemann_c3_linecuts_with_weno7_z.csv",
        },
    },
    "07_riemann_c4": {
        "figures": {
            "riemann_c4.pdf": (
                "weno_z_borges_p2_results/figures/riemann/c4/"
                "fields_with_weno7_z_hybrid.pdf"
            ),
        },
        "tables": {
            "linecuts.csv": "weno_z_borges_p2_results/tables/riemann_c4_linecuts_with_weno7_z.csv",
        },
    },
    "08_riemann_c5": {
        "figures": {
            "riemann_c5.pdf": (
                "weno_z_borges_p2_results/figures/riemann/c5/"
                "fields_with_weno7_z_hybrid.pdf"
            ),
        },
        "tables": {
            "linecuts.csv": "weno_z_borges_p2_results/tables/riemann_c5_linecuts_with_weno7_z.csv",
        },
    },
    "09_riemann_c6": {
        "figures": {
            "riemann_c6.pdf": (
                "weno_z_borges_p2_results/figures/riemann/c6/"
                "fields_with_weno7_z_hybrid.pdf"
            ),
        },
        "tables": {
            "linecuts.csv": "weno_z_borges_p2_results/tables/riemann_c6_linecuts_with_weno7_z.csv",
        },
    },
    "10_double_mach_reflection": {
        "figures": {
            "double_mach_weno5.pdf": (
                "weno_z_borges_p2_results/figures/double_mach/"
                "double_mach_weno5_full_and_contour_zoom.pdf"
            ),
            "double_mach_weno7.pdf": (
                "weno_z_borges_p2_results/figures/double_mach/"
                "double_mach_weno7_full_and_contour_zoom.pdf"
            ),
        },
        "tables": {
            "linecuts.csv": "weno_z_borges_p2_results/tables/double_mach_linecuts_with_weno_z.csv",
        },
    },
    "11_shock_bubble_2d_ma122": {
        "figures": {
            "mock_schlieren.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_2d/ma122/"
                "mock_schlieren_with_weno_z.pdf"
            ),
            "selected_density_linecuts.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_2d/ma122/"
                "selected_density_linecuts_with_weno_z.pdf"
            ),
        },
        "tables": {
            "linecut_l1.csv": (
                "weno_z_borges_p2_results/tables/"
                "shockbubble_2d_ma122_linecuts_with_weno_z.csv"
            ),
        },
    },
    "12_shock_bubble_2d_ma30": {
        "figures": {
            "mock_schlieren.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_2d/ma30/"
                "mock_schlieren_with_weno_z.pdf"
            ),
            "selected_density_linecuts.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_2d/ma30/"
                "selected_density_linecuts_with_weno_z.pdf"
            ),
        },
        "tables": {
            "linecut_l1.csv": (
                "weno_z_borges_p2_results/tables/"
                "shockbubble_2d_ma30_linecuts_with_weno_z.csv"
            ),
        },
    },
    "13_shock_bubble_3d_ma30": {
        "figures": {
            "initial_configuration.png": "for_paper_results/figures/shock_bubble.png",
            "volume_rendering.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "volume_render_all_methods_with_reference.pdf"
            ),
            "mock_schlieren.png": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "mock_schlieren_all_methods_with_weno_z_with_red.png"
            ),
            "density_line_x0085.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "reference_line_profiles/"
                "density_line_reference_compare_liney_x0p0850_z0p0445_auto_full.pdf"
            ),
            "density_line_x0097.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "reference_line_profiles/"
                "density_line_reference_compare_liney_x0p0970_z0p0445_auto_full.pdf"
            ),
            "density_line_x0114.pdf": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "reference_line_profiles/"
                "density_line_reference_compare_liney_x0p1140_z0p0445_auto_full.pdf"
            ),
        },
        "tables": {
            "density_line_x0085_errors.csv": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "reference_line_profiles/"
                "density_line_reference_compare_liney_x0p0850_z0p0445_auto_full_errors.csv"
            ),
            "density_line_x0097_errors.csv": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "reference_line_profiles/"
                "density_line_reference_compare_liney_x0p0970_z0p0445_auto_full_errors.csv"
            ),
            "density_line_x0114_errors.csv": (
                "weno_z_borges_p2_results/figures/shockbubble_3d/N224x88x88/"
                "reference_line_profiles/"
                "density_line_reference_compare_liney_x0p1140_z0p0445_auto_full_errors.csv"
            ),
        },
    },
    "14_mixed_precision_timing": {
        "figures": {},
        "tables": {
            "weno5_precision_timing.tex": "for_paper_results/tables/weno5_precision_timing.tex",
        },
    },
}

PAPER_LOCATIONS = {
    "01_gste_long_time_advection": "Section 4, subsection label subsec:gste_results",
    "02_sod_shock_tube": "Section 4, subsection label subsec:sod_results",
    "03_lax_shock_tube": "Section 4, subsection label subsec:lax_results",
    "04_titarev_toro": "Section 4, subsection label subsec:titarev_toro_results",
    "05_isentropic_vortex": "Section 4, subsection label subsec:vortex_results",
    "06_riemann_c3": "Section 4, subsection label subsec:riemann_results (C.3)",
    "07_riemann_c4": "Section 4, subsection label subsec:riemann_results (C.4)",
    "08_riemann_c5": "Section 4, subsection label subsec:riemann_results (C.5)",
    "09_riemann_c6": "Section 4, subsection label subsec:riemann_results (C.6)",
    "10_double_mach_reflection": "Section 4, subsection label subsec:double_mach_status",
    "11_shock_bubble_2d_ma122": "Section 4, subsection label subsec:shock_bubble_results (Mach 1.22)",
    "12_shock_bubble_2d_ma30": "Section 4, subsection label subsec:shock_bubble_results (Mach 3)",
    "13_shock_bubble_3d_ma30": "Section 4, subsection label subsec:shock_bubble_3d_results",
    "14_mixed_precision_timing": "Section 4, subsection label subsec:mixed_precision_cost",
}

PLOT_ENTRY_POINTS = {
    "01_gste_long_time_advection": ["weno_z_borges_p2_results.plot_gste"],
    "02_sod_shock_tube": ["weno_z_borges_p2_results.plot_riemann_1d_zoom"],
    "03_lax_shock_tube": ["weno_z_borges_p2_results.plot_lax_paper"],
    "04_titarev_toro": ["weno_z_borges_p2_results.plot_titarev_toro"],
    "05_isentropic_vortex": ["weno_z_borges_p2_results.plot_vortex"],
    "06_riemann_c3": ["weno_z_borges_p2_results.plot_riemann"],
    "07_riemann_c4": ["weno_z_borges_p2_results.plot_riemann"],
    "08_riemann_c5": ["weno_z_borges_p2_results.plot_riemann"],
    "09_riemann_c6": ["weno_z_borges_p2_results.plot_riemann"],
    "10_double_mach_reflection": ["weno_z_borges_p2_results.plot_double_mach"],
    "11_shock_bubble_2d_ma122": ["weno_z_borges_p2_results.plot_shockbubble_2d"],
    "12_shock_bubble_2d_ma30": ["weno_z_borges_p2_results.plot_shockbubble_2d"],
    "13_shock_bubble_3d_ma30": [
        "weno_z_borges_p2_results.plot_shockbubble_3d",
        "weno_z_borges_p2_results.plot_shockbubble_3d_linecuts",
    ],
    "14_mixed_precision_timing": [],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source_relative: str, destination: Path) -> dict[str, object]:
    source = SOURCE / source_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": source_relative,
        "destination": destination.relative_to(REPOSITORY).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def copy_code_tree(relative: str) -> list[dict[str, object]]:
    source_root = SOURCE / relative
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    records: list[dict[str, object]] = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.suffix not in CODE_SUFFIXES:
            continue
        local = source.relative_to(source_root)
        if any(part in EXCLUDED_PARTS or part.startswith("raw") for part in local.parts):
            continue
        if relative == "teacherfree_lab_weno7_rk4_distance_balanced_fast" and (
            "teacherfree_lab_weno7_rk4_distance_balanced_fast" in local.parts
        ):
            continue
        allowlist = TREE_ALLOWLISTS.get(relative)
        if allowlist is not None and local.as_posix() not in allowlist:
            continue
        destination = SNAPSHOT / relative / local
        records.append(copy_file(source.relative_to(SOURCE).as_posix(), destination))
    return records


def main() -> None:
    records: list[dict[str, object]] = []

    if SNAPSHOT.exists():
        shutil.rmtree(SNAPSHOT)

    records.append(copy_file("WENO_NN/main.pdf", REPOSITORY / "paper" / "WENO_SR_current_draft.pdf"))

    for output_name, source_relative in MODELS.items():
        records.append(copy_file(source_relative, REPOSITORY / "models" / output_name))

    for relative in CODE_TREES:
        records.extend(copy_code_tree(relative))
    for relative in ROOT_CODE_FILES:
        records.append(copy_file(relative, SNAPSHOT / relative))
    for relative in EXTRA_CODE_FILES:
        records.append(copy_file(relative, SNAPSHOT / relative))

    for alias, model_name in MODEL_ALIASES.items():
        source = REPOSITORY / "models" / model_name
        destination = SNAPSHOT / alias
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "source": f"models/{model_name}",
                "destination": destination.relative_to(REPOSITORY).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "runtime_alias": True,
            }
        )

    for experiment, groups in EXPERIMENT_ASSETS.items():
        experiment_root = REPOSITORY / "experiments" / experiment
        for generated_group in ("figures", "tables"):
            generated_path = experiment_root / generated_group
            if generated_path.exists():
                shutil.rmtree(generated_path)
        experiment_records: list[dict[str, object]] = []
        for group, assets in groups.items():
            for output_name, source_relative in assets.items():
                record = copy_file(source_relative, experiment_root / group / output_name)
                records.append(record)
                experiment_records.append(record)
        provenance = {
            "experiment": experiment,
            "artifact_policy": "Derived figures and compact tables only; raw numerical fields are excluded.",
            "selected_checkpoints": {
                "weno5_sr_fp64": "models/weno5_sr_fp64_step012250.npz",
                "weno5_sr_fp32": "models/weno5_sr_fp32_step016500.npz",
                "weno7_sr_fp64": "models/weno7_sr_fp64_step016750.npz",
            },
            "run_entry_point": f"experiments/{experiment}/run.sh",
            "plot_entry_points": PLOT_ENTRY_POINTS[experiment],
            "manuscript_source": "WENO_NN/section4.tex",
            "manuscript_location": PAPER_LOCATIONS[experiment],
            "artifacts": experiment_records,
        }
        path = experiment_root / "provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    inventory = REPOSITORY / "scripts" / "copy_inventory.json"
    inventory.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"copied {len(records)} immutable files into {REPOSITORY}")


if __name__ == "__main__":
    main()
