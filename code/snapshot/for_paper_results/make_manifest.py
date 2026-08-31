#!/usr/bin/env python3
from __future__ import annotations

from for_paper_results import config
from for_paper_results.common import environment_manifest, sha256, write_json


def main() -> None:
    config.ensure_output_dirs()
    config.validate_models()
    path = config.PACKAGE / "manifest.json"
    payload = environment_manifest()
    payload["protocols"] = {
        "euler_common": {
            "riemann_solver": "hllc",
            "weno_space": "characteristic",
            "eno_cutoff": False,
            "hllc_state_floors": True,
            "hllc_denominator_regularization": True,
            "hllc_tiny": 1.0e-16,
            "hllc_wave_speed_estimate": "direct_davis_minmax",
            "cfl": 0.4,
        },
        "gste": {
            "cells": 200, "t_end": 10.0, "periodic": True,
            "methods": [
                "weno5_js", "weno5_sr_f64", "weno5_sr_f32",
                "weno7_js", "weno7_sr_f64",
            ],
            "cfl": 0.6, "initial_quadrature": 15,
            "selected_results": "for_paper_results/raw/gste",
        },
        "sod": {"cells": [51], "t_end": 0.2, "initial_quadrature": 15,
                "geometry": "one-dimensional", "domain": [0.0, 1.0],
                "discontinuity": 0.5},
        "vortex": {"meshes": [25, 50, 100, 200], "t_end": 2.0, "periodic": True},
        "riemann_2d": {
            "configurations": {
                "C.3 (q400)": 0.50,
                "C.4": 0.25,
                "C.5": 0.23,
                "C.6": 0.30,
            },
            "nx": 400, "ny": 400, "initial_quadrature": 15,
            "display": "shared pressure scale, density contours, velocity vectors",
        },
        "weno5_timing": {
            "cases": ["C.4"], "nx": 400, "ny": 400,
            "methods": ["weno5_js", "weno5_sr_f64", "weno5_sr_f32"],
            "warmups": 1, "repeats": 3,
        },
        "double_mach": {
            "nx": 1200, "ny": 300, "t_end": 0.2, "cfl": 0.4,
            "initial_quadrature": 15,
            "learned_methods": ["weno5_sr_f64", "weno5_sr_f32", "weno7_sr_f64"],
            "displayed_methods": [
                "weno5_js", "weno5_sr_f64", "weno5_sr_f32",
                "weno7_js", "weno7_sr_f64",
            ],
            "selected_results": "for_paper_results/raw/double_mach",
            "reused_existing_validated_fp64_result": False,
            "classical_baselines": [
                str(config.DOUBLE_MACH_WENO5_JS_STATE.relative_to(config.ROOT)),
            ],
        },
    }
    source_paths = [
        config.PACKAGE / "solvers/weno5_hllc.py",
        config.PACKAGE / "solvers/weno5_hllc_mixed.py",
        config.PACKAGE / "solvers/euler_methods.py",
        config.PACKAGE / "run_gste.py",
        config.PACKAGE / "run_sod.py",
        config.PACKAGE / "run_vortex.py",
        config.PACKAGE / "run_quadrant.py",
        config.PACKAGE / "run_double_mach.py",
        config.PACKAGE / "run_weno5_timing.py",
        config.PACKAGE / "make_figures.py",
        config.PACKAGE / "make_riemann_figures.py",
        config.PACKAGE / "make_sod_point_figures.py",
        config.PACKAGE / "make_double_mach_figures.py",
        config.ROOT / "teacherfree_lab_weno5_v20_distance_balanced/train_weno5_v20.py",
        config.ROOT / "teacherfree_lab_weno5_v20_distance_balanced/v20_losses.py",
        config.ROOT / "teacherfree_lab_weno5_v20_distance_balanced/weno5_rk3_diff_v20_deploy.py",
        config.ROOT / "teacherfree_lab_weno5_v20_distance_balanced/weno5_hllc_refsym.py",
        config.ROOT / "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/train.py",
        config.ROOT / "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/fast_losses.py",
        config.ROOT / "teacherfree_lab_weno7_rk4_distance_balanced_fast/train.py",
        config.ROOT / "teacherfree_lab_weno7_rk4_distance_balanced_fast/fast_losses.py",
        config.ROOT / "teacherfree_lab_weno7_rk4_distance_balanced_fast/weno7_core.py",
        config.ROOT / "teacherfree_lab_weno7_rk4_distance_balanced_fast/rk4_advection.py",
        config.ROOT / "warp_weno5_helpers.py",
        config.ROOT / "weno5_rk3_diff.py",
        config.ROOT / "weno5_rk3_forward.py",
        config.ROOT / "weno7_external_clean/warp_weno7_ader4_helpers_classical_only.py",
    ]
    payload["source_sha256"] = {
        str(source.relative_to(config.ROOT)): sha256(source) for source in source_paths
    }
    write_json(path, payload)
    print(path)


if __name__ == "__main__":
    main()
