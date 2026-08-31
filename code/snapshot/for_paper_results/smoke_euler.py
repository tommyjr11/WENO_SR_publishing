#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from teacherfree_lab_weno5 import warp_sod_validation as sod5
from weno7_mlp_retrain_claude import sod_eval as sod7

from for_paper_results import config
from for_paper_results.common import state_health
from for_paper_results.solvers import euler_methods


def main() -> None:
    device = "cuda"
    nx, ny, cfl, t_end = 40, 4, 0.4, 0.01
    results: dict[str, dict] = {}
    for key in config.EULER_METHODS:
        if key.startswith("weno5"):
            mixed = key == "weno5_sr_f32"
            params = euler_methods.make_weno5_params(nx, ny, 1.0, ny / nx, cfl, t_end, mixed)
            initial = sod5.make_exact_sod_state(params, 0.0, "x")
            final, summary = euler_methods.run_weno5(
                key, initial, params, device=device, boundary="transmissive",
            )
        else:
            params = sod7.make_sod_params(nx, ny, t_end, cfl)
            initial = sod7.make_sod_u0(params)
            final, summary = euler_methods.run_weno7(
                key, initial, params, device=device, boundary="outflow",
            )
        health = state_health(final, params.ghost, nx, ny)
        complete_time = abs(float(summary["t"]) - t_end) < 1.0e-12
        results[key] = {**summary, **health, "complete_time": complete_time}
        if not complete_time or not health["complete"]:
            raise RuntimeError(f"smoke failed for {key}: {results[key]}")
        print(f"PASS {key}: {results[key]}", flush=True)
    path = config.PACKAGE / "verification_smoke_euler.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()

