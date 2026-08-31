from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ShockBubbleConfig
from .solver import run_solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warp float64 3-D characteristic WENO5--SSPRK3 Ma=3 shock--bubble run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=Path("warp_weno5_3d_rk3/runs/ma3_evilin"))
    parser.add_argument("--reference", type=Path, default=Path("ADER_TR_Project/data/step_0140.bin"))
    parser.add_argument("--nx", type=int, default=224)
    parser.add_argument("--ny", type=int, default=88)
    parser.add_argument("--nz", type=int, default=88)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--t-end", type=float, default=2.0e-5)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--no-strict-sync", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = ShockBubbleConfig()
    config = replace(base, nx=args.nx, ny=args.ny, nz=args.nz, cfl=args.cfl, t_end=args.t_end)
    report = run_solver(
        config=config,
        device=args.device,
        out_dir=args.out_dir,
        reference=args.reference,
        stop_step=args.stop_step,
        strict_sync=not args.no_strict_sync,
    )
    summary = {
        "step": report["step"],
        "time": report["time"],
        "output": report["output"],
        "final_diagnostics": report["final_diagnostics"],
    }
    comparison = report.get("comparison")
    if isinstance(comparison, dict):
        summary["file_bitwise_identical"] = comparison["file_bitwise_identical"]
        summary["normalized_l1"] = comparison["normalized_l1"]
        summary["max_absolute_error"] = comparison["max_absolute_error"]
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
