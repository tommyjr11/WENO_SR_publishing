from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ShockBubbleConfig
from .plot_midplane import plot_file
from .solver import run_solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warp float64 3-D characteristic WENO7-JS--RK4 Ma=3 shock--bubble run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=Path("warp_weno7_3d_rk4/runs/ma3_js_rk4_t00002"))
    parser.add_argument("--nx", type=int, default=224)
    parser.add_argument("--ny", type=int, default=88)
    parser.add_argument("--nz", type=int, default=88)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--t-end", type=float, default=2.0e-5)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--strict-sync", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--eno-cutoff", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(ShockBubbleConfig(), nx=args.nx, ny=args.ny, nz=args.nz, cfl=args.cfl, t_end=args.t_end)
    report = run_solver(
        config,
        device=args.device,
        out_dir=args.out_dir,
        stop_step=args.stop_step,
        strict_sync=args.strict_sync,
        model=args.model,
        eno_cutoff=args.eno_cutoff,
    )
    if not args.no_plot:
        scheme_label = "WENO7-JS--RK4" if args.model is None else "WENO7-SR--RK4 (FP64)"
        report["figures"] = [
            str(path)
            for path in plot_file(
                report["output"], args.out_dir / "figures", scheme_label=scheme_label
            )
        ]
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
