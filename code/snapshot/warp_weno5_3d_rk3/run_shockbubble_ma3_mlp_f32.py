from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ShockBubbleConfig
from .solver import run_solver


DEFAULT_MODEL = Path(
    "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/runs/"
    "apost_weno5_v20_mlp_f32_fast_200k/checkpoints/"
    "model_step_016500.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warp 3-D characteristic WENO5-SR--SSPRK3 Ma=3 shock--bubble run; "
            "only the MLP is float32 and the solver remains float64"
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("warp_weno5_3d_rk3/runs/ma3_v20_mlp_f32_step016500"),
    )
    parser.add_argument("--nx", type=int, default=224)
    parser.add_argument("--ny", type=int, default=88)
    parser.add_argument("--nz", type=int, default=88)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--t-end", type=float, default=2.0e-5)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--eno-cutoff", action="store_true")
    parser.add_argument("--no-strict-sync", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        ShockBubbleConfig(),
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        cfl=args.cfl,
        t_end=args.t_end,
    )
    report = run_solver(
        config=config,
        device=args.device,
        out_dir=args.out_dir,
        model=args.model,
        eno_cutoff=args.eno_cutoff,
        stop_step=args.stop_step,
        strict_sync=not args.no_strict_sync,
    )
    print(
        json.dumps(
            {
                "step": report["step"],
                "time": report["time"],
                "elapsed_seconds": report["elapsed_seconds"],
                "output": report["output"],
                "model": report["model"],
                "final_diagnostics": report["final_diagnostics"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
