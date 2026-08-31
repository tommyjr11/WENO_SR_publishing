#!/usr/bin/env python3
"""Run trusted mixed-precision Warp Sod validation for selected checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import warp_sod_validation as SV


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.25)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    SV.prepare_warp(args.device)
    records = []
    for step in args.steps:
        checkpoint = (
            args.run_dir
            / "checkpoints"
            / f"model_step_{step:06d}.npz"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        with np.load(checkpoint, allow_pickle=True) as data:
            payload = {name: data[name] for name in data.files}
        parameters = SV.wp_params_from_payload(payload, args.device)
        record = SV.run_warp_sod_validation(
            step,
            step,
            parameters,
            args.out_dir,
            args.device,
            nx=args.nx,
            ny=args.ny,
            cfl=args.cfl,
            t_end=args.t_end,
            axis=args.axis,
            eno_cutoff=False,
            weno_space="characteristic",
            riemann_solver="evilin",
        )
        records.append(record)
        print(
            f"SOD_DONE step={step:06d} "
            f"gain_L2={100.0 * float(record['rel_gain_vs_reference_l2']):.2f}% "
            f"failed={int(float(record['failed']))}",
            flush=True,
        )
    SV.write_warp_sod_outputs(args.out_dir, records)


if __name__ == "__main__":
    main()
