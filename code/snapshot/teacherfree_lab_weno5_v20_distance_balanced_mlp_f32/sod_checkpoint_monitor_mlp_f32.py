#!/usr/bin/env python3
"""Run mixed-precision symmetric Warp Sod diagnostics for V20 checkpoints."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from teacherfree_lab_weno5_v20_distance_balanced_mlp_f32 import (
    warp_sod_validation_mlp_f32 as SV,
)


STEP_PATTERN = re.compile(r"model_step_(\d+)\.npz$")


def load_records(run_dir: Path) -> dict[int, dict[str, object]]:
    path = run_dir / "circle_validation_metrics.npz"
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=True) as data:
        records = data["records"].tolist()
    return {int(record["raw_step"]): record for record in records}


def ready_checkpoints(run_dir: Path, interval: int) -> list[tuple[int, Path]]:
    ready = []
    for path in (run_dir / "checkpoints").glob("model_step_*.npz"):
        match = STEP_PATTERN.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step > 0 and step % interval == 0:
            ready.append((step, path))
    return sorted(ready)


def run(args: argparse.Namespace) -> None:
    if args.interval <= 0 or args.poll_seconds <= 0:
        raise ValueError("interval and poll-seconds must be positive")
    if not args.run_dir.is_dir():
        raise FileNotFoundError(args.run_dir)

    records = load_records(args.run_dir)
    SV.prepare_warp(args.device)
    print(
        f"sod_monitor_start run={args.run_dir} interval={args.interval} "
        f"grid={args.nx}x{args.ny} cfl={args.cfl} t_end={args.t_end} "
        f"space=characteristic solver=evilin cutoff=False existing={len(records)} "
        "precision=mlp_float32_state_float64",
        flush=True,
    )

    while True:
        pending = [
            item for item in ready_checkpoints(args.run_dir, args.interval)
            if item[0] not in records
        ]
        for step, checkpoint in pending:
            print(f"SOD_START step={step:06d} checkpoint={checkpoint}", flush=True)
            try:
                with np.load(checkpoint, allow_pickle=True) as data:
                    payload = {name: data[name] for name in data.files}
                mlp_params = SV.wp_params_from_payload(payload, args.device)
                record = SV.run_warp_sod_validation(
                    step,
                    step,
                    mlp_params,
                    args.run_dir,
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
            except Exception as error:
                print(
                    f"SOD_RETRY step={step:06d} error={type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                break
            records[step] = record
            ordered = [records[key] for key in sorted(records)]
            SV.write_warp_sod_outputs(args.run_dir, ordered)
            print(
                f"SOD_DONE step={step:06d} "
                f"gain_L2={100.0 * float(record['rel_gain_vs_reference_l2']):.2f}% "
                f"failed={int(float(record['failed']))}",
                flush=True,
            )

        if args.max_step in records:
            print(f"sod_monitor_complete step={args.max_step:06d}", flush=True)
            return
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=250)
    parser.add_argument("--max-step", type=int, default=200000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.25)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
