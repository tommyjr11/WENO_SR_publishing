#!/usr/bin/env python3
"""Compare trusted double and mixed classical WENO5 paths bitwise on case6."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warp_weno5_helpers as trusted_wh
from run_weno5_quadrant_mlp_only import make_quadrant_state_fast as trusted_make_quadrant_state
from weno5_rk3_forward import run_forward_to_time as trusted_run_forward_to_time
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32 import warp_weno5_helpers_mlp_f32 as mixed_wh
from teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.weno5_rk3_forward_mlp_f32 import (
    run_forward_to_time as mixed_run_forward_to_time,
)

wp = trusted_wh.wp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=400)
    p.add_argument("--ny", type=int, default=400)
    p.add_argument("--cfl", type=float, default=0.4)
    p.add_argument("--t-end", type=float, default=0.25)
    p.add_argument("--case", choices=("case6", "case12"), default="case6")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--out-dir", type=Path, default=Path("plots/WENO5_MLP/teacherfree_weno5_mlp_f32/bitwise_classical_case6"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trusted_wh.require_warp()
    mixed_wh.require_warp()
    wp.init()
    wp.set_device(args.device)

    trusted_params = trusted_wh.Params(nx=args.nx, ny=args.ny, x_length=1.0, y_length=1.0, cfl=args.cfl, t_end=args.t_end)
    mixed_params = mixed_wh.Params(nx=args.nx, ny=args.ny, x_length=1.0, y_length=1.0, cfl=args.cfl, t_end=args.t_end)
    u0 = trusted_make_quadrant_state(trusted_params, args.case)

    trusted, trusted_dt, trusted_steps, trusted_t = trusted_run_forward_to_time(
        u0,
        trusted_params,
        args.t_end,
        args.device,
        True,
        None,
        False,
        "transmissive",
        "evilin",
        0,
        None,
    )
    mixed, mixed_dt, mixed_steps, mixed_t = mixed_run_forward_to_time(
        u0.copy(),
        mixed_params,
        args.t_end,
        args.device,
        True,
        None,
        False,
        "transmissive",
        "evilin",
        0,
        None,
    )

    equal = bool(np.array_equal(trusted, mixed))
    diff = trusted - mixed
    max_abs = float(np.max(np.abs(diff)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "classical_bitwise_compare.npz",
        trusted=trusted,
        mixed=mixed,
        diff=diff,
        trusted_dt=np.array(trusted_dt),
        mixed_dt=np.array(mixed_dt),
        bitwise_equal=np.array(equal),
        max_abs_diff=np.array(max_abs),
        trusted_steps=np.array(trusted_steps),
        mixed_steps=np.array(mixed_steps),
        trusted_t=np.array(trusted_t),
        mixed_t=np.array(mixed_t),
    )
    with (args.out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"case: {args.case}\n")
        f.write(f"nx: {args.nx}\n")
        f.write(f"ny: {args.ny}\n")
        f.write(f"t_end: {args.t_end}\n")
        f.write("weno_space: characteristic\n")
        f.write("riemann_solver: evilin\n")
        f.write(f"trusted_steps: {trusted_steps}\n")
        f.write(f"mixed_steps: {mixed_steps}\n")
        f.write(f"trusted_t: {trusted_t:.17e}\n")
        f.write(f"mixed_t: {mixed_t:.17e}\n")
        f.write(f"bitwise_equal: {equal}\n")
        f.write(f"max_abs_diff: {max_abs:.17e}\n")
    print(f"bitwise_equal={equal}")
    print(f"max_abs_diff={max_abs:.17e}")
    print(f"out={args.out_dir}")
    if not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
