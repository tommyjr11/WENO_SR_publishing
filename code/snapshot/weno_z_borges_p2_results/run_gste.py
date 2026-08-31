#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from for_paper_results.run_gste import LENGTH, cell_averages
from . import torch_weno_z as Z


ROOT = Path(__file__).resolve().parent


def integrate(initial: np.ndarray, t_end: float, cfl: float, stepper, device: str):
    nx = initial.size
    dx = LENGTH / float(nx)
    steps = int(math.ceil(t_end / (cfl * dx)))
    dt = t_end / float(steps)
    state = torch.as_tensor(initial, device=device, dtype=torch.float64)[None, :]
    with torch.no_grad():
        for _ in range(steps):
            state = stepper(state, dt, 1.0 / dx)
    return state[0].cpu().numpy(), steps, dt, dt / dx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--cfl", type=float, default=0.6)
    parser.add_argument("--quadrature", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    Z.self_test()
    out = ROOT / "raw/gste/N200_t10_cfl06"
    out.mkdir(parents=True, exist_ok=True)
    x, initial = cell_averages(args.nx, 0.0, args.quadrature)
    _, exact = cell_averages(args.nx, args.t_end, args.quadrature)
    dx = LENGTH / float(args.nx)
    methods = (
        ("weno5_z_p2", "WENO5-Z-RK3", 2, Z.ssprk3_step),
        ("weno7_z_p3", "WENO7-Z-RK4", 3, Z.shu_rk4_step),
    )
    rows = []
    for key, label, power, stepper in methods:
        final, steps, dt, actual_cfl = integrate(
            initial, args.t_end, args.cfl, stepper, args.device
        )
        error = final - exact
        row = {
            "method": key,
            "label": label,
            "nx": args.nx,
            "t_end": args.t_end,
            "steps": steps,
            "dt": dt,
            "cfl": actual_cfl,
            "l1": float(np.mean(np.abs(error))),
            "l2": float(np.sqrt(np.mean(error * error))),
            "linf": float(np.max(np.abs(error))),
            "tv": float(np.sum(np.abs(final - np.roll(final, 1)))),
            "min": float(final.min()),
            "max": float(final.max()),
            "complete": bool(np.all(np.isfinite(final))),
            "weno_z_p": power,
            "weno_z_epsilon": dx**power,
            "weno_z_tau": (
                "abs(beta0-beta2)" if power == 2
                else "abs(-beta0-3*beta1+3*beta2+beta3)"
            ),
        }
        np.savez(
            out / f"{key}.npz", x=x, initial=initial, exact=exact, final=final,
            metadata_json=np.array(json.dumps(row, sort_keys=True)),
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
