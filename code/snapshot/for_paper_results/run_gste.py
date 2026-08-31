#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from teacherfree_lab_weno5 import apost_advect as adv5
from teacherfree_lab_weno5 import weno5_core as core5
from teacherfree_lab_weno5_mlp_f32 import apost_advect as adv5mixed
from teacherfree_lab_weno5_mlp_f32 import weno5_core as core5mixed
from teacherfree_lab_weno5_v20_distance_balanced_mlp_f32.v20_mlp_f32_model import (
    load_checkpoint as load_reflection_symmetric_weno5_f32,
)
from teacherfree_lab_weno5_v12_reflection_sym.v12_losses import (
    ssprk3_step_signed,
)
from teacherfree_lab_weno5_v12_reflection_sym.v12_model import (
    load_checkpoint as load_reflection_symmetric_weno5,
)
from teacherfree_lab_weno7_rk4_distance_balanced_fast import (
    rk4_advection as adv7,
)
from teacherfree_lab_weno7_rk4_distance_balanced_fast import (
    weno7_core as core7,
)

from for_paper_results import config
from for_paper_results.common import write_json


X_MIN = -1.0
X_MAX = 1.0
LENGTH = X_MAX - X_MIN
PAPER_METHODS = (
    "weno5_js",
    "weno5_sr_f64",
    "weno5_sr_f32",
    "weno7_js",
    "weno7_sr_f64",
    "weno9_tt",
    "weno11_tt",
)
FORMAL_METHODS = ("weno5_js", "weno5_sr_f64")
ALL_METHODS = PAPER_METHODS + ("weno5_offline137k",)
ENO_CUTOFF_THRESHOLD = 4.0e-7
WENO5_OFFLINE137K = (
    config.ROOT
    / "plots/WENO5_MLP/weno5_offline_power2_5_10_6_6_3_400k_disc15"
    / "checkpoints/model_step_137000.npz"
)


def wrap(x: np.ndarray) -> np.ndarray:
    return X_MIN + np.mod(x - X_MIN, LENGTH)


def gste_point(x: np.ndarray) -> np.ndarray:
    x = wrap(np.asarray(x, dtype=np.float64))
    delta = 0.005
    beta = math.log(2.0) / (36.0 * delta * delta)
    z = -0.7
    a = 0.5
    alpha = 10.0

    def gaussian(center: float) -> np.ndarray:
        return np.exp(-beta * (x - center) ** 2)

    def ellipse(center: float) -> np.ndarray:
        return np.sqrt(np.maximum(1.0 - alpha * alpha * (x - center) ** 2, 0.0))

    out = np.zeros_like(x)
    mask = (-0.8 < x) & (x < -0.6)
    out[mask] = (
        gaussian(z - delta)[mask] + 4.0 * gaussian(z)[mask] + gaussian(z + delta)[mask]
    ) / 6.0
    mask = (-0.4 < x) & (x < -0.2)
    out[mask] = 1.0
    mask = (0.0 < x) & (x < 0.2)
    out[mask] = 1.0 - np.abs(10.0 * (x[mask] - 0.1))
    mask = (0.4 < x) & (x < 0.6)
    out[mask] = (
        ellipse(a - delta)[mask] + 4.0 * ellipse(a)[mask] + ellipse(a + delta)[mask]
    ) / 6.0
    return out


def cell_averages(nx: int, time: float, quadrature: int = 6) -> tuple[np.ndarray, np.ndarray]:
    dx = LENGTH / nx
    centers = X_MIN + (np.arange(nx) + 0.5) * dx
    xi, weights = np.polynomial.legendre.leggauss(quadrature)
    samples = centers[:, None] + 0.5 * dx * xi[None, :] - time
    averages = 0.5 * np.sum(weights[None, :] * gste_point(samples), axis=1)
    return centers, averages


def integrate_ssprk3(model, stepper, u0: np.ndarray, cfl_limit: float,
                     t_end: float, device: torch.device) -> tuple[np.ndarray, dict]:
    nx = u0.size
    dx = LENGTH / nx
    steps = int(math.ceil(t_end / (cfl_limit * dx)))
    dt = t_end / steps
    actual_cfl = dt / dx
    u = torch.as_tensor(u0, device=device, dtype=torch.float64).reshape(1, nx)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for _ in range(steps):
            u = stepper(model, u, dt, 1.0 / dx)
    return u[0].detach().cpu().numpy(), {
        "steps": steps, "dt": dt, "cfl": actual_cfl, "t": t_end,
    }


def reflection_symmetric_ssprk3(
    model, u: torch.Tensor, dt: float, dxinv: float
) -> torch.Tensor:
    velocities = torch.ones(
        u.shape[0], device=u.device, dtype=torch.float64
    )
    return ssprk3_step_signed(model, u, dt, dxinv, velocities)


def reflection_symmetric_shu_rk4(
    model, u: torch.Tensor, dt: float, dxinv: float
) -> torch.Tensor:
    velocities = torch.ones(
        u.shape[0], device=u.device, dtype=torch.float64
    )
    return adv7.shu_rk4_step_signed(
        model, u, dt, dxinv, velocities, eno_cutoff=False
    )


def make_weno5_cutoff_stepper(core, standard_stepper, enabled: bool):
    if not enabled:
        return standard_stepper

    def reconstruction_omega(model, q: torch.Tensor) -> torch.Tensor:
        d = core.optimal_d(1, q.device).reshape(1, 3).expand(q.shape[0], 3)
        omega = core.omega_from_ratio(model(core.weno5_features(q)), 1)
        kept = torch.where(
            omega > ENO_CUTOFF_THRESHOLD,
            omega,
            torch.zeros_like(omega),
        )
        denom = torch.sum(kept, dim=1, keepdim=True)
        cutoff = torch.where(denom > 0.0, kept / denom, omega)
        return torch.where(core.plateau_mask(q).reshape(-1, 1), d, cutoff)

    def reconstruct_iplus(model, u: torch.Tensor) -> torch.Tensor:
        bsz, n = u.shape
        periodic = torch.cat([u[:, -2:], u, u[:, :2]], dim=1)
        q = periodic.unfold(1, 5, 1).reshape(bsz * n, 5)
        value = torch.sum(
            reconstruction_omega(model, q) * core.candidate_values(q, 1),
            dim=1,
        )
        return value.reshape(bsz, n)

    def rhs(model, u: torch.Tensor, dxinv: float) -> torch.Tensor:
        flux = reconstruct_iplus(model, u)
        return -(flux - torch.roll(flux, 1, dims=1)) * dxinv

    def stepper(model, u: torch.Tensor, dt: float, dxinv: float) -> torch.Tensor:
        u1 = u + dt * rhs(model, u, dxinv)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(model, u1, dxinv))
        return u / 3.0 + (2.0 / 3.0) * (u2 + dt * rhs(model, u2, dxinv))

    return stepper


def run_torch_methods(
    nx: int,
    t_end: float,
    device: torch.device,
    out_dir: Path,
    quadrature: int,
    methods: set[str],
    weno5_cfl: float,
    weno7_cfl: float,
    weno5_sr_f64_model: Path | None = None,
    weno5_offline137k_model: Path | None = None,
    eno_cutoff: bool = False,
) -> list[dict]:
    x, u0 = cell_averages(nx, 0.0, quadrature)
    _, exact = cell_averages(nx, t_end, quadrature)
    definitions = []
    stepper_f64 = make_weno5_cutoff_stepper(core5, adv5.ssprk3, eno_cutoff)
    stepper_f32 = make_weno5_cutoff_stepper(core5mixed, adv5mixed.ssprk3, eno_cutoff)
    if "weno5_js" in methods:
        definitions.append(("weno5_js", "classical", adv5.ssprk3, weno5_cfl, None, False))
    if "weno5_offline137k" in methods:
        model_path = weno5_offline137k_model or WENO5_OFFLINE137K
        definitions.append((
            "weno5_offline137k",
            core5.load_checkpoint(model_path, device),
            stepper_f64,
            weno5_cfl,
            model_path,
            eno_cutoff,
        ))
    if "weno5_sr_f64" in methods:
        model_path = weno5_sr_f64_model or config.METHODS["weno5_sr_f64"].model
        definitions.append((
            "weno5_sr_f64",
            load_reflection_symmetric_weno5(model_path, device),
            reflection_symmetric_ssprk3,
            weno5_cfl,
            model_path,
            False,
        ))
    if "weno5_sr_f32" in methods:
        definitions.append((
            "weno5_sr_f32",
            load_reflection_symmetric_weno5_f32(
                config.METHODS["weno5_sr_f32"].model, device
            ),
            stepper_f32,
            weno5_cfl,
            config.METHODS["weno5_sr_f32"].model,
            False,
        ))
    if "weno7_js" in methods:
        definitions.append((
            "weno7_js", "classical", reflection_symmetric_shu_rk4,
            weno7_cfl, None, False,
        ))
    if "weno7_sr_f64" in methods:
        definitions.append((
            "weno7_sr_f64",
            core7.load_checkpoint(config.METHODS["weno7_sr_f64"].model, device),
            reflection_symmetric_shu_rk4,
            weno7_cfl,
            config.METHODS["weno7_sr_f64"].model,
            False,
        ))
    rows: list[dict] = []
    for key, model, stepper, cfl, model_path, cutoff_applied in definitions:
        final, meta = integrate_ssprk3(model, stepper, u0, cfl, t_end, device)
        diff = final - exact
        row = {
            "method": key, **meta,
            "l1": float(np.mean(np.abs(diff))),
            "l2": float(np.sqrt(np.mean(diff * diff))),
            "linf": float(np.max(np.abs(diff))),
            "tv": float(np.sum(np.abs(final - np.roll(final, 1)))),
            "min": float(np.min(final)), "max": float(np.max(final)),
            "complete": bool(np.all(np.isfinite(final))),
            "eno_cutoff": cutoff_applied,
            "eno_cutoff_threshold": ENO_CUTOFF_THRESHOLD if cutoff_applied else None,
            "model": str(model_path) if model_path is not None else None,
        }
        np.savez(out_dir / f"{key}.npz", x=x, initial=u0, final=final, exact=exact,
                 metadata_json=np.array(json.dumps(row, sort_keys=True)))
        rows.append(row)
        print(row, flush=True)
    return rows


def run_tt(method: str, order: int, q: int, nx: int, t_end: float, out_dir: Path) -> dict:
    binary = config.ROOT / "advection/advection_tt_runner/advection_trq"
    profile = out_dir / f"{method}_profile.csv"
    command = [
        str(binary), "--order", str(order), "--q", str(q), "--nx", str(nx),
        "--cfl", "0.8", "--final-time", str(t_end), "--ic", "jscomposite",
        # delta_IS<2.4 is a smooth-wave CFL detector.  The discontinuous GSTE
        # profile starts far above that threshold, so it is not a valid gate here.
        "--delta-threshold", "1e300",
        "--check-every", "1000000", "--dump-profile", str(profile),
    ]
    completed = subprocess.run(command, cwd=config.ROOT, check=False, text=True,
                               capture_output=True)
    (out_dir / f"{method}.log").write_text(completed.stdout + completed.stderr)
    lines = [line for line in completed.stdout.splitlines() if line and not line.startswith("order,")]
    if completed.returncode != 0 or not lines:
        return {"method": method, "complete": False, "returncode": completed.returncode}
    keys = completed.stdout.splitlines()[-2].split(",") if completed.stdout.splitlines()[-1].startswith("order,") else None
    # The runner emits a two-line CSV summary; locate it explicitly.
    output_lines = completed.stdout.splitlines()
    header_index = next(i for i, line in enumerate(output_lines) if line.startswith("order,q,ic,"))
    values = next(csv.reader([output_lines[header_index + 1]]))
    names = next(csv.reader([output_lines[header_index]]))
    data = dict(zip(names, values))
    arr = np.genfromtxt(profile, delimiter=",", names=True)
    final = np.asarray(arr["q_final"], dtype=np.float64)
    exact = np.asarray(arr["exact_final"], dtype=np.float64)
    diff = final - exact
    row = {
        "method": method,
        "steps": int(data["steps"]), "dt": float(data["dt"]),
        "cfl": float(data["cfl"]), "t": float(data["final_time"]),
        "l1": float(np.mean(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(diff * diff))),
        "linf": float(np.max(np.abs(diff))),
        "tv": float(np.sum(np.abs(final - np.roll(final, 1)))),
        "min": float(np.min(final)), "max": float(np.max(final)),
        "complete": bool(np.all(np.isfinite(final)) and data["stable"] == "true"),
        "runner_stable": data["stable"] == "true", "reason": data["reason"],
    }
    np.savez(out_dir / f"{method}.npz", x=arr["x"], initial=arr["q_initial"],
             final=final, exact=exact, metadata_json=np.array(json.dumps(row, sort_keys=True)))
    print(row, flush=True)
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "method", "complete", "steps", "cfl", "dt", "t", "l1", "l2",
        "linf", "tv", "min", "max", "eno_cutoff", "eno_cutoff_threshold",
        "model",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--t-end", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--weno5-cfl", type=float, default=0.8)
    parser.add_argument("--weno7-cfl", type=float, default=0.6)
    parser.add_argument(
        "--weno5-sr-f64-model",
        type=Path,
        default=None,
        help="override the configured WENO5-SR FP64 checkpoint",
    )
    parser.add_argument(
        "--weno5-offline137k-model",
        type=Path,
        default=WENO5_OFFLINE137K,
        help="checkpoint used by the weno5_offline137k method",
    )
    parser.add_argument(
        "--eno-cutoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply the trusted 4e-7 cutoff to WENO5 MLP weights",
    )
    parser.add_argument(
        "--methods",
        default=",".join(FORMAL_METHODS),
        help="comma-separated subset of paper GSTE methods",
    )
    parser.add_argument(
        "--out-tag",
        default=None,
        help="optional subdirectory below raw/gste (keeps the paper result intact)",
    )
    args = parser.parse_args()
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if not methods or unknown:
        raise ValueError(f"invalid GSTE methods: {unknown or methods}")
    if args.weno5_cfl <= 0.0 or args.weno7_cfl <= 0.0:
        raise ValueError("GSTE CFL values must be positive")
    if args.weno5_sr_f64_model is not None and not args.weno5_sr_f64_model.is_file():
        raise FileNotFoundError(args.weno5_sr_f64_model)
    if "weno5_offline137k" in methods and not args.weno5_offline137k_model.is_file():
        raise FileNotFoundError(args.weno5_offline137k_model)
    if {"weno9_tt", "weno11_tt"} & set(methods) and args.init_quadrature != 6:
        raise ValueError(
            "the TT GSTE runners require the six-point Gauss-Legendre rule "
            "used by advection_trq"
        )
    config.ensure_output_dirs()
    config.validate_models()
    out_dir = config.RAW / "gste"
    if args.out_tag:
        out_dir /= args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_torch_methods(
        args.nx,
        args.t_end,
        torch.device(args.device),
        out_dir,
        args.init_quadrature,
        set(methods),
        args.weno5_cfl,
        args.weno7_cfl,
        args.weno5_sr_f64_model,
        args.weno5_offline137k_model,
        args.eno_cutoff,
    )
    if "weno9_tt" in methods:
        rows.append(run_tt("weno9_tt", 5, 4, args.nx, args.t_end, out_dir))
    if "weno11_tt" in methods:
        rows.append(run_tt("weno11_tt", 6, 5, args.nx, args.t_end, out_dir))
    write_csv(rows, out_dir / "metrics.csv")
    write_json(
        out_dir / "summary.json",
        {
            "nx": args.nx,
            "t_end": args.t_end,
            "init_quadrature": args.init_quadrature,
            "methods": methods,
            "weno5_cfl": args.weno5_cfl,
            "weno7_cfl": args.weno7_cfl,
            "eno_cutoff": args.eno_cutoff,
            "eno_cutoff_threshold": ENO_CUTOFF_THRESHOLD,
            "weno5_sr_f64_model": (
                str(args.weno5_sr_f64_model.resolve())
                if args.weno5_sr_f64_model is not None else None
            ),
            "out_tag": args.out_tag,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
