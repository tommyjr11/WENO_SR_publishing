#!/usr/bin/env python3
"""Run WENO7 point-value Shu-RK4 quadrant test with optional MLP beta."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weno7_point_rk4_shu import point_rk4 as classical
from weno7_point_rk4_shu import point_rk4_mlp as mlp_solver
from weno7_point_rk4_shu import run_quadrant_point_rk4 as qrun


wp = classical.wp


def diff_metrics(classical_u: np.ndarray, mlp_u: np.ndarray, params: classical.Params) -> dict[str, float]:
    c_pri = qrun.primitive_interior(classical_u, params)
    m_pri = qrun.primitive_interior(mlp_u, params)
    out: dict[str, float] = {}
    for name, idx in (("rho", 0), ("p", 3)):
        diff = m_pri[..., idx] - c_pri[..., idx]
        out[f"{name}_diff_l1"] = float(np.mean(np.abs(diff)))
        out[f"{name}_diff_l2"] = float(np.sqrt(np.mean(diff * diff)))
        out[f"{name}_diff_linf"] = float(np.max(np.abs(diff)))
    return out


def run(args: argparse.Namespace) -> None:
    classical.wh.require_warp()
    wp.init()
    wp.set_device(args.device)

    params = classical.Params(
        nx=args.nx,
        ny=args.ny,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        cfl=args.cfl,
        t_end=args.t_end,
    )
    if args.run_mlp and args.model is None:
        raise ValueError("--run-mlp requires --model")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    initial = qrun.make_quadrant_state(params, args.quadrant_case, args.init_quadrature)

    print(
        f"run_start WENO7 point-RK4 model={args.model} nx={params.nx} ny={params.ny} "
        f"cfl={params.cfl} t_end={params.t_end} case={args.quadrant_case} "
        f"solver={args.riemann_solver} boundary={args.boundary} init_quadrature={args.init_quadrature} "
        f"eno_cutoff={args.eno_cutoff} run_classical={args.run_classical} run_mlp={args.run_mlp} device={args.device}",
        flush=True,
    )

    results: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, object]] = {}

    if args.run_mlp:
        beta_model = mlp_solver.TorchWeno7PointBeta(args.model, args.device, params.gamma)
        final_mlp, summaries["mlp"] = mlp_solver.run_from_initial_mlp(
            initial.copy(),
            params,
            device=args.device,
            riemann_solver=args.riemann_solver,
            beta_model=beta_model,
            report_interval=args.report_interval,
            max_steps=args.max_steps,
            boundary=args.boundary,
            eno_cutoff=args.eno_cutoff,
        )
        results["mlp"] = final_mlp
        np.savez(args.out_dir / "mlp_quadrant_point_rk4_results.npz", initial=initial, mlp=final_mlp, **summaries["mlp"])
        qrun.plot_qstyle(
            final_mlp,
            params,
            args.out_dir / "mlp_point_rk4_pressure_rho_quiver_rho016_171_step005.png",
            f"WENO7 MLP point-RK4 {args.quadrant_case} {args.riemann_solver} {params.nx}x{params.ny} t={summaries['mlp']['t']:.3f}",
        )

    if args.run_classical:
        final_classical, summaries["classical"] = classical.run_from_initial(
            initial.copy(),
            params,
            device=args.device,
            riemann_solver=args.riemann_solver,
            characteristic=True,
            report_interval=args.report_interval,
            max_steps=args.max_steps,
            boundary=args.boundary,
        )
        results["classical"] = final_classical
        np.savez(args.out_dir / "classical_quadrant_point_rk4_results.npz", initial=initial, classical=final_classical, **summaries["classical"])
        qrun.plot_qstyle(
            final_classical,
            params,
            args.out_dir / "classical_point_rk4_pressure_rho_quiver_rho016_171_step005.png",
            f"WENO7 classical point-RK4 {args.quadrant_case} {args.riemann_solver} {params.nx}x{params.ny} t={summaries['classical']['t']:.3f}",
        )

    if "classical" in results and "mlp" in results:
        summaries["mlp_minus_classical"] = diff_metrics(results["classical"], results["mlp"], params)

    qrun.write_summary(
        args.out_dir / "summary.txt",
        {
            "method": "WENO7 point-value Shu-RK4 MLP beta",
            "model": str(args.model) if args.model is not None else "None",
            "quadrant_case": args.quadrant_case,
            "riemann_solver": args.riemann_solver,
            "boundary": args.boundary,
            "nx": params.nx,
            "ny": params.ny,
            "cfl": params.cfl,
            "t_end": params.t_end,
            "init_quadrature": args.init_quadrature,
            "eno_cutoff": args.eno_cutoff,
            "device": args.device,
        },
        summaries,
    )
    print(f"summary={args.out_dir / 'summary.txt'}", flush=True)
    if "mlp" in results:
        print(f"mlp_plot={args.out_dir / 'mlp_point_rk4_pressure_rho_quiver_rho016_171_step005.png'}", flush=True)
    if "classical" in results:
        print(f"classical_plot={args.out_dir / 'classical_point_rk4_pressure_rho_quiver_rho016_171_step005.png'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.5)
    parser.add_argument("--quadrant-case", choices=("case12", "case6"), default="case12")
    parser.add_argument("--init-quadrature", type=int, default=15)
    parser.add_argument("--riemann-solver", choices=("evilin", "hllc"), default="evilin")
    parser.add_argument("--boundary", choices=("outflow", "periodic"), default="outflow")
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-classical", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-interval", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
