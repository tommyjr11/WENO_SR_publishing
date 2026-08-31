#!/usr/bin/env python3
"""Teacher-free a-posteriori training for the deployed WENO5 5->10->6->6->3 MLP."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5.apost_advect import IC_DEFAULT_PROBS, IC_KINDS, make_ic, rollout_loss
from teacherfree_lab_weno5 import warp_sod_validation as SV

torch.set_default_dtype(torch.float64)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def kl_to_d(r_s: torch.Tensor) -> torch.Tensor:
    kl = torch.zeros(r_s.shape[0], device=r_s.device, dtype=r_s.dtype)
    for lr in W.LR_VALUES:
        om = torch.clamp(W.omega_from_ratio(r_s, lr), min=1.0e-12)
        d = W.optimal_d(lr, r_s.device).reshape(1, 3)
        kl = kl + torch.sum(d * (torch.log(d) - torch.log(om)), dim=1)
    return kl / float(len(W.LR_VALUES))


def bound_violation(omega_fn, q: torch.Tensor, tol: float = 1.0e-3) -> torch.Tensor:
    dq = q[:, 1:] - q[:, :-1]
    mono = ((dq >= 0.0).all(dim=1) | (dq <= 0.0).all(dim=1)).to(q.dtype)
    lo = torch.min(q, dim=1).values
    hi = torch.max(q, dim=1).values
    rng = torch.clamp(hi - lo, min=1.0e-13)
    total = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    for lr in W.LR_VALUES:
        val = torch.sum(omega_fn(lr) * W.candidate_values(q, lr), dim=1)
        total = total + torch.relu((val - hi) / rng - tol) + torch.relu((lo - val) / rng - tol)
    return mono * total / float(len(W.LR_VALUES))


def _rand(gen: torch.Generator, device: torch.device, lo: float, hi: float, shape: tuple[int, int]) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen, dtype=torch.float64)


def make_smooth_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    x = torch.arange(5.0, device=device, dtype=torch.float64).reshape(1, 5) - 2.0
    base = _rand(gen, device, -2.0, 4.0, (n, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    slope = _rand(gen, device, -0.4, 0.4, (n, 1)) * scale
    quad = _rand(gen, device, -0.04, 0.04, (n, 1)) * scale
    cubic = _rand(gen, device, -0.01, 0.01, (n, 1)) * scale
    sine_amp = torch.pow(10.0, _rand(gen, device, -5.0, -2.0, (n, 1))) * scale
    phase = _rand(gen, device, 0.0, 2.0 * np.pi, (n, 1))
    return base + slope * x + quad * x * x + cubic * x * x * x + sine_amp * torch.sin(np.pi * x / 4.0 + phase)


def _sigmoid_cell_avg(lo: torch.Tensor, hi: torch.Tensor, c: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    sp = torch.nn.functional.softplus
    return w * (sp((hi - c) / w) - sp((lo - c) / w)) / (hi - lo)


def make_front_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    x = torch.arange(5.0, device=device, dtype=torch.float64).reshape(1, 5)
    lo = x - 0.5
    hi = x + 0.5
    base = _rand(gen, device, -2.0, 4.0, (n, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    amp = torch.pow(10.0, _rand(gen, device, -3.0, 0.0, (n, 1))) * scale
    sign = torch.where(torch.rand((n, 1), device=device, generator=gen) < 0.5, -1.0, 1.0)
    c = _rand(gen, device, 0.4, 3.6, (n, 1))
    w = torch.pow(10.0, _rand(gen, device, -1.3, 0.18, (n, 1)))
    return base + sign * amp * _sigmoid_cell_avg(lo, hi, c, w)


def make_monotone_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    base = _rand(gen, device, -2.0, 4.0, (n, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    amp = torch.pow(10.0, _rand(gen, device, -4.0, 0.0, (n, 1))) * scale
    inc = torch.rand((n, 4), device=device, generator=gen, dtype=torch.float64)
    inc = inc / torch.clamp(torch.sum(inc, dim=1, keepdim=True), min=1.0e-13)
    q = torch.cat([torch.zeros((n, 1), device=device, dtype=torch.float64), torch.cumsum(inc, dim=1)], dim=1)
    q = base + amp * q
    flip_sign = torch.rand((n, 1), device=device, generator=gen) < 0.5
    return torch.where(flip_sign, base - (q - base), q)


def make_bound_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    half = n // 2
    return torch.cat([make_front_stencils(half, device, gen), make_monotone_stencils(n - half, device, gen)], dim=0)


def make_ampgate_stencils(
    n: int,
    device: torch.device,
    gen: torch.Generator,
    amp_min: float,
    amp_max: float,
) -> torch.Tensor:
    q = make_bound_stencils(n, device, gen)
    mean = q.mean(dim=1, keepdim=True)
    rng = torch.clamp(torch.max(q, dim=1).values - torch.min(q, dim=1).values, min=1.0e-13)
    q_scale = torch.clamp(torch.max(torch.abs(q), dim=1).values, min=1.0)
    lo, hi = np.log10(amp_min), np.log10(amp_max)
    amp = torch.pow(10.0, lo + (hi - lo) * torch.rand(q.shape[0], device=device, generator=gen))
    scale = (amp * q_scale / rng).reshape(-1, 1)
    return mean + (q - mean) * scale


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    W.check_weno5_coefficients()
    device = W.torch_device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = W.SharedBadnessMLP(seed=args.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = None
    if 0.0 < args.lr_final < args.lr:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr_final)

    horizons = tuple(int(h) for h in args.horizons)
    for h in horizons:
        assert abs(h * args.cfl - round(h * args.cfl)) < 1.0e-12, (
            f"horizon {h} * cfl {args.cfl} must be an integer shift"
        )

    ic_probs = tuple(float(x) for x in args.ic_probs) if args.ic_probs is not None else IC_DEFAULT_PROBS
    probe_gen = torch.Generator(device=device)
    probe_gen.manual_seed(999)
    probe_h = horizons[len(horizons) // 2]
    probes = {}
    for name in IC_KINDS:
        ic = make_ic(64, args.grid, device, probe_gen, kind=name)
        # probes use plain L2 (default err_power) so vs_cls ratios are
        # directly comparable with the WENO7 teacherfree_lab monitors
        with torch.no_grad():
            cls, _ = rollout_loss("classical", ic, probe_h, args.cfl)
        probes[name] = (ic, max(float(cls), 1.0e-300))
        print(f"probe[{name}] h={probe_h} classical={float(cls):.4e}", flush=True)

    calib_gen = torch.Generator(device=device)
    calib_gen.manual_seed(4242)
    q_cal = make_bound_stencils(8192, device, calib_gen)
    with torch.no_grad():
        d_lin = {lr: W.optimal_d(lr, device).reshape(1, 3).expand(q_cal.shape[0], 3) for lr in W.LR_VALUES}
        v_lin = bound_violation(lambda lr: d_lin[lr], q_cal, args.bound_tol).mean()
        v_cls = bound_violation(lambda lr: W.classical_omega(q_cal, lr), q_cal, args.bound_tol).mean()
    print(f"bound_viol calibration: LINEAR={float(v_lin):.4e} CLASSICAL={float(v_cls):.4e}", flush=True)

    if args.ampgate_lambda > 0.0:
        ag_gen = torch.Generator(device=device)
        ag_gen.manual_seed(4343)
        q_ag = make_ampgate_stencils(8192, device, ag_gen, args.ampgate_amp_min, args.ampgate_amp_max)
        with torch.no_grad():
            kl_cls = torch.zeros(q_ag.shape[0], device=device, dtype=torch.float64)
            for lr in W.LR_VALUES:
                om_c = torch.clamp(W.classical_omega(q_ag, lr), min=1.0e-12)
                d = W.optimal_d(lr, device).reshape(1, 3)
                kl_cls = kl_cls + torch.sum(d * (torch.log(d) - torch.log(om_c)), dim=1)
            kl_cls = kl_cls / float(len(W.LR_VALUES))
        print(
            f"ampgate calibration: CLASSICAL={float(kl_cls.mean()):.4e} "
            f"(amp {args.ampgate_amp_min:.1e}..{args.ampgate_amp_max:.1e}, floor {args.ampgate_floor:.1e})",
            flush=True,
        )

    hist_csv = args.out_dir / "history.csv"
    eval_csv = args.out_dir / "eval.csv"
    hist_fields = [
        "step",
        "loss",
        "evolve_err",
        "tv_excess",
        "tv_pen",
        "kl_smooth",
        "bound",
        "kl_ampgate",
        "horizon",
        "grad",
        "skipped",
    ]
    eval_fields = ["step"] + [f"vs_cls_{k}" for k in IC_KINDS] + [
        "warp_sod_mlp_ref_l2",
        "warp_sod_cls_ref_l2",
        "warp_sod_rel_gain_l2",
        "warp_sod_failed",
        "warp_sod_plot_dir",
    ]

    skipped = 0
    sod_records: list[dict[str, object]] = []
    print(
        f"apost_weno5_start steps={args.steps} batch={args.batch} grid={args.grid} "
        f"horizons={horizons} cfl={args.cfl} lr={args.lr:.2e} tv_bg={args.tv_bg_lambda} "
        f"ic_probs={ic_probs} sod_eval={args.sod_eval} "
        f"sod={args.sod_nx}x{args.sod_ny}@t{args.sod_t_end}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        model.train()
        h = horizons[int(torch.randint(len(horizons), (1,), device=device, generator=gen).item())]
        u0 = make_ic(args.batch, args.grid, device, gen, probs=ic_probs)
        loss, stats = rollout_loss(
            model,
            u0,
            h,
            args.cfl,
            args.tv_lambda,
            args.err_power,
            args.tv_floor,
            args.tv_bg_lambda,
        )

        q_sm = make_smooth_stencils(args.smooth_batch, device, gen)
        sel = W.weno5_gamma_s(q_sm) < args.smooth_gamma_max
        kl_sm = torch.zeros((), device=device, dtype=torch.float64)
        if args.smooth_anchor_lambda > 0.0 and bool(sel.any()):
            kl_sm = kl_to_d(model(W.weno5_features(q_sm[sel]))).mean()
            loss = loss + args.smooth_anchor_lambda * torch.relu(kl_sm - args.anchor_floor)

        bnd = torch.zeros((), device=device, dtype=torch.float64)
        if args.bound_lambda > 0.0:
            q_b = make_bound_stencils(args.bound_batch, device, gen)
            r_b = model(W.weno5_features(q_b))
            bnd = bound_violation(lambda lr: W.omega_from_ratio(r_b, lr), q_b, args.bound_tol).mean()
            loss = loss + args.bound_lambda * torch.relu(bnd - args.bound_floor)

        kl_ag = torch.zeros((), device=device, dtype=torch.float64)
        if args.ampgate_lambda > 0.0:
            q_g = make_ampgate_stencils(args.ampgate_batch, device, gen, args.ampgate_amp_min, args.ampgate_amp_max)
            kl_ag = kl_to_d(model(W.weno5_features(q_g))).mean()
            loss = loss + args.ampgate_lambda * torch.relu(kl_ag - args.ampgate_floor)

        if not bool(torch.isfinite(loss)):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if sched is not None:
                sched.step()
            print(f"step={step:06d} NONFINITE loss, skipped (total {skipped})", flush=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not np.isfinite(grad) or grad > args.grad_skip:
            skipped += 1
            opt.zero_grad(set_to_none=True)
            print(f"step={step:06d} grad={grad:.2e} > {args.grad_skip:.1e}, skipped (total {skipped})", flush=True)
        else:
            opt.step()
        if sched is not None:
            sched.step()

        if step == 1 or step % args.log_interval == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "evolve_err": stats["evolve_err"],
                "tv_excess": stats.get("tv_excess", 0.0),
                "tv_pen": stats.get("tv_pen", 0.0),
                "kl_smooth": float(kl_sm.detach()),
                "bound": float(bnd.detach()),
                "kl_ampgate": float(kl_ag.detach()),
                "horizon": h,
                "grad": grad,
                "skipped": skipped,
            }
            append_csv(hist_csv, row, hist_fields)
            print(
                f"step={step:06d} loss={row['loss']:.3e} evolve={row['evolve_err']:.3e} "
                f"tv={row['tv_excess']:.3e} kl_sm={row['kl_smooth']:.3e} "
                f"bnd={row['bound']:.3e} kl_ag={row['kl_ampgate']:.3e} h={h} grad={grad:.2e}",
                flush=True,
            )

        if step == 1 or step % args.checkpoint_interval == 0 or step == args.steps:
            ckpt = args.out_dir / "checkpoints" / f"model_step_{step:06d}.npz"
            W.save_checkpoint(ckpt, model, {"raw_step": step, "recipe": "teacherfree_weno5_apost_advect"})

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            ckpt = args.out_dir / "checkpoints" / f"model_step_{step:06d}.npz"
            if not ckpt.exists():
                W.save_checkpoint(ckpt, model, {"raw_step": step, "recipe": "teacherfree_weno5_apost_advect"})
            model.eval()
            with torch.no_grad():
                ratios = {}
                for name, (ic, cls) in probes.items():
                    pe, _ = rollout_loss(model, ic, probe_h, args.cfl)
                    ratios[name] = float(pe) / cls
            row = {"step": step}
            for name in IC_KINDS:
                row[f"vs_cls_{name}"] = ratios[name]
            ratio_str = " ".join(f"{name}={ratios[name]:.2f}" for name in IC_KINDS)
            sod_str = ""
            if args.sod_eval and args.sod_nx > 0 and args.sod_ny > 0:
                sod_device = args.sod_device or args.device
                SV.prepare_warp(sod_device)
                payload = W.checkpoint_payload(
                    model,
                    {"raw_step": step, "recipe": "teacherfree_weno5_apost_advect"},
                )
                mlp_params = SV.wp_params_from_payload(payload, sod_device)
                sod = SV.run_warp_sod_validation(
                    step,
                    step,
                    mlp_params,
                    args.out_dir,
                    sod_device,
                    nx=args.sod_nx,
                    ny=args.sod_ny,
                    cfl=args.sod_cfl,
                    t_end=args.sod_t_end,
                    axis=args.sod_axis,
                    eno_cutoff=args.sod_eno_cutoff,
                    weno_space=args.sod_weno_space,
                    riemann_solver=args.sod_riemann_solver,
                )
                sod_records.append(sod)
                SV.write_warp_sod_outputs(args.out_dir, sod_records)
                row.update(
                    {
                        "warp_sod_mlp_ref_l2": sod.get("mlp_vs_reference_l2", float("nan")),
                        "warp_sod_cls_ref_l2": sod.get("classical_vs_reference_l2", float("nan")),
                        "warp_sod_rel_gain_l2": sod.get("rel_gain_vs_reference_l2", float("nan")),
                        "warp_sod_failed": sod.get("failed", float("nan")),
                        "warp_sod_plot_dir": sod.get("plot_dir", ""),
                    }
                )
                sod_str = (
                    f" warpSod[MLP_ref_L2={float(row['warp_sod_mlp_ref_l2']):.3e} "
                    f"CLS_ref_L2={float(row['warp_sod_cls_ref_l2']):.3e} "
                    f"relGain_L2={100.0 * float(row['warp_sod_rel_gain_l2']):.2f}% "
                    f"failed={int(float(row['warp_sod_failed']))}]"
                )
            append_csv(eval_csv, row, eval_fields)
            print(f"EVAL step={step:06d} vsCls[{ratio_str}]{sod_str}", flush=True)

    print(f"done out_dir={args.out_dir} skipped={skipped}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--cfl", type=float, default=0.2)
    p.add_argument("--horizons", type=int, nargs="+", default=[20, 40, 80, 120])
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--lr-final", type=float, default=1.0e-5)
    p.add_argument("--err-power", type=float, default=4.0)
    p.add_argument("--tv-lambda", type=float, default=0.0)
    p.add_argument("--tv-floor", type=float, default=0.0)
    p.add_argument("--tv-bg-lambda", type=float, default=0.03)
    p.add_argument("--smooth-anchor-lambda", type=float, default=1.0)
    p.add_argument("--smooth-batch", type=int, default=4096)
    p.add_argument("--smooth-gamma-max", type=float, default=0.1)
    p.add_argument("--anchor-floor", type=float, default=1.0e-3)
    p.add_argument("--bound-lambda", type=float, default=3.0)
    p.add_argument("--bound-batch", type=int, default=4096)
    p.add_argument("--bound-floor", type=float, default=2.0e-4)
    p.add_argument("--bound-tol", type=float, default=1.0e-3)
    p.add_argument("--ampgate-lambda", type=float, default=1.0)
    p.add_argument("--ampgate-batch", type=int, default=4096)
    p.add_argument("--ampgate-amp-min", type=float, default=1.0e-7)
    p.add_argument("--ampgate-amp-max", type=float, default=1.0e-3)
    p.add_argument("--ampgate-floor", type=float, default=1.0e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-skip", type=float, default=10.0)
    p.add_argument("--ic-probs", type=float, nargs=5, default=None)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--checkpoint-interval", type=int, default=250)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--sod-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sod-nx", type=int, default=100)
    p.add_argument("--sod-ny", type=int, default=10)
    p.add_argument("--sod-t-end", type=float, default=0.25)
    p.add_argument("--sod-cfl", type=float, default=0.4)
    p.add_argument("--sod-axis", choices=("x", "y"), default="x")
    p.add_argument("--sod-eno-cutoff", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sod-weno-space", choices=("characteristic", "conserved"), default="characteristic")
    p.add_argument("--sod-riemann-solver", choices=("force", "evilin"), default="evilin")
    p.add_argument("--sod-device", choices=("cuda", "cpu"), default="")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
