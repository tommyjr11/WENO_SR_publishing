#!/usr/bin/env python3
"""True-FVM teacher-free WENO5 training with full-step analytic supervision."""
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

from teacherfree_lab_weno5 import warp_sod_validation as SV
from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v4_fvm_e2e.apost_advect_fvm import (
    final_state_error,
    shortwave_stability_loss,
    trajectory_loss,
)
from teacherfree_lab_weno5_v4_fvm_e2e.fvm_profiles import (
    PROFILE_DEFAULT_PROBS,
    PROFILE_NAMES,
    make_profiles,
    make_shortwave_profiles,
)

torch.set_default_dtype(torch.float64)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def kl_to_d(ratios: torch.Tensor) -> torch.Tensor:
    kl = torch.zeros(ratios.shape[0], device=ratios.device)
    for lr in W.LR_VALUES:
        omega = torch.clamp(W.omega_from_ratio(ratios, lr), min=1.0e-12)
        d = W.optimal_d(lr, ratios.device).reshape(1, 3)
        kl = kl + torch.sum(d * (torch.log(d) - torch.log(omega)), dim=1)
    return kl / float(len(W.LR_VALUES))


def bound_violation(omega_fn, q: torch.Tensor, tol: float) -> torch.Tensor:
    dq = q[:, 1:] - q[:, :-1]
    monotone = ((dq >= 0.0).all(dim=1) | (dq <= 0.0).all(dim=1)).to(q.dtype)
    lo = torch.min(q, dim=1).values
    hi = torch.max(q, dim=1).values
    rng = torch.clamp(hi - lo, min=1.0e-13)
    total = torch.zeros(q.shape[0], device=q.device)
    for lr in W.LR_VALUES:
        value = torch.sum(omega_fn(lr) * W.candidate_values(q, lr), dim=1)
        total = total + torch.relu((value - hi) / rng - tol) + torch.relu((lo - value) / rng - tol)
    return monotone * total / float(len(W.LR_VALUES))


def _rand(
    gen: torch.Generator,
    device: torch.device,
    lo: float,
    hi: float,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen, dtype=torch.float64)


def make_smooth_fvm_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    """Cell averages of random cubics plus a weak sine on five unit cells."""
    c = torch.arange(-2.0, 3.0, device=device).reshape(1, 5)
    avg_x = c
    avg_x2 = torch.square(c) + 1.0 / 12.0
    avg_x3 = torch.pow(c, 3) + 0.25 * c
    base = _rand(gen, device, -2.0, 4.0, (n, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    slope = _rand(gen, device, -0.4, 0.4, (n, 1)) * scale
    quad = _rand(gen, device, -0.04, 0.04, (n, 1)) * scale
    cubic = _rand(gen, device, -0.01, 0.01, (n, 1)) * scale
    sine_amp = torch.pow(10.0, _rand(gen, device, -5.0, -2.0, (n, 1))) * scale
    phase = _rand(gen, device, 0.0, 2.0 * np.pi, (n, 1))
    kappa = _rand(gen, device, np.pi / 12.0, np.pi / 2.0, (n, 1))
    sinc = 2.0 * torch.sin(0.5 * kappa) / kappa
    sine_avg = sinc * torch.sin(kappa * c + phase)
    return base + slope * avg_x + quad * avg_x2 + cubic * avg_x3 + sine_amp * sine_avg


def _sigmoid_cell_average(lo: torch.Tensor, hi: torch.Tensor, center: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    sp = torch.nn.functional.softplus
    return width * (sp((hi - center) / width) - sp((lo - center) / width)) / (hi - lo)


def make_front_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    x = torch.arange(5.0, device=device).reshape(1, 5)
    lo, hi = x - 0.5, x + 0.5
    base = _rand(gen, device, -2.0, 4.0, (n, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    amp = torch.pow(10.0, _rand(gen, device, -3.0, 0.0, (n, 1))) * scale
    amp = amp * torch.where(torch.rand((n, 1), device=device, generator=gen) < 0.5, -1.0, 1.0)
    center = _rand(gen, device, 0.4, 3.6, (n, 1))
    width = torch.pow(10.0, _rand(gen, device, -1.3, 0.18, (n, 1)))
    return base + amp * _sigmoid_cell_average(lo, hi, center, width)


def make_monotone_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    base = _rand(gen, device, -2.0, 4.0, (n, 1))
    scale = torch.clamp(torch.abs(base), min=1.0)
    amp = torch.pow(10.0, _rand(gen, device, -4.0, 0.0, (n, 1))) * scale
    inc = torch.rand((n, 4), device=device, generator=gen, dtype=torch.float64)
    inc = inc / torch.clamp(torch.sum(inc, dim=1, keepdim=True), min=1.0e-13)
    q = torch.cat([torch.zeros((n, 1), device=device), torch.cumsum(inc, dim=1)], dim=1)
    q = base + amp * q
    return torch.where(torch.rand((n, 1), device=device, generator=gen) < 0.5, base - (q - base), q)


def make_bound_stencils(n: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    half = n // 2
    return torch.cat((make_front_stencils(half, device, gen), make_monotone_stencils(n - half, device, gen)), dim=0)


def make_ampgate_stencils(
    n: int,
    device: torch.device,
    gen: torch.Generator,
    amp_min: float,
    amp_max: float,
) -> torch.Tensor:
    q = make_bound_stencils(n, device, gen)
    mean = torch.mean(q, dim=1, keepdim=True)
    rng = torch.clamp(torch.max(q, dim=1).values - torch.min(q, dim=1).values, min=1.0e-13)
    qscale = torch.clamp(torch.max(torch.abs(q), dim=1).values, min=1.0)
    amp = torch.pow(10.0, _rand(gen, device, np.log10(amp_min), np.log10(amp_max), (n,)))
    factor = (amp * qscale / rng).reshape(-1, 1)
    return mean + factor * (q - mean)


def save_model(path: Path, model: W.SharedBadnessMLP, step: int, args: argparse.Namespace) -> None:
    W.save_checkpoint(
        path,
        model,
        {
            "raw_step": step,
            "recipe": "weno5_v4_true_fvm_fullstep_e2e",
            "training_cfl": args.cfl,
            "true_fvm_cell_averages": True,
            "full_step_exact_state_and_point_targets": True,
            "shortwave_stability": True,
        },
    )


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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.lr_final,
    )

    horizons = tuple(int(value) for value in args.horizons)
    profile_probs = tuple(float(value) for value in args.profile_probs)
    probe_gen = torch.Generator(device=device)
    probe_gen.manual_seed(991)
    probe_horizon = horizons[len(horizons) // 2]
    probes = {
        name: make_profiles(32, args.grid, device, probe_gen, kind=name)
        for name in PROFILE_NAMES
    }
    probe_cls = {
        name: max(final_state_error("classical", profile, args.grid, probe_horizon, args.cfl), 1.0e-300)
        for name, profile in probes.items()
    }
    for name in PROFILE_NAMES:
        print(f"probe[{name}] h={probe_horizon} classical={probe_cls[name]:.4e}", flush=True)

    eval_short = make_shortwave_profiles(48, args.grid, device, probe_gen)
    history_fields = [
        "step", "loss", "trajectory", "face_path", "exact_recon", "tv_excess",
        "shortwave", "short_gain_model", "short_gain_linear", "short_harmonic",
        "kl_smooth", "bound", "kl_ampgate", "horizon", "grad", "skipped",
    ]
    eval_fields = ["step"] + [f"vs_cls_{name}" for name in PROFILE_NAMES] + [
        "short_gain_model", "short_gain_linear", "short_harmonic",
        "warp_sod_mlp_ref_l2", "warp_sod_cls_ref_l2", "warp_sod_rel_gain_l2",
        "warp_sod_failed", "warp_sod_plot_dir",
    ]
    history_csv = args.out_dir / "history.csv"
    eval_csv = args.out_dir / "eval.csv"
    sod_records: list[dict[str, object]] = []
    skipped = 0

    print(
        f"weno5_v4_start steps={args.steps} batch={args.batch} grid={args.grid} "
        f"cfl={args.cfl} horizons={horizons} true_fvm=True fullstep_targets=True "
        f"face_lambda={args.face_path_lambda} recon_lambda={args.exact_recon_lambda} "
        f"shortwave_lambda={args.shortwave_lambda} tv_bg={args.tv_bg_lambda} "
        f"profile_probs={profile_probs}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        model.train()
        horizon = horizons[int(torch.randint(len(horizons), (1,), device=device, generator=gen).item())]
        profiles = make_profiles(args.batch, args.grid, device, gen, probs=profile_probs)
        loss, stats = trajectory_loss(
            model,
            profiles,
            args.grid,
            horizon,
            args.cfl,
            args.face_path_lambda,
            args.exact_recon_lambda,
            args.tv_bg_lambda,
        )

        short_profiles = make_shortwave_profiles(args.shortwave_batch, args.grid, device, gen)
        short_loss, short_stats = shortwave_stability_loss(
            model,
            short_profiles,
            args.grid,
            args.shortwave_horizon,
            args.cfl,
        )
        loss = loss + args.shortwave_lambda * short_loss

        q_smooth = make_smooth_fvm_stencils(args.smooth_batch, device, gen)
        smooth_select = W.weno5_gamma_s(q_smooth) < args.smooth_gamma_max
        kl_smooth = torch.zeros((), device=device)
        if args.smooth_anchor_lambda > 0.0 and bool(smooth_select.any()):
            kl_smooth = torch.mean(kl_to_d(model(W.weno5_features(q_smooth[smooth_select]))))
            loss = loss + args.smooth_anchor_lambda * torch.relu(kl_smooth - args.anchor_floor)

        bound = torch.zeros((), device=device)
        if args.bound_lambda > 0.0:
            q_bound = make_bound_stencils(args.bound_batch, device, gen)
            ratios = model(W.weno5_features(q_bound))
            bound = torch.mean(
                bound_violation(lambda lr: W.omega_from_ratio(ratios, lr), q_bound, args.bound_tol)
            )
            loss = loss + args.bound_lambda * torch.relu(bound - args.bound_floor)

        kl_ampgate = torch.zeros((), device=device)
        if args.ampgate_lambda > 0.0:
            q_gate = make_ampgate_stencils(
                args.ampgate_batch,
                device,
                gen,
                args.ampgate_amp_min,
                args.ampgate_amp_max,
            )
            kl_ampgate = torch.mean(kl_to_d(model(W.weno5_features(q_gate))))
            loss = loss + args.ampgate_lambda * torch.relu(kl_ampgate - args.ampgate_floor)

        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            skipped += 1
            scheduler.step()
            print(f"step={step:06d} NONFINITE loss, skipped={skipped}", flush=True)
            continue
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not np.isfinite(grad) or grad > args.grad_skip:
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            print(f"step={step:06d} grad={grad:.3e} skipped={skipped}", flush=True)
        else:
            optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_interval == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                **stats,
                **short_stats,
                "kl_smooth": float(kl_smooth.detach()),
                "bound": float(bound.detach()),
                "kl_ampgate": float(kl_ampgate.detach()),
                "horizon": horizon,
                "grad": grad,
                "skipped": skipped,
            }
            append_csv(history_csv, row, history_fields)
            print(
                f"step={step:06d} loss={row['loss']:.3e} traj={row['trajectory']:.3e} "
                f"face={row['face_path']:.3e} recon={row['exact_recon']:.3e} "
                f"tv={row['tv_excess']:.3e} short={row['shortwave']:.3e} "
                f"gain={row['short_gain_model']:.6f}/{row['short_gain_linear']:.6f} "
                f"sm={row['kl_smooth']:.3e} bnd={row['bound']:.3e} "
                f"gate={row['kl_ampgate']:.3e} h={horizon} grad={grad:.3e}",
                flush=True,
            )

        checkpoint = args.out_dir / "checkpoints" / f"model_step_{step:06d}.npz"
        if step == 1 or step % args.checkpoint_interval == 0 or step == args.steps:
            save_model(checkpoint, model, step, args)

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            if not checkpoint.exists():
                save_model(checkpoint, model, step, args)
            model.eval()
            with torch.no_grad():
                ratios = {
                    name: final_state_error(model, profile, args.grid, probe_horizon, args.cfl) / probe_cls[name]
                    for name, profile in probes.items()
                }
                _, sw_eval = shortwave_stability_loss(
                    model,
                    eval_short,
                    args.grid,
                    args.shortwave_horizon,
                    args.cfl,
                )
            eval_row: dict[str, object] = {"step": step, **sw_eval}
            for name in PROFILE_NAMES:
                eval_row[f"vs_cls_{name}"] = ratios[name]
            sod_text = ""
            if args.sod_eval:
                sod_device = args.sod_device or args.device
                SV.prepare_warp(sod_device)
                payload = W.checkpoint_payload(
                    model,
                    {"raw_step": step, "recipe": "weno5_v4_true_fvm_fullstep_e2e"},
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
                    weno_space="characteristic",
                    riemann_solver="evilin",
                )
                sod_records.append(sod)
                SV.write_warp_sod_outputs(args.out_dir, sod_records)
                eval_row.update(
                    {
                        "warp_sod_mlp_ref_l2": sod.get("mlp_vs_reference_l2", float("nan")),
                        "warp_sod_cls_ref_l2": sod.get("classical_vs_reference_l2", float("nan")),
                        "warp_sod_rel_gain_l2": sod.get("rel_gain_vs_reference_l2", float("nan")),
                        "warp_sod_failed": sod.get("failed", float("nan")),
                        "warp_sod_plot_dir": sod.get("plot_dir", ""),
                    }
                )
                sod_text = (
                    f" SOD[gain={100.0 * float(eval_row['warp_sod_rel_gain_l2']):.2f}% "
                    f"failed={int(float(eval_row['warp_sod_failed']))}]"
                )
            append_csv(eval_csv, eval_row, eval_fields)
            ratio_text = " ".join(f"{name}={ratios[name]:.2f}" for name in PROFILE_NAMES)
            print(
                f"EVAL step={step:06d} vsCls[{ratio_text}] "
                f"shortGain={sw_eval['short_gain_model']:.6f}/{sw_eval['short_gain_linear']:.6f}{sod_text}",
                flush=True,
            )

    print(f"done out_dir={args.out_dir} skipped={skipped}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--cfl", type=float, default=0.6)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 10, 20, 40, 80, 120))
    parser.add_argument("--profile-probs", type=float, nargs=len(PROFILE_NAMES), default=PROFILE_DEFAULT_PROBS)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--lr-final", type=float, default=1.0e-5)
    parser.add_argument("--face-path-lambda", type=float, default=0.20)
    parser.add_argument("--exact-recon-lambda", type=float, default=0.10)
    parser.add_argument("--shortwave-lambda", type=float, default=0.20)
    parser.add_argument("--shortwave-batch", type=int, default=24)
    parser.add_argument("--shortwave-horizon", type=int, default=20)
    parser.add_argument("--tv-bg-lambda", type=float, default=0.03)
    parser.add_argument("--smooth-anchor-lambda", type=float, default=1.0)
    parser.add_argument("--smooth-batch", type=int, default=2048)
    parser.add_argument("--smooth-gamma-max", type=float, default=0.1)
    parser.add_argument("--anchor-floor", type=float, default=1.0e-3)
    parser.add_argument("--bound-lambda", type=float, default=3.0)
    parser.add_argument("--bound-batch", type=int, default=2048)
    parser.add_argument("--bound-floor", type=float, default=2.0e-4)
    parser.add_argument("--bound-tol", type=float, default=1.0e-3)
    parser.add_argument("--ampgate-lambda", type=float, default=1.0)
    parser.add_argument("--ampgate-batch", type=int, default=2048)
    parser.add_argument("--ampgate-amp-min", type=float, default=1.0e-7)
    parser.add_argument("--ampgate-amp-max", type=float, default=3.0e-3)
    parser.add_argument("--ampgate-floor", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-skip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sod-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sod-nx", type=int, default=100)
    parser.add_argument("--sod-ny", type=int, default=10)
    parser.add_argument("--sod-t-end", type=float, default=0.25)
    parser.add_argument("--sod-cfl", type=float, default=0.4)
    parser.add_argument("--sod-axis", choices=("x", "y"), default="x")
    parser.add_argument("--sod-eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sod-device", choices=("", "cuda", "cpu"), default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
