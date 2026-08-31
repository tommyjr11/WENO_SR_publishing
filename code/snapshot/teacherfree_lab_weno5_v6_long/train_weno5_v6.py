#!/usr/bin/env python3
"""From-scratch WENO5-v6 long training at SSPRK3 CFL 0.5."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teacherfree_lab_weno5 import warp_sod_validation as SV
from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v4_fvm_e2e.apost_advect_fvm import final_state_error
from teacherfree_lab_weno5_v4_fvm_e2e.fvm_profiles import PROFILE_NAMES, make_profiles
from teacherfree_lab_weno5_v4_fvm_e2e.train_weno5_v4 import (
    append_csv,
    bound_violation,
    kl_to_d,
    make_ampgate_stencils,
    make_bound_stencils,
    make_smooth_fvm_stencils,
)
from teacherfree_lab_weno5_v5_fvm_e2e.v5_losses import (
    jump_shedding_loss,
    make_jump_wavepackets,
    make_shortwave_profiles_v5,
    shortwave_corridor_loss,
    trajectory_loss_v5,
)
from teacherfree_lab_weno5_v6_long.gste_monitor import GsteMonitor

torch.set_default_dtype(torch.float64)


def scheduled_lr(completed_step: int, args: argparse.Namespace) -> float:
    """V5's 30k cosine followed by a long low-rate refinement tail."""
    if completed_step <= args.lr_phase1_end:
        phase = completed_step / float(args.lr_phase1_end)
        blend = 0.5 * (1.0 + math.cos(math.pi * phase))
        return args.lr_phase1_final + (args.lr - args.lr_phase1_final) * blend
    if completed_step <= args.lr_tail_start:
        return args.lr_phase1_final
    phase = min(
        1.0,
        (completed_step - args.lr_tail_start) / float(args.steps - args.lr_tail_start),
    )
    blend = 0.5 * (1.0 + math.cos(math.pi * phase))
    return args.lr_tail_final + (args.lr_phase1_final - args.lr_tail_final) * blend


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda completed_step: scheduled_lr(completed_step, args) / args.lr,
    )


def truncate_csv_after(path: Path, completed_step: int) -> None:
    """Discard rows newer than a restored state to prevent duplicate history."""
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["step"]) <= completed_step]
    if not fields:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_state(
    path: Path,
    latest: Path,
    step: int,
    model: W.SharedBadnessMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    gen: torch.Generator,
    skipped: int,
    sod_records: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    payload: dict[str, object] = {
        "format_version": 1,
        "recipe": "weno5_v6_long_v5_objective_cfl05",
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_generator_state": gen.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "skipped": skipped,
        "sod_records": sod_records,
        "schedule": {
            "lr": args.lr,
            "phase1_end": args.lr_phase1_end,
            "phase1_final": args.lr_phase1_final,
            "tail_start": args.lr_tail_start,
            "tail_final": args.lr_tail_final,
            "steps": args.steps,
        },
    }
    atomic_torch_save(payload, path)
    atomic_torch_save(payload, latest)


def save_model(path: Path, model: W.SharedBadnessMLP, step: int, args: argparse.Namespace) -> None:
    W.save_checkpoint(
        path,
        model,
        {
            "raw_step": step,
            "recipe": "weno5_v6_long_v5_objective_cfl05",
            "training_cfl": args.cfl,
            "true_fvm_cell_averages": True,
            "from_scratch_linear_d_initialization": True,
            "full_step_targets": True,
            "flat_d2_loss": True,
            "shortwave_damping_corridor": True,
            "jump_wavepacket_shedding": True,
            "training_steps": args.steps,
            "lr_phase1_end": args.lr_phase1_end,
            "lr_phase1_final": args.lr_phase1_final,
            "lr_tail_start": args.lr_tail_start,
            "lr_tail_final": args.lr_tail_final,
        },
    )


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not (0 < args.lr_phase1_end <= args.lr_tail_start < args.steps):
        raise ValueError("LR schedule requires 0 < phase1_end <= tail_start < steps")
    if not (args.lr > args.lr_phase1_final > args.lr_tail_final > 0.0):
        raise ValueError("LR values must satisfy lr > phase1_final > tail_final > 0")
    if args.state_interval <= 0:
        raise ValueError("state interval must be positive")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    W.check_weno5_coefficients()
    device = W.torch_device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # w4=b4=0 in this constructor, so the first checkpoint is exactly linear d.
    model = W.SharedBadnessMLP(seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args)

    horizons = tuple(int(value) for value in args.horizons)
    profile_probs = tuple(float(value) for value in args.profile_probs)
    if abs(sum(profile_probs) - 1.0) > 1.0e-12:
        raise ValueError("profile probabilities must sum to one")

    probe_gen = torch.Generator(device=device)
    probe_gen.manual_seed(991)
    probe_horizon = horizons[len(horizons) // 2]
    probes = {
        name: make_profiles(32, args.grid, device, probe_gen, kind=name)
        for name in PROFILE_NAMES
    }
    probe_cls = {
        cfl: {
            name: max(final_state_error("classical", profile, args.grid, probe_horizon, cfl), 1.0e-300)
            for name, profile in probes.items()
        }
        for cfl in (0.4, args.cfl, 0.6)
    }
    eval_short, eval_cpw = make_shortwave_profiles_v5(48, args.grid, device, probe_gen)
    eval_jump = make_jump_wavepackets(32, args.grid, device, probe_gen)

    history_fields = [
        "step", "loss", "trajectory", "face_path", "exact_recon", "flat_d2", "tv_excess",
        "shortwave", "short_upper", "short_lower", "short_gain_model", "short_gain_linear",
        "short_harmonic", "jump_shedding", "jump_flat_pen", "jump_tv_pen",
        "kl_smooth", "bound", "kl_ampgate", "horizon", "lr", "grad", "skipped",
    ]
    eval_fields = ["step"] + [f"vs_cls_{name}" for name in PROFILE_NAMES] + [
        "short_gain_model", "short_gain_linear", "short_upper", "short_lower", "short_harmonic",
        "jump_shedding", "jump_flat_pen", "jump_tv_pen",
        "cfl04_mean_vs_cls", "cfl06_mean_vs_cls",
        "warp_sod_mlp_ref_l2", "warp_sod_cls_ref_l2", "warp_sod_rel_gain_l2",
        "warp_sod_failed", "warp_sod_plot_dir",
    ]
    history_csv = args.out_dir / "history.csv"
    eval_csv = args.out_dir / "eval.csv"
    gste_csv = args.out_dir / "gste_eval.csv"
    gste_fields = [
        "step", "l1", "l2", "linf", "tv", "min", "max", "complete",
        "gain_l1_vs_js", "gain_l2_vs_js", "js_l1", "js_l2", "js_tv",
        "steps", "cfl", "t",
    ]
    sod_records: list[dict[str, object]] = []
    skipped = 0
    start_step = 1

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("recipe") != "weno5_v6_long_v5_objective_cfl05":
            raise ValueError(f"incompatible resume state: {state.get('recipe')!r}")
        expected_schedule = {
            "lr": args.lr,
            "phase1_end": args.lr_phase1_end,
            "phase1_final": args.lr_phase1_final,
            "tail_start": args.lr_tail_start,
            "tail_final": args.lr_tail_final,
            "steps": args.steps,
        }
        if state.get("schedule") != expected_schedule:
            raise ValueError(
                f"resume LR schedule mismatch: saved={state.get('schedule')} "
                f"requested={expected_schedule}"
            )
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        gen.set_state(state["training_generator_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if args.device == "cuda" and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
            )
        completed_step = int(state["step"])
        start_step = completed_step + 1
        skipped = int(state.get("skipped", 0))
        sod_records = list(state.get("sod_records", []))
        for csv_path in (history_csv, eval_csv, gste_csv):
            truncate_csv_after(csv_path, completed_step)
        print(
            f"RESUME state={args.resume} completed_step={completed_step} "
            f"next_step={start_step} lr={scheduler.get_last_lr()[0]:.3e} skipped={skipped}",
            flush=True,
        )

    gste_monitor = None
    if args.gste_interval > 0:
        gste_monitor = GsteMonitor(
            args.gste_nx,
            args.gste_t_end,
            args.gste_cfl,
            args.gste_quadrature,
            device,
        )
        print(
            f"GSTE monitor nx={args.gste_nx} t={args.gste_t_end} "
            f"cfl={gste_monitor.actual_cfl:.8f} interval={args.gste_interval} "
            f"JS_L1={float(gste_monitor.classical['l1']):.6e} "
            f"JS_L2={float(gste_monitor.classical['l2']):.6e}",
            flush=True,
        )

    print(
        f"weno5_v6_long_start from_scratch={args.resume is None} steps={args.steps} "
        f"batch={args.batch} grid={args.grid} cfl={args.cfl} horizons={horizons} "
        f"lr={args.lr:.2e}->{args.lr_phase1_final:.2e}@{args.lr_phase1_end} "
        f"hold_to={args.lr_tail_start} ->{args.lr_tail_final:.2e}@{args.steps} "
        f"face={args.face_path_lambda} recon={args.exact_recon_lambda} flat={args.flat_d2_lambda} "
        f"short={args.shortwave_lambda} jump={args.jump_lambda} tv={args.tv_bg_lambda} "
        f"profile_probs={profile_probs}",
        flush=True,
    )

    run_end = args.steps if args.stop_after_step is None else min(args.stop_after_step, args.steps)
    if start_step > run_end:
        print(f"already complete at step={start_step - 1}", flush=True)
        return

    for step in range(start_step, run_end + 1):
        model.train()
        horizon = horizons[int(torch.randint(len(horizons), (1,), device=device, generator=gen).item())]
        profiles = make_profiles(args.batch, args.grid, device, gen, probs=profile_probs)
        loss, stats = trajectory_loss_v5(
            model,
            profiles,
            args.grid,
            horizon,
            args.cfl,
            args.face_path_lambda,
            args.exact_recon_lambda,
            args.flat_d2_lambda,
            args.flat_tolerance,
            args.tv_bg_lambda,
        )

        short_profiles, cpw = make_shortwave_profiles_v5(args.shortwave_batch, args.grid, device, gen)
        short_loss, short_stats = shortwave_corridor_loss(
            model,
            short_profiles,
            cpw,
            args.grid,
            args.shortwave_horizon,
            args.cfl,
            args.shortwave_lower_margin,
        )
        loss = loss + args.shortwave_lambda * short_loss

        jump_batch = make_jump_wavepackets(args.jump_batch, args.grid, device, gen)
        jump_loss, jump_stats = jump_shedding_loss(
            model,
            jump_batch,
            args.grid,
            args.jump_horizon,
            args.cfl,
            args.flat_tolerance,
        )
        loss = loss + args.jump_lambda * jump_loss

        q_smooth = make_smooth_fvm_stencils(args.smooth_batch, device, gen)
        smooth_select = W.weno5_gamma_s(q_smooth) < args.smooth_gamma_max
        kl_smooth = torch.zeros((), device=device)
        if args.smooth_anchor_lambda > 0.0 and bool(smooth_select.any()):
            kl_smooth = torch.mean(kl_to_d(model(W.weno5_features(q_smooth[smooth_select]))))
            loss = loss + args.smooth_anchor_lambda * torch.relu(kl_smooth - args.anchor_floor)

        q_bound = make_bound_stencils(args.bound_batch, device, gen)
        ratios_bound = model(W.weno5_features(q_bound))
        bound = torch.mean(
            bound_violation(lambda lr: W.omega_from_ratio(ratios_bound, lr), q_bound, args.bound_tol)
        )
        loss = loss + args.bound_lambda * torch.relu(bound - args.bound_floor)

        q_gate = make_ampgate_stencils(
            args.ampgate_batch, device, gen, args.ampgate_amp_min, args.ampgate_amp_max
        )
        kl_ampgate = torch.mean(kl_to_d(model(W.weno5_features(q_gate))))
        loss = loss + args.ampgate_lambda * torch.relu(kl_ampgate - args.ampgate_floor)

        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            skipped += 1
            scheduler.step()
            print(f"step={step:06d} NONFINITE loss skipped={skipped}", flush=True)
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
                **jump_stats,
                "kl_smooth": float(kl_smooth.detach()),
                "bound": float(bound.detach()),
                "kl_ampgate": float(kl_ampgate.detach()),
                "horizon": horizon,
                "lr": scheduler.get_last_lr()[0],
                "grad": grad,
                "skipped": skipped,
            }
            append_csv(history_csv, row, history_fields)
            print(
                f"step={step:06d} loss={row['loss']:.3e} traj={row['trajectory']:.3e} "
                f"face={row['face_path']:.3e} recon={row['exact_recon']:.3e} "
                f"flat={row['flat_d2']:.3e} tv={row['tv_excess']:.3e} "
                f"short={row['shortwave']:.3e}[up={row['short_upper']:.1e},lo={row['short_lower']:.1e}] "
                f"jump={row['jump_shedding']:.3e} sm={row['kl_smooth']:.3e} "
                f"bnd={row['bound']:.3e} gate={row['kl_ampgate']:.3e} "
                f"h={horizon} lr={row['lr']:.2e} grad={grad:.3e}",
                flush=True,
            )

        checkpoint = args.out_dir / "checkpoints" / f"model_step_{step:06d}.npz"
        if step == 1 or step % args.checkpoint_interval == 0 or step == run_end:
            save_model(checkpoint, model, step, args)

        if step == 1 or step % args.eval_interval == 0 or step == run_end:
            if not checkpoint.exists():
                save_model(checkpoint, model, step, args)
            model.eval()
            with torch.no_grad():
                ratios = {
                    name: final_state_error(model, profile, args.grid, probe_horizon, args.cfl)
                    / probe_cls[args.cfl][name]
                    for name, profile in probes.items()
                }
                _, short_eval = shortwave_corridor_loss(
                    model, eval_short, eval_cpw, args.grid, args.shortwave_horizon,
                    args.cfl, args.shortwave_lower_margin,
                )
                _, jump_eval = jump_shedding_loss(
                    model, eval_jump, args.grid, args.jump_horizon,
                    args.cfl, args.flat_tolerance,
                )
            eval_row: dict[str, object] = {"step": step, **short_eval, **jump_eval}
            for name in PROFILE_NAMES:
                eval_row[f"vs_cls_{name}"] = ratios[name]

            if step == 1 or step % args.cross_cfl_interval == 0 or step == run_end:
                with torch.no_grad():
                    for cfl, key in ((0.4, "cfl04_mean_vs_cls"), (0.6, "cfl06_mean_vs_cls")):
                        values = [
                            final_state_error(model, profile, args.grid, probe_horizon, cfl)
                            / probe_cls[cfl][name]
                            for name, profile in probes.items()
                        ]
                        eval_row[key] = float(np.mean(values))

            if args.sod_eval:
                sod_device = args.sod_device or args.device
                SV.prepare_warp(sod_device)
                payload = W.checkpoint_payload(
                    model,
                    {"raw_step": step, "recipe": "weno5_v6_long_v5_objective_cfl05"},
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
                    eno_cutoff=False,
                    weno_space="characteristic",
                    riemann_solver="evilin",
                )
                sod_records.append(sod)
                SV.write_warp_sod_outputs(args.out_dir, sod_records)
                eval_row.update({
                    "warp_sod_mlp_ref_l2": sod.get("mlp_vs_reference_l2", float("nan")),
                    "warp_sod_cls_ref_l2": sod.get("classical_vs_reference_l2", float("nan")),
                    "warp_sod_rel_gain_l2": sod.get("rel_gain_vs_reference_l2", float("nan")),
                    "warp_sod_failed": sod.get("failed", float("nan")),
                    "warp_sod_plot_dir": sod.get("plot_dir", ""),
                })
            append_csv(eval_csv, eval_row, eval_fields)
            ratio_text = " ".join(f"{name}={ratios[name]:.2f}" for name in PROFILE_NAMES)
            print(
                f"EVAL step={step:06d} vsCls[{ratio_text}] "
                f"shortGain={short_eval['short_gain_model']:.6f}/{short_eval['short_gain_linear']:.6f} "
                f"jump={jump_eval['jump_shedding']:.2e} "
                + (
                    f"SOD[gain={100.0 * float(eval_row['warp_sod_rel_gain_l2']):.2f}% "
                    f"failed={int(float(eval_row['warp_sod_failed']))}]"
                    if args.sod_eval else "SOD[disabled]"
                ),
                flush=True,
            )

        if gste_monitor is not None and (
            step % args.gste_interval == 0 or step == run_end
        ):
            gste_row = gste_monitor.evaluate(model, step)
            append_csv(gste_csv, gste_row, gste_fields)
            print(
                f"GSTE step={step:06d} L1={float(gste_row['l1']):.6e} "
                f"L2={float(gste_row['l2']):.6e} TV={float(gste_row['tv']):.6f} "
                f"range=[{float(gste_row['min']):.6e},{float(gste_row['max']):.6e}] "
                f"gain=[{100.0 * float(gste_row['gain_l1_vs_js']):.2f}%,"
                f"{100.0 * float(gste_row['gain_l2_vs_js']):.2f}%]",
                flush=True,
            )

        if step % args.state_interval == 0 or step == run_end:
            state_dir = args.out_dir / "training_state"
            state_path = state_dir / f"state_step_{step:06d}.pt"
            save_training_state(
                state_path,
                state_dir / "latest.pt",
                step,
                model,
                optimizer,
                scheduler,
                gen,
                skipped,
                sod_records,
                args,
            )
            print(f"STATE step={step:06d} path={state_path}", flush=True)

    status = "complete" if run_end == args.steps else "stopped"
    print(f"{status} step={run_end} out_dir={args.out_dir} skipped={skipped}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 10, 20, 40, 80, 120))
    parser.add_argument(
        "--profile-probs", type=float, nargs=len(PROFILE_NAMES),
        default=(0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10),
    )
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lr-phase1-end", type=int, default=30000)
    parser.add_argument("--lr-phase1-final", type=float, default=3.0e-6)
    parser.add_argument("--lr-tail-start", type=int, default=120000)
    parser.add_argument("--lr-tail-final", type=float, default=3.0e-7)
    parser.add_argument("--face-path-lambda", type=float, default=0.04)
    parser.add_argument("--exact-recon-lambda", type=float, default=0.15)
    parser.add_argument("--flat-d2-lambda", type=float, default=0.05)
    parser.add_argument("--flat-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--shortwave-lambda", type=float, default=0.20)
    parser.add_argument("--shortwave-batch", type=int, default=24)
    parser.add_argument("--shortwave-horizon", type=int, default=20)
    parser.add_argument("--shortwave-lower-margin", type=float, default=0.02)
    parser.add_argument("--jump-lambda", type=float, default=0.10)
    parser.add_argument("--jump-batch", type=int, default=16)
    parser.add_argument("--jump-horizon", type=int, default=20)
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
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--cross-cfl-interval", type=int, default=1000)
    parser.add_argument("--state-interval", type=int, default=2500)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--gste-interval", type=int, default=2500)
    parser.add_argument("--gste-nx", type=int, default=200)
    parser.add_argument("--gste-t-end", type=float, default=10.0)
    parser.add_argument("--gste-cfl", type=float, default=0.5)
    parser.add_argument("--gste-quadrature", type=int, default=6)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--sod-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sod-nx", type=int, default=100)
    parser.add_argument("--sod-ny", type=int, default=10)
    parser.add_argument("--sod-t-end", type=float, default=0.25)
    parser.add_argument("--sod-cfl", type=float, default=0.4)
    parser.add_argument("--sod-axis", choices=("x", "y"), default="x")
    parser.add_argument("--sod-device", choices=("", "cuda", "cpu"), default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
