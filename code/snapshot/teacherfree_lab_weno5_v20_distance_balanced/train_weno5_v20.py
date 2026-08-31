#!/usr/bin/env python3
"""Train WENO5 V20 with equal-probability propagation distances at CFL 0.5."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teacherfree_lab_weno5 import weno5_core as W
from teacherfree_lab_weno5_v4_fvm_e2e.fvm_profiles import (
    PROFILE_NAMES,
    make_profiles,
)
from teacherfree_lab_weno5_v4_fvm_e2e.train_weno5_v4 import append_csv
from teacherfree_lab_weno5_v6_long.train_weno5_v6 import (
    atomic_torch_save,
    truncate_csv_after,
)
from teacherfree_lab_weno5_v12_reflection_sym.v12_losses import (
    balanced_velocities,
    final_state_error_signed,
)
from teacherfree_lab_weno5_v12_reflection_sym.v12_model import (
    ReflectionSymmetricBadnessMLP,
    reflection_defect,
)
from teacherfree_lab_weno5_v20_distance_balanced.v20_losses import (
    checkpointed_autoregressive_trajectory_loss,
)

torch.set_default_dtype(torch.float64)

RECIPE = "weno5_v20_single_cfl_equal_probability_distance_balanced"


def make_scheduler(optimizer, start_lr: float, final_lr: float, steps: int):
    ratio = final_lr / start_lr

    def factor(index: int) -> float:
        progress = min(max(index / float(max(1, steps)), 0.0), 1.0)
        return ratio + 0.5 * (1.0 - ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def steps_for_distance(distance: float, cfl: float) -> int:
    raw = float(distance) / float(cfl)
    rounded = int(round(raw))
    if rounded < 1 or abs(raw - rounded) > 1.0e-12:
        raise ValueError(
            f"distance={distance:g} must contain an integer number of "
            f"full SSPRK3 steps at CFL={cfl:g}"
        )
    return rounded


def distance_index_for_step(step: int, count: int, seed: int) -> int:
    """Visit every distance once per cycle with reproducible shuffled order."""
    if step < 1 or count < 1:
        raise ValueError("step and distance count must be positive")
    cycle, offset = divmod(step - 1, count)
    cycle_gen = torch.Generator(device="cpu")
    cycle_gen.manual_seed(seed + 1_000_003 * cycle)
    order = torch.randperm(count, generator=cycle_gen).tolist()
    return int(order[offset])


def signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "steps": args.steps,
        "grid": args.grid,
        "distances": tuple(args.distances),
        "distance_batches": tuple(args.distance_batches),
        "profile_probs": tuple(args.profile_probs),
        "primary_cfl": args.primary_cfl,
        "edge_cfls": tuple(args.edge_cfls),
        "edge_max_steps": args.edge_max_steps,
        "edge_lambda": args.edge_lambda,
        "global_guard_lambda": args.global_guard_lambda,
        "local_guard_lambda": args.local_guard_lambda,
        "local_window": args.local_window,
        "tv_lambda": args.tv_lambda,
        "reflection": True,
        "velocities": (-1.0, 1.0),
    }


def save_model(path: Path, model, step: int, args: argparse.Namespace) -> None:
    W.save_checkpoint(
        path,
        model,
        {
            "raw_step": step,
            "recipe": RECIPE,
            "from_scratch_linear_d_initialization": True,
            "true_fvm_cell_averages": True,
            "autoregressive_model_state_in_graph": True,
            "rollout_activation_checkpointing": True,
            "rollout_state_detached": False,
            "exact_state_target_at_every_complete_ssprk3_step": True,
            "primary_cfl": args.primary_cfl,
            "propagation_distances_cells": list(args.distances),
            "distance_sampling": "one_each_per_shuffled_cycle",
            "distance_batches": list(args.distance_batches),
            "classical_baseline": "WENO5-JS with the same SSPRK3 step and CFL",
            "classical_global_nonregression": True,
            "classical_local_nonregression_window": args.local_window,
            "classical_guard_aggregation": "0.25_family_mean_plus_0.75_cvar25",
            "paper_test_profiles_used_for_training": False,
            "training_profiles": list(PROFILE_NAMES),
            "reflection_symmetrized_forward": True,
            "reflection_formula": "0.5*(M(x)+P*M(P*x))",
            "checkpoint_stores_shared_raw_parameters": True,
            "deployment_requires_reflection_symmetrization": True,
            "advection_velocities": [-1.0, 1.0],
        },
    )


def save_state(
    path: Path,
    latest: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    gen: torch.Generator,
    skipped: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "format_version": 1,
        "recipe": RECIPE,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "training_generator_state": gen.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "skipped": skipped,
        "training_signature": signature(args),
    }
    atomic_torch_save(payload, path)
    atomic_torch_save(payload, latest)


def validate_args(args: argparse.Namespace) -> None:
    if len(args.distances) != len(args.distance_batches):
        raise ValueError("distances and distance-batches must have equal length")
    if not args.distances or min(args.distances) <= 0.0:
        raise ValueError("all propagation distances must be positive")
    if len(set(args.distances)) != len(args.distances):
        raise ValueError("propagation distances must be unique")
    for batch in args.distance_batches:
        if batch < 2 or batch % 2:
            raise ValueError("every dynamic batch must be positive and even")
    for distance in args.distances:
        steps_for_distance(distance, args.primary_cfl)
    if len(args.profile_probs) != len(PROFILE_NAMES):
        raise ValueError("one profile probability is required per profile family")
    if abs(sum(args.profile_probs) - 1.0) > 1.0e-12:
        raise ValueError("profile probabilities must sum to one")
    if tuple(args.edge_cfls) != (args.primary_cfl,):
        raise ValueError(
            "V20 is a strict single-CFL experiment: edge-cfls must equal "
            "(primary-cfl,)"
        )
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar-fraction must lie in (0,1]")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validate_args(args)
    W.check_weno5_coefficients()
    device = W.torch_device(args.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = ReflectionSymmetricBadnessMLP(seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(optimizer, args.lr, args.lr_final, args.steps)
    history_csv = args.out_dir / "history.csv"
    eval_csv = args.out_dir / "eval.csv"
    start_step = 1
    skipped = 0

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("recipe") != RECIPE:
            raise ValueError("incompatible resume recipe")
        if state.get("training_signature") != signature(args):
            raise ValueError("resume training signature mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        gen.set_state(state["training_generator_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [item.cpu() for item in state["cuda_rng_state_all"]]
            )
        start_step = int(state["step"]) + 1
        skipped = int(state.get("skipped", 0))
        truncate_csv_after(history_csv, start_step - 1)
        truncate_csv_after(eval_csv, start_step - 1)
        print(f"RESUME completed={start_step - 1} next={start_step}", flush=True)
    else:
        save_model(
            args.out_dir / "checkpoints/model_step_000000.npz",
            model,
            0,
            args,
        )

    probe_gen = torch.Generator(device=device)
    probe_gen.manual_seed(args.validation_seed)
    probes = {
        name: make_profiles(
            args.eval_batch, args.eval_grid, device, probe_gen, kind=name
        )
        for name in PROFILE_NAMES
    }
    probe_velocities = {
        name: balanced_velocities(profile.batch, device)
        for name, profile in probes.items()
    }
    classical_errors = {
        name: max(
            final_state_error_signed(
                "classical",
                profile,
                args.eval_grid,
                args.eval_steps * args.primary_cfl,
                args.primary_cfl,
                probe_velocities[name],
            )[0],
            1.0e-300,
        )
        for name, profile in probes.items()
    }
    symmetry_features = torch.rand(
        (4096, 5),
        device=device,
        generator=probe_gen,
        dtype=torch.float64,
    )

    history_fields = [
        "step",
        "loss",
        "primary_loss",
        "edge_loss",
        "trajectory",
        "face_path",
        "exact_recon",
        "flat_d2",
        "tv_excess",
        "global_js_guard",
        "local_js_guard",
        "final_model_error",
        "final_classical_error",
        "final_vs_classical",
        "edge_global_js_guard",
        "edge_local_js_guard",
        "edge_final_vs_classical",
        "distance_cells",
        "horizon",
        "batch",
        "edge_steps",
        "edge_cfl",
        "lr",
        "grad",
        "reflection_defect",
        "skipped",
    ]
    eval_fields = ["step"]
    tag = f"cfl{int(round(10 * args.primary_cfl)):02d}"
    eval_fields.extend(f"{tag}_vs_cls_{name}" for name in PROFILE_NAMES)
    eval_fields.extend((f"{tag}_mean_vs_cls", "reflection_defect"))

    pairs = tuple(zip(args.distances, args.distance_batches))
    print(
        f"weno5_v20_start initialization=linear_d steps={args.steps} "
        f"grid={args.grid} primary_cfl={args.primary_cfl} "
        f"distance_batch_pairs={pairs} sampling=one_each_per_shuffled_cycle "
        f"profiles={PROFILE_NAMES} probs={tuple(args.profile_probs)} "
        "objective=checkpointed_autoregressive_exact_cell_averages_each_step "
        f"js_guard=global+local(window={args.local_window},"
        f"cvar={args.cvar_fraction}) reflection=enabled "
        "velocities=[+1,-1] test_profiles=excluded",
        flush=True,
    )

    run_end = (
        args.steps
        if args.stop_after_step is None
        else min(args.steps, args.stop_after_step)
    )
    for step in range(start_step, run_end + 1):
        model.train()
        pair_index = distance_index_for_step(
            step, len(pairs), args.seed
        )
        distance = float(args.distances[pair_index])
        batch = int(args.distance_batches[pair_index])
        horizon = steps_for_distance(distance, args.primary_cfl)
        edge_cfl = args.primary_cfl
        edge_steps = min(horizon, args.edge_max_steps)
        profiles = make_profiles(
            batch,
            args.grid,
            device,
            gen,
            probs=tuple(float(value) for value in args.profile_probs),
        )
        velocities = balanced_velocities(batch, device)

        primary_loss, primary = checkpointed_autoregressive_trajectory_loss(
            model,
            profiles,
            args.grid,
            horizon,
            args.primary_cfl,
            velocities,
            state_lambda=1.0,
            face_path_lambda=args.face_path_lambda,
            exact_recon_lambda=args.exact_recon_lambda,
            flat_d2_lambda=args.flat_d2_lambda,
            flat_tolerance=args.flat_tolerance,
            tv_lambda=args.tv_lambda,
            global_guard_lambda=args.global_guard_lambda,
            local_guard_lambda=args.local_guard_lambda,
            local_window=args.local_window,
            cvar_fraction=args.cvar_fraction,
            guard_tolerance=args.guard_tolerance,
        )
        edge_loss, edge = checkpointed_autoregressive_trajectory_loss(
            model,
            profiles,
            args.grid,
            edge_steps,
            edge_cfl,
            velocities,
            state_lambda=0.0,
            face_path_lambda=0.0,
            exact_recon_lambda=0.0,
            flat_d2_lambda=0.0,
            flat_tolerance=args.flat_tolerance,
            tv_lambda=0.0,
            global_guard_lambda=args.global_guard_lambda,
            local_guard_lambda=args.local_guard_lambda,
            local_window=args.local_window,
            cvar_fraction=args.cvar_fraction,
            guard_tolerance=args.guard_tolerance,
        )
        loss = primary_loss + args.edge_lambda * edge_loss

        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            skipped += 1
            scheduler.step()
            print(f"step={step:06d} NONFINITE skipped={skipped}", flush=True)
            continue
        loss.backward()
        grad = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        )
        if not np.isfinite(grad) or grad > args.grad_skip:
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"step={step:06d} grad={grad:.3e} skipped={skipped}",
                flush=True,
            )
        else:
            optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_interval == 0:
            defect = reflection_defect(model, symmetry_features)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "primary_loss": float(primary_loss.detach()),
                "edge_loss": float(edge_loss.detach()),
                **primary,
                "edge_global_js_guard": edge["global_js_guard"],
                "edge_local_js_guard": edge["local_js_guard"],
                "edge_final_vs_classical": edge["final_vs_classical"],
                "distance_cells": distance,
                "horizon": horizon,
                "batch": batch,
                "edge_steps": edge_steps,
                "edge_cfl": edge_cfl,
                "lr": scheduler.get_last_lr()[0],
                "grad": grad,
                "reflection_defect": defect,
                "skipped": skipped,
            }
            append_csv(history_csv, row, history_fields)
            print(
                f"step={step:06d} loss={row['loss']:.3e} "
                f"traj={row['trajectory']:.2e} "
                f"guards=[{row['global_js_guard']:.2e},"
                f"{row['local_js_guard']:.2e}] "
                f"final/js={row['final_vs_classical']:.3f} "
                f"distance={distance:g} horizon={horizon} batch={batch} "
                f"edge(n={edge_steps})/js={row['edge_final_vs_classical']:.3f} "
                f"lr={row['lr']:.2e} grad={grad:.2e}",
                flush=True,
            )

        checkpoint_path = (
            args.out_dir
            / "checkpoints"
            / f"model_step_{step:06d}.npz"
        )
        if (
            step == 1
            or step % args.checkpoint_interval == 0
            or step == run_end
        ):
            save_model(checkpoint_path, model, step, args)

        if args.eval_interval > 0 and (
            step == 1
            or step % args.eval_interval == 0
            or step == run_end
        ):
            model.eval()
            eval_row: dict[str, object] = {"step": step}
            with torch.no_grad():
                ratios = []
                for name, profile in probes.items():
                    error = final_state_error_signed(
                        model,
                        profile,
                        args.eval_grid,
                        args.eval_steps * args.primary_cfl,
                        args.primary_cfl,
                        probe_velocities[name],
                    )[0]
                    ratio = error / classical_errors[name]
                    eval_row[f"{tag}_vs_cls_{name}"] = ratio
                    ratios.append(ratio)
                eval_row[f"{tag}_mean_vs_cls"] = float(np.mean(ratios))
                eval_row["reflection_defect"] = reflection_defect(
                    model, symmetry_features
                )
            append_csv(eval_csv, eval_row, eval_fields)
            print(
                f"EVAL step={step:06d} {tag}_mean_vs_cls="
                f"{float(eval_row[f'{tag}_mean_vs_cls']):.4f} "
                f"sym={float(eval_row['reflection_defect']):.1e}",
                flush=True,
            )

        if step % args.state_interval == 0 or step == run_end:
            state_dir = args.out_dir / "training_state"
            save_state(
                state_dir / f"state_step_{step:06d}.pt",
                state_dir / "latest.pt",
                step,
                model,
                optimizer,
                scheduler,
                gen,
                skipped,
                args,
            )
            print(f"STATE step={step:06d}", flush=True)

    print(
        f"complete step={run_end} out={args.out_dir} skipped={skipped}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument(
        "--distances",
        type=float,
        nargs="+",
        default=(2, 8, 64, 128, 256, 512, 1024),
    )
    parser.add_argument(
        "--distance-batches",
        type=int,
        nargs="+",
        default=(32, 32, 16, 8, 4, 2, 2),
    )
    parser.add_argument(
        "--profile-probs",
        type=float,
        nargs=len(PROFILE_NAMES),
        default=(0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10),
    )
    parser.add_argument("--primary-cfl", type=float, default=0.5)
    parser.add_argument(
        "--edge-cfls", type=float, nargs="+", default=(0.5,)
    )
    parser.add_argument("--edge-max-steps", type=int, default=40)
    parser.add_argument("--edge-lambda", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--lr-final", type=float, default=2.0e-5)
    parser.add_argument("--face-path-lambda", type=float, default=0.04)
    parser.add_argument("--exact-recon-lambda", type=float, default=0.15)
    parser.add_argument("--flat-d2-lambda", type=float, default=0.05)
    parser.add_argument("--flat-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--tv-lambda", type=float, default=0.03)
    parser.add_argument("--global-guard-lambda", type=float, default=1.0)
    parser.add_argument("--local-guard-lambda", type=float, default=1.0)
    parser.add_argument("--guard-tolerance", type=float, default=0.0)
    parser.add_argument("--local-window", type=int, default=8)
    parser.add_argument("--cvar-fraction", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-skip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--validation-seed", type=int, default=104729)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--state-interval", type=int, default=2500)
    parser.add_argument("--eval-grid", type=int, default=96)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
