#!/usr/bin/env python3
"""Train reflection-equivariant WENO7 with exact-FVM Shu-RK4 rollouts."""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

import fast_losses
import fvm_profiles as profiles_module
import losses
import rk4_advection as A
import weno7_core as W

torch.set_default_dtype(torch.float64)

RECIPE = "weno7_rk4_distance_balanced_fast_execution_v1"


def append_csv(path: Path, row: dict[str, object], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def truncate_csv_after(path: Path, completed_step: int) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = [
            row
            for row in reader
            if int(float(row.get("step", "-1"))) <= completed_step
        ]
    if not fieldnames:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    start_lr: float,
    final_lr: float,
    steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
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
            f"distance={distance:g} must contain full Shu-RK4 steps "
            f"at CFL={cfl:g}"
        )
    return rounded


def distance_index_for_step(step: int, count: int, seed: int) -> int:
    if step < 1 or count < 1:
        raise ValueError("step and count must be positive")
    cycle, offset = divmod(step - 1, count)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003 * cycle)
    order = torch.randperm(count, generator=generator).tolist()
    return int(order[offset])


def resolve_compile_mode(
    requested: str, device: torch.device
) -> str:
    """Select a portable default while allowing an explicit override."""
    if requested != "auto":
        return requested
    if device.type != "cuda":
        return "none"
    # TorchScript removes Python launch overhead without depending on Triton,
    # so the same path is portable across V100, RTX 4090, and newer GPUs.
    return "jit"


def signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "steps": args.steps,
        "grid": args.grid,
        "hidden": tuple(args.hidden),
        "distances": tuple(args.distances),
        "distance_batches": tuple(args.distance_batches),
        "profile_probs": tuple(args.profile_probs),
        "primary_cfl": args.primary_cfl,
        "edge_max_steps": args.edge_max_steps,
        "edge_lambda": args.edge_lambda,
        "face_path_lambda": args.face_path_lambda,
        "exact_recon_lambda": args.exact_recon_lambda,
        "flat_d2_lambda": args.flat_d2_lambda,
        "flat_tolerance": args.flat_tolerance,
        "global_guard_lambda": args.global_guard_lambda,
        "local_guard_lambda": args.local_guard_lambda,
        "guard_tolerance": args.guard_tolerance,
        "local_window": args.local_window,
        "cvar_fraction": args.cvar_fraction,
        "tv_lambda": args.tv_lambda,
        "lr": args.lr,
        "lr_final": args.lr_final,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "reflection": True,
        "velocities": (-1.0, 1.0),
        "time_integrator": "shu_tvd_rk4_with_reverse_operator",
        "loss_execution": "precomputed_targets_chunked_checkpoint_edge_reuse_v1",
    }


def save_model(
    path: Path,
    model: W.ReflectionSymmetricBadnessMLP,
    step: int,
    args: argparse.Namespace,
) -> None:
    W.save_checkpoint(
        path,
        model,
        {
            "raw_step": step,
            "recipe": RECIPE,
            "from_scratch_linear_d_initialization": True,
            "true_fvm_cell_averages": True,
            "cell_average_quadrature": 15,
            "autoregressive_model_state_in_graph": True,
            "rollout_activation_checkpointing": True,
            "activation_checkpoint_chunk_steps": args.chunk_steps,
            "exact_target_precompute_chunk": args.target_chunk,
            "primary_edge_prefix_reused": True,
            "compile_mode_requested": args.compile_mode,
            "rollout_state_detached": False,
            "exact_state_target_at_every_complete_rk_step": True,
            "internal_rk_stages_supervised": False,
            "time_integrator": "Shu fourth-order TVD RK with L and L_tilde",
            "primary_cfl": args.primary_cfl,
            "propagation_distances_cells": list(args.distances),
            "distance_sampling": "one_each_per_shuffled_cycle",
            "distance_batches": list(args.distance_batches),
            "classical_baseline": (
                "WENO7-JS with identical Shu-RK4, CFL, and trajectory"
            ),
            "classical_global_nonregression": True,
            "classical_local_nonregression_window": args.local_window,
            "classical_guard_aggregation": (
                "0.25_family_mean_plus_0.75_cvar25"
            ),
            "paper_gste_profile_used_for_training": False,
            "training_profiles": list(profiles_module.PROFILE_NAMES),
            "advection_velocities": [-1.0, 1.0],
            "i_plus_linear_weights": W.OPTIMAL_D_NP[1].tolist(),
            "i_minus_linear_weights": W.OPTIMAL_D_NP[2].tolist(),
        },
    )


def save_state(
    path: Path,
    latest: Path,
    step: int,
    model: W.ReflectionSymmetricBadnessMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
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
        "training_generator_state": generator.get_state(),
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
            raise ValueError("all dynamic batches must be positive and even")
    for distance in args.distances:
        steps_for_distance(distance, args.primary_cfl)
    if len(args.profile_probs) != len(profiles_module.PROFILE_NAMES):
        raise ValueError("one probability is required per profile family")
    if abs(sum(args.profile_probs) - 1.0) > 1.0e-12:
        raise ValueError("profile probabilities must sum to one")
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar-fraction must lie in (0,1]")
    if args.chunk_steps < 1 or args.target_chunk < 1:
        raise ValueError("chunk-steps and target-chunk must be positive")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(args.resume)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validate_args(args)
    W.check_weno7_coefficients()
    rk_order = A.check_shu_rk4_order()
    device = W.torch_device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = W.ReflectionSymmetricBadnessMLP(
        hidden=tuple(args.hidden), seed=args.seed
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(
        optimizer, args.lr, args.lr_final, args.steps
    )
    history_csv = args.out_dir / "history.csv"
    eval_csv = args.out_dir / "eval.csv"
    start_step = 1
    skipped = 0
    initialization_label = "linear_d"

    if args.resume is not None:
        state = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if state.get("recipe") != RECIPE:
            raise ValueError("incompatible resume recipe")
        if state.get("training_signature") != signature(args):
            raise ValueError("resume training signature mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        generator.set_state(state["training_generator_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [item.cpu() for item in state["cuda_rng_state_all"]]
            )
        start_step = int(state["step"]) + 1
        skipped = int(state.get("skipped", 0))
        initialization_label = f"resume_step_{start_step - 1:06d}"
        truncate_csv_after(history_csv, start_step - 1)
        truncate_csv_after(eval_csv, start_step - 1)
        print(
            f"RESUME completed={start_step - 1} next={start_step}",
            flush=True,
        )
    else:
        save_model(
            args.out_dir / "checkpoints/model_step_000000.npz",
            model,
            0,
            args,
        )

    dt = args.primary_cfl / float(args.grid)
    step_module = fast_losses.TrajectoryStep(
        model,
        dt=dt,
        dxinv=float(args.grid),
        flat_tolerance=args.flat_tolerance,
        local_window=args.local_window,
        cvar_fraction=args.cvar_fraction,
        guard_tolerance=args.guard_tolerance,
    )
    classical_module = fast_losses.ClassicalStep(
        dt=dt, dxinv=float(args.grid)
    )
    effective_compile_mode = resolve_compile_mode(
        args.compile_mode, device
    )
    compiled_active = effective_compile_mode != "none"
    if effective_compile_mode == "jit":
        step_operator = fast_losses.ShapeTracedCallable(
            step_module, "trajectory"
        )
        classical_step = fast_losses.ShapeTracedCallable(
            classical_module, "classical"
        )
    elif compiled_active:
        try:
            torch._dynamo.config.cache_size_limit = max(
                int(torch._dynamo.config.cache_size_limit), 64
            )
        except (AttributeError, TypeError):
            pass
        step_operator = torch.compile(
            step_module,
            mode=effective_compile_mode,
            fullgraph=args.compile_fullgraph,
            dynamic=False,
        )
        classical_step = torch.compile(
            classical_module,
            mode=effective_compile_mode,
            fullgraph=args.compile_fullgraph,
            dynamic=False,
        )
    else:
        step_operator = step_module
        classical_step = classical_module

    probe_generator = torch.Generator(device=device)
    probe_generator.manual_seed(args.validation_seed)
    probes = {
        name: profiles_module.make_profiles(
            args.eval_batch,
            args.eval_grid,
            device,
            probe_generator,
            kind=name,
        )
        for name in profiles_module.PROFILE_NAMES
    }
    probe_velocities = {
        name: A.balanced_velocities(profile.batch, device)
        for name, profile in probes.items()
    }
    eval_distance = args.eval_steps * args.primary_cfl
    classical_errors = {
        name: max(
            losses.final_state_error_signed(
                "classical",
                profile,
                args.eval_grid,
                eval_distance,
                args.primary_cfl,
                probe_velocities[name],
            )[0],
            1.0e-300,
        )
        for name, profile in probes.items()
    }
    symmetry_features = torch.rand(
        (4096, W.INPUTS),
        device=device,
        generator=probe_generator,
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
        "lr",
        "grad",
        "reflection_defect",
        "skipped",
    ]
    tag = f"cfl{int(round(10.0 * args.primary_cfl)):02d}"
    eval_fields = ["step"]
    eval_fields.extend(
        f"{tag}_vs_cls_{name}" for name in profiles_module.PROFILE_NAMES
    )
    eval_fields.extend((f"{tag}_mean_vs_cls", "reflection_defect"))

    pairs = tuple(zip(args.distances, args.distance_batches))
    print(
        f"weno7_rk4_start initialization={initialization_label} "
        f"rk_order_check={rk_order:.3f} "
        f"steps={args.steps} grid={args.grid} hidden={tuple(args.hidden)} "
        f"primary_cfl={args.primary_cfl} distance_batch_pairs={pairs} "
        "sampling=one_each_per_shuffled_cycle "
        f"profiles={profiles_module.PROFILE_NAMES} "
        f"probs={tuple(args.profile_probs)} "
        "integrator=shu_tvd_rk4_with_reverse_operator "
        "objective=checkpointed_autoregressive_exact_cell_averages_each_step "
        f"execution=fast(chunk={args.chunk_steps},"
        f"target_chunk={args.target_chunk},"
        f"compile={effective_compile_mode}) "
        f"js_guard=global+local(window={args.local_window},"
        f"cvar={args.cvar_fraction}) reflection=enabled "
        "velocities=[+1,-1] GSTE=excluded",
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
        edge_steps = min(horizon, args.edge_max_steps)
        profile = profiles_module.make_profiles(
            batch,
            args.grid,
            device,
            generator,
            probs=tuple(float(value) for value in args.profile_probs),
        )
        velocities = A.balanced_velocities(batch, device)

        loss_kwargs = {
            "state_lambda": 1.0,
            "face_path_lambda": args.face_path_lambda,
            "exact_recon_lambda": args.exact_recon_lambda,
            "flat_d2_lambda": args.flat_d2_lambda,
            "tv_lambda": args.tv_lambda,
            "global_guard_lambda": args.global_guard_lambda,
            "local_guard_lambda": args.local_guard_lambda,
            "edge_steps": edge_steps,
            "edge_lambda": args.edge_lambda,
            "chunk_steps": args.chunk_steps,
            "target_chunk": args.target_chunk,
        }
        try:
            loss, metrics = fast_losses.fast_autoregressive_trajectory_loss(
                step_operator,
                classical_step,
                profile,
                args.grid,
                horizon,
                args.primary_cfl,
                velocities,
                **loss_kwargs,
            )
        except Exception as error:
            if not compiled_active:
                raise
            print(
                "ACCELERATOR_FALLBACK "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            compiled_active = False
            effective_compile_mode = "none-after-fallback"
            step_operator = step_module
            classical_step = classical_module
            try:
                torch._dynamo.reset()
            except AttributeError:
                pass
            loss, metrics = fast_losses.fast_autoregressive_trajectory_loss(
                step_operator,
                classical_step,
                profile,
                args.grid,
                horizon,
                args.primary_cfl,
                velocities,
                **loss_kwargs,
            )

        optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            skipped += 1
            scheduler.step()
            print(
                f"step={step:06d} NONFINITE skipped={skipped}", flush=True
            )
            continue
        try:
            loss.backward()
        except Exception as error:
            if not compiled_active:
                raise
            print(
                "ACCELERATOR_BACKWARD_FALLBACK "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            compiled_active = False
            effective_compile_mode = "none-after-fallback"
            step_operator = step_module
            classical_step = classical_module
            try:
                torch._dynamo.reset()
            except AttributeError:
                pass
            loss, metrics = fast_losses.fast_autoregressive_trajectory_loss(
                step_operator,
                classical_step,
                profile,
                args.grid,
                horizon,
                args.primary_cfl,
                velocities,
                **loss_kwargs,
            )
            if not bool(torch.isfinite(loss)):
                skipped += 1
                scheduler.step()
                print(
                    f"step={step:06d} NONFINITE_AFTER_FALLBACK "
                    f"skipped={skipped}",
                    flush=True,
                )
                continue
            loss.backward()
        grad = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
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
            defect = W.reflection_defect(model, symmetry_features)
            logged_metrics = {
                name: float(value)
                for name, value in metrics.items()
            }
            row = {
                "step": step,
                "loss": float(loss.detach()),
                **logged_metrics,
                "distance_cells": distance,
                "horizon": horizon,
                "batch": batch,
                "edge_steps": edge_steps,
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
                f"lr={row['lr']:.2e} grad={grad:.2e} sym={defect:.1e}",
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
                for name, probe in probes.items():
                    error = losses.final_state_error_signed(
                        model,
                        probe,
                        args.eval_grid,
                        eval_distance,
                        args.primary_cfl,
                        probe_velocities[name],
                    )[0]
                    ratio = error / classical_errors[name]
                    eval_row[f"{tag}_vs_cls_{name}"] = ratio
                    ratios.append(ratio)
                eval_row[f"{tag}_mean_vs_cls"] = float(np.mean(ratios))
                eval_row["reflection_defect"] = W.reflection_defect(
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
                generator,
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
        "--hidden", type=int, nargs=3, default=W.DEFAULT_HIDDEN
    )
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
        nargs=len(profiles_module.PROFILE_NAMES),
        default=profiles_module.PROFILE_DEFAULT_PROBS,
    )
    parser.add_argument("--primary-cfl", type=float, default=0.5)
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
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument("--target-chunk", type=int, default=128)
    parser.add_argument(
        "--compile-mode",
        choices=("auto", "none", "jit", "default"),
        default="auto",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
