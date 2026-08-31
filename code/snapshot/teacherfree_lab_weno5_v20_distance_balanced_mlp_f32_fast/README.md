# WENO5 V20 mixed-FP32 fast execution package

This directory is an isolated, code-only fast version of the validated WENO5
V20 distance-balanced training setup. It does not modify or stop the original
FP64 or mixed-FP32 runs.

## Numerical contract

The training problem is unchanged:

- true finite-volume GL15 cell averages on a periodic grid;
- WENO5 candidates, optimal weights, features, and
  `5 -> 10 -> 6 -> 6 -> 3` MLP;
- linear-weight initialization from a zero final layer;
- reflection-symmetric two-pass forward
  `0.5 * (M(x) + P * M(P*x))`;
- both advection directions and the same SSPRK3 update;
- the same seven propagation distances, dynamic batches, profile
  probabilities, exact target at every complete SSPRK3 step, loss
  coefficients, JS guards, Adam optimizer, scheduler, and save cadence;
- no GSTE profile is used in training.

The precision boundary is also unchanged from the controlled mixed-FP32 run:

- MLP parameters and hidden activations: `float32`;
- FVM state, analytic targets, WENO normalization, SSPRK3, JS reference,
  losses, and reflection average: `float64`;
- TF32 and autocast: disabled.

Default distance sampling:

| distance (cells) | 2 | 8 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| complete SSPRK3 steps | 4 | 16 | 128 | 256 | 512 | 1024 | 2048 |
| batch | 32 | 32 | 16 | 8 | 4 | 2 | 2 |

## Execution-only changes

The fast path:

1. evaluates exact cell-average and point targets in time blocks;
2. advances the WENO5-JS reference once;
3. reuses the first 40 primary states for the identical edge guard;
4. checkpoints 16 complete SSPRK3 steps per block;
5. traces one SSPRK3-plus-loss operator for each dynamic batch shape.

These changes reduce Python dispatch and repeated work. They do not reduce
the number of supervised distances, alter a loss, detach a state, enable
lower-precision solver arithmetic, or approximate a target.

`equivalence_test.py` compares the original and fast implementations,
including exact targets, every loss component, total loss, and all MLP
gradients.

## Preflight

```bash
cd teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast
python3 self_test.py --device cpu
python3 equivalence_test.py --device cpu --compile-mode none
python3 equivalence_test.py --device cuda --compile-mode jit
```

The first encounter with each of the five batch shapes prints
`JIT_TRACE_READY`. Tracing is then reused.

On the local RTX 5080 while another training job shared the GPU, a
`grid=48`, `batch=2`, 16-step check changed from about `1.48 s` in the
original path to `0.134 s` in the JIT path (`11.1x`). This is a small
execution benchmark, not a performance promise for the A800.

## A800 nohup training

Copy this entire directory to the A800 machine, enter it, and run:

```bash
RUN=runs/apost_weno5_v20_mlp_f32_fast_a800_200k

nohup env \
  MODULE_NAME="" \
  GPU_ID=0 \
  OUT_DIR="$RUN" \
  RESUME=fresh \
  COMPILE_MODE=jit \
  CHUNK_STEPS=16 \
  TARGET_CHUNK=128 \
  START_SOD_MONITOR=1 \
  WAIT_FOR_TRAIN=1 \
  bash run_train.sh \
  > a800_fast_launcher.log 2>&1 &

echo $! | tee a800_fast_launcher.pid
disown
```

The launcher uses the current Python environment by default. Set
`CONDA_ENV=name` or `MODULE_NAME=name` only when that machine requires it.

Monitor without attaching to the training process:

```bash
tail -f "$RUN/nohup.out"
tail -f "$RUN/sod_monitor.log"
```

The online validation is the trusted two-dimensional Warp WENO5-RK3 path:
`100x8`, CFL `0.4`, `t=0.25`, characteristic reconstruction, Evilin, no ENO
cutoff, FP64 state, FP32 MLP, and reflection-symmetric two-pass inference.

To resume after an interruption, use the same command with `RESUME=auto` and
append the launcher log:

```bash
nohup env \
  MODULE_NAME="" \
  GPU_ID=0 \
  OUT_DIR="$RUN" \
  RESUME=auto \
  COMPILE_MODE=jit \
  CHUNK_STEPS=16 \
  TARGET_CHUNK=128 \
  START_SOD_MONITOR=1 \
  WAIT_FOR_TRAIN=1 \
  bash run_train.sh \
  >> a800_fast_launcher.log 2>&1 &

echo $! | tee a800_fast_launcher.pid
disown
```

The resumable state includes the MLP, Adam, scheduler, profile RNG, CPU RNG,
and CUDA RNG. `RESUME=fresh` refuses to overwrite an existing run.

## Slurm server

On the `granularmech` partition, submit one GPU from inside the package:

```bash
cd /home/rliuca/teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast

RUN=runs/apost_weno5_v20_mlp_f32_fast_200k

sbatch --export=ALL,OUT_DIR="$RUN",RESUME=fresh,COMPILE_MODE=jit,CHUNK_STEPS=16,TARGET_CHUNK=128,START_SOD_MONITOR=1 \
  submit_train.sbatch
```

Monitor:

```bash
squeue -u "$USER"
tail -f "$RUN/nohup.out"
tail -f "$RUN/sod_monitor.log"
```

After an interruption, submit the same run with `RESUME=auto`:

```bash
sbatch --export=ALL,OUT_DIR="$RUN",RESUME=auto,COMPILE_MODE=jit,CHUNK_STEPS=16,TARGET_CHUNK=128,START_SOD_MONITOR=1 \
  submit_train.sbatch
```

## Checkpoint validation

GSTE, excluded from training:

```bash
MODULE_NAME="" bash validate_gste.sh \
  "$RUN" "2500 3500 4000 10000 12250"
```

Selected trusted Warp Sod checks:

```bash
MODULE_NAME="" bash validate_sod.sh \
  "$RUN" "2500 3500 4000 10000 12250"
```

The fast package has a distinct recipe identifier. Resume only a state
created by this package; old checkpoints remain usable for inference, but an
old optimizer state is intentionally rejected.
