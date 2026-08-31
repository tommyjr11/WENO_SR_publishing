# WENO7-SR / Shu-RK4 fast execution package

This is an isolated, code-only copy of the validated WENO7 distance-balanced
training package. It does not modify or import the original package, and it
contains no checkpoints or run outputs.

## What Is Unchanged

- True finite-volume GL15 cell averages on a periodic grid.
- WENO7 heads, optimal weights, features, and `6 -> 24 -> 16 -> 16 -> 4`
  float64 MLP.
- Linear-weight initialization from a zero final layer.
- Reflection-symmetric forward:
  `0.5 * (M(x) + P4 * M(P6*x))`.
- Both velocities, the exact Shu TVD RK4 coefficients, and the reverse
  operator used by the trusted two-dimensional WENO7 code.
- The seven independently sampled distances, batches, profile probabilities,
  every-step exact targets, all loss coefficients, JS guards, optimizer,
  scheduler, checkpoint cadence, and trusted HLLC Sod monitor.
- GSTE is excluded from training.

Default sampling remains:

| distance (cells) | 2 | 8 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| complete RK4 steps | 4 | 16 | 128 | 256 | 512 | 1024 | 2048 |
| batch | 32 | 32 | 16 | 8 | 4 | 2 | 2 |

## What Is Faster

The old path launched many small CUDA operations from a serial Python loop
and used three activation-checkpoint calls per RK step. It also repeated the
first 40 steps solely for the edge guard.

The fast path:

1. evaluates analytic cell averages and four point targets in time blocks;
2. computes the classical JS trajectory once;
3. reuses the first 40 states of the primary trajectory for the identical
   edge guard;
4. checkpoints 16 complete RK4 steps at a time;
5. traces one full RK4-plus-loss step per batch shape so CUDA launches are
   issued by TorchScript rather than the Python interpreter.

The training objective is not approximated. `equivalence_test.py` compares
the old and fast paths, including every target, every loss component, total
loss, and all MLP gradients.

A single autoregressive trajectory cannot be advanced by several CPU threads:
step `n+1` needs the CUDA result from step `n`. Multiple host threads would
add synchronization and race hazards rather than queueing useful independent
work. The correct optimization is to submit fewer, larger compiled units.

## Preflight

```bash
cd teacherfree_lab_weno7_rk4_distance_balanced_fast
python3 self_test.py --device cpu
python3 equivalence_test.py --device cpu
python3 equivalence_test.py --device cuda
```

Optional local timing:

```bash
python3 benchmark_fast.py \
  --device cuda --grid 96 --batch 2 --steps 16 \
  --chunk-steps 16 --target-chunk 128 --compile-mode jit
```

The first encounter with each of the five batch shapes emits
`JIT_TRACE_READY`; tracing takes seconds and is then reused.

On the local RTX 5080, with another training process sharing the GPU, the
`grid=96`, `batch=2`, 16-step benchmark changed from about `3.57 s` in the
old path to `0.179 s` in the JIT path. This is a workload measurement, not a
promise for every driver/GPU combination. If tracing is unsupported in a
particular environment, the launcher reports `ACCELERATOR_*_FALLBACK` and
recomputes that same sample with the equivalent chunked eager path.

## RTX 4090: Slurm

Copy the whole directory to `/home/rliuca/`, then submit from inside it:

```bash
cd /home/rliuca/teacherfree_lab_weno7_rk4_distance_balanced_fast

RUN=runs/apost_weno7_rk4_distance_balanced_fast_4090_200k

sbatch --export=ALL,OUT_DIR="$RUN",RESUME=fresh,COMPILE_MODE=jit,CHUNK_STEPS=16,TARGET_CHUNK=128,START_SOD_MONITOR=1 \
  submit_train.sbatch
```

Monitor:

```bash
squeue -u "$USER"
tail -f "$RUN/nohup.out"
tail -f "$RUN/sod_monitor.log"
```

If Slurm or a node stops the job, submit the same directory again with
`RESUME=auto`:

```bash
sbatch --export=ALL,OUT_DIR="$RUN",RESUME=auto,COMPILE_MODE=jit,CHUNK_STEPS=16,TARGET_CHUNK=128,START_SOD_MONITOR=1 \
  submit_train.sbatch
```

`RESUME=fresh` refuses to overwrite an existing history or training state.
`RESUME=auto` restores `training_state/latest.pt`, including model, Adam,
scheduler, profile RNG, and CUDA RNG.

## Tesla V100: nohup

The same TorchScript path is portable to V100 because it does not depend on
Triton or Tensor Cores. Keep `COMPILE_MODE=jit`.

```bash
cd /root/teacherfree_lab_weno7_rk4_distance_balanced_fast

RUN=runs/apost_weno7_rk4_distance_balanced_fast_v100_200k

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
  > v100_fast_launcher.log 2>&1 &

echo $! | tee v100_fast_launcher.pid
disown
```

Monitor:

```bash
tail -f "$RUN/nohup.out"
tail -f "$RUN/sod_monitor.log"
```

Resume after interruption:

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
  >> v100_fast_launcher.log 2>&1 &

echo $! | tee v100_fast_launcher.pid
disown
```

## Validation

GSTE checkpoint sweep:

```bash
MODULE_NAME="" bash validate_gste.sh \
  "$RUN" "2500 3500 4000 6250 10000"
```

Trusted two-dimensional characteristic/HLLC Sod:

```bash
MODULE_NAME="" bash validate_sod.sh \
  "$RUN" "2500 3500 4000 6250 10000"
```

The online monitor already runs Sod at every 250-step checkpoint using
`100x10`, CFL `0.4`, `t=0.25`, characteristic reconstruction, HLLC, no ENO
cutoff, and the reflection-symmetric two-pass MLP.

Fast training states use a separate recipe identifier. Resume within this
fast package; do not point it at an optimizer state produced by the old
execution package.
