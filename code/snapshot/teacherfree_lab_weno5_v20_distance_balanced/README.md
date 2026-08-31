# WENO5 V20: Single-CFL Distance-Balanced Training

V20 keeps the V19 network, data families, reflection-symmetric forward pass,
exact FVM cell-average targets, TV term, and WENO5-JS non-regression guards.
The only learning-objective change is the rollout support.

- Training CFL: `0.5`
- Propagation distances, each visited exactly once per shuffled seven-step cycle:
  `2, 8, 64, 128, 256, 512, 1024` cells
- Full SSPRK3 steps:
  `4, 16, 128, 256, 512, 1024, 2048`
- Dynamic batches:
  `32, 32, 16, 8, 4, 2, 2`
- The autoregressive state is never detached.
- Activation checkpointing preserves the complete long-range gradient.
- GSTE and all paper-test profiles remain excluded from training.

Start or auto-resume:

```bash
bash teacherfree_lab_weno5_v20_distance_balanced/run_weno5_v20.sh
```

Monitor:

```bash
tail -f teacherfree_lab_weno5_v20_distance_balanced/runs/apost_weno5_v20_distance_balanced_cfl05_200k/nohup.out
```
