# WENO5 V20 Distance-Balanced MLP-FP32

This is a controlled precision copy of
`teacherfree_lab_weno5_v20_distance_balanced/`.

The training recipe is unchanged:

- CFL `0.5`;
- propagation distances `2, 8, 64, 128, 256, 512, 1024` cells, each visited
  once per shuffled seven-step cycle;
- dynamic batches `32, 32, 16, 8, 4, 2, 2`;
- exact FVM cell-average targets at every complete SSPRK3 step;
- the same profile families and probabilities;
- the same face, reconstruction, flat-region, TV, global JS and local JS
  objectives;
- balanced velocities `a=+1/-1`;
- `0.5*(M(x)+P*M(P*x))` reflection-symmetric inference;
- linear-d initialization and the same optimizer/LR schedule.

Only the MLP precision changes:

- FVM states, exact targets, feature construction, reflection averaging, WENO
  normalization, reconstruction, SSPRK3 and losses remain `float64`;
- MLP parameters, hidden activations, logits and softmax are `float32`;
- the three softmax ratios are converted back to `float64` immediately after
  each MLP call;
- checkpoints store all eight MLP arrays as `float32`.

The online Sod monitor uses the same trusted 2D Warp WENO5/RK3,
characteristic/EVILIN path as V20, with only its MLP path converted to FP32.

Start or auto-resume:

```bash
bash teacherfree_lab_weno5_v20_distance_balanced_mlp_f32/run_weno5_v20_mlp_f32.sh
```

Monitor:

```bash
tail -f teacherfree_lab_weno5_v20_distance_balanced_mlp_f32/runs/apost_weno5_v20_distance_balanced_mlp_f32_cfl05_200k/nohup.out
```

GSTE validation:

```bash
python3 -u \
  teacherfree_lab_weno5_v20_distance_balanced_mlp_f32/evaluate_gste_cfl_sweep.py \
  --model path/to/model_step_XXXXXX.npz \
  --out-dir path/to/gste_output
```
