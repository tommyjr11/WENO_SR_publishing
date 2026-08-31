# WENO7 External Normal MLP Experiment

This folder is an isolated copy used to test a faster compile path without
touching the main WENO7 files.

The split is:

- `warp_weno7_ader4_helpers_classical_only.py`: classical WENO7/ADER4 Warp
  helper with the embedded MLP kernels removed.
- `weno7_ader4_warp_classical_only.py`: classical formal runner utilities and
  model loading.
- `warp_weno7_mlp_beta_provider.py`: small external MLP beta precompute kernels.
- `run_weno7_quadrant_external_normal_clean.py`: quadrant runner using MLP beta
  only for normal-direction characteristic face values. ADER4 cross/Gauss
  reconstruction stays classical.

This is equivalent to the old `mlp_derivative_mode=normal` path. It is not the
full `all` MLP path.
