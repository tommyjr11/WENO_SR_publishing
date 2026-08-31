# Immutable source snapshot

`snapshot/` mirrors the original repository-relative module layout so that existing imports and command lines remain valid. Files are copied verbatim; publication wrappers and documentation live outside the snapshot.

The three final training packages are:

- `teacherfree_lab_weno5_v20_distance_balanced/` -- WENO5-SR FP64;
- `teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/` -- WENO5-SR with an FP32 MLP and FP64 solver state;
- `teacherfree_lab_weno7_rk4_distance_balanced_fast-2/` -- the exact WENO7-SR training directory associated with the selected run.

The package-compatible `teacherfree_lab_weno7_rk4_distance_balanced_fast/` namespace is also retained because the paper inference modules import that name. The selected checkpoint in both source runs is byte-identical; the canonical distributed file is `../../models/weno7_sr_fp64_step016750.npz`.

Historical WENO5 directories are present only where the final FP64 training entry point imports a loss, model, profile generator, or checkpoint helper from that module. They are dependency provenance, not additional selected experiments.

The remaining directories provide the exact two- and three-dimensional inference solvers, JS/Z baselines, HLLC and EVILIN flux paths, plotting utilities, and convergence/validation entry points used to create the archived results.

Runtime aliases of the three selected checkpoints are placed at the original paths expected by the unchanged source. Regenerated `runs/`, `raw/`, figures, logs, and caches are ignored by the release repository.
