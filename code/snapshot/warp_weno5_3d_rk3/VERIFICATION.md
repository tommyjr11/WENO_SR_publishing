# Verification record

## Reference

- C++ source: `ADER_TR_Project/ADER_TR_3D.cu`
- Active branch: `step <= history_size + stable_time`, `HEOC_TR == 4`
- Reference: `ADER_TR_Project/data/step_0140.bin`
- Shape: `88 x 88 x 224 x 5`
- Final time: `2e-5`
- Reference SHA-256:
  `b977c367c59417155719ce53a7e787c7c3fb8088bc713bd4194c2d7b8218e7c9`

The C++ reference was rebuilt in an isolated temporary directory before this
port was accepted; the rebuilt file had the same SHA-256 as the supplied
reference.

## Localization sequence

1. The initialized Warp and C++ files matched byte for byte. Their shared
   step-zero SHA-256 was
   `39bcd77c4cfa80761e4c4af530f07495f5417f7c1b0e3d637ebf0d96290c23ae`.
2. Before the final source-order correction, the first mismatch appeared in
   SSPRK3 stage 1. Direct flux dumps localized it before the RK update.
3. The `LR=2` normal reconstruction matched exactly, while `LR=1` differed.
4. The difference came from C++'s mutable stencil: `LR=2` performs an
   `R*(L*U)` round trip before `LR=1` reads the same local array.
5. After reproducing that sequence, all x/y/z face-point states, all three
   directional flux arrays, and the one-step output matched bitwise.
6. The complete 140-step run then matched the supplied reference byte for
   byte.

## Final result

```text
file_bitwise_identical = true
array_bitwise_identical = true
unequal_count = 0
max_ulp = 0
max_absolute_error = 0
normalized_l1 = 0
candidate_sha256 = b977c367c59417155719ce53a7e787c7c3fb8088bc713bd4194c2d7b8218e7c9
```

The accepted output and machine-readable reports are in
`runs/classical_regression_t00002_after_mlp/`.

## WENO5-SR deployment

- Checkpoint: V20 FP64 step 12,250
- Checkpoint SHA-256:
  `368759415a7dcf5567af76db06fd49c0b89260150f0fab7d9ada7f067ab3f74e`
- Architecture: `5 -> 10 -> 6 -> 6 -> 3`
- Deployment: `0.5*(M(x)+P*M(P*x))`
- Coverage: characteristic normal reconstruction and both transverse Gauss
  reconstructions, with all four reconstruction heads
- ENO cutoff: disabled

The four scalar heads agree with the trusted Torch deployment to a maximum
absolute error of `1.7763568394002505e-15`. A three-dimensional uniform-flow
step has maximum absolute drift `4.440892098500626e-16` and no NaNs.

The complete Ma=3 run reached `t=2e-5` in 142 steps:

```text
nan_count = 0
rho_min = 0.21399985723055326
rho_max = 5.0002172598094266
p_min = 101324.29710995934
p_max = 1047991.989218728
elapsed_seconds = 514.0135662649991
```

Its result and manifest are in `runs/ma3_v20_step012250_t00002/`.
