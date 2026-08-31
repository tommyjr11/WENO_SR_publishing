# Verification record

The independent checks were run on an NVIDIA GeForce RTX 5080 with Warp 1.13.0.

```text
binary round-trip:                 pass
WENO7 face reflection defect:     0.0000000000000000e+00
WENO7 Gauss reflection defect:    1.1102230246251565e-16
four-layer transmissive boundary: pass
four-layer periodic boundary:     pass (faces, edges, and corners)
uniform-flow scaled defect:       2.7755575615628914e-17
2-D reduction relative defect:    1.2176926134088717e-12
2-D reduction z spread:           0.0000000000000000e+00
initial binary SHA-256:            39bcd77c4cfa80761e4c4af530f07495f5417f7c1b0e3d637ebf0d96290c23ae
```

The initialization hash is identical to the trusted three-dimensional WENO5
port because both solvers use the same Ma=3 cell-average initializer and binary
schema.

## Formal Ma=3 run

```text
grid:                  224 x 88 x 88
CFL:                   0.25
target time:           2.0000000000000000e-05
steps:                 141
elapsed:               281.9279 s
NaN count:             0
minimum density:       2.1399933974421573e-01
minimum pressure:      1.0130427773470749e+05
output SHA-256:        d2954068302a4c11eea28cdb076a40698c286cb51b9e7f4839c12782150074ed
```

The last time step was clipped from `1.1963378367806386e-07` to
`2.6057620057755423e-08`, reaching the requested terminal time exactly in the
binary header and `dt_trace.csv`.

## Periodic isentropic vortex in three planes

The same two-dimensional vortex was embedded in the xy, yz, and xz planes on
`[-10,10]^3`. The inactive direction used 10 cells. Both active directions used
`N=25,50,100,200`, with CFL 0.4, terminal time 2, and 15x15 Gauss-Legendre
finite-volume averages for both initialization and the exact solution.

The following values are identical for all three planes at the shown precision:

| N | density L1 | order | density L2 | order |
|---:|---:|---:|---:|---:|
| 25  | 1.45537744e-03 | -      | 6.94575839e-03 | -      |
| 50  | 1.26838875e-04 | 3.5203 | 6.56355429e-04 | 3.4036 |
| 100 | 4.40764362e-06 | 4.8468 | 2.06143026e-05 | 4.9928 |
| 200 | 9.83641566e-08 | 5.4857 | 5.84936920e-07 | 5.1392 |

At N=200, the maximum pointwise density differences after mapping the active
coordinates into the same orientation were `1.23e-13` for xy versus yz and
`1.39e-13` for xy versus xz. The largest density-error discrepancy against the
trusted two-dimensional WENO7-EVILIN-RK4 table was `1.49e-14` in L1 and
`6.62e-14` in L2. Thus all three directional implementations reproduce the
trusted two-dimensional method to roundoff and exceed fourth-order convergence
over the two finest refinement intervals.
