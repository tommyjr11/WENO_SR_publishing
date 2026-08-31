# WENO7-Z formula audit

## Source check

Castro, Costa, and Don (JCP 230, 2011) give the generic even-`r` indicator in
Theorem 2, Eq. (28):

```text
tau7_generic = abs(beta0 - beta1 - beta2 + beta3) = O(h^6).
```

This is a valid seventh-order WENO-Z indicator.  It is not, however, the
indicator used in the subsequent numerical experiments.  Remark 10 states
that the rest of the paper uses the optimal indicator from Table 2.  For
`r=4`, the coefficient vector is `(-1,-3,3,1)`:

```text
tau7_opt = abs(-beta0 - 3*beta1 + 3*beta2 + beta3) = O(h^7).
```

Section 5 also states that, except for the weight-only experiment in Section
5.1, time-dependent tests use `epsilon=h^(r-1)`.  Thus WENO5 uses `h^2` and
WENO7 uses `h^3`.  The local WENO7 beta implementation carries a common
factor 240, so its code uses `240*h^3`, which produces exactly the same
ratios.

The paper's Double-Mach figure contains WENO-Z orders 5, 9, and 11; it does
not show order 7.  Its solver is characteristic finite-difference WENO with
SSPRK3 and CFL 0.45.  The local benchmark instead uses finite-volume
reconstruction, HLLC, downwind Shu-RK4, and CFL 0.4.  The comparison therefore
tests the WENO-Z weights in a different trusted solver path; it is not a
bitwise reproduction of the paper.

## Controlled Double-Mach audit

| WENO7 global indicator | epsilon | Grid and horizon | Outcome |
|---|---|---|---|
| generic Eq. (28) | `1e-40` | `1200x300`, first step | non-finite, values reached `O(1e54)` |
| generic Eq. (28) | `h^3` | `120x30`, `t=0.02` | failed at step 12 with 612 non-finite entries |
| optimal Table 2 | `h^3` | `120x30`, `t=0.02` | complete, 20 steps, no non-finite values |
| optimal Table 2 | `h^3` | `1200x300`, `t=0.2` | complete, 3177 steps, no non-finite values |

The generic indicator is mathematically admissible; "generic" means it is a
closed formula valid for every order, not that it has the same cancellation
order or nonlinear stability margin as the optimal indicator.  In this
finite-volume/HLLC/RK4 Double-Mach path, the difference is decisive.

The failed pre-audit outputs were removed before the full corrected rerun. The
corrected formal result is `raw/double_mach/N1200x300/weno7_z_p3.npz`.
