import sympy as sp
from math import log10
from functools import lru_cache


def _check_order(order: int) -> int:
    if order < 3 or order % 2 == 0:
        raise ValueError("WENO order must be an odd integer >= 3, e.g. 3, 5, 7, 9.")
    return (order + 1) // 2


def _lcm_denominators(expr, symbols):
    poly = sp.Poly(sp.expand(expr), *symbols)
    den = 1
    for c in poly.coeffs():
        den = sp.ilcm(den, sp.Rational(c).q)
    return den


@lru_cache(None)
def _beta_matrix_in_poly_coeffs(r: int):
    """
    JS beta matrix for a degree-(r-1) polynomial

        p(x) = a0 + a1 x + ... + a_{r-1} x^{r-1}

    on the target cell [-1/2, 1/2], with dx = 1:

        beta = sum_{ell=1}^{r-1} int_{-1/2}^{1/2} (d^ell p / dx^ell)^2 dx.

    Return G such that

        beta = [a1,...,a_{r-1}]^T G [a1,...,a_{r-1}].
    """
    x = sp.Symbol("x")
    a = sp.symbols(f"a0:{r}")
    p = sum(a[m] * x**m for m in range(r))

    beta = 0
    for ell in range(1, r):
        beta += sp.integrate(
            sp.diff(p, x, ell)**2,
            (x, -sp.Rational(1, 2), sp.Rational(1, 2))
        )
    beta = sp.expand(beta)

    av = a[1:]
    G = sp.zeros(r - 1, r - 1)

    for i in range(r - 1):
        for j in range(r - 1):
            if i == j:
                G[i, j] = beta.coeff(av[i] * av[j])
            else:
                G[i, j] = beta.coeff(av[i] * av[j]) / 2

    return G


@lru_cache(None)
def get_decomposition(order: int):
    """
    Build exact symbolic decomposition for WENO-(2r-1) JS smoothness indicators.

    Returns data so that for each substencil k,

        beta_k = sum_m coeff_{k,m} * D_{k,m}**2.

    Here D_{k,m} is an integer linear form of q0,...,q_{2r-2}.
    """
    r = _check_order(order)
    n = 2 * r - 1

    q = sp.symbols(f"q0:{n}")
    x = sp.Symbol("x")
    a = sp.symbols(f"a0:{r}")

    # Orthogonalization of JS beta in polynomial coefficient space.
    G = _beta_matrix_in_poly_coeffs(r)
    L, Dmat = G.LDLdecomposition()

    # If beta = a^T G a = y^T D y, then y = L^T a.
    y = L.T * sp.Matrix(a[1:])

    centers = [j - (r - 1) for j in range(n)]
    items = []

    for k in range(r):
        # Polynomial p_k whose cell averages over the substencil cells equal q[k:k+r].
        M = sp.zeros(r, r)
        b = sp.Matrix(q[k:k + r])

        for row, c in enumerate(centers[k:k + r]):
            left = sp.Rational(c) - sp.Rational(1, 2)
            right = sp.Rational(c) + sp.Rational(1, 2)

            for m in range(r):
                M[row, m] = sp.integrate(x**m, (x, left, right))

        avec = M.LUsolve(b)
        sol = {a[m]: sp.simplify(avec[m]) for m in range(r)}

        modes = []

        for m in range(r - 1):
            exact_form = sp.expand(y[m].subs(sol))

            den = _lcm_denominators(exact_form, q)
            D_int = sp.expand(den * exact_form)
            coeff = sp.simplify(Dmat[m, m] / (den**2))

            modes.append({
                "derivative_order": m + 1,
                "coeff": coeff,
                "D": D_int,
                "exact_form": exact_form,
            })

        # Direct JS beta check.
        p = sum(avec[m] * x**m for m in range(r))
        beta_direct = 0

        for ell in range(1, r):
            beta_direct += sp.integrate(
                sp.diff(p, x, ell)**2,
                (x, -sp.Rational(1, 2), sp.Rational(1, 2))
            )

        beta_direct = sp.expand(beta_direct)
        beta_decomp = sp.expand(sum(mode["coeff"] * mode["D"]**2 for mode in modes))

        if sp.simplify(beta_direct - beta_decomp) != 0:
            raise RuntimeError(f"Decomposition check failed for substencil {k}.")

        items.append(modes)

    return {
        "order": order,
        "r": r,
        "q": q,
        "items": items,
    }


def cpp_coeff(c) -> str:
    c = sp.Rational(c)

    if c.q == 1:
        return f"{c.p}.0"

    return f"({c.p}.0/{c.q}.0)"


def cpp_expr(expr, order: int, index_fmt: str = "q[{j}]") -> str:
    """
    Convert sympy expression to C/C++ expression.

    Examples
    --------
    index_fmt = "q[{j}]"
    index_fmt = "U_con[{j}][i]"
    """
    r = _check_order(order)
    n = 2 * r - 1

    s = sp.ccode(sp.expand(expr))

    # Replace q10 before q1.
    for j in reversed(range(n)):
        s = s.replace(f"q{j}", index_fmt.format(j=j))

    return s


def nested_fmax(names):
    expr = names[0]
    for name in names[1:]:
        expr = f"fmax({expr}, {name})"
    return expr


def print_decomposition(order: int, index_fmt: str = "q[{j}]"):
    """
    Print C/CUDA-style code for

        D_k_m,
        delta_k = sum_m coeff_m * fabs(D_k_m),
        beta_k  = sum_m coeff_m * D_k_m^2.

    Use index_fmt="U_con[{j}][i]" if you want your code style.
    """
    data = get_decomposition(order)
    r = data["r"]

    print(f"// WENO{order}: substencil size r = {r}; full stencil length = {2*r - 1}")
    print(f"// beta_k  = sum_m c_m * D_k_m^2")
    print(f"// delta_k = sum_m c_m * fabs(D_k_m)")
    print()

    for k, modes in enumerate(data["items"]):
        print(f"// substencil S{k}")

        for mode in modes:
            m = mode["derivative_order"]
            D_code = cpp_expr(mode["D"], order, index_fmt)
            print(f"double D{k}_{m} = {D_code};")

        terms_delta = [
            f"{cpp_coeff(mode['coeff'])} * fabs(D{k}_{mode['derivative_order']})"
            for mode in modes
        ]

        print(f"double delta{k} = " + "\n              + ".join(terms_delta) + ";")

        terms_beta = [
            f"{cpp_coeff(mode['coeff'])} * D{k}_{mode['derivative_order']} * D{k}_{mode['derivative_order']}"
            for mode in modes
        ]

        print(f"double beta{k}  = " + "\n              + ".join(terms_beta) + ";")
        print()

    max_expr = nested_fmax([f"delta{k}" for k in range(r)])
    print(f"double max_delta = {max_expr};")


def integer_scaled_beta(order: int, k: int):
    """
    Return scale and expanded expression such that

        scale * beta_k

    has integer coefficients.

    For WENO7, scale should be 240, so this matches your old 'bate' form.
    """
    data = get_decomposition(order)
    q = data["q"]

    beta = sp.expand(sum(
        mode["coeff"] * mode["D"]**2
        for mode in data["items"][k]
    ))

    scale = _lcm_denominators(beta, q)
    return scale, sp.expand(scale * beta)


def print_integer_scaled_beta(order: int, index_fmt: str = "q[{j}]"):
    """
    Print expanded integer-scaled beta expressions.
    Useful for checking against existing hand-written bate0, bate1, ...
    """
    r = _check_order(order)

    for k in range(r):
        scale, expr = integer_scaled_beta(order, k)
        print(f"// {scale} * beta{k}")
        print(f"double bate{k} = {cpp_expr(expr, order, index_fmt)};")
        print()


def evaluate_delta_features(order: int, q_values, eps: float = 1e-12):
    """
    Numerically evaluate your MLP input features for WENO-(order).

    q_values must have length 2r-1.

    Returns
    -------
    {
        "delta": [delta0,...],
        "beta": [beta0,...],
        "features": [delta0/max_delta,...,gamma_s,scale_feature],
        "gamma_s": gamma_s,
        "scale_feature": scale_feature,
        "max_delta": max_delta
    }
    """
    data = get_decomposition(order)
    r = data["r"]
    n = 2 * r - 1

    if len(q_values) != n:
        raise ValueError(f"WENO{order} needs {n} stencil values, got {len(q_values)}.")

    q_values = [float(v) for v in q_values]
    q_symbols = data["q"]
    subs = {q_symbols[j]: q_values[j] for j in range(n)}

    deltas = []
    betas = []

    for modes in data["items"]:
        delta = 0.0
        beta = 0.0

        for mode in modes:
            D_val = float(mode["D"].subs(subs))
            c_val = float(mode["coeff"])

            delta += c_val * abs(D_val)
            beta += c_val * D_val * D_val

        deltas.append(delta)
        betas.append(beta)

    max_delta = max(deltas)

    if max_delta <= 0.0:
        norm_delta = [0.0] * r
    else:
        norm_delta = [d / max_delta for d in deltas]

    gammas = []

    for j in range(n - 2):
        d2 = q_values[j] - 2.0 * q_values[j + 1] + q_values[j + 2]
        denom = (
            abs(q_values[j + 1] - q_values[j])
            + abs(q_values[j + 2] - q_values[j + 1])
            + eps
        )
        gammas.append(abs(d2) / denom)

    gamma_s = min(1.0, max(gammas) if gammas else 0.0)

    q_scale = max(max(abs(v) for v in q_values), 1.0)
    relative_scale = max(max_delta / q_scale, 1e-30)
    scale_feature = min(1.0, max(0.0, (log10(relative_scale) + 16.0) / 16.0))

    features = norm_delta + [gamma_s, scale_feature]

    return {
        "delta": deltas,
        "beta": betas,
        "features": features,
        "gamma_s": gamma_s,
        "scale_feature": scale_feature,
        "max_delta": max_delta,
    }


if __name__ == "__main__":
    print_decomposition(5, index_fmt="U_con[{j}][i]")
    # 1. Generate C/CUDA code for WENO7.
    print_decomposition(7, index_fmt="U_con[{j}][i]")

    # 2. Check expanded integer beta. For WENO7 this gives 240 * beta,
    #    which should match your hand-written bate0~bate3.
    print_integer_scaled_beta(7, index_fmt="U_con[{j}][i]")

    # 3. Numerically evaluate MLP input features.
    q_example = [1.0, 1.1, 1.05, 0.8, 0.4, 0.2, 0.1]
    out = evaluate_delta_features(7, q_example)

    print("delta =", out["delta"])
    print("beta =", out["beta"])
    print("features =", out["features"])