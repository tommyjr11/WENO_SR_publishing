from __future__ import annotations

import warp as wp


# Match the trusted nvcc build (`--fmad=false` and no fast math).  Warp keeps
# floating-point contraction enabled independently of its fast-math switch.
wp.set_module_options({"fast_math": False, "fuse_fp": False})


Vec5d = wp.types.vector(length=5, dtype=wp.float64)
Mat55d = wp.types.matrix(shape=(5, 5), dtype=wp.float64)


@wp.func
def load_vec(q: wp.array4d(dtype=wp.float64), k: int, j: int, i: int) -> Vec5d:
    return Vec5d(q[k, j, i, 0], q[k, j, i, 1], q[k, j, i, 2], q[k, j, i, 3], q[k, j, i, 4])


@wp.func
def store_vec(q: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, value: Vec5d):
    q[k, j, i, 0] = value[0]
    q[k, j, i, 1] = value[1]
    q[k, j, i, 2] = value[2]
    q[k, j, i, 3] = value[3]
    q[k, j, i, 4] = value[4]


@wp.func
def conserved_to_primitive(con: Vec5d) -> Vec5d:
    pri = Vec5d()
    pri[0] = con[0]
    pri[1] = con[1] / con[0]
    pri[2] = con[2] / con[0]
    pri[3] = con[3] / con[0]
    pri[4] = (wp.float64(1.4) - wp.float64(1.0)) * (
        con[4]
        - wp.float64(0.5)
        * con[0]
        * (pri[1] * pri[1] + pri[2] * pri[2] + pri[3] * pri[3])
    )
    return pri


@wp.func
def primitive_to_conserved(pri: Vec5d) -> Vec5d:
    con = Vec5d()
    con[0] = pri[0]
    con[1] = pri[0] * pri[1]
    con[2] = pri[0] * pri[2]
    con[3] = pri[0] * pri[3]
    con[4] = pri[4] / (wp.float64(1.4) - wp.float64(1.0)) + wp.float64(0.5) * pri[0] * (
        pri[1] * pri[1] + pri[2] * pri[2] + pri[3] * pri[3]
    )
    return con


@wp.func
def flux_from_primitive(pri: Vec5d, direction: int) -> Vec5d:
    rho = pri[0]
    u = pri[1]
    v = pri[2]
    w = pri[3]
    p = pri[4]
    energy = wp.float64(0.5) * rho * (u * u + v * v + w * w) + p / (wp.float64(1.4) - wp.float64(1.0))
    flux = Vec5d()
    if direction == 1:
        flux[0] = rho * u
        flux[1] = rho * u * u + p
        flux[2] = rho * u * v
        flux[3] = rho * u * w
        flux[4] = u * (p + energy)
    elif direction == 2:
        flux[0] = rho * v
        flux[1] = rho * u * v
        flux[2] = rho * v * v + p
        flux[3] = rho * v * w
        flux[4] = v * (p + energy)
    else:
        flux[0] = rho * w
        flux[1] = rho * u * w
        flux[2] = rho * v * w
        flux[3] = rho * w * w + p
        flux[4] = w * (p + energy)
    return flux


@wp.func
def clamped_flux_from_primitive(pri_in: Vec5d, direction: int) -> Vec5d:
    eps = wp.float64(1.0e-15)
    pri = Vec5d()
    pri[0] = wp.max(pri_in[0], eps)
    pri[1] = pri_in[1]
    pri[2] = pri_in[2]
    pri[3] = pri_in[3]
    pri[4] = wp.max(pri_in[4], eps)
    return flux_from_primitive(pri, direction)


@wp.func
def normal_velocity(pri: Vec5d, direction: int) -> wp.float64:
    result = pri[3]
    if direction == 1:
        result = pri[1]
    elif direction == 2:
        result = pri[2]
    return result


@wp.func
def set_normal_velocity(pri_in: Vec5d, direction: int, value: wp.float64) -> Vec5d:
    pri = pri_in
    if direction == 1:
        pri[1] = value
    elif direction == 2:
        pri[2] = value
    else:
        pri[3] = value
    return pri


@wp.func
def evilin_state(ul: Vec5d, ur: Vec5d, direction: int, dt_dx: wp.float64) -> Vec5d:
    gamma = wp.float64(1.4)
    eps = wp.float64(1.0e-15)

    ul0 = ul
    ur0 = ur
    wl0 = conserved_to_primitive(ul0)
    wr0 = conserved_to_primitive(ur0)
    wl0[0] = wp.max(wl0[0], eps)
    wl0[4] = wp.max(wl0[4], eps)
    wr0[0] = wp.max(wr0[0], eps)
    wr0[4] = wp.max(wr0[4], eps)

    unl0 = normal_velocity(wl0, direction)
    unr0 = normal_velocity(wr0, direction)
    al0 = wp.sqrt(gamma * wl0[4] / wl0[0])
    ar0 = wp.sqrt(gamma * wr0[4] / wr0[0])
    smax = wp.max(wp.abs(unl0) + al0, wp.abs(unr0) + ar0)
    smax = wp.max(smax, eps)

    dx_dt = wp.float64(1.0) / dt_dx
    courant = smax * dt_dx
    fl = clamped_flux_from_primitive(wl0, direction)
    fr = clamped_flux_from_primitive(wr0, direction)

    u_lw = Vec5d()
    for c in range(5):
        u_lw[c] = wp.float64(0.5) * (ul0[c] + ur0[c]) - wp.float64(0.5) * dt_dx * (fr[c] - fl[c])

    w_lw = conserved_to_primitive(u_lw)
    w_lw[0] = wp.max(w_lw[0], eps)
    w_lw[4] = wp.max(w_lw[4], eps)
    f_lw = clamped_flux_from_primitive(w_lw, direction)

    f_fo = Vec5d()
    for c in range(5):
        f_fo[c] = wp.float64(0.25) * (
            fl[c] + wp.float64(2.0) * f_lw[c] + fr[c] - dx_dt * (ur0[c] - ul0[c])
        )

    theta = (wp.float64(1.0) - courant) / (wp.float64(1.0) + courant)
    f_gf = Vec5d()
    for c in range(5):
        f_gf[c] = theta * f_lw[c] + (wp.float64(1.0) - theta) * f_fo[c]

    ulh = Vec5d()
    urh = Vec5d()
    for c in range(5):
        ulh[c] = ul0[c] - dt_dx * (f_gf[c] - fl[c])
        urh[c] = ur0[c] - dt_dx * (fr[c] - f_gf[c])

    wl = conserved_to_primitive(ulh)
    wr = conserved_to_primitive(urh)
    wl[0] = wp.max(wl[0], eps)
    wl[4] = wp.max(wl[4], eps)
    wr[0] = wp.max(wr[0], eps)
    wr[4] = wp.max(wr[4], eps)

    rho_bar = wp.float64(0.5) * (wl[0] + wr[0])
    p_bar = wp.float64(0.5) * (wl[4] + wr[4])
    rho_bar = wp.max(rho_bar, eps)
    p_bar = wp.max(p_bar, eps)
    a_bar = wp.sqrt(gamma * p_bar / rho_bar)
    a_bar = wp.max(a_bar, eps)
    c1 = rho_bar * a_bar
    c2 = rho_bar / a_bar

    unle = normal_velocity(wl, direction)
    unre = normal_velocity(wr, direction)
    u_star = wp.float64(0.5) * (unle + unre) - wp.float64(0.5) * (wr[4] - wl[4]) / c1
    p_star = wp.float64(0.5) * (wl[4] + wr[4]) - wp.float64(0.5) * (unre - unle) * c1
    rho_star_l = wl[0] + (unle - u_star) * c2
    rho_star_r = wr[0] + (u_star - unre) * c2
    rho_star_l = wp.max(rho_star_l, eps)
    rho_star_r = wp.max(rho_star_r, eps)
    p_star = wp.max(p_star, eps)

    u_bar = wp.float64(0.5) * (unle + unre)
    lam1 = u_bar - a_bar
    lam2 = u_bar
    lam5 = u_bar + a_bar
    wface = wr
    if lam1 >= wp.float64(0.0):
        wface = wl
    elif lam2 >= wp.float64(0.0):
        wface = wl
        wface[0] = rho_star_l
        wface = set_normal_velocity(wface, direction, u_star)
        wface[4] = p_star
    elif lam5 >= wp.float64(0.0):
        wface = wr
        wface[0] = rho_star_r
        wface = set_normal_velocity(wface, direction, u_star)
        wface[4] = p_star
    wface[0] = wp.max(wface[0], eps)
    wface[4] = wp.max(wface[4], eps)
    return primitive_to_conserved(wface)


@wp.func
def matvec5(matrix: Mat55d, vector: Vec5d) -> Vec5d:
    result = Vec5d()
    for row in range(5):
        total = wp.float64(0.0)
        for column in range(5):
            total = total + matrix[row, column] * vector[column]
        result[row] = total
    return result


@wp.func
def roe_average(left: Vec5d, right: Vec5d) -> Vec5d:
    gamma = wp.float64(1.4)
    rho_l = left[0]
    u_l = left[1] / rho_l
    v_l = left[2] / rho_l
    w_l = left[3] / rho_l
    e_l = left[4]
    p_l = (gamma - wp.float64(1.0)) * (
        e_l - wp.float64(0.5) * rho_l * (u_l * u_l + v_l * v_l + w_l * w_l)
    )
    h_l = (e_l + p_l) / rho_l

    rho_r = right[0]
    u_r = right[1] / rho_r
    v_r = right[2] / rho_r
    w_r = right[3] / rho_r
    e_r = right[4]
    p_r = (gamma - wp.float64(1.0)) * (
        e_r - wp.float64(0.5) * rho_r * (u_r * u_r + v_r * v_r + w_r * w_r)
    )
    h_r = (e_r + p_r) / rho_r

    sqrt_rho_l = wp.sqrt(rho_l)
    sqrt_rho_r = wp.sqrt(rho_r)
    inv_denom = wp.float64(1.0) / (sqrt_rho_l + sqrt_rho_r)
    u_roe = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) * inv_denom
    v_roe = (sqrt_rho_l * v_l + sqrt_rho_r * v_r) * inv_denom
    w_roe = (sqrt_rho_l * w_l + sqrt_rho_r * w_r) * inv_denom
    h_roe = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) * inv_denom
    rho_roe = sqrt_rho_l * sqrt_rho_r
    q2_roe = u_roe * u_roe + v_roe * v_roe + w_roe * w_roe
    e_roe = (
        rho_roe * h_roe
        + wp.float64(0.5) * (gamma - wp.float64(1.0)) * rho_roe * q2_roe
    ) / gamma
    return Vec5d(rho_roe, rho_roe * u_roe, rho_roe * v_roe, rho_roe * w_roe, e_roe)


@wp.func
def eigen_left_x(state: Vec5d) -> Mat55d:
    rho = state[0]
    u = state[1] / rho
    v = state[2] / rho
    w = state[3] / rho
    energy = state[4]
    gm1 = wp.float64(1.4) - wp.float64(1.0)
    vtot = u * u + v * v + w * w
    pressure = gm1 * (energy - wp.float64(0.5) * rho * vtot)
    sound = wp.sqrt(wp.float64(1.4) * pressure / rho)
    sound2 = sound * sound
    enthalpy = wp.float64(0.5) * vtot + sound2 / gm1
    matrix = Mat55d()
    matrix[0, 0] = enthalpy + sound * (u - sound) / gm1
    matrix[0, 1] = -(u + sound / gm1)
    matrix[0, 2] = -v
    matrix[0, 3] = -w
    matrix[0, 4] = wp.float64(1.0)
    matrix[1, 0] = -wp.float64(2.0) * enthalpy + wp.float64(4.0) * sound2 / gm1
    matrix[1, 1] = wp.float64(2.0) * u
    matrix[1, 2] = wp.float64(2.0) * v
    matrix[1, 3] = wp.float64(2.0) * w
    matrix[1, 4] = -wp.float64(2.0)
    matrix[2, 0] = -wp.float64(2.0) * v * sound2 / gm1
    matrix[2, 1] = wp.float64(0.0)
    matrix[2, 2] = wp.float64(2.0) * sound2 / gm1
    matrix[2, 3] = wp.float64(0.0)
    matrix[2, 4] = wp.float64(0.0)
    matrix[3, 0] = -wp.float64(2.0) * w * sound2 / gm1
    matrix[3, 1] = wp.float64(0.0)
    matrix[3, 2] = wp.float64(0.0)
    matrix[3, 3] = wp.float64(2.0) * sound2 / gm1
    matrix[3, 4] = wp.float64(0.0)
    matrix[4, 0] = enthalpy - sound * (u + sound) / gm1
    matrix[4, 1] = -(u - sound / gm1)
    matrix[4, 2] = -v
    matrix[4, 3] = -w
    matrix[4, 4] = wp.float64(1.0)
    scale = gm1 / (wp.float64(2.0) * sound2)
    for row in range(5):
        for column in range(5):
            matrix[row, column] = matrix[row, column] * scale
    return matrix


@wp.func
def eigen_right_x(state: Vec5d) -> Mat55d:
    rho = state[0]
    u = state[1] / rho
    v = state[2] / rho
    w = state[3] / rho
    energy = state[4]
    gm1 = wp.float64(1.4) - wp.float64(1.0)
    vtot = u * u + v * v + w * w
    pressure = gm1 * (energy - wp.float64(0.5) * rho * vtot)
    sound = wp.sqrt(wp.float64(1.4) * pressure / rho)
    sound2 = sound * sound
    enthalpy = wp.float64(0.5) * vtot + sound2 / gm1
    matrix = Mat55d()
    matrix[0, 0] = wp.float64(1.0); matrix[0, 1] = wp.float64(1.0); matrix[0, 2] = wp.float64(0.0); matrix[0, 3] = wp.float64(0.0); matrix[0, 4] = wp.float64(1.0)
    matrix[1, 0] = u - sound; matrix[1, 1] = u; matrix[1, 2] = wp.float64(0.0); matrix[1, 3] = wp.float64(0.0); matrix[1, 4] = u + sound
    matrix[2, 0] = v; matrix[2, 1] = v; matrix[2, 2] = wp.float64(1.0); matrix[2, 3] = wp.float64(0.0); matrix[2, 4] = v
    matrix[3, 0] = w; matrix[3, 1] = w; matrix[3, 2] = wp.float64(0.0); matrix[3, 3] = wp.float64(1.0); matrix[3, 4] = w
    matrix[4, 0] = enthalpy - u * sound
    matrix[4, 1] = wp.float64(0.5) * vtot
    matrix[4, 2] = v
    matrix[4, 3] = w
    matrix[4, 4] = enthalpy + u * sound
    return matrix


@wp.func
def eigen_left_y(state: Vec5d) -> Mat55d:
    rho = state[0]
    u = state[1] / rho
    v = state[2] / rho
    w = state[3] / rho
    energy = state[4]
    gm1 = wp.float64(1.4) - wp.float64(1.0)
    vtot = u * u + v * v + w * w
    pressure = gm1 * (energy - wp.float64(0.5) * rho * vtot)
    sound = wp.sqrt(wp.float64(1.4) * pressure / rho)
    sound2 = sound * sound
    enthalpy = wp.float64(0.5) * vtot + sound2 / gm1
    normal = v
    tangent1 = u
    tangent2 = w
    matrix = Mat55d()
    matrix[0, 0] = enthalpy + sound * (normal - sound) / gm1
    matrix[0, 1] = -tangent1
    matrix[0, 2] = -(normal + sound / gm1)
    matrix[0, 3] = -tangent2
    matrix[0, 4] = wp.float64(1.0)
    matrix[1, 0] = -wp.float64(2.0) * enthalpy + wp.float64(4.0) * sound2 / gm1
    matrix[1, 1] = wp.float64(2.0) * tangent1
    matrix[1, 2] = wp.float64(2.0) * normal
    matrix[1, 3] = wp.float64(2.0) * tangent2
    matrix[1, 4] = -wp.float64(2.0)
    matrix[2, 0] = -wp.float64(2.0) * tangent1 * sound2 / gm1
    matrix[2, 1] = wp.float64(2.0) * sound2 / gm1
    matrix[2, 2] = wp.float64(0.0)
    matrix[2, 3] = wp.float64(0.0)
    matrix[2, 4] = wp.float64(0.0)
    matrix[3, 0] = -wp.float64(2.0) * tangent2 * sound2 / gm1
    matrix[3, 1] = wp.float64(0.0)
    matrix[3, 2] = wp.float64(0.0)
    matrix[3, 3] = wp.float64(2.0) * sound2 / gm1
    matrix[3, 4] = wp.float64(0.0)
    matrix[4, 0] = enthalpy - sound * (normal + sound) / gm1
    matrix[4, 1] = -tangent1
    matrix[4, 2] = -normal + sound / gm1
    matrix[4, 3] = -tangent2
    matrix[4, 4] = wp.float64(1.0)
    scale = gm1 / (wp.float64(2.0) * sound2)
    for row in range(5):
        for column in range(5):
            matrix[row, column] = matrix[row, column] * scale
    return matrix


@wp.func
def eigen_right_y(state: Vec5d) -> Mat55d:
    rho = state[0]
    u = state[1] / rho
    v = state[2] / rho
    w = state[3] / rho
    energy = state[4]
    gm1 = wp.float64(1.4) - wp.float64(1.0)
    vtot = u * u + v * v + w * w
    pressure = gm1 * (energy - wp.float64(0.5) * rho * vtot)
    sound = wp.sqrt(wp.float64(1.4) * pressure / rho)
    sound2 = sound * sound
    enthalpy = wp.float64(0.5) * vtot + sound2 / gm1
    normal = v
    tangent1 = u
    tangent2 = w
    matrix = Mat55d()
    matrix[0, 0] = wp.float64(1.0); matrix[0, 1] = wp.float64(1.0); matrix[0, 2] = wp.float64(0.0); matrix[0, 3] = wp.float64(0.0); matrix[0, 4] = wp.float64(1.0)
    matrix[1, 0] = tangent1; matrix[1, 1] = tangent1; matrix[1, 2] = wp.float64(1.0); matrix[1, 3] = wp.float64(0.0); matrix[1, 4] = tangent1
    matrix[2, 0] = normal - sound; matrix[2, 1] = normal; matrix[2, 2] = wp.float64(0.0); matrix[2, 3] = wp.float64(0.0); matrix[2, 4] = normal + sound
    matrix[3, 0] = tangent2; matrix[3, 1] = tangent2; matrix[3, 2] = wp.float64(0.0); matrix[3, 3] = wp.float64(1.0); matrix[3, 4] = tangent2
    matrix[4, 0] = enthalpy - normal * sound
    matrix[4, 1] = wp.float64(0.5) * vtot
    matrix[4, 2] = tangent1
    matrix[4, 3] = tangent2
    matrix[4, 4] = enthalpy + normal * sound
    return matrix


@wp.func
def eigen_left_z(state: Vec5d) -> Mat55d:
    rho = state[0]
    u = state[1] / rho
    v = state[2] / rho
    w = state[3] / rho
    energy = state[4]
    gm1 = wp.float64(1.4) - wp.float64(1.0)
    vtot = u * u + v * v + w * w
    pressure = gm1 * (energy - wp.float64(0.5) * rho * vtot)
    sound = wp.sqrt(wp.float64(1.4) * pressure / rho)
    sound2 = sound * sound
    enthalpy = wp.float64(0.5) * vtot + sound2 / gm1
    normal = w
    tangent1 = u
    tangent2 = v
    matrix = Mat55d()
    matrix[0, 0] = enthalpy + sound * (normal - sound) / gm1
    matrix[0, 1] = -tangent1
    matrix[0, 2] = -tangent2
    matrix[0, 3] = -(normal + sound / gm1)
    matrix[0, 4] = wp.float64(1.0)
    matrix[1, 0] = -wp.float64(2.0) * enthalpy + wp.float64(4.0) * sound2 / gm1
    matrix[1, 1] = wp.float64(2.0) * tangent1
    matrix[1, 2] = wp.float64(2.0) * tangent2
    matrix[1, 3] = wp.float64(2.0) * normal
    matrix[1, 4] = -wp.float64(2.0)
    matrix[2, 0] = -wp.float64(2.0) * tangent1 * sound2 / gm1
    matrix[2, 1] = wp.float64(2.0) * sound2 / gm1
    matrix[2, 2] = wp.float64(0.0)
    matrix[2, 3] = wp.float64(0.0)
    matrix[2, 4] = wp.float64(0.0)
    matrix[3, 0] = -wp.float64(2.0) * tangent2 * sound2 / gm1
    matrix[3, 1] = wp.float64(0.0)
    matrix[3, 2] = wp.float64(2.0) * sound2 / gm1
    matrix[3, 3] = wp.float64(0.0)
    matrix[3, 4] = wp.float64(0.0)
    matrix[4, 0] = enthalpy - sound * (normal + sound) / gm1
    matrix[4, 1] = -tangent1
    matrix[4, 2] = -tangent2
    matrix[4, 3] = -normal + sound / gm1
    matrix[4, 4] = wp.float64(1.0)
    scale = gm1 / (wp.float64(2.0) * sound2)
    for row in range(5):
        for column in range(5):
            matrix[row, column] = matrix[row, column] * scale
    return matrix


@wp.func
def eigen_right_z(state: Vec5d) -> Mat55d:
    rho = state[0]
    u = state[1] / rho
    v = state[2] / rho
    w = state[3] / rho
    energy = state[4]
    gm1 = wp.float64(1.4) - wp.float64(1.0)
    vtot = u * u + v * v + w * w
    pressure = gm1 * (energy - wp.float64(0.5) * rho * vtot)
    sound = wp.sqrt(wp.float64(1.4) * pressure / rho)
    sound2 = sound * sound
    enthalpy = wp.float64(0.5) * vtot + sound2 / gm1
    normal = w
    tangent1 = u
    tangent2 = v
    matrix = Mat55d()
    matrix[0, 0] = wp.float64(1.0); matrix[0, 1] = wp.float64(1.0); matrix[0, 2] = wp.float64(0.0); matrix[0, 3] = wp.float64(0.0); matrix[0, 4] = wp.float64(1.0)
    matrix[1, 0] = tangent1; matrix[1, 1] = tangent1; matrix[1, 2] = wp.float64(1.0); matrix[1, 3] = wp.float64(0.0); matrix[1, 4] = tangent1
    matrix[2, 0] = tangent2; matrix[2, 1] = tangent2; matrix[2, 2] = wp.float64(0.0); matrix[2, 3] = wp.float64(1.0); matrix[2, 4] = tangent2
    matrix[3, 0] = normal - sound; matrix[3, 1] = normal; matrix[3, 2] = wp.float64(0.0); matrix[3, 3] = wp.float64(0.0); matrix[3, 4] = normal + sound
    matrix[4, 0] = enthalpy - normal * sound
    matrix[4, 1] = wp.float64(0.5) * vtot
    matrix[4, 2] = tangent1
    matrix[4, 3] = tangent2
    matrix[4, 4] = enthalpy + normal * sound
    return matrix


@wp.func
def safe_rcp(beta: wp.float64) -> wp.float64:
    return wp.float64(1.0) / (wp.float64(1.0e-6) + beta)


@wp.func
def weno_beta0(q0: wp.float64, q1: wp.float64, q2: wp.float64) -> wp.float64:
    a = q0 - wp.float64(2.0) * q1 + q2
    b = q0 - wp.float64(4.0) * q1 + wp.float64(3.0) * q2
    return (wp.float64(13.0) / wp.float64(12.0)) * a * a + (wp.float64(1.0) / wp.float64(4.0)) * b * b


@wp.func
def weno_beta1(q1: wp.float64, q2: wp.float64, q3: wp.float64) -> wp.float64:
    a = q1 - wp.float64(2.0) * q2 + q3
    b = q1 - q3
    return (wp.float64(13.0) / wp.float64(12.0)) * a * a + (wp.float64(1.0) / wp.float64(4.0)) * b * b


@wp.func
def weno_beta2(q2: wp.float64, q3: wp.float64, q4: wp.float64) -> wp.float64:
    a = q2 - wp.float64(2.0) * q3 + q4
    b = wp.float64(3.0) * q2 - wp.float64(4.0) * q3 + q4
    return (wp.float64(13.0) / wp.float64(12.0)) * a * a + (wp.float64(1.0) / wp.float64(4.0)) * b * b


@wp.func
def weno_face_scalar(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, lr: int, h: wp.float64) -> wp.float64:
    beta0 = weno_beta0(q0, q1, q2)
    beta1 = weno_beta1(q1, q2, q3)
    beta2 = weno_beta2(q2, q3, q4)
    s0 = wp.float64(0.0)
    s1 = wp.float64(0.0)
    s2 = wp.float64(0.0)
    d0 = wp.float64(0.0)
    d1 = wp.float64(3.0) / wp.float64(5.0)
    d2 = wp.float64(0.0)
    if lr == 1:
        s0 = (wp.float64(1.0) / wp.float64(3.0)) * q0 - (wp.float64(7.0) / wp.float64(6.0)) * q1 + (wp.float64(11.0) / wp.float64(6.0)) * q2
        s1 = (-wp.float64(1.0) / wp.float64(6.0)) * q1 + (wp.float64(5.0) / wp.float64(6.0)) * q2 + (wp.float64(1.0) / wp.float64(3.0)) * q3
        s2 = (wp.float64(1.0) / wp.float64(3.0)) * q2 + (wp.float64(5.0) / wp.float64(6.0)) * q3 - (wp.float64(1.0) / wp.float64(6.0)) * q4
        d0 = wp.float64(1.0) / wp.float64(10.0)
        d2 = wp.float64(3.0) / wp.float64(10.0)
    else:
        s0 = (-wp.float64(1.0) / wp.float64(6.0)) * q0 + (wp.float64(5.0) / wp.float64(6.0)) * q1 + (wp.float64(1.0) / wp.float64(3.0)) * q2
        s1 = (wp.float64(1.0) / wp.float64(3.0)) * q1 + (wp.float64(5.0) / wp.float64(6.0)) * q2 - (wp.float64(1.0) / wp.float64(6.0)) * q3
        s2 = (wp.float64(11.0) / wp.float64(6.0)) * q2 - (wp.float64(7.0) / wp.float64(6.0)) * q3 + (wp.float64(1.0) / wp.float64(3.0)) * q4
        d0 = wp.float64(3.0) / wp.float64(10.0)
        d2 = wp.float64(1.0) / wp.float64(10.0)
    tau5 = wp.abs(beta0 - beta2)
    eps_z = h * h
    ratio0 = tau5 / (beta0 + eps_z)
    ratio1 = tau5 / (beta1 + eps_z)
    ratio2 = tau5 / (beta2 + eps_z)
    alpha0 = d0 * (wp.float64(1.0) + ratio0 * ratio0)
    alpha1 = d1 * (wp.float64(1.0) + ratio1 * ratio1)
    alpha2 = d2 * (wp.float64(1.0) + ratio2 * ratio2)
    total = alpha0 + alpha1 + alpha2
    weight0 = alpha0 / total
    weight1 = alpha1 / total
    weight2 = alpha2 / total
    return weight0 * s0 + weight1 * s1 + weight2 * s2


@wp.func
def weno_face_vec(q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d, q4: Vec5d, lr: int, h: wp.float64) -> Vec5d:
    result = Vec5d()
    for component in range(5):
        result[component] = weno_face_scalar(q0[component], q1[component], q2[component], q3[component], q4[component], lr, h)
    return result


@wp.func
def characteristic_weno_face_x(q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d, q4: Vec5d, lr: int, h: wp.float64) -> Vec5d:
    left_index_state = q1
    right_index_state = q2
    if lr == 1:
        left_index_state = q2
        right_index_state = q3
    average = roe_average(left_index_state, right_index_state)
    left_matrix = eigen_left_x(average)
    right_matrix = eigen_right_x(average)
    c0 = matvec5(left_matrix, q0)
    c1 = matvec5(left_matrix, q1)
    c2 = matvec5(left_matrix, q2)
    c3 = matvec5(left_matrix, q3)
    c4 = matvec5(left_matrix, q4)
    characteristic = weno_face_vec(c0, c1, c2, c3, c4, lr, h)
    return matvec5(right_matrix, characteristic)


@wp.func
def characteristic_weno_face_y(q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d, q4: Vec5d, lr: int, h: wp.float64) -> Vec5d:
    left_index_state = q1
    right_index_state = q2
    if lr == 1:
        left_index_state = q2
        right_index_state = q3
    average = roe_average(left_index_state, right_index_state)
    left_matrix = eigen_left_y(average)
    right_matrix = eigen_right_y(average)
    c0 = matvec5(left_matrix, q0)
    c1 = matvec5(left_matrix, q1)
    c2 = matvec5(left_matrix, q2)
    c3 = matvec5(left_matrix, q3)
    c4 = matvec5(left_matrix, q4)
    characteristic = weno_face_vec(c0, c1, c2, c3, c4, lr, h)
    return matvec5(right_matrix, characteristic)


@wp.func
def characteristic_weno_face_z(q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d, q4: Vec5d, lr: int, h: wp.float64) -> Vec5d:
    left_index_state = q1
    right_index_state = q2
    if lr == 1:
        left_index_state = q2
        right_index_state = q3
    average = roe_average(left_index_state, right_index_state)
    left_matrix = eigen_left_z(average)
    right_matrix = eigen_right_z(average)
    c0 = matvec5(left_matrix, q0)
    c1 = matvec5(left_matrix, q1)
    c2 = matvec5(left_matrix, q2)
    c3 = matvec5(left_matrix, q3)
    c4 = matvec5(left_matrix, q4)
    characteristic = weno_face_vec(c0, c1, c2, c3, c4, lr, h)
    return matvec5(right_matrix, characteristic)


@wp.func
def characteristic_roundtrip_x_lr2(q: Vec5d, roe_left: Vec5d, roe_right: Vec5d) -> Vec5d:
    """Reproduce the in-place U_con mutation in the trusted C++ LR=2 call."""
    average = roe_average(roe_left, roe_right)
    left_matrix = eigen_left_x(average)
    right_matrix = eigen_right_x(average)
    return matvec5(right_matrix, matvec5(left_matrix, q))


@wp.func
def characteristic_roundtrip_y_lr2(q: Vec5d, roe_left: Vec5d, roe_right: Vec5d) -> Vec5d:
    average = roe_average(roe_left, roe_right)
    left_matrix = eigen_left_y(average)
    right_matrix = eigen_right_y(average)
    return matvec5(right_matrix, matvec5(left_matrix, q))


@wp.func
def characteristic_roundtrip_z_lr2(q: Vec5d, roe_left: Vec5d, roe_right: Vec5d) -> Vec5d:
    average = roe_average(roe_left, roe_right)
    left_matrix = eigen_left_z(average)
    right_matrix = eigen_right_z(average)
    return matvec5(right_matrix, matvec5(left_matrix, q))


@wp.func
def gauss_candidate(q0: wp.float64, q1: wp.float64, q2: wp.float64, location: int) -> wp.float64:
    result = wp.float64(0.0)
    if location == 1:
        result = -wp.float64(0.144337567297407) * q0 + wp.float64(0.577350269189626) * q1 + wp.float64(0.56698729810778) * q2
    elif location == 2:
        result = wp.float64(0.144337567297406) * q0 + wp.float64(1.0) * q1 - wp.float64(0.144337567297407) * q2
    elif location == 3:
        result = wp.float64(1.43301270189222) * q0 - wp.float64(0.577350269189626) * q1 + wp.float64(0.144337567297406) * q2
    elif location == 4:
        result = wp.float64(0.144337567297406) * q0 - wp.float64(0.577350269189626) * q1 + wp.float64(1.43301270189222) * q2
    elif location == 5:
        result = -wp.float64(0.144337567297407) * q0 + wp.float64(1.0) * q1 + wp.float64(0.144337567297406) * q2
    else:
        result = wp.float64(0.566987298107781) * q0 + wp.float64(0.577350269189625) * q1 - wp.float64(0.144337567297406) * q2
    return result


@wp.func
def gauss_weno_scalar(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64, q4: wp.float64, lr: int, h: wp.float64) -> wp.float64:
    beta0 = weno_beta0(q0, q1, q2)
    beta1 = weno_beta1(q1, q2, q3)
    beta2 = weno_beta2(q2, q3, q4)
    root3 = wp.sqrt(wp.float64(3.0))
    d0 = (wp.float64(210.0) + root3) / wp.float64(1080.0)
    d1 = wp.float64(11.0) / wp.float64(18.0)
    d2 = (wp.float64(210.0) - root3) / wp.float64(1080.0)
    location0 = 1
    location1 = 2
    location2 = 3
    if lr == 2:
        d0 = (wp.float64(210.0) - root3) / wp.float64(1080.0)
        d2 = (wp.float64(210.0) + root3) / wp.float64(1080.0)
        location0 = 4
        location1 = 5
        location2 = 6
    tau5 = wp.abs(beta0 - beta2)
    eps_z = h * h
    ratio0 = tau5 / (beta0 + eps_z)
    ratio1 = tau5 / (beta1 + eps_z)
    ratio2 = tau5 / (beta2 + eps_z)
    alpha0 = d0 * (wp.float64(1.0) + ratio0 * ratio0)
    alpha1 = d1 * (wp.float64(1.0) + ratio1 * ratio1)
    alpha2 = d2 * (wp.float64(1.0) + ratio2 * ratio2)
    total = alpha0 + alpha1 + alpha2
    weight0 = alpha0 / total
    weight1 = alpha1 / total
    weight2 = alpha2 / total
    s0 = gauss_candidate(q0, q1, q2, location0)
    s1 = gauss_candidate(q1, q2, q3, location1)
    s2 = gauss_candidate(q2, q3, q4, location2)
    return weight0 * s0 + weight1 * s1 + weight2 * s2


@wp.func
def gauss_weno_vec(q0: Vec5d, q1: Vec5d, q2: Vec5d, q3: Vec5d, q4: Vec5d, lr: int, h: wp.float64) -> Vec5d:
    result = Vec5d()
    for component in range(5):
        result[component] = gauss_weno_scalar(q0[component], q1[component], q2[component], q3[component], q4[component], lr, h)
    return result


@wp.func
def gauss_node(index: int) -> wp.float64:
    value = wp.float64(0.0)
    if index == 0:
        value = wp.float64(-0.77459666924148340427791481488384306430816650390625)
    elif index == 2:
        value = wp.float64(0.77459666924148340427791481488384306430816650390625)
    return value


@wp.func
def gauss_weight(index: int) -> wp.float64:
    value = wp.float64(0.55555555555555546920487586248782463371753692626953125)
    if index == 1:
        value = wp.float64(0.88888888888888917261255073754000477492809295654296875)
    return value


@wp.func
def initial_shockbubble_component(x: wp.float64, y: wp.float64, z: wp.float64, component: int) -> wp.float64:
    gamma = wp.float64(1.4)
    mach = wp.float64(3.0)
    rho_air = wp.float64(1.29)
    p_air = wp.float64(101325.0)
    rho_he = wp.float64(0.214)
    rho_post = (((gamma + wp.float64(1.0)) * mach * mach) / ((gamma - wp.float64(1.0)) * mach * mach + wp.float64(2.0))) * rho_air
    p_post = ((((wp.float64(2.0) * gamma) * mach * mach) - (gamma - wp.float64(1.0))) / (gamma + wp.float64(1.0))) * p_air

    rho = rho_air
    u = wp.float64(0.0)
    v = wp.float64(0.0)
    w = wp.float64(0.0)
    pressure = p_air
    if x < wp.float64(0.005):
        rho = rho_post
        u = wp.float64(736.911)
        pressure = p_post

    dx_bubble = x - wp.float64(0.035)
    dy_bubble = y - wp.float64(0.0445)
    dz_bubble = z - wp.float64(0.0445)
    if dx_bubble * dx_bubble + dy_bubble * dy_bubble + dz_bubble * dz_bubble <= wp.float64(0.025) * wp.float64(0.025):
        rho = rho_he
        u = wp.float64(0.0)
        pressure = p_air

    value = rho
    if component == 1:
        value = rho * u
    elif component == 2:
        value = rho * v
    elif component == 3:
        value = rho * w
    elif component == 4:
        value = wp.float64(0.5) * rho * (u * u + v * v + w * w) + pressure / (gamma - wp.float64(1.0))
    return value


@wp.kernel
def initialize_shockbubble_kernel(
    q: wp.array4d(dtype=wp.float64),
    ghost: int,
    x_start: wp.float64,
    y_start: wp.float64,
    z_start: wp.float64,
    dx: wp.float64,
    dy: wp.float64,
    dz: wp.float64,
):
    k, j, i = wp.tid()
    x_center = x_start + (wp.float64(i - ghost) + wp.float64(0.5)) * dx
    y_center = y_start + (wp.float64(j - ghost) + wp.float64(0.5)) * dy
    z_center = z_start + (wp.float64(k - ghost) + wp.float64(0.5)) * dz

    for component in range(5):
        total = wp.float64(0.0)
        for gi in range(3):
            x = x_center + wp.float64(0.5) * dx * gauss_node(gi)
            wi = gauss_weight(gi)
            for gj in range(3):
                y = y_center + wp.float64(0.5) * dy * gauss_node(gj)
                wj = gauss_weight(gj)
                for gk in range(3):
                    z = z_center + wp.float64(0.5) * dz * gauss_node(gk)
                    wk = gauss_weight(gk)
                    value = initial_shockbubble_component(x, y, z, component)
                    total = total + wi * wj * wk * value
        q[k, j, i, component] = wp.float64(0.125) * total


@wp.kernel
def conserved_to_primitive_kernel(
    q: wp.array4d(dtype=wp.float64),
    primitive: wp.array4d(dtype=wp.float64),
):
    k, j, i = wp.tid()
    store_vec(primitive, k, j, i, conserved_to_primitive(load_vec(q, k, j, i)))


@wp.kernel
def boundary_x_kernel(q: wp.array4d(dtype=wp.float64), nx: int):
    k, j = wp.tid()
    for component in range(5):
        value_left = q[k, j, 3, component]
        q[k, j, 0, component] = value_left
        q[k, j, 1, component] = value_left
        q[k, j, 2, component] = value_left
        value_right = q[k, j, nx + 2, component]
        q[k, j, nx + 3, component] = value_right
        q[k, j, nx + 4, component] = value_right
        q[k, j, nx + 5, component] = value_right


@wp.kernel
def boundary_y_kernel(q: wp.array4d(dtype=wp.float64), ny: int):
    k, i = wp.tid()
    for component in range(5):
        value_bottom = q[k, 3, i, component]
        q[k, 0, i, component] = value_bottom
        q[k, 1, i, component] = value_bottom
        q[k, 2, i, component] = value_bottom
        value_top = q[k, ny + 2, i, component]
        q[k, ny + 3, i, component] = value_top
        q[k, ny + 4, i, component] = value_top
        q[k, ny + 5, i, component] = value_top


@wp.kernel
def boundary_z_kernel(q: wp.array4d(dtype=wp.float64), nz: int):
    j, i = wp.tid()
    for component in range(5):
        value_front = q[3, j, i, component]
        q[0, j, i, component] = value_front
        q[1, j, i, component] = value_front
        q[2, j, i, component] = value_front
        value_back = q[nz + 2, j, i, component]
        q[nz + 3, j, i, component] = value_back
        q[nz + 4, j, i, component] = value_back
        q[nz + 5, j, i, component] = value_back


@wp.kernel
def max_speed_kernel(
    primitive: wp.array4d(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float64),
    ghost: int,
):
    k, j, i = wp.tid()
    state = load_vec(primitive, k + ghost, j + ghost, i + ghost)
    sound = wp.sqrt(wp.float64(1.4) * state[4] / state[0])
    speed_x = wp.abs(state[1]) + sound
    speed_y = wp.abs(state[2]) + sound
    speed_z = wp.abs(state[3]) + sound
    local_maximum = wp.max(speed_x, wp.max(speed_y, speed_z))
    wp.atomic_max(maximum, 0, local_maximum)


@wp.kernel
def normal_x_kernel(
    q: wp.array4d(dtype=wp.float64),
    left: wp.array4d(dtype=wp.float64),
    right: wp.array4d(dtype=wp.float64),
    h: wp.float64,
):
    k, j, i = wp.tid()
    q0 = load_vec(q, k, j, i)
    q1 = load_vec(q, k, j, i + 1)
    q2 = load_vec(q, k, j, i + 2)
    q3 = load_vec(q, k, j, i + 3)
    q4 = load_vec(q, k, j, i + 4)
    left_value = characteristic_weno_face_x(q0, q1, q2, q3, q4, 2, h)
    # The C++ wrapper restores its mutable stencil with R*(L*U) after LR=2,
    # then invokes LR=1 on that round-tripped stencil.  Preserve that order.
    r0 = characteristic_roundtrip_x_lr2(q0, q1, q2)
    r1 = characteristic_roundtrip_x_lr2(q1, q1, q2)
    r2 = characteristic_roundtrip_x_lr2(q2, q1, q2)
    r3 = characteristic_roundtrip_x_lr2(q3, q1, q2)
    r4 = characteristic_roundtrip_x_lr2(q4, q1, q2)
    right_value = characteristic_weno_face_x(r0, r1, r2, r3, r4, 1, h)
    store_vec(left, k, j, i, left_value)
    store_vec(right, k, j, i, right_value)


@wp.kernel
def normal_y_kernel(
    q: wp.array4d(dtype=wp.float64),
    left: wp.array4d(dtype=wp.float64),
    right: wp.array4d(dtype=wp.float64),
    h: wp.float64,
):
    k, j, i = wp.tid()
    q0 = load_vec(q, k, j, i)
    q1 = load_vec(q, k, j + 1, i)
    q2 = load_vec(q, k, j + 2, i)
    q3 = load_vec(q, k, j + 3, i)
    q4 = load_vec(q, k, j + 4, i)
    left_value = characteristic_weno_face_y(q0, q1, q2, q3, q4, 2, h)
    r0 = characteristic_roundtrip_y_lr2(q0, q1, q2)
    r1 = characteristic_roundtrip_y_lr2(q1, q1, q2)
    r2 = characteristic_roundtrip_y_lr2(q2, q1, q2)
    r3 = characteristic_roundtrip_y_lr2(q3, q1, q2)
    r4 = characteristic_roundtrip_y_lr2(q4, q1, q2)
    right_value = characteristic_weno_face_y(r0, r1, r2, r3, r4, 1, h)
    store_vec(left, k, j, i, left_value)
    store_vec(right, k, j, i, right_value)


@wp.kernel
def normal_z_kernel(
    q: wp.array4d(dtype=wp.float64),
    left: wp.array4d(dtype=wp.float64),
    right: wp.array4d(dtype=wp.float64),
    h: wp.float64,
):
    k, j, i = wp.tid()
    q0 = load_vec(q, k, j, i)
    q1 = load_vec(q, k + 1, j, i)
    q2 = load_vec(q, k + 2, j, i)
    q3 = load_vec(q, k + 3, j, i)
    q4 = load_vec(q, k + 4, j, i)
    left_value = characteristic_weno_face_z(q0, q1, q2, q3, q4, 2, h)
    r0 = characteristic_roundtrip_z_lr2(q0, q1, q2)
    r1 = characteristic_roundtrip_z_lr2(q1, q1, q2)
    r2 = characteristic_roundtrip_z_lr2(q2, q1, q2)
    r3 = characteristic_roundtrip_z_lr2(q3, q1, q2)
    r4 = characteristic_roundtrip_z_lr2(q4, q1, q2)
    right_value = characteristic_weno_face_z(r0, r1, r2, r3, r4, 1, h)
    store_vec(left, k, j, i, left_value)
    store_vec(right, k, j, i, right_value)


@wp.func
def transverse_y_value(source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, lr: int, h: wp.float64) -> Vec5d:
    return gauss_weno_vec(
        load_vec(source, k, j + 1, i),
        load_vec(source, k, j + 2, i),
        load_vec(source, k, j + 3, i),
        load_vec(source, k, j + 4, i),
        load_vec(source, k, j + 5, i),
        lr, h,
    )


@wp.func
def transverse_x_value(source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, lr: int, h: wp.float64) -> Vec5d:
    return gauss_weno_vec(
        load_vec(source, k, j, i + 1),
        load_vec(source, k, j, i + 2),
        load_vec(source, k, j, i + 3),
        load_vec(source, k, j, i + 4),
        load_vec(source, k, j, i + 5),
        lr, h,
    )


@wp.func
def transverse_z_value(source: wp.array4d(dtype=wp.float64), k: int, j: int, i: int, lr: int, h: wp.float64) -> Vec5d:
    return gauss_weno_vec(
        load_vec(source, k + 1, j, i),
        load_vec(source, k + 2, j, i),
        load_vec(source, k + 3, j, i),
        load_vec(source, k + 4, j, i),
        load_vec(source, k + 5, j, i),
        lr, h,
    )


@wp.kernel
def transverse_y_kernel(
    source_left: wp.array4d(dtype=wp.float64),
    source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64),
    target_right: wp.array4d(dtype=wp.float64),
    lr: int,
    h: wp.float64,
):
    k, j, i = wp.tid()
    store_vec(target_left, k, j, i, transverse_y_value(source_left, k, j, i, lr, h))
    store_vec(target_right, k, j, i, transverse_y_value(source_right, k, j, i, lr, h))


@wp.kernel
def transverse_x_kernel(
    source_left: wp.array4d(dtype=wp.float64),
    source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64),
    target_right: wp.array4d(dtype=wp.float64),
    lr: int,
    h: wp.float64,
):
    k, j, i = wp.tid()
    store_vec(target_left, k, j, i, transverse_x_value(source_left, k, j, i, lr, h))
    store_vec(target_right, k, j, i, transverse_x_value(source_right, k, j, i, lr, h))


@wp.kernel
def transverse_z_kernel(
    source_left: wp.array4d(dtype=wp.float64),
    source_right: wp.array4d(dtype=wp.float64),
    target_left: wp.array4d(dtype=wp.float64),
    target_right: wp.array4d(dtype=wp.float64),
    lr: int,
    h: wp.float64,
):
    k, j, i = wp.tid()
    store_vec(target_left, k, j, i, transverse_z_value(source_left, k, j, i, lr, h))
    store_vec(target_right, k, j, i, transverse_z_value(source_right, k, j, i, lr, h))


@wp.kernel
def flux_x_kernel(
    left_points: wp.array4d(dtype=wp.float64),
    right_points: wp.array4d(dtype=wp.float64),
    flux: wp.array4d(dtype=wp.float64),
    dt_dx: wp.float64,
    flag: int,
):
    k, j, i = wp.tid()
    left_state = load_vec(left_points, k, j, i + 1)
    right_state = load_vec(right_points, k, j, i)
    interface_state = evilin_state(right_state, left_state, 1, dt_dx)
    value = flux_from_primitive(conserved_to_primitive(interface_state), 1)
    for component in range(5):
        scaled = value[component] * dt_dx
        if flag == 1:
            flux[k, j, i, component] = scaled
        else:
            flux[k, j, i, component] = flux[k, j, i, component] + scaled


@wp.kernel
def flux_y_kernel(
    left_points: wp.array4d(dtype=wp.float64),
    right_points: wp.array4d(dtype=wp.float64),
    flux: wp.array4d(dtype=wp.float64),
    dt_dy: wp.float64,
    flag: int,
):
    k, j, i = wp.tid()
    left_state = load_vec(left_points, k, j + 1, i)
    right_state = load_vec(right_points, k, j, i)
    interface_state = evilin_state(right_state, left_state, 2, dt_dy)
    value = flux_from_primitive(conserved_to_primitive(interface_state), 2)
    for component in range(5):
        scaled = value[component] * dt_dy
        if flag == 1:
            flux[k, j, i, component] = scaled
        else:
            flux[k, j, i, component] = flux[k, j, i, component] + scaled


@wp.kernel
def flux_z_kernel(
    left_points: wp.array4d(dtype=wp.float64),
    right_points: wp.array4d(dtype=wp.float64),
    flux: wp.array4d(dtype=wp.float64),
    dt_dz: wp.float64,
    flag: int,
):
    k, j, i = wp.tid()
    left_state = load_vec(left_points, k + 1, j, i)
    right_state = load_vec(right_points, k, j, i)
    interface_state = evilin_state(right_state, left_state, 3, dt_dz)
    value = flux_from_primitive(conserved_to_primitive(interface_state), 3)
    for component in range(5):
        scaled = value[component] * dt_dz
        if flag == 1:
            flux[k, j, i, component] = scaled
        else:
            flux[k, j, i, component] = flux[k, j, i, component] + scaled


@wp.func
def four_point_flux_difference(
    old: wp.float64,
    flux_x_left: wp.float64,
    flux_x_right: wp.float64,
    flux_y_left: wp.float64,
    flux_y_right: wp.float64,
    flux_z_left: wp.float64,
    flux_z_right: wp.float64,
) -> wp.float64:
    return old - wp.float64(0.25) * (flux_x_right - flux_x_left) - wp.float64(0.25) * (flux_y_right - flux_y_left) - wp.float64(0.25) * (flux_z_right - flux_z_left)


@wp.kernel
def rk_stage1_kernel(
    q: wp.array4d(dtype=wp.float64),
    q0: wp.array4d(dtype=wp.float64),
    flux_x: wp.array4d(dtype=wp.float64),
    flux_y: wp.array4d(dtype=wp.float64),
    flux_z: wp.array4d(dtype=wp.float64),
    ghost: int,
):
    k, j, i = wp.tid()
    kp = k + ghost
    jp = j + ghost
    ip = i + ghost
    for component in range(5):
        old = q[kp, jp, ip, component]
        q0[kp, jp, ip, component] = old
        q[kp, jp, ip, component] = four_point_flux_difference(
            old,
            flux_x[k, j, i, component], flux_x[k, j, i + 1, component],
            flux_y[k, j, i, component], flux_y[k, j + 1, i, component],
            flux_z[k, j, i, component], flux_z[k + 1, j, i, component],
        )


@wp.kernel
def rk_stage2_kernel(
    q: wp.array4d(dtype=wp.float64),
    q0: wp.array4d(dtype=wp.float64),
    flux_x: wp.array4d(dtype=wp.float64),
    flux_y: wp.array4d(dtype=wp.float64),
    flux_z: wp.array4d(dtype=wp.float64),
    ghost: int,
):
    k, j, i = wp.tid()
    kp = k + ghost
    jp = j + ghost
    ip = i + ghost
    for component in range(5):
        advanced = four_point_flux_difference(
            q[kp, jp, ip, component],
            flux_x[k, j, i, component], flux_x[k, j, i + 1, component],
            flux_y[k, j, i, component], flux_y[k, j + 1, i, component],
            flux_z[k, j, i, component], flux_z[k + 1, j, i, component],
        )
        q[kp, jp, ip, component] = (wp.float64(3.0) / wp.float64(4.0)) * q0[kp, jp, ip, component] + (wp.float64(1.0) / wp.float64(4.0)) * advanced


@wp.kernel
def rk_stage3_kernel(
    q: wp.array4d(dtype=wp.float64),
    q0: wp.array4d(dtype=wp.float64),
    primitive: wp.array4d(dtype=wp.float64),
    flux_x: wp.array4d(dtype=wp.float64),
    flux_y: wp.array4d(dtype=wp.float64),
    flux_z: wp.array4d(dtype=wp.float64),
    ghost: int,
):
    k, j, i = wp.tid()
    kp = k + ghost
    jp = j + ghost
    ip = i + ghost
    value = Vec5d()
    for component in range(5):
        advanced = four_point_flux_difference(
            q[kp, jp, ip, component],
            flux_x[k, j, i, component], flux_x[k, j, i + 1, component],
            flux_y[k, j, i, component], flux_y[k, j + 1, i, component],
            flux_z[k, j, i, component], flux_z[k + 1, j, i, component],
        )
        value[component] = (wp.float64(1.0) / wp.float64(3.0)) * q0[kp, jp, ip, component] + (wp.float64(2.0) / wp.float64(3.0)) * advanced
        q[kp, jp, ip, component] = value[component]
    store_vec(primitive, kp, jp, ip, conserved_to_primitive(value))


@wp.kernel
def diagnostics_kernel(
    primitive: wp.array4d(dtype=wp.float64),
    output: wp.array(dtype=wp.float64),
    ghost: int,
):
    k, j, i = wp.tid()
    state = load_vec(primitive, k + ghost, j + ghost, i + ghost)
    wp.atomic_min(output, 0, state[0])
    wp.atomic_max(output, 1, state[0])
    wp.atomic_min(output, 2, state[4])
    wp.atomic_max(output, 3, state[4])
