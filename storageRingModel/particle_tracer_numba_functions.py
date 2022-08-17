import numba
from constants import GRAVITATIONAL_ACCELERATION
from math import sqrt,isnan


@numba.njit()
def norm_3D(vec):
    return sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


@numba.njit()
def dot_Prod_3D(veca, vecb):
    return veca[0] * vecb[0] + veca[1] * vecb[1] + veca[2] * vecb[2]


@numba.njit()
def fast_qNew(q, F, p, h):
    return q + p * h + .5 * F * h ** 2


@numba.njit()
def fast_pNew(p, F, F_new, h):
    return p + .5 * (F + F_new) * h


@numba.njit()
def _transform_To_Next_Element(q, p, r01, r02, ROutEl1, RInEl2):
    # don't try and condense. Because of rounding, results won't agree with other methods and tests will fail
    q = q.copy()
    p = p.copy()
    q[:2] = ROutEl1 @ q[:2]
    q += r01
    q -= r02
    q[:2] = RInEl2 @ q[:2]
    p[:2] = ROutEl1 @ p[:2]
    p[:2] = RInEl2 @ p[:2]
    return q, p

@numba.njit()
def multi_step_verlet(qEln, pEln, T, T_max, h, force_func):
    # pylint: disable = E, W, R, C
    # collisionRate = 0.0 if np.isnan(collision_params[0]) else collision_params[1]
    x, y, z = qEln
    px, py, pz = pEln
    Fx, Fy, Fz = force_func(x, y, z)
    Fz = Fz - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always
    if isnan(Fx) or T >= T_max:
        is_particle_clipped = True
        qEln, pEln = (x, y, z), (px, py, pz)
        return qEln, qEln, pEln, T, is_particle_clipped
    is_particle_clipped = False
    while True:
        if T >= T_max:
            p_el, q_el = (px, py, pz), (x, y, z)
            return q_el, q_el, p_el, T, is_particle_clipped
        x = x + px * h + .5 * Fx * h ** 2
        y = y + py * h + .5 * Fy * h ** 2
        z = z + pz * h + .5 * Fz * h ** 2

        Fx_n, Fy_n, Fz_n = force_func(x, y, z)
        Fz_n = Fz_n - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always

        if isnan(Fx_n):
            xo = x - (px * h + .5 * Fx * h ** 2)
            yo = y - (py * h + .5 * Fy * h ** 2)
            zo = z - (pz * h + .5 * Fz * h ** 2)
            p_el, q_el, qEl_o = (px, py, pz), (x, y, z), (xo, yo, zo)
            is_particle_clipped = True
            return q_el, qEl_o, p_el, T, is_particle_clipped
        px = px + .5 * (Fx_n + Fx) * h
        py = py + .5 * (Fy_n + Fy) * h
        pz = pz + .5 * (Fz_n + Fz) * h
        Fx, Fy, Fz = Fx_n, Fy_n, Fz_n
        T += h
        # if collisionRate!=0.0 and np.random.rand() < h * collisionRate:
        #     px, py, pz = post_collision_momentum((px, py, pz), (x, y, z), collision_params)
        
        
@numba.njit()
def multi_step_verlet_with_logging(qEln, pEln, T, T_max, h, force_func):
    # pylint: disable = E, W, R, C
    # collisionRate = 0.0 if np.isnan(collision_params[0]) else collision_params[1]
    x, y, z = qEln
    px, py, pz = pEln
    Fx, Fy, Fz = force_func(x, y, z)
    Fz = Fz - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always
    q_vals,p_vals=[[x,y,z]],[[px,py,pz]]
    if isnan(Fx) or T >= T_max:
        is_particle_clipped = True
        qEln, pEln = (x, y, z), (px, py, pz)
        return qEln, qEln, pEln, T, is_particle_clipped,q_vals,p_vals
    is_particle_clipped = False
    while True:
        if T >= T_max:
            p_el, q_el = (px, py, pz), (x, y, z)
            return q_el, q_el, p_el, T, is_particle_clipped,q_vals,p_vals
        x = x + px * h + .5 * Fx * h ** 2
        y = y + py * h + .5 * Fy * h ** 2
        z = z + pz * h + .5 * Fz * h ** 2


        Fx_n, Fy_n, Fz_n = force_func(x, y, z)
        Fz_n = Fz_n - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always

        if isnan(Fx_n):
            xo = x - (px * h + .5 * Fx * h ** 2)
            yo = y - (py * h + .5 * Fy * h ** 2)
            zo = z - (pz * h + .5 * Fz * h ** 2)
            p_el, q_el, qEl_o = (px, py, pz), (x, y, z), (xo, yo, zo)
            is_particle_clipped = True
            return q_el, qEl_o, p_el, T, is_particle_clipped,q_vals,p_vals
        px = px + .5 * (Fx_n + Fx) * h
        py = py + .5 * (Fy_n + Fy) * h
        pz = pz + .5 * (Fz_n + Fz) * h
        q_vals.append([x,y,z])
        p_vals.append([px,py,pz])
        Fx, Fy, Fz = Fx_n, Fy_n, Fz_n
        T += h
        # if collisionRate!=0.0 and np.random.rand() < h * collisionRate:
        #     px, py, pz = post_collision_momentum((px, py, pz), (x, y, z), collision_params)