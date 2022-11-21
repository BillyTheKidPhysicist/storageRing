"""
Functions to model the effects of collisions between lithium atoms. Current version of the model is rather simple and
described in electronic lab book. Briefly, the model assumptions are:

- lithium is uniformly distributed in magnets up to .7rp
- relative velocity comes from a combination of a minimum "geometric" value and from a thermal distirbution
- effect is only modeled in lens and arc section of bender to keep things simple

"""
# pylint: disable=too-many-locals, too-many-arguments

import numba
import numpy as np

from constants import MASS_LITHIUM_7, BOLTZMANN_CONSTANT
from helper_tools import full_arctan2
# import latticeElements.elementPT
from lattice_elements.elements import HalbachLensSim, Drift, BenderSim, LensIdeal, BenderIdeal

full_arctan2 = numba.njit(full_arctan2)


@numba.njit()
def collision_partner_momentum_bender(q_el, nominal_speed: float, T: float):
    """Calculate a collision partner's momentum for colliding with lithium traveling in the bender. The collision
    partner is sampled assuming a random gas with nominal speeds in the bender given by geometry and angular momentum.
    """
    delta_pso, pxo, pyo = momentum_sample_3D(T)
    pso = nominal_speed + delta_pso
    theta = full_arctan2(q_el[1], q_el[0])
    px = pxo * np.cos(theta) + pso * np.sin(theta)
    py = pxo * np.sin(theta) - pso * np.cos(theta)
    pz = pyo
    p_collision = (px, py, pz)
    return p_collision


@numba.njit()
def momentum_sample_3D(T: float):
    """Sample momentum in 3D based on temperature T"""
    sigma = np.sqrt(BOLTZMANN_CONSTANT * T / MASS_LITHIUM_7)
    pi, pj, pk = np.random.normal(loc=0.0, scale=sigma, size=3)
    return pi, pj, pk


def generate_collision_params(particle_tracer):
    element = particle_tracer.current_el
    if type(element) in (Drift, LensIdeal, HalbachLensSim):
        which_el_flag = 0
    elif type(element) in (BenderSim, BenderIdeal):
        which_el_flag = 1
    else:
        which_el_flag = -1
    nominal_speed = particle_tracer.PTL.design_speed
    params = np.array([particle_tracer.collision_rate, particle_tracer.temperature,
                       nominal_speed, which_el_flag])
    return params


@numba.njit()
def is_collision(h, collision_params):
    collision_rate, _, _, _ = collision_params
    col_prob = h * collision_rate
    assert col_prob < .1
    return np.random.rand() < col_prob


@numba.njit()
def post_collision_momentum(p, q, collision_params):
    """Get the momentum after a collision. The collision partner momentum is generated, and then Jeremy's collision
    algorithm is applied to find the new momentum. There is some wonkiness here from using numba"""
    collision_rate, temperature, nominal_speed, which_el_flag = collision_params
    if which_el_flag == 0:
        dpx, dpy, dpz = momentum_sample_3D(temperature)
        p_col_partner = (nominal_speed + dpx, dpy, dpz)
        p_new = collision(*p, *p_col_partner)
    elif which_el_flag == 1:
        theta = full_arctan2(q[1], q[0])
        raise NotImplementedError #the bender angle needs to be included
        if 0 < theta < np.pi * .9:
            p_col_partner = collision_partner_momentum_bender(q, nominal_speed, temperature)
            p_new = collision(*p, *p_col_partner)
        else:
            p_new = p
    elif which_el_flag == -1:
        p_new = p
    else:
        raise NotImplementedError
    return p_new


@numba.njit()
def vel_comp_after_collision(v_rel):
    """Provided by Jeremy"""
    cos_theta = 2 * np.random.random() - 1
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = 2 * np.pi * np.random.random()

    vx_final = v_rel * sin_theta * np.cos(phi)
    vy_final = v_rel * sin_theta * np.sin(phi)
    vz_final = v_rel * cos_theta

    return vx_final, vy_final, vz_final


@numba.njit()
def collision(p1_vx, p1_vy, p1_vz, p2_vx, p2_vy, p2_vz):
    """ Elastic collision of two particles with random scattering angle phi and theta. Inputs are the two particles
        x,y,z components of their velocity. Output is the particles final velocity components. Output coordinate
        system matches whatever is used as the input so long as it's cartesian."""
    vx_cm = 0.5 * (p1_vx + p2_vx)
    vy_cm = 0.5 * (p1_vy + p2_vy)
    vz_cm = 0.5 * (p1_vz + p2_vz)

    v_rel = np.sqrt((p1_vx - p2_vx) ** 2 + (p1_vy - p2_vy) ** 2 + (p1_vz - p2_vz) ** 2)

    vx_final, vy_final, vz_final = vel_comp_after_collision(v_rel)

    p1_vx_final = vx_cm + 0.5 * vx_final
    p1_vy_final = vy_cm + 0.5 * vy_final
    p1_vz_final = vz_cm + 0.5 * vz_final

    return p1_vx_final, p1_vy_final, p1_vz_final
