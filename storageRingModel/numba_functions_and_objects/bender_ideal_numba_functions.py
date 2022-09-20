"""
Contains Numba functions for use with corresponding element. These are called by attributes of the element, and used in
the fast time stepping method in particle_tracer.py
"""

import numba
import numpy as np

from numba_functions_and_objects.utilities import full_arctan2, eps_fact


@numba.njit()
def magnetic_potential(x, y, z, params):
    rb, ap, ang, K, field_fact = params
    phi = full_arctan2(y, x)
    r_polar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
    r_toroidal = np.sqrt((r_polar - rb) ** 2 + z ** 2)
    if phi <= ang * eps_fact and r_toroidal < ap:
        V0 = .5 * K * r_toroidal ** 2
    else:
        V0 = np.nan
    V0 *= field_fact
    return V0


@numba.njit()
def force(x, y, z, params):
    rb, ap, ang, K, field_fact = params
    phi = full_arctan2(y, x)
    r_polar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
    r_toroidal = np.sqrt((r_polar - rb) ** 2 + z ** 2)
    if phi <= ang * eps_fact and r_toroidal < ap:
        F0 = -K * (r_polar - rb)  # force in x y plane
        Fx = np.cos(phi) * F0
        Fy = np.sin(phi) * F0
        Fz = -K * z
    else:
        Fx, Fy, Fz = np.nan, np.nan, np.nan
    Fx *= field_fact
    Fy *= field_fact
    Fz *= field_fact
    return Fx, Fy, Fz


@numba.njit()
def is_coord_in_vacuum(x, y, z, params):
    rb, ap, ang, K, field_fact = params
    phi = full_arctan2(y, x)
    if phi <= ang * eps_fact:  # if particle is in bending segment
        rh = np.sqrt(x ** 2 + y ** 2) - rb  # horizontal radius
        r = np.sqrt(rh ** 2 + z ** 2)  # particle displacement from center of apeture
        return r < ap
    else:
        return False
