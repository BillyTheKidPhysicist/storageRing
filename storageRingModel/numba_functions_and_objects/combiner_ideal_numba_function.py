"""
Contains Numba functions for use with corresponding element. These are called by attributes of the element, and used in
the fast time stepping method in particle_tracer.py
"""
import numba
import numpy as np

from constants import SIMULATION_MAGNETON
from numba_functions_and_objects.utilities import eps, eps_fact


@numba.njit()
def combiner_Ideal_Force(x, y, z, Lm, c1, c2) -> tuple[float, float, float]:
    Fx, Fy, Fz = 0.0, 0.0, 0.0
    if 0 < x < Lm:
        B0 = np.sqrt((c2 * z) ** 2 + (c1 + c2 * y) ** 2)
        Fy = SIMULATION_MAGNETON * c2 * (c1 + c2 * y) / B0
        Fz = SIMULATION_MAGNETON * c2 ** 2 * z / B0
    return Fx, Fy, Fz


@numba.njit()
def force(x, y, z, params):
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan, np.nan, np.nan
    else:
        return force_without_isinside_check(x, y, z, params)


@numba.njit()
def force_without_isinside_check(x, y, z, params):
    c1, c2, ang, La, Lb, apz, ap_left, ap_right, field_fact = params
    Fx, Fy, Fz = combiner_Ideal_Force(x, y, z, Lb, c1, c2)
    Fx *= field_fact
    Fy *= field_fact
    Fz *= field_fact
    return Fx, Fy, Fz


@numba.njit()
def magnetic_potential(x, y, z, params):
    c1, c2, ang, La, Lb, apz, ap_left, ap_right, field_fact = params
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan
    if 0 < x < Lb:
        V0 = SIMULATION_MAGNETON * np.sqrt((c2 * z) ** 2 + (c1 + c2 * y) ** 2)
    else:
        V0 = 0.0
    V0 *= field_fact
    return V0


@numba.njit()
def is_coord_in_vacuum(x, y, z, params):
    c1, c2, ang, La, Lb, apz, ap_left, ap_right, field_fact = params
    if not -apz < z < apz:  # if outside the z apeture (vertical)
        return False
    elif -eps <= x <= Lb * eps_fact:  # particle is in the horizontal section (in element frame) that passes
        # through the combiner. Simple square apeture
        if -ap_left < y < ap_right and -apz < z < apz:  # if inside the y (width) apeture
            return True
        else:
            return False
    elif x < 0:
        return False
    else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
        m = np.tan(ang)
        Y1 = m * x + (ap_right - m * Lb)  # upper limit
        Y2 = (-1 / m) * x + La * np.sin(ang) + (Lb + La * np.cos(ang)) / m
        Y3 = m * x + (-ap_left - m * Lb)
        if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
            return True
        elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
            return True
        else:
            return False
