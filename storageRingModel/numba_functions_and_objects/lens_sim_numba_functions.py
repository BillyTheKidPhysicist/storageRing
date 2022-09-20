"""
Contains Numba functions for use with corresponding element. These are called by attributes of the element, and used in
the fast time stepping method in particle_tracer.py
"""
import numba
import numpy as np

from numba_functions_and_objects.interpFunctions import magnetic_potential_interp_3D, force_interp_3D, \
    magnetic_potential_interp_2D, force_interp_2D
from numba_functions_and_objects.utilities import TupleOf3Floats, eps, eps_fact


@numba.njit()
def is_coord_in_vacuum(x: float, y: float, z: float, params) -> bool:
    L, ap, L_cap, field_fact, use_symmetry = params
    return -eps <= x <= L * eps_fact and y ** 2 + z ** 2 < ap ** 2


@numba.njit()
def force(x0: float, y0: float, z0: float, params, field_data) -> TupleOf3Floats:
    """
    Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper.
    Symmetry is used to simplify the computation of force. Either end of the lens is identical, so coordinates
    falling within some range are mapped to an interpolation of the force field at the lenses end. If the lens is
    long enough, the inner region is modeled as a single plane as well. (nan,nan,nan) is returned if coordinate
    is outside vacuum tube
    """
    field_data_2D, field_data_3D = field_data
    L, ap, L_cap, field_fact, use_symmetry = params

    if not is_coord_in_vacuum(x0, y0, z0, params):
        return np.nan, np.nan, np.nan
    if use_symmetry:
        x = x0
        y = abs(y0)  # confine to upper right quadrant
        z = abs(z0)
        if -eps <= x <= L_cap * eps_fact:  # at beginning of lens
            Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_3D)
        elif L_cap < x <= L - L_cap:  # if long enough, model interior as uniform in x
            Fx, Fy, Fz = force_interp_2D(y, z, field_data_2D)
        elif L - L_cap <= x <= L * eps_fact:  # at end of lens
            x = L - x
            Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_3D)
            Fx = -Fx
        else:
            raise Exception("Particle outside field region")  # this may be triggered when itentionally misligned
        Fy_symmetry_fact = 1.0 if y0 >= 0.0 else -1.0  # take advantage of symmetry
        Fz_symmetry_fact = 1.0 if z0 >= 0.0 else -1.0
        Fx *= field_fact
        Fy *= Fy_symmetry_fact * field_fact
        Fz *= Fz_symmetry_fact * field_fact
    else:
        Fx, Fy, Fz = force_interp_3D(x0, y0, z0, field_data_3D)

    return Fx, Fy, Fz


@numba.njit()
def magnetic_potential(x0: float, y0: float, z0: float, params, field_data) -> float:
    """
    Magnetic potential energy of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper

    Symmetry if used to simplify the computation of potential. Either end of the lens is identical, so coordinates
    falling within some range are mapped to an interpolation of the potential at the lenses end. If the lens is
    long enough, the inner region is modeled as a single plane as well. nan is returned if coordinate
    is outside vacuum tube

    """
    field_data_2D, field_data_3D = field_data
    L, ap, L_cap, field_fact, use_symmetry = params
    if not is_coord_in_vacuum(x0, y0, z0, params):
        return np.nan
    if use_symmetry:
        x = x0
        y = abs(y0)
        z = abs(z0)
        if -eps <= x <= L_cap:
            V0 = magnetic_potential_interp_3D(x, y, z, field_data_3D)
        elif L_cap < x <= L - L_cap:
            V0 = magnetic_potential_interp_2D(x, y, z, field_data_2D)
        elif 0 <= x <= L * eps_fact:
            x = L - x
            V0 = magnetic_potential_interp_3D(x, y, z, field_data_3D)
        else:
            raise Exception("Particle outside field region")
    else:
        V0 = magnetic_potential_interp_3D(x0, y0, z0, field_data_3D)
    V0 *= field_fact
    return V0
