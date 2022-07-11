import numba
import numpy as np

from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, interp2D, scalar_interp3D
from numbaFunctionsAndObjects.utilities import tupleOf3Floats


# todo: refactor the interpolation data stuff


@numba.njit(cache=False)
def is_coord_in_vacuum(x: float, y: float, z: float, params) -> bool:
    L, ap, L_cap, extra_field_length, field_fact, magnetImperfections = params
    return 0 <= x <= L and y ** 2 + z ** 2 < ap ** 2


@numba.njit(cache=False)
def _force_Func_Inner(y: float, z: float, field_data) -> tupleOf3Floats:
    """Wrapper for interpolation of force fields of plane at center lens. see self.force"""
    y_arr, z_arr, FyArr, Fz_arr, V_arr = field_data
    Fx = 0.0
    Fy = interp2D(y, z, y_arr, z_arr, FyArr)
    Fz = interp2D(y, z, y_arr, z_arr, Fz_arr)
    return Fx, Fy, Fz


@numba.njit(cache=False)
def _force_Func_Outer(x, y, z, field_data) -> tupleOf3Floats:
    """Wrapper for interpolation of force fields at ends of lens. see force"""
    x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = field_data
    Fx, Fy, Fz = vec_interp3D(x, y, z, x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr)
    return Fx, Fy, Fz


@numba.njit(cache=False)
def _magnetic_potential_Func_Fringe(x: float, y: float, z: float, field_data) -> float:
    """Wrapper for interpolation of magnetic fields at ends of lens. see magnetic_potential"""
    x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = field_data
    V = scalar_interp3D(x, y, z, x_arr, y_arr, z_arr, V_arr)
    return V


@numba.njit(cache=False)
def _magnetic_potential_Func_Inner(x: float, y: float, z: float, field_data) -> float:
    """Wrapper for interpolation of magnetic fields of plane at center lens.see magnetic_potential"""
    y_arr, z_arr, FyArr, Fz_arr, V_arr = field_data
    V = interp2D(y, z, y_arr, z_arr, V_arr)
    return V


@numba.njit(cache=False)
def force(x: float, y: float, z: float, params, field_data) -> tupleOf3Floats:
    L, ap, L_cap, extra_field_length, field_fact, magnetImperfections = params
    field_data_3D, field_data_2D, field_data_perturbations = field_data
    Fx, Fy, Fz = _force(x, y, z, params, field_data_2D, field_data_3D)
    if magnetImperfections and not np.isnan(Fx):
        delta_Fx, delta_Fy, delta_Fz = _force_Func_Outer(x, y, z, field_data_perturbations)
        Fx += delta_Fx
        Fy += delta_Fy
        Fz += delta_Fz
    return Fx, Fy, Fz


@numba.njit(cache=False)
def _force(x: float, y: float, z: float, params, field_data_2D, field_data_3D) -> tupleOf3Floats:
    """
    Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper

    Symmetry is used to simplify the computation of force. Either end of the lens is identical, so coordinates
    falling within some range are mapped to an interpolation of the force field at the lenses end. If the lens is
    long enough, the inner region is modeled as a single plane as well. (nan,nan,nan) is returned if coordinate
    is outside vacuum tube

    :param x: x cartesian coordinate, m
    :param y: y cartesian coordinate, m
    :param z: z cartesian coordinate, m
    :return: tuple of length 3 of the force vector, simulation units. contents are nan if coordinate is outside
    vacuum
    """
    L, ap, L_cap, extra_field_length, field_fact, magnetImperfections = params

    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan, np.nan, np.nan
    # x, y, z = baseClass.misalign_Coords(x, y, z)
    FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
    FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
    y = abs(y)  # confine to upper right quadrant
    z = abs(z)
    if -extra_field_length <= x <= L_cap:  # at beginning of lens
        Fx, Fy, Fz = _force_Func_Outer(x, y, z, field_data_3D)
    elif L_cap < x <= L - L_cap:  # if long enough, model interior as uniform in x
        Fx, Fy, Fz = _force_Func_Inner(y, z, field_data_2D)
    elif L - L_cap <= x <= L + extra_field_length:  # at end of lens
        x = L - x
        Fx, Fy, Fz = _force_Func_Outer(x, y, z, field_data_3D)
        Fx = -Fx
    else:
        raise Exception("Particle outside field region")  # this may be triggered when itentionally misligned
    Fx *= field_fact
    Fy *= FySymmetryFact * field_fact
    Fz *= FzSymmetryFact * field_fact
    # Fx, Fy, Fz = baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
    return Fx, Fy, Fz


@numba.njit(cache=False)
def magnetic_potential(x: float, y: float, z: float, params, field_data) -> float:
    L, ap, L_cap, extra_field_length, field_fact, magnetImperfections = params
    field_data_3D, field_data_2D, field_data_perturbations = field_data
    V = _magnetic_potential(x, y, z, params, field_data_2D, field_data_3D)
    if magnetImperfections and not np.isnan(V):
        delta_V = _magnetic_potential_Func_Fringe(x, y, z, field_data_perturbations)
        V += delta_V

    return V


@numba.njit(cache=False)
def _magnetic_potential(x: float, y: float, z: float, params, field_data_2D, field_data_3D) -> float:
    """
    Magnetic potential energy of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper

    Symmetry if used to simplify the computation of potential. Either end of the lens is identical, so coordinates
    falling within some range are mapped to an interpolation of the potential at the lenses end. If the lens is
    long enough, the inner region is modeled as a single plane as well. nan is returned if coordinate
    is outside vacuum tube

    """
    L, ap, L_cap, extra_field_length, field_fact, magnetImperfections = params
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan
    # x, y, z = baseClass.misalign_Coords(x, y, z)
    y = abs(y)
    z = abs(z)
    if -extra_field_length <= x <= L_cap:
        V0 = _magnetic_potential_Func_Fringe(x, y, z, field_data_3D)
    elif L_cap < x <= L - L_cap:
        V0 = _magnetic_potential_Func_Inner(x, y, z, field_data_2D)
    elif 0 <= x <= L + extra_field_length:
        x = L - x
        V0 = _magnetic_potential_Func_Fringe(x, y, z, field_data_3D)
    else:
        raise Exception("Particle outside field region")
    V0 *= field_fact
    return V0

# @numba.njit(cache=False)
# def _force_Field_Perturbations(x0: float, y0: float, z0: float, params, field_data) -> tupleOf3Floats:
#     L, ap, L_cap, extra_field_length, field_fact = params
#     if not is_coord_in_vacuum(x0, y0, z0, L, ap):
#         return np.nan, np.nan, np.nan
#     # x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
#     x, y, z = x0, y0, z0
#     x = x - L / 2
#     Fx, Fy, Fz = _force_Func_Outer(x, y, z, field_data, fieldPerturbationData,
#                                    useImperfectInterp=True)  # being used to hold fields for entire lens
#     Fx = Fx * field_fact
#     Fy = Fy * field_fact
#     Fz = Fz * field_fact
#     # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
#     return Fx, Fy, Fz
# @numba.njit(cache=False)
# def force( x: float, y: float, z: float,params,field_data) -> tupleOf3Floats:
#     """Force on lithium atom. Functions to combine perfect force and extra force from imperfections.
#      Perturbation force is messed up force minus perfect force."""
# fieldPerturbationData = fieldPerturbationData if fieldPerturbationData is not None else nanArr7Tuple
# Fx, Fy, Fz = _force(x, y, z,params,field_data)
# if use_field_perturbations:
#     deltaFx, deltaFy, deltaFz = _force_Field_Perturbations(x, y,z,params,field_data,fieldPerturbationData)
#     Fx, Fy, Fz = Fx + deltaFx, Fy + deltaFy, Fz + deltaFz
# return Fx, Fy, Fz
