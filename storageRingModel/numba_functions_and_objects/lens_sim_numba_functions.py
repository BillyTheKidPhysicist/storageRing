import numba
import numpy as np

from numba_functions_and_objects.interpFunctions import magnetic_potential_interp_3D, force_interp_3D, \
    magnetic_potential_interp_2D, force_interp_2D
from numba_functions_and_objects.utilities import TupleOf3Floats


@numba.njit()
def is_coord_in_vacuum(x: float, y: float, z: float, params) -> bool:
    L, ap, L_cap, field_fact, use_only_symmetry = params
    return 0 <= x <= L and y ** 2 + z ** 2 < ap ** 2


@numba.njit()
def force(x: float, y: float, z: float, params, field_data) -> TupleOf3Floats:
    L, ap, L_cap, field_fact, use_only_symmetry = params
    field_data_3D, field_data_2D, field_data_perturbations = field_data
    Fx, Fy, Fz = _force_symmetry(x, y, z, params, field_data_2D, field_data_3D)
    if not use_only_symmetry and not np.isnan(Fx):
        delta_Fx, delta_Fy, delta_Fz = force_interp_3D(x, y, z, field_data_perturbations)
        Fx += delta_Fx
        Fy += delta_Fy
        Fz += delta_Fz
    return Fx, Fy, Fz


@numba.njit()
def _force_symmetry(x: float, y: float, z: float, params, field_data_2D, field_data_3D) -> TupleOf3Floats:
    """
    Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper.
    Symmetry is used to simplify the computation of force. Either end of the lens is identical, so coordinates
    falling within some range are mapped to an interpolation of the force field at the lenses end. If the lens is
    long enough, the inner region is modeled as a single plane as well. (nan,nan,nan) is returned if coordinate
    is outside vacuum tube
    """
    L, ap, L_cap, field_fact, use_only_symmetry = params

    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan, np.nan, np.nan
    # x, y, z = baseClass.misalign_Coords(x, y, z)
    Fy_symmetry_fact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
    Fz_symmetry_fact = 1.0 if z >= 0.0 else -1.0
    y = abs(y)  # confine to upper right quadrant
    z = abs(z)
    if 0.0 <= x <= L_cap:  # at beginning of lens
        Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_3D)
    elif L_cap < x <= L - L_cap:  # if long enough, model interior as uniform in x
        Fx, Fy, Fz = force_interp_2D(y, z, field_data_2D)
    elif L - L_cap <= x <= L:  # at end of lens
        x = L - x
        Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_3D)
        Fx = -Fx
    else:
        raise Exception("Particle outside field region")  # this may be triggered when itentionally misligned
    Fx *= field_fact
    Fy *= Fy_symmetry_fact * field_fact
    Fz *= Fz_symmetry_fact * field_fact
    # Fx, Fy, Fz = baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
    return Fx, Fy, Fz


@numba.njit()
def magnetic_potential(x: float, y: float, z: float, params, field_data) -> float:
    L, ap, L_cap, field_fact, use_only_symmetry = params
    field_data_3D, field_data_2D, field_data_perturbations = field_data
    V = _magnetic_potential_symmetry(x, y, z, params, field_data_2D, field_data_3D)
    if not use_only_symmetry and not np.isnan(V):
        delta_V = magnetic_potential_interp_3D(x, y, z, field_data_perturbations)
        V += delta_V

    return V


@numba.njit()
def _magnetic_potential_symmetry(x: float, y: float, z: float, params, field_data_2D, field_data_3D) -> float:
    """
    Magnetic potential energy of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper

    Symmetry if used to simplify the computation of potential. Either end of the lens is identical, so coordinates
    falling within some range are mapped to an interpolation of the potential at the lenses end. If the lens is
    long enough, the inner region is modeled as a single plane as well. nan is returned if coordinate
    is outside vacuum tube

    """
    L, ap, L_cap, field_fact, use_only_symmetry = params
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan
    # x, y, z = baseClass.misalign_Coords(x, y, z)
    y = abs(y)
    z = abs(z)
    if 0.0 <= x <= L_cap:
        V0 = magnetic_potential_interp_3D(x, y, z, field_data_3D)
    elif L_cap < x <= L - L_cap:
        V0 = magnetic_potential_interp_2D(x, y, z, field_data_2D)
    elif 0 <= x <= L:
        x = L - x
        V0 = magnetic_potential_interp_3D(x, y, z, field_data_3D)
    else:
        raise Exception("Particle outside field region")
    V0 *= field_fact
    return V0

# @numba.njit(cache=False)
# def _force_Field_Perturbations(x0: float, y0: float, z0: float, params, field_data) -> TupleOf3Floats:
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
# def force( x: float, y: float, z: float,params,field_data) -> TupleOf3Floats:
#     """Force on lithium atom. Functions to combine perfect force and extra force from imperfections.
#      Perturbation force is messed up force minus perfect force."""
# fieldPerturbationData = fieldPerturbationData if fieldPerturbationData is not None else nanArr7Tuple
# Fx, Fy, Fz = _force(x, y, z,params,field_data)
# if use_field_perturbations:
#     deltaFx, deltaFy, deltaFz = _force_Field_Perturbations(x, y,z,params,field_data,fieldPerturbationData)
#     Fx, Fy, Fz = Fx + deltaFx, Fy + deltaFy, Fz + deltaFz
# return Fx, Fy, Fz
