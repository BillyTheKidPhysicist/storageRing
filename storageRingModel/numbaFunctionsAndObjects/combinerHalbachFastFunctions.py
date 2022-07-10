import numba
import numpy as np

from constants import FLAT_WALL_VACUUM_THICKNESS
from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, scalar_interp3D

#todo: refactor the field data stuff 


@numba.njit()
def _force_Func_Internal( x, y, z,fieldDataInternal):
    x_arr, y_arr, zArr, FxArr, FyArr, FzArr, V_arr = fieldDataInternal
    Fx, Fy, Fz = vec_interp3D(x, y, z, x_arr, y_arr, zArr, FxArr, FyArr, FzArr)
    return Fx, Fy, Fz
@numba.njit()
def _force_Func_External( x, y, z,fieldDataExternal):
    x_arr, y_arr, zArr, FxArr, FyArr, FzArr, V_arr = fieldDataExternal
    Fx, Fy, Fz = vec_interp3D(x, y, z, x_arr, y_arr, zArr, FxArr, FyArr, FzArr)
    return Fx, Fy, Fz
@numba.njit()
def _magnetic_potential_Func_Internal( x, y, z,fieldDataInternal):
    x_arr, y_arr, zArr, FxArr, FyArr, FzArr, V_arr = fieldDataInternal
    return scalar_interp3D(x, y, z, x_arr, y_arr, zArr, V_arr)
@numba.njit()
def _magnetic_potential_Func_External( x, y, z,fieldDataExternal):
    x_arr, y_arr, zArr, FxArr, FyArr, FzArr, V_arr = fieldDataExternal
    return scalar_interp3D(x, y, z, x_arr, y_arr, zArr, V_arr)

#todo: I think this is unnecesary
@numba.njit()
def force( x, y, z,params,field_data):
    if not is_coord_in_vacuum(x, y, z,params):
        return np.nan, np.nan, np.nan
    else:
        return force_Without_isInside_Check(x, y, z,params,field_data)
@numba.njit()
def force_Without_isInside_Check( x0, y0, z0,params,field_data):
    # this function uses the symmetry of the combiner to extract the force everywhere.
    # I believe there are some redundancies here that could be trimmed to save time.
    # x, y, z = baseClass.misalign_Coords(x0, y0, z0)
    ap, Lm, La,Lb,space,ang,acceptance_width, field_fact, useSymmetry, extra_field_length = params
    fieldDataInternal,fieldDataExternal=field_data
    x, y, z = x0, y0, z0
    symmetryPlaneX = Lm / 2 + space  # field symmetry plane location
    if useSymmetry:
        FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)

        if -extra_field_length <= x <= space:
            # print(x,y,z,Lm,space)
            Fx, Fy, Fz = _force_Func_External(x, y, z,fieldDataExternal)
        elif space < x <= symmetryPlaneX:
            Fx, Fy, Fz = _force_Func_Internal(x, y, z,fieldDataInternal)
        elif symmetryPlaneX < x <= Lm + space:
            x = 2 * symmetryPlaneX - x
            Fx, Fy, Fz = _force_Func_Internal(x, y, z,fieldDataInternal)
            Fx = -Fx
        elif space + Lm < x:
            x = 2 * symmetryPlaneX - x
            Fx, Fy, Fz = _force_Func_External(x, y, z,fieldDataExternal)
            Fx = -Fx
        else:
            print(x, y, z, Lm, space)
            raise ValueError
        Fy = Fy * FySymmetryFact
        Fz = Fz * FzSymmetryFact
    else:
        Fx, Fy, Fz = _force_Func_Internal(x, y, z,fieldDataInternal)
    # Fx, Fy, Fz = baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
    Fx *= field_fact
    Fy *= field_fact
    Fz *= field_fact
    return Fx, Fy, Fz
@numba.njit()
def magnetic_potential( x, y, z,params,field_data):
    if not is_coord_in_vacuum(x, y, z,params):
        return np.nan
    # x, y, z = baseClass.misalign_Coords(x, y, z)
    ap, Lm, La,Lb,space,ang,acceptance_width, field_fact, useSymmetry, extra_field_length = params
    fieldDataInternal, fieldDataExternal = field_data
    y = abs(y)  # confine to upper right quadrant
    z = abs(z)
    symmetryPlaneX = Lm / 2 + space  # field symmetry plane location
    if useSymmetry:
        if -extra_field_length <= x <= space:
            V = _magnetic_potential_Func_External(x, y, z,fieldDataExternal)
        elif space < x <= symmetryPlaneX:
            V = _magnetic_potential_Func_Internal(x, y, z,fieldDataInternal)
        elif symmetryPlaneX < x <= Lm + space:
            x = 2 * symmetryPlaneX - x
            V = _magnetic_potential_Func_Internal(x, y, z,fieldDataInternal)
        elif Lm + space < x:  # particle can extend past 2*symmetryPlaneX
            x = 2 * symmetryPlaneX - x
            V = _magnetic_potential_Func_External(x, y, z,fieldDataExternal)
        else:
            print(x, y, z, Lm, space)
            raise ValueError
    else:
        V = _magnetic_potential_Func_Internal(x, y, z,fieldDataInternal)
    V = V * field_fact
    return V

@numba.njit()
def is_coord_in_vacuum( x, y, z,params):
    # q: coordinate to test in element's frame
    ap, Lm, La,Lb,space,ang,acceptance_width, field_fact, useSymmetry, extra_field_length = params
    standOff = 10e-6  # first valid (non np.nan) interpolation point on face of lens is 1e-6 off the surface of the lens
    assert FLAT_WALL_VACUUM_THICKNESS > standOff
    if not -ap <= z <= ap:  # if outside the z apeture (vertical)
        return False
    elif 0 <= x <= Lb + FLAT_WALL_VACUUM_THICKNESS:  # particle is in the horizontal section (in element frame) that passes
        # through the combiner.
        if np.sqrt(y ** 2 + z ** 2) < ap:
            return True
        else:
            return False
    elif x < 0:
        return False
    else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
        # todo: these should be in functions if they are used elsewhere
        m = np.tan(ang)
        Y1 = m * x + (acceptance_width - m * Lb)  # upper limit
        Y2 = (-1 / m) * x + La * np.sin(ang) + (Lb + La * np.cos(ang)) / m
        Y3 = m * x + (-acceptance_width - m * Lb)
        if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
            return True
        elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
            return True
        else:
            return False
