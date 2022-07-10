import numba
import numpy as np

from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, scalar_interp3D

@numba.njit()
def _force_Func(x, y, z,field_data):
    """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
    x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr,Varr=field_data
    return vec_interp3D(x, y, z, x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr)

@numba.njit()
def _magnetic_potential_Func( x, y, z,field_data):
    x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = field_data
    return scalar_interp3D(x, y, z, x_arr, y_arr, z_arr, V_arr)

@numba.njit()
def force( x, y, z,params,field_data):
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan, np.nan, np.nan
    else:
        return force_Without_isInside_Check(x, y, z,params,field_data)
@numba.njit()
def force_Without_isInside_Check( x, y, z,params, field_data):
    # this function uses the symmetry of the combiner to extract the force everywhere.
    # I believe there are some redundancies here that could be trimmed to save time.
    ang, La, Lb, Lm,apz,apL,apR, space, field_fact = params
    xFact = 1  # value to modify the force based on symmetry
    zFact = 1
    if 0 <= x <= (Lm / 2 + space):  # if the particle is in the first half of the magnet
        if z < 0:  # if particle is in the lower plane
            z = -z  # flip position to upper plane
            zFact = -1  # z force is opposite in lower half
    elif (Lm / 2 + space) < x:  # if the particle is in the last half of the magnet
        x = (Lm / 2 + space) - (x - (Lm / 2 + space))  # use the reflection of the particle
        xFact = -1  # x force is opposite in back plane
        if z < 0:  # if in the lower plane, need to use symmetry
            z = -z
            zFact = -1  # z force is opposite in lower half
    Fx, Fy, Fz = _force_Func(x, y, z,field_data)
    Fx = field_fact * xFact * Fx
    Fy = field_fact * Fy
    Fz = field_fact * zFact * Fz
    return Fx, Fy, Fz

@numba.njit()
def magnetic_potential( x, y, z,params,field_data):
    # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
    ang, La, Lb, Lm,apz,apL,apR, space, field_fact = params
    if 0 <= x <= (Lm / 2 + space):  # if the particle is in the first half of the magnet
        if z < 0:  # if particle is in the lower plane
            z = -z  # flip position to upper plane
    if (Lm / 2 + space) < x:  # if the particle is in the last half of the magnet
        x = (Lm / 2 + space) - (
                x - (Lm / 2 + space))  # use the reflection of the particle
        if z < 0:  # if in the lower plane, need to use symmetry
            z = -z
    return field_fact * _magnetic_potential_Func(x, y, z,field_data)
@numba.njit()
def is_coord_in_vacuum(x, y, z, params) -> bool:
    # q: coordinate to test in element's frame
    ang, La, Lb, Lm,apz,apL,apR, space, field_fact = params
    if not -apz <= z <= apz:  # if outside the z apeture (vertical)
        return False
    elif 0 <= x <= Lb:  # particle is in the horizontal section (in element frame) that passes
        # through the combiner. Simple square apeture
        if -apL < y < apR:  # if inside the y (width) apeture
            return True
        else:
            return False
    elif x < 0:
        return False
    else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
        m = np.tan(ang)
        Y1 = m * x + (apR - m * Lb)  # upper limit
        Y2 = (-1 / m) * x + La * np.sin(ang) + (Lb + La * np.cos(ang)) / m
        Y3 = m * x + (-apL - m * Lb)
        if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
            return True
        elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
            return True
        else:
            return False
