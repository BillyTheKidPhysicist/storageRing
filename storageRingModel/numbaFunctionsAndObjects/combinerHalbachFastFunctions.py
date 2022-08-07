import numba
import numpy as np

from constants import FLAT_WALL_VACUUM_THICKNESS
from numbaFunctionsAndObjects.interpFunctions import magnetic_potential_interp_3D,force_interp_3D


@numba.njit()
def force(x0, y0, z0, params, field_data):
    # this function uses the symmetry of the combiner to extract the force everywhere.
    # I believe there are some redundancies here that could be trimmed to save time.
    # x, y, z = baseClass.misalign_Coords(x0, y0, z0)
    if not is_coord_in_vacuum(x0, y0, z0, params):
        return np.nan, np.nan, np.nan
    ap, Lm, La, Lb, space, ang, acceptance_width, field_fact, use_symmetry = params
    field_data_internal, field_data_external, field_data_full  = field_data
    x, y, z = x0, y0, z0
    symmetry_plane_x = Lm / 2 + space  # field symmetry plane location
    if use_symmetry:
        Fy_symmetry_fact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
        Fz_symmetry_fact = 1.0 if z >= 0.0 else -1.0
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)

        if 0.0 <= x <= space:
            # print(x,y,z,Lm,space)
            Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_external)
        elif space < x <= symmetry_plane_x:
            Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_internal)
        elif symmetry_plane_x < x <= Lm + space:
            x = 2 * symmetry_plane_x - x
            Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_internal)
            Fx = -Fx
        elif space + Lm < x:
            x = 2 * symmetry_plane_x - x
            Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_external)
            Fx = -Fx
        else:
            print(x, y, z, Lm, space)
            raise ValueError
        Fy = Fy * Fy_symmetry_fact
        Fz = Fz * Fz_symmetry_fact
    else:
        Fx, Fy, Fz = force_interp_3D(x, y, z, field_data_full)
    Fx *= field_fact
    Fy *= field_fact
    Fz *= field_fact
    return Fx, Fy, Fz


@numba.njit()
def magnetic_potential(x, y, z, params, field_data):
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan
    # x, y, z = baseClass.misalign_Coords(x, y, z)
    ap, Lm, La, Lb, space, ang, acceptance_width, field_fact, use_symmetry = params
    field_data_internal, field_data_external, field_data_full  = field_data

    if use_symmetry:
        symmetry_plane_x = Lm / 2 + space  # field symmetry plane location
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if 0.0 <= x <= space:
            V = magnetic_potential_interp_3D(x, y, z, field_data_external)
        elif space < x <= symmetry_plane_x:
            V = magnetic_potential_interp_3D(x, y, z, field_data_internal)
        elif symmetry_plane_x < x <= Lm + space:
            x = 2 * symmetry_plane_x - x
            V = magnetic_potential_interp_3D(x, y, z, field_data_internal)
        elif Lm + space < x:  # particle can extend past 2*symmetryPlaneX
            x = 2 * symmetry_plane_x - x
            V = magnetic_potential_interp_3D(x, y, z, field_data_external)
        else:
            print(x, y, z, Lm, space)
            raise ValueError
    else:
        V = magnetic_potential_interp_3D(x, y, z, field_data_full)
    V = V * field_fact
    return V


@numba.njit()
def is_coord_in_vacuum(x, y, z, params):
    # q: coordinate to test in element's frame
    ap, Lm, La, Lb, space, ang, acceptance_width, field_fact, use_symmetry = params
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
