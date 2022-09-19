import numba
import numpy as np

from numba_functions_and_objects.interpFunctions import vec_interp3D, scalar_interp3D
from numba_functions_and_objects.utilities import full_arctan2,eps

@numba.njit()
def cartesian_To_Center(x, y, z, params):
    """Convert from cartesian coords to HalbachLensClass.BenderSim coored, ie "center coords" for
    evaluation by interpolator"""
    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_symmetry = params

    if x > 0.0 and -L_cap <= y <= 0.0:
        s = L_cap + y
        xc = x - rb
        yc = z
    else:
        theta = full_arctan2(y, x)
        if theta <= ang:
            s = theta * rb + L_cap
            xc = np.sqrt(x ** 2 + y ** 2) - rb
            yc = z
        elif ang < theta <= 2 * np.pi:  # i'm being lazy here and not limiting the real end
            x0, y0 = np.cos(ang) * rb, np.sin(ang) * rb
            theta_end_perp = np.pi - np.arctan(-1 / np.tan(ang))
            x, y = x - x0, y - y0
            deltaS, xc = np.cos(theta_end_perp) * x + np.sin(-theta_end_perp) * y, np.sin(theta_end_perp) * x + np.cos(
                theta_end_perp) * y
            yc = z
            xc = -xc
            s = (ang * rb + L_cap) + deltaS
        else:
            raise ValueError
    return s, xc, yc


@numba.njit()
def _force_func(x, y, z, field_data_seg):
    Fx, Fy, Fz = vec_interp3D(x, y, z, *field_data_seg[:6])
    return Fx, Fy, Fz


@numba.njit()
def _magnetic_potential_func(x, y, z, field_data_seg):
    return scalar_interp3D(x, y, z, *field_data_seg[:3], field_data_seg[-1])


@numba.njit()
def transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(Fx, Fy, Fz, x, y, ucAng, M_uc):
    phi = full_arctan2(y, x)  # calling a fast numba version that is global
    cellNum = int(phi / ucAng) + 1  # cell number that particle is in, starts at one
    if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
        rot_angle = 2 * (cellNum // 2) * ucAng
    else:  # otherwise it needs to be reflected. This is the algorithm for reflections
        Fx0 = Fx
        Fy0 = Fy
        Fx = M_uc[0, 0] * Fx0 + M_uc[0, 1] * Fy0
        Fy = M_uc[1, 0] * Fx0 + M_uc[1, 1] * Fy0
        rot_angle = 2 * ((cellNum - 1) // 2) * ucAng
    Fx0 = Fx
    Fy0 = Fy
    Fx = np.cos(rot_angle) * Fx0 - np.sin(rot_angle) * Fy0
    Fy = np.sin(rot_angle) * Fx0 + np.cos(rot_angle) * Fy0
    return Fx, Fy, Fz


@numba.njit()
def transform_Element_Coords_Into_Unit_Cell_Frame(x, y, z, ang, ucAng):
    phi = ang - full_arctan2(y, x)
    revs = int(phi / ucAng)  # number of revolutions through unit cell
    if revs % 2 == 0:  # if even
        theta = phi - ucAng * revs
    else:  # if odd
        theta = ucAng - (phi - ucAng * revs)
    r = np.sqrt(x ** 2 + y ** 2)
    x = r * np.cos(theta)  # cartesian coords in unit cell frame
    y = r * np.sin(theta)  # cartesian coords in unit cell frame
    return x, y, z


@numba.njit()
def is_coord_in_vacuum(x, y, z, params):
    phi = full_arctan2(y, x)  

    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_symmetry = params
    if phi <= ang+eps:  # if particle is inside bending angle region
        r_minor=np.sqrt((np.sqrt(x ** 2 + y ** 2) - rb) ** 2 + z ** 2)
        return r_minor< ap 
    else:  # if outside bender's angle range
        r_minor_east=np.sqrt((x - rb) ** 2 + z ** 2)
        if r_minor_east <= ap  and (eps >= y >= -L_cap-eps):  # If inside the cap on
            # eastward side
            return True
        else:
            x_rot = RIn_Ang[0, 0] * x + RIn_Ang[0, 1] * y
            y_rot = RIn_Ang[1, 0] * x + RIn_Ang[1, 1] * y
            r_minor_west = np.sqrt((x_rot - rb) ** 2 + z ** 2)
            return r_minor_west <= ap  and (L_cap +eps>= y_rot >= -eps)
            # if on the westwards side


@numba.njit()
def magnetic_potential(x0, y0, z0, params, field_data):
    # magnetic potential at point q in element frame
    # q: particle's position in element frame
    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_symmetry = params
    field_data_seg, field_data_internal, field_data_cap, field_data_full = field_data

    if not is_coord_in_vacuum(x0, y0, z0, params):
        return np.nan
    if use_symmetry:
        x, y = x0, y0
        z = abs(z0)
        phi = full_arctan2(y, x)  # calling a fast numba version that is global
        if phi < ang:  # if particle is inside bending angle region
            revs = int((ang - phi) / ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position = 'FIRST'
            elif revs == num_magnets * 2 - 1 or revs == num_magnets * 2 - 2:
                position = 'LAST'
            else:
                position = 'INNER'
            if position == 'INNER':
                quc = transform_Element_Coords_Into_Unit_Cell_Frame(x, y, z, ang, ucAng)  # get unit cell coords
                V0 = _magnetic_potential_func(quc[0], quc[1], quc[2], field_data_seg)
            elif position == 'FIRST' or position == 'LAST':
                V0 = magnetic_potential_First_And_Last(x, y, z, position, M_ang, field_data_internal)
            else:
                V0 = np.nan
        elif phi > ang:  # if outside bender's angle range
            if (rb - ap < x < rb + ap) and (0 > y > -L_cap):  # If inside the cap on
                # eastward side
                V0 = _magnetic_potential_func(x, y, z, field_data_cap)
            else:
                xTest = RIn_Ang[0, 0] * x + RIn_Ang[0, 1] * y
                yTest = RIn_Ang[1, 0] * x + RIn_Ang[1, 1] * y
                if (rb - ap < xTest < rb + ap) and (
                        L_cap > yTest > 0):  # if on the westwards side
                    yTest = -yTest
                    V0 = _magnetic_potential_func(xTest, yTest, z, field_data_cap)
                else:  # if not in either cap
                    V0 = np.nan
    else:
        s, xc, yc = cartesian_To_Center(x0, y0, z0, params)
        V0 = _magnetic_potential_func(s, xc, yc, field_data_full)
    V0 *= field_fact
    return V0


@numba.njit()
def magnetic_potential_First_And_Last(x, y, z, position, M_ang, field_data_internal):
    if position == 'FIRST':
        xNew = M_ang[0, 0] * x + M_ang[0, 1] * y
        yNew = M_ang[1, 0] * x + M_ang[1, 1] * y
        V0 = _magnetic_potential_func(xNew, yNew, z, field_data_internal)
    elif position == 'LAST':
        V0 = _magnetic_potential_func(x, y, z, field_data_internal)
    else:
        raise ValueError('Invalid position')
    return V0


@numba.njit()
def force(x0, y0, z0, params, field_data):
    # force at point q in element frame
    # q: particle's position in element frame

    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_symmetry = params
    field_data_seg, field_data_internal, field_data_cap, field_data_full = field_data

    if use_symmetry:
        Fz_symmetry_fact = 1.0 if z0 >= 0.0 else -1.0
        x, y = x0, y0
        z = abs(z0)
        phi = full_arctan2(y, x)  # calling a fast numba version that is global
        if phi <= ang:  # if particle is inside bending angle region
            rXYPlane = np.sqrt(x ** 2 + y ** 2)  # radius in xy plane
            if np.sqrt((rXYPlane - rb) ** 2 + z ** 2) < ap:
                psi = ang - phi
                revs = int(psi / ucAng)  # number of revolutions through unit cell
                if revs == 0 or revs == 1:
                    position = 'FIRST'
                elif revs == num_magnets * 2 - 1 or revs == num_magnets * 2 - 2:
                    position = 'LAST'
                else:
                    position = 'INNER'
                if position == 'INNER':
                    if revs % 2 == 0:  # if even
                        theta = psi - ucAng * revs
                    else:  # if odd
                        theta = ucAng - (psi - ucAng * revs)
                    xuc = rXYPlane * np.cos(theta)  # cartesian coords in unit cell frame
                    yuc = rXYPlane * np.sin(theta)  # cartesian coords in unit cell frame
                    Fx, Fy, Fz = _force_func(xuc, yuc, z, field_data_seg)
                    Fx, Fy, Fz = transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(Fx, Fy, Fz, x, y, ucAng, M_uc)
                else:
                    if position == 'FIRST':
                        x, y = M_ang[0, 0] * x + M_ang[0, 1] * y, M_ang[1, 0] * x + M_ang[1, 1] * y
                        Fx, Fy, Fz = _force_func(x, y, z, field_data_internal)
                        Fx0 = Fx
                        Fy0 = Fy
                        Fx = M_ang[0, 0] * Fx0 + M_ang[0, 1] * Fy0
                        Fy = M_ang[1, 0] * Fx0 + M_ang[1, 1] * Fy0
                    else:
                        Fx, Fy, Fz = _force_func(x, y, z, field_data_internal)
            else:
                Fx, Fy, Fz = np.nan, np.nan, np.nan
        else:  # if outside bender's angle range
            if np.sqrt((x - rb) ** 2 + z ** 2) < ap and (eps >= y >= -L_cap-eps):  # If inside the cap on
                # eastward side
                Fx, Fy, Fz = _force_func(x, y, z, field_data_cap)
            else:
                x, y = M_ang[0, 0] * x + M_ang[0, 1] * y, M_ang[1, 0] * x + M_ang[1, 1] * y
                if np.sqrt((x - rb) ** 2 + z ** 2) < ap and (
                        -L_cap-eps <= y <= eps):  # if on the westwards side
                    Fx, Fy, Fz = _force_func(x, y, z, field_data_cap)
                    Fx0 = Fx
                    Fy0 = Fy
                    Fx = M_ang[0, 0] * Fx0 + M_ang[0, 1] * Fy0
                    Fy = M_ang[1, 0] * Fx0 + M_ang[1, 1] * Fy0
                else:  # if not in either cap, then outside the bender
                    Fx, Fy, Fz = np.nan, np.nan, np.nan
        Fz = Fz * Fz_symmetry_fact
    else:
        if not is_coord_in_vacuum(x0, y0, z0, params):
            Fx, Fy, Fz= np.nan,np.nan,np.nan
        else:
            s, xc, yc = cartesian_To_Center(x0, y0, z0, params)
            Fx, Fy, Fz = _force_func(s, xc, yc, field_data_full)
    Fx *= field_fact
    Fy *= field_fact
    Fz *= field_fact
    return Fx, Fy, Fz
