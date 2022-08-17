import numba
import numpy as np

from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, scalar_interp3D
from numbaFunctionsAndObjects.utilities import full_arctan2


@numba.njit()
def cartesian_To_Center(x, y, z, params):
    """Convert from cartesian coords to HalbachLensClass.BenderSim coored, ie "center coords" for
    evaluation by interpolator"""
    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_field_perturbations = params

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
            thetaEndPerp = np.pi - np.arctan(-1 / np.tan(ang))
            x, y = x - x0, y - y0
            deltaS, xc = np.cos(thetaEndPerp) * x + np.sin(-thetaEndPerp) * y, np.sin(thetaEndPerp) * x + np.cos(
                thetaEndPerp) * y
            yc = z
            xc = -xc
            s = (ang * rb + L_cap) + deltaS
        else:
            raise ValueError
    return s, xc, yc


@numba.njit()
def _force_Func_Seg(x, y, z, field_data_seg):
    Fx, Fy, Fz = vec_interp3D(x, y, z, *field_data_seg[:6])
    return Fx, Fy, Fz


@numba.njit()
def _force_Func_Internal_Fringe(x, y, z, field_data_internal):
    Fx, Fy, Fz = vec_interp3D(x, y, z, *field_data_internal[:6])
    return Fx, Fy, Fz


@numba.njit()
def _force_Func_Perturbation(x, y, z, params, fieldPerturbationData):
    s, xc, yc = cartesian_To_Center(x, y, z, params)
    Fx, Fy, Fz = vec_interp3D(s, xc, yc, *fieldPerturbationData[:6])
    return Fx, Fy, Fz


@numba.njit()
def _Force_Func_Cap(x, y, z, field_data_cap):
    Fx, Fy, Fz = vec_interp3D(x, y, z, *field_data_cap[:6])
    return Fx, Fy, Fz


@numba.njit()
def _magnetic_potential_Func_Seg(x, y, z, field_data_seg):
    return scalar_interp3D(x, y, z, *field_data_seg[:3], field_data_seg[-1])


@numba.njit()
def _magnetic_potential_Func_Internal_Fringe(x, y, z, field_data_internal):
    return scalar_interp3D(x, y, z, *field_data_internal[:3], field_data_internal[-1])


@numba.njit()
def _magnetic_potential_Func_Cap(x, y, z, field_data_cap):
    return scalar_interp3D(x, y, z, *field_data_cap[:3], field_data_cap[-1])


@numba.njit()
def _magnetic_potential_Func_Perturbation(x, y, z, fieldPerturbationData, params):
    s, xc, yc = cartesian_To_Center(x, y, z, params)
    return scalar_interp3D(s, xc, yc, *fieldPerturbationData[:3], fieldPerturbationData[-1])


@numba.njit()
def transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(Fx, Fy, Fz, x, y, ucAng, M_uc):
    # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
    # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
    # or leaving the element interface as mirror images of each other.
    # FNew: Force to be rotated out of unit cell frame
    # q: particle's position in the element frame where the force is acting
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
    phi = full_arctan2(y, x)  # calling a fast numba version that is global

    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_field_perturbations = params
    if phi < ang:  # if particle is inside bending angle region
        return (np.sqrt(x ** 2 + y ** 2) - rb) ** 2 + z ** 2 < ap ** 2
    else:  # if outside bender's angle range
        if (x - rb) ** 2 + z ** 2 <= ap ** 2 and (0 >= y >= -L_cap):  # If inside the cap on
            # eastward side
            return True
        else:
            qTestx = RIn_Ang[0, 0] * x + RIn_Ang[0, 1] * y
            qTesty = RIn_Ang[1, 0] * x + RIn_Ang[1, 1] * y
            return (qTestx - rb) ** 2 + z ** 2 <= ap ** 2 and (L_cap >= qTesty >= 0)
            # if on the westwards side


@numba.njit()
def magnetic_potential(x0, y0, z0, params, field_data):
    # magnetic potential at point q in element frame
    # q: particle's position in element frame
    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_field_perturbations = params
    field_data_seg, field_data_internal, field_data_cap, fieldPerturbationData = field_data
    x, y, z = x0, y0, z0
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan
    z = abs(z)
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
            V0 = _magnetic_potential_Func_Seg(quc[0], quc[1], quc[2], field_data_seg)
        elif position == 'FIRST' or position == 'LAST':
            V0 = magnetic_potential_First_And_Last(x, y, z, position, M_ang, field_data_internal)
        else:
            V0 = np.nan
    elif phi > ang:  # if outside bender's angle range
        if (rb - ap < x < rb + ap) and (0 > y > -L_cap):  # If inside the cap on
            # eastward side
            V0 = _magnetic_potential_Func_Cap(x, y, z, field_data_cap)
        else:
            xTest = RIn_Ang[0, 0] * x + RIn_Ang[0, 1] * y
            yTest = RIn_Ang[1, 0] * x + RIn_Ang[1, 1] * y
            if (rb - ap < xTest < rb + ap) and (
                    L_cap > yTest > 0):  # if on the westwards side
                yTest = -yTest
                V0 = _magnetic_potential_Func_Cap(xTest, yTest, z, field_data_cap)
            else:  # if not in either cap
                V0 = np.nan
    if use_field_perturbations and not np.isnan(V0):
        deltaV = _magnetic_potential_Func_Perturbation(x0, y0, z0, fieldPerturbationData,
                                                       params)  # extra force from design imperfections
        V0 = V0 + deltaV
    V0 *= field_fact
    return V0


@numba.njit()
def magnetic_potential_First_And_Last(x, y, z, position, M_ang, field_data_internal):
    if position == 'FIRST':
        xNew = M_ang[0, 0] * x + M_ang[0, 1] * y
        yNew = M_ang[1, 0] * x + M_ang[1, 1] * y
        V0 = _magnetic_potential_Func_Internal_Fringe(xNew, yNew, z, field_data_internal)
    elif position == 'LAST':
        V0 = _magnetic_potential_Func_Internal_Fringe(x, y, z, field_data_internal)
    else:
        raise Exception('INVALID POSITION SUPPLIED')
    return V0


def update_Element_Perturb_Params(shift_y, shift_z, rot_angle_y, rot_angle_z):
    """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
    raise NotImplementedError


@numba.njit()
def force(x0, y0, z0, params, field_data):
    # force at point q in element frame
    # q: particle's position in element frame
    x, y, z = x0, y0, z0
    rb, ap, L_cap, ang, num_magnets, ucAng, M_ang, RIn_Ang, M_uc, field_fact, use_field_perturbations = params
    field_data_seg, field_data_internal, field_data_cap, fieldPerturbationData = field_data
    FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
    # todo: I think I need to get rid of this symmetry stuff for the magnet imperfections to work right
    z = abs(z)
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
                Fx, Fy, Fz = _force_Func_Seg(xuc, yuc, z, field_data_seg)
                Fx, Fy, Fz = transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(Fx, Fy, Fz, x, y, ucAng, M_uc)
            else:
                if position == 'FIRST':
                    x, y = M_ang[0, 0] * x + M_ang[0, 1] * y, M_ang[1, 0] * x + M_ang[1, 1] * y
                    Fx, Fy, Fz = _force_Func_Internal_Fringe(x, y, z, field_data_internal)
                    Fx0 = Fx
                    Fy0 = Fy
                    Fx = M_ang[0, 0] * Fx0 + M_ang[0, 1] * Fy0
                    Fy = M_ang[1, 0] * Fx0 + M_ang[1, 1] * Fy0
                else:
                    Fx, Fy, Fz = _force_Func_Internal_Fringe(x, y, z, field_data_internal)
        else:
            Fx, Fy, Fz = np.nan, np.nan, np.nan
    else:  # if outside bender's angle range
        if np.sqrt((x - rb) ** 2 + z ** 2) < ap and (0 >= y >= -L_cap):  # If inside the cap on
            # eastward side
            Fx, Fy, Fz = _Force_Func_Cap(x, y, z, field_data_cap)
        else:
            x, y = M_ang[0, 0] * x + M_ang[0, 1] * y, M_ang[1, 0] * x + M_ang[1, 1] * y
            if np.sqrt((x - rb) ** 2 + z ** 2) < ap and (
                    -L_cap <= y <= 0):  # if on the westwards side
                Fx, Fy, Fz = _Force_Func_Cap(x, y, z, field_data_cap)
                Fx0 = Fx
                Fy0 = Fy
                Fx = M_ang[0, 0] * Fx0 + M_ang[0, 1] * Fy0
                Fy = M_ang[1, 0] * Fx0 + M_ang[1, 1] * Fy0
            else:  # if not in either cap, then outside the bender
                Fx, Fy, Fz = np.nan, np.nan, np.nan
    Fz = Fz * FzSymmetryFact
    Fx *= field_fact
    Fy *= field_fact
    Fz *= field_fact
    if use_field_perturbations and not np.isnan(Fx):
        deltaFx, deltaFy, deltaFz = _force_Func_Perturbation(x0, y0,
                                                             z0, params,
                                                             fieldPerturbationData)  # extra force from design imperfections
        Fx, Fy, Fz = Fx + deltaFx, Fy + deltaFy, Fz + deltaFz
    return Fx, Fy, Fz
