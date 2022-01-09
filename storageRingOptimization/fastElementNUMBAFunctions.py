import numba
import numpy as np


@numba.njit(numba.types.UniTuple(numba.float64, 3)(numba.float64, numba.float64, numba.float64, numba.float64[:],
                                                   numba.types.Array(numba.types.float64, 2, 'C', readonly=True),
                                                   numba.float64))
def transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(Fx, Fy, Fz, q, M_uc, ucAng):
    # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
    # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
    # or leaving the element interface as mirror images of each other.
    # FNew: Force to be rotated out of unit cell frame
    # q: particle's position in the element frame where the force is acting
    phi = np.arctan2(q[1], q[0])  # the anglular displacement from output of bender to the particle. I use
    # output instead of input because the unit cell is conceptually located at the output so it's easier to visualize
    if phi < 0:  # restrict range to between 0 and 2pi
        phi += 2 * np.pi
    cellNum = int(phi // ucAng) + 1  # cell number that particle is in, starts at one
    if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
        rotAngle = 2 * (cellNum // 2) * ucAng
    else:  # otherwise it needs to be reflected. This is the algorithm for reflections
        Fx0 = Fx
        Fy0 = Fy
        Fx = M_uc[0, 0] * Fx0 + M_uc[0, 1] * Fy0
        Fy = M_uc[1, 0] * Fx0 + M_uc[1, 1] * Fy0
        rotAngle = 2 * ((cellNum - 1) // 2) * ucAng
    Fx0 = Fx
    Fy0 = Fy
    Fx = np.cos(rotAngle) * Fx0 - np.sin(rotAngle) * Fy0
    Fy = np.sin(rotAngle) * Fx0 + np.cos(rotAngle) * Fy0
    return Fx, Fy, Fz


@numba.njit()
def transform_Element_Coords_Into_Unit_Cell_Frame_NUMBA(q, ang, ucAng):
    quc = np.empty(3)
    angle = np.arctan2(q[1], q[0])
    if angle < 0:  # restrict range to between 0 and 2pi
        angle += 2 * np.pi
    phi = ang - angle
    revs = int(phi // ucAng)  # number of revolutions through unit cell
    if revs % 2 == 0:  # if even
        theta = phi - ucAng * revs
    else:  # if odd
        theta = ucAng - (phi - ucAng * revs)
    r = np.sqrt(q[0] ** 2 + q[1] ** 2)
    x = r * np.cos(theta)  # cartesian coords in unit cell frame
    y = r * np.sin(theta)  # cartesian coords in unit cell frame
    quc[0] = x
    quc[1] = y
    quc[2] = q[2]
    return quc


@numba.njit()
def segmented_Bender_Sim_Force_NUMBA(q, ang, ucAng, numMagnets, rb, ap, M_ang, M_uc, RIn_Ang, Lcap, Force_Func_Seg,
                                     Force_Func_Internal_Fringe, Force_Func_Cap):
    # force at point q in element frame
    # q: particle's position in element frame
    x, y, z = q
    FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
    z = abs(z)
    phi = np.arctan2(y, x)
    if phi < 0:  # restrict range to between 0 and 2pi
        phi += 2 * np.pi
    if phi <= ang:  # if particle is inside bending angle region
        rXYPlane = np.sqrt(x ** 2 + y ** 2)  # radius in xy plane
        if np.sqrt((rXYPlane - rb) ** 2 + z ** 2) < ap:
            psi = ang - phi
            revs = int(psi // ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position = 'FIRST'
            elif revs == numMagnets * 2 - 1 or revs == numMagnets * 2 - 2:
                position = 'LAST'
            else:
                position = 'INNER'
            if position == 'INNER':
                if revs % 2 == 0:  # if even
                    theta = psi - ucAng * revs
                else:  # if odd
                    theta = ucAng - (psi - ucAng * revs)
                x = rXYPlane * np.cos(theta)  # cartesian coords in unit cell frame
                y = rXYPlane * np.sin(theta)  # cartesian coords in unit cell frame
                Fx, Fy, Fz = Force_Func_Seg(x, y, z)
                Fx, Fy, Fz = transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(Fx, Fy, Fz, q, M_uc,
                                                                                ucAng)  # transform unit cell coordinates into  element frame
            else:
                if position == 'FIRST':
                    x0 = x
                    y0 = y
                    x = M_ang[0, 0] * x0 + M_ang[0, 1] * y0
                    y = M_ang[1, 0] * x0 + M_ang[1, 1] * y0

                    Fx, Fy, Fz = Force_Func_Internal_Fringe(x, y, z)
                    Fx0 = Fx
                    Fy0 = Fy
                    Fx = M_ang[0, 0] * Fx0 + M_ang[0, 1] * Fy0
                    Fy = M_ang[1, 0] * Fx0 + M_ang[1, 1] * Fy0
                else:
                    Fx, Fy, Fz = Force_Func_Internal_Fringe(x, y, z)
        else:
            Fx, Fy, Fz = np.nan, np.nan, np.nan
    else:  # if outside bender's angle range
        if np.sqrt((x - rb) ** 2 + z ** 2) < ap and (0 >= y >= -Lcap):  # If inside the cap on
            # eastward side
            Fx, Fy, Fz = Force_Func_Cap(x, y, z)
        else:
            x0=x
            y0=y
            x = M_ang[0, 0] * x0 + M_ang[0, 1] * y0
            y = M_ang[1, 0] * x0 + M_ang[1, 1] * y0
            if np.sqrt((x - rb) ** 2 + z ** 2) < ap and (-Lcap <= y <= 0):  # if on the westwards side
                Fx, Fy, Fz = Force_Func_Cap(x, y, z)
                Fx0 = Fx
                Fy0 = Fy
                Fx = M_ang[0, 0] * Fx0 + M_ang[0, 1] * Fy0
                Fy = M_ang[1, 0] * Fx0 + M_ang[1, 1] * Fy0
            else:  # if not in either cap, then outside the bender
                Fx, Fy, Fz = np.nan, np.nan, np.nan
    Fz = Fz * FzSymmetryFact
    return Fx, Fy, Fz


@numba.njit()
def lens_Ideal_Force_NUMBA(q, L, ap, K):
    # fast numba function for idea lens. Simple hard edge model
    # q: 3d coordinates in element frame
    # L: length of lens
    # ap: size of aperture, or bore of vacuum tube, of lens
    # K: the 'spring' constant of the lens
    if 0 <= q[0] <= L and q[1] ** 2 + q[2] ** 2 < ap ** 2:
        Fx = 0
        Fy = -K * q[1]
        Fz = -K * q[2]
        return Fx, Fy, Fz
    else:
        return np.nan, np.nan, np.nan


@numba.njit()
def lens_Halbach_Force_NUMBA(q, Lcap, L, ap, force_Func_Inner, force_Func_Outer):
    x, y, z = q
    # print(q)
    if np.sqrt(y ** 2 + z ** 2) > ap:
        return np.nan, np.nan, np.nan
    FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
    FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
    y = abs(y)  # confine to upper right quadrant
    z = abs(z)
    if 0 <= x <= Lcap:
        x = Lcap - x
        Fx, Fy, Fz = force_Func_Outer(x, y, z)
        Fx = -Fx
    elif Lcap < x <= L - Lcap:
        Fx, Fy, Fz = force_Func_Inner(x, y, z)
    elif 0 <= q[0] <= L:
        x = Lcap - (L - x)
        Fx, Fy, Fz = force_Func_Outer(x, y, z)
    else:
        return np.nan, np.nan, np.nan
    Fy = Fy * FySymmetryFact
    Fz = Fz * FzSymmetryFact
    return Fx, Fy, Fz
@numba.njit()
def lens_Shim_Halbach_Force_NUMBA(q,L,ap,force_Func):
    x,y,z=q
    if np.sqrt(y**2+z**2)>=ap:
        return np.nan,np.nan,np.nan
    FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
    FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
    y = abs(y)  # confine to upper right quadrant
    z = abs(z)
    if 0<=x <=L/2:
        x = L/2 - x
        Fx,Fy,Fz= force_Func(x, y, z)
        Fx=-Fx
    elif L/2<x <= L:
        x=x-L/2
        Fx,Fy,Fz = force_Func(x, y, z)
    else:
        return np.nan,np.nan,np.nan
    Fy = Fy * FySymmetryFact
    Fz = Fz * FzSymmetryFact
    return Fx,Fy,Fz
@numba.njit()
def combiner_Sim_Hexapole_Force_NUMBA(q, La, Lb, Lm, space, ang, ap, searchIsCoordInside, force_Func):
    # this function uses the symmetry of the combiner to extract the force everywhere.
    # I believe there are some redundancies here that could be trimmed to save time.
    if searchIsCoordInside == True:
        if not -ap <= q[2] <= ap:  # if outside the z apeture (vertical)
            return np.nan, np.nan, np.nan
        elif 0 <= q[0] <= Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner.
            if np.sqrt(q[1]**2+q[2]**2)<ap:
                pass
            else:
                return np.nan, np.nan, np.nan
        elif q[0] < 0:
            return np.nan, np.nan, np.nan
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            #todo: For now a square aperture, update to circular. Use a simple rotation
            m = np.tan(ang)
            Y1 = m * q[0] + (ap - m * Lb)  # upper limit
            Y2 = (-1 / m) * q[0] + La * np.sin(ang) + (Lb + La * np.cos(ang)) / m
            Y3 = m * q[0] + (-ap - m * Lb)
            if np.sign(m) < 0.0 and (q[1] < Y1 and q[1] > Y2 and q[1] > Y3):  # if the inlet is tilted 'down'
                pass
            elif np.sign(m) > 0.0 and (q[1] < Y1 and q[1] < Y2 and q[1] > Y3):  # if the inlet is tilted 'up'
                pass
            else:
                return np.nan, np.nan, np.nan
    x,y,z=q
    FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
    FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
    y = abs(y)  # confine to upper right quadrant
    z = abs(z)
    symmetryLength=Lm+2*space
    if 0<=x <=symmetryLength/2:
        x = symmetryLength/2 - x
        Fx,Fy,Fz= force_Func(x, y, z)
        Fx=-Fx
    elif symmetryLength/2<x:
        x=x-symmetryLength/2
        Fx,Fy,Fz = force_Func(x, y, z)
    else:
        raise Exception(ValueError)
    Fy = Fy * FySymmetryFact
    Fz = Fz * FzSymmetryFact
    return Fx,Fy,Fz




@numba.njit()
def combiner_Sim_Force_NUMBA(q, La, Lb, Lm, space, ang, apz, apL, apR, searchIsCoordInside, force_Func):
    # this function uses the symmetry of the combiner to extract the force everywhere.
    # I believe there are some redundancies here that could be trimmed to save time.
    if searchIsCoordInside == True:
        if not -apz <= q[2] <= apz:  # if outside the z apeture (vertical)
            return np.nan, np.nan, np.nan
        elif 0 <= q[0] <= Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -apL <= q[1] <= apR:  # if inside the y (width) apeture
                pass
            else:
                return np.nan, np.nan, np.nan
        elif q[0] < 0:
            return np.nan, np.nan, np.nan
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            #todo: better modeled as a simpler rotation
            m = np.tan(ang)
            Y1 = m * q[0] + (apR - m * Lb)  # upper limit
            Y2 = (-1 / m) * q[0] + La * np.sin(ang) + (Lb + La * np.cos(ang)) / m
            Y3 = m * q[0] + (-apL - m * Lb)
            if np.sign(m) < 0.0 and (q[1] < Y1 and q[1] > Y2 and q[1] > Y3):  # if the inlet is tilted 'down'
                pass
            elif np.sign(m) > 0.0 and (q[1] < Y1 and q[1] < Y2 and q[1] > Y3):  # if the inlet is tilted 'up'
                pass
            else:
                return np.nan, np.nan, np.nan
    x, y, z = q
    xFact = 1  # value to modify the force based on symmetry
    zFact = 1
    if 0 <= x <= (Lm / 2 + space):  # if the particle is in the first half of the magnet
        if z < 0:  # if particle is in the lower plane
            z = -z  # flip position to upper plane
            zFact = -1  # z force is opposite in lower half
    elif (Lm / 2 + space) < x:  # if the particle is in the last half of the magnet
        x = (Lm / 2 + space) - (
                x - (Lm / 2 + space))  # use the reflection of the particle
        xFact = -1  # x force is opposite in back plane
        if z < 0:  # if in the lower plane, need to use symmetry
            z = -z
            zFact = -1  # z force is opposite in lower half
    Fx, Fy, Fz = force_Func(x, y, z)
    Fx = xFact * Fx
    Fz = zFact * Fz
    return Fx, Fy, Fz


