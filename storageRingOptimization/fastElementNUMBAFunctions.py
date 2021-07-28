import numba
import numpy as np


@numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.types.Array(numba.types.float64, 2, 'C', readonly=True),numba.float64))
def transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(FNew, q, M_uc, ucAng):
    # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
    # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
    # or leaving the element interface as mirror images of each other.
    # FNew: Force to be rotated out of unit cell frame
    # q: particle's position in the element frame where the force is acting
    phi = np.arctan2(q[1], q[0])  # the anglular displacement from output of bender to the particle. I use
    # output instead of input because the unit cell is conceptually located at the output so it's easier to visualize
    cellNum = int(phi // ucAng) + 1  # cell number that particle is in, starts at one
    if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
        rotAngle = 2 * (cellNum // 2) * ucAng
    else:  # otherwise it needs to be reflected. This is the algorithm for reflections
        Fx = FNew[0]
        Fy = FNew[1]
        FNew[0] = M_uc[0, 0] * Fx + M_uc[0, 1] * Fy
        FNew[1] = M_uc[1, 0] * Fx + M_uc[1, 1] * Fy
        rotAngle = 2 * ((cellNum - 1) // 2) * ucAng
    Fx = FNew[0]
    Fy = FNew[1]
    FNew[0] = Fx * np.cos(rotAngle) - Fy * np.sin(rotAngle)
    FNew[1] = Fx * np.sin(rotAngle) + Fy * np.cos(rotAngle)
    return FNew

@numba.njit(numba.float64[:](numba.float64[:],numba.float64,numba.float64))
def transform_Element_Coords_Into_Unit_Cell_Frame_NUMBA(q, ang, ucAng):
    phi = ang - np.arctan2(q[1], q[0])
    revs = int(phi // ucAng)  # number of revolutions through unit cell
    if revs % 2 == 0:  # if even
        theta = phi - ucAng * revs
    else:  # if odd
        theta = ucAng - (phi - ucAng * revs)
    r = np.sqrt(q[0] ** 2 + q[1] ** 2)
    x = r * np.cos(theta)  # cartesian coords in unit cell frame
    y = r * np.sin(theta)  # cartesian coords in unit cell frame
    return np.asarray([x,y,q[2]])


@numba.njit()
def segmented_Bender_Sim_Force_NUMBA(q, ang, ucAng, numMagnets, rb, ap, M_ang,M_uc, RIn_Ang, Lcap,Force_Func_Seg,
                                     Force_Func_Internal_Fringe, Force_Func_Cap):
    # force at point q in element frame
    # q: particle's position in element frame
    phi = np.arctan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    if phi < ang:  # if particle is inside bending angle region
        if np.sqrt((np.sqrt(q[0] ** 2 + q[1] ** 2) - rb) ** 2 + q[2] ** 2) < ap:
            revs = int((ang - phi) // ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position = 'FIRST'
            elif revs == numMagnets * 2 - 1 or revs == numMagnets * 2 - 2:
                position = 'LAST'
            else:
                position = 'INNER'
            if position == 'INNER':
                x,y,z=q
                phi = ang - np.arctan2(y, x)
                revs = int(phi // ucAng)  # number of revolutions through unit cell
                if revs % 2 == 0:  # if even
                    theta = phi - ucAng * revs
                else:  # if odd
                    theta = ucAng - (phi - ucAng * revs)
                r = np.sqrt(x ** 2 + y ** 2)
                x = r * np.cos(theta)  # cartesian coords in unit cell frame
                y = r * np.sin(theta)  # cartesian coords in unit cell frame
                F = Force_Func_Seg(x,y,z)
                F = transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(F,q,M_uc,ucAng)  # transform unit cell coordinates into  element frame
            elif position == 'FIRST' or position == 'LAST':
                if position == 'FIRST':
                    x0 = q[0]
                    y0 = q[1]
                    x = M_ang[0, 0] * x0 + M_ang[0, 1] * y0
                    y = M_ang[1, 0] * x0 + M_ang[1, 1] * y0

                    F = Force_Func_Internal_Fringe(x, y, q[2])
                    Fx = F[0]
                    Fy = F[1]
                    F[0] = M_ang[0, 0] * Fx + M_ang[0, 1] * Fy
                    F[1] = M_ang[1, 0] * Fx + M_ang[1, 1] * Fy
                elif position == 'LAST':
                    F = Force_Func_Internal_Fringe(q[0], q[1], q[2])
        else:
            F = np.asarray([np.nan])
    else:  # if outside bender's angle range
        if (rb - ap < q[0] < rb + ap) and (0 > q[1] > -Lcap):  # If inside the cap on
            # eastward side
            F = Force_Func_Cap(q[0],q[1],q[2])
        else:
            qTestx = RIn_Ang[0, 0] * q[0] + RIn_Ang[0, 1] * q[1]
            qTesty = RIn_Ang[1, 0] * q[0] + RIn_Ang[1, 1] * q[1]
            if (rb - ap < qTestx < rb + ap) and (
                    Lcap > qTesty > 0):  # if on the westwards side
                x, y, z = qTestx, qTesty, q[2]
                y = -y
                F = Force_Func_Cap(x, y, z)
                Fx = F[0]
                Fy = F[1]
                F[0] = M_ang[0, 0] * Fx + M_ang[0, 1] * Fy
                F[1] = M_ang[1, 0] * Fx + M_ang[1, 1] * Fy
            else:  # if not in either cap, then outside the bender
                F = np.asarray([np.nan])
    return F


@numba.njit(numba.float64[:](numba.float64[:] ,numba.float64 ,numba.float64 ,numba.float64))
def lens_Ideal_Force_NUMBA(q ,L ,ap ,K):
    #fast numba function for idea lens. Simple hard edge model
    #q: 3d coordinates in element frame
    #L: length of lens
    #ap: size of aperture, or bore of vacuum tube, of lens
    #K: the 'spring' constant of the lens
    F=np.zeros(3)
    if 0 <= q[0] <= L and q[1] ** 2 + q[2] ** 2 < ap**2:
        F[1] = -K * q[1]
        F[2] = -K * q[2]
        return F
    else:
        return np.asarray([np.nan])
    


@numba.njit()
def lens_Halbach_Force_NUMBA(q,Lcap,L,force_Func_Inner,force_Func_Outer):
    x, y, z = q
    if q[0] <= Lcap:
        x = Lcap - x
        F = force_Func_Outer(x, y, z)
        F[0]=-F[0]
    elif Lcap < q[0] <= L - Lcap:
        F = force_Func_Inner(x,y,z)
    elif q[0] <= L:
        x = Lcap - (L - x)
        F = force_Func_Outer(x, y, z)
    else:
        F = np.asarray([np.nan])
    return F




@numba.njit()
def combiner_Sim_Force_NUMBA(q, La,Lb,Lm,space,ang,apz,apL,apR,searchIsCoordInside,force_Func):
    # this function uses the symmetry of the combiner to extract the force everywhere.
    # I believe there are some redundancies here that could be trimmed to save time.
    qNew = q.copy()
    if searchIsCoordInside == True:
        if not -apz < q[2] < apz:  # if outside the z apeture (vertical)
            return np.asarray([np.nan])
        elif 0 <= q[0] <= Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -apL < q[1] < apR:  # if inside the y (width) apeture
                pass
            else:
                return np.asarray([np.nan])
        elif q[0] < 0:
            return np.asarray([np.nan])
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            m = np.tan(ang)
            Y1 = m * q[0] + (apR - m * Lb)  # upper limit
            Y2 = (-1 / m) * q[0] + La * np.sin(ang) + (Lb + La * np.cos(ang)) / m
            Y3 = m * q[0] + (-apL - m * Lb)
            if q[1] < Y1 and q[1] < Y2 and q[1] > Y3:
                pass
            else:
                return np.asarray([np.nan])

    xFact = 1  # value to modify the force based on symmetry
    zFact = 1
    if 0 <= qNew[0] <= (Lm / 2 + space):  # if the particle is in the first half of the magnet
        if qNew[2] < 0:  # if particle is in the lower plane
            qNew[2] = -qNew[2]  # flip position to upper plane
            zFact = -1  # z force is opposite in lower half
    elif (Lm / 2 + space) < qNew[0]:  # if the particle is in the last half of the magnet
        qNew[0] = (Lm / 2 + space) - (
                qNew[0] - (Lm / 2 + space))  # use the reflection of the particle
        xFact = -1  # x force is opposite in back plane
        if qNew[2] < 0:  # if in the lower plane, need to use symmetry
            qNew[2] = -qNew[2]
            zFact = -1  # z force is opposite in lower half
    F=force_Func(qNew[0],qNew[1],qNew[2])
    F[0] = xFact * F[0]
    F[2] = zFact * F[2]
    return F


