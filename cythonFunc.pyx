from libc.math cimport atan2, sin, cos,atan, M_PI,sqrt
from cython cimport cdivision,boundscheck,wraparound,nonecheck
import numpy as np
cimport numpy as np
ctypedef double DTYPE_FLOAT
ctypedef int DTYPE_INT

@cdivision(True)
@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def transform_Unit_Cell_Force_Into_Element_Frame(np.ndarray[DTYPE_FLOAT,ndim=1] FNew,np.ndarray[DTYPE_FLOAT,ndim=1] q
                ,np.ndarray[DTYPE_FLOAT,ndim=2]M_uc, DTYPE_FLOAT ucAng):
    # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
    # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
    # or leaving the element interface as mirror images of each other.
    # FNew: Force to be rotated out of unit cell frame
    # q: particle's position in the element frame where the force is acting
    cdef double Fx, Fy, phi,rotAngle,x,y
    cdef int cellNum
    phi = atan2(q[1], q[0])  # the anglular displacement from output of bender to the particle. I use
    # output instead of input because the unit cell is conceptually located at the output so it's easier to visualize

    cellNum = <DTYPE_INT>(phi/ucAng) + 1  # cell number that particle is in, starts at one
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
    FNew[0] = Fx * cos(rotAngle) - Fy * sin(rotAngle)
    FNew[1] = Fx * sin(rotAngle) + Fy * cos(rotAngle)
    return FNew

@cdivision(True)
@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def is_Coord_Inside_CYTHON_BenderIdealSegmentedWithCap(np.ndarray[DTYPE_FLOAT,ndim=1]q,np.ndarray[DTYPE_FLOAT,ndim=2]RIn_Ang,
    DTYPE_FLOAT ap,DTYPE_FLOAT ang,DTYPE_FLOAT rb,DTYPE_FLOAT Lcap):
    cdef double phi,qx,qy,qz,qxTest,qyTest
    qx=q[0]
    qy=q[1]
    qz=q[2]
    if not -ap<qz<ap:  # if clipping in z direction
        return False
    phi=atan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * M_PI
    if phi<ang: #if inside the bending region
        r = sqrt(qx ** 2 + qy ** 2)
        if rb-ap<r<rb+ap:
            return True
    if phi>ang:  # if outside bender's angle range
        if (rb - ap < qx < rb + ap) and (0 > qy > -Lcap): #If inside the cap on
            #the eastward side
            return True
        qxTest = RIn_Ang[0, 0] * qx + RIn_Ang[0, 1] * qy
        qyTest = RIn_Ang[1, 0] * qx + RIn_Ang[1, 1] * qy
        if (rb - ap < qxTest < rb + ap) and (Lcap > qyTest > 0): #if inside on the westward side
            return True
    return False

@cdivision(True)
@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def transform_Element_Coords_Into_Unit_Cell_Frame_CYTHON(np.ndarray[DTYPE_FLOAT,ndim=1] qNew,DTYPE_FLOAT ang,DTYPE_FLOAT ucAng):
    cdef double phi,theta
    cdef int revs
    phi=ang-atan2(qNew[1],qNew[0])
    revs=int(phi//ucAng) #number of revolutions through unit cell
    if revs%2==0: #if even
        theta = phi - ucAng * revs
    else: #if odd
        theta = ucAng-(phi - ucAng * revs)
    r=sqrt(qNew[0]**2+qNew[1]**2)
    qNew[0]=r*cos(theta) #cartesian coords in unit cell frame
    qNew[1]=r*sin(theta) #cartesian coords in unit cell frame
    return qNew
