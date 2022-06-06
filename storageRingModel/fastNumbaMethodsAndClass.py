import numpy.linalg as npl
import numpy as np
import numba
from math import floor
from constants import SIMULATION_MAGNETON, FLAT_WALL_VACUUM_THICKNESS

# this needs to be refactored

tupleOf3Floats = tuple[float, float, float]
nanArr7Tuple = tuple([np.ones(1) * np.nan] * 7)


@numba.njit()
def scalar_interp3D(x, y, z, xCoords, yCoords, zCoords, vec):
    X, Y, Z = len(xCoords), len(yCoords), len(zCoords)
    assert 2 < X and 2 < Y and 2 < Z, "need at least 2 points to interpolate"
    min_x, max_x = xCoords[0], xCoords[-1]
    min_y, max_y = yCoords[0], yCoords[-1]
    min_z, max_z = zCoords[0], zCoords[-1]
    delta_x = (max_x - min_x) / (xCoords.shape[0] - 1)
    delta_y = (max_y - min_y) / (yCoords.shape[0] - 1)
    delta_z = (max_z - min_z) / (zCoords.shape[0] - 1)

    x = (x - min_x) / delta_x
    y = (y - min_y) / delta_y
    z = (z - min_z) / delta_z
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    z0 = int(z)
    z1 = z0 + 1
    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)
    indexA = Y * Z * x0 + Z * y0 + z0
    indexB = Y * Z * x1 + Z * y0 + z0
    indexC = Y * Z * x0 + Z * y0 + z1
    indexD = Y * Z * x1 + Z * y0 + z1
    indexE = Y * Z * x0 + Z * y1 + z0
    indexF = Y * Z * x1 + Z * y1 + z0
    indexG = Y * Z * x0 + Z * y1 + z1
    indexH = Y * Z * x1 + Z * y1 + z1

    if x >= 0.0 and y >= 0.0 and z >= 0.0 and x1 < X and y1 < Y and z1 < Z:
        c00 = vec[indexA] * (1 - xd) + vec[indexB] * xd
        c01 = vec[indexC] * (1 - xd) + vec[indexD] * xd
        c10 = vec[indexE] * (1 - xd) + vec[indexF] * xd
        c11 = vec[indexG] * (1 - xd) + vec[indexH] * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        c = c0 * (1 - zd) + c1 * zd
    else:
        raise Exception('out of bounds')

    return c


@numba.njit()
def vec_interp3D(xLoc, yLoc, zLoc, xCoords, yCoords, zCoords, vecX, vecY, vecZ):
    X, Y, Z = len(xCoords), len(yCoords), len(zCoords)
    assert 2 < X and 2 < Y and 2 < Z, "need at least 2 points to interpolate"
    min_x, max_x = xCoords[0], xCoords[-1]
    min_y, max_y = yCoords[0], yCoords[-1]
    min_z, max_z = zCoords[0], zCoords[-1]
    delta_x = (max_x - min_x) / (len(xCoords) - 1)
    delta_y = (max_y - min_y) / (len(yCoords) - 1)
    delta_z = (max_z - min_z) / (len(zCoords) - 1)
    x = (xLoc - min_x) / delta_x
    y = (yLoc - min_y) / delta_y
    z = (zLoc - min_z) / delta_z
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    z0 = int(z)
    z1 = z0 + 1
    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)
    indexA = Y * Z * x0 + Z * y0 + z0
    indexB = Y * Z * x1 + Z * y0 + z0
    indexC = Y * Z * x0 + Z * y0 + z1
    indexD = Y * Z * x1 + Z * y0 + z1
    indexE = Y * Z * x0 + Z * y1 + z0
    indexF = Y * Z * x1 + Z * y1 + z0
    indexG = Y * Z * x0 + Z * y1 + z1
    indexH = Y * Z * x1 + Z * y1 + z1

    if x >= 0.0 and y >= 0.0 and z >= 0.0 and x1 < X and y1 < Y and z1 < Z:
        c00_x = vecX[indexA] * (1 - xd) + vecX[indexB] * xd
        c01_x = vecX[indexC] * (1 - xd) + vecX[indexD] * xd
        c10_x = vecX[indexE] * (1 - xd) + vecX[indexF] * xd
        c11_x = vecX[indexG] * (1 - xd) + vecX[indexH] * xd
        c0_x = c00_x * (1 - yd) + c10_x * yd
        c1_x = c01_x * (1 - yd) + c11_x * yd
        c_x = c0_x * (1 - zd) + c1_x * zd
        c00_y = vecY[indexA] * (1 - xd) + vecY[indexB] * xd
        c01_y = vecY[indexC] * (1 - xd) + vecY[indexD] * xd
        c10_y = vecY[indexE] * (1 - xd) + vecY[indexF] * xd
        c11_y = vecY[indexG] * (1 - xd) + vecY[indexH] * xd
        c0_y = c00_y * (1 - yd) + c10_y * yd
        c1_y = c01_y * (1 - yd) + c11_y * yd
        c_y = c0_y * (1 - zd) + c1_y * zd
        c00_z = vecZ[indexA] * (1 - xd) + vecZ[indexB] * xd
        c01_z = vecZ[indexC] * (1 - xd) + vecZ[indexD] * xd
        c10_z = vecZ[indexE] * (1 - xd) + vecZ[indexF] * xd
        c11_z = vecZ[indexG] * (1 - xd) + vecZ[indexH] * xd
        c0_z = c00_z * (1 - yd) + c10_z * yd
        c1_z = c01_z * (1 - yd) + c11_z * yd
        c_z = c0_z * (1 - zd) + c1_z * zd
    else:
        print(xLoc, yLoc, zLoc)
        print(xCoords.min(), xCoords.max())
        print(yCoords.min(), yCoords.max())
        print(zCoords.min(), zCoords.max())
        raise Exception('out of bounds')
    return c_x, c_y, c_z


@numba.njit()
def interp2D(xLoc, yLoc, xCoords, yCoords, v_c):
    X, Y = len(xCoords), len(yCoords)
    min_x, max_x = xCoords[0], xCoords[-1]
    min_y, max_y = yCoords[0], yCoords[-1]
    delta_x = (max_x - min_x) / (xCoords.shape[0] - 1)
    delta_y = (max_y - min_y) / (yCoords.shape[0] - 1)
    x = (xLoc - min_x) / delta_x
    y = (yLoc - min_y) / delta_y
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    if x0 >= 0 and y0 >= 0 and x1 < X and y1 < Y:
        c00 = v_c[Y * x0 + y0] * (1 - xd) + v_c[Y * x1 + y0] * xd
        c10 = v_c[Y * x0 + y1] * (1 - xd) + v_c[Y * x1 + y1] * xd
        c = c00 * (1 - yd) + c10 * yd
    else:
        # print(xLoc, yLoc)
        # print(xCoords.min(), xCoords.max())
        # print(yCoords.min(), yCoords.max())
        raise Exception('out of bounds')
    return c


@numba.njit()
def full_Arctan2(y, x):
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


from numba.experimental import jitclass

spec = [
    ('shiftY', numba.float64),
    ('shiftZ', numba.float64),
    ('rotY', numba.float64),
    ('rotZ', numba.float64),
    ('fieldFact', numba.float64),
    ('dummyArg', numba.none)
]


@jitclass(spec)
class BaseClassFieldHelper_Numba:
    """
    Base jitclass helper class for elementPT objects to accelerate particel tracing

    A fully numba approach is used to accelerate particle. This class, in conjunction if a numba function in
    ParticleTracerClass, enables ~100 time faster particle tracing. This requires returning scalar values ( or tuples of
    scalar values) instead of arrays/vectors. Functions used here for that are recycled for general use by casting
    there output to an array in elementPT classes. There are two major downsides however:

    1: jitclass does no allow for inheritance, so instead I use this class as a variable in other classes.
    NotImplementedError could still be raised in elementPT.py with inheritance there

    2: jitclass is not pickleable, and so multiprocessing does not work. This is fine in most cases because particle
    tracing is so fast that multiprocessing is a poor use case.
    """

    def __init__(self, dummyArg):
        self.dummyArg = dummyArg
        self.fieldFact = 1.0
        self.shiftY, self.shiftZ, self.rotY, self.rotZ = (0.0, 0.0, 0.0, 0.0)

    def get_Init_Params(self):
        """Jitclass can't be pickled, so it must be able to be rebuilt with its parameters if being pickled.
        Initialization parameters are externally with this method"""
        return self.dummyArg

    def get_Rotation_Params(self):
        """Jitclass can't be pickled, so it must be able to be rebuilt with its parameters if being pickled.
        misalingment parameters are recorded externally with this method"""
        return self.shiftY, self.shiftZ, self.rotY, self.rotZ

    def set_Misalignment_Params(self, params):
        """
        Set misalignment params to simulate actual misalignment of element. see self.misalign_Coords

        :param params: Tuple of misalignment as [shiftY, shiftZ, rotY,rotZ]
        :return:
        """
        self.shiftY, self.shiftZ, self.rotY, self.rotZ = params

    def magnetic_Potential(self, x: float, y: float, z: float) -> float:
        """
        Return magnetic potential energy at position qEl.

        Return magnetic potential energy of a lithium atom in simulation units, where the mass of a lithium-7 atom is
        1kg, at cartesian 3D coordinate qEl in the local element frame. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.

        :param x: x cartesian coords
        :param y: y cartesian coords
        :param z: z cartesian coords
        :return: magnetic potential energy of a lithium atom in simulation units, float
        """
        raise NotImplementedError

    def force(self, x: float, y: float, z: float) -> tuple:
        """
        Return force at position qEl.

        Return 3D cartesian force of a lithium at cartesian 3D coordinate qEl in the local element frame. Force vector
        has simulation units where lithium-7 mass is 1kg. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.



        :param x: x cartesian coords
        :param y: y cartesian coords
        :param z: z cartesian coords
        :return: Tuple of floats as (Fx,Fy,Fz)
        """
        raise NotImplementedError

    def is_Coord_Inside_Vacuum(self, x: float, y: float, z: float) -> bool:
        """
        Check if a 3D cartesian element frame coordinate is contained within an element's vacuum tube

        :param x: x cartesian coords
        :param y: y cartesian coords
        :param z: z cartesian coords
        :return: True if the coordinate is inside, False if outside
        """
        raise NotImplementedError

    def update_Element_Perturb_Params(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float):
        """
        perturb the alignment of the element relative to the vacuum tube. The vacuum tube will remain unchanged, but
        the element will be shifted, and therefore the force it applies will be as well. This is modeled as shifting
        and rotating the supplied coordinates to force and magnetic field function, then rotating the force

        :param shiftY: Shift in the y direction in element frame
        :param shiftZ: Shift in the z direction in the element frame
        :param rotY: Rotation about y axis of the element
        :param rotZ: Rotation about z axis of the element
        :return:
        """
        self.shiftY, self.shiftZ, self.rotY, self.rotZ = shiftY, shiftZ, rotY, rotZ

    def misalign_Coords(self, x: float, y: float, z: float):
        """Model element misalignment by misaligning coords. First do rotations about (0,0,0), then displace. Element
        misalignment has the opposite applied effect. Force will be needed to be rotated"""
        x, y = np.cos(-self.rotZ) * x - np.sin(-self.rotZ) * y, np.sin(-self.rotZ) * x + np.cos(
            -self.rotZ) * y  # rotate about z
        x, z = np.cos(-self.rotY) * x - np.sin(-self.rotY) * z, np.sin(-self.rotY) * x + np.cos(
            -self.rotY) * z  # rotate about y
        y -= self.shiftY
        z -= self.shiftZ
        return x, y, z

    def rotate_Force_For_Misalignment(self, Fx: float, Fy: float, Fz: float):
        """After rotating and translating coords to model element misalignment, the force must now be rotated as well"""
        Fx, Fy = np.cos(self.rotZ) * Fx - np.sin(self.rotZ) * Fy, np.sin(self.rotZ) * Fx + np.cos(
            self.rotZ) * Fy  # rotate about z
        Fx, Fz = np.cos(self.rotY) * Fx - np.sin(self.rotY) * Fz, np.sin(self.rotY) * Fx + np.cos(
            self.rotY) * Fz  # rotate about y
        return Fx, Fy, Fz


spec = [
    ('L', numba.float64),
    ('ap', numba.float64),
    ('fieldFact', numba.float64),
    ('inputAngleTilt', numba.float64),
    ('outputAngleTilt', numba.float64),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None)))
]


@jitclass(spec)
class DriftFieldHelper_Numba:
    """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""

    def __init__(self, L, ap,inputAngleTilt,outputAngleTilt):
        self.L = L
        self.ap = ap
        self.inputAngleTilt,self.outputAngleTilt= inputAngleTilt,outputAngleTilt
        self.fieldFact = 1.0
        self.baseClass = BaseClassFieldHelper_Numba(None)

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.L, self.ap,self.inputAngleTilt,self.outputAngleTilt

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def magnetic_Potential(self, x, y, z):
        """Magnetic potential of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if self.is_Coord_Inside_Vacuum(x, y, z) == False:
            return np.nan
        return 0.0

    def force(self, x, y, z):
        """Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if self.is_Coord_Inside_Vacuum(x, y, z) == False:
            return np.nan, np.nan, np.nan
        else:
            return 0.0, 0.0, 0.0

    def is_Coord_Inside_Vacuum(self, x, y, z):
        """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
        if self.inputAngleTilt==self.outputAngleTilt==0.0: #drift is a simple cylinder
                return 0 <= x <= self.L and np.sqrt(y ** 2 + z ** 2) < self.ap
        else:
            #min max of purely cylinderical portion of drift region
            xMinCylinder=abs(np.tan(self.inputAngleTilt)*self.ap)
            xMaxCylinder=self.L-abs(np.tan(self.outputAngleTilt)*self.ap)
            if xMinCylinder<=x<=xMaxCylinder: #if in simple straight section
                return np.sqrt(y ** 2 + z ** 2) < self.ap
            else: #if in the tilted ends, our outside, along x
                xMinDrift,xMaxDrift=-xMinCylinder,self.L+abs(np.tan(self.outputAngleTilt)*self.ap)
                if not xMinDrift<=x<=xMaxDrift: #if entirely outside
                    return False
                else: #maybe it's in the tilted slivers now
                    slopeInput,slopeOutput=np.tan(np.pi/2+self.inputAngleTilt),np.tan(np.pi/2+self.outputAngleTilt)
                    yInput=slopeInput*x
                    yOutput=slopeOutput*x-slopeOutput*self.L
                    if ((slopeInput>0 and y<yInput) or (slopeInput<0 and y>yInput)) and x<xMinCylinder:
                        return np.sqrt(y ** 2 + z ** 2) < self.ap
                    elif ((slopeOutput > 0 and y > yOutput) or (slopeOutput < 0 and y < yOutput)) and x > xMaxCylinder:
                        return np.sqrt(y ** 2 + z ** 2) < self.ap
                    else:
                        return False

    def update_Element_Perturb_Params(self, shiftY, shiftZ, rotY, rotZ):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        self.baseClass.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)


spec = [
    ('L', numba.float64),
    ('K', numba.float64),
    ('ap', numba.float64),
    ('fieldFact', numba.float64),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None)))
]


@jitclass(spec)
class IdealLensFieldHelper_Numba:
    """Helper for elementPT.LensIdeal. Psuedo-inherits from BaseClassFieldHelper"""

    def __init__(self, L, K, ap):
        self.L = L
        self.K = K
        self.ap = ap
        self.fieldFact = 1.0
        self.baseClass = BaseClassFieldHelper_Numba(None)

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.L, self.K, self.ap

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def update_Element_Perturb_Params(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        self.baseClass.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)

    def is_Coord_Inside_Vacuum(self, x: float, y: float, z: float) -> bool:
        """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
        if 0 <= x <= self.L and y ** 2 + z ** 2 < self.ap ** 2:
            return True
        else:
            return False

    def magnetic_Potential(self, x: float, y: float, z: float) -> float:
        """Magnetic potential of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if self.is_Coord_Inside_Vacuum(x, y, z):
            x, y, z = self.baseClass.misalign_Coords(x, y, z)
            r = np.sqrt(y ** 2 + z ** 2)
            V0 = .5 * self.K * r ** 2
        else:
            V0 = np.nan
        V0 = self.fieldFact * V0
        return V0

    def force(self, x: float, y: float, z: float) -> tuple:
        """Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if self.is_Coord_Inside_Vacuum(x, y, z) == True:
            x, y, z = self.baseClass.misalign_Coords(x, y, z)
            Fx = 0.0
            Fy = -self.K * y
            Fz = -self.K * z
            Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
            Fx *= self.fieldFact
            Fy *= self.fieldFact
            Fz *= self.fieldFact
            return Fx, Fy, Fz
        else:
            return np.nan, np.nan, np.nan


spec = [
    ('xArrEnd', numba.float64[::1]),
    ('yArrEnd', numba.float64[::1]),
    ('zArrEnd', numba.float64[::1]),
    ('FxArrEnd', numba.float64[::1]),
    ('FyArrEnd', numba.float64[::1]),
    ('FzArrEnd', numba.float64[::1]),
    ('VArrEnd', numba.float64[::1]),
    ('xArrIn', numba.float64[::1]),
    ('yArrIn', numba.float64[::1]),
    ('FxArrIn', numba.float64[::1]),
    ('FyArrIn', numba.float64[::1]),
    ('VArrIn', numba.float64[::1]),
    ('fieldPerturbationData', numba.types.UniTuple(numba.float64[::1], 7)),
    ('L', numba.float64),
    ('Lcap', numba.float64),
    ('ap', numba.float64),
    ('fieldFact', numba.float64),
    ('extraFieldLength', numba.float64),
    ('useFieldPerturbations', numba.boolean),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None)))
]


@jitclass(spec)
class LensHalbachFieldHelper_Numba:
    """Helper for elementPT.HalbachLensSim. Psuedo-inherits from BaseClassFieldHelper"""

    def __init__(self, fieldData, fieldPerturbationData, L, Lcap, ap, extraFieldLength):
        self.xArrEnd, self.yArrEnd, self.zArrEnd, self.FxArrEnd, self.FyArrEnd, self.FzArrEnd, self.VArrEnd, self.xArrIn, \
        self.yArrIn, self.FxArrIn, self.FyArrIn, self.VArrIn = fieldData
        self.L = L
        self.Lcap = Lcap
        self.ap = ap
        self.fieldFact = 1.0
        self.extraFieldLength = extraFieldLength
        self.useFieldPerturbations = True if fieldPerturbationData is not None else False
        self.fieldPerturbationData = fieldPerturbationData if fieldPerturbationData is not None else nanArr7Tuple
        self.baseClass = BaseClassFieldHelper_Numba(None)

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldData = self.xArrEnd, self.yArrEnd, self.zArrEnd, self.FxArrEnd, self.FyArrEnd, self.FzArrEnd, self.VArrEnd, \
                    self.xArrIn, self.yArrIn, self.FxArrIn, self.FyArrIn, self.VArrIn
        fieldPerturbationData = None if self.useFieldPerturbations == False else self.fieldPerturbationData
        return fieldData, fieldPerturbationData, self.L, self.Lcap, self.ap, self.extraFieldLength

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def update_Element_Perturb_Params(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        self.baseClass.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)

    def is_Coord_Inside_Vacuum(self, x: float, y: float, z: float) -> bool:
        """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
        if 0 <= x <= self.L and y ** 2 + z ** 2 < self.ap ** 2:
            return True
        else:
            return False

    def _magnetic_Potential_Func_Fringe(self, x: float, y: float, z: float, useImperfectInterp: bool = False) -> float:
        """Wrapper for interpolation of magnetic fields at ends of lens. see self.magnetic_Potential"""
        if useImperfectInterp == False:
            V = scalar_interp3D(-z, y, x, self.xArrEnd, self.yArrEnd, self.zArrEnd, self.VArrEnd)
        else:
            xArr, yArr, zArr, FxArr, FyArr, FzArr, VArr = self.fieldPerturbationData
            V = scalar_interp3D(-z, y, x, xArr, yArr, zArr, VArr)
        return V

    def _magnetic_Potential_Func_Inner(self, x: float, y: float, z: float) -> float:
        """Wrapper for interpolation of magnetic fields of plane at center lens.see self.magnetic_Potential"""
        V = interp2D(-z, y, self.xArrIn, self.yArrIn, self.VArrIn)
        return V

    def _force_Func_Outer(self, x, y, z, useImperfectInterp=False) -> tupleOf3Floats:
        """Wrapper for interpolation of force fields at ends of lens. see self.force"""
        if useImperfectInterp == False:
            Fx0, Fy0, Fz0 = vec_interp3D(-z, y, x, self.xArrEnd, self.yArrEnd, self.zArrEnd,
                                         self.FxArrEnd, self.FyArrEnd, self.FzArrEnd)
        else:
            xArr, yArr, zArr, FxArr, FyArr, FzArr, VArr = self.fieldPerturbationData
            Fx0, Fy0, Fz0 = vec_interp3D(-z, y, x, xArr, yArr, zArr, FxArr, FyArr, FzArr)
        Fx = Fz0
        Fy = Fy0
        Fz = -Fx0
        return Fx, Fy, Fz

    def _force_Func_Inner(self, x: float, y: float, z: float) -> tupleOf3Floats:
        """Wrapper for interpolation of force fields of plane at center lens. see self.force"""
        Fx = 0.0
        Fy = interp2D(-z, y, self.xArrIn, self.yArrIn, self.FyArrIn)
        Fz = -interp2D(-z, y, self.xArrIn, self.yArrIn, self.FxArrIn)
        return Fx, Fy, Fz

    def force(self, x: float, y: float, z: float) -> tupleOf3Floats:
        """Force on lithium atom. Functions to combine perfect force and extra force from imperfections.
         Perturbation force is messed up force minus perfect force."""

        Fx, Fy, Fz = self._force(x, y, z)
        if self.useFieldPerturbations:
            deltaFx, deltaFy, deltaFz = self._force_Field_Perturbations(x, y,
                                                                        z)  # extra force from design imperfections
            Fx, Fy, Fz = Fx + deltaFx, Fy + deltaFy, Fz + deltaFz
        return Fx, Fy, Fz

    def _force(self, x: float, y: float, z: float) -> tupleOf3Floats:
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
        if self.is_Coord_Inside_Vacuum(x, y, z) == False:
            return np.nan, np.nan, np.nan
        x, y, z = self.baseClass.misalign_Coords(x, y, z)
        FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if -self.extraFieldLength <= x <= self.Lcap:  # at beginning of lens
            x = self.Lcap - x
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
            Fx = -Fx
        elif self.Lcap < x <= self.L - self.Lcap:  # if long enough, model interior as uniform in x
            Fx, Fy, Fz = self._force_Func_Inner(x, y, z)
        elif self.L - self.Lcap <= x <= self.L + self.extraFieldLength:  # at end of lens
            x = self.Lcap - (self.L - x)
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
        else:
            raise Exception("Particle outside field region")  # this may be triggered when itentionally misligned
        Fx = Fx * self.fieldFact
        Fy = Fy * FySymmetryFact * self.fieldFact
        Fz = Fz * FzSymmetryFact * self.fieldFact
        Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        return Fx, Fy, Fz

    def _force_Field_Perturbations(self, x0: float, y0: float, z0: float) -> tupleOf3Floats:
        if self.is_Coord_Inside_Vacuum(x0, y0, z0) == False:
            return np.nan, np.nan, np.nan
        x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x = x - self.L / 2
        Fx, Fy, Fz = self._force_Func_Outer(x, y, z,
                                            useImperfectInterp=True)  # being used to hold fields for entire lens
        Fx = Fx * self.fieldFact
        Fy = Fy * self.fieldFact
        Fz = Fz * self.fieldFact
        Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        return Fx, Fy, Fz

    def magnetic_Potential(self, x: float, y: float, z: float) -> float:
        """Magnetic potential of lithium atom. Functions to combine perfect potential and extra potential from
        imperfections. Perturbation potential is messed up potential minus perfect potential."""

        V = self._magnetic_Potential(x, y, z)
        if self.useFieldPerturbations == True:
            deltaV = self._magnetic_Potential_Perturbations(x, y, z)  # extra potential from design imperfections
            V += deltaV
        return V

    def _magnetic_Potential(self, x: float, y: float, z: float) -> float:
        """
        Magnetic potential energy of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper

        Symmetry if used to simplify the computation of potential. Either end of the lens is identical, so coordinates
        falling within some range are mapped to an interpolation of the potential at the lenses end. If the lens is
        long enough, the inner region is modeled as a single plane as well. nan is returned if coordinate
        is outside vacuum tube

        :param x: x cartesian coordinate, m
        :param y: y cartesian coordinate, m
        :param z: z cartesian coordinate, m
        :return: potential energy, simulation units. returns nan if the coordinate is outside the vacuum tube
        """
        if self.is_Coord_Inside_Vacuum(x, y, z) == False:
            return np.nan
        x, y, z = self.baseClass.misalign_Coords(x, y, z)
        y = abs(y)
        z = abs(z)
        if -self.extraFieldLength <= x <= self.Lcap:
            x = self.Lcap - x
            V0 = self._magnetic_Potential_Func_Fringe(x, y, z)
        elif self.Lcap < x <= self.L - self.Lcap:
            V0 = self._magnetic_Potential_Func_Inner(x, y, z)
        elif 0 <= x <= self.L + self.extraFieldLength:
            x = self.Lcap - (self.L - x)
            V0 = self._magnetic_Potential_Func_Fringe(x, y, z)
        else:
            raise Exception("Particle outside field region")
        V0 = V0 * self.fieldFact
        return V0

    def _magnetic_Potential_Perturbations(self, x0: float, y0: float, z0: float) -> float:
        if self.is_Coord_Inside_Vacuum(x0, y0, z0) == False:
            return np.nan
        x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x = x - self.L / 2
        V0 = self._magnetic_Potential_Func_Fringe(x, y, z, useImperfectInterp=True)
        return V0


spec = [
    ('ang', numba.float64),
    ('K', numba.float64),
    ('rp', numba.float64),
    ('rb', numba.float64),
    ('ap', numba.float64),
    ('fieldFact', numba.float64),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None)))
]


@jitclass(spec)
class BenderIdealFieldHelper_Numba:

    def __init__(self, ang, K, rp, rb, ap):
        self.ang = ang
        self.K = K
        self.rp = rp
        self.rb = rb
        self.ap = ap
        self.fieldFact = 1.0
        self.baseClass = BaseClassFieldHelper_Numba(None)

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.ang, self.K, self.rp, self.rb, self.ap

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def magnetic_Potential(self, x, y, z):
        # potential energy at provided coordinates
        # q coords in element frame
        phi = full_Arctan2(y, x)
        rPolar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
        rToroidal = np.sqrt((rPolar - self.rb) ** 2 + z ** 2)
        if phi < self.ang and rToroidal < self.ap:
            V0 = .5 * self.K * SIMULATION_MAGNETON * rToroidal ** 2
        else:
            V0 = np.nan
        V0 *= self.fieldFact
        return V0

    def force(self, x, y, z):
        # force at point q in element frame
        # q: particle's position in element frame
        phi = full_Arctan2(y, x)
        rPolar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
        rToroidal = np.sqrt((rPolar - self.rb) ** 2 + z ** 2)
        if phi < self.ang and rToroidal < self.ap:
            F0 = -self.K * (rPolar - self.rb)  # force in x y plane
            Fx = np.cos(phi) * F0
            Fy = np.sin(phi) * F0
            Fz = -self.K * z
        else:
            Fx, Fy, Fz = np.nan, np.nan, np.nan
        Fx *= self.fieldFact
        Fy *= self.fieldFact
        Fz *= self.fieldFact
        return Fx, Fy, Fz

    def is_Coord_Inside_Vacuum(self, x, y, z):
        phi = full_Arctan2(y, x)
        if phi < 0:  # constraint to between zero and 2pi
            phi += 2 * np.pi
        if phi <= self.ang:  # if particle is in bending segment
            rh = np.sqrt(x ** 2 + y ** 2) - self.rb  # horizontal radius
            r = np.sqrt(rh ** 2 + z ** 2)  # particle displacement from center of apeture
            if r > self.ap:
                return False
            else:
                return True
        else:
            return False

    def update_Element_Perturb_Params(self, shiftY, shiftZ, rotY, rotZ):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        raise NotImplementedError


spec = [
    ('c1', numba.float64),
    ('c2', numba.float64),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('apL', numba.float64),
    ('apR', numba.float64),
    ('apz', numba.float64),
    ('ang', numba.float64),
    ('fieldFact', numba.float64),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None)))
]


@jitclass(spec)
class CombinerIdealFieldHelper_Numba:

    def __init__(self, c1, c2, La, Lb, apL, apR, apz, ang):
        self.c1 = c1
        self.c2 = c2
        self.La = La
        self.Lb = Lb
        self.apL = apL
        self.apR = apR
        self.apz = apz
        self.ang = ang
        self.fieldFact = 1.0
        self.baseClass = BaseClassFieldHelper_Numba(None)

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.c1, self.c2, self.La, self.Lb, self.apL, self.apR, self.apz, self.ang

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def force(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan, np.nan, np.nan
        else:
            return self.force_Without_isInside_Check(x, y, z)

    def force_Without_isInside_Check(self, x, y, z):
        # force at point q in element frame
        # q: particle's position in element frame
        Fx, Fy, Fz = 0.0, 0.0, 0.0
        if 0 < x < self.Lb:
            B0 = np.sqrt((self.c2 * z) ** 2 + (self.c1 + self.c2 * y) ** 2)
            Fy = SIMULATION_MAGNETON * self.c2 * (self.c1 + self.c2 * y) / B0
            Fz = SIMULATION_MAGNETON * self.c2 ** 2 * z / B0
        else:
            pass
        Fx *= self.fieldFact
        Fy *= self.fieldFact
        Fz *= self.fieldFact
        return Fx, Fy, Fz

    def magnetic_Potential(self, x, y, z):
        if self.is_Coord_Inside_Vacuum(x, y, z) == False:
            return np.nan
        if 0 < x < self.Lb:
            V0 = SIMULATION_MAGNETON * np.sqrt((self.c2 * z) ** 2 + (self.c1 + self.c2 * y) ** 2)
        else:
            V0 = 0.0
        V0 *= self.fieldFact
        return V0

    def is_Coord_Inside_Vacuum(self, x, y, z):
        # q: coordinate to test in element's frame
        if not -self.apz < z < self.apz:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -self.apL < y < self.apR and -self.apz < z < self.apz:  # if inside the y (width) apeture
                return True
            else:
                return False
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            m = np.tan(self.ang)
            Y1 = m * x + (self.apR - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-self.apL - m * self.Lb)
            if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                return True
            elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                return True
            else:
                return False


spec = [
    ('VArr', numba.float64[::1]),
    ('FxArr', numba.float64[::1]),
    ('FyArr', numba.float64[::1]),
    ('FzArr', numba.float64[::1]),
    ('xArr', numba.float64[::1]),
    ('yArr', numba.float64[::1]),
    ('zArr', numba.float64[::1]),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('Lm', numba.float64),
    ('space', numba.float64),
    ('apL', numba.float64),
    ('apR', numba.float64),
    ('apz', numba.float64),
    ('ang', numba.float64),
    ('fieldFact', numba.float64),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None)))
]


@jitclass(spec)
class CombinerSimFieldHelper_Numba:

    def __init__(self, fieldData, La, Lb, Lm, space, apL, apR, apz, ang, fieldFact):
        self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr, self.VArr = fieldData
        self.La = La
        self.Lb = Lb
        self.Lm = Lm
        self.space = space
        self.apL = apL
        self.apR = apR
        self.apz = apz
        self.ang = ang
        self.fieldFact = fieldFact
        self.baseClass = BaseClassFieldHelper_Numba(None)

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldData = self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr, self.VArr
        return fieldData, self.La, self.Lb, self.Lm, self.space, self.apL, self.apR, self.apz, self.ang, self.fieldFact

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def _force_Func(self, x, y, z):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return vec_interp3D(x, y, z, self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr)

    def _magnetic_Potential_Func(self, x, y, z):
        return scalar_interp3D(x, y, z, self.xArr, self.yArr, self.zArr, self.VArr)

    def force(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan, np.nan, np.nan
        else:
            return self.force_Without_isInside_Check(x, y, z)

    def force_Without_isInside_Check(self, x, y, z):
        # this function uses the symmetry of the combiner to extract the force everywhere.
        # I believe there are some redundancies here that could be trimmed to save time.
        xFact = 1  # value to modify the force based on symmetry
        zFact = 1
        if 0 <= x <= (self.Lm / 2 + self.space):  # if the particle is in the first half of the magnet
            if z < 0:  # if particle is in the lower plane
                z = -z  # flip position to upper plane
                zFact = -1  # z force is opposite in lower half
        elif (self.Lm / 2 + self.space) < x:  # if the particle is in the last half of the magnet
            x = (self.Lm / 2 + self.space) - (x - (self.Lm / 2 + self.space))  # use the reflection of the particle
            xFact = -1  # x force is opposite in back plane
            if z < 0:  # if in the lower plane, need to use symmetry
                z = -z
                zFact = -1  # z force is opposite in lower half
        Fx, Fy, Fz = self._force_Func(x, y, z)
        Fx = self.fieldFact * xFact * Fx
        Fy = self.fieldFact * Fy
        Fz = self.fieldFact * zFact * Fz
        return Fx, Fy, Fz

    def magnetic_Potential(self, x, y, z):
        # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
        if 0 <= x <= (self.Lm / 2 + self.space):  # if the particle is in the first half of the magnet
            if z < 0:  # if particle is in the lower plane
                z = -z  # flip position to upper plane
        if (self.Lm / 2 + self.space) < x:  # if the particle is in the last half of the magnet
            x = (self.Lm / 2 + self.space) - (
                    x - (self.Lm / 2 + self.space))  # use the reflection of the particle
            if z < 0:  # if in the lower plane, need to use symmetry
                z = -z
        return self.fieldFact * self._magnetic_Potential_Func(x, y, z)

    def is_Coord_Inside_Vacuum(self, x, y, z) -> bool:
        # q: coordinate to test in element's frame
        if not -self.apz <= z <= self.apz:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -self.apL < y < self.apR:  # if inside the y (width) apeture
                return True
            else:
                return False
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            m = np.tan(self.ang)
            Y1 = m * x + (self.apR - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-self.apL - m * self.Lb)
            if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                return True
            elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                return True
            else:
                return False


spec = [
    ('VArr', numba.float64[::1]),
    ('FxArr', numba.float64[::1]),
    ('FyArr', numba.float64[::1]),
    ('FzArr', numba.float64[::1]),
    ('xArr', numba.float64[::1]),
    ('yArr', numba.float64[::1]),
    ('zArr', numba.float64[::1]),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('Lm', numba.float64),
    ('space', numba.float64),
    ('ap', numba.float64),
    ('ang', numba.float64),
    ('fieldFact', numba.float64),
    ('extraFieldLength', numba.float64),
    ('useSymmetry', numba.boolean),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None)))
]


@jitclass(spec)
class CombinerHalbachLensSimFieldHelper_Numba:

    def __init__(self, fieldData, La, Lb, Lm, space, ap, ang, fieldFact, extraFieldLength, useSymmetry):
        self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr, self.VArr = fieldData
        self.La = La
        self.Lb = Lb
        self.Lm = Lm
        self.space = space
        self.ap = ap
        self.ang = ang
        self.fieldFact = fieldFact
        self.extraFieldLength = extraFieldLength
        self.useSymmetry = useSymmetry
        self.baseClass = self.baseClass = BaseClassFieldHelper_Numba(None)

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldData = self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr, self.VArr
        return fieldData, self.La, self.Lb, self.Lm, self.space, self.ap, self.ang, self.fieldFact, \
               self.extraFieldLength, self.useSymmetry

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def _force_Func(self, x, y, z):
        Fx0, Fy0, Fz0 = vec_interp3D(-z, y, x, self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr)
        Fx = Fz0
        Fy = Fy0
        Fz = -Fx0
        return Fx, Fy, Fz

    def _magnetic_Potential_Func(self, x, y, z):
        return scalar_interp3D(-z, y, x, self.xArr, self.yArr, self.zArr, self.VArr)

    def force(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan, np.nan, np.nan
        else:
            return self.force_Without_isInside_Check(x, y, z)

    def force_Without_isInside_Check(self, x0, y0, z0):
        # this function uses the symmetry of the combiner to extract the force everywhere.
        # I believe there are some redundancies here that could be trimmed to save time.
        x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        symmetryPlaneX = self.Lm / 2 + self.space  # field symmetry plane location
        if self.useSymmetry:
            FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
            FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
            y = abs(y)  # confine to upper right quadrant
            z = abs(z)
            if -self.extraFieldLength <= x <= symmetryPlaneX:
                x = symmetryPlaneX - x
                Fx, Fy, Fz = self._force_Func(x, y, z)
                Fx = -Fx
            elif symmetryPlaneX < x:
                x = x - symmetryPlaneX
                Fx, Fy, Fz = self._force_Func(x, y, z)
            else:
                raise ValueError
            Fy = Fy * FySymmetryFact
            Fz = Fz * FzSymmetryFact
        else:
            x = x - symmetryPlaneX
            Fx, Fy, Fz = self._force_Func(x, y, z)
        Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        Fx *= self.fieldFact
        Fy *= self.fieldFact
        Fz *= self.fieldFact
        return Fx, Fy, Fz

    def magnetic_Potential(self, x, y, z):
        if self.is_Coord_Inside_Vacuum(x, y, z) == False:
            return np.nan
        x, y, z = self.baseClass.misalign_Coords(x, y, z)
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        symmetryPlaneX = self.Lm / 2 + self.space  # field symmetry plane location
        if self.useSymmetry:
            if -self.extraFieldLength <= x <= symmetryPlaneX:
                x = symmetryPlaneX - x
                V = self._magnetic_Potential_Func(x, y, z)
            elif symmetryPlaneX < x:  # particle can extend past 2*symmetryPlaneX
                x = x - symmetryPlaneX
                V = self._magnetic_Potential_Func(x, y, z)
            else:
                raise Exception(ValueError)
        else:
            x = x - symmetryPlaneX
            V = self._magnetic_Potential_Func(x, y, z)
        V = V * self.fieldFact
        return V

    def is_Coord_Inside_Vacuum(self, x, y, z):
        # q: coordinate to test in element's frame
        standOff = 10e-6  # first valid (non np.nan) interpolation point on face of lens is 1e-6 off the surface of the lens
        assert FLAT_WALL_VACUUM_THICKNESS > standOff
        if not -self.ap <= z <= self.ap:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb + FLAT_WALL_VACUUM_THICKNESS:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner.
            if np.sqrt(y ** 2 + z ** 2) < self.ap:
                return True
            else:
                return False
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            # todo: For now a square aperture, update to circular. Use a simple rotation
            m = np.tan(self.ang)
            Y1 = m * x + (self.ap - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-1.5 * self.ap - m * self.Lb)
            if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                return True
            elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                return True
            else:
                return False

    def update_Element_Perturb_Params(self, shiftY, shiftZ, rotY, rotZ):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        self.baseClass.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)


spec = [
    ('fieldDataSeg', numba.types.UniTuple(numba.float64[::1], 7)),
    ('fieldDataInternal', numba.types.UniTuple(numba.float64[::1], 7)),
    ('fieldDataCap', numba.types.UniTuple(numba.float64[::1], 7)),
    ('ap', numba.float64),
    ('ang', numba.float64),
    ('ucAng', numba.float64),
    ('rb', numba.float64),
    ('numMagnets', numba.float64),
    ('M_uc', numba.float64[:, ::1]),
    ('M_ang', numba.float64[:, ::1]),
    ('Lcap', numba.float64),
    ('RIn_Ang', numba.float64[:, ::1]),
    ('misalignmentCoeffArr', numba.float64[:, ::1]),
    ('fieldFact', numba.float64),
    ('baseClass', numba.typeof(BaseClassFieldHelper_Numba(None))),
    ('fieldPerturbationData', numba.types.UniTuple(numba.float64[::1], 7)),
    ('useFieldPerturbations', numba.boolean)
]


@jitclass(spec)
class SegmentedBenderSimFieldHelper_Numba:

    def __init__(self, fieldDataSeg, fieldDataInternal, fieldDataCap, fieldPerturbationData, ap, ang, ucAng, rb,
                 numMagnets, Lcap, M_uc, M_ang, RIn_Ang):
        self.fieldDataSeg = fieldDataSeg
        self.fieldDataInternal = fieldDataInternal
        self.fieldDataCap = fieldDataCap
        self.ap = ap
        self.ang = ang
        self.ucAng = ucAng
        self.rb = rb
        self.numMagnets = numMagnets
        self.M_uc = M_uc
        self.M_ang = M_ang
        self.Lcap = Lcap
        self.RIn_Ang = RIn_Ang
        self.fieldFact = 1.0
        self.baseClass = BaseClassFieldHelper_Numba(None)
        self.misalignmentCoeffArr = 2.0 * (np.random.random_sample((numMagnets + 1, 12)) - .5)
        self.useFieldPerturbations = True if fieldPerturbationData is not None else False  # apply magnet Perturbation data
        self.fieldPerturbationData = fieldPerturbationData if fieldPerturbationData is not None else nanArr7Tuple

    def get_Init_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldPerturbationData = None if self.useFieldPerturbations == False else self.fieldPerturbationData
        return self.fieldDataSeg, self.fieldDataInternal, self.fieldDataCap, fieldPerturbationData, self.ap, self.ang, \
               self.ucAng, self.rb, self.numMagnets, self.Lcap, self.M_uc, self.M_ang, self.RIn_Ang

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, self.baseClass.get_Rotation_Params()

    def set_Internal_Params(self, params):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""

        self.fieldFact, alignmentParams = params
        self.baseClass.set_Misalignment_Params(alignmentParams)

    def cartesian_To_Center(self, x, y, z):
        """Convert from cartesian coords to HalbachLensClass.SegmentedBenderHalbach coored, ie "center coords" for
        evaluation by interpolator"""

        if x > 0.0 and -self.Lcap <= z <= 0.0:
            s = self.Lcap + z
            xc = x - self.rb
            yc = y
        else:
            theta = full_Arctan2(z, x)
            if theta <= self.ang:
                s = theta * self.rb + self.Lcap
                xc = np.sqrt(x ** 2 + z ** 2) - self.rb
                yc = y
            elif self.ang < theta <= 2 * np.pi:  # i'm being lazy here and not limiting the real end
                x0, z0 = np.cos(self.ang) * self.rb, np.sin(self.ang) * self.rb
                thetaEndPerp = np.pi - np.arctan(-1 / np.tan(self.ang))
                x, z = x - x0, z - z0
                deltaS, xc = np.cos(thetaEndPerp) * x + np.sin(-thetaEndPerp) * z, np.sin(thetaEndPerp) * x + np.cos(
                    thetaEndPerp) * z
                yc = y
                xc = -xc
                s = (self.ang * self.rb + self.Lcap) + deltaS
            else:
                raise ValueError
        return s, xc, yc

    def _force_Func_Seg(self, x, y, z):
        Fx0, Fy0, Fz0 = vec_interp3D(x, -z, y, *self.fieldDataSeg[:6])
        Fx = Fx0
        Fy = Fz0
        Fz = -Fy0
        return Fx, Fy, Fz

    def _force_Func_Internal_Fringe(self, x, y, z):
        Fx0, Fy0, Fz0 = vec_interp3D(x, -z, y, *self.fieldDataInternal[:6])
        Fx = Fx0
        Fy = Fz0
        Fz = -Fy0
        return Fx, Fy, Fz

    def _force_Func_Perturbation(self, x, y, z):
        s, xc, yc = self.cartesian_To_Center(x, -z, y)
        Fx0, Fy0, Fz0 = vec_interp3D(s, xc, yc, *self.fieldPerturbationData[:6])
        Fx = Fx0
        Fy = Fz0
        Fz = -Fy0
        return Fx, Fy, Fz

    def _Force_Func_Cap(self, x, y, z):
        Fx0, Fy0, Fz0 = vec_interp3D(x, -z, y, *self.fieldDataCap[:6])
        Fx = Fx0
        Fy = Fz0
        Fz = -Fy0
        return Fx, Fy, Fz

    def _magnetic_Potential_Func_Seg(self, x, y, z):
        return scalar_interp3D(x, -z, y, *self.fieldDataSeg[:3], self.fieldDataSeg[-1])

    def _magnetic_Potential_Func_Internal_Fringe(self, x, y, z):
        return scalar_interp3D(x, -z, y, *self.fieldDataInternal[:3], self.fieldDataInternal[-1])

    def _magnetic_Potential_Func_Cap(self, x, y, z):
        return scalar_interp3D(x, -z, y, *self.fieldDataCap[:3], self.fieldDataCap[-1])

    def _magnetic_Potential_Func_Perturbation(self, x, y, z):
        s, xc, yc = self.cartesian_To_Center(x, -z, y)
        return scalar_interp3D(s, xc, yc, *self.fieldPerturbationData[:3], self.fieldPerturbationData[-1])

    def transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(self, Fx, Fy, Fz, x, y):
        # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
        # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
        # or leaving the element interface as mirror images of each other.
        # FNew: Force to be rotated out of unit cell frame
        # q: particle's position in the element frame where the force is acting
        phi = np.arctan2(y, x)  # the anglular displacement from output of bender to the particle. I use
        # output instead of input because the unit cell is conceptually located at the output so it's easier to visualize
        if phi < 0:  # restrict range to between 0 and 2pi
            phi += 2 * np.pi
        cellNum = int(phi // self.ucAng) + 1  # cell number that particle is in, starts at one
        if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
            rotAngle = 2 * (cellNum // 2) * self.ucAng
        else:  # otherwise it needs to be reflected. This is the algorithm for reflections
            Fx0 = Fx
            Fy0 = Fy
            Fx = self.M_uc[0, 0] * Fx0 + self.M_uc[0, 1] * Fy0
            Fy = self.M_uc[1, 0] * Fx0 + self.M_uc[1, 1] * Fy0
            rotAngle = 2 * ((cellNum - 1) // 2) * self.ucAng
        Fx0 = Fx
        Fy0 = Fy
        Fx = np.cos(rotAngle) * Fx0 - np.sin(rotAngle) * Fy0
        Fy = np.sin(rotAngle) * Fx0 + np.cos(rotAngle) * Fy0
        return Fx, Fy, Fz

    def force(self, x0, y0, z0):
        # force at point q in element frame
        # q: particle's position in element frame

        # todo: change this x0,y0,z0 stuff

        x, y, z = x0, y0, z0
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        z = abs(z)
        phi = np.arctan2(y, x)
        if phi < 0:  # restrict range to between 0 and 2pi
            phi += 2 * np.pi
        if phi <= self.ang:  # if particle is inside bending angle region
            rXYPlane = np.sqrt(x ** 2 + y ** 2)  # radius in xy plane
            if np.sqrt((rXYPlane - self.rb) ** 2 + z ** 2) < self.ap:
                psi = self.ang - phi
                revs = int(psi // self.ucAng)  # number of revolutions through unit cell
                if revs == 0 or revs == 1:
                    position = 'FIRST'
                elif revs == self.numMagnets * 2 - 1 or revs == self.numMagnets * 2 - 2:
                    position = 'LAST'
                else:
                    position = 'INNER'
                if position == 'INNER':
                    if revs % 2 == 0:  # if even
                        theta = psi - self.ucAng * revs
                    else:  # if odd
                        theta = self.ucAng - (psi - self.ucAng * revs)
                    xuc = rXYPlane * np.cos(theta)  # cartesian coords in unit cell frame
                    yuc = rXYPlane * np.sin(theta)  # cartesian coords in unit cell frame
                    Fx, Fy, Fz = self._force_Func_Seg(xuc, yuc, z)
                    Fx, Fy, Fz = self.transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(Fx, Fy, Fz, x, y)
                else:
                    if position == 'FIRST':
                        x, y = self.M_ang[0, 0] * x + self.M_ang[0, 1] * y, self.M_ang[1, 0] * x + self.M_ang[1, 1] * y
                        Fx, Fy, Fz = self._force_Func_Internal_Fringe(x, y, z)
                        Fx0 = Fx
                        Fy0 = Fy
                        Fx = self.M_ang[0, 0] * Fx0 + self.M_ang[0, 1] * Fy0
                        Fy = self.M_ang[1, 0] * Fx0 + self.M_ang[1, 1] * Fy0
                    else:
                        Fx, Fy, Fz = self._force_Func_Internal_Fringe(x, y, z)
            else:
                Fx, Fy, Fz = np.nan, np.nan, np.nan
        else:  # if outside bender's angle range
            if np.sqrt((x - self.rb) ** 2 + z ** 2) < self.ap and (0 >= y >= -self.Lcap):  # If inside the cap on
                # eastward side
                Fx, Fy, Fz = self._Force_Func_Cap(x, y, z)
            else:
                x, y = self.M_ang[0, 0] * x + self.M_ang[0, 1] * y, self.M_ang[1, 0] * x + self.M_ang[1, 1] * y
                if np.sqrt((x - self.rb) ** 2 + z ** 2) < self.ap and (
                        -self.Lcap <= y <= 0):  # if on the westwards side
                    Fx, Fy, Fz = self._Force_Func_Cap(x, y, z)
                    Fx0 = Fx
                    Fy0 = Fy
                    Fx = self.M_ang[0, 0] * Fx0 + self.M_ang[0, 1] * Fy0
                    Fy = self.M_ang[1, 0] * Fx0 + self.M_ang[1, 1] * Fy0
                else:  # if not in either cap, then outside the bender
                    Fx, Fy, Fz = np.nan, np.nan, np.nan
        Fz = Fz * FzSymmetryFact
        Fx *= self.fieldFact
        Fy *= self.fieldFact
        Fz *= self.fieldFact
        if self.useFieldPerturbations == True and np.isnan(Fx) == False:
            deltaFx, deltaFy, deltaFz = self._force_Func_Perturbation(x0, y0,
                                                                      z0)  # extra force from design imperfections
            Fx, Fy, Fz = Fx + deltaFx, Fy + deltaFy, Fz + deltaFz
        return Fx, Fy, Fz

    def transform_Element_Coords_Into_Unit_Cell_Frame(self, x, y, z):
        angle = np.arctan2(y, x)
        if angle < 0:  # restrict range to between 0 and 2pi
            angle += 2 * np.pi
        phi = self.ang - angle
        revs = int(phi // self.ucAng)  # number of revolutions through unit cell
        if revs % 2 == 0:  # if even
            theta = phi - self.ucAng * revs
        else:  # if odd
            theta = self.ucAng - (phi - self.ucAng * revs)
        r = np.sqrt(x ** 2 + y ** 2)
        x = r * np.cos(theta)  # cartesian coords in unit cell frame
        y = r * np.sin(theta)  # cartesian coords in unit cell frame
        return x, y, z

    def is_Coord_Inside_Vacuum(self, x, y, z):
        phi = full_Arctan2(y, x)  # calling a fast numba version that is global
        if phi < self.ang:  # if particle is inside bending angle region
            if (np.sqrt(x ** 2 + y ** 2) - self.rb) ** 2 + z ** 2 < self.ap ** 2:
                return True
            else:
                return False
        else:  # if outside bender's angle range
            if (x - self.rb) ** 2 + z ** 2 <= self.ap ** 2 and (0 >= y >= -self.Lcap):  # If inside the cap on
                # eastward side
                return True
            else:
                qTestx = self.RIn_Ang[0, 0] * x + self.RIn_Ang[0, 1] * y
                qTesty = self.RIn_Ang[1, 0] * x + self.RIn_Ang[1, 1] * y
                if (qTestx - self.rb) ** 2 + z ** 2 <= self.ap ** 2 and (
                        self.Lcap >= qTesty >= 0):  # if on the westwards side
                    return True
                else:  # if not in either cap, then outside the bender
                    return False

    def magnetic_Potential(self, x0, y0, z0):
        # magnetic potential at point q in element frame
        # q: particle's position in element frame
        x, y, z = x0, y0, z0
        if self.is_Coord_Inside_Vacuum(x, y, z) == False:
            return np.nan
        z = abs(z)
        phi = full_Arctan2(y, x)  # calling a fast numba version that is global
        if phi < self.ang:  # if particle is inside bending angle region
            revs = int((self.ang - phi) // self.ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position = 'FIRST'
            elif revs == self.numMagnets * 2 - 1 or revs == self.numMagnets * 2 - 2:
                position = 'LAST'
            else:
                position = 'INNER'
            if position == 'INNER':
                quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(x, y, z)  # get unit cell coords
                V0 = self._magnetic_Potential_Func_Seg(quc[0], quc[1], quc[2])
            elif position == 'FIRST' or position == 'LAST':
                V0 = self.magnetic_Potential_First_And_Last(x, y, z, position)
            else:
                V0 = np.nan
        elif phi > self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < x < self.rb + self.ap) and (0 > y > -self.Lcap):  # If inside the cap on
                # eastward side
                V0 = self._magnetic_Potential_Func_Cap(x, y, z)
            else:
                xTest = self.RIn_Ang[0, 0] * x + self.RIn_Ang[0, 1] * y
                yTest = self.RIn_Ang[1, 0] * x + self.RIn_Ang[1, 1] * y
                if (self.rb - self.ap < xTest < self.rb + self.ap) and (
                        self.Lcap > yTest > 0):  # if on the westwards side
                    yTest = -yTest
                    V0 = self._magnetic_Potential_Func_Cap(xTest, yTest, z)
                else:  # if not in either cap
                    V0 = np.nan
        if self.useFieldPerturbations == True and np.isnan(V0) == False:
            deltaV = self._magnetic_Potential_Func_Perturbation(x0, y0, z0)  # extra force from design imperfections
            V0 = V0 + deltaV
        V0 *= self.fieldFact
        return V0

    def magnetic_Potential_First_And_Last(self, x, y, z, position):
        if position == 'FIRST':
            xNew = self.M_ang[0, 0] * x + self.M_ang[0, 1] * y
            yNew = self.M_ang[1, 0] * x + self.M_ang[1, 1] * y
            V0 = self._magnetic_Potential_Func_Internal_Fringe(xNew, yNew, z)
        elif position == 'LAST':
            V0 = self._magnetic_Potential_Func_Internal_Fringe(x, y, z)
        else:
            raise Exception('INVALID POSITION SUPPLIED')
        return V0

    def update_Element_Perturb_Params(self, shiftY, shiftZ, rotY, rotZ):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        raise NotImplementedError

# for genetic lens
# @numba.njit()
# def genetic_Lens_Force_NUMBA(x,y,z, L,ap,  force_Func):
#     FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
#     FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
#     y = abs(y)  # confine to upper right quadrant
#     z = abs(z)
#     if np.sqrt(y**2+z**2)>ap:
#         return np.nan,np.nan,np.nan
#     if 0<=x <=L/2:
#         x = L/2 - x
#         Fx,Fy,Fz= force_Func(x, y, z)
#         Fx=-Fx
#     elif L/2<x<L:
#         x=x-L/2
#         Fx,Fy,Fz = force_Func(x, y, z)
#     else:
#         return np.nan,np.nan,np.nan
#     Fy = Fy * FySymmetryFact
#     Fz = Fz * FzSymmetryFact
#     return Fx,Fy,Fz
