import numba
import numpy as np

from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, scalar_interp3D
from numbaFunctionsAndObjects.utilities import nanArr7Tuple, full_Arctan2

spec_Bender_Halbach = [
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
    ('M_uc', numba.float64[:, ::1]),
    ('M_ang', numba.float64[:, ::1]),
    ('fieldFact', numba.float64),
    ('fieldPerturbationData', numba.types.UniTuple(numba.float64[::1], 7)),
    ('useFieldPerturbations', numba.boolean)
]


class SegmentedBenderSimFieldHelper_Numba:

    def __init__(self, fieldDataSeg, fieldDataInternal, fieldDataCap, fieldPerturbationData, ap, ang, ucAng, rb,
                 numMagnets, Lcap):
        self.fieldDataSeg = fieldDataSeg
        self.fieldDataInternal = fieldDataInternal
        self.fieldDataCap = fieldDataCap
        self.ap = ap
        self.ang = ang
        self.ucAng = ucAng
        self.rb = rb
        self.numMagnets = numMagnets
        m = np.tan(self.ucAng)
        self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        m = np.tan(self.ang / 2)
        self.M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        self.Lcap = Lcap
        self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
        self.fieldFact = 1.0
        self.useFieldPerturbations = True if fieldPerturbationData is not None else False  # apply magnet Perturbation data
        self.fieldPerturbationData = fieldPerturbationData if fieldPerturbationData is not None else nanArr7Tuple

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldPerturbationData = None if not self.useFieldPerturbations else self.fieldPerturbationData
        initParams = (
        self.fieldDataSeg, self.fieldDataInternal, self.fieldDataCap, fieldPerturbationData, self.ap, self.ang,
        self.ucAng, self.rb, self.numMagnets, self.Lcap)
        internalParams = (self.fieldFact,)
        return initParams, internalParams

    def set_Internal_State(self, params):
        self.fieldFact = params[0]

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
        phi = full_Arctan2(y, x)  # calling a fast numba version that is global
        cellNum = int(phi / self.ucAng) + 1  # cell number that particle is in, starts at one
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

        x, y, z = x0, y0, z0
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        z = abs(z)
        phi = full_Arctan2(y, x)  # calling a fast numba version that is global
        if phi <= self.ang:  # if particle is inside bending angle region
            rXYPlane = np.sqrt(x ** 2 + y ** 2)  # radius in xy plane
            if np.sqrt((rXYPlane - self.rb) ** 2 + z ** 2) < self.ap:
                psi = self.ang - phi
                revs = int(psi / self.ucAng)  # number of revolutions through unit cell
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
        if self.useFieldPerturbations and not np.isnan(Fx):
            deltaFx, deltaFy, deltaFz = self._force_Func_Perturbation(x0, y0,
                                                                      z0)  # extra force from design imperfections
            Fx, Fy, Fz = Fx + deltaFx, Fy + deltaFy, Fz + deltaFz
        return Fx, Fy, Fz

    def transform_Element_Coords_Into_Unit_Cell_Frame(self, x, y, z):
        phi = self.ang - full_Arctan2(y, x)
        revs = int(phi / self.ucAng)  # number of revolutions through unit cell
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
            return (np.sqrt(x ** 2 + y ** 2) - self.rb) ** 2 + z ** 2 < self.ap ** 2
        else:  # if outside bender's angle range
            if (x - self.rb) ** 2 + z ** 2 <= self.ap ** 2 and (0 >= y >= -self.Lcap):  # If inside the cap on
                # eastward side
                return True
            else:
                qTestx = self.RIn_Ang[0, 0] * x + self.RIn_Ang[0, 1] * y
                qTesty = self.RIn_Ang[1, 0] * x + self.RIn_Ang[1, 1] * y
                return (qTestx - self.rb) ** 2 + z ** 2 <= self.ap ** 2 and (self.Lcap >= qTesty >= 0)
                # if on the westwards side

    def magnetic_Potential(self, x0, y0, z0):
        # magnetic potential at point q in element frame
        # q: particle's position in element frame
        x, y, z = x0, y0, z0
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan
        z = abs(z)
        phi = full_Arctan2(y, x)  # calling a fast numba version that is global
        if phi < self.ang:  # if particle is inside bending angle region
            revs = int((self.ang - phi) / self.ucAng)  # number of revolutions through unit cell
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
        if self.useFieldPerturbations and not np.isnan(V0):
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
