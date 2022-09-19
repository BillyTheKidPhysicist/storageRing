import numba
import numpy as np

from numba_functions_and_objects.interpFunctions import vec_interp3D, scalar_interp3D
from numba_functions_and_objects.utilities import nanArr7Tuple, full_arctan2

spec_Bender_Halbach = [
    ('field_data_seg', numba.types.UniTuple(numba.float64[::1], 7)),
    ('field_data_internal', numba.types.UniTuple(numba.float64[::1], 7)),
    ('field_data_cap', numba.types.UniTuple(numba.float64[::1], 7)),
    ('ap', numba.float64),
    ('ang', numba.float64),
    ('ucAng', numba.float64),
    ('rb', numba.float64),
    ('num_lenses', numba.float64),
    ('M_uc', numba.float64[:, ::1]),
    ('M_ang', numba.float64[:, ::1]),
    ('L_cap', numba.float64),
    ('RIn_Ang', numba.float64[:, ::1]),
    ('M_uc', numba.float64[:, ::1]),
    ('M_ang', numba.float64[:, ::1]),
    ('field_fact', numba.float64),
    ('fieldPerturbationData', numba.types.UniTuple(numba.float64[::1], 7)),
    ('use_field_perturbations', numba.boolean)
]


class SegmentedBenderSimFieldHelper_Numba:

    def __init__(self, field_data_seg, field_data_internal, field_data_cap, fieldPerturbationData, ap, ang, ucAng, rb,
                 num_lenses, L_cap):
        self.field_data_seg = field_data_seg
        self.field_data_internal = field_data_internal
        self.field_data_cap = field_data_cap
        self.ap = ap
        self.ang = ang
        self.ucAng = ucAng
        self.rb = rb
        self.num_lenses = num_lenses
        m = np.tan(self.ucAng)
        self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        m = np.tan(self.ang / 2)
        self.M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        self.L_cap = L_cap
        self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
        self.field_fact = 1.0
        self.use_field_perturbations = True if fieldPerturbationData is not None else False  # apply magnet Perturbation data
        self.fieldPerturbationData = fieldPerturbationData if fieldPerturbationData is not None else nanArr7Tuple

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldPerturbationData = None if not self.use_field_perturbations else self.fieldPerturbationData
        initParams = (
            self.field_data_seg, self.field_data_internal, self.field_data_cap, fieldPerturbationData, self.ap,
            self.ang,
            self.ucAng, self.rb, self.num_lenses, self.L_cap)
        internalParams = (self.field_fact,)
        return initParams, internalParams

    def set_Internal_State(self, params):
        self.field_fact = params[0]

    def cartesian_To_Center(self, x, y, z):
        """Convert from cartesian coords to HalbachLensClass.BenderSim coored, ie "center coords" for
        evaluation by interpolator"""

        if x > 0.0 and -self.L_cap <= y <= 0.0:
            s = self.L_cap + y
            xc = x - self.rb
            yc = z
        else:
            theta = full_arctan2(y, x)
            if theta <= self.ang:
                s = theta * self.rb + self.L_cap
                xc = np.sqrt(x ** 2 + y ** 2) - self.rb
                yc = z
            elif self.ang < theta <= 2 * np.pi:  # i'm being lazy here and not limiting the real end
                x0, y0 = np.cos(self.ang) * self.rb, np.sin(self.ang) * self.rb
                thetaEndPerp = np.pi - np.arctan(-1 / np.tan(self.ang))
                x, y = x - x0, y - y0
                deltaS, xc = np.cos(thetaEndPerp) * x + np.sin(-thetaEndPerp) * y, np.sin(thetaEndPerp) * x + np.cos(
                    thetaEndPerp) * y
                yc = z
                xc = -xc
                s = (self.ang * self.rb + self.L_cap) + deltaS
            else:
                raise ValueError
        return s, xc, yc

    def _force_Func_Seg(self, x, y, z):
        Fx, Fy, Fz = vec_interp3D(x, y, z, *self.field_data_seg[:6])
        return Fx, Fy, Fz

    def _force_func_internal_fringe(self, x, y, z):
        Fx, Fy, Fz = vec_interp3D(x, y, z, *self.field_data_internal[:6])
        return Fx, Fy, Fz

    def _force_func_perturbation(self, x, y, z):
        s, xc, yc = self.cartesian_To_Center(x, y, z)
        Fx, Fy, Fz = vec_interp3D(s, xc, yc, *self.fieldPerturbationData[:6])
        return Fx, Fy, Fz

    def _force_func_cap(self, x, y, z):
        Fx, Fy, Fz = vec_interp3D(x, y, z, *self.field_data_cap[:6])
        return Fx, Fy, Fz

    def _magnetic_potential_Func_Seg(self, x, y, z):
        return scalar_interp3D(x, y, z, *self.field_data_seg[:3], self.field_data_seg[-1])

    def _magnetic_potential_Func_Internal_Fringe(self, x, y, z):
        return scalar_interp3D(x, y, z, *self.field_data_internal[:3], self.field_data_internal[-1])

    def _magnetic_potential_Func_Cap(self, x, y, z):
        return scalar_interp3D(x, y, z, *self.field_data_cap[:3], self.field_data_cap[-1])

    def _magnetic_potential_Func_Perturbation(self, x, y, z):
        s, xc, yc = self.cartesian_To_Center(x, y, z)
        return scalar_interp3D(s, xc, yc, *self.fieldPerturbationData[:3], self.fieldPerturbationData[-1])

    def transform_Unit_Cell_Force_Into_Element_Frame_NUMBA(self, Fx, Fy, Fz, x, y):
        # transform the coordinates in the unit cell frame into element frame. The crux of the logic is to notice
        # that exploiting the unit cell symmetry requires dealing with the condition where the particle is approaching
        # or leaving the element interface as mirror images of each other.
        # FNew: Force to be rotated out of unit cell frame
        # q: particle's position in the element frame where the force is acting
        phi = full_arctan2(y, x)  # calling a fast numba version that is global
        cellNum = int(phi / self.ucAng) + 1  # cell number that particle is in, starts at one
        if cellNum % 2 == 1:  # if odd number cell. Then the unit cell only needs to be rotated into that position
            rot_angle = 2 * (cellNum // 2) * self.ucAng
        else:  # otherwise it needs to be reflected. This is the algorithm for reflections
            Fx0 = Fx
            Fy0 = Fy
            Fx = self.M_uc[0, 0] * Fx0 + self.M_uc[0, 1] * Fy0
            Fy = self.M_uc[1, 0] * Fx0 + self.M_uc[1, 1] * Fy0
            rot_angle = 2 * ((cellNum - 1) // 2) * self.ucAng
        Fx0 = Fx
        Fy0 = Fy
        Fx = np.cos(rot_angle) * Fx0 - np.sin(rot_angle) * Fy0
        Fy = np.sin(rot_angle) * Fx0 + np.cos(rot_angle) * Fy0
        return Fx, Fy, Fz

    def force(self, x0, y0, z0):
        # force at point q in element frame
        # q: particle's position in element frame

        x, y, z = x0, y0, z0
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        # todo: I think I need to get rid of this symmetry stuff for the magnet imperfections to work right
        z = abs(z)
        phi = full_arctan2(y, x)  # calling a fast numba version that is global
        if phi <= self.ang:  # if particle is inside bending angle region
            rXYPlane = np.sqrt(x ** 2 + y ** 2)  # radius in xy plane
            if np.sqrt((rXYPlane - self.rb) ** 2 + z ** 2) < self.ap:
                psi = self.ang - phi
                revs = int(psi / self.ucAng)  # number of revolutions through unit cell
                if revs == 0 or revs == 1:
                    position = 'FIRST'
                elif revs == self.num_lenses * 2 - 1 or revs == self.num_lenses * 2 - 2:
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
                        Fx, Fy, Fz = self._force_func_internal_fringe(x, y, z)
                        Fx0 = Fx
                        Fy0 = Fy
                        Fx = self.M_ang[0, 0] * Fx0 + self.M_ang[0, 1] * Fy0
                        Fy = self.M_ang[1, 0] * Fx0 + self.M_ang[1, 1] * Fy0
                    else:
                        Fx, Fy, Fz = self._force_func_internal_fringe(x, y, z)
            else:
                Fx, Fy, Fz = np.nan, np.nan, np.nan
        else:  # if outside bender's angle range
            if np.sqrt((x - self.rb) ** 2 + z ** 2) < self.ap and (0 >= y >= -self.L_cap):  # If inside the cap on
                # eastward side
                Fx, Fy, Fz = self._force_func_cap(x, y, z)
            else:
                x, y = self.M_ang[0, 0] * x + self.M_ang[0, 1] * y, self.M_ang[1, 0] * x + self.M_ang[1, 1] * y
                if np.sqrt((x - self.rb) ** 2 + z ** 2) < self.ap and (
                        -self.L_cap <= y <= 0):  # if on the westwards side
                    Fx, Fy, Fz = self._force_func_cap(x, y, z)
                    Fx0 = Fx
                    Fy0 = Fy
                    Fx = self.M_ang[0, 0] * Fx0 + self.M_ang[0, 1] * Fy0
                    Fy = self.M_ang[1, 0] * Fx0 + self.M_ang[1, 1] * Fy0
                else:  # if not in either cap, then outside the bender
                    Fx, Fy, Fz = np.nan, np.nan, np.nan
        Fz = Fz * FzSymmetryFact
        Fx *= self.field_fact
        Fy *= self.field_fact
        Fz *= self.field_fact
        if self.use_field_perturbations and not np.isnan(Fx):
            deltaFx, deltaFy, deltaFz = self._force_func_perturbation(x0, y0,
                                                                      z0)  # extra force from design imperfections
            Fx, Fy, Fz = Fx + deltaFx, Fy + deltaFy, Fz + deltaFz
        return Fx, Fy, Fz

    def transform_Element_Coords_Into_Unit_Cell_Frame(self, x, y, z):
        phi = self.ang - full_arctan2(y, x)
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
        phi = full_arctan2(y, x)  # calling a fast numba version that is global
        if phi < self.ang:  # if particle is inside bending angle region
            return (np.sqrt(x ** 2 + y ** 2) - self.rb) ** 2 + z ** 2 < self.ap ** 2
        else:  # if outside bender's angle range
            if (x - self.rb) ** 2 + z ** 2 <= self.ap ** 2 and (0 >= y >= -self.L_cap):  # If inside the cap on
                # eastward side
                return True
            else:
                qTestx = self.RIn_Ang[0, 0] * x + self.RIn_Ang[0, 1] * y
                qTesty = self.RIn_Ang[1, 0] * x + self.RIn_Ang[1, 1] * y
                return (qTestx - self.rb) ** 2 + z ** 2 <= self.ap ** 2 and (self.L_cap >= qTesty >= 0)
                # if on the westwards side

    def magnetic_potential(self, x0, y0, z0):
        # magnetic potential at point q in element frame
        # q: particle's position in element frame
        x, y, z = x0, y0, z0
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan
        z = abs(z)
        phi = full_arctan2(y, x)  # calling a fast numba version that is global
        if phi < self.ang:  # if particle is inside bending angle region
            revs = int((self.ang - phi) / self.ucAng)  # number of revolutions through unit cell
            if revs == 0 or revs == 1:
                position = 'FIRST'
            elif revs == self.num_lenses * 2 - 1 or revs == self.num_lenses * 2 - 2:
                position = 'LAST'
            else:
                position = 'INNER'
            if position == 'INNER':
                quc = self.transform_Element_Coords_Into_Unit_Cell_Frame(x, y, z)  # get unit cell coords
                V0 = self._magnetic_potential_Func_Seg(quc[0], quc[1], quc[2])
            elif position == 'FIRST' or position == 'LAST':
                V0 = self.magnetic_potential_First_And_Last(x, y, z, position)
            else:
                V0 = np.nan
        elif phi > self.ang:  # if outside bender's angle range
            if (self.rb - self.ap < x < self.rb + self.ap) and (0 > y > -self.L_cap):  # If inside the cap on
                # eastward side
                V0 = self._magnetic_potential_Func_Cap(x, y, z)
            else:
                xTest = self.RIn_Ang[0, 0] * x + self.RIn_Ang[0, 1] * y
                yTest = self.RIn_Ang[1, 0] * x + self.RIn_Ang[1, 1] * y
                if (self.rb - self.ap < xTest < self.rb + self.ap) and (
                        self.L_cap > yTest > 0):  # if on the westwards side
                    yTest = -yTest
                    V0 = self._magnetic_potential_Func_Cap(xTest, yTest, z)
                else:  # if not in either cap
                    V0 = np.nan
        if self.use_field_perturbations and not np.isnan(V0):
            deltaV = self._magnetic_potential_Func_Perturbation(x0, y0, z0)  # extra force from design imperfections
            V0 = V0 + deltaV
        V0 *= self.field_fact
        return V0

    def magnetic_potential_First_And_Last(self, x, y, z, position):
        if position == 'FIRST':
            xNew = self.M_ang[0, 0] * x + self.M_ang[0, 1] * y
            yNew = self.M_ang[1, 0] * x + self.M_ang[1, 1] * y
            V0 = self._magnetic_potential_Func_Internal_Fringe(xNew, yNew, z)
        elif position == 'LAST':
            V0 = self._magnetic_potential_Func_Internal_Fringe(x, y, z)
        else:
            raise Exception('INVALID POSITION SUPPLIED')
        return V0

    def update_Element_Perturb_Params(self, shift_y, shift_z, rot_angle_y, rot_angle_z):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        raise NotImplementedError
