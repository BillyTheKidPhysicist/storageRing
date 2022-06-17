import numba
import numpy as np

from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, interp2D, scalar_interp3D
from numbaFunctionsAndObjects.utilities import tupleOf3Floats, nanArr7Tuple

spec_Lens_Halbach = [
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
    ('useFieldPerturbations', numba.boolean)
]


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

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldData = self.xArrEnd, self.yArrEnd, self.zArrEnd, self.FxArrEnd, self.FyArrEnd, self.FzArrEnd, self.VArrEnd, \
                    self.xArrIn, self.yArrIn, self.FxArrIn, self.FyArrIn, self.VArrIn
        fieldPerturbationData = None if not self.useFieldPerturbations else self.fieldPerturbationData
        return (fieldData, fieldPerturbationData, self.L, self.Lcap, self.ap, self.extraFieldLength), (self.fieldFact,)

    def set_Internal_State(self, params):
        self.fieldFact = params[0]

    def is_Coord_Inside_Vacuum(self, x: float, y: float, z: float) -> bool:
        """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
        return 0 <= x <= self.L and y ** 2 + z ** 2 < self.ap ** 2

    def _magnetic_Potential_Func_Fringe(self, x: float, y: float, z: float, useImperfectInterp: bool = False) -> float:
        """Wrapper for interpolation of magnetic fields at ends of lens. see self.magnetic_Potential"""
        if not useImperfectInterp:
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
        if not useImperfectInterp:
            Fx0, Fy0, Fz0 = vec_interp3D(-z, y, x, self.xArrEnd, self.yArrEnd, self.zArrEnd,
                                         self.FxArrEnd, self.FyArrEnd, self.FzArrEnd)
        else:
            xArr, yArr, zArr, FxArr, FyArr, FzArr, VArr = self.fieldPerturbationData
            Fx0, Fy0, Fz0 = vec_interp3D(-z, y, x, xArr, yArr, zArr, FxArr, FyArr, FzArr)
        Fx = Fz0
        Fy = Fy0
        Fz = -Fx0
        return Fx, Fy, Fz

    def _force_Func_Inner(self, y: float, z: float) -> tupleOf3Floats:
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
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan, np.nan, np.nan
        # x, y, z = self.baseClass.misalign_Coords(x, y, z)
        FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if -self.extraFieldLength <= x <= self.Lcap:  # at beginning of lens
            x = self.Lcap - x
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
            Fx = -Fx
        elif self.Lcap < x <= self.L - self.Lcap:  # if long enough, model interior as uniform in x
            Fx, Fy, Fz = self._force_Func_Inner(y, z)
        elif self.L - self.Lcap <= x <= self.L + self.extraFieldLength:  # at end of lens
            x = self.Lcap - (self.L - x)
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
        else:
            raise Exception("Particle outside field region")  # this may be triggered when itentionally misligned
        Fx *= self.fieldFact
        Fy *= FySymmetryFact * self.fieldFact
        Fz *= FzSymmetryFact * self.fieldFact
        # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        return Fx, Fy, Fz

    def _force_Field_Perturbations(self, x0: float, y0: float, z0: float) -> tupleOf3Floats:
        if not self.is_Coord_Inside_Vacuum(x0, y0, z0):
            return np.nan, np.nan, np.nan
        # x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x, y, z = x0, y0, z0
        x = x - self.L / 2
        Fx, Fy, Fz = self._force_Func_Outer(x, y, z,
                                            useImperfectInterp=True)  # being used to hold fields for entire lens
        Fx = Fx * self.fieldFact
        Fy = Fy * self.fieldFact
        Fz = Fz * self.fieldFact
        # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        return Fx, Fy, Fz

    def magnetic_Potential(self, x: float, y: float, z: float) -> float:
        """Magnetic potential of lithium atom. Functions to combine perfect potential and extra potential from
        imperfections. Perturbation potential is messed up potential minus perfect potential."""

        V = self._magnetic_Potential(x, y, z)
        if self.useFieldPerturbations:
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
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan
        # x, y, z = self.baseClass.misalign_Coords(x, y, z)
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
        V0 *= self.fieldFact
        return V0

    def _magnetic_Potential_Perturbations(self, x0: float, y0: float, z0: float) -> float:
        if not self.is_Coord_Inside_Vacuum(x0, y0, z0):
            return np.nan
        # x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x, y, z = x0, y0, z0
        x = x - self.L / 2
        V0 = self._magnetic_Potential_Func_Fringe(x, y, z, useImperfectInterp=True)
        return V0
