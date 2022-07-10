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
    ('yArrIn', numba.float64[::1]),
    ('zArrIn', numba.float64[::1]),
    ('FyArrIn', numba.float64[::1]),
    ('FzArrIn', numba.float64[::1]),
    ('VArrIn', numba.float64[::1]),
    ('fieldPerturbationData', numba.types.UniTuple(numba.float64[::1], 7)),
    ('L', numba.float64),
    ('Lcap', numba.float64),
    ('ap', numba.float64),
    ('field_fact', numba.float64),
    ('extra_field_length', numba.float64),
    ('useFieldPerturbations', numba.boolean)
]


class LensHalbachFieldHelper_Numba:
    """Helper for elementPT.HalbachLensSim. Psuedo-inherits from BaseClassFieldHelper"""

    def __init__(self, field_data, fieldPerturbationData, L, Lcap, ap, extra_field_length):
        self.xArrEnd, self.yArrEnd, self.zArrEnd, self.FxArrEnd, self.FyArrEnd, self.FzArrEnd, self.VArrEnd, self.yArrIn, \
        self.zArrIn, self.FyArrIn, self.FzArrIn, self.VArrIn = field_data
        self.L = L
        self.Lcap = Lcap
        self.ap = ap
        self.field_fact = 1.0
        self.extra_field_length = extra_field_length
        self.useFieldPerturbations = True if fieldPerturbationData is not None else False
        self.fieldPerturbationData = fieldPerturbationData if fieldPerturbationData is not None else nanArr7Tuple

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        field_data = self.xArrEnd, self.yArrEnd, self.zArrEnd, self.FxArrEnd, self.FyArrEnd, self.FzArrEnd, self.VArrEnd, \
                    self.yArrIn, self.zArrIn, self.FyArrIn, self.FzArrIn, self.VArrIn
        fieldPerturbationData = None if not self.useFieldPerturbations else self.fieldPerturbationData
        return (field_data, fieldPerturbationData, self.L, self.Lcap, self.ap, self.extra_field_length), (self.field_fact,)

    def set_Internal_State(self, params):
        self.field_fact = params[0]

    def is_Coord_Inside_Vacuum(self, x: float, y: float, z: float) -> bool:
        """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
        return 0 <= x <= self.L and y ** 2 + z ** 2 < self.ap ** 2

    def _magnetic_potential_Func_Fringe(self, x: float, y: float, z: float, useImperfectInterp: bool = False) -> float:
        """Wrapper for interpolation of magnetic fields at ends of lens. see self.magnetic_potential"""
        if not useImperfectInterp:
            V = scalar_interp3D(x, y, z, self.xArrEnd, self.yArrEnd, self.zArrEnd, self.VArrEnd)
        else:
            x_arr, y_arr, zArr, FxArr, FyArr, FzArr, VArr = self.fieldPerturbationData
            V = scalar_interp3D(x, y, z, x_arr, y_arr, zArr, VArr)
        return V

    def _magnetic_potential_Func_Inner(self, x: float, y: float, z: float) -> float:
        """Wrapper for interpolation of magnetic fields of plane at center lens.see self.magnetic_potential"""
        V = interp2D(y, z, self.yArrIn, self.zArrIn, self.VArrIn)
        return V

    def _force_Func_Outer(self, x, y, z, useImperfectInterp=False) -> tupleOf3Floats:
        """Wrapper for interpolation of force fields at ends of lens. see self.force"""
        if not useImperfectInterp:
            Fx, Fy, Fz = vec_interp3D(x, y, z, self.xArrEnd, self.yArrEnd, self.zArrEnd,
                                         self.FxArrEnd, self.FyArrEnd, self.FzArrEnd)
        else:
            x_arr, y_arr, zArr, FxArr, FyArr, FzArr, VArr = self.fieldPerturbationData
            Fx, Fy, Fz = vec_interp3D(x, y, z, x_arr, y_arr, zArr, FxArr, FyArr, FzArr)
        return Fx, Fy, Fz

    def _force_Func_Inner(self, y: float, z: float) -> tupleOf3Floats:
        """Wrapper for interpolation of force fields of plane at center lens. see self.force"""
        Fx = 0.0
        Fy = interp2D(y, z, self.yArrIn, self.zArrIn, self.FyArrIn)
        Fz = interp2D(y, z, self.yArrIn, self.zArrIn, self.FzArrIn)
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
        if -self.extra_field_length <= x <= self.Lcap:  # at beginning of lens
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
        elif self.Lcap < x <= self.L - self.Lcap:  # if long enough, model interior as uniform in x
            Fx, Fy, Fz = self._force_Func_Inner(y, z)
        elif self.L - self.Lcap <= x <= self.L + self.extra_field_length:  # at end of lens
            x = self.L - x
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
            Fx=-Fx
        else:
            raise Exception("Particle outside field region")  # this may be triggered when itentionally misligned
        Fx *= self.field_fact
        Fy *= FySymmetryFact * self.field_fact
        Fz *= FzSymmetryFact * self.field_fact
        # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        return Fx, Fy, Fz

    def _force_Field_Perturbations(self, x0: float, y0: float, z0: float) -> tupleOf3Floats:
        if not self.is_Coord_Inside_Vacuum(x0, y0, z0):
            return np.nan, np.nan, np.nan
        # x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x, y, z = x0, y0, z0
        Fx, Fy, Fz = self._force_Func_Outer(x, y, z,
                                            useImperfectInterp=True)  # being used to hold fields for entire lens
        Fx = Fx * self.field_fact
        Fy = Fy * self.field_fact
        Fz = Fz * self.field_fact
        # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        return Fx, Fy, Fz

    def magnetic_potential(self, x: float, y: float, z: float) -> float:
        """Magnetic potential of lithium atom. Functions to combine perfect potential and extra potential from
        imperfections. Perturbation potential is messed up potential minus perfect potential."""

        V = self._magnetic_potential(x, y, z)
        if self.useFieldPerturbations:
            deltaV = self._magnetic_potential_Perturbations(x, y, z)  # extra potential from design imperfections
            V += deltaV
        return V

    def _magnetic_potential(self, x: float, y: float, z: float) -> float:
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
        if -self.extra_field_length <= x <= self.Lcap:
            V0 = self._magnetic_potential_Func_Fringe(x, y, z)
        elif self.Lcap < x <= self.L - self.Lcap:
            V0 = self._magnetic_potential_Func_Inner(x, y, z)
        elif 0 <= x <= self.L + self.extra_field_length:
            x = self.L - x
            V0 = self._magnetic_potential_Func_Fringe(x, y, z)
        else:
            raise Exception("Particle outside field region")
        V0 *= self.field_fact
        return V0

    def _magnetic_potential_Perturbations(self, x0: float, y0: float, z0: float) -> float:
        if not self.is_Coord_Inside_Vacuum(x0, y0, z0):
            return np.nan
        # x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x, y, z = x0, y0, z0
        V0 = self._magnetic_potential_Func_Fringe(x, y, z, useImperfectInterp=True)
        return V0
