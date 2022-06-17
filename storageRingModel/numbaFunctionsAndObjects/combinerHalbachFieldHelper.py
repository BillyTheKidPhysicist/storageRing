from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, scalar_interp3D
import numba
import numpy as np

from constants import FLAT_WALL_VACUUM_THICKNESS
from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, scalar_interp3D

spec_Combiner_Halbach = [
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
    ('useSymmetry', numba.boolean)
]


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

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldData = self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr, self.VArr
        return (fieldData, self.La, self.Lb, self.Lm, self.space, self.ap, self.ang, self.fieldFact,
                self.extraFieldLength, self.useSymmetry), ()

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.fieldFact, ()

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
        # x, y, z = self.baseClass.misalign_Coords(x0, y0, z0)
        x, y, z = x0, y0, z0
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
        # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        Fx *= self.fieldFact
        Fy *= self.fieldFact
        Fz *= self.fieldFact
        return Fx, Fy, Fz

    def magnetic_Potential(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan
        # x, y, z = self.baseClass.misalign_Coords(x, y, z)
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
