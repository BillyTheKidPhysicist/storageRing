import numba
import numpy as np

from numbaFunctionsAndObjects.interpFunctions import vec_interp3D, scalar_interp3D

spec_Combiner_Sim = [
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
    ('fieldFact', numba.float64)
]


# @jitclass(spec)
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

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        fieldData = self.xArr, self.yArr, self.zArr, self.FxArr, self.FyArr, self.FzArr, self.VArr
        return (fieldData, self.La, self.Lb, self.Lm, self.space, self.apL, self.apR, self.apz, self.ang,
                self.fieldFact), ()

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
