import numba
import numpy as np

from constants import SIMULATION_MAGNETON


@numba.njit()
def combiner_Ideal_Force(x, y, z, Lm, c1, c2) -> tuple[float, float, float]:
    Fx, Fy, Fz = 0.0, 0.0, 0.0
    if 0 < x < Lm:
        B0 = np.sqrt((c2 * z) ** 2 + (c1 + c2 * y) ** 2)
        Fy = SIMULATION_MAGNETON * c2 * (c1 + c2 * y) / B0
        Fz = SIMULATION_MAGNETON * c2 ** 2 * z / B0
    return Fx, Fy, Fz


spec_Combiner_Ideal = [
    ('c1', numba.float64),
    ('c2', numba.float64),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('ap_left', numba.float64),
    ('ap_right', numba.float64),
    ('apz', numba.float64),
    ('ang', numba.float64),
    ('field_fact', numba.float64)
]


class CombinerIdealFieldHelper_Numba:

    def __init__(self, c1, c2, La, Lb, ap_left, ap_right, apz, ang):
        self.c1 = c1
        self.c2 = c2
        self.La = La
        self.Lb = Lb
        self.ap_left = ap_left
        self.ap_right = ap_right
        self.apz = apz
        self.ang = ang
        self.field_fact = 1.0

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return (self.c1, self.c2, self.La, self.Lb, self.ap_left, self.ap_right, self.apz, self.ang), (self.field_fact,)

    def set_Internal_State(self, params):
        self.field_fact = params[0]

    def get_Internal_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return self.field_fact, (self.field_fact,)

    def force(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan, np.nan, np.nan
        else:
            return self.force_without_isinside_check(x, y, z)

    def force_without_isinside_check(self, x, y, z):
        # force at point q in element frame
        # q: particle's position in element frame
        Fx, Fy, Fz = combiner_Ideal_Force(x, y, z, self.Lb, self.c1, self.c2)
        Fx *= self.field_fact
        Fy *= self.field_fact
        Fz *= self.field_fact
        return Fx, Fy, Fz

    def magnetic_potential(self, x, y, z):
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan
        if 0 < x < self.Lb:
            V0 = SIMULATION_MAGNETON * np.sqrt((self.c2 * z) ** 2 + (self.c1 + self.c2 * y) ** 2)
        else:
            V0 = 0.0
        V0 *= self.field_fact
        return V0

    def is_Coord_Inside_Vacuum(self, x, y, z):
        # q: coordinate to test in element's frame
        if not -self.apz < z < self.apz:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -self.ap_left < y < self.ap_right and -self.apz < z < self.apz:  # if inside the y (width) apeture
                return True
            else:
                return False
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            m = np.tan(self.ang)
            Y1 = m * x + (self.ap_right - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-self.ap_left - m * self.Lb)
            if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                return True
            elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                return True
            else:
                return False
