import numba
import numpy as np

from constants import SIMULATION_MAGNETON
from numba_functions_and_objects.utilities import full_arctan2

spec_Bender_Ideal = [
    ('ang', numba.float64),
    ('K', numba.float64),
    ('rp', numba.float64),
    ('rb', numba.float64),
    ('ap', numba.float64),
    ('field_fact', numba.float64)
]


class BenderIdealFieldHelper_Numba:

    def __init__(self, ang, K, rp, rb, ap):
        self.ang = ang
        self.K = K
        self.rp = rp
        self.rb = rb
        self.ap = ap
        self.field_fact = 1.0

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return (self.ang, self.K, self.rp, self.rb, self.ap), (self.field_fact,)

    def set_Internal_State(self, params):
        self.field_fact = params[0]

    def magnetic_potential(self, x, y, z):
        # potential energy at provided coordinates
        # q coords in element frame
        phi = full_arctan2(y, x)
        rPolar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
        rToroidal = np.sqrt((rPolar - self.rb) ** 2 + z ** 2)
        if phi < self.ang and rToroidal < self.ap:
            V0 = .5 * self.K * SIMULATION_MAGNETON * rToroidal ** 2
        else:
            V0 = np.nan
        V0 *= self.field_fact
        return V0

    def force(self, x, y, z):
        # force at point q in element frame
        # q: particle's position in element frame
        phi = full_arctan2(y, x)
        rPolar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
        rToroidal = np.sqrt((rPolar - self.rb) ** 2 + z ** 2)
        if phi < self.ang and rToroidal < self.ap:
            F0 = -self.K * (rPolar - self.rb)  # force in x y plane
            Fx = np.cos(phi) * F0
            Fy = np.sin(phi) * F0
            Fz = -self.K * z
        else:
            Fx, Fy, Fz = np.nan, np.nan, np.nan
        Fx *= self.field_fact
        Fy *= self.field_fact
        Fz *= self.field_fact
        return Fx, Fy, Fz

    def is_Coord_Inside_Vacuum(self, x, y, z):
        phi = full_arctan2(y, x)
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

    def update_Element_Perturb_Params(self, shift_y, shift_z, rot_angle_y, rot_angle_z):
        """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
        raise NotImplementedError
