import numba
import numpy as np

spec_Ideal_Lens = [
    ('L', numba.float64),
    ('K', numba.float64),
    ('ap', numba.float64),
    ('field_fact', numba.float64)
]


class IdealLensFieldHelper:
    """Helper for elementPT.LensIdeal. Psuedo-inherits from BaseClassFieldHelper"""

    def __init__(self, L, K, ap):
        self.L = L
        self.K = K
        self.ap = ap
        self.field_fact = 1.0

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return (self.L, self.K, self.ap), (self.field_fact,)

    def set_Internal_State(self, params):
        self.field_fact = params[0]

    def is_Coord_Inside_Vacuum(self, x: float, y: float, z: float) -> bool:
        """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
        if 0 <= x <= self.L and y ** 2 + z ** 2 < self.ap ** 2:
            return True
        else:
            return False

    def magnetic_potential(self, x: float, y: float, z: float) -> float:
        """Magnetic potential of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if self.is_Coord_Inside_Vacuum(x, y, z):
            r = np.sqrt(y ** 2 + z ** 2)
            V0 = .5 * self.K * r ** 2
        else:
            V0 = np.nan
        V0 = self.field_fact * V0
        return V0

    def force(self, x: float, y: float, z: float) -> tuple:
        """Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if self.is_Coord_Inside_Vacuum(x, y, z) == True:
            Fx = 0.0
            Fy = -self.K * y
            Fz = -self.K * z
            # Fx, Fy, Fz = self.baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
            Fx *= self.field_fact
            Fy *= self.field_fact
            Fz *= self.field_fact
            return Fx, Fy, Fz
        else:
            return np.nan, np.nan, np.nan
