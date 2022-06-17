import numba
import numpy as np

spec_Drift = [
    ('L', numba.float64),
    ('ap', numba.float64),
    ('inputAngleTilt', numba.float64),
    ('outputAngleTilt', numba.float64)
]


class DriftFieldHelper_Numba:
    """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""

    def __init__(self, L, ap, inputAngleTilt, outputAngleTilt):
        self.L = L
        self.ap = ap
        self.inputAngleTilt, self.outputAngleTilt = inputAngleTilt, outputAngleTilt

    def get_State_Params(self):
        """Helper for a elementPT.Drift. Psuedo-inherits from BaseClassFieldHelper"""
        return (self.L, self.ap, self.inputAngleTilt, self.outputAngleTilt), ()

    def magnetic_Potential(self, x, y, z):
        """Magnetic potential of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan
        return 0.0

    def force(self, x, y, z):
        """Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
        if not self.is_Coord_Inside_Vacuum(x, y, z):
            return np.nan, np.nan, np.nan
        else:
            return 0.0, 0.0, 0.0

    def is_Coord_Inside_Vacuum(self, x, y, z):
        """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
        if self.inputAngleTilt == self.outputAngleTilt == 0.0:  # drift is a simple cylinder
            return 0 <= x <= self.L and np.sqrt(y ** 2 + z ** 2) < self.ap
        else:
            # min max of purely cylinderical portion of drift region
            xMinCylinder = abs(np.tan(self.inputAngleTilt) * self.ap)
            xMaxCylinder = self.L - abs(np.tan(self.outputAngleTilt) * self.ap)
            if xMinCylinder <= x <= xMaxCylinder:  # if in simple straight section
                return np.sqrt(y ** 2 + z ** 2) < self.ap
            else:  # if in the tilted ends, our outside, along x
                xMinDrift, xMaxDrift = -xMinCylinder, self.L + abs(np.tan(self.outputAngleTilt) * self.ap)
                if not xMinDrift <= x <= xMaxDrift:  # if entirely outside
                    return False
                else:  # maybe it's in the tilted slivers now
                    slopeInput, slopeOutput = np.tan(np.pi / 2 + self.inputAngleTilt), np.tan(
                        np.pi / 2 + self.outputAngleTilt)
                    yInput = slopeInput * x
                    yOutput = slopeOutput * x - slopeOutput * self.L
                    if ((slopeInput > 0 and y < yInput) or (slopeInput < 0 and y > yInput)) and x < xMinCylinder:
                        return np.sqrt(y ** 2 + z ** 2) < self.ap
                    elif ((slopeOutput > 0 and y > yOutput) or (slopeOutput < 0 and y < yOutput)) and x > xMaxCylinder:
                        return np.sqrt(y ** 2 + z ** 2) < self.ap
                    else:
                        return False
