import numba
import numpy as np


@numba.njit()
def is_coord_in_vacuum(x, y, z, params):
    ap, L, inputAngleTilt, outputAngleTilt = params
    if inputAngleTilt == outputAngleTilt == 0.0:  # drift is a simple cylinder
        return 0 <= x <= L and np.sqrt(y ** 2 + z ** 2) < ap
    else:
        # min max of purely cylinderical portion of drift region
        xMinCylinder = abs(np.tan(inputAngleTilt) * ap)
        xMaxCylinder = L - abs(np.tan(outputAngleTilt) * ap)
        if xMinCylinder <= x <= xMaxCylinder:  # if in simple straight section
            return np.sqrt(y ** 2 + z ** 2) < ap
        else:  # if in the tilted ends, our outside, along x
            xMinDrift, xMaxDrift = -xMinCylinder, L + abs(np.tan(outputAngleTilt) * ap)
            if not xMinDrift <= x <= xMaxDrift:  # if entirely outside
                return False
            else:  # maybe it's in the tilted slivers now
                slopeInput, slopeOutput = np.tan(np.pi / 2 + inputAngleTilt), np.tan(
                    np.pi / 2 + outputAngleTilt)
                yInput = slopeInput * x
                yOutput = slopeOutput * x - slopeOutput * L
                if ((slopeInput > 0 and y < yInput) or (slopeInput < 0 and y > yInput)) and x < xMinCylinder:
                    return np.sqrt(y ** 2 + z ** 2) < ap
                elif ((slopeOutput > 0 and y > yOutput) or (slopeOutput < 0 and y < yOutput)) and x > xMaxCylinder:
                    return np.sqrt(y ** 2 + z ** 2) < ap
                else:
                    return False


@numba.njit()
def magnetic_potential(x, y, z, params):
    """Magnetic potential of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
    ap, L, inputAngleTilt, outputAngleTilt = params
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan
    return 0.0


@numba.njit()
def force(x, y, z, params):
    """Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
    ap, L, inputAngleTilt, outputAngleTilt = params
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan, np.nan, np.nan
    else:
        return 0.0, 0.0, 0.0
