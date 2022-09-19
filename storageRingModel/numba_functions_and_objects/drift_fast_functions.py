import numba
import numpy as np

from numba_functions_and_objects.interpFunctions import force_interp_3D, magnetic_potential_interp_3D
from numba_functions_and_objects.utilities import eps

@numba.njit()
def is_coord_in_vacuum(x, y, z, params):
    #IMPROVEMENT: epsilon could be better implemented here
    ap, L, input_angle_tilt, output_angle_tilt = params
    if input_angle_tilt == output_angle_tilt == 0.0:  # drift is a simple cylinder
        return -eps <= x <= L+eps and np.sqrt(y ** 2 + z ** 2) < ap
    else:
        # min max of purely cylinderical portion of drift region
        x_min_cylinder = abs(np.tan(input_angle_tilt) * ap)-eps
        x_max_cylinder = L - abs(np.tan(output_angle_tilt) * ap)+eps
        if x_min_cylinder <= x <= x_max_cylinder:  # if in simple straight section
            return np.sqrt(y ** 2 + z ** 2) < ap
        else:  # if in the tilted ends, our outside, along x
            x_min_drift, x_max_drift = -x_min_cylinder-eps, L + abs(np.tan(output_angle_tilt) * ap)+eps
            if not x_min_drift <= x <= x_max_drift:  # if entirely outside
                return False
            else:  # maybe it's in the tilted slivers now
                slope_input, slope_output = np.tan(np.pi / 2 + input_angle_tilt), np.tan(
                    np.pi / 2 + output_angle_tilt)
                y_input = slope_input * x
                y_output = slope_output * x - slope_output * L
                if ((slope_input > 0 and y < y_input) or (slope_input < 0 and y > y_input)) and x < x_min_cylinder:
                    return np.sqrt(y ** 2 + z ** 2) < ap
                elif ((slope_output > 0 and y > y_output) or (slope_output < 0 and y < y_output)) \
                        and x > x_max_cylinder:
                    return np.sqrt(y ** 2 + z ** 2) < ap
                else:
                    return False


@numba.njit()
def magnetic_potential(x, y, z, params, field_data):
    """Magnetic potential of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
    ap, L, inputAngleTilt, outputAngleTilt = params
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan
    else:
        return magnetic_potential_interp_3D(x, y, z, field_data)


@numba.njit()
def force(x, y, z, params, field_data):
    """Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
    ap, L, inputAngleTilt, outputAngleTilt = params
    if not is_coord_in_vacuum(x, y, z, params):
        return np.nan, np.nan, np.nan
    else:
        return force_interp_3D(x, y, z, field_data)
