import numba
import numpy as np
from numba_functions_and_objects.utilities import eps


@numba.njit()
def is_coord_in_vacuum(x: float, y: float, z: float, params) -> bool:
    """Check if coord is inside vacuum tube."""
    K, L, ap, field_fact = params
    return -eps <= x <= L+eps and np.sqrt(y ** 2 + z ** 2) < ap


@numba.njit()
def magnetic_potential(x: float, y: float, z: float, params) -> float:
    """Magnetic potential of Li7 in simulation units at x,y,z."""
    K, L, ap, field_fact = params
    if is_coord_in_vacuum(x, y, z, params):
        r = np.sqrt(y ** 2 + z ** 2)
        V0 = .5 * K * r ** 2
    else:
        V0 = np.nan
    V0 = field_fact * V0
    return V0


@numba.njit()
def force(x: float, y: float, z: float, params) -> tuple:
    """Force on Li7 in simulation units at x,y,z."""
    K, L, ap, field_fact = params
    if is_coord_in_vacuum(x, y, z, params):
        Fx = 0.0
        Fy = -K * y
        Fz = -K * z
        Fx *= field_fact
        Fy *= field_fact
        Fz *= field_fact
        return Fx, Fy, Fz
    else:
        return np.nan, np.nan, np.nan
