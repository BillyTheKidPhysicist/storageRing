import numba
import numpy as np


@numba.njit()
def is_coord_in_vacuum( x: float, y: float, z: float,params) -> bool:
    """Check if coord is inside vacuum tube. pseudo-overrides BaseClassFieldHelper"""
    K, L, ap, field_fact = params
    if 0 <= x <= L and y ** 2 + z ** 2 < ap ** 2:
        return True
    else:
        return False

@numba.njit()
def magnetic_potential( x: float, y: float, z: float,params) -> float:
    """Magnetic potential of Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
    K, L, ap, field_fact = params
    if is_coord_in_vacuum(x, y, z,params):
        # x, y, z = baseClass.misalign_Coords(x, y, z)
        r = np.sqrt(y ** 2 + z ** 2)
        V0 = .5 * K * r ** 2
    else:
        V0 = np.nan
    V0 = field_fact * V0
    return V0

@numba.njit()
def force( x: float, y: float, z: float,params) -> tuple:
    """Force on Li7 in simulation units at x,y,z. pseudo-overrides BaseClassFieldHelper"""
    K,L,ap,field_fact=params
    if is_coord_in_vacuum(x, y, z,params):
        # x, y, z = baseClass.misalign_Coords(x, y, z)
        Fx = 0.0
        Fy = -K * y
        Fz = -K * z
        # Fx, Fy, Fz = baseClass.rotate_Force_For_Misalignment(Fx, Fy, Fz)
        Fx *= field_fact
        Fy *= field_fact
        Fz *= field_fact
        return Fx, Fy, Fz
    else:
        return np.nan, np.nan, np.nan
