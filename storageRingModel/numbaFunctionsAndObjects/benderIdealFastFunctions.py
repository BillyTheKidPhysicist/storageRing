import numba
import numpy as np

from constants import SIMULATION_MAGNETON
from numbaFunctionsAndObjects.utilities import full_arctan2

@numba.njit()
def magnetic_potential( x, y, z,params):
    # potential energy at provided coordinates
    # q coords in element frame
    rb, ap, ang,K, fieldFact = params
    phi = full_arctan2(y, x)
    rPolar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
    rToroidal = np.sqrt((rPolar - rb) ** 2 + z ** 2)
    if phi < ang and rToroidal < ap:
        V0 = .5 * K * SIMULATION_MAGNETON * rToroidal ** 2
    else:
        V0 = np.nan
    V0 *= fieldFact
    return V0

@numba.njit()
def force( x, y, z,params):
    # force at point q in element frame
    # q: particle's position in element frame
    rb, ap, ang,K, fieldFact = params
    phi = full_arctan2(y, x)
    rPolar = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
    rToroidal = np.sqrt((rPolar - rb) ** 2 + z ** 2)
    if phi < ang and rToroidal < ap:
        F0 = -K * (rPolar - rb)  # force in x y plane
        Fx = np.cos(phi) * F0
        Fy = np.sin(phi) * F0
        Fz = -K * z
    else:
        Fx, Fy, Fz = np.nan, np.nan, np.nan
    Fx *= fieldFact
    Fy *= fieldFact
    Fz *= fieldFact
    return Fx, Fy, Fz

@numba.njit()
def is_coord_in_vacuum( x, y, z,params):
    rb, ap, ang, K, fieldFact = params
    phi = full_arctan2(y, x)
    if phi < 0:  # constraint to between zero and 2pi
        phi += 2 * np.pi
    if phi <= ang:  # if particle is in bending segment
        rh = np.sqrt(x ** 2 + y ** 2) - rb  # horizontal radius
        r = np.sqrt(rh ** 2 + z ** 2)  # particle displacement from center of apeture
        if r > ap:
            return False
        else:
            return True
    else:
        return False

# def update_Element_Perturb_Params( shiftY, shiftZ, rotY, rotZ):
#     """update rotations and shifts of element relative to vacuum. pseudo-overrides BaseClassFieldHelper"""
#     raise NotImplementedError
