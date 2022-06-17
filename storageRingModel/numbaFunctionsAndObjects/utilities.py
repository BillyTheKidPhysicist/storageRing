import numba
import numpy as np
from numba.experimental import jitclass


@numba.njit()
def full_Arctan2(y, x):
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


tupleOf3Floats = tuple[float, float, float]
nanArr7Tuple = tuple([np.ones(1) * np.nan] * 7)


class jitclass_Wrapper:
    def __init__(self, initParams, Class, Spec):
        self.numbaJitClass = jitclass(Spec)(Class)(*initParams)
        self.Class = Class
        self.Spec = Spec

    def __getstate__(self):
        jitClassStateParams = self.numbaJitClass.get_State_Params()
        return jitClassStateParams[0], jitClassStateParams[1], self.Class, self.Spec

    def __setstate__(self, state):
        initParams, internalParams, Class, Spec = state
        self.numbaJitClass = jitclass(Spec)(Class)(*initParams)
        if len(internalParams) > 0:
            self.numbaJitClass.set_Internal_State(internalParams)


def misalign_Coords(x: float, y: float, z: float, shiftY: float, shiftZ: float, rotY: float,
                    rotZ: float) -> tupleOf3Floats:
    """Model element misalignment by misaligning coords. First do rotations about (0,0,0), then displace. Element
    misalignment has the opposite applied effect. Force will be needed to be rotated"""
    x, y = np.cos(-rotZ) * x - np.sin(-rotZ) * y, np.sin(-rotZ) * x + np.cos(
        -rotZ) * y  # rotate about z
    x, z = np.cos(-rotY) * x - np.sin(-rotY) * z, np.sin(-rotY) * x + np.cos(
        -rotY) * z  # rotate about y
    y -= shiftY
    z -= shiftZ
    return x, y, z


def rotate_Force(Fx: float, Fy: float, Fz: float, rotY: float, rotZ: float) -> tupleOf3Floats:
    """After rotating and translating coords to model element misalignment, the force must now be rotated as well"""
    Fx, Fy = np.cos(rotZ) * Fx - np.sin(rotZ) * Fy, np.sin(rotZ) * Fx + np.cos(
        rotZ) * Fy  # rotate about z
    Fx, Fz = np.cos(rotY) * Fx - np.sin(rotY) * Fz, np.sin(rotY) * Fx + np.cos(
        rotY) * Fz  # rotate about y
    return Fx, Fy, Fz
