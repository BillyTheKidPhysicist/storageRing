import numba
import numpy as np
from numba.experimental import jitclass


@numba.njit()
def full_arctan2(y, x):
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


TupleOf3Floats = tuple[float, float, float]
nanArr7Tuple = tuple([np.ones(1) * np.nan] * 7)
DUMMY_FIELD_DATA_3D = (np.ones(1) * np.nan,) * 7


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


def misalign_Coords(x: float, y: float, z: float, shift_y: float, shift_z: float, rot_angle_y: float,
                    rot_angle_z: float) -> TupleOf3Floats:
    """Model element misalignment by misaligning coords. First do rotations about (0,0,0), then displace. Element
    misalignment has the opposite applied effect. Force will be needed to be rotated"""
    x, y = np.cos(-rot_angle_z) * x - np.sin(-rot_angle_z) * y, np.sin(-rot_angle_z) * x + np.cos(
        -rot_angle_z) * y  # rotate about z
    x, z = np.cos(-rot_angle_y) * x - np.sin(-rot_angle_y) * z, np.sin(-rot_angle_y) * x + np.cos(
        -rot_angle_y) * z  # rotate about y
    y -= shift_y
    z -= shift_z
    return x, y, z


def rotate_Force(Fx: float, Fy: float, Fz: float, rot_angle_y: float, rot_angle_z: float) -> TupleOf3Floats:
    """After rotating and translating coords to model element misalignment, the force must now be rotated as well"""
    Fx, Fy = np.cos(rot_angle_z) * Fx - np.sin(rot_angle_z) * Fy, np.sin(rot_angle_z) * Fx + np.cos(
        rot_angle_z) * Fy  # rotate about z
    Fx, Fz = np.cos(rot_angle_y) * Fx - np.sin(rot_angle_y) * Fz, np.sin(rot_angle_y) * Fx + np.cos(
        rot_angle_y) * Fz  # rotate about y
    return Fx, Fy, Fz
