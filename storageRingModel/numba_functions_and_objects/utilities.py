"""
Contains functions, objects and parameters used in Numba methods
"""
import sys

import numba
import numpy as np
from numba.experimental import jitclass


@numba.njit()
def full_arctan2(y, x):
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


eps = sys.float_info.epsilon
eps_fact = (1 + eps)
eps_fact_reduced = (1 - eps)
TupleOf3Floats = tuple[float, float, float]
nanArr7Tuple = tuple([np.ones(1) * np.nan] * 7)
DUMMY_FIELD_DATA_3D = (np.ones(1) * np.nan,) * 7
DUMMY_FIELD_DATA_2D = (np.ones(1) * np.nan,) * 5


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