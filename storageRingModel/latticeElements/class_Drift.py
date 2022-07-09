from typing import Optional, Union

import numpy as np

from numbaFunctionsAndObjects import driftFastFunctions

from constants import TUBE_WALL_THICKNESS
from latticeElements.class_LensIdeal import LensIdeal
from numbaFunctionsAndObjects.fieldHelpers import get_Drift_Field_Helper

# todo: this needs a good scrubbing and refactoring


realNumber = (int, float)
sequence = Union[list, tuple, np.ndarray]

TINY_STEP = 1e-9
TINY_OFFSET = 1e-12  # tiny offset to avoid out of bounds right at edges of element
SMALL_OFFSET = 1e-9  # small offset to avoid out of bounds right at edges of element
MAGNET_ASPECT_RATIO = 4  # length of individual neodymium magnet relative to width of magnet


class Drift(LensIdeal):
    """
    Simple model of free space. Effectively a cylinderical vacuum tube
    """

    def __init__(self, PTL, L: float, ap: float, outerHalfWidth: Optional[float],
                 inputTiltAngle: float, outputTiltAngle: float):
        super().__init__(PTL, L, 0, np.inf, ap)  # set Bp to zero and bore radius to infinite
        self.inputTiltAngle, self.outputTiltAngle = inputTiltAngle, outputTiltAngle
        self.outerHalfWidth = ap + TUBE_WALL_THICKNESS if outerHalfWidth is None else outerHalfWidth
        assert self.outerHalfWidth > ap

    def build_Fast_Field_Helper(self) -> None:

        numba_func_constants = (self.ap,self.L,self.inputTiltAngle,self.outputTiltAngle)

        force_args = (numba_func_constants,)
        potential_args = (numba_func_constants,)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(driftFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)


    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.Lo = self.L
