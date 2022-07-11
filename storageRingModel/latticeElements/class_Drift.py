from typing import Optional

import numpy as np

from constants import TUBE_WALL_THICKNESS
from latticeElements.class_LensIdeal import LensIdeal
from numbaFunctionsAndObjects import driftFastFunctions


class Drift(LensIdeal):
    """
    Simple model of free space. Effectively a cylinderical vacuum tube
    """

    def __init__(self, PTL, L: float, ap: float, outer_half_width: Optional[float],
                 input_tilt_angle: float, output_tilt_angle: float):
        super().__init__(PTL, L, 0, np.inf, ap)  # set Bp to zero and bore radius to infinite
        self.input_tilt_angle, self.output_tilt_angle = input_tilt_angle, output_tilt_angle
        self.outer_half_width = ap + TUBE_WALL_THICKNESS if outer_half_width is None else outer_half_width
        assert self.outer_half_width > ap

    def build_fast_field_helper(self) -> None:
        numba_func_constants = (self.ap, self.L, self.input_tilt_angle, self.output_tilt_angle)

        force_args = (numba_func_constants,)
        potential_args = (numba_func_constants,)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(driftFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.Lo = self.L
