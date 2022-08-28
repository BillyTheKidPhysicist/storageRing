from typing import Optional

import numpy as np

from constants import TUBE_WALL_THICKNESS
from field_generators import ElementMagnetCollection
from helper_tools import arr_product, round_and_make_odd, is_odd_length
from lattice_elements.lens_ideal import LensIdeal
from lattice_elements.utilities import TINY_INTERP_STEP, B_GRAD_STEP_SIZE, shape_field_data_3D
from numba_functions_and_objects import drift_fast_functions
from type_hints import ndarray

POINTS_PER_M_r = 1000  # 1 point per mm
POINTS_PER_M_x = 500  # 1 point per 2 mm
NUM_VALS_MAX_r_AND_x = 101
NUM_VALS_MIN_r_AND_x = 11


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

    def make_field_data(self, extra_magnets: Optional[list]) -> ndarray:

        if extra_magnets is not None and len(extra_magnets) != 0:
            col = ElementMagnetCollection(extra_magnets)

            num_vals_r = round_and_make_odd(self.ap * POINTS_PER_M_r * self.PTL.field_dens_mult)
            num_vals_x = round_and_make_odd(self.ap * POINTS_PER_M_x * self.PTL.field_dens_mult)
            num_vals_r = np.clip(num_vals_r, NUM_VALS_MIN_r_AND_x, NUM_VALS_MAX_r_AND_x)
            num_vals_x = np.clip(num_vals_x, NUM_VALS_MIN_r_AND_x, NUM_VALS_MAX_r_AND_x)
            x_vals = np.linspace(-TINY_INTERP_STEP, self.L + TINY_INTERP_STEP, num_vals_x)
            r_vals = np.linspace(-(self.ap + TINY_INTERP_STEP), (self.ap + TINY_INTERP_STEP), num_vals_r)
            assert is_odd_length(x_vals) and is_odd_length(r_vals)
            coords = arr_product(x_vals, r_vals, r_vals)
            B_norm_grad, B_norm = col.B_norm_grad(coords, return_norm=True, dx=B_GRAD_STEP_SIZE, use_approx=True)

        else:
            big_pos_val = 1e12  # to prevent almost any chance of out of bounds issue with big drift regions
            dummy_pos_vals = [-big_pos_val, 0.0, big_pos_val]
            coords = arr_product(dummy_pos_vals, dummy_pos_vals, dummy_pos_vals)
            B_norm_grad, B_norm = np.zeros((len(coords), 3)), np.zeros(len(coords))
        unshaped_data = np.column_stack((coords, B_norm_grad, B_norm))
        return shape_field_data_3D(unshaped_data)

    def build_fast_field_helper(self, extra_magnets: list = None) -> None:
        numba_func_constants = (self.ap, self.L, self.input_tilt_angle, self.output_tilt_angle)
        field_data = self.make_field_data(extra_magnets)
        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(drift_fast_functions, force_args, potential_args, is_coord_in_vacuum_args)

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.Lo = self.L
        self.make_orbit()
