import numpy as np
import pandas as pd

from lattice_elements.combiner_ideal import CombinerIdeal
from numba_functions_and_objects import combiner_quad_sim_numba_function


class CombinerSim(CombinerIdeal):

    def __init__(self, PTL, combiner_file_name: str, mode: str, size_scale: float = 1.0):
        # PTL: particle tracing lattice object
        # combinerFile: File with data with dimensions (n,6) where n is the number of points and each row is
        # (x,y,z,gradxB,gradyB,gradzB,B). Data must have come from a grid. Data must only be from the upper quarter
        # quadrant, ie the portion with z>0 and x< length/2
        # mode: wether the combiner is functioning as a loader, or a circulator.
        # sizescale: factor to scale up or down all dimensions. This modifies the field strength accordingly, ie
        # doubling dimensions halves the gradient
        assert mode in ('injector', 'storage_ring')
        assert size_scale > 0 and isinstance(combiner_file_name, str)
        Lm = .187
        ap_left = .015
        ap_right = .025
        ap_z = 6e-3
        super().__init__(PTL, Lm, np.nan, np.nan, ap_left, ap_right, ap_z, size_scale)
        self.fringeSpace = 5 * 1.1e-2
        self.combiner_file_name = combiner_file_name

    def open_and_shape_field_data(self):
        data = np.asarray(pd.read_csv(self.combiner_file_name, delim_whitespace=True, header=None))

        # use the new size scaling to adjust the provided data
        data[:, :3] = data[:, :3] * self.size_scale  # scale the dimensions
        data[:, 3:6] = data[:, 3:6] / self.size_scale  # scale the field gradient
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        field_data = self.shape_field_data_3D(data)
        return field_data

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        from lattice_elements.combiner_characterizer import characterize_combiner_sim
        self.space = self.fringeSpace * self.size_scale  # extra space past the hard edge on either end to account for fringe fields
        self.ap_left = self.ap_left * self.size_scale
        self.ap_right = self.ap_right * self.size_scale
        self.apz = self.apz * self.size_scale

        input_ang, input_offset, trajectory_length = characterize_combiner_sim(self)
        self.L = self.Lo = trajectory_length
        self.ang = input_ang
        y0 = input_offset
        x0 = self.space
        theta = input_ang
        self.La = (y0 + x0 / np.tan(theta)) / (np.sin(theta) + np.cos(theta) ** 2 / np.sin(theta))

        self.input_offset = input_offset - np.tan(
            input_ang) * self.space  # the input offset is measured at the end of the hard edge

    def fill_post_constrained_parameters(self):
        pass

    def build_fast_field_helper(self, extra_magnets=None) -> None:
        numba_func_constants = (
            self.ang, self.La, self.Lb, self.Lm, self.apz, self.ap_left, self.ap_right, self.space, self.field_fact)

        field_data = self.open_and_shape_field_data()

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(combiner_quad_sim_numba_function, force_args, potential_args,
                                    is_coord_in_vacuum_args)

    def update_field_fact(self, field_strength_fact) -> None:
        raise NotImplementedError
        self.fast_field_helper.numbaJitClass.field_fact = field_strength_fact
        self.field_fact = field_strength_fact
