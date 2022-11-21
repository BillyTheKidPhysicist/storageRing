"""
Contains ideal model of combiner. This is simply a quadrupole and dipole multi-pole combination.
"""
import numpy as np

from lattice_elements.base_element import BaseElement
from lattice_elements.utilities import STATE_FIELD_FACT
from numba_functions_and_objects import combiner_ideal_numba_function


class CombinerIdeal(BaseElement):

    def __init__(self, PTL, Lm: float, c1: float, c2: float, ap_left: float, ap_right: float, ap_z: float,
                 size_scale: float, atom_state: str):
        super().__init__(PTL)
        self.field_fact = STATE_FIELD_FACT[atom_state]
        self.size_scale = size_scale  # the fraction that the combiner is scaled up or down to. A combiner
        # twice the size would use size_scale=2.0
        self.ap_right = ap_right
        self.ap_left = ap_left
        self.apz = ap_z
        self.ap = None
        self.Lm = Lm
        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet
        self.c1 = c1
        self.c2 = c2
        self.space = 0  # space at the end of the combiner to account for fringe fields

        self.shape = 'COMBINER_SQUARE'
        self.input_offset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        from lattice_elements.combiner_characterizer import characterize_combiner_ideal
        self.ap_right, self.ap_left, self.apz, self.Lm = [val * self.size_scale for val in
                                                          (self.ap_right, self.ap_left, self.apz, self.Lm)]
        self.c1, self.c2 = self.c1 / self.size_scale, self.c2 / self.size_scale
        self.Lb = self.Lm  # length of segment after kink after the inlet
        # self.fast_field_helper = get_Combiner_Ideal([self.c1, self.c2, np.nan, self.Lb,
        #                                                   self.ap_left, self.ap_right, np.nan, np.nan])
        input_ang, input_offset, trajectory_length = characterize_combiner_ideal(self)

        self.Lo = trajectory_length  # np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = input_ang
        self.input_offset = input_offset
        self.La = .5 * (self.ap_right + self.ap_left) * np.sin(self.ang)
        self.L = self.La * np.cos(
            self.ang) + self.Lb  # TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING. Is it used?

    def fill_post_constrained_parameters(self):
        pass

    def build_fast_field_helper(self, extra_magnets=None) -> None:
        numba_func_constants = self.c1, self.c2, self.ang, self.La, self.Lb, self.apz, self.ap_left, self.ap_right, self.field_fact
        force_args = (numba_func_constants,)
        potential_args = (numba_func_constants,)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(combiner_ideal_numba_function, force_args, potential_args, is_coord_in_vacuum_args)

    def compute_trajectory_length(self, qTracedArr: np.ndarray) -> float:
        # to find the trajectory length model the trajectory as a bunch of little deltas for each step and add up their
        # length
        x = qTracedArr[:, 0]
        y = qTracedArr[:, 1]
        x_delta = np.append(x[0], x[1:] - x[:-1])  # have to add the first value to the length of difference because
        # it starts at zero
        y_delta = np.append(y[0], y[1:] - y[:-1])
        dL_arr = np.sqrt(x_delta ** 2 + y_delta ** 2)
        Lo = float(np.sum(dL_arr))
        return Lo

    def transform_lab_coords_into_element_frame(self, q_lab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        q_el = self.transform_lab_frame_vector_into_element_frame(q_lab - self.r2)  # a simple vector trick
        return q_el

    def transform_element_coords_into_local_orbit_frame(self, q_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        # NOTE: THIS NOT GOING TO BE CORRECT IN GENERALY BECAUSE THE TRAJECTORY IS NOT SMOOTH AND I HAVE NOT WORKED IT OUT
        # YET
        # IMPROVEMENT: make this work for everyone and be less crude
        if self.orbit_trajectory is not None:
            x_orbit_traj = self.orbit_trajectory[:, 0]
            idx = np.argmin(np.abs(q_el[0] - x_orbit_traj))
            qo = q_el.copy()
            qo[1] = self.orbit_trajectory[idx,1]-qo[1]
            qo[0] = self.Lo - qo[0]
        else:
            qo = q_el.copy()
            qo[0] = self.Lo - qo[0]
        return qo

    def transform_element_momentum_into_local_orbit_frame(self, q_el: np.ndarray, p_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Not supported at the moment, so returns np.nan array instead"""
        # IMPROVEMENT: THIS IS ONLY APPROXIMATELY CORRECT
        return p_el.copy()

    def transform_element_coords_into_lab_frame(self, q_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        q_new = q_el.copy()
        q_new[:2] = self.R_Out @ q_new[:2] + self.r2[:2]
        return q_new

    def transform_orbit_frame_into_lab_frame(self, q_orbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        q_new = q_orbit.copy()
        q_new[0] = -q_new[0]
        q_new[:2] = self.R_Out @ q_new[:2]
        q_new += self.r1
        return q_new
