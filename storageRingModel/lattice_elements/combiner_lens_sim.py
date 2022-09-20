"""
Contains simulated hexapole lens combiner. Particle orbit is deflected by traveling through the lens offset
from the centerline.
"""

from math import sin, cos, tan, isclose
from typing import Optional

import numpy as np

from constants import MIN_MAGNET_MOUNT_THICKNESS, COMBINER_TUBE_WALL_THICKNESS
from field_generators import Collection
from helper_tools import is_close_all, arr_product, round_and_make_odd, is_odd_length
from lattice_elements.combiner_ideal import CombinerIdeal
from lattice_elements.element_magnets import MagneticLens
from lattice_elements.utilities import (CombinerDimensionError, CombinerIterExceededError,
                                        get_halbach_layers_radii_and_magnet_widths, round_down_to_nearest_tube_OD,
                                        TINY_INTERP_STEP,
                                        B_GRAD_STEP_SIZE, INTERP_MAGNET_MATERIAL_OFFSET, shape_field_data_3D)
from lattice_elements.utilities import STATE_FIELD_FACT
from numba_functions_and_objects import combiner_lens_sim_numba_functions
from numba_functions_and_objects.utilities import DUMMY_FIELD_DATA_3D
from type_hints import ndarray


# IMPROVEMENT: MOVE THE CHARACTERIZE STUFF FROM HERE TO THE combiner_characterizer
# IMPROVEMENT: THERE IS SOME GOOFINESS GOING ON WITH THE ATOM STATE STUFF AND COMBINER CHARACTERIZER

def make_arrays(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z) -> tuple[ndarray, ndarray, ndarray]:
    assert x_max > x_min and y_max > y_min and z_max > z_min
    x_arr = np.linspace(x_min, x_max, num_x)  # this becomes z in element frame, with sign change
    y_arr = np.linspace(y_min, y_max, num_y)  # this remains y in element frame
    z_arr = np.linspace(z_min, z_max, num_z)  # this becomes x in element frame
    assert all(is_odd_length(values) for values in (x_arr, y_arr, z_arr))
    return x_arr, y_arr, z_arr


class CombinerLensSim(CombinerIdeal):
    fringe_frac_outer: float = 1.5
    num_grid_points_r: int = 30
    num_points_per_rad_in_x: int = 15

    def __init__(self, PTL, Lm: float, rp: float, load_beam_offset: float, num_layers: int, ap: Optional[float],
                 seed, atom_state):
        # PTL: object of ParticleTracerLatticeClass
        # Lm: hardedge length of magnet.
        # load_beam_offset: Expected diameter of loading beam. Used to set the maximum combiner bending
        # layers: Number of concentric layers
        # mode: wether storage ring or injector. Injector uses high field seeking, storage ring used low field seeking

        assert all(val > 0 for val in (Lm, rp, load_beam_offset, num_layers))
        assert ap < rp if ap is not None else True
        CombinerIdeal.__init__(self, PTL, Lm, None, None, None, None, None, 1.0, atom_state)

        # ----num points depends on a few paremters to be the same as when I determined the optimal values

        self.Lm = Lm
        self.rp = rp
        self.num_layers = num_layers
        self.ap = ap
        self.load_beam_offset = load_beam_offset
        self.PTL = PTL
        self.magnet_widths = None
        self.field_fact: float = STATE_FIELD_FACT[atom_state]
        self.space = None
        self.magnet: MagneticLens = None

        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet

        self.shape: str = 'COMBINER_CIRCULAR'
        self.input_offset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

        self.acceptance_width = None
        self.seed = seed

    def fill_pre_constrained_parameters(self) -> None:
        rp_layers, magnet_widths = get_halbach_layers_radii_and_magnet_widths(self.rp, self.num_layers)
        self.magnet_widths = magnet_widths
        self.space = max(rp_layers) * self.fringe_frac_outer
        self.ap = self.max_valid_aperture() if self.ap is None else self.ap
        assert self.is_apeture_valid(self.ap)

        self.magnet = MagneticLens(self.Lm, rp_layers, magnet_widths, self.PTL.magnet_grade,
                                   self.PTL.use_solenoid_field, self.space, seed=self.seed)

        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input in a straight line. This is that section
        # or down
        # because field values are computed twice from same lens. Otherwise, magnet errors would change
        input_ang, input_offset, trajectory_length = self.compute_input_orbit_characteristics()
        self.Lo = trajectory_length
        self.ang = input_ang
        self.La = (input_offset + self.space / tan(input_ang)) / (
                sin(input_ang) + cos(input_ang) ** 2 / sin(input_ang))
        self.input_offset = input_offset - tan(
            input_ang) * self.space  # the input offset is measured at the end of the hard edge

        self.L = self.Lo
        self.outer_half_width = max(rp_layers) + max(magnet_widths) + MIN_MAGNET_MOUNT_THICKNESS
        self.acceptance_width = self.get_acceptance_width()

    def fill_post_constrained_parameters(self):
        # The way I fill the coordinates for magnet is kind of crazy and circular for the combiner
        self.make_orbit()

    def get_acceptance_width(self) -> float:
        extra_large_ange = 2 * self.ang
        width_overshoot = self.rp + (self.La + self.rp * sin(abs(extra_large_ange))) * sin(abs(extra_large_ange))
        return width_overshoot

    def is_apeture_valid(self, ap) -> bool:
        return ap <= self.rp - INTERP_MAGNET_MATERIAL_OFFSET and ap <= self.max_ap_internal_interp_region()

    def max_valid_aperture(self):
        vac_tube_OD = 2 * self.rp
        ap_largest_tube = round_down_to_nearest_tube_OD(vac_tube_OD) / 2.0 - COMBINER_TUBE_WALL_THICKNESS
        ap_max_good_interp_region = self.max_ap_internal_interp_region()
        assert ap_largest_tube < ap_max_good_interp_region
        return ap_largest_tube

    def num_points_x_interp(self, x_min: float, x_max: float) -> int:
        assert x_max > x_min
        min_points_x = 11
        num_points_x = round_and_make_odd(
            self.PTL.field_dens_mult * self.num_points_per_rad_in_x * (x_max - x_min) / self.rp)
        return num_points_x if num_points_x >= min_points_x else min_points_x

    def make_internal_symmetry_field_data(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_grid_coords_arrays_internal_symmetry()
        field_data = self.make_field_data(x_arr, y_arr, z_arr)
        return field_data

    def make_full_field_data(self, extra_magnets: Collection = None) -> tuple[ndarray, ...]:
        # this does not work because I am not carefully accounting for how the interpolation region intrudes into
        # the end of the magnet when using a full grid
        x_vals, y_vals, z_vals = self.make_full_grid_coord_arrays()
        x_mag_edge_min, x_mag_edge_max = self.space, self.space + self.Lm
        x_ind_mag_min = np.argwhere(x_vals == x_mag_edge_min)[0][0]
        x_ind_mag_max = np.argwhere(x_vals == x_mag_edge_max)[0][0]
        # temporarily shift x value that lay on the edge of magnet material so interp is still valid
        x_vals_temp = x_vals.copy()
        x_vals_temp[x_ind_mag_min] -= INTERP_MAGNET_MATERIAL_OFFSET
        x_vals_temp[x_ind_mag_max] += INTERP_MAGNET_MATERIAL_OFFSET
        field_data = self.make_field_data(x_vals_temp, y_vals, z_vals, extra_magnets=extra_magnets,
                                          include_mag_errors=self.PTL.include_mag_errors,
                                          include_misalignments=self.PTL.include_misalignments)
        field_data[0][:] = x_vals  # replace the values
        return field_data

    def full_interpolation_length(self) -> float:
        return self.Lb + (self.La + self.acceptance_width * sin(abs(self.ang))) * cos(abs(self.ang))

    def y_z_interpolation_outer_max(self) -> tuple[float, float]:
        z_max = self.rp - INTERP_MAGNET_MATERIAL_OFFSET
        m = abs(np.tan(self.ang))
        y_max = m * z_max + (self.acceptance_width + m * self.Lb) + INTERP_MAGNET_MATERIAL_OFFSET
        return y_max, z_max

    def make_external_symmetry_field_data(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_grid_coords_arrays_external_symmetry()
        x_arr_interp = x_arr.copy()
        assert x_arr_interp[-1] - TINY_INTERP_STEP == self.space  # interp must have overshoot of this size for 
        # trick below to work
        x_arr_interp[-1] -= TINY_INTERP_STEP + INTERP_MAGNET_MATERIAL_OFFSET  # to avoid interpolating inside the
        # magnetic material I cheat here by shifting the point the field is calculated at a tiny bit
        field_data = self.make_field_data(x_arr_interp, y_arr, z_arr)
        field_data[0][:] = x_arr
        return field_data

    def build_fast_field_helper(self, extra_magnets: Collection = None) -> None:
        use_symmetry = False if (
                self.PTL.include_mag_errors or extra_magnets is not None or self.PTL.include_misalignments) else True
        if use_symmetry:
            field_data_internal = self.make_internal_symmetry_field_data()
            field_data_external = self.make_external_symmetry_field_data()
            field_data_full = DUMMY_FIELD_DATA_3D
        else:
            field_data_internal = field_data_external = DUMMY_FIELD_DATA_3D
            field_data_full = self.make_full_field_data(extra_magnets=extra_magnets)

        field_data = (field_data_internal, field_data_external, field_data_full)

        numba_func_constants = (self.ap, self.Lm, self.La, self.Lb, self.space, self.ang, self.acceptance_width,
                                self.field_fact, use_symmetry)

        # IMPROVEMENT: there's repeated code here between modules with the force stuff, not sure if I can sanely remove that

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(combiner_lens_sim_numba_functions, force_args, potential_args,
                                    is_coord_in_vacuum_args)

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lm / 2 + self.space, self.ap / 2, .0])))
        assert F_edge / F_center < .01

    def combiner_x_interp_vals(self) -> np.ndarray:
        """Make x values (through axis) of interpolation grid for full combiner. Trick is that values must extend beyond
        the limits of the combiner interpolation region (x_interp_min,x_interp_max) and line up exactly at the hard 
        edges of the combiner (x_mag_min,x_mag_max)"""
        full_interp_field_length = self.full_interpolation_length()
        magnet_center_x = self.space + self.Lm / 2
        x_interp_min = magnet_center_x - full_interp_field_length / 2.0 - TINY_INTERP_STEP
        x_interp_max = magnet_center_x + full_interp_field_length / 2.0 + TINY_INTERP_STEP
        delta_nominal = self.rp / (self.PTL.field_dens_mult * self.num_points_per_rad_in_x)
        x_mag_min, x_mag_max = self.space, self.space + self.Lm

        intervals_b = round((x_mag_max - x_mag_min) / delta_nominal)
        steps_b = intervals_b + 1
        x_vals_b = np.linspace(x_mag_min, x_mag_max, steps_b)
        delta_b = x_vals_b[1] - x_vals_b[0]
        x_vals_c = np.arange(x_mag_max, x_interp_max, delta_b) + delta_b
        vals_a = np.flip(np.arange(x_mag_min, x_interp_min, -delta_b)) - delta_b
        x_vals = np.concatenate((vals_a, x_vals_b, x_vals_c))
        deltas = np.gradient(x_vals)
        is_close_all(deltas, delta_b, 1e-12)
        assert x_vals[0] < x_interp_min and x_vals[-1] > x_interp_max
        return x_vals

    def make_full_grid_coord_arrays(self) -> tuple[ndarray, ndarray, ndarray]:

        y_max, z_max = self.y_z_interpolation_outer_max()
        y_min, z_min = -y_max, -z_max
        num_z = round_and_make_odd(2 * self.num_grid_points_r * self.PTL.field_dens_mult)
        y_fraction_of_r = y_max / self.rp
        num_y = round_and_make_odd(y_fraction_of_r * 2 * self.num_grid_points_r * self.PTL.field_dens_mult)
        x_arr = self.combiner_x_interp_vals()
        y_arr = np.linspace(y_min, y_max, num_y)
        z_arr = np.linspace(z_min, z_max, num_z)
        return x_arr, y_arr, z_arr

    def make_grid_coords_arrays_external_symmetry(self) -> tuple[ndarray, ndarray, ndarray]:
        # IMPROVEMENT: something is wrong here with the interp stuff. There is out of bounds error very rarely in tests
        x_min = self.space + self.Lm / 2 - self.full_interpolation_length() / 2.0
        x_max = self.space + TINY_INTERP_STEP
        num_x = self.num_points_x_interp(x_min, x_max)
        z_min = - TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max, z_max = self.y_z_interpolation_outer_max()
        num_y = num_z = round_and_make_odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        scaling_factor = y_max / self.rp
        num_y = round_and_make_odd(num_y * scaling_factor)
        return make_arrays(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z)

    def make_grid_coords_arrays_internal_symmetry(self) -> tuple[ndarray, ndarray, ndarray]:

        num_y = num_z = round_and_make_odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        x_min = self.space - TINY_INTERP_STEP
        x_max = self.space + self.Lm / 2.0 + TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max = (self.rp - INTERP_MAGNET_MATERIAL_OFFSET)
        z_min = -TINY_INTERP_STEP
        z_max = (self.rp - INTERP_MAGNET_MATERIAL_OFFSET)
        num_x = self.num_points_x_interp(x_min, x_max)
        return make_arrays(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z)

    def max_ap_internal_interp_region(self) -> float:
        _, y_arr, z_arr = self.make_grid_coords_arrays_internal_symmetry()
        assert max(np.abs(y_arr)) == max(np.abs(z_arr))  # must be same for logic below or using radius
        radius_interp_region = max(np.abs(y_arr))
        assert radius_interp_region < self.rp - B_GRAD_STEP_SIZE  # interp must not reach into material for logic below
        ap_max_interp = radius_interp_region - np.sqrt(2) * (y_arr[1] - y_arr[0])
        return ap_max_interp

    def make_field_data(self, x_arr, y_arr, z_arr, extra_magnets: Collection = None,
                        include_mag_errors: bool = False, include_misalignments: bool = False) -> tuple[ndarray, ...]:
        """Make field data as [[x,y,z,Fx,Fy,Fz,V]..] to be used in fast grid interpolator"""
        volume_coords = arr_product(x_arr, y_arr, z_arr)
        B_norm_grad, B_norm = self.magnet.get_valid_field_values(volume_coords,
                                                                 include_mag_errors=include_mag_errors,
                                                                 extra_magnets=extra_magnets,
                                                                 include_misalignments=include_misalignments)
        field_data_unshaped = np.column_stack((volume_coords, B_norm_grad, B_norm))
        field_data = shape_field_data_3D(field_data_unshaped)
        return field_data

    def compute_input_orbit_characteristics(self) -> tuple:
        """compute characteristics of the input orbit. This applies for injected beam, or recirculating beam"""
        from lattice_elements.combiner_characterizer import characterize_combiner_halbach

        self.output_offset = self.find_Ideal_Offset()

        input_ang, input_offset, trajectory_length, _ = characterize_combiner_halbach(self)
        assert input_ang * self.field_fact > 0  # satisfied if low field is positive angle and high is negative.
        # Sometimes this can be triggered because the lens is to long so an oscilattory behaviour is required by
        # injector
        return input_ang, input_offset, trajectory_length

    def update_field_fact(self, field_strength_fact) -> None:
        raise NotImplementedError

    def find_Ideal_Offset(self) -> float:
        """use newton's method to find where the vertical translation of the combiner wher the minimum seperation
        between atomic beam path and lens is equal to the specified beam diameter for INJECTED beam. This requires
        modeling high field seekers. Particle is traced backwards from the output of the combiner to the input.
        Can possibly error out from modeling magnet or assembly error"""
        from lattice_elements.combiner_characterizer import characterize_combiner_halbach

        if self.load_beam_offset >= self.ap:  # beam doens't fit in combiner
            raise CombinerDimensionError
        y_initial = self.ap / 10.0
        try:
            input_ang, _, _, seperation_initial = characterize_combiner_halbach(self, atom_state='HIGH_FIELD_SEEKER',
                                                                                particleOffset=y_initial)
        except:
            raise CombinerDimensionError
        assert input_ang < 0  # loading beam enters from y<0, if positive then this is circulating beam
        gradient_initial = (seperation_initial - self.ap) / (y_initial - 0.0)
        y = y_initial
        seperation = seperation_initial  # initial value of lens/atom seperation.
        gradient = gradient_initial
        i, iter_max = 0, 20  # to prevent possibility of ifnitne loop
        tol_absolute = 1e-6  # m
        target_sep = self.load_beam_offset
        while not isclose(seperation, target_sep, abs_tol=tol_absolute):
            delta_x = -(seperation - target_sep) / gradient  # I like to use a little damping
            delta_x = -y / 2 if y + delta_x < 0 else delta_x  # restrict deltax to allow value
            y = y + delta_x
            input_ang, _, _, seperation_new = characterize_combiner_halbach(self, atom_state='HIGH_FIELD_SEEKER',
                                                                            particleOffset=y)
            assert input_ang < 0  # loading beam enters from y<0, if positive then this is circulating beam
            gradient = (seperation_new - seperation) / delta_x
            seperation = seperation_new
            i += 1
            if i > iter_max:
                raise CombinerIterExceededError
        assert 0.0 < y < self.ap
        return y
