import warnings
from math import sqrt, inf
from typing import Optional

import numpy as np

from constants import TUBE_WALL_THICKNESS
from field_generators import Collection
from helper_tools import round_and_make_odd, is_odd_length
from lattice_elements.element_magnets import MagneticLens
from lattice_elements.lens_ideal import LensIdeal
from lattice_elements.utilities import MAGNET_ASPECT_RATIO, ElementTooShortError, halbach_magnet_width, \
    round_down_to_nearest_tube_OD, B_GRAD_STEP_SIZE, TINY_INTERP_STEP, \
    INTERP_MAGNET_MATERIAL_OFFSET, shape_field_data_2D, shape_field_data_3D
from numba_functions_and_objects import lens_sim_numba_functions
from numba_functions_and_objects.utilities import DUMMY_FIELD_DATA_2D


# IMPROVEMENT: the structure here is confusing and brittle because of the extra field length logic


class HalbachLensSim(LensIdeal):
    fringe_frac_outer: float = 1.5
    fringe_frac_inner_min = 4.0  # if the total hard edge magnet length is longer than this value * rp, then it can
    num_points_per_rp_x = 25
    num_points_r = 25

    # can safely be modeled as a magnet "cap" with a 2D model of the interior

    def __init__(self, PTL, rp_layers: tuple, L: Optional[float], ap: Optional[float],
                 magnet_widths: Optional[tuple]):
        assert all(rp > 0 for rp in rp_layers)
        # if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        # to accomdate the new rp such as force values and positions
        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        self.rp = min(rp_layers)
        # self.numGridPointsX = round_and_make_odd(21 * PTL.field_dens_mult)
        self.fringe_field_length = max(rp_layers) * self.fringe_frac_outer
        super().__init__(PTL, L, None, self.rp, None)
        self.magnet_widths = self.make_or_check_magnet_widths(rp_layers, magnet_widths)
        self.ap = self.max_valid_aperture() if ap is None else ap
        assert self.ap <= self.max_interp_radius()
        self.rp_layers = rp_layers  # can be multiple bore radius for different layers
        self.Lm = None
        self.L_cap: Optional[float] = None  # todo: ridiculous and confusing name
        self.individualMagnetLength = None
        self.magnet = None
        # or down

    def max_interp_radius(self) -> float:
        """ from geometric arguments of grid inside circle.
        imagine two concentric rings on a grid, such that no grid box which has a portion outside the outer ring
        has any portion inside the inner ring. This is to prevent interpolation reaching into magnetic material"""
        # todo: why is this so different from the combiner version? It should be like that version instead
        ap_max = (self.rp - INTERP_MAGNET_MATERIAL_OFFSET) * (1 - sqrt(2) / (self.num_points_r - 1))
        return ap_max

    def fill_pre_constrained_parameters(self):
        pass

    def fill_post_constrained_parameters(self):
        self.fill_geometric_params()
        self.magnet = MagneticLens(self.Lm, self.rp_layers, self.magnet_widths, self.PTL.magnet_grade,
                                   self.PTL.use_solenoid_field, self.fringe_field_length)
        self.make_orbit()

    def set_length(self, L: float) -> None:
        assert L > 0.0
        self.L = L

    def max_valid_aperture(self):
        """Get a valid magnet aperture. This is either limited by the good field region, or the available dimensions
        of standard vacuum tubes if specified"""

        # todo: this may give results which are not limited by aperture, but by interpolation region validity
        vac_tube_OD = 2 * self.rp
        ap_largest_tube = round_down_to_nearest_tube_OD(
            vac_tube_OD) / 2.0 - TUBE_WALL_THICKNESS if self.PTL.use_standard_tube_OD else inf
        ap_largest_interp = self.max_interp_radius()
        ap_valid = ap_largest_tube if ap_largest_tube < ap_largest_interp else ap_largest_interp
        return ap_valid

    def effective_material_length(self) -> float:
        """If a lens is very long, then longitudinal symmetry can possibly be exploited because the interior region
        is effectively isotropic a sufficient depth inside. This is then modeled as a 2d slice, and the outer edges
        as 3D slice"""
        min_effective_material_length = self.fringe_frac_inner_min * max(self.rp_layers)
        return min_effective_material_length if min_effective_material_length < self.Lm else self.Lm

    def make_or_check_magnet_widths(self, rp_layers: tuple[float, ...],
                                    magnet_widths_proposed: Optional[tuple[float, ...]]) \
            -> tuple[float, ...]:
        """
        Return transverse width(w in L x w x w) of individual neodymium permanent magnets used in each layer to
        build lens. Check that sizes are valid

        :param rp_layers: tuple of bore radius of each concentric layer
        :param magnet_widths_proposed: tuple of magnet widths in each concentric layer, or None, in which case the maximum value
            will be calculated based on geometry
        :return: tuple of transverse widths of magnets
        """
        if magnet_widths_proposed is None:
            magnet_widths = tuple(halbach_magnet_width(rp, use_standard_sizes=self.PTL.use_standard_mag_size) for
                                  rp in rp_layers)
        else:
            assert len(magnet_widths_proposed) == len(rp_layers)
            max_magnet_widths = tuple(halbach_magnet_width(rp, magnetSeparation=0.0) for rp in rp_layers)
            assert all(width <= maxWidth for width, maxWidth in zip(magnet_widths_proposed, max_magnet_widths))
            if len(rp_layers) > 1:
                for indexPrev, rp in enumerate(rp_layers[1:]):
                    assert rp >= rp_layers[indexPrev] + magnet_widths_proposed[indexPrev] - 1e-12
            magnet_widths = magnet_widths_proposed
        return magnet_widths

    def fill_geometric_params(self) -> None:
        """Compute dependent geometric values"""
        assert self.L is not None  # must be initialized at this point
        self.Lm = self.L - 2 * self.fringe_frac_outer * max(self.rp_layers)  # hard edge length of magnet
        if self.Lm < .5 * self.rp:  # If less than zero, unphysical. If less than .5rp, this can screw up my assumption
            # about fringe fields
            raise ElementTooShortError
        self.individualMagnetLength = min(
            [(MAGNET_ASPECT_RATIO * min(self.magnet_widths)), self.Lm])  # this may get rounded
        # up later to satisfy that the total length is Lm
        self.Lo = self.L
        self.L_cap = self.effective_material_length() / 2 + self.fringe_frac_outer * max(self.rp_layers)
        mount_thickness = 1e-3  # outer thickness of mount, likely from space required by epoxy and maybe clamp
        self.outer_half_width = max(self.rp_layers) + self.magnet_widths[np.argmax(self.rp_layers)] + mount_thickness

    def make_grid_coord_arrays(self, use_symmetry: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        """

        y_min, y_max = -TINY_INTERP_STEP, self.rp - INTERP_MAGNET_MATERIAL_OFFSET  # must be self.ap so that
        # misalingment stuff works
        x_min, x_max = -TINY_INTERP_STEP, self.L_cap + TINY_INTERP_STEP
        points_per_length = self.num_points_per_rp_x / self.rp
        num_points_x = round_and_make_odd(points_per_length * x_max)
        num_points_r = self.num_points_r
        if not use_symmetry:  # range will have to fully capture lens.
            y_min = -y_max
            x_max = self.L + TINY_INTERP_STEP
            x_min = -TINY_INTERP_STEP
            num_points_x = round_and_make_odd(self.L * points_per_length)
            num_points_x_max = 1001
            if num_points_x > num_points_x_max:
                warnings.warn("number of z points is being truncated.\n desired is " + str(num_points_x) +
                              " but limit is " + str(num_points_x_max))
            num_points_x = np.clip(num_points_x, 0, num_points_x_max)
            num_points_r = round_and_make_odd(self.num_points_r * 2)
        x_arr = np.linspace(x_min, x_max, num_points_x)
        y_arr_quadrant = np.linspace(y_min, y_max, num_points_r)
        z_arr_quadrant = y_arr_quadrant.copy()
        assert all(is_odd_length(val) for val in [x_arr, y_arr_quadrant, z_arr_quadrant])
        return x_arr, y_arr_quadrant, z_arr_quadrant

    def make_unshaped_interp_data_2D(self) -> np.ndarray:

        # ignore fringe fields for interior  portion inside then use a 2D plane to represent the inner portion to
        # save resources
        _, y_arr, z_arr = self.make_grid_coord_arrays(True)
        lens_center = self.magnet.Lm / 2.0 + self.magnet.x_in_offset
        plane_coords = np.asarray(np.meshgrid(lens_center, y_arr, z_arr)).T.reshape(-1, 3)
        B_norm_grad, B_norm = self.magnet.get_valid_field_values(plane_coords)
        data_2D = np.column_stack((plane_coords[:, 1:], B_norm_grad[:, 1:], B_norm))  # 2D is formated as
        # [[x,y,z,B0Gx,B0Gy,B0],..]
        return data_2D

    def make_unshaped_interp_data_3D(self, use_symmetry, use_mag_errors,
                                     extra_magnets: Collection,
                                     include_misalignments) -> np.ndarray:
        """
        Make 3d field data for interpolation from end of lens region

        If the lens is sufficiently long compared to bore radius then this is only field data from the end region
        (fringe frields and interior near end) because the interior region is modeled as a single plane to exploit
        longitudinal symmetry. Otherwise, it is exactly half of the lens and fringe fields

        """
        assert not ((use_mag_errors or include_misalignments or extra_magnets is not None) and use_symmetry)
        x_arr, y_arr, z_arr = self.make_grid_coord_arrays(use_symmetry)

        volume_coords = np.asarray(np.meshgrid(x_arr, y_arr, z_arr)).T.reshape(-1, 3)  # note that these coordinates
        # can have the wrong value for z if the magnet length is longer than the fringe field effects.
        B_norm_grad, B_norm = self.magnet.get_valid_field_values(volume_coords,
                                                                 use_mag_errors=use_mag_errors,
                                                                 extra_magnets=extra_magnets,
                                                                 include_misalignments=include_misalignments)
        data_3D = np.column_stack((volume_coords, B_norm_grad, B_norm))

        return data_3D

    def make_interp_data(self, use_symmetry, extra_magnets) -> tuple[tuple, tuple]:
        exploit_very_long_lens = self.effective_material_length() < self.Lm and use_symmetry

        interp_data_3D = shape_field_data_3D(self.make_unshaped_interp_data_3D(use_symmetry, self.PTL.use_mag_errors,
                                                                               extra_magnets,
                                                                               self.PTL.include_misalignments))

        if exploit_very_long_lens:
            interp_data_2D = shape_field_data_2D(self.make_unshaped_interp_data_2D())
        else:
            interp_data_2D = DUMMY_FIELD_DATA_2D  # dummy data to make Numba happy

        y_arr, z_arr = interp_data_3D[1], interp_data_3D[2]
        max_grid_sep = np.sqrt(2) * (y_arr[1] - y_arr[0])
        assert self.rp - B_GRAD_STEP_SIZE - max_grid_sep > self.max_interp_radius()
        return interp_data_2D, interp_data_3D

    def build_fast_field_helper(self, extra_magnets: Collection = None) -> None:
        """Generate magnetic field gradients and norms for numba jitclass field helper. Low density sampled imperfect
        data may added on top of high density symmetry exploiting perfect data. """
        use_symmetry = not (self.PTL.use_mag_errors or extra_magnets is not None or self.PTL.include_misalignments)
        field_data = self.make_interp_data(use_symmetry, extra_magnets)

        numba_func_constants = (self.L, self.ap, self.L_cap, self.field_fact, use_symmetry)

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(lens_sim_numba_functions, force_args, potential_args, is_coord_in_vacuum_args)

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.L_cap, self.ap / 2, .0])))
        assert F_edge / F_center < .015

    def update_field_fact(self, field_strength_fact: float) -> None:
        """Update value used to model magnet strength tunability. field_fact multiplies force and magnetic potential to
        model increasing or reducing magnet strength """
        warnings.warn("extra field sources are being ignore here. Funcitnality is currently broken")
        self.field_fact = field_strength_fact
        warnings.warn("this method does not account for neigboring magnets!!")
        self.build_fast_field_helper()
