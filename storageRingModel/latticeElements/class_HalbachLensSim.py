import warnings
from math import tan, sqrt, inf
from typing import Optional

import numpy as np

from constants import TUBE_WALL_THICKNESS
from helperTools import is_close_all
from helperTools import round_and_make_odd
from latticeElements.Magnets import MagneticLens
from latticeElements.class_LensIdeal import LensIdeal
from latticeElements.utilities import MAGNET_ASPECT_RATIO, is_even, \
    ElementTooShortError, halbach_magnet_width, round_down_to_nearest_tube_OD, B_GRAD_STEP_SIZE, TINY_INTERP_STEP, \
    INTERP_MAGNET_OFFSET
from numbaFunctionsAndObjects import halbachLensFastFunctions


# todo: the structure here is confusing and brittle because of the extra field length logic


class HalbachLensSim(LensIdeal):
    fringe_frac_outer: float = 1.5
    fringe_frac_inner_min = 4.0  # if the total hard edge magnet length is longer than this value * rp, then it can

    # can safely be modeled as a magnet "cap" with a 2D model of the interior

    def __init__(self, PTL, rp_layers: tuple, L: Optional[float], ap: Optional[float],
                 magnet_widths: Optional[tuple]):
        assert all(rp > 0 for rp in rp_layers)
        # if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        # to accomdate the new rp such as force values and positions
        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        # assert self.fringe_frac_outer == 1.5 and self.fringe_frac_inner_min == 4.0, "May need to change numgrid points if " \
        #                                                                             "this changes"
        self.rp = min(rp_layers)
        self.numGridPointsX = round_and_make_odd(21 * PTL.field_dens_mult)
        self.num_grid_points_r = round_and_make_odd(25 * PTL.field_dens_mult)
        self.fringeFieldLength = max(rp_layers) * self.fringe_frac_outer
        super().__init__(PTL, L, None, self.rp,
                         None)  # todo: there should be multiple inheritance here for geometries
        self.magnet_widths = self.make_or_check_magnet_widths(rp_layers, magnet_widths)
        self.ap = self.max_valid_aperture() if ap is None else ap
        assert self.ap <= self.max_interp_radius()
        assert self.ap > 5 * self.rp / self.num_grid_points_r  # ap shouldn't be too small. Value below may be dubiuos from interpolation
        self.rp_layers = rp_layers  # can be multiple bore radius for different layers
        self.Lm = None
        self.L_cap: Optional[float] = None  # todo: ridiculous and confusing name
        self.extra_field_length: Optional[float] = None  # extra field added to end of lens to account misalignment
        self.individualMagnetLength = None
        self.magnet = None
        # or down

    def max_interp_radius(self) -> float:
        """ from geometric arguments of grid inside circle.
        imagine two concentric rings on a grid, such that no grid box which has a portion outside the outer ring
        has any portion inside the inner ring. This is to prevent interpolation reaching into magnetic material"""
        # todo: why is this so different from the combiner version? It should be like that version instead
        ap_max = (self.rp - INTERP_MAGNET_OFFSET) * (1 - sqrt(2) / (self.num_grid_points_r - 1))
        return ap_max

    def fill_pre_constrained_parameters(self):
        pass

    def fill_post_constrained_parameters(self):
        self.set_extra_field_length()
        self.fill_geometric_params()
        self.magnet = MagneticLens(self.Lm, self.rp_layers, self.magnet_widths, self.PTL.magnet_grade,
                                   self.PTL.use_solenoid_field, self.fringeFieldLength)
        self.magnet.fill_position_and_orientation_params(self.r1, self.r2, self.nb, self.ne)

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

    def set_extra_field_length(self) -> None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped"""

        jitter_amp = self.get_valid_jitter_amplitude(Print=True)
        tilt_max = np.arctan(jitter_amp / self.L)
        assert 0.0 <= tilt_max < .1  # small angle. Not sure if valid outside that range
        self.extra_field_length = self.rp * tilt_max * 1.5  # safety factor for approximations

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

        y_min, y_max = -TINY_INTERP_STEP, self.rp - INTERP_MAGNET_OFFSET
        x_min, x_max = -(self.extra_field_length + TINY_INTERP_STEP), self.L_cap + TINY_INTERP_STEP
        num_points_xy, num_points_z = self.num_grid_points_r, self.numGridPointsX
        if not use_symmetry:  # range will have to fully capture lens.
            y_min = -y_max
            x_max = self.L + self.extra_field_length + TINY_INTERP_STEP
            x_min = -(self.extra_field_length + TINY_INTERP_STEP)
            assert self.fringe_frac_outer == 1.5  # pointsperslice mildly depends on this value
            points_per_bore_radius = 5
            num_points_z = round_and_make_odd(max([points_per_bore_radius * self.Lm / self.rp, 2 * num_points_z - 1]))
            assert num_points_z < 150  # things might start taking unreasonably long if not careful
            num_points_xy = 45
        assert not is_even(num_points_xy) and not is_even(num_points_z)
        x_arr = np.linspace(x_min, x_max, num_points_z)
        y_arr_quadrant = np.linspace(y_min, y_max, num_points_xy)
        z_arr_quadrant = y_arr_quadrant.copy()
        return x_arr, y_arr_quadrant, z_arr_quadrant

    def make_unshaped_interp_data_2D(self) -> np.ndarray:

        # ignore fringe fields for interior  portion inside then use a 2D plane to represent the inner portion to
        # save resources
        x_arr, y_arr, z_arr = self.make_grid_coord_arrays(True)
        lens_center = self.magnet.Lm / 2.0 + self.magnet.x_in_offset
        plane_coords = np.asarray(np.meshgrid(lens_center, y_arr, z_arr)).T.reshape(-1, 3)
        B_norm_grad, B_norm = self.magnet.get_valid_field_values(plane_coords, B_GRAD_STEP_SIZE, False)
        data_2D = np.column_stack((plane_coords[:, 1:], B_norm_grad[:, 1:], B_norm))  # 2D is formated as
        # [[x,y,z,B0Gx,B0Gy,B0],..]
        return data_2D

    def make_unshaped_interp_data_3D(self, use_symmetry=True, use_magnet_errors=False,
                                     extra_magnets=None) -> np.ndarray:
        """
        Make 3d field data for interpolation from end of lens region

        If the lens is sufficiently long compared to bore radius then this is only field data from the end region
        (fringe frields and interior near end) because the interior region is modeled as a single plane to exploit
        longitudinal symmetry. Otherwise, it is exactly half of the lens and fringe fields

        """
        assert not (use_magnet_errors and use_symmetry)
        x_arr, y_arr, z_arr = self.make_grid_coord_arrays(use_symmetry)

        volume_coords = np.asarray(np.meshgrid(x_arr, y_arr, z_arr)).T.reshape(-1, 3)  # note that these coordinates
        # can have the wrong value for z if the magnet length is longer than the fringe field effects.
        B_norm_grad, B_norm = self.magnet.get_valid_field_values(volume_coords, B_GRAD_STEP_SIZE,
                                                                 use_magnet_errors, extra_magnets=extra_magnets)
        data_3D = np.column_stack((volume_coords, B_norm_grad, B_norm))

        return data_3D

    def make_interp_data_ideal(self) -> tuple[tuple, tuple]:
        exploit_very_long_lens = True if self.effective_material_length() < self.Lm else False
        interp_data_3D = self.shape_field_data_3D(self.make_unshaped_interp_data_3D())

        if exploit_very_long_lens:
            interp_data_2D = self.shape_field_data_2D(self.make_unshaped_interp_data_2D())
        else:
            interp_data_2D = (np.ones(1) * np.nan,) * 5  # dummy data to make Numba happy

        y_arr, z_arr = interp_data_3D[1], interp_data_3D[2]
        max_grid_sep = np.sqrt(2) * (y_arr[1] - y_arr[0])
        assert self.rp - B_GRAD_STEP_SIZE - max_grid_sep > self.max_interp_radius()
        return interp_data_2D, interp_data_3D

    def make_interp_data(self, apply_perturbation, extra_magnets) -> tuple[tuple, tuple, tuple]:

        data3D_no_perturb = (np.ones(1) * np.nan,) * 7

        field_data_2D, field_data_3D = self.make_interp_data_ideal()
        field_data_perturbations = self.make_field_perturbation_data(extra_magnets) if apply_perturbation \
            else data3D_no_perturb
        return field_data_3D, field_data_2D, field_data_perturbations

    def build_fast_field_helper(self) -> None:
        """Generate magnetic field gradients and norms for numba jitclass field helper. Low density sampled imperfect
        data may added on top of high density symmetry exploiting perfect data. """
        extra_magnets = None
        apply_perturbation = True if self.PTL.use_mag_errors or extra_magnets is not None else False
        field_data = self.make_interp_data(apply_perturbation, extra_magnets)

        numba_func_constants = (
            self.L, self.ap, self.L_cap, self.extra_field_length, self.field_fact, apply_perturbation)

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(halbachLensFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.L_cap, self.ap / 2, .0])))
        assert F_edge / F_center < .015

    def make_field_perturbation_data(self, extra_magnets) -> tuple:
        """Make data for fields coming from magnet imperfections and misalingnmet. Imperfect field values are calculated
        and perfect fiel values are subtracted. The difference is then added later on top of perfect field values. This
        force is small, and so I can get away with interpolating with low density, while keeping my high density
        symmetry region. interpolation points inside magnet material are set to zero, so the interpolation may be poor
        near bore of magnet. This is done to avoid dealing with mistmatch  between good field region of ideal and
        perturbation interpolation"""
        data_3D_unperturbed = self.make_unshaped_interp_data_3D(use_symmetry=False, use_magnet_errors=False)
        data_3D_perturbed = self.make_unshaped_interp_data_3D(use_symmetry=False,
                                                              use_magnet_errors=self.PTL.use_mag_errors,
                                                              extra_magnets=extra_magnets)

        assert len(data_3D_perturbed) == len(data_3D_unperturbed)
        assert is_close_all(data_3D_perturbed[:, :3], data_3D_unperturbed[:, :3], 1e-12)

        data_3D_perturbed[np.isnan(data_3D_perturbed)] = 0.0
        data_3D_unperturbed[np.isnan(data_3D_unperturbed)] = 0.0
        coords = data_3D_unperturbed[:, :3]
        field_vals_difference = data_3D_perturbed[:, 3:] - data_3D_unperturbed[:, 3:]
        data_3D_diff = np.column_stack((coords, field_vals_difference))
        data_3D_diff[np.isnan(data_3D_diff)] = 0.0
        data_3D_diff = tuple(self.shape_field_data_3D(data_3D_diff))
        return data_3D_diff

    def update_field_fact(self, field_strength_fact: float) -> None:
        """Update value used to model magnet strength tunability. field_fact multiplies force and magnetic potential to
        model increasing or reducing magnet strength """
        warnings.warn("extra field sources are being ignore here. Funcitnality is currently broken")
        self.field_fact = field_strength_fact
        warnings.warn("this method does not account for neigboring magnets!!")
        self.build_fast_field_helper()

    def get_valid_jitter_amplitude(self, Print=False):
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        jitter_amp_proposed = self.PTL.jitter_amp
        assert jitter_amp_proposed >= 0.0
        max_jitter_amp = self.max_interp_radius() - self.ap
        if max_jitter_amp == 0.0 and jitter_amp_proposed != 0.0:
            print('Aperture is set to maximum, no room to misalign element')
        jitter_amp = max_jitter_amp if jitter_amp_proposed > max_jitter_amp else jitter_amp_proposed
        if Print:
            if jitter_amp_proposed == max_jitter_amp and jitter_amp_proposed != 0.0:
                print(
                    'jitter amplitude of:' + str(jitter_amp_proposed) + ' clipped to maximum value:' + str(
                        max_jitter_amp))
        return jitter_amp

    def perturb_element(self, shift_y: float, shift_z: float, rot_angle_y: float, rot_angle_z: float) -> None:
        """Overrides abstract method from Element. Add catches for ensuring particle stays in good field region of
        interpolation"""

        raise NotImplementedError

        if self.PTL.jitter_amp == 0.0 and self.PTL.jitter_amp != 0.0:
            warnings.warn("No jittering was accomodated for, so their will be no effect")
        assert abs(rot_angle_z) < .05 and abs(rot_angle_z) < .05  # small angle
        totalshift_y = shift_y + tan(rot_angle_z) * self.L
        totalshift_z = shift_z + tan(rot_angle_y) * self.L
        total_shift = sqrt(totalshift_y ** 2 + totalshift_z ** 2)
        max_shift = self.get_valid_jitter_amplitude()
        if total_shift > max_shift:
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            safety_fact = .95 * max_shift / total_shift  # safety factor
            print('proposed', total_shift, 'new', safety_fact * max_shift)
            shift_y, shift_z, rot_angle_y, rot_angle_z = [val * safety_fact for val in
                                                          [shift_y, shift_z, rot_angle_y, rot_angle_z]]
        self.fast_field_helper.update_Element_Perturb_Params(shift_y, shift_z, rot_angle_y, rot_angle_z)
