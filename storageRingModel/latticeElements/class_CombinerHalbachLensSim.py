import warnings
from math import sin, sqrt, cos, atan, tan, isclose
from typing import Optional

from latticeElements.Magnets import MagneticLens

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from constants import MIN_MAGNET_MOUNT_THICKNESS, COMBINER_TUBE_WALL_THICKNESS
from helperTools import round_and_make_odd

from latticeElements.class_CombinerIdeal import CombinerIdeal
from latticeElements.utilities import MAGNET_ASPECT_RATIO, CombinerDimensionError, \
    CombinerIterExceededError, is_even, get_halbach_layers_radii_and_magnet_widths, round_down_to_nearest_tube_OD, \
    TINY_INTERP_STEP, B_GRAD_STEP_SIZE, INTERP_MAGNET_OFFSET
from numbaFunctionsAndObjects import combinerHalbachFastFunctions

DEFAULT_SEED = 42
ndarray = np.ndarray


# todo: think much more carefully about interp offset stuff and how it affects aperture, and in which direction it is
# affected

def make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z) -> tuple[
    ndarray, ndarray, ndarray]:
    assert x_max > x_min and y_max > y_min and z_max > z_min
    x_arr = np.linspace(x_min, x_max, num_x)  # this becomes z in element frame, with sign change
    y_arr = np.linspace(y_min, y_max, num_y)  # this remains y in element frame
    z_arr = np.linspace(z_min, z_max, num_z)  # this becomes x in element frame
    assert not is_even(len(x_arr)) and not is_even(len(y_arr)) and not is_even(len(z_arr))
    return x_arr, y_arr, z_arr


class CombinerHalbachLensSim(CombinerIdeal):
    outerFringeFrac: float = 1.5
    num_grid_points_r: int = 30
    pointsPerRadiusX: int = 5

    def __init__(self, PTL, Lm: float, rp: float, load_beam_offset: float, numLayers: int, ap: Optional[float], seed):
        # PTL: object of ParticleTracerLatticeClass
        # Lm: hardedge length of magnet.
        # load_beam_offset: Expected diameter of loading beam. Used to set the maximum combiner bending
        # layers: Number of concentric layers
        # mode: wether storage ring or injector. Injector uses high field seeking, storage ring used low field seeking

        assert all(val > 0 for val in (Lm, rp, load_beam_offset, numLayers))
        assert ap < rp if ap is not None else True
        CombinerIdeal.__init__(self, PTL, Lm, None, None, None, None, None, 1.0)

        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        assert self.outerFringeFrac == 1.5, "May need to change numgrid points if this changes"

        self.Lm = Lm
        self.rp = rp
        self.numLayers = numLayers
        self.ap = ap
        self.load_beam_offset = load_beam_offset
        self.PTL = PTL
        self.magnet_widths = None
        self.field_fact: float = -1.0 if PTL.lattice_type == 'injector' else 1.0
        self.space = None
        self.extra_field_length = 0.0
        self.extraLoadApFrac = 1.5
        self.magnet: MagneticLens=None

        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet

        self.shape: str = 'COMBINER_CIRCULAR'
        self.input_offset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

        self.acceptance_width = None
        self.seed = seed

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        rp_layers, magnet_widths = get_halbach_layers_radii_and_magnet_widths(self.rp, self.numLayers,
                                                                              use_standard_sizes=self.PTL.use_standard_mag_size)
        self.magnet_widths = magnet_widths
        self.space = max(rp_layers) * self.outerFringeFrac
        self.ap = self.max_valid_aperture() if self.ap is None else self.ap
        assert self.is_apeture_valid(self.ap)

        seed = DEFAULT_SEED if self.seed is None else self.seed
        self.magnet=MagneticLens(self.Lm,rp_layers,magnet_widths,self.PTL.magnet_grade,self.PTL.use_solenoid_field,self.space,seed=seed)

        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input in a straight line. This is that section
        # or down
        # because field values are computed twice from same lens. Otherwise, magnet errors would change
        input_ang, input_offset, trajectory_length = self.compute_input_orbit_characteristics()
        self.Lo = trajectory_length
        self.L = self.Lo
        self.ang = input_ang
        self.La = (input_offset + self.space / tan(input_ang)) / (
                sin(input_ang) + cos(input_ang) ** 2 / sin(input_ang))
        self.input_offset = input_offset - tan(
            input_ang) * self.space  # the input offset is measured at the end of the hard edge
        self.outer_half_width = max(rp_layers) + max(magnet_widths) + MIN_MAGNET_MOUNT_THICKNESS
        self.acceptance_width = self.get_acceptance_width()

    def get_acceptance_width(self) -> float:
        extra_large_ange = 2 * self.ang
        width_overshoot = self.rp + (self.La + self.rp * sin(abs(extra_large_ange))) * sin(abs(extra_large_ange))
        return width_overshoot

    def is_apeture_valid(self, ap) -> bool:
        return ap <= self.rp - INTERP_MAGNET_OFFSET and ap <= self.max_ap_internal_interp_region()

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
            self.PTL.field_dens_mult * self.pointsPerRadiusX * (x_max - x_min) / self.rp)
        return num_points_x if num_points_x >= min_points_x else min_points_x

    def make_internal_symmetry_field_data(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_grid_coords_arrays_internal_symmetry()
        field_data = self.make_field_data(x_arr, y_arr, z_arr)
        return field_data

    def make_full_field_data(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_full_grid_coord_arrays()
        field_data = self.make_field_data(x_arr, y_arr, z_arr)
        return field_data

    def make_external_symmetry_field_data(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_grid_coords_arrays_external_symmetry()
        x_arr_interp = x_arr.copy()
        assert x_arr_interp[-1] - TINY_INTERP_STEP == self.space  # interp must have overshoot of this size for 
        # trick below to work
        x_arr_interp[-1] -= TINY_INTERP_STEP + INTERP_MAGNET_OFFSET  # to avoid interpolating inside the magnetic
        # material I cheat here by shifting the point the field is calcuated at a tiny bit
        field_data = self.make_field_data(x_arr_interp, y_arr, z_arr)
        field_data[0][:] = x_arr
        return field_data

    def build_fast_field_helper(self) -> None:
        self.set_extra_field_length()
        if self.PTL.use_mag_errors:
            field_data_internal = self.make_full_field_data()
        else:
            field_data_internal = self.make_internal_symmetry_field_data()
        field_data_external = self.make_external_symmetry_field_data()
        field_data = (field_data_internal, field_data_external)

        use_symmetry = not self.PTL.use_mag_errors

        numba_func_constants = (self.ap, self.Lm, self.La, self.Lb, self.space, self.ang, self.acceptance_width,
                                self.field_fact, use_symmetry, self.extra_field_length)

        # todo: there's repeated code here between modules with the force stuff, not sure if I can sanely remove that

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(combinerHalbachFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lm / 2 + self.space, self.ap / 2, .0])))
        assert F_edge / F_center < .01

    def make_full_grid_coord_arrays(self) -> tuple[ndarray, ndarray, ndarray]:
        # todo: WET
        full_interp_field_length = (
                    self.Lb + (self.La + self.acceptance_width * sin(abs(self.ang))) * cos(abs(self.ang)))
        magnet_center_x = self.space + self.Lm / 2

        x_min = magnet_center_x - full_interp_field_length / 2.0 - TINY_INTERP_STEP
        x_max = magnet_center_x + full_interp_field_length / 2.0 + TINY_INTERP_STEP

        z_min = -(self.rp - INTERP_MAGNET_OFFSET)
        z_max = -z_min
        m = abs(np.tan(self.ang))
        y_max = m * z_max + (self.acceptance_width + m * self.Lb) + TINY_INTERP_STEP
        y_min = -y_max
        num_y0 = num_z = round_and_make_odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        num_y = round_and_make_odd(num_y0 * y_max / self.rp)
        num_x = self.num_points_x_interp(x_min, x_max)
        num_x, num_y, num_z = [round_and_make_odd(val * 2 - 1) for val in (num_x, num_y, num_z)]
        return make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z)

    def make_grid_coords_arrays_external_symmetry(self) -> tuple[ndarray, ndarray, ndarray]:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant

        num_y0 = num_z = round_and_make_odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        # todo: something is wrong here with the interp stuff. There is out of bounds error
        x_min = self.space + self.Lm / 2 - (
                self.Lb + (self.La + self.acceptance_width * sin(abs(self.ang))) * cos(abs(self.ang))) / 2.0
        x_max = self.space + TINY_INTERP_STEP
        z_min = - TINY_INTERP_STEP
        z_max = self.rp - INTERP_MAGNET_OFFSET
        num_x = self.num_points_x_interp(x_min, x_max)
        m = abs(np.tan(self.ang))
        y_max = m * z_max + (self.acceptance_width + m * self.Lb) + INTERP_MAGNET_OFFSET
        y_min = -TINY_INTERP_STEP
        num_y = round_and_make_odd(num_y0 * y_max / self.rp)
        return make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z)

    def make_grid_coords_arrays_internal_symmetry(self) -> tuple[ndarray, ndarray, ndarray]:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant

        num_y = num_z = round_and_make_odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        x_min = self.space - TINY_INTERP_STEP
        x_max = self.space + self.Lm / 2.0 + TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max = (self.rp - INTERP_MAGNET_OFFSET)
        z_min = -TINY_INTERP_STEP
        z_max = (self.rp - INTERP_MAGNET_OFFSET)
        num_x = self.num_points_x_interp(x_min, x_max)
        return make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, num_x, num_y, num_z)

    def max_ap_internal_interp_region(self) -> float:
        _, y_arr, z_arr = self.make_grid_coords_arrays_internal_symmetry()
        assert max(np.abs(y_arr)) == max(np.abs(z_arr))  # must be same for logic below or using radius
        radius_interp_region = max(np.abs(y_arr))
        assert radius_interp_region < self.rp - B_GRAD_STEP_SIZE  # interp must not reach into material for logic below
        ap_max_interp = radius_interp_region - np.sqrt(2) * (y_arr[1] - y_arr[0])
        return ap_max_interp

    def make_field_data(self, x_arr, y_arr, z_arr) -> tuple[ndarray, ...]:
        """Make field data as [[x,y,z,Fx,Fy,Fz,V]..] to be used in fast grid interpolator"""
        volume_coords = np.asarray(np.meshgrid(x_arr, y_arr, z_arr)).T.reshape(-1, 3)
        B_norm_grad, B_norm = np.nan * np.zeros((len(volume_coords), 3)), np.nan * np.zeros(len(volume_coords))
        valid_r = np.linalg.norm(volume_coords[:, 1:], axis=1) <= self.rp - B_GRAD_STEP_SIZE
        valid_x = np.logical_or(volume_coords[:, 0] < self.space - B_GRAD_STEP_SIZE,
                                volume_coords[:, 0] > self.space + self.Lm - B_GRAD_STEP_SIZE)
        valid_indices = np.logical_or(valid_x, valid_r)  # tricky
        field_generator=self.magnet.make_magpylib_magnets(self.PTL.use_mag_errors)
        B_norm_grad[valid_indices], B_norm[valid_indices] = field_generator.B_norm_grad(volume_coords[valid_indices],
                                                                                         return_norm=True,
                                                                                         dx=B_GRAD_STEP_SIZE)
        field_data_unshaped = np.column_stack((volume_coords, B_norm_grad, B_norm))
        field_data = self.shape_field_data_3D(field_data_unshaped)
        return field_data

    def compute_input_orbit_characteristics(self) -> tuple:
        """compute characteristics of the input orbit. This applies for injected beam, or recirculating beam"""
        from latticeElements.combiner_characterizer import characterize_combiner_halbach

        self.output_offset = self.find_Ideal_Offset()

        input_ang, input_offset, trajectory_length, _ = characterize_combiner_halbach(self)
        assert input_ang * self.field_fact > 0  # satisfied if low field is positive angle and high is negative.
        # Sometimes this can be triggered because the lens is to long so an oscilattory behaviour is required by
        # injector
        return input_ang, input_offset, trajectory_length

    def update_field_fact(self, field_strength_fact) -> None:
        raise NotImplementedError

    def get_valid_jitter_amplitude(self, show_warnings=True) -> float:
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        assert self.PTL.jitter_amp >= 0.0
        jitter_amp_proposed = self.PTL.jitter_amp
        max_jitter_amp = self.max_ap_internal_interp_region() - self.ap
        jitter_amp = max_jitter_amp if jitter_amp_proposed > max_jitter_amp else jitter_amp_proposed
        if show_warnings:
            if jitter_amp_proposed == max_jitter_amp and jitter_amp_proposed != 0.0:
                warnings.warn('jitter amplitude of:' + str(jitter_amp_proposed) +
                              ' clipped to maximum value:' + str(max_jitter_amp))

        return jitter_amp

    def set_extra_field_length(self) -> None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped. Misalignment is a translational and/or rotational,
        so extra length needs to be accounted for in the case of rotational."""
        jitter_amp = self.get_valid_jitter_amplitude(show_warnings=True)
        tiltMax1D = atan(jitter_amp / self.L)  # Tilt in x,y can be higher but I only need to consider 1D
        # because interpolation grid is square
        assert tiltMax1D < .05  # insist small angle approx
        self.extra_field_length = self.rp * np.tan(tiltMax1D) * 1.5  # safety factor

    def perturb_element(self, shift_y: float, shift_z: float, rot_angle_y: float, rot_angle_z: float) -> None:
        """Overrides abstract method from Element. Add catches for ensuring particle stays in good field region of
        interpolation"""
        raise NotImplementedError  # need to reimplement the accomodate jitter stuff

        assert abs(rot_angle_z) < .05 and abs(rot_angle_z) < .05  # small angle
        totalshift_y = shift_y + np.tan(rot_angle_z) * self.L
        totalshift_z = shift_z + np.tan(rot_angle_y) * self.L
        totalShift = sqrt(totalshift_y ** 2 + totalshift_z ** 2)
        maxShift = self.get_valid_jitter_amplitude()
        if maxShift == 0.0 and self.PTL.jitter_amp != 0.0:
            warnings.warn("No jittering was accomodated for, so their will be no effect")
        if totalShift > maxShift:
            show_warnings('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact = .95 * maxShift / totalShift  # safety factor
            show_warnings('proposed', totalShift, 'new', reductionFact * totalShift)
            shift_y, shift_z, rot_angle_y, rot_angle_z = [val * reductionFact for val in
                                                          [shift_y, shift_z, rot_angle_y, rot_angle_z]]
        self.fast_field_helper.numbaJitClass.update_Element_Perturb_Params(shift_y, shift_z, rot_angle_y, rot_angle_z)

    def find_Ideal_Offset(self) -> float:
        """use newton's method to find where the vertical translation of the combiner wher the minimum seperation
        between atomic beam path and lens is equal to the specified beam diameter for INJECTED beam. This requires
        modeling high field seekers. Particle is traced backwards from the output of the combiner to the input.
        Can possibly error out from modeling magnet or assembly error"""
        from latticeElements.combiner_characterizer import characterize_combiner_halbach

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
