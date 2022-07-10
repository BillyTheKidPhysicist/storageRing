from math import sin, sqrt, cos, atan, tan, isclose
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from constants import MIN_MAGNET_MOUNT_THICKNESS, COMBINER_TUBE_WALL_THICKNESS
from helperTools import round_And_Make_Odd
from latticeElements.class_CombinerIdeal import CombinerIdeal
from latticeElements.utilities import MAGNET_ASPECT_RATIO, CombinerDimensionError, \
    CombinerIterExceededError, is_even, get_halbach_layers_radii_and_magnet_widths, round_down_to_nearest_valid_tube_OD, \
    TINY_INTERP_STEP, B_GRAD_STEP_SIZE, INTERP_MAGNET_OFFSET
from numbaFunctionsAndObjects import combinerHalbachFastFunctions

DEFAULT_SEED = 42
ndarray = np.ndarray


# todo: think much more carefully about interp offset stuff and how it affects aperture, and in which direction it is
# affected

def make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, numX, numY, numZ) -> tuple[
    ndarray, ndarray, ndarray]:
    assert x_max > x_min and y_max > y_min and z_max > z_min
    x_arr = np.linspace(x_min, x_max, numX)  # this becomes z in element frame, with sign change
    y_arr = np.linspace(y_min, y_max, numY)  # this remains y in element frame
    z_arr = np.linspace(z_min, z_max, numZ)  # this becomes x in element frame
    assert not is_even(len(x_arr)) and not is_even(len(y_arr)) and not is_even(len(z_arr))
    return x_arr, y_arr, z_arr


class CombinerHalbachLensSim(CombinerIdeal):
    outerFringeFrac: float = 1.5
    num_grid_points_r: int = 30
    pointsPerRadiusX: int = 5

    def __init__(self, PTL, Lm: float, rp: float, loadBeamOffset: float, numLayers: int, ap: Optional[float], seed):
        # PTL: object of ParticleTracerLatticeClass
        # Lm: hardedge length of magnet.
        # loadBeamOffset: Expected diameter of loading beam. Used to set the maximum combiner bending
        # layers: Number of concentric layers
        # mode: wether storage ring or injector. Injector uses high field seeking, storage ring used low field seeking

        assert all(val > 0 for val in (Lm, rp, loadBeamOffset, numLayers))
        assert ap < rp if ap is not None else True
        CombinerIdeal.__init__(self, PTL, Lm, None, None, None, None, None, 1.0)

        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        assert self.outerFringeFrac == 1.5, "May need to change numgrid points if this changes"

        self.Lm = Lm
        self.rp = rp
        self.numLayers = numLayers
        self.ap = ap
        self.loadBeamOffset = loadBeamOffset
        self.PTL = PTL
        self.magnet_widths = None
        self.field_fact: float = -1.0 if PTL.lattice_type == 'injector' else 1.0
        self.space = None
        self.extra_field_length = 0.0
        self.extraLoadApFrac = 1.5

        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet

        self.shape: str = 'COMBINER_CIRCULAR'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

        self.acceptance_width = None
        self.seed = seed

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        rp_layers, magnet_widths = get_halbach_layers_radii_and_magnet_widths(self.rp, self.numLayers, use_standard_sizes=
        self.PTL.use_standard_mag_size)
        self.magnet_widths = magnet_widths
        self.space = max(rp_layers) * self.outerFringeFrac
        self.ap = self.valid_aperture() if self.ap is None else self.ap
        assert self.is_apeture_valid(self.ap)
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input in a straight line. This is that section
        # or down
        # because field values are computed twice from same lens. Otherwise, magnet errors would change
        inputAngle, inputOffset, trajectoryLength = self.compute_Input_Orbit_Characteristics()
        self.Lo = trajectoryLength
        self.L = self.Lo
        self.ang = inputAngle
        self.La = (inputOffset + self.space / tan(inputAngle)) / (
                sin(inputAngle) + cos(inputAngle) ** 2 / sin(inputAngle))
        self.inputOffset = inputOffset - tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge
        self.outer_half_width = max(rp_layers) + max(magnet_widths) + MIN_MAGNET_MOUNT_THICKNESS
        self.acceptance_width = self.get_Acceptance_Width()

    def get_Acceptance_Width(self) -> float:
        extra_large_ange = 2 * self.ang
        width_overshoot = self.rp + (self.La + self.rp * sin(abs(extra_large_ange))) * sin(abs(extra_large_ange))
        return width_overshoot

    def is_apeture_valid(self, ap) -> bool:
        return ap <= self.rp - INTERP_MAGNET_OFFSET and ap <= self.max_ap_internal_interp_region()

    def valid_aperture(self):
        boreOD = 2 * self.rp
        apLargestTube = round_down_to_nearest_valid_tube_OD(boreOD) / 2.0 - COMBINER_TUBE_WALL_THICKNESS
        ap_maxInterp = self.max_ap_internal_interp_region()
        assert apLargestTube < ap_maxInterp
        return apLargestTube

    def make_Lens(self) -> _HalbachLensFieldGenerator:
        """Make field generating lens. A seed is required to reproduce the same magnet if magnet errors are being
        used because this is called multiple times."""
        rp_layers, magnet_widths = get_halbach_layers_radii_and_magnet_widths(self.rp, self.numLayers)
        individualMagnetLengthApprox = min([(MAGNET_ASPECT_RATIO * min(magnet_widths)), self.Lm])
        numDisks = 1 if not self.PTL.use_mag_errors else round(self.Lm / individualMagnetLengthApprox)
        lensCenter = self.Lm / 2 + self.space

        seed = DEFAULT_SEED if self.seed is None else self.seed
        state = np.random.get_state()
        np.random.seed(seed)
        lens = _HalbachLensFieldGenerator(rp_layers, magnet_widths, self.Lm, self.PTL.magnet_grade,
                                          use_method_of_moments=True,
                                          use_standard_mag_errors=self.PTL.use_mag_errors,
                                          numDisks=numDisks,
                                          use_solenoid_field=self.PTL.use_solenoid_field, position=(lensCenter, 0, 0),
                                          orientation=Rot.from_rotvec([0, np.pi / 2.0, 0.0]))  # must reuse lens

        np.random.set_state(state)
        return lens

    def num_points_x_interp(self, x_min: float, x_max: float) -> int:
        assert x_max > x_min
        minPointsX=11
        numPointsX = round_And_Make_Odd(
            self.PTL.field_dens_mult * self.pointsPerRadiusX * (x_max - x_min) / self.rp)
        return numPointsX if numPointsX>=minPointsX else minPointsX

    def make_Internal_Field_data_symmetry(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_Grid_Coords_Arrays_Internal_Symmetry()
        field_data = self.make_Field_Data(x_arr, y_arr, z_arr)
        return field_data

    def make_Field_data_full(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_Full_Grid_Coord_Arrays()
        field_data = self.make_Field_Data(x_arr, y_arr, z_arr)
        return field_data

    def make_External_Field_data_symmetry(self) -> tuple[ndarray, ...]:
        x_arr, y_arr, z_arr = self.make_Grid_Coords_Arrays_External_Symmetry()
        xArrInterp = x_arr.copy()
        assert xArrInterp[
                   -1] - TINY_INTERP_STEP == self.space  # interp must have overshoot of this size for trick below
        # to work
        xArrInterp[-1] -= TINY_INTERP_STEP + INTERP_MAGNET_OFFSET  # to avoid interpolating inside the magnetic
        # material I cheat here by shifting the point the field is calcuated at a tiny bit
        field_data = self.make_Field_Data(xArrInterp, y_arr, z_arr)
        field_data[0][:] = x_arr
        return field_data

    def build_fast_field_helper(self) -> None:
        self.set_extraFieldLength()
        if self.PTL.use_mag_errors:
            fieldDataInternal = self.make_Field_data_full()
        else:
            fieldDataInternal = self.make_Internal_Field_data_symmetry()
        fieldDataExternal = self.make_External_Field_data_symmetry()
        field_data = (fieldDataInternal, fieldDataExternal)

        useSymmetry = not self.PTL.use_mag_errors

        numba_func_constants = (self.ap, self.Lm, self.La, self.Lb, self.space, self.ang, self.acceptance_width,
                                self.field_fact, useSymmetry, self.extra_field_length)

        # todo: there's repeated code here between modules with the force stuff, not sure if I can sanely remove that

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(combinerHalbachFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lm / 2 + self.space, self.ap / 2, .0])))
        assert F_edge / F_center < .01

    def make_Full_Grid_Coord_Arrays(self) -> tuple[ndarray, ndarray, ndarray]:
        #todo: WET
        full_interp_field_length = (self.Lb + (self.La + self.acceptance_width * sin(abs(self.ang))) * cos(abs(self.ang)))
        magnet_center_x = self.space + self.Lm / 2

        x_min = magnet_center_x - full_interp_field_length / 2.0 - TINY_INTERP_STEP
        x_max = magnet_center_x + full_interp_field_length / 2.0 + TINY_INTERP_STEP

        z_min = -(self.rp - INTERP_MAGNET_OFFSET)
        z_max = -z_min
        m = abs(np.tan(self.ang))
        y_max = m * z_max + (self.acceptance_width + m * self.Lb) + TINY_INTERP_STEP
        y_min = -y_max
        numY0 = numZ = round_And_Make_Odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        numY = round_And_Make_Odd(numY0 * y_max / self.rp)
        numX = self.num_points_x_interp(x_min, x_max)
        numX, numY, numZ = [round_And_Make_Odd(val * 2 - 1) for val in (numX, numY, numZ)]
        return make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, numX, numY, numZ)

    def make_Grid_Coords_Arrays_External_Symmetry(self) -> tuple[ndarray, ndarray, ndarray]:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant

        numY0 = numZ = round_And_Make_Odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        #todo: something is wrong here with the interp stuff. There is out of bounds error
        x_min = self.space + self.Lm / 2 - (
                self.Lb + (self.La + self.acceptance_width * sin(abs(self.ang))) * cos(abs(self.ang))) / 2.0
        x_max = self.space + TINY_INTERP_STEP
        z_min = - TINY_INTERP_STEP
        z_max = self.rp - INTERP_MAGNET_OFFSET
        numX = self.num_points_x_interp(x_min, x_max)
        m = abs(np.tan(self.ang))
        y_max = m * z_max + (self.acceptance_width + m * self.Lb) + INTERP_MAGNET_OFFSET
        y_min = -TINY_INTERP_STEP
        numY = round_And_Make_Odd(numY0 * y_max / self.rp)
        return make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, numX, numY, numZ)

    def make_Grid_Coords_Arrays_Internal_Symmetry(self) -> tuple[ndarray, ndarray, ndarray]:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant

        numY = numZ = round_And_Make_Odd(self.num_grid_points_r * self.PTL.field_dens_mult)
        x_min = self.space - TINY_INTERP_STEP
        x_max = self.space + self.Lm / 2.0 + TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max = (self.rp - INTERP_MAGNET_OFFSET)
        z_min = -TINY_INTERP_STEP
        z_max = (self.rp - INTERP_MAGNET_OFFSET)
        numX = self.num_points_x_interp(x_min, x_max)
        return make_and_check_arrays_are_odd(x_min, x_max, y_min, y_max, z_min, z_max, numX, numY, numZ)

    def max_ap_internal_interp_region(self) -> float:
        _, y_arr, z_arr = self.make_Grid_Coords_Arrays_Internal_Symmetry()
        assert max(np.abs(y_arr)) == max(np.abs(z_arr))  # must be same for logic below or using radius
        radius_interp_region = max(np.abs(y_arr))
        assert radius_interp_region < self.rp - B_GRAD_STEP_SIZE  # interp must not reach into material for logic below
        ap_maxGoodField = radius_interp_region - np.sqrt(2) * (y_arr[1] - y_arr[0])
        return ap_maxGoodField

    def make_Field_Data(self, x_arr, y_arr, z_arr) -> tuple[ndarray, ...]:
        """Make field data as [[x,y,z,Fx,Fy,Fz,V]..] to be used in fast grid interpolator"""
        volume_coords = np.asarray(np.meshgrid(x_arr, y_arr, z_arr)).T.reshape(-1, 3)
        B_norm_grad, B_norm = np.nan * np.zeros((len(volume_coords), 3)), np.nan * np.zeros(len(volume_coords))
        validR = np.linalg.norm(volume_coords[:, 1:], axis=1) <= self.rp - B_GRAD_STEP_SIZE
        validX = np.logical_or(volume_coords[:, 0] < self.space - B_GRAD_STEP_SIZE,
                               volume_coords[:, 0] > self.space + self.Lm - B_GRAD_STEP_SIZE)
        validIndices = np.logical_or(validX, validR)  # tricky
        B_norm_grad[validIndices], B_norm[validIndices] = self.make_Lens().B_norm_grad(volume_coords[validIndices],
                                                                                       return_norm=True,
                                                                                       dx=B_GRAD_STEP_SIZE)
        data3D = np.column_stack((volume_coords, B_norm_grad, B_norm))
        field_data = self.shape_field_data_3D(data3D)
        return field_data

    def compute_Input_Orbit_Characteristics(self) -> tuple:
        """compute characteristics of the input orbit. This applies for injected beam, or recirculating beam"""
        from latticeElements.combiner_characterizer import characterize_CombinerHalbach

        self.output_offset = self.find_Ideal_Offset()

        inputAngle, inputOffset, trajectoryLength, _ = characterize_CombinerHalbach(self)
        assert inputAngle * self.field_fact > 0  # satisfied if low field is positive angle and high is negative.
        # Sometimes this can be triggered because the lens is to long so an oscilattory behaviour is required by
        # injector
        return inputAngle, inputOffset, trajectoryLength

    def update_Field_Fact(self, fieldStrengthFact) -> None:
        self.fast_field_helper.numbaJitClass.field_fact = fieldStrengthFact
        self.field_fact = fieldStrengthFact

    def get_valid_jitter_amplitude(self, Print=False) -> float:
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        assert self.PTL.jitter_amp >= 0.0
        jitter_amp_proposed = self.PTL.jitter_amp
        max_jitter_amp = self.max_ap_internal_interp_region() - self.ap
        jitter_amp = max_jitter_amp if jitter_amp_proposed > max_jitter_amp else jitter_amp_proposed
        if Print:
            if jitter_amp_proposed == max_jitter_amp and jitter_amp_proposed != 0.0:
                print(
                    'jitter amplitude of:' + str(jitter_amp_proposed) + ' clipped to maximum value:' + str(max_jitter_amp))
        return jitter_amp

    def set_extraFieldLength(self) -> None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped. Misalignment is a translational and/or rotational,
        so extra length needs to be accounted for in the case of rotational."""
        jitter_amp = self.get_valid_jitter_amplitude(Print=True)
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
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact = .95 * maxShift / totalShift  # safety factor
            print('proposed', totalShift, 'new', reductionFact * totalShift)
            shift_y, shift_z, rot_angle_y, rot_angle_z = [val * reductionFact for val in [shift_y, shift_z, rot_angle_y, rot_angle_z]]
        self.fast_field_helper.numbaJitClass.update_Element_Perturb_Params(shift_y, shift_z, rot_angle_y, rot_angle_z)

    def find_Ideal_Offset(self) -> float:
        """use newton's method to find where the vertical translation of the combiner wher the minimum seperation
        between atomic beam path and lens is equal to the specified beam diameter for INJECTED beam. This requires
        modeling high field seekers. Particle is traced backwards from the output of the combiner to the input.
        Can possibly error out from modeling magnet or assembly error"""
        from latticeElements.combiner_characterizer import characterize_CombinerHalbach

        if self.loadBeamOffset >= self.ap:  # beam doens't fit in combiner
            raise CombinerDimensionError
        yInitial = self.ap / 10.0
        try:
            inputAngle, _, _, seperationInitial = characterize_CombinerHalbach(self, atomState='HIGH_FIELD_SEEKING',
                                                                               particleOffset=yInitial)
        except:
            raise CombinerDimensionError
        assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
        gradientInitial = (seperationInitial - self.ap) / (yInitial - 0.0)
        y = yInitial
        seperation = seperationInitial  # initial value of lens/atom seperation.
        gradient = gradientInitial
        i, iterMax = 0, 20  # to prevent possibility of ifnitne loop
        tolAbsolute = 1e-6  # m
        targetSep = self.loadBeamOffset
        while not isclose(seperation, targetSep, abs_tol=tolAbsolute):
            deltaX = -(seperation - targetSep) / gradient  # I like to use a little damping
            deltaX = -y / 2 if y + deltaX < 0 else deltaX  # restrict deltax to allow value
            y = y + deltaX
            inputAngle, _, _, seperationNew = characterize_CombinerHalbach(self, atomState='HIGH_FIELD_SEEKING',
                                                                           particleOffset=y)
            assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
            gradient = (seperationNew - seperation) / deltaX
            seperation = seperationNew
            i += 1
            if i > iterMax:
                raise CombinerIterExceededError
        assert 0.0 < y < self.ap
        return y
