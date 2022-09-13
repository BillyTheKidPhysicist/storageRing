from math import cos, sin, sqrt, tan
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import constants
from field_generators import ElementMagnetCollection, Collection, HalbachLens
from field_generators import HalbachBender
from helper_tools import random_num_for_seeding
from helper_tools import temporary_seed, is_even
from lattice_elements.utilities import MAGNET_ASPECT_RATIO, B_GRAD_STEP_SIZE, INTERP_MAGNET_MATERIAL_OFFSET
from type_hints import ndarray

Dim1_Arr = ndarray


# IMPROVEMENT: the naming of valid is silly

def valid_field_values_col(element_magnet, coords, valid_indices, use_approx,
                           extra_elements: list[Collection] = None):
    """Get valid field values, ie those specified by 'valid_indices, from 'element_magnet' and possible extra
     magnets"""
    col = ElementMagnetCollection(element_magnet)
    if extra_elements is not None:
        col.add_magnets(extra_elements)

    B_norm_grad_arr, B_norm_arr = np.zeros((len(coords), 3)) * np.nan, np.zeros(len(coords)) * np.nan
    B_norm_grad_arr[valid_indices], B_norm_arr[valid_indices] = col.B_norm_grad(coords[valid_indices],
                                                                                return_norm=True,
                                                                                use_approx=use_approx,
                                                                                dx=B_GRAD_STEP_SIZE)
    return B_norm_grad_arr, B_norm_arr


def misalignment_transform_parameters(self) -> tuple[ndarray, Rot, Rot, ndarray]:
    """Return parameters (shifts and rotations) to move magnet to misaligned position. To apply these rotations,
    first rotate about rotation_origin using Ry and Rz, then shift with r_shift"""
    dx, dy1, dz1, dy2, dz2 = self.alignment_shifts
    rotation_origin = self.r_in_el.copy()
    r_shift = np.array([dx, dy1, dz1])
    angle_z = np.arctan((dy2 - dy1) / self.Lm)
    angle_y = -np.arctan((dz2 - dz1) / self.Lm)  # rotation about y requires (-)
    R1 = Rot.from_rotvec([0, angle_y, 0])
    R2 = Rot.from_rotvec([0, 0, angle_z])
    return rotation_origin, R1, R2, r_shift


def transform_coords_to_misaligned(self, coords):
    """Transform coords into the misalignment frame to ease analysis of wether they are valid.
    Go in reverse order of the misalignment procedure"""
    rotation_origin, R1, R2, r_shift = misalignment_transform_parameters(self)
    R1, R2 = R1.as_matrix(), R2.as_matrix()
    coords -= r_shift
    coords = (coords - rotation_origin).T
    coords = np.linalg.inv(R2) @ coords
    coords = np.linalg.inv(R1) @ coords
    coords = coords.T + rotation_origin
    return coords


def yz_random_samp() -> tuple[float, float]:
    """Generate 2 random samples, y and z,  in circle"""
    r_shift = sqrt(np.random.random()) * constants.ASSEMBLY_TOLERANCE  # sqrt neccesary cause it's polar
    angle = np.random.random() * 2 * np.pi
    return r_shift * cos(angle), r_shift * sin(angle)


def alignment_shifts() -> tuple[float, float, float, float, float]:
    """Generate the 5 shift values that represent misalingment of the lens. dx, dy1,dz1,dy2,dz2"""
    dy1, dz1 = yz_random_samp()
    dy2, dz2 = yz_random_samp()
    dx = 2 * (np.random.random() - .5) * constants.ASSEMBLY_TOLERANCE
    return dx, dy1, dy2, dz1, dz2


class MagneticOptic:
    def __init__(self, seed: Optional[int]):
        self.seed = random_num_for_seeding() if seed is None else seed
        self.neighbors: list[MagneticOptic] = []


class MagneticLens(MagneticOptic):
    def __init__(self, Lm, rp_layers, magnet_widths, magnet_grade, use_solenoid, x_in_offset, seed=None):
        assert len(rp_layers) == len(magnet_widths)
        super().__init__(seed)
        self.Lm = Lm
        self.rp_layers = rp_layers
        self.magnet_widths = magnet_widths
        self.magnet_grade = magnet_grade
        self.use_solenoid = use_solenoid
        with temporary_seed(self.seed):
            self.alignment_shifts = alignment_shifts()  # dx, dy1, dz1, dy2, dz2
        self.x_in_offset = x_in_offset
        self.r_in_el = np.array([x_in_offset, 0.0, 0.0])

    def num_disks(self, magnet_errors) -> int:
        if magnet_errors:
            L_magnets = min([(MAGNET_ASPECT_RATIO * min(self.magnet_widths)), self.Lm])
            return round(self.Lm / L_magnets)
        else:
            return 1

    def misalignment_transform_parameters(self) -> tuple[ndarray, Rot, Rot, ndarray]:
        """Return parameters (shifts and rotations) to move magnet to misaligned position. To apply these rotations,
        first rotate about rotation_origin using Ry and Rz, then shift with r_shift"""
        dx, dy1, dz1, dy2, dz2 = self.alignment_shifts
        rotation_origin = self.r_in_el.copy()
        r_shift = np.array([dx, dy1, dz1])
        angle_z = np.arctan((dy2 - dy1) / self.Lm)
        angle_y = -np.arctan((dz2 - dz1) / self.Lm)  # rotation about y requires (-)
        R1 = Rot.from_rotvec([0, angle_y, 0])
        R2 = Rot.from_rotvec([0, 0, angle_z])
        return rotation_origin, R1, R2, r_shift

    def transform_coords_to_misaligned(self, coords):
        """Transform coords into the misalignment frame to ease analysis of wether they are valid.
        Go in reverse order of the misalignment procedure"""
        rotation_origin, R1, R2, r_shift = self.misalignment_transform_parameters()
        R1, R2 = R1.as_matrix(), R2.as_matrix()
        coords -= r_shift
        coords = (coords - rotation_origin).T
        coords = np.linalg.inv(R2) @ coords
        coords = np.linalg.inv(R1) @ coords
        coords = coords.T + rotation_origin
        return coords

    def make_magpylib_magnets(self, magnet_errors, include_misalignments) -> Collection:
        position = (self.Lm / 2.0 + self.x_in_offset, 0, 0)
        orientation = Rot.from_rotvec([0, np.pi / 2.0, 0.0])

        with temporary_seed(self.seed):
            magnets = HalbachLens(self.rp_layers, self.magnet_widths, self.Lm, self.magnet_grade,
                                  use_method_of_moments=True, use_standard_mag_errors=magnet_errors,
                                  num_disks=self.num_disks(magnet_errors), use_solenoid_field=self.use_solenoid,
                                  orientation=orientation, position=position)
        if include_misalignments:
            rotation_origin, Ry, Rz, r_shift = self.misalignment_transform_parameters()
            for R in [Ry, Rz]:
                magnets.rotate(R, anchor=rotation_origin)
            magnets.move_meters(r_shift)

        return magnets

    def magpylib_magnets_model(self, magnet_errors, include_misalignments):
        return self.make_magpylib_magnets(magnet_errors, include_misalignments)

    def get_valid_coord_indices(self, coords: ndarray, include_misalignments) -> Dim1_Arr:
        if include_misalignments:
            coords = coords.copy()  # to not modify original
            coords = self.transform_coords_to_misaligned(coords)
        valid_x_a = coords[:, 0] < self.x_in_offset - INTERP_MAGNET_MATERIAL_OFFSET
        valid_x_b = coords[:, 0] > self.x_in_offset + self.Lm + INTERP_MAGNET_MATERIAL_OFFSET
        rarr = np.linalg.norm(coords[:, 1:], axis=1)

        r_inner = np.min(self.rp_layers)
        r_outer = np.max(self.rp_layers) + self.magnet_widths[np.argmax(self.rp_layers)]
        valid_r_a = rarr < r_inner - INTERP_MAGNET_MATERIAL_OFFSET
        valid_r_b = rarr > r_outer + INTERP_MAGNET_MATERIAL_OFFSET
        valid_indices = valid_x_a + valid_x_b + valid_r_a + valid_r_b
        return valid_indices

    def get_valid_field_values(self, coords: ndarray, use_mag_errors: bool = False,
                               extra_magnets: Collection = None,
                               include_misalignments=False,use_approx=True) -> tuple[ndarray, ndarray]:
        valid_indices = self.get_valid_coord_indices(coords, include_misalignments)
        magnets = self.magpylib_magnets_model(use_mag_errors, include_misalignments)
        return valid_field_values_col(magnets, coords, valid_indices, use_approx, extra_elements=extra_magnets)


class MagnetBender(MagneticOptic):
    """Model of the magnetic system of a segmented bender magnet"""
    num_model_lenses = 7

    def __init__(self, rp, rb, uc_angle, Lm, magnet_width, magnet_grade, num_lenses, use_solenoid, seed=None):
        super().__init__(seed)
        self.rp = rp
        self.rb = rb
        self.uc_angle = uc_angle
        self.Lm = Lm
        self.magnet_grade = magnet_grade
        self.num_lenses = num_lenses
        self.use_solenoid = use_solenoid
        self.magnet_width = magnet_width

    def make_magpylib_magnets(self, use_pos_mag_angs_only: bool, use_half_cap_end: tuple[bool, bool],
                              num_lenses: int, use_mag_errors: bool, use_full_method_of_moments: bool = False):
        """Return magpylib magnet model representing a portion or all of the bender"""
        use_approx_method_of_moments = not use_full_method_of_moments
        with temporary_seed(self.seed):
            bender_field_generator = HalbachBender(self.rp, self.rb, self.uc_angle, self.Lm,
                                                   self.magnet_grade,
                                                   num_lenses, use_half_cap_end,
                                                   use_pos_mag_angs_only=use_pos_mag_angs_only,
                                                   use_solenoid_field=self.use_solenoid,
                                                   use_mag_errors=use_mag_errors,
                                                   magnet_width=self.magnet_width,
                                                   use_approx_method_of_moments=use_approx_method_of_moments)
        return bender_field_generator

    def magpylib_magnets_model(self, use_mag_errors, use_full_method_of_moments=False) -> Collection:
        """Return full magpylib magnet model of bender"""
        return self.make_magpylib_magnets(True, (True, True), self.num_lenses,
                                          use_mag_errors, use_full_method_of_moments=use_full_method_of_moments)

    def magpylib_magnets_internal_model(self) -> Collection:
        """Return full magpylib magnet model representing repeating interior region of bender"""
        num_lenses = self.num_model_lenses
        assert not is_even(num_lenses)
        return self.make_magpylib_magnets(False, (False, False), num_lenses, False)

    def magpylib_magnets_fringe_cap_model(self) -> Collection:
        """Return full magpylib magnet model representing input section of bender"""
        return self.make_magpylib_magnets(True, (True, False), self.num_model_lenses, False)

    def is_valid_in_lens_of_bender(self, x: bool, y: bool, z: bool) -> bool:
        """Check that the coordinates x,y,z are valid for a lens in the bender. The lens is centered on (self.rb,0,0)
        aligned with the z axis. If the coordinates are outside the double unit cell containing the lens, or inside
        the toirodal cylinder enveloping the magnet material, the coordinate is invalid"""
        y_uc_line = tan(self.uc_angle) * x
        minor_radius = sqrt((x - self.rb) ** 2 + z ** 2)
        lens_radius_valid_inner = self.rp - INTERP_MAGNET_MATERIAL_OFFSET
        lens_radius_valid_outer = self.rp + self.magnet_width + INTERP_MAGNET_MATERIAL_OFFSET
        if abs(y) < (self.Lm + B_GRAD_STEP_SIZE) / 2.0 and \
                lens_radius_valid_inner <= minor_radius < lens_radius_valid_outer:
            return False
        elif abs(y) <= y_uc_line:
            return True
        else:
            return False

    def get_valid_indices_internal(self, coords: ndarray, max_rotations: int) -> list[bool]:
        """Check if coords are not in the magnetic material region of the bender. Check up to max_rotations of the
        coords going counterclockwise about y axis by rotating coords"""
        R = Rot.from_rotvec([0, 0, -self.uc_angle]).as_matrix()
        valid_indices = []
        for [x, y, z] in coords:
            if self.is_valid_in_lens_of_bender(x, y, z):
                valid_indices.append(True)
            else:
                loop_start, loop_stop = 1, max_rotations + 1
                for i in range(loop_start, loop_stop):
                    num_rotations = (i + 1) // 2
                    x, y, z = (R ** num_rotations) @ [x, y, z]
                    if self.is_valid_in_lens_of_bender(x, y, z):
                        valid_indices.append(True)
                        break
                    elif i == loop_stop - 1:
                        valid_indices.append(False)
        return valid_indices

    def valid_internal_fringe_field_values(self, coords) -> tuple[ndarray, ndarray]:
        valid_indices = self.get_valid_indices_internal(coords, 3)
        magnets = self.magpylib_magnets_fringe_cap_model()
        B_norm_grad_arr, B_norm_arr = valid_field_values_col(magnets, coords, valid_indices, False)
        return B_norm_grad_arr, B_norm_arr

    def valid_segment_field_values(self, coords) -> tuple[ndarray, ndarray]:
        valid_indices = self.get_valid_indices_internal(coords, 1)
        magnets = self.magpylib_magnets_internal_model()
        B_norm_grad_arr, B_norm_arr = valid_field_values_col(magnets, coords, valid_indices, False)
        return B_norm_grad_arr, B_norm_arr

    def valid_field_values_cap(self, coords) -> tuple[ndarray, ndarray]:
        # IMPROVEMENT: CAP NAMING IS AWFUL. BOTH HERE AND IN LENS
        valid_indices = np.sqrt((coords[:, 0] - self.rb) ** 2 + coords[:, 2] ** 2) < self.rp - B_GRAD_STEP_SIZE
        magnets = self.magpylib_magnets_fringe_cap_model()
        B_norm_grad_arr, B_norm_arr = valid_field_values_col(magnets, coords, valid_indices, False)
        return B_norm_grad_arr, B_norm_arr

    def valid_field_values_full(self, coords_center, coords_cartesian,
                                extra_elements: Optional[list[Collection]],
                                use_mag_errors) -> tuple[ndarray, ndarray]:

        magnets = self.magpylib_magnets_model(use_mag_errors=use_mag_errors)

        r_center_arr = np.linalg.norm(coords_center[:, 1:], axis=1)
        valid_indices = r_center_arr < self.rp  # IMPROVEMENT: implement the more accurate valid checker thing
        B_norm_grad_arr, B_norm_arr = valid_field_values_col(magnets, coords_cartesian, valid_indices, True,
                                                             extra_elements=extra_elements)
        return B_norm_grad_arr, B_norm_arr
