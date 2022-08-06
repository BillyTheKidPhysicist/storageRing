from math import cos, sin

from scipy.spatial.transform import Rotation as Rot

from HalbachLensClass import Collection
from HalbachLensClass import HalbachLens
from constants import ASSEMBLY_TOLERANCE
from helperTools import *
from latticeElements.utilities import MAGNET_ASPECT_RATIO

B_Vec_Arr, B_Norm_Arr = np.ndarray, np.ndarray
Dim1_Arr = np.ndarray


def misalignment_transform_parameters(self) -> tuple[np.ndarray, Rot, Rot, np.ndarray]:
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
    r_shift = np.random.random() * ASSEMBLY_TOLERANCE
    angle = np.random.random() * 2 * np.pi
    return r_shift * cos(angle), r_shift * sin(angle)


def alignment_shifts() -> tuple[float, float, float, float, float]:
    """Generate the 5 shift values that represent misalingment of the lens. dx, dy1,dz1,dy2,dz2"""
    dy1, dz1 = yz_random_samp()
    dy2, dz2 = yz_random_samp()
    dx = np.random.random() * ASSEMBLY_TOLERANCE
    return dx, dy1, dy2, dz1, dz2


class MagneticOptic:
    def __init__(self, seed: int = None):
        self.r_in_lab = None  # input of magnet system in lab coordinates.
        self.r_out_lab = None  # output of magnet system in lab coordinates
        self.r_in_el = None
        self.r_out_el = None
        self.norm_in_lab = None  # input of magnet system in lab coordinates.
        self.norm_out_lab = None  # output of magnet system in lab coordinates
        self.norm_in_el = None
        self.norm_out_el = None
        self.seed = int(time.time()) if seed is None else seed
        self.neighbors: list[MagneticOptic] = []


class MagneticLens(MagneticOptic):
    def __init__(self, Lm, rp_layers, magnet_widths, magnet_grade, use_solenoid, x_in_offset, seed=None,
                 num_slices=1):
        assert all(rp1 == rp2 for rp1, rp2 in zip(rp_layers, sorted(rp_layers)))
        assert len(rp_layers) == len(magnet_widths)
        super().__init__()
        self.Lm = Lm
        self.rp_layers = rp_layers
        self.magnet_widths = magnet_widths
        self.magnet_grade = magnet_grade
        self.use_solenoid = use_solenoid
        self.seed = seed
        self.num_slices = num_slices
        self.alignment_shifts = alignment_shifts()  # dx, dy1, dz1, dy2, dz2

        self.x_in_offset = x_in_offset
        self.norm_in_el, self.norm_out_el = np.array([-1.0, 0, 0]), np.array([1.0, 0, 0])
        self.combiner = False

    def fill_position_and_orientation_params(self, pos_in_lab_for_element, norm_in_lab_for_element):
        self.r_in_el = np.array([self.x_in_offset, 0.0, 0.0])
        self.r_out_el = np.array([self.x_in_offset + self.Lm, 0.0, 0.0])
        self.r_in_lab = pos_in_lab_for_element - self.x_in_offset * norm_in_lab_for_element
        self.r_out_lab = self.r_in_lab + (-norm_in_lab_for_element) * self.Lm
        self.norm_in_lab = norm_in_lab_for_element
        self.norm_out_lab = -self.norm_in_lab

    def num_disks(self, magnet_errors) -> int:
        if magnet_errors:
            L_magnets = min([(MAGNET_ASPECT_RATIO * min(self.magnet_widths)), self.Lm])
            return round(self.Lm / L_magnets)
        else:
            return 1

    def misalignment_transform_parameters(self) -> tuple[np.ndarray, Rot, Rot, np.ndarray]:
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

        magnets = HalbachLens(self.rp_layers, self.magnet_widths, self.Lm, self.magnet_grade,
                              use_method_of_moments=True, use_standard_mag_errors=magnet_errors,
                              num_disks=self.num_disks(magnet_errors), use_solenoid_field=self.use_solenoid,
                              orientation=orientation, position=position)
        if include_misalignments:
            rotation_origin, Ry, Rz, r_shift = self.misalignment_transform_parameters()
            for R in [Ry, Rz]:
                magnets.rotate(R, anchor=rotation_origin)
            magnets.move(r_shift)

        return magnets

    def get_valid_coord_indices(self, coords: np.ndarray, interp_step_size: float, include_misalignments) -> Dim1_Arr:
        if include_misalignments:
            coords = coords.copy()  # to not modify original
            coords = self.transform_coords_to_misaligned(coords)

        valid_x_a = coords[:, 0] < self.x_in_offset - interp_step_size
        valid_x_b = coords[:, 0] > self.x_in_offset + self.Lm + interp_step_size
        rarr = np.linalg.norm(coords[:, 1:], axis=1)

        r_inner = self.rp_layers[0]
        r_outer = self.rp_layers[-1] + self.magnet_widths[-1]
        valid_r_a = rarr < r_inner - interp_step_size
        valid_r_b = rarr > r_outer + interp_step_size
        valid_indices = valid_x_a + valid_x_b + valid_r_a + valid_r_b
        return valid_indices

    def get_valid_field_values(self, coords: np.ndarray, interp_step_size: float, use_mag_errors: bool = False,
                               extra_magnets: Collection = None, interp_rounding_guard: float = 1e-12,
                               include_misalignments=False) -> tuple[
        B_Vec_Arr, B_Norm_Arr]:
        assert interp_step_size > 0.0 and interp_rounding_guard > 0.0
        interp_step_size_valid = interp_step_size + interp_rounding_guard
        valid_indices = self.get_valid_coord_indices(coords, interp_step_size_valid, include_misalignments)
        col = Collection([self.make_magpylib_magnets(use_mag_errors,include_misalignments)])
        if extra_magnets is not None:
            col.add(extra_magnets)
            # col.show()
        B_norm_grad, B_norm = np.zeros((len(valid_indices), 3)) * np.nan, np.ones(len(valid_indices)) * np.nan
        B_norm_grad[valid_indices], B_norm[valid_indices] = col.B_norm_grad(coords[valid_indices],
                                                                            return_norm=True, dx=interp_step_size)
        return B_norm_grad, B_norm
