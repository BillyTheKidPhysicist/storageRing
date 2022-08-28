import copy
from numbers import Number

import magpylib.current
import numba
import numpy as np
from magpylib import Collection as _Collection
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib.magnet import Cuboid as _Cuboid
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from constants import MAGNETIC_PERMEABILITY, MAGNET_WIRE_DIAM, SPIN_FLIP_AVOIDANCE_FIELD, GRADE_MAGNETIZATION
from demag_functions import apply_demag
from helper_tools import Union, Optional, math, inch_to_meter, radians, within_tol
from helper_tools import temporary_seed
from lattice_elements.utilities import halbach_magnet_width, max_tube_IR_in_segmented_bend
from type_hints import ndarray, sequence

magpyMagnetization_ToSI: float = 1 / (1e3 * MAGNETIC_PERMEABILITY)
SI_MagnetizationToMagpy: float = 1 / magpyMagnetization_ToSI
METER_TO_MM = 1e3  # magpy takes distance in mm
MM_TO_METER = 1e-3  # magpy takes distance in mm

COILS_PER_RADIUS = 4  # number of longitudinal coils per length is this number divided by radius of element

BORE_RADII_MAX_DISTANCE_FACTORS = 7.0


@numba.njit()
def B_NUMBA(r: ndarray, r0: ndarray, m: ndarray) -> ndarray:
    r = r - r0  # convert to difference vector
    r_norm_temp = np.sqrt(np.sum(r ** 2, axis=1))
    r_norm = np.empty((r_norm_temp.shape[0], 1))
    r_norm[:, 0] = r_norm_temp
    mr_dot_temp = np.sum(m * r, axis=1)
    mr_dot = np.empty((r_norm_temp.shape[0], 1))
    mr_dot[:, 0] = mr_dot_temp
    B_vec = (MAGNETIC_PERMEABILITY / (4 * np.pi)) * (3 * r * mr_dot / r_norm ** 5 - m / r_norm ** 3)
    return B_vec


class Sphere:

    def __init__(self, radius: float, magnet_grade: str = 'legacy'):
        # angle: symmetry plane angle. There is a negative and positive one
        # radius: radius in inches
        # M_norm: magnetization
        assert radius > 0
        self.angle: Optional[float] = None  # angular location of the magnet
        self.radius: float = radius
        self.volume: float = (4 * np.pi / 3) * self.radius ** 3  # m^3
        self.m0: float = MAGNETIC_PERMEABILITY[magnet_grade] * self.volume  # dipole moment
        self.r0: Optional[ndarray] = None  # location of sphere
        self.n: Optional[ndarray] = None  # orientation
        self.m: Optional[ndarray] = None  # vector sphere moment
        self.theta: Optional[float] = None  # phi position
        self.phi: Optional[float] = None  # orientation of dipole. From lab z axis
        self.psi: Optional[float] = None  # orientation of dipole. in lab xy plane
        self.z: Optional[float] = None
        self.r: Optional[float] = None

    def position_sphere(self, r: float, theta: float, z: float) -> None:
        self.r, self.theta, self.z = r, theta, z
        assert None not in (theta, z, r)
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)
        self.r0 = np.asarray([x, y, self.z])

    def update_size(self, radius: float) -> None:
        self.radius = radius
        self.volume = (4 * np.pi / 3) * self.radius ** 3
        M_norm = 1.15e6  # magnetization density
        self.m0 = M_norm * (4 / 3) * np.pi * self.radius ** 3  # dipole moment
        self.m = self.m0 * self.n  # vector sphere moment

    def orient(self, phi: float, psi: float) -> None:
        # tilt the sphere in spherical coordinates. These are in the lab frame, there is no sphere frame
        # phi,psi =(pi/2,0) is along +x
        # phi,psi=(pi/2,pi/2) is along +y
        # phi,psi=(0.0,anything) is along +z
        self.phi = phi
        self.psi = psi
        self.n = np.asarray([np.sin(phi) * np.cos(psi), np.sin(phi) * np.sin(psi), np.cos(phi)])  # x,y,z
        self.m = self.m0 * self.n

    def B(self, r: ndarray) -> ndarray:
        assert len(r.shape) == 2 and r.shape[1] == 3
        return B_NUMBA(r, self.r0, self.m)

    def B_shim(self, r: ndarray, plane_symmetry: bool = True, negative_symmetry: bool = True,
               rotation_angle: float = np.pi / 3) -> ndarray:
        # a single magnet actually represents 12 magnet
        # r: array of N position vectors to get field at. Shape (N,3)
        # plane_symmetry: Wether to exploit z symmetry or not
        # plt.quiver(self.r0[0],self.r0[1],self.m[0],self.m[1],color='r')
        arr = np.zeros(r.shape)
        arr += self.B(r)
        arr += self.B_symmetry(r, 1, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 2, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 3, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 4, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 5, negative_symmetry, rotation_angle, not plane_symmetry)

        if plane_symmetry:
            arr += self.B_symmetry(r, 0, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 1, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 2, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 3, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 4, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 5, negative_symmetry, rotation_angle, plane_symmetry)

        return arr

    def B_symmetry(self, r: ndarray, rotations: float, negative_symmetry: float, rotation_angle: float,
                   plane_reflection: float) -> ndarray:
        rot_angle = rotation_angle * rotations
        R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
        r0_sym = self.r0.copy()
        r0_sym[:2] = R @ r0_sym[:2]
        m_sym = self.m.copy()
        m_sym[:2] = R @ m_sym[:2]
        if negative_symmetry:
            m_sym[:2] *= (-1) ** rotations
        if plane_reflection:  # another dipole on the other side of the z=0 line
            r0_sym[2] = -r0_sym[2]
            m_sym[-1] *= -1
        # plt.quiver(r0_sym[0], r0_sym[1], m_sym[0], m_sym[1])
        B_vec_arr = B_NUMBA(r, r0_sym, m_sym)

        return B_vec_arr


class Collection(_Collection):

    def __init__(self, *sources, **kwargs):
        super().__init__(*sources, **kwargs)

    def rotate(self, rotation, anchor=None, start=-1):
        if anchor is not None and not isinstance(anchor, (int, float)):  # make more general
            anchor = [entry * METER_TO_MM for entry in anchor]
        super().rotate(rotation, anchor=anchor, start=start)

    def move_meters(self, displacement, start="auto") -> None:
        """Move an object assuming displacement is in units of meters"""
        displacement_mm = [entry * METER_TO_MM for entry in displacement]
        super().move(displacement_mm, start=start)

    def position_meters(self) -> ndarray:
        return np.array(self.position) * MM_TO_METER

    def _getB_wrapper(self, eval_coords_mm: ndarray, size_max: float = 500_000) -> ndarray:
        """To reduce ram usage, split the sources up into smaller chunks. A bit slower, but works realy well. Only
        applied to sources when ram usage would be too hight"""
        sources_all = self.sources_all
        size = len(eval_coords_mm) * len(self.sources_all)
        splits = min([int(size / size_max), len(sources_all)])
        splits = 1 if splits < 1 else splits
        split_size = math.ceil(len(sources_all) / splits)
        split_sources = [sources_all[split_size * i:split_size * (i + 1)] for i in range(splits)] if splits > 1 else [
            sources_all]
        B_vec = np.zeros(eval_coords_mm.shape)
        counter = 0
        for sources in split_sources:
            if len(sources) == 0:
                break
            counter += len(sources)
            B_vec += getBH_level2(sources, eval_coords_mm, sumup=True, squeeze=True, pixel_agg=None, field="B")
        B_vec = B_vec[0] if len(B_vec) == 1 else B_vec
        assert counter == len(sources_all)
        return B_vec

    def B_vec(self, eval_coords: ndarray, use_approx: int = False) -> ndarray:
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        assert len(self) > 0 and isinstance(eval_coords, ndarray)
        if use_approx:
            raise NotImplementedError
        mTESLA_TO_TESLA = 1e-3
        eval_coords_mm = METER_TO_MM * eval_coords
        b_vec = mTESLA_TO_TESLA * self._getB_wrapper(eval_coords_mm)
        return b_vec

    def B_norm(self, eval_coords: ndarray, use_approx: bool = False) -> ndarray:
        # r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array

        B_vec = self.B_vec(eval_coords, use_approx=use_approx)
        if len(eval_coords.shape) == 1:
            return norm(B_vec)
        elif len(eval_coords) == 1:
            return np.asarray([norm(B_vec)])
        else:
            return norm(B_vec, axis=1)

    def central_diff(self, eval_coords: ndarray, return_norm: bool, use_approx: bool, dx: float = 1e-7) -> \
            Union[tuple[ndarray, ...], ndarray]:
        assert dx > 0.0

        def grad(index: int) -> ndarray:
            coord_b = eval_coords.copy()  # upper step
            coord_b[:, index] += dx
            B_norm_b = self.B_norm(coord_b, use_approx=use_approx)
            coord_a = eval_coords.copy()  # upper step
            coord_a[:, index] += -dx
            B_norm_a = self.B_norm(coord_a, use_approx=use_approx)
            return (B_norm_b - B_norm_a) / (2 * dx)

        B_norm_grad = np.column_stack((grad(0), grad(1), grad(2)))
        if return_norm:
            B_norm = self.B_norm(eval_coords, use_approx=use_approx)
            return B_norm_grad, B_norm
        else:
            return B_norm_grad

    def forward_diff(self, eval_coords: ndarray, return_norm: bool,
                     use_approx: bool, dx: float = 1e-7) -> Union[tuple[ndarray, ...], ndarray]:
        assert dx > 0.0
        B_norm = self.B_norm(eval_coords, use_approx=use_approx)

        def grad(index):
            coord_b = eval_coords.copy()  # upper step
            coord_b[:, index] += dx
            B_norm_b = self.B_norm(coord_b, use_approx=use_approx)
            return (B_norm_b - B_norm) / dx

        B_norm_grad = np.column_stack((grad(0), grad(1), grad(2)))
        if return_norm:
            return B_norm_grad, B_norm
        else:
            return B_norm_grad

    def shape_eval_coords(self, eval_coords: ndarray) -> ndarray:
        """Shape the coordinates that the field values are evaluated at. valid input shapes are (3) and (N,3) where N
        is the number of points to evaluate. (3) is converted to (1,3)"""

        assert eval_coords.ndim in (1, 2)
        eval_coords_shaped = np.array([eval_coords]) if eval_coords.ndim != 2 else eval_coords
        return eval_coords_shaped

    def B_norm_grad(self, eval_coords: ndarray, return_norm: bool = False, diff_method='forward',
                    use_approx: bool = False, dx: float = 1e-7) -> Union[ndarray, tuple]:
        # Return the gradient of the norm of the B field. use forward difference theorom
        # r: (N,3) vector of coordinates or (3) vector of coordinates.
        # return_norm: Wether to return the norm as well as the gradient.
        # dr: step size
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array

        eval_coords_shaped = self.shape_eval_coords(eval_coords)

        if diff_method == 'central':
            results = self.central_diff(eval_coords_shaped, return_norm, use_approx, dx=dx)
        elif diff_method == 'forward':
            results = self.forward_diff(eval_coords_shaped, return_norm, use_approx, dx=dx)
        else:
            raise NotImplementedError

        if len(eval_coords.shape) == 1:
            if return_norm:
                [[B_grad_x, B_grad_y, B_grad_z]], [B0] = results
                results = (np.array([B_grad_x, B_grad_y, B_grad_z]), B0)
            else:
                [[B_grad_x, B_grad_y, B_grad_z]] = results
                results = np.array([B_grad_x, B_grad_y, B_grad_z])
        return results

    def apply_method_of_moments(self):
        apply_demag(self)


class Cuboid(_Cuboid):
    def __init__(self, mur: float = 1.0, *args, **kwargs):  # todo: change default to 1.05
        super().__init__(*args, **kwargs)
        self.mur = mur
        self.magnetization0 = self.magnetization.copy()


class Layer(Collection):
    # class object for a layer of the magnet. Uses the RectangularPrism object
    num_magnets_in_layer = 12

    def __init__(self, rp: float, magnet_width: float, length: float, magnet_grade: str, position: sequence = None,
                 orientation: Rotation = None, mur: float = 1.05,
                 r_magnet_shift=None, theta_shift=None, phi_shift=None, M_norm_shift_rel=None, dim_shift=None,
                 R_angle_shift=None, use_method_of_moments=False):
        super().__init__()
        assert magnet_width > 0.0 and length > 0.0
        assert isinstance(orientation, (type(None), Rotation))
        position = (0.0, 0.0, 0.0) if position is None else position
        self.r_magnet_shift: ndarray = self.make_Arr_If_None_Else_Copy(r_magnet_shift)
        self.theta_shift: ndarray = self.make_Arr_If_None_Else_Copy(theta_shift)
        self.phi_shift: ndarray = self.make_Arr_If_None_Else_Copy(phi_shift)
        self.M_norm_shift_relative: ndarray = self.make_Arr_If_None_Else_Copy(M_norm_shift_rel)
        self.dim_shift: ndarray = self.make_Arr_If_None_Else_Copy(dim_shift, num_params=3)
        self.R_angle_shift: ndarray = self.make_Arr_If_None_Else_Copy(R_angle_shift, num_params=2)
        self.rp: tuple = (rp,) * self.num_magnets_in_layer
        self.mur = mur  # relative permeability
        self.position_to_set = position
        self.orientation_to_set = orientation  # orientation about body frame #todo: this "to set" stuff is pretty wonky
        self.magnet_width: float = magnet_width
        self.length: float = length
        self.use_method_of_moments = use_method_of_moments
        self.M_norm: float = GRADE_MAGNETIZATION[magnet_grade]
        self.build()

    def make_Arr_If_None_Else_Copy(self, variable: Optional[ndarray], num_params: int = 1) -> ndarray:
        """If no misalignment is supplied, making correct shaped array of zeros, also check shape of provided array
        is correct"""
        assert variable.shape[1] == num_params if variable is not None else True
        variable_arr = np.zeros((self.num_magnets_in_layer, num_params)) if variable is None else copy.copy(variable)
        assert len(variable_arr) == self.num_magnets_in_layer and len(variable_arr.shape) == 2
        return variable_arr

    def build(self) -> None:
        # build the elements that form the layer. The 'home' magnet's center is located at x=r0+width/2,y=0, and its
        # magnetization points along positive x
        # how I do this is confusing
        magnetization_arr = self.make_Mag_Vec_Arr_Magpy()
        dimension_arr = self.make_cuboid_dimensions_magpy()
        position_arr = self.make_cuboid_positions_magpy()
        orientation_list = self.make_cuboid_orientation_magpy()
        for M_norm, dim, pos, orientation in zip(magnetization_arr, dimension_arr, position_arr, orientation_list):
            box = Cuboid(magnetization=M_norm, dimension=dim,
                         position=pos, orientation=orientation, mur=self.mur)
            self.add(box)

        if self.orientation_to_set is not None:
            self.rotate(self.orientation_to_set, anchor=0.0)
        self.move_meters(self.position_to_set)
        if self.use_method_of_moments:
            self.apply_method_of_moments()

    def make_cuboid_orientation_magpy(self):
        """Make orientations of each magpylib cuboid. A list of scipy Rotation objects. add error effects
        (may be zero though)"""
        phi_arr = np.pi + np.arange(0, 12) * 2 * np.pi / 3  # direction of magnetization
        phi_arr += np.ravel(self.phi_shift)  # add specified rotation, typically errors
        rotation_all = [Rotation.from_rotvec([0.0, 0.0, phi]) for phi in phi_arr]
        assert len(rotation_all) == self.num_magnets_in_layer
        return rotation_all

    def make_cuboid_positions_magpy(self) -> ndarray:
        """Array of position of each magpylib cuboid, in units of mm. add error effects (may be zero though)"""
        theta_arr = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # location of 12 magnets.
        theta_arr += np.ravel(self.theta_shift)  # add specified rotation, typically errors
        r_arr = self.rp + np.ravel(self.r_magnet_shift)

        r_magnet_center = r_arr + self.magnet_width / 2  # add specified rotation, typically errors
        x_center, y_center = r_magnet_center * np.cos(theta_arr), r_magnet_center * np.sin(theta_arr)
        position_all = np.column_stack((x_center, y_center, np.zeros(self.num_magnets_in_layer)))
        position_all *= METER_TO_MM
        assert position_all.shape == (self.num_magnets_in_layer, 3)
        return position_all

    def make_cuboid_dimensions_magpy(self):
        """Make array of dimension of each magpylib cuboid in units of mm. add error effects (may be zero though)"""
        dimension_single = np.asarray([self.magnet_width, self.magnet_width, self.length])
        dimension_all = dimension_single * np.ones((self.num_magnets_in_layer, 3))
        dimension_all += self.dim_shift
        assert np.all(np.abs(self.dim_shift) < dimension_all)  # model probably doesn't work
        dimension_all *= METER_TO_MM
        assert dimension_all.shape == (self.num_magnets_in_layer, 3)
        return dimension_all

    def make_Mag_Vec_Arr_Magpy(self):
        """Make array of magnetization vector of each magpylib cuboid in units of mT. add error effects (may be zero
        though)"""
        mag_magpy_units = self.M_norm * SI_MagnetizationToMagpy  # uses units of mT for magnetization.
        magnetization_single = np.asarray([mag_magpy_units, 0.0, 0.0])
        magnetization_all = magnetization_single * np.ones((self.num_magnets_in_layer, 3))
        magnetization_all *= (1 + self.M_norm_shift_relative)  # add specified fraction shifts, typically errors
        for M_norm, M_Angle in zip(magnetization_all, self.R_angle_shift):
            R = Rotation.from_rotvec([0.0, M_Angle[0], M_Angle[1]])
            M_norm[:] = R.as_matrix() @ M_norm[:]  # edit in place
        assert magnetization_all.shape == (self.num_magnets_in_layer, 3)
        return magnetization_all

    def valid_indices(self, coords: ndarray, max_dist: float, x0: float, y0: float, z0: float) -> ndarray:
        def valid_indices_range(vals, delta, center):
            return (vals > center - delta) & (vals < center + delta)

        x_vals, y_vals, z_vals = coords.T
        x_valid_indices = valid_indices_range(x_vals, max_dist, x0)
        y_valid_indices = valid_indices_range(y_vals, max_dist, y0)
        z_valid_indices = valid_indices_range(z_vals, max_dist, z0)
        valid_indices = x_valid_indices & y_valid_indices & z_valid_indices
        return valid_indices

    def B_vec(self, coords: ndarray, use_approx: bool = False) -> ndarray:
        """Return B field vector at each coordinate in 'coords'. If 'use_approx' is specified, then magnets that
        are further than a maximum distance from each coordinate in 'coords' are ignored"""
        if use_approx:
            max_dist = self.length / 2 + max(self.rp) * BORE_RADII_MAX_DISTANCE_FACTORS
            x0, y0, z0 = np.array(self.position_meters())
            valid_indices = self.valid_indices(coords, max_dist, x0, y0, z0)
            _B_vec = np.zeros(coords.shape)
            if np.sum(valid_indices) == 0:
                return _B_vec
            else:
                _B_vec[valid_indices] = super().B_vec(coords[valid_indices])
                return _B_vec
        else:
            return super().B_vec(coords)


class HalbachLens(Collection):
    num_magnets_in_layer = 12

    def __init__(self, rp: Union[float, tuple], magnet_width: Union[float, tuple], length: float, magnet_grade: str,
                 position: sequence = None, orientation: Rotation = None,
                 num_disks: int = 1, use_method_of_moments=False, use_standard_mag_errors=False,
                 use_solenoid_field: bool = False, seed=None):
        super().__init__()
        assert length > 0.0
        assert (isinstance(num_disks, int) and num_disks >= 1)
        assert isinstance(orientation, (type(None), Rotation))
        assert isinstance(rp, (float, tuple)) and isinstance(magnet_width, (float, tuple))
        position = (0.0, 0.0, 0.0) if position is None else position
        self.rp: tuple = rp if isinstance(rp, tuple) else (rp,)
        assert length / min(self.rp) >= .5 if use_solenoid_field else True  # shorter than this and the solenoid model
        # is dubious
        self.length: float = length

        self.position_to_set = position
        self.orientation_to_set = orientation  # orientation about body frame
        self.magnet_width: tuple = magnet_width if isinstance(magnet_width, tuple) else (magnet_width,)
        self.use_method_of_moments: bool = use_method_of_moments
        self.use_standard_mag_errors: bool = use_standard_mag_errors
        self.seed = seed
        self.magnet_grade = magnet_grade
        self.num_disks = num_disks
        self.num_layers = len(self.rp)
        self.use_solenoid_field = use_solenoid_field
        self.mur = 1.05

        self.layer_list: list[Layer] = []
        self.build()

    def build(self):
        z_arr, length_arr = self.subdivide_Lens()
        for z_layer, length in zip(z_arr, length_arr):
            for radius_layer, widthLayer in zip(self.rp, self.magnet_width):
                if self.use_standard_mag_errors:
                    dim_variation, mag_vec_angle_variation, mag_norm_variation = self.standard_Magnet_Errors()
                else:
                    dim_variation, mag_vec_angle_variation, mag_norm_variation = np.zeros((12, 3)), np.zeros(
                        (12, 2)), np.zeros((12, 1))
                layer = Layer(radius_layer, widthLayer, length, magnet_grade=self.magnet_grade,
                              position=(0, 0, z_layer),
                              R_angle_shift=mag_vec_angle_variation, dim_shift=dim_variation,
                              M_norm_shift_rel=mag_norm_variation, mur=self.mur)
                self.add(layer)
                self.layer_list.append(layer)
        if self.use_method_of_moments:  # this must come before adding solenoids because the demag does not play nice with
            # coils
            self.apply_method_of_moments()
        if self.use_solenoid_field:
            self.add_solenoid_coils()
        if self.orientation_to_set is not None:
            self.rotate(self.orientation_to_set, anchor=0.0)
        self.move_meters(self.position_to_set)

    def standard_Magnet_Errors(self):
        """Make standard tolerances for permanent magnets. From various sources, particularly K&J magnetics"""
        dim_tol = inch_to_meter(.004)  # dimension variation,inch to meter, +/- meters
        mag_vec_angle_tol = radians(2)  # magnetization vector angle tolerane,degree to radian,, +/- radians
        mag_norm_tol = .0125  # magnetization value tolerance, +/- fraction
        with temporary_seed(self.seed):  # if None, then no special seeding happens
            dim_variation = self.make_Base_Error_Arr_Cartesian(num_params=3) * dim_tol
            MagVecAngleVariation = self.make_Base_Error_Arr_Circular() * mag_vec_angle_tol
            mag_norm_variation = self.make_Base_Error_Arr_Cartesian() * mag_norm_tol
        return dim_variation, MagVecAngleVariation, mag_norm_variation

    def make_Base_Error_Arr_Cartesian(self, num_params: int = 1) -> ndarray:
        """values range between -1 and 1 with shape (12,numParams)"""
        return 2 * (np.random.random_sample((self.num_magnets_in_layer, num_params)) - .5)

    def make_Base_Error_Arr_Circular(self) -> ndarray:
        """Make error array confined inside unit circle. Return results in cartesian with shape (12,2)"""
        theta = 2 * np.pi * np.random.random_sample(self.num_magnets_in_layer)
        radius = np.random.random_sample(self.num_magnets_in_layer)
        x, y = np.cos(theta) * radius, np.sin(theta) * radius
        return np.column_stack((x, y))

    def subdivide_Lens(self) -> tuple[ndarray, ndarray]:
        """To improve accuracu of magnetostatic method of moments, divide the layers into smaller layers. Also used
         if the lens is composed of slices"""
        length_arr = np.ones(self.num_disks) * self.length / self.num_disks
        z_arr = np.cumsum(length_arr) - .5 * self.length - .5 * self.length / self.num_disks
        assert within_tol(np.sum(length_arr), self.length) and within_tol(np.mean(z_arr), 0.0)
        return z_arr, length_arr

    def add_solenoid_coils(self) -> None:
        """Add simple coils through length of lens. This is to remove the region of non zero magnetic field to prevent
        spin flips"""
        coil_diam = 1.95 * min(self.rp) * METER_TO_MM  # slightly smaller than magnet bore
        z_arr, length_arr = self.subdivide_Lens()
        z_lens_min, z_lens_max = z_arr[0] - length_arr[0] / 2, z_arr[-1] + length_arr[-1] / 2
        num_coils = max([round(COILS_PER_RADIUS * (z_lens_max - z_lens_min) / min(self.rp)), 1])
        B_dot_dl = SPIN_FLIP_AVOIDANCE_FIELD * (z_lens_max - z_lens_min)  # amperes law
        current_infinite_solenoid = B_dot_dl / (MAGNETIC_PERMEABILITY * num_coils)  # amperes law
        current = current_infinite_solenoid * np.sqrt(1 + (2 * min(self.rp) / self.length) ** 2)
        coil_locations_z_arr = METER_TO_MM * np.linspace(z_lens_min, z_lens_max, num_coils)
        for coilPosZ in coil_locations_z_arr:
            loop = magpylib.current.Loop(current=current, diameter=coil_diam, position=(0, 0, coilPosZ))
            self.add(loop)

    def B_vec(self, eval_coords: ndarray, use_approx: bool = False) -> ndarray:
        if use_approx:
            _B_vec = np.zeros(eval_coords.shape)
            for layer in self.layer_list:
                _B_vec += layer.B_vec(eval_coords, use_approx=True)
            return _B_vec

        else:
            return super().B_vec(eval_coords)


class HalbachBender(Collection):
    # a model of odd number lenses to represent the symmetry of the segmented bender. The inner lens represents the fully
    # symmetric field

    def __init__(self, rp: float, rb: float, UCAngle: float, Lm: float, magnet_grade: str, num_lenses,
                 use_half_cap_end: tuple[bool, bool],
                 use_pos_mag_angs_only: bool = False, use_method_of_moments=False, use_mag_errors: bool = False,
                 use_solenoid_field: bool = False, magnet_width: float = None):
        # todo: by default I think it should be positive angles only
        super().__init__()
        assert all(isinstance(value, Number) for value in (rp, rb, UCAngle, Lm)) and isinstance(num_lenses, int)
        self.use_half_cap_end = (False, False) if use_half_cap_end is None else use_half_cap_end
        self.rp: float = rp  # radius of bore of magnet, ie to the pole
        self.rb: float = rb  # bending radius
        self.UCAngle: float = UCAngle  # unit cell angle of a HALF single magnet, ie HALF the bending angle of a single magnet. It
        # is called the unit cell because obviously one only needs to use half the magnet and can use symmetry to
        # solve the rest
        self.Lm: float = Lm  # length of single magnet
        self.magnet_grade = magnet_grade
        self.use_solenoid_field = use_solenoid_field
        self.use_pos_mag_angs_only: bool = use_pos_mag_angs_only  # This is used to model the cap amgnet, and the first full
        # segment. No magnets can be below z=0, but a magnet can be right at z=0. Very different behavious wether negative
        # or positive
        self.magnet_width: float = halbach_magnet_width(rp,
                                                        magnetSeparation=0.0) if magnet_width is None else magnet_width
        assert np.tan(.5 * Lm / (rb - self.magnet_width)) <= UCAngle  # magnets should not overlap!
        self.num_lenses: int = num_lenses  # number of lenses in the model
        self.lens_list: list[HalbachLens] = []  # list to hold lenses
        self.lens_angles_arr: ndarray = self.make_lens_angle_array()
        self.use_method_of_moments = use_method_of_moments
        self.useStandardMagnetErrors = use_mag_errors
        self._build()

    def make_lens_angle_array(self) -> ndarray:
        if self.num_lenses == 1:
            if self.use_pos_mag_angs_only:
                raise Exception('Not applicable with only 1 magnet')
            angle_arr = np.asarray([0.0])
        else:
            angle_arr = np.linspace(-2 * self.UCAngle * (self.num_lenses - 1) / 2,
                                    2 * self.UCAngle * (self.num_lenses - 1) / 2, num=self.num_lenses)
        angle_arr = angle_arr - angle_arr.min() if self.use_pos_mag_angs_only else angle_arr
        return angle_arr

    def lens_length_and_angle_iter(self) -> iter:
        """Create an iterable for length of lenses and their angles. This handles the case of using only a half length
        lens at the beginning and/or end of the bender"""

        angle_arr = self.lens_angles_arr.copy()
        assert len(angle_arr) > 1
        length_magnets = [self.Lm] * self.num_lenses
        angle_sep = (angle_arr[1] - angle_arr[0])

        length_magnets[0] = length_magnets[0] / 2 if self.use_half_cap_end[0] else length_magnets[0]
        angle_arr[0] = angle_arr[0] + angle_sep * .25 if self.use_half_cap_end[0] else angle_arr[0]

        length_magnets[-1] = length_magnets[-1] / 2 if self.use_half_cap_end[1] else length_magnets[-1]
        angle_arr[-1] = angle_arr[-1] - angle_sep * .25 if self.use_half_cap_end[1] else angle_arr[-1]

        return zip(length_magnets, angle_arr)

    def _build(self) -> None:
        for Lm, angle in self.lens_length_and_angle_iter():
            lens = HalbachLens(self.rp, self.magnet_width, Lm, magnet_grade=self.magnet_grade,
                               position=(self.rb, 0.0, 0.0),
                               use_standard_mag_errors=self.useStandardMagnetErrors,
                               use_method_of_moments=False)
            R = Rotation.from_rotvec([0.0, -angle, 0.0])
            lens.rotate(R, anchor=0)
            # my angle convention is unfortunately opposite what it should be here. positive theta
            # is clockwise about y axis in the xz plane looking from the negative side of y
            # lens.position(r0)
            self.lens_list.append(lens)
            self.add(lens)
        if self.use_method_of_moments:  # must be done before adding coils because coils dont' play nice
            self.apply_method_of_moments()
        if self.use_solenoid_field:
            self.add_solenoid_coils()

    def B_vec(self, eval_coords: ndarray, use_approx=False) -> ndarray:
        """
        overrides Collection

        :param eval_coords: Coordinate to evaluate magnetic field vector at, m. shape (n,3)
        :param use_approx: Wether to use the approximately true, within 1%, method of neglecting lenses that are
        far from a given coordinate in evalCorods
        :return: The magnetic field vector,  shape (n,3)
        """
        if use_approx:
            _B_vec = np.zeros(eval_coords.shape)
            for lens in self.lens_list:
                _B_vec += lens.B_vec(eval_coords, use_approx=True)
            return _B_vec
        else:
            return super().B_vec(eval_coords)

    def add_solenoid_coils(self) -> None:
        """Add simple coils through length of lens. This is to remove the region of non zero magnetic field to prevent
        spin flips. Solenoid wraps around an imaginary vacuum tube such that the wires but up against the inside edge
        of the magnets where they approach the bending radius the closest"""

        coil_diam = METER_TO_MM * 2 * max_tube_IR_in_segmented_bend(self.rb, self.rp, self.Lm,
                                                                    tube_wall_thickness=MAGNET_WIRE_DIAM / 2.0)
        angle_start = self.lens_angles_arr[0] if self.use_half_cap_end[0] else self.lens_angles_arr[0] - self.UCAngle
        angle_end = self.lens_angles_arr[-1] if self.use_half_cap_end[1] else self.lens_angles_arr[-1] + self.UCAngle
        circumference = self.rb * (angle_end - angle_start)
        num_coils = max([round(COILS_PER_RADIUS * circumference / self.rp), 1])
        B_dot_dl = SPIN_FLIP_AVOIDANCE_FIELD * circumference  # amperes law
        current = B_dot_dl / (MAGNETIC_PERMEABILITY * num_coils)  # amperes law
        for theta in np.linspace(angle_start, angle_end, num_coils):
            loop = magpylib.current.Loop(current=current, diameter=coil_diam, position=(self.rb * METER_TO_MM, 0, 0))
            loop.rotate(Rotation.from_rotvec([0, -theta, 0]), anchor=0)
            self.add(loop)


element_magnets = (HalbachBender, HalbachLens)


def can_magnet_use_approx(magnet):
    return type(magnet) in element_magnets


class ElementMagnetCollection(Collection):
    """Class to hold only magnets used by elements. This allows me to use the use_approx keyword correctly and easily.
    There is a more general way, but I don't need it."""

    def __init__(self, extra_magnets):
        super().__init__()
        self.add_magnets(extra_magnets)

    def add_magnets(self, extra_magnets: Union[list, Collection]):
        """Extra magnets must be an iterable of collections, with each entry being a type of 'element_magnets'"""
        if can_magnet_use_approx(extra_magnets):
            super().add(extra_magnets)
        else:
            assert not isinstance(extra_magnets, Collection)
            for magnet in extra_magnets:
                assert can_magnet_use_approx(magnet)
                self.add(magnet)

    def B_vec(self, eval_coords: ndarray, use_approx: bool = False) -> ndarray:
        _B_vec = np.zeros(eval_coords.shape)
        for el in self:
            _B_vec += el.B_vec(eval_coords, use_approx=use_approx)
        return _B_vec
