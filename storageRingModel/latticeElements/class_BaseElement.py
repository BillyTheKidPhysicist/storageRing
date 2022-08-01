from typing import Optional

import numba
import numpy as np
from shapely.geometry import Polygon

from constants import SIMULATION_MAGNETON


# todo: a base geometry inheritance is most logical


def wrap_numba_func(func, args):
    @numba.njit()
    def func_wrapper(x, y, z):
        return func(x, y, z, *args)

    return func_wrapper


class BaseElement:
    """
    Base class for other elements. Contains universal attributes and methods.

    An element is the fundamental component of a neutral atom storage ring/injector. An arrangment of elements is called
    a lattice, as is done in accelerator physics. Elements are intended to be combined together such that particles can
    smoothly move from one to another, and many class variables serves this purpose. An element also contains methods
    for force vectors and magnetic potential at a point in space. It will also contain methods to generate fields values
    and construct itself, which is not always trivial.
    """

    def __init__(self, PTL, ang: float = 0.0, L=None):
        self.theta = None  # angle that describes an element's rotation in the xy plane.
        # SEE EACH ELEMENT FOR MORE DETAILS
        # -Straight elements like lenses and drifts: theta=0 is the element's input at the origin and the output pointing
        # east. for theta=90 the output is pointing up.
        # -Bending elements without caps: at theta=0 the outlet is at (bending radius,0) pointing south with the input
        # at some angle counterclockwise. a 180 degree bender would have the inlet at (-bending radius,0) pointing south.
        # force is a continuous function of r and theta, ie a revolved cross section of a hexapole
        # -Bending  elements with caps: same as without caps, but keep in mind that the cap on the output would be BELOW
        # y=0
        # combiner: theta=0 has the outlet at the origin and pointing to the west, with the inlet some distance to the right
        # and pointing in the NE direction
        # todo: r1,r2,ne,nb are not consistent. They describe either orbit coordinates, or physical element coordinates
        self.PTL = PTL  # particle tracer lattice object. Used for various constants
        self.nb: Optional[np.ndarray] = None  # normal vector to beginning (clockwise sense) of element.
        self.ne: Optional[np.ndarray] = None  # normal vector to end (clockwise sense) of element

        self.R_Out: Optional[np.ndarray] = None  # 2d matrix to rotate a vector out of the element's reference frame
        self.R_In: Optional[np.ndarray] = None  # 2d matrix to rotate a vector into the element's reference frame
        self.r1: Optional[np.ndarray] = None  # 3D coordinates of beginning (clockwise sense) of element in lab frame
        self.r2: Optional[np.ndarray] = None  # 3D coordinates of ending (clockwise sense) of element in lab frame
        self.SO: Optional[Polygon] = None  # the shapely object for the element. These are used for plotting, and for
        # finding if the coordinates
        # # are inside an element that can't be found with simple geometry
        self.SO_outer: Optional[Polygon] = None  # shapely object that represents the outer edge of the element
        self.outer_half_width: Optional[
            float] = None  # outer diameter/width of the element, where applicable. For example,
        # outer diam of lens is the bore radius plus magnets and mount material radial thickness
        self.ang = ang  # bending angle of the element. 0 for lenses and drifts
        self.L: Optional[float] = L
        self.index: Optional[int] = None
        self.Lo: Optional[float] = None  # length of orbit for particle. For lenses and drifts this is the same as the
        # length. This is a nominal value because for segmented benders the path length is not simple to compute
        self.output_offset: float = 0.0  # some elements have an output offset, like from bender's centrifugal force or
        # #lens combiner
        self.field_fact: float = 1.0  # factor to modify field values everywhere in space by, including force
        self.numba_functions: dict = {}

    def build_fast_field_helper(self, extra_magnets=None) -> None:
        raise NotImplementedError

    def set_field_fact(self, field_fact: bool):
        assert field_fact > 0.0
        self.field_fact = field_fact

    def perturb_element(self, shift_y: float, shift_z: float, rot_angle_y: float, rot_angle_z: float):
        """
        perturb the alignment of the element relative to the vacuum tube. The vacuum tube will remain unchanged, but
        the element will be shifted, and therefore the force it applies will be as well. This is modeled as shifting
        and rotating the supplied coordinates to force and magnetic field function, then rotating the force

        :param shift_y: Shift in the y direction in element frame
        :param shift_z: Shift in the z direction in the element frame
        :param rot_angle_y: Rotation about y axis of the element
        :param rot_angle_z: Rotation about z axis of the element
        :return:
        """

        raise NotImplementedError
        self.fast_field_helper.numbaJitClass.numbaJitClass.update_Element_Perturb_args(shift_y, shift_z, rot_angle_y,
                                                                                       rot_angle_z)

    def magnetic_potential(self, q_el: np.ndarray) -> float:
        """
        Return magnetic potential energy at position q_el.

        Return magnetic potential energy of a lithium atom in simulation units, where the mass of a lithium-7 atom is
        1kg, at cartesian 3D coordinate q_el in the local element frame. This is done by calling up fast_field_helper, a
        jitclass, which does the actual math/interpolation.

        :param q_el: 3D cartesian position vector in local element frame, numpy.array([x,y,z])
        :return: magnetic potential energy of a lithium atom in simulation units, float
        """
        return self.numba_functions['magnetic_potential'](*q_el)  # will raise NotImplementedError if called

    def force(self, q_el: np.ndarray) -> np.ndarray:
        """
        Return force at position q_el.

        Return 3D cartesian force of a lithium at cartesian 3D coordinate q_el in the local element frame. Force vector
        has simulation units where lithium-7 mass is 1kg. This is done by calling up fast_field_helper, a
        jitclass, which does the actual math/interpolation.


        :param q_el: 3D cartesian position vector in local element frame,numpy.array([x,y,z])
        :return: New 3D cartesian force vector, numpy.array([Fx,Fy,Fz])
        """
        return np.asarray(self.numba_functions['force'](*q_el))  # will raise NotImplementedError if called

    def transform_element_coords_into_global_orbit_frame(self, q_el: np.ndarray,
                                                         cumulative_length: float) -> np.ndarray:
        """
        Generate coordinates in the non-cartesian global orbit frame that grows cumulatively with revolutions, from
        observer/lab cartesian coordinates.

        :param q_el: 3D cartesian position vector in observer/lab frame,numpy.array([x,y,z])
        :param cumulative_length: total length in orbit frame traveled so far. For a series of linear elements this
        would simply be the sum of their length, float
        :return: New 3D global orbit frame position, numpy.ndarray([x,y,z])
        """

        q_orbit = self.transform_element_coords_into_local_orbit_frame(q_el)
        q_orbit[0] = q_orbit[0] + cumulative_length  # longitudinal component grows
        return q_orbit

    def transform_element_momentum_into_global_orbit_frame(self, q_el: np.ndarray, p_el: np.ndarray) -> np.ndarray:
        """wraps self.transform_element_momentum_into_local_orbit_frame"""

        return self.transform_element_momentum_into_local_orbit_frame(q_el, p_el)

    def transform_lab_coords_into_element_frame(self, q_lab: np.ndarray) -> np.ndarray:
        """
        Generate local cartesian element frame coordinates from cartesian observer/lab frame coordinates

        :param q_lab: 3D cartesian position vector in observer/lab frame,numpy.array([x,y,z])
        :return: New 3D cartesian element frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_element_coords_into_lab_frame(self, q_el: np.ndarray) -> np.ndarray:
        """
        Generate cartesian observer/lab frame coordinates from local cartesian element frame coordinates

        :param q_el: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_orbit_frame_into_lab_frame(self, q_orbit: np.ndarray) -> np.ndarray:
        """
        Generate global cartesian observer/lab frame coords from non-cartesian local orbit frame coords. Orbit coords
        are similiar to the Frenet-Serret Frame.

        :param q_orbit: 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_element_coords_into_local_orbit_frame(self, q_el: np.ndarray) -> np.ndarray:
        """
        Generate non-cartesian local orbit frame coords from local cartesian element frame coords. Orbit coords are
        similiar to the Frenet-Serret Frame.

        :param q_el: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        """
        raise NotImplementedError

    def transform_element_momentum_into_local_orbit_frame(self, q_el: np.ndarray, p_el: np.ndarray) -> np.ndarray:
        """
        Transform momentum vector in element frame in frame moving along with nominal orbit. In this frame px is the
        momentum tangent to the orbit, py is perpindicular and horizontal, pz is vertical.

        :param q_el: 3D Position vector in element frame
        :param p_el: 3D Momentum vector in element frame
        :return: New 3D momentum vector in orbit frame
        """
        raise NotImplementedError

    def transform_Lab_Frame_Vector_Into_Element_Frame(self, vecLab: np.ndarray) -> np.ndarray:
        """
        Generate element frame vector from observer/lab frame vector.

        :param vecLab: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz])
        :return: 3D cartesian vector in element frame,numpy.array([vx,vy,vz])
        """""
        vec_new = vecLab.copy()  # prevent editing
        vec_new[:2] = self.R_In @ vec_new[:2]
        return vec_new

    def transform_Element_Frame_Vector_Into_Lab_Frame(self, vecEl: np.ndarray) -> np.ndarray:
        """
        Generate observer/lab frame vector from element frame vector.

        :param vecEl: 3D cartesian vector in element frame,numpy.array([vx,vy,vz])
        :return: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz])
        """
        vec_new = vecEl.copy()  # prevent editing
        vec_new[:2] = self.R_Out @ vec_new[:2]
        return vec_new

    def is_Coord_Inside(self, q_el: np.ndarray) -> bool:
        """
        Check if a 3D cartesian element frame coordinate is contained within an element's vacuum tube

        :param q_el: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: True if the coordinate is inside, False if outside
        """
        return self.numba_functions['is_coord_in_vacuum'](*q_el)  # will raise NotImplementedError if called

    def assign_numba_functions(self, func_module, force_args, potential_args, is_coord_in_vacuum_args) -> None:
        self.numba_functions['force'] = wrap_numba_func(func_module.force, force_args)
        self.numba_functions['magnetic_potential'] = wrap_numba_func(func_module.magnetic_potential, potential_args)
        self.numba_functions['is_coord_in_vacuum'] = wrap_numba_func(func_module.is_coord_in_vacuum,
                                                                     is_coord_in_vacuum_args)

    def fill_pre_constrained_parameters(self):
        """Fill available geometric parameters before constrained lattice layout is solved. Fast field helper, shapely
        objects, and positions still need to be solved for/computed after this point. Most elements call compute all
        their internal parameters before the floorplan is solved, but lenses may have length unspecified and bending
        elements may have bending angle or number of magnets unspecified for example"""
        raise NotImplementedError

    def fill_post_constrained_parameters(self):
        """Fill internal parameters after constrained lattice layout is solved. See fill_Pre_Constrainted_Parameters.
        At this point everything about the geometry of the element is specified"""

    def shape_field_data_3D(self, data: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Shape 3D field data for fast linear interpolation method

        Take an array with the shape (n,7) where n is the number of points in space. Each row
        must have the format [x,y,z,gradxB,gradyB,gradzB,B] where B is the magnetic field norm at x,y,z and grad is the
        partial derivative. The data must be from a 3D grid of points with no missing points or any other funny business
        and the order of points doesn't matter. Return arrays are raveled for use by fast interpolater

        :param data: (n,7) numpy array of points originating from a 3d grid
        :return: tuple of 7 arrays, first 3 are grid edge coords (x,y,z) and last 4 are flattened field values
        (Fx,Fy,Fz,V)
        """
        assert data.shape[1] == 7 and len(data) > 2 ** 3
        x_arr = np.unique(data[:, 0])
        y_arr = np.unique(data[:, 1])
        z_arr = np.unique(data[:, 2])
        assert all(not np.any(np.isnan(arr)) for arr in (x_arr, y_arr, z_arr))

        num_x = x_arr.shape[0]
        num_y = y_arr.shape[0]
        num_z = z_arr.shape[0]
        Fx_matrix = np.empty((num_x, num_y, num_z))
        Fy_matrix = np.empty((num_x, num_y, num_z))
        Fz_matrix = np.empty((num_x, num_y, num_z))
        V_matrix = np.zeros((num_x, num_y, num_z))
        x_indices = np.argwhere(data[:, 0][:, None] == x_arr)[:, 1]
        y_indices = np.argwhere(data[:, 1][:, None] == y_arr)[:, 1]
        z_indices = np.argwhere(data[:, 2][:, None] == z_arr)[:, 1]
        Fx_matrix[x_indices, y_indices, z_indices] = -SIMULATION_MAGNETON * data[:, 3]
        Fy_matrix[x_indices, y_indices, z_indices] = -SIMULATION_MAGNETON * data[:, 4]
        Fz_matrix[x_indices, y_indices, z_indices] = -SIMULATION_MAGNETON * data[:, 5]
        V_matrix[x_indices, y_indices, z_indices] = SIMULATION_MAGNETON * data[:, 6]
        V_Flat, Fx_Flat, Fy_Flat, Fz_Flat = V_matrix.ravel(), Fx_matrix.ravel(), Fy_matrix.ravel(), Fz_matrix.ravel()
        return x_arr, y_arr, z_arr, Fx_Flat, Fy_Flat, Fz_Flat, V_Flat

    def shape_field_data_2D(self, data: np.ndarray) -> tuple[np.ndarray, ...]:
        """2D version of shape_field_data_3D. Data must be shape (n,5), with each row [x,y,Fx,Fy,V]"""
        assert data.shape[1] == 5 and len(data) > 2 ** 3
        x_arr = np.unique(data[:, 0])
        y_arr = np.unique(data[:, 1])
        num_x = x_arr.shape[0]
        num_y = y_arr.shape[0]
        B_grad_x_matrix = np.zeros((num_x, num_y))
        B_grad_y_matrix = np.zeros((num_x, num_y))
        B0_matrix = np.zeros((num_x, num_y))
        x_indices = np.argwhere(data[:, 0][:, None] == x_arr)[:, 1]
        y_indices = np.argwhere(data[:, 1][:, None] == y_arr)[:, 1]

        B_grad_x_matrix[x_indices, y_indices] = data[:, 2]
        B_grad_y_matrix[x_indices, y_indices] = data[:, 3]
        B0_matrix[x_indices, y_indices] = data[:, 4]
        Fx_matrix = -SIMULATION_MAGNETON * B_grad_x_matrix
        FyMatrix = -SIMULATION_MAGNETON * B_grad_y_matrix
        V_matrix = SIMULATION_MAGNETON * B0_matrix
        V_Flat, Fx_Flat, Fy_Flat = np.ravel(V_matrix), np.ravel(Fx_matrix), np.ravel(FyMatrix)
        return x_arr, y_arr, Fx_Flat, Fy_Flat, V_Flat

    def get_valid_jitter_amplitude(self):
        """If jitter (radial misalignment) amplitude is too large, it is clipped."""
        return self.PTL.jitter_amp
