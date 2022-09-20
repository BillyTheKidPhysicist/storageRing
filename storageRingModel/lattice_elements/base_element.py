"""Contains base element object from which others inherit. An element has the following features and
parameters:

- Attributes relating to element placement such as location and normals on input/output.
- Methods for force and magnetic field calculation.
- Methods for transforming coordinates and vectors between the lab, element and orbit frame.
- For simulated elements, a magnet object that contains a magpylib model of the element used for computing magnetic
    fields.

This is not that well done and could use refactoring. A major issue is the attributes r1,r2, nb,ne are not
consistently defined between elements. For some elements they represent the input/output of the elements' centerline,
for others they represent the input/output of the design orbit."""

import numba
import numpy as np

from type_hints import ndarray


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
        # IMPROVEMENT: r1,r2,ne,nb are not consistent. They describe either orbit coordinates, or physical element coordinates
        # IMPROVEMENT: change naming of the parameters, they are weird
        self.PTL = PTL  # particle tracer lattice object. Used for various constants
        self.nb = None  # normal vector to beginning (clockwise sense) of element.
        self.ne = None  # normal vector to end (clockwise sense) of element

        self.R_Out = None  # 2d matrix to rotate a vector out of the element's reference frame
        self.R_In = None  # 2d matrix to rotate a vector into the element's reference frame
        self.r1 = None  # 3D coordinates of beginning (clockwise sense) of element in lab frame
        self.r2 = None  # 3D coordinates of ending (clockwise sense) of element in lab frame
        self.SO = None  # the shapely object for the element. These are used for plotting, and for
        # finding if the coordinates
        # # are inside an element that can't be found with simple geometry
        self.SO_outer = None  # shapely object that represents the outer edge of the element
        self.outer_half_width = None  # outer diameter/width of the element, where applicable. For example,
        # outer diam of lens is the bore radius plus magnets and mount material radial thickness
        self.ang = ang  # bending angle of the element. 0 for lenses and drifts
        self.orbit_trajectory = None
        self.L = L
        self.index = None
        self.Lo = None  # length of orbit for particle. For lenses and drifts this is the same as the
        # length. This is a nominal value because for segmented benders the path length is not simple to compute
        self.output_offset = 0.0  # some elements have an output offset, like from bender's centrifugal force or
        # #lens combiner
        self.field_fact = 1.0  # factor to modify field values everywhere in space by, including force
        self.numba_functions = {}

    def build_fast_field_helper(self, extra_magnets=None) -> None:
        raise NotImplementedError

    def set_field_fact(self, field_fact: bool):
        assert field_fact > 0.0
        self.field_fact = field_fact

    def magnetic_potential(self, q_el: ndarray) -> float:
        """
        Return magnetic potential energy at position q_el.

        Return magnetic potential energy of a lithium atom in simulation units, where the mass of a lithium-7 atom is
        1kg, at cartesian 3D coordinate q_el in the local element frame. This is done by calling up fast_field_helper, a
        jitclass, which does the actual math/interpolation.

        :param q_el: 3D cartesian position vector in local element frame, numpy.array([x,y,z])
        :return: magnetic potential energy of a lithium atom in simulation units, float
        """
        return self.numba_functions['magnetic_potential'](*q_el)  # will raise NotImplementedError if called

    def force(self, q_el: ndarray) -> ndarray:
        """
        Return force at position q_el.

        Return 3D cartesian force of a lithium at cartesian 3D coordinate q_el in the local element frame. Force vector
        has simulation units where lithium-7 mass is 1kg. This is done by calling up fast_field_helper, a
        jitclass, which does the actual math/interpolation.


        :param q_el: 3D cartesian position vector in local element frame,numpy.array([x,y,z])
        :return: New 3D cartesian force vector, numpy.array([Fx,Fy,Fz])
        """
        return np.asarray(self.numba_functions['force'](*q_el))  # will raise NotImplementedError if called

    def transform_element_coords_into_global_orbit_frame(self, q_el: ndarray,
                                                         cumulative_length: float) -> ndarray:
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

    def transform_element_momentum_into_global_orbit_frame(self, q_el: ndarray, p_el: ndarray) -> ndarray:
        """wraps self.transform_element_momentum_into_local_orbit_frame"""

        return self.transform_element_momentum_into_local_orbit_frame(q_el, p_el)

    def transform_lab_coords_into_element_frame(self, q_lab: ndarray) -> ndarray:
        """
        Generate local cartesian element frame coordinates from cartesian observer/lab frame coordinates

        :param q_lab: 3D cartesian position vector in observer/lab frame,numpy.array([x,y,z])
        :return: New 3D cartesian element frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_element_coords_into_lab_frame(self, q_el: ndarray) -> ndarray:
        """
        Generate cartesian observer/lab frame coordinates from local cartesian element frame coordinates

        :param q_el: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_orbit_frame_into_lab_frame(self, q_orbit: ndarray) -> ndarray:
        """
        Generate global cartesian observer/lab frame coords from non-cartesian local orbit frame coords. Orbit coords
        are similiar to the Frenet-Serret Frame.

        :param q_orbit: 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_element_coords_into_local_orbit_frame(self, q_el: ndarray) -> ndarray:
        """
        Generate non-cartesian local orbit frame coords from local cartesian element frame coords. Orbit coords are
        similiar to the Frenet-Serret Frame.

        :param q_el: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        """
        raise NotImplementedError

    def transform_element_momentum_into_local_orbit_frame(self, q_el: ndarray, p_el: ndarray) -> ndarray:
        """
        Transform momentum vector in element frame in frame moving along with nominal orbit. In this frame px is the
        momentum tangent to the orbit, py is perpindicular and horizontal, pz is vertical.

        :param q_el: 3D Position vector in element frame
        :param p_el: 3D Momentum vector in element frame
        :return: New 3D momentum vector in orbit frame
        """
        raise NotImplementedError

    def transform_lab_frame_vector_into_element_frame(self, vecLab: ndarray) -> ndarray:
        """
        Generate element frame vector from observer/lab frame vector.

        :param vecLab: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz])
        :return: 3D cartesian vector in element frame,numpy.array([vx,vy,vz])
        """""
        vec_new = vecLab.copy()  # prevent editing
        vec_new[:2] = self.R_In @ vec_new[:2]
        return vec_new

    def transform_element_frame_vector_into_lab_frame(self, vecEl: ndarray) -> ndarray:
        """
        Generate observer/lab frame vector from element frame vector.

        :param vecEl: 3D cartesian vector in element frame,numpy.array([vx,vy,vz])
        :return: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz])
        """
        vec_new = vecEl.copy()  # prevent editing
        vec_new[:2] = self.R_Out @ vec_new[:2]
        return vec_new

    def is_coord_inside(self, q_el: ndarray) -> bool:
        """
        Check if a 3D cartesian element frame coordinate is contained within an element's vacuum tube

        :param q_el: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: True if the coordinate is inside, False if outside
        """
        return self.numba_functions['is_coord_in_vacuum'](*q_el)

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
        raise NotImplementedError

    def make_orbit(self):
        from lattice_elements.orbit_trajectories import nominal_particle_trajectory
        self.orbit_trajectory = nominal_particle_trajectory(self)
