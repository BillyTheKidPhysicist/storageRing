from math import sqrt, sin, cos

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from constants import SIMULATION_MAGNETON
from lattice_elements.base_element import BaseElement
from lattice_elements.utilities import full_arctan2
from numbaFunctionsAndObjects import benderIdealFastFunctions
from typeHints import ndarray


class BenderIdeal(BaseElement):
    """
        Element representing a bender/waveguide. Base class for other benders

        Simple ideal model of bending/waveguide element modeled as a toroid. The force is linearly proportional to the
        particle's minor radius in the toroid. In this way, it works as a bent lens. Particles will travel through the
        bender displaced from the center of the  potential (minor radius=0.0) because of the centrifugal effect. Thus,
        to minimize oscillations, the nominal particle trajectory is not straight through the bender, but offset a small
        distance. For the ideal bender, this offset can be calculated analytically. Because there are no fringe fields,
        energy conservation is not expected

        Attributes
        ----------
        Bp: Magnetic field at poleface of bender bore, Teslas.

        rp: Radius (minor) to poleface of bender bore, meters.

        ap: Radius (minor) of aperture bender bore, meters. Effectively the vacuum tube inner radius

        rb: Nominal ending radius of bender/waveguide, meters. This is major radius of the toroid. Note that atoms will
            revolve at a slightly larger radius because of centrifugal effect

        shape: Gemeotric shape of element used for placement. ParticleTracerLatticeClass uses this to assemble lattice

        ro: Orbit bending radius, meter. Larger than self.rb because of centrifugal effect

        """

    def __init__(self, PTL, ang: float, Bp: float, rp: float, rb: float, ap: float):
        super().__init__(PTL, ang=ang)
        self.Bp = Bp
        self.rp = rp
        self.ap = self.rp if ap is None else ap
        self.rb = rb
        self.K = None
        self.shape = 'BEND'
        self.ro = None  # bending radius of orbit, ie rb + r_offset.
        self.r0 = None  # coordinates of center of bender, minus any caps

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.K = (2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
        self.output_offset = sqrt(
            self.rb ** 2 / 4 + self.PTL.speed_nominal ** 2 / self.K) - self.rb / 2  # self.output_Offset(self.rb)
        self.ro = self.rb + self.output_offset
        if self.ang is not None:  # calculation is being delayed until constraints are solved
            self.L = self.rb * self.ang
            self.Lo = self.ro * self.ang

    def build_fast_field_helper(self, extra_magnets=None) -> None:
        numba_func_constants = (self.rb, self.ap, self.ang, self.K, self.field_fact)

        force_args = (numba_func_constants,)
        potential_args = (numba_func_constants,)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(benderIdealFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

    def fill_post_constrained_parameters(self) -> None:
        self.fill_in_and_out_rotation_matrices()

    def fill_in_and_out_rotation_matrices(self) -> None:
        rot = self.theta - self.ang + np.pi / 2
        self.R_Out = Rot.from_rotvec([0.0, 0.0, rot]).as_matrix()[:2, :2]
        self.R_In = Rot.from_rotvec([0.0, 0.0, -rot]).as_matrix()[:2, :2]

    def transform_lab_coords_into_element_frame(self, q_lab: ndarray) -> ndarray:
        """Overrides abstract method from Element."""
        q_new = q_lab - self.r0
        q_new = self.transform_lab_frame_vector_into_element_frame(q_new)
        return q_new

    def transform_element_coords_into_lab_frame(self, q_el: ndarray) -> ndarray:
        """Overrides abstract method from Element."""
        q_new = q_el.copy()
        q_new = self.transform_element_frame_vector_into_lab_frame(q_new)
        q_new = q_new + self.r0
        return q_new

    def transform_element_coords_into_local_orbit_frame(self, q_el: ndarray) -> ndarray:
        """Overrides abstract method from Element."""
        qo = q_el.copy()
        phi = self.ang - full_arctan2(qo)  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = sqrt(q_el[0] ** 2 + q_el[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo

    def transform_orbit_frame_into_lab_frame(self, q_orbit: ndarray) -> ndarray:
        """Overrides abstract method from Element."""
        raise NotImplementedError  # there is an error here with yo
        xo, yo, zo = q_orbit
        phi = self.ang - xo / self.ro
        x_lab = self.ro * cos(phi)
        y_lab = self.ro * sin(phi)
        z_lab = zo
        q_lab = np.asarray([x_lab, y_lab, z_lab])
        q_lab[:2] = self.R_Out @ q_lab[:2]
        q_lab += self.r0
        return q_lab

    def transform_element_momentum_into_local_orbit_frame(self, q_el: ndarray, p_el: ndarray) -> ndarray:
        """Overrides abstract method from Element class. Simple cartesian to cylinderical coordinates"""

        x, y = q_el[:2]
        x_dot, y_dot, z_dot = p_el
        r = sqrt(x ** 2 + y ** 2)
        r_dot = (x * x_dot + y * y_dot) / r
        theta_dot = (x * y_dot - x_dot * y) / r ** 2
        velocity_tangent = -r * theta_dot
        return np.array([velocity_tangent, r_dot, z_dot])  # tanget, perpindicular horizontal, perpindicular vertical
