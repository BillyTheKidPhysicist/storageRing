import warnings
from math import isnan
from typing import Optional

import numba
import numpy as np


from collision_physics import post_collision_momentum, make_collision_params
from constants import GRAVITATIONAL_ACCELERATION
from helper_tools import is_close_all,full_arctan2
from lattice_elements.elements import LensIdeal, CombinerIdeal, Element, BenderIdeal, BenderSim, \
    CombinerSim, CombinerLensSim
from particle import Particle
from particle_tracer_numba_functions import multi_step_verlet, _transform_To_Next_Element, norm_3D, fast_pNew, \
    fast_qNew, dot_Prod_3D




class ElementTooShortError(Exception):
    pass


TINY_TIME_STEP = 1e-9  # nanosecond time step to move particle from one element to another


# this class does the work of tracing the particles through the lattice with timestepping algorithms.
# it utilizes fast numba functions that are compiled and saved at the moment that the lattice is passed. If the lattice
# is changed, then the particle tracer needs to be updated.

class ParticleTracer:
    minTimeStepsPerElement: int = 2  # if an element is shorter than this, throw an error

    def __init__(self, PTL):
        # lattice: ParticleTracerLattice object typically
        self.el_list = PTL.el_list  # list containing the elements in the lattice in order from first to last (order added)
        self.total_lattice_length = PTL.total_length

        self.PTL = PTL

        self.use_collisions = None
        self.accelerated = None

        self.T = None  # total time elapsed
        self.h = None  # step size

        self.el_has_changed = False  # to record if the particle has changed to another element in the previous step
        self.E0 = None  # total initial energy of particle

        self.particle = None  # particle object being traced
        self.use_fast_mode = None  # wether to use the fast and memory light version that doesn't record parameters of the particle
        self.q_el = None  # this is in the element frame
        self.p_el = None  # this is in the element frame
        self.current_el = None
        self.force_last = None  # the last force value. this is used to save computing time by reusing force

        self.use_fast_mode = None  # wether to log particle positions
        self.T0 = None  # total time to trace
        self.logTracker = None
        self.steps_per_logging = None

        self.log_el_phase_space_coords = False  # wether to log lab frame phase space coords at element inputs

    def transform_To_Next_Element(self, q: np.ndarray, p: np.ndarray, nextEll: Element) \
            -> tuple[np.ndarray, np.ndarray]:
        el1 = self.current_el
        el2 = nextEll
        if type(el1) in (BenderIdeal, BenderSim):
            r01 = el1.r0
        elif type(el1) in (CombinerLensSim, CombinerSim, CombinerIdeal):
            r01 = el1.r2
        else:
            r01 = el1.r1
        if type(el2) in (BenderIdeal, BenderSim):
            r02 = el2.r0
        elif type(el2) in (CombinerLensSim, CombinerSim, CombinerIdeal):
            r02 = el2.r2
        else:
            r02 = el2.r1
        return _transform_To_Next_Element(q, p, r01, r02, el1.R_Out, el2.R_In)

    def initialize(self) -> None:
        assert self.PTL.are_fast_field_helpers_built
        self.T = self.particle.T
        if self.particle.clipped is not None:
            self.particle.clipped = False
        LMin = norm_3D(self.particle.pi) * self.h * self.minTimeStepsPerElement
        for el in self.el_list:
            if el.Lo <= LMin:  # have at least a few steps in each element
                raise ElementTooShortError
        self.current_el = self.which_element_lab_coords(self.particle.qi)
        self.particle.current_el = self.current_el
        self.particle.data_logging = not self.use_fast_mode  # if using fast mode, there will NOT be logging
        self.logTracker = 0
        if self.log_el_phase_space_coords:
            self.particle.el_phase_space_log.append((self.particle.qi.copy(), self.particle.pi.copy()))
        if self.current_el is None:
            self.particle.clipped = True
        else:
            self.particle.clipped = False
            self.q_el = self.current_el.transform_lab_coords_into_element_frame(self.particle.qi)
            self.p_el = self.current_el.transform_lab_frame_vector_into_element_frame(self.particle.pi)
        if self.use_fast_mode is False and self.particle.clipped is False:
            self.particle.log_params(self.current_el, self.q_el, self.p_el)

    def trace(self, particle: Optional[Particle], h: float, T0: float, fast_mode: bool = False,
              accelerated: bool = False,steps_between_logging: int = 1, use_collisions: bool = False,
              log_el_phase_space_coords: bool = False) -> Particle:
        if use_collisions:
            raise NotImplementedError  # the heterogenous tuple was killing performance. Need a new method
        assert 0 < h < 1e-4 and T0 > 0.0  # reasonable ranges
        self.use_collisions = use_collisions
        self.steps_per_logging = steps_between_logging
        self.log_el_phase_space_coords = log_el_phase_space_coords

        if particle is None:
            particle = Particle()
        if particle.traced:
            raise Exception('Particle has previously been traced. Tracing a second time is not supported')
        self.particle = particle
        if self.particle.clipped:  # some particles may come in clipped so ignore them
            self.particle.finished(self.current_el, self.q_el, self.p_el, total_lattice_length=0)
            return self.particle
        self.use_fast_mode = fast_mode
        self.h = h
        self.T0 = float(T0)
        self.initialize()
        self.accelerated = accelerated
        if self.particle.clipped:  # some a particles may be clipped after initializing them because they were about
            # to become clipped
            self.particle.finished(self.current_el, self.q_el, self.p_el, total_lattice_length=0,
                                   was_clipped_immediately=True)
            return particle
        self.time_step_loop()
        self.force_last = None  # reset last force to zero
        # self.particle.log_params(self.current_el,self.q_el,self.p_el)
        self.particle.finished(self.current_el, self.q_el, self.p_el, total_lattice_length=self.total_lattice_length)

        if self.log_el_phase_space_coords:
            self.particle.el_phase_space_log.append((self.particle.qf, self.particle.pf))

        return self.particle

    def does_particle_survive_to_end(self):
        """
        Check if a particle survived to the end of a lattice. This is only intended for lattices that aren't closed.
        This isn't straight forward because the particle tracing stops when the particle is outside the lattice, then
        the previous position is the final position. Thus, it takes a little extra logic to find out wether a particle
        actually survived to the end in an unclosed lattice

        :return: True if particle survived to end, False if not
        """

        assert not self.PTL.is_closed
        el_last = self.el_list[-1]
        if isinstance(el_last, LensIdeal):
            time_step_to_end = (el_last.L - self.q_el[0]) / self.p_el[0]
        elif isinstance(el_last, CombinerIdeal):
            time_step_to_end = self.q_el[0] / -self.p_el[0]
        elif type(el_last) is BenderIdeal:
            time_step_to_end = -self.q_el[1] / self.p_el[1]
        elif type(el_last) is BenderSim:
            time_step_to_end=(-el_last.L_cap-self.q_el[1])/ self.p_el[1]
        else:
            warnings.warn('not implemented, falling back to previous behaviour')
            return self.particle.clipped

        if not 0 <= time_step_to_end <= self.h:
            clipped = True
        else:
            qElEnd = self.q_el + .99 * time_step_to_end * self.p_el
            clipped = not el_last.is_coord_inside(qElEnd)
        return clipped

    def time_step_loop(self) -> None:
        while True:
            if self.T >= self.T0:  # if out of time
                self.particle.clipped = False
                break
            if self.use_fast_mode is False:  # either recording data at each step
                # or the element does not have the capability to be evaluated with the much faster multi_step_verlet
                self.time_step_verlet()
                if self.use_fast_mode is False and self.logTracker % self.steps_per_logging == 0:
                    if not self.particle.clipped:  # nothing to update if particle clipped
                        self.particle.log_params(self.current_el, self.q_el, self.p_el)
                self.logTracker += 1
            else:
                self.multi_step_verlet()
            if self.particle.clipped:
                break
            self.T += self.h
            self.particle.T = self.T

        if not self.PTL.is_closed:
            if self.current_el is self.el_list[-1] and self.particle.clipped:  # only bother if particle is
                # in last element
                self.particle.clipped = self.does_particle_survive_to_end()

    def multi_step_verlet(self) -> None:
        # collision_params = get_Collision_Params(self.current_el, self.PTL.speed_nominal) if \
        #     self.use_collisions else (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
        results = multi_step_verlet(self.q_el, self.p_el, self.T, self.T0, self.h,
                                    self.current_el.numba_functions['force'])
        q_el_new, self.q_el[:], self.p_el[:], self.T, particleOutside = results
        q_el_new = np.array(q_el_new)
        self.particle.T = self.T
        if particleOutside:
            self.check_if_particle_is_outside_and_handle_edge_event(q_el_new, self.q_el, self.p_el)

    def time_step_verlet(self) -> None:
        # the velocity verlet time stepping algorithm. This version recycles the force from the previous step when
        # possible
        q_el = self.q_el  # q old or q sub n
        p_el = self.p_el  # p old or p sub n
        if not self.el_has_changed and self.force_last is not None:  # if the particle is inside the lement it was in
            # last time step, and it's not the first time step, then recycle the force. The particle is starting at the
            # same position it stopped at last time, thus same force
            F = self.force_last
        else:  # the last force is invalid because the particle is at a new position
            F = self.current_el.force(q_el)
            F[2] = F[2] - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always
        # a = F # acceleration old or acceleration sub n
        q_el_new = fast_qNew(q_el, F, p_el, self.h)  # q new or q sub n+1
        F_new = self.current_el.force(q_el_new)
        F_new[2] = F_new[2] - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always
        if isnan(F_new[0]):  # particle is outside element if an array of length 1 with np.nan is returned
            self.check_if_particle_is_outside_and_handle_edge_event(q_el_new, q_el,
                                                                    p_el)  # check if element has changed.
            return
        # a_n = F_new  # acceleration new or acceleration sub n+1
        p_el_new = fast_pNew(p_el, F, F_new, self.h)
        if self.use_collisions:
            collision_params = make_collision_params(self.current_el, self.PTL.speed_nominal)
            if collision_params[0] != 'NONE':
                if np.random.rand() < self.h * collision_params[1]:
                    p_el_new[:] = post_collision_momentum(tuple(p_el_new), tuple(q_el_new), collision_params)

        self.q_el = q_el_new
        self.p_el = p_el_new
        self.force_last = F_new  # record the force to be recycled
        self.el_has_changed = False

    def check_if_particle_is_outside_and_handle_edge_event(self, q_el_next: np.ndarray, q_el: np.ndarray,
                                                           p_el: np.ndarray) -> None:
        # todo: goofy ass naming convention going on here with q_el_next vs q_next_el

        # q_el_next: coordinates that are outside the current element and possibley in the next
        # q_el: coordinates right before this method was called, should still be in the element
        # p_el: momentum for both q_el_next and q_el

        if self.accelerated:
            if self.log_el_phase_space_coords:
                qElLab = self.current_el.transform_element_coords_into_lab_frame(
                    q_el_next)  # use the old  element for transform
                pElLab = self.current_el.transform_element_frame_vector_into_lab_frame(
                    p_el)  # use the old  element for transform
                self.particle.el_phase_space_log.append((qElLab, pElLab))
            next_el = self.get_next_element()
            q_next_el, p_nextEl = self.transform_To_Next_Element(q_el_next, p_el, next_el)
            if not next_el.is_coord_inside(q_next_el):
                self.particle.clipped = True
            else:
                self.particle.cumulative_length += self.current_el.Lo  # add the previous orbit length
                self.current_el = next_el
                self.particle.current_el = next_el
                self.q_el = q_next_el
                self.p_el = p_nextEl
                self.el_has_changed = True
        else:
            el = self.which_element(q_el_next)
            if el is None:  # if outside the lattice
                self.particle.clipped = True
            elif el is not self.current_el:  # element has changed
                next_el = el
                self.particle.cumulative_length += self.current_el.Lo  # add the previous orbit length
                qElLab = self.current_el.transform_element_coords_into_lab_frame(
                    q_el_next)  # use the old  element for transform
                pElLab = self.current_el.transform_element_frame_vector_into_lab_frame(
                    p_el)  # use the old  element for transform
                if self.log_el_phase_space_coords:
                    self.particle.el_phase_space_log.append((qElLab, pElLab))
                self.current_el = next_el
                self.particle.current_el = next_el
                self.q_el = self.current_el.transform_lab_coords_into_element_frame(
                    qElLab)  # at the beginning of the next element
                self.p_el = self.current_el.transform_lab_frame_vector_into_element_frame(
                    pElLab)  # at the beginning of the next
                # element
                self.el_has_changed = True
            else:
                raise Exception('Particle is likely in a region of magnetic field which is invalid because its '
                                'interpolation extends into the magnetic material. Particle is also possibly frozen '
                                'because of broken logic that returns it to the same location.')

    def which_element_lab_coords(self, q_lab: np.ndarray) -> Optional[Element]:
        for el in self.el_list:
            if el.is_coord_inside(el.transform_lab_coords_into_element_frame(q_lab)):
                return el
        return None

    def get_next_element(self) -> Element:
        if self.current_el.index + 1 >= len(self.el_list):
            next_el = self.el_list[0]
        else:
            next_el = self.el_list[self.current_el.index + 1]
        return next_el

    def which_element(self, q_el: np.ndarray) -> Optional[Element]:
        # find which element the particle is in, but check the next element first ,which save time
        # and will be the case most of the time. Also, recycle the element coordinates for use in force evaluation later
        q_lab = self.current_el.transform_element_coords_into_lab_frame(q_el)
        next_el = self.get_next_element()
        if next_el.is_coord_inside(next_el.transform_lab_coords_into_element_frame(q_lab)):  # try the next element
            return next_el
        else:
            # now instead look everywhere, except the next element we already checked
            for el in self.el_list:
                if el is not next_el:  # don't waste rechecking current element or next element
                    if el.is_coord_inside(el.transform_lab_coords_into_element_frame(q_lab)):
                        return el
            return None


def step_particle_past_end(particle: Particle, lattice) -> None:
    """Walk a particle up to just past the end of the lattice"""
    dx = lattice[-1].r2[0] - particle.qf[0]
    overshoot_frac = 1e-6
    dt = dx * (1 + overshoot_frac) / particle.pf[0]
    particle.qf = particle.qf + particle.pf * dt
    particle.T += dt
    particle.cumulative_length += lattice[-1].Lo


def apply_periodicity_to_particle(particle, PTL):
    """Assuming a periodic lattice return a particle to the beginning and correctly adjust the particle time and
    x position. Particle should be very close to end because it will be walked up to just past the end as the first
    step"""
    assert not particle.clipped
    assert is_close_all(PTL[-1].ne, np.array([-1.0, 0, 0]), 1e-12)
    total_lattice_angle = sum(abs(el.ang) for el in PTL)
    assert total_lattice_angle == 0
    step_particle_past_end(particle, PTL)
    particle.qi = particle.qf.copy()
    particle.qi[0] -= PTL[-1].r2[0]
    particle.pi = particle.pf.copy()
    particle.qf = particle.pf = None
    particle.traced = False
    particle.q_vals = list(particle.q_vals)
    particle.p_vals = list(particle.p_vals)
    particle.po_vals = list(particle.po_vals)
    particle.qo_vals = list(particle.qo_vals)
    particle.E_vals = list(particle.E_vals)
    particle.KE_vals = list(particle.KE_vals)
    particle.V_vals = list(particle.V_vals)


def trace_particle_periodic_linear_lattice(particle, pt, h, T, fast_mode=False):
    """Trace a particle through a linear lattice assuming it is periodic"""
    lattice = pt.PTL
    assert lattice.is_lattice_straight()
    qi, pi = particle.qi.copy(), particle.pi.copy()
    particle = pt.trace(particle, h, T, fast_mode=fast_mode)
    iter, max_iters = 0, 10_000
    x = particle.qf[0]

    while not particle.clipped and particle.T < T:
        apply_periodicity_to_particle(particle, lattice)
        particle = pt.trace(particle, h, T, fast_mode=fast_mode)
        x += particle.qf[0]
        iter += 1
        assert iter < max_iters
    particle.qf[0] = x
    particle.qi, particle.pi = qi, pi
    return particle
