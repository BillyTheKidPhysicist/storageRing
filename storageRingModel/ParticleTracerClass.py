import math
import warnings
from math import isnan, sqrt
from typing import Optional

import numba
import numpy as np
from numba.core.errors import NumbaPerformanceWarning

from ParticleClass import Particle
from collisionPhysics import post_collision_momentum, make_collision_params
from constants import GRAVITATIONAL_ACCELERATION
from latticeElements.elements import LensIdeal, CombinerIdeal, Element, BenderIdeal, HalbachBenderSimSegmented, \
    CombinerSim, CombinerHalbachLensSim

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


@numba.njit(numba.float64(numba.float64[:]), cache=False)
def norm_3D(vec):
    return sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


@numba.njit(numba.float64(numba.float64[:], numba.float64[:]), cache=False)
def dot_Prod_3D(veca, vecb):
    return veca[0] * vecb[0] + veca[1] * vecb[1] + veca[2] * vecb[2]


TINY_TIME_STEP = 1e-9  # nanosecond time step to move particle from one element to another


@numba.njit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:], numba.float64), cache=False)
def fast_qNew(q, F, p, h):
    return q + p * h + .5 * F * h ** 2


@numba.njit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:], numba.float64), cache=False)
def fast_pNew(p, F, F_new, h):
    return p + .5 * (F + F_new) * h


@numba.njit(cache=False)
def _transform_To_Next_Element(q, p, r01, r02, ROutEl1, RInEl2):
    # don't try and condense. Because of rounding, results won't agree with other methods and tests will fail
    q = q.copy()
    p = p.copy()
    q[:2] = ROutEl1 @ q[:2]
    q += r01
    q -= r02
    q[:2] = RInEl2 @ q[:2]
    p[:2] = ROutEl1 @ p[:2]
    p[:2] = RInEl2 @ p[:2]
    return q, p


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
        self.use_energy_correction = None

        self.elHasChanged = False  # to record if the particle has changed to another element in the previous step
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

        self.log_phase_space_coords = False  # wether to log lab frame phase space coords at element inputs

    def transform_To_Next_Element(self, q: np.ndarray, p: np.ndarray, nextEll: Element) \
            -> tuple[np.ndarray, np.ndarray]:
        el1 = self.current_el
        el2 = nextEll
        if type(el1) in (BenderIdeal, HalbachBenderSimSegmented):
            r01 = el1.r0
        elif type(el1) in (CombinerHalbachLensSim, CombinerSim, CombinerIdeal):
            r01 = el1.r2
        else:
            r01 = el1.r1
        if type(el2) in (BenderIdeal, HalbachBenderSimSegmented):
            r02 = el2.r0
        elif type(el2) in (CombinerHalbachLensSim, CombinerSim, CombinerIdeal):
            r02 = el2.r2
        else:
            r02 = el2.r1
        return _transform_To_Next_Element(q, p, r01, r02, el1.R_Out, el2.R_In)

    def initialize(self) -> None:
        # prepare for a single particle to be traced
        self.T = 0.0
        if self.particle.clipped is not None:
            self.particle.clipped = False
        LMin = norm_3D(self.particle.pi) * self.h * self.minTimeStepsPerElement
        for el in self.el_list:
            if el.Lo <= LMin:  # have at least a few steps in each element
                raise Exception('element too short for time steps size')
        if self.particle.qi[0] == 0.0:
            raise Exception("a particle appears to be starting with x=0 exactly. This can cause unpredictable "
                            "behaviour")
        self.current_el = self.which_element_lab_coords(self.particle.qi)
        self.particle.current_el = self.current_el
        self.particle.data_logging = not self.use_fast_mode  # if using fast mode, there will NOT be logging
        self.logTracker = 0
        if self.log_phase_space_coords:
            self.particle.el_phase_space_log.append((self.particle.qi.copy(), self.particle.pi.copy()))
        if self.current_el is None:
            self.particle.clipped = True
        else:
            self.particle.clipped = False
            self.q_el = self.current_el.transform_lab_coords_into_element_frame(self.particle.qi)
            self.p_el = self.current_el.transform_Lab_Frame_Vector_Into_Element_Frame(self.particle.pi)
            self.E0 = self.particle.get_energy(self.current_el, self.q_el, self.p_el)
        if self.use_fast_mode is False and self.particle.clipped is False:
            self.particle.log_params(self.current_el, self.q_el, self.p_el)

    def trace(self, particle: Optional[Particle], h: float, T0: float, fast_mode: bool = False,
              accelerated: bool = False,
              use_energy_correction: bool = False, steps_between_logging: int = 1, use_collisions: bool = False,
              log_phase_space_coords: bool = False) -> Particle:
        # trace the particle through the lattice. This is done in lab coordinates. Elements affect a particle by having
        # the particle's position transformed into the element frame and then the force is transformed out. This is obviously
        # not very efficient.
        # qi: initial position coordinates
        # vi: initial velocity coordinates
        # h: timestep
        # T0: total tracing time
        # use_fast_mode: wether to use the performance optimized versoin that doesn't track paramters
        if use_collisions:
            raise NotImplementedError  # the heterogenous tuple was killing performance. Need a new method
        assert 0 < h < 1e-4 and T0 > 0.0  # reasonable ranges
        assert not (use_energy_correction and use_collisions)
        self.use_collisions = use_collisions
        self.use_energy_correction = use_energy_correction
        self.steps_per_logging = steps_between_logging
        self.log_phase_space_coords = log_phase_space_coords
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
        self.particle.finished(self.current_el, self.q_el, self.p_el, total_lattice_length=self.total_lattice_length)

        if self.log_phase_space_coords:
            self.particle.el_phase_space_log.append((self.particle.qf, self.particle.pf))

        return self.particle

    def did_Particle_Survive_To_End(self):
        """
        Check if a particle survived to the end of a lattice. This is only intended for lattices that aren't closed.
        This isn't straight forward because the particle tracing stops when the particle is outside the lattice, then
        the previous position is the final position. Thus, it takes a little extra logic to find out wether a particle
        actually survived to the end in an unclosed lattice

        :return: wether particle has survived to end or not
        """

        assert not self.PTL.is_closed
        el_last = self.el_list[-1]
        # q_el = el_last.transform_lab_coords_into_element_frame(self.q_el)
        # p_el = el_last.transform_Lab_Frame_Vector_Into_Element_Frame(self.p_el)
        if isinstance(el_last, LensIdeal):
            timeStepToEnd = (el_last.L - self.q_el[0]) / self.p_el[0]
        elif isinstance(el_last, CombinerIdeal):
            timeStepToEnd = self.q_el[0] / -self.p_el[0]
        else:
            print('not implemented, falling back to previous behaviour')
            return self.particle.clipped

        if not 0 <= timeStepToEnd <= self.h:
            clipped = True
        else:
            qElEnd = self.q_el + .99 * timeStepToEnd * self.p_el
            clipped = not el_last.is_Coord_Inside(qElEnd)
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
                self.particle.clipped = self.did_Particle_Survive_To_End()

    def multi_step_verlet(self) -> None:
        # collision_params = get_Collision_Params(self.current_el, self.PTL.speed_nominal) if \
        #     self.use_collisions else (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
        results = self._multi_step_verlet(self.q_el, self.p_el, self.T, self.T0, self.h,
                                          self.current_el.numba_functions['force'])
        q_el_new, self.q_el[:], self.p_el[:], self.T, particleOutside = results
        q_el_new = np.array(q_el_new)
        self.particle.T = self.T
        if particleOutside:
            self.check_if_particle_is_outside_and_handle_edge_event(q_el_new, self.q_el, self.p_el)

    @staticmethod
    @numba.njit(cache=False)
    def _multi_step_verlet(qEln, pEln, T, T0, h, force):
        # pylint: disable = E, W, R, C
        # collisionRate = 0.0 if np.isnan(collision_params[0]) else collision_params[1]
        x, y, z = qEln
        px, py, pz = pEln
        Fx, Fy, Fz = force(x, y, z)
        Fz = Fz - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always
        if math.isnan(Fx) or T >= T0:
            particleOutside = True
            qEln, pEln = (x, y, z), (px, py, pz)
            return qEln, qEln, pEln, T, particleOutside
        particleOutside = False
        while True:
            if T >= T0:
                p_el, q_el = (px, py, pz), (x, y, z)
                return q_el, q_el, p_el, T, particleOutside
            x = x + px * h + .5 * Fx * h ** 2
            y = y + py * h + .5 * Fy * h ** 2
            z = z + pz * h + .5 * Fz * h ** 2

            Fx_n, Fy_n, Fz_n = force(x, y, z)
            Fz_n = Fz_n - GRAVITATIONAL_ACCELERATION  # simulated mass is 1kg always

            if math.isnan(Fx_n):
                xo = x - (px * h + .5 * Fx * h ** 2)
                yo = y - (py * h + .5 * Fy * h ** 2)
                zo = z - (pz * h + .5 * Fz * h ** 2)
                p_el, q_el, qEl_o = (px, py, pz), (x, y, z), (xo, yo, zo)
                particleOutside = True
                return q_el, qEl_o, p_el, T, particleOutside
            px = px + .5 * (Fx_n + Fx) * h
            py = py + .5 * (Fy_n + Fy) * h
            pz = pz + .5 * (Fz_n + Fz) * h
            Fx, Fy, Fz = Fx_n, Fy_n, Fz_n
            T += h
            # if collisionRate!=0.0 and np.random.rand() < h * collisionRate:
            #     px, py, pz = post_collision_momentum((px, py, pz), (x, y, z), collision_params)

    def time_step_verlet(self) -> None:
        # the velocity verlet time stepping algorithm. This version recycles the force from the previous step when
        # possible
        q_el = self.q_el  # q old or q sub n
        p_el = self.p_el  # p old or p sub n
        if not self.elHasChanged and self.force_last is not None:  # if the particle is inside the lement it was in
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
        self.elHasChanged = False

    def check_if_particle_is_outside_and_handle_edge_event(self, q_el_next: np.ndarray, q_el: np.ndarray,
                                                           p_el: np.ndarray) -> None:
        # todo: goofy ass naming convention going on here with q_el_next vs q_next_el

        # q_el_next: coordinates that are outside the current element and possibley in the next
        # q_el: coordinates right before this method was called, should still be in the element
        # p_el: momentum for both q_el_next and q_el

        if self.accelerated:
            if self.use_energy_correction:
                p_el[:] += self.momentum_correction_at_bounday(self.E0, q_el, p_el,
                                                               self.current_el.numba_functions['magnetic_potential'],
                                                               self.current_el.numba_functions['force'],
                                                               'leaving')
            if self.log_phase_space_coords:
                qElLab = self.current_el.transform_element_coords_into_lab_frame(
                    q_el_next)  # use the old  element for transform
                pElLab = self.current_el.transform_Element_Frame_Vector_Into_Lab_Frame(
                    p_el)  # use the old  element for transform
                self.particle.el_phase_space_log.append((qElLab, pElLab))
            next_el = self.get_next_element()
            q_next_el, p_nextEl = self.transform_To_Next_Element(q_el_next, p_el, next_el)
            if not next_el.is_Coord_Inside(q_next_el):
                self.particle.clipped = True
            else:
                if self.use_energy_correction:
                    p_nextEl[:] += self.momentum_correction_at_bounday(self.E0, q_next_el, p_nextEl,
                                                                       next_el.numba_functions['magnetic_potential'],
                                                                       next_el.numba_functions['force'],
                                                                       'entering')
                self.particle.cumulative_length += self.current_el.Lo  # add the previous orbit length
                self.current_el = next_el
                self.particle.current_el = next_el
                self.q_el = q_next_el
                self.p_el = p_nextEl
                self.elHasChanged = True
        else:
            el = self.which_element(q_el_next)
            if el is None:  # if outside the lattice
                self.particle.clipped = True
            elif el is not self.current_el:  # element has changed
                if self.use_energy_correction:
                    p_el[:] += self.momentum_correction_at_bounday(self.E0, q_el, p_el,
                                                                   self.current_el.numba_functions[
                                                                       'magnetic_potential'],
                                                                   self.current_el.numba_functions['force'],
                                                                   'leaving')
                next_el = el
                self.particle.cumulative_length += self.current_el.Lo  # add the previous orbit length
                qElLab = self.current_el.transform_element_coords_into_lab_frame(
                    q_el_next)  # use the old  element for transform
                pElLab = self.current_el.transform_Element_Frame_Vector_Into_Lab_Frame(
                    p_el)  # use the old  element for transform
                if self.log_phase_space_coords:
                    self.particle.el_phase_space_log.append((qElLab, pElLab))
                self.current_el = next_el
                self.particle.current_el = next_el
                self.q_el = self.current_el.transform_lab_coords_into_element_frame(
                    qElLab)  # at the beginning of the next element
                self.p_el = self.current_el.transform_Lab_Frame_Vector_Into_Element_Frame(
                    pElLab)  # at the beginning of the next
                # element
                self.elHasChanged = True
                if self.use_energy_correction:
                    self.p_el[:] += self.momentum_correction_at_bounday(self.E0, self.q_el, self.p_el,
                                                                        self.current_el.numba_functions[
                                                                            'magnetic_potential'],
                                                                        self.current_el.numba_functions['force'],
                                                                        'entering')
            else:
                raise Exception('Particle is likely in a region of magnetic field which is invalid because its '
                                'interpolation extends into the magnetic material. Particle is also possibly frozen '
                                'because of broken logic that returns it to the same location.')

    @staticmethod
    @numba.njit(cache=False)
    def momentum_correction_at_bounday(E0, q_el: np.ndarray, p_el: np.ndarray, magnetic_potential, force,
                                       direction: str) -> \
            tuple[float, float, float]:
        # a small momentum correction because the potential doesn't go to zero, nor do i model overlapping potentials
        assert direction in ('entering', 'leaving')
        Fx, Fy, Fz = force(q_el[0], q_el[1], q_el[2])
        F_norm = np.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2)
        if F_norm < 1e-6:  # force is too small, and may cause a division error
            return 0.0, 0.0, 0.0
        else:
            if direction == 'leaving':  # go from zero to non zero potential instantly
                E_mew = dot_Prod_3D(p_el, p_el) / 2.0  # ideally, no magnetic field right at border
                delta_E = E0 - E_mew  # need to lose energy to maintain E0 when the potential turns off
            else:  # go from zero to non zero potentially instantly
                delta_E = -np.array(
                    magnetic_potential(q_el[0], q_el[1], q_el[2]))  # need to lose this energy
            Fx_unit, Fy_unit, Fz_unit = Fx / F_norm, Fy / F_norm, Fz / F_norm
            delta_p_norm = delta_E / (Fx_unit * p_el[0] + Fy_unit * p_el[1] + Fz_unit * p_el[2])
            delta_px, delta_py, delta_pz = delta_p_norm * Fx_unit, delta_p_norm * Fy_unit, delta_p_norm * Fz_unit
            return delta_px, delta_py, delta_pz

    def which_element_lab_coords(self, q_lab: np.ndarray) -> Optional[Element]:
        for el in self.el_list:
            if el.is_Coord_Inside(el.transform_lab_coords_into_element_frame(q_lab)):
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
        if next_el.is_Coord_Inside(next_el.transform_lab_coords_into_element_frame(q_lab)):  # try the next element
            return next_el
        else:
            # now instead look everywhere, except the next element we already checked
            for el in self.el_list:
                if el is not next_el:  # don't waste rechecking current element or next element
                    if el.is_Coord_Inside(el.transform_lab_coords_into_element_frame(q_lab)):
                        return el
            return None
