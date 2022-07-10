import copy
import warnings
from math import isnan
from typing import Union, Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl

from constants import DEFAULT_ATOM_SPEED
from latticeElements.elements import Element


class Particle:
    # This object represents a single particle with unit mass. It can track parameters such as position, momentum, and
    # energies, though these are computationally intensive and are not enabled by default. It also tracks where it was
    # clipped if a collision with an apeture occured, the number of revolutions before clipping and other parameters of
    # interest.

    def __init__(self, qi: Optional[np.ndarray] = None, pi: Optional[np.ndarray] = None, probability: float = 1.0):
        if qi is None:
            qi = np.array([-1e-10, 0, 0])
        if pi is None:
            pi = np.asarray([-DEFAULT_ATOM_SPEED, 0.0, 0.0])
        assert len(qi) == 3 and len(pi) == 3 and 0.0 <= probability <= 1.0
        if probability != 1.0:
            raise NotImplementedError  # i'm not' sure how correct the methods depending on this are anymore
        self.qi = qi.copy()  # initial position, lab frame, meters
        self.pi = pi.copy()  # initial momentu, lab frame, meters*kg/s, where mass=1
        self.qf = None  # final position
        self.pf = None  # final momentum
        self.T = 0  # time of particle in simulation
        self.traced = False  # recored wether the particle has already been sent throught the particle tracer
        self.color = None  # color that can be added to each particle for plotting

        self.force = None  # current force on the particle
        self.current_el = None  # which element the particle is ccurently in
        self.current_el_index = None  # Index of the elmenent that the particle is curently in. THis remains unchanged even
        # after the particle leaves the tracing algorithm and such can be used to record where it clipped
        self.cumulative_length = 0  # total length traveled by the particle IN TERMS of lattice elements. It updates after
        # the particle leaves an element by adding that elements length (particle trajectory length that is)
        self.revolutions = 0  # revolutions particle makd around lattice. Initially zero
        self.clipped = None  # wether particle clipped an apeture
        self.data_logging = None  # wether the particle is loggin parameters such as position and energy. This will typically be
        # false when use_fast_mode is being used in the particle tracer class
        # these lists track the particles momentum, position etc during the simulation if that feature is enable. Later
        # they are converted into arrays
        self._p_list = []  # List of momentum vectors
        self._q_list = []  # List of position vector
        self._qo_list = []  # List of position in orbit frame vectors
        self._po_list = []
        self._t_list = []  # kinetic energy list. Each entry contains the element index and corresponding energy
        self._v_list = []  # potential energy list. Each entry contains the element index and corresponding energy
        # array versions
        self.p_arr = None
        self.p0Arr = None  # array of norm of momentum.
        self.q_arr = None
        self.qo_arr = None
        self.po_arr = None
        self.T_arr = None
        self.V_arr = None
        self.E_arr = None  # total energy
        self.elDeltaEDict = {}  # dictionary to hold energy changes that occur traveling through an element. Entries are
        # element index and list of energy changes for each pass
        self.probability = probability  # used for swarm behaviour based on probability
        self.el_phase_space_log = []  # to log the phase space coords at the beginning of each element. Lab frame
        self.total_lattice_length = None

    def reset(self) -> None:
        # reset the particle
        self.__init__(qi=self.qi, pi=self.pi, probability=self.probability)

    def __str__(self) -> str:
        string = '------particle-------\n'
        string += 'qi: ' + str(self.qi) + '\n'
        string += 'pi: ' + str(self.pi) + '\n'
        string += 'p: ' + str(self.pf) + '\n'
        string += 'q: ' + str(self.qf) + '\n'
        string += 'current element: ' + str(self.current_el) + ' \n '
        string += 'revolution: ' + str(self.revolutions) + ' \n'
        return string

    def log_Params(self, currentEl: Element, q_el: np.ndarray, p_el: np.ndarray) -> None:
        q_lab = currentEl.transform_element_coords_into_lab_frame(q_el)
        pLab = currentEl.transform_Element_Frame_Vector_Into_Lab_Frame(p_el)
        self._q_list.append(q_lab.copy())
        self._p_list.append(pLab.copy())
        self._t_list.append((currentEl.index, np.sum(pLab ** 2) / 2.0))
        if currentEl is not None:
            q_el = currentEl.transform_lab_coords_into_element_frame(q_lab)
            elIndex = currentEl.index
            self._qo_list.append(currentEl.transform_element_coords_into_global_orbit_frame(q_el, self.cumulative_length))
            self._po_list.append(currentEl.transform_element_momentum_into_global_orbit_frame(q_el, p_el))
            self._v_list.append((elIndex, currentEl.magnetic_potential(q_el)))

    def get_Energy(self, currentEl: Element, q_el: np.ndarray, p_el: np.ndarray) -> float:
        V = currentEl.magnetic_potential(q_el)
        T = np.sum(p_el ** 2) / 2.0
        return T + V

    def fill_Energy_Array_And_Dicts(self) -> None:
        self.T_arr = np.asarray([entry[1] for entry in self._t_list])
        self.V_arr = np.asarray([entry[1] for entry in self._v_list])
        self.E_arr = self.T_arr.copy() + self.V_arr.copy()

        if self.E_arr.shape[0] > 1:
            elementIndexPrev = self._t_list[0][0]
            E_AfterEnteringEl = self.E_arr[0]

            for i, _ in enumerate(self._t_list):
                if self._t_list[i][0] != elementIndexPrev:
                    E_BeforeLeavingEl = self.E_arr[i - 1]
                    deltaE = E_BeforeLeavingEl - E_AfterEnteringEl
                    if str(elementIndexPrev) not in self.elDeltaEDict:
                        self.elDeltaEDict[str(elementIndexPrev)] = [deltaE]
                    else:
                        self.elDeltaEDict[str(elementIndexPrev)].append(deltaE)
                    E_AfterEnteringEl = self.E_arr[i]
                    elementIndexPrev = self._t_list[i][0]
        self._t_list, self._v_list = [], []

    def finished(self, currentEl: Optional[Element], q_el: np.ndarray, p_el: np.ndarray,
                 total_lattice_length: Optional[float] = None, clippedImmediately=False) -> None:
        # finish tracing with the particle, tie up loose ends
        # totalLaticeLength: total length of periodic lattice
        self.traced = True
        self.force = None
        if clippedImmediately:
            self.qf, self.pf = self.qi.copy(), self.pi.copy()
        if self.data_logging:
            self.q_arr = np.asarray(self._q_list)
            self._q_list = []  # save memory
            self.p_arr = np.asarray(self._p_list)
            self._p_list = []
            self.qo_arr = np.asarray(self._qo_list)
            self._qo_list = []
            self.po_arr = np.asarray(self._po_list)
            self._po_list = []
            if self.p_arr.shape[0] != 0:
                self.p0Arr = npl.norm(self.p_arr, axis=1)
            self.fill_Energy_Array_And_Dicts()
        if self.current_el is not None:
            self.current_el = currentEl
            self.qf = self.current_el.transform_element_coords_into_lab_frame(q_el)
            self.pf = self.current_el.transform_Element_Frame_Vector_Into_Lab_Frame(p_el)
            self.current_el_index = self.current_el.index
            if total_lattice_length is not None:
                self.total_lattice_length = total_lattice_length
                qoFinal = self.current_el.transform_element_coords_into_global_orbit_frame(q_el, self.cumulative_length)
                self.revolutions = qoFinal[0] / total_lattice_length
            self.current_el = None  # to save memory

    def plot_Energies(self, showOnlyTotalEnergy: bool = False) -> None:
        if self.E_arr.shape[0] == 0:
            raise Exception('PARTICLE HAS NO LOGGED POSITION')
        E_arr = self.E_arr
        T_arr = self.T_arr
        V_arr = self.V_arr
        qo_arr = self.qo_arr
        plt.close('all')
        plt.title(
            'Particle energies vs position. \n Total initial energy is ' + str(np.round(E_arr[0], 1)) + ' energy units')
        distFact = self.total_lattice_length if self.total_lattice_length is not None else 1.0
        plt.plot(qo_arr[:, 0] / distFact, E_arr - E_arr[0], label='E')
        if not showOnlyTotalEnergy:
            plt.plot(qo_arr[:, 0] / distFact, T_arr - T_arr[0], label='T')
            plt.plot(qo_arr[:, 0] / distFact, V_arr - V_arr[0], label='V')
        plt.ylabel("Energy, simulation units")
        if self.total_lattice_length is not None:
            plt.xlabel("Distance along lattice, revolutions")
        else:
            plt.xlabel("Distance along lattice, meters")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_Orbit_Reference_Frame_Position(self, plotYAxis: bool = 'y') -> None:
        if plotYAxis not in ('y', 'z'):
            raise Exception('plotYAxis MUST BE EITHER \'y\' or \'z\'')
        if self.qo_arr.shape[0] == 0:
            warnings.warn('Particle has no logged position values')
            qo_arr = np.zeros((1, 3)) + np.nan
        else:
            qo_arr = self.qo_arr
        if plotYAxis == 'y':
            yPlot = qo_arr[:, 1]
        else:
            yPlot = qo_arr[:, 2]
        plt.close('all')
        plt.plot(qo_arr[:, 0], yPlot)
        plt.ylabel('Trajectory offset, m')
        plt.xlabel('Trajectory length, m')
        plt.grid()
        plt.show()

    def copy(self):
        return copy.deepcopy(self)


class Swarm:

    def __init__(self):
        self.particles: list[Particle] = []  # list of particles in swarm

    def add_New_Particle(self, qi: Optional[np.ndarray] = None, pi: Union[np.ndarray] = None) -> None:
        # add an additional particle to phase space
        # qi: spatial coordinates
        # pi: momentum coordinates
        if pi is None:
            pi = np.asarray([-DEFAULT_ATOM_SPEED, 0.0, 0.0])
        if qi is None:
            qi = np.asarray([-1e-10, 0.0, 0.0])
        self.particles.append(Particle(qi, pi))

    def add(self, particle: Particle):
        self.particles.append(particle)

    def survival_Rev(self) -> float:
        # return average number of revolutions of particles
        revs = 0
        for particle in self.particles:
            if particle.clipped is None:
                raise Exception('PARTICLE HAS NOT BEEN TRACED')
            if isnan(particle.revolutions):
                raise Exception('Particle revolutions have an issue')
            if particle.revolutions is not None:
                revs += particle.revolutions

        meanRevs = revs / self.num_Particles()
        return meanRevs

    def longest_Particle_Life_Revolutions(self) -> float:
        # return number of revolutions of longest lived particle
        maxList = []
        for particle in self.particles:
            if particle.revolutions is not None:
                maxList.append(particle.revolutions)
        if len(maxList) == 0:
            return 0.0
        else:
            return max(maxList)

    def survival_Bool(self, frac: bool = True) -> float:
        # returns fraction of particles that have survived, ie not clipped.
        # frac: if True, return the value as a fraction, the number of surviving particles divided by total particles
        numSurvived = 0.0
        for particle in self.particles:
            if particle.clipped is None:
                raise Exception('PARTICLE HAS NO DATA ON SURVIVAL')
            numSurvived += float(not particle.clipped)  # if it has NOT clipped then turn that into a 1.0
        if frac:
            return numSurvived / len(self.particles)
        else:
            return numSurvived

    def __iter__(self) -> Iterable[Particle]:
        return (particle for particle in self.particles)

    def __len__(self):
        return len(self.particles)

    def copy(self):
        return copy.deepcopy(self)

    def quick_Copy(self):  # only copy the initial conditions. For swarms that havn't been traced or been monkeyed
        # with at all
        swarm_new = Swarm()
        for particle in self.particles:
            assert not particle.traced
            particleNew = Particle(qi=particle.qi.copy(), pi=particle.pi.copy())
            particleNew.probability = particle.probability
            particleNew.color = particle.color
            swarm_new.particles.append(particleNew)
        return swarm_new

    def num_Particles(self, weighted: bool = False, unClippedOnly: bool = False) -> float:

        if weighted and unClippedOnly:
            return sum([(not p.clipped) * p.probability for p in self.particles])
        elif weighted and not unClippedOnly:
            return sum([p.probability for p in self.particles])
        elif not weighted and unClippedOnly:
            return sum([not p.clipped for p in self.particles])
        else:
            return len(self.particles)

    def num_Revs(self, weighted: bool = False) -> int:
        if weighted:
            return sum([particle.revolutions for particle in self.particles])
        else:
            return sum([particle.revolutions * particle.probability for particle in self.particles])

    def weighted_flux_mult(self) -> float:
        # only for circular lattice
        if self.num_Particles() == 0:
            return 0.0
        assert all(particle.traced for particle in self.particles)
        numWeighedtRevs = self.num_Revs(weighted=True)
        numWeightedParticles = self.num_Particles(weighted=True)
        return numWeighedtRevs / numWeightedParticles

    def lattice_Flux(self, weighted: bool = False) -> float:
        # only for circular lattice. This gives the average flux in a cross section of the lattice. Only makes sense
        # for many more than one revolutions
        totalFlux = 0
        for particle in self.particles:
            flux = particle.revolutions / np.linalg.norm(particle.pi)
            flux = flux * particle.probability if weighted else flux
            totalFlux += flux
        return totalFlux

    def reset(self) -> None:
        # reset the swarm.
        for particle in self.particles:
            particle.reset()

    def plot(self, yAxis: bool = True, zAxis: bool = False) -> None:
        for particle in self.particles:
            if yAxis:
                plt.plot(particle.qo_arr[:, 0], particle.qo_arr[:, 1], c='red')
            if zAxis:
                plt.plot(particle.qo_arr[:, 0], particle.qo_arr[:, 2], c='blue')
        plt.grid()
        plt.title('ideal orbit displacement. red is y position, blue is z positon. \n total particles: ' +
                  str(len(self.particles)))
        plt.ylabel('displacement from ideal orbit')
        plt.xlabel("distance along orbit,m")
        plt.show()
