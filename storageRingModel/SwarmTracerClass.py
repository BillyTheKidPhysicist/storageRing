import os
from collections.abc import Iterable
from typing import Union

import multiprocess as mp
import numpy as np

from ParticleClass import Swarm, Particle
from ParticleTracerClass import ParticleTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from helperTools import low_discrepancy_sample
from typeHints import RealNum


def lorentz_function(x, gamma):
    # returns a value of 1.0 for x=0
    return (gamma / 2) ** 2 / (x ** 2 + (gamma / 2) ** 2)


def normal(v, sigma, v0=0.0):
    return np.exp(-.5 * ((v - v0) / sigma) ** 2)


TupleOrNum = Union[tuple[float, float], RealNum]
realNumbers = (int, float)


class SwarmTracer:

    def __init__(self, lattice: ParticleTracerLattice):
        self.lattice = lattice
        self.particle_tracer = ParticleTracer(self.lattice)

    def time_step_swarm_distance_along_x(self, swarm, distance: RealNum, hold_position_in_x: bool = False) -> Swarm:
        """Particles are time stepped, forward or backward, to move 'distance' along the x axis"""
        for particle in swarm:
            t = distance / particle.pi[0]
            if hold_position_in_x:
                particle.qi[1:] += t * particle.pi[1:]
            else:
                particle.qi += t * particle.pi
        return swarm

    def initialize_stablity_testing_swarm(self, q_max: RealNum) -> Swarm:
        small_offset = -1e-10  # this prevents setting a particle right at a boundary which is takes time to sort out
        swarm_test = Swarm()
        swarm_test.add_New_Particle(qi=np.asarray([small_offset, 0.0, 0.0]))
        swarm_test.add_New_Particle(qi=np.asarray([small_offset, q_max / 2, q_max / 2]))
        swarm_test.add_New_Particle(qi=np.asarray([small_offset, -q_max / 2, q_max / 2]))
        swarm_test.add_New_Particle(qi=np.asarray([small_offset, q_max / 2, -q_max / 2]))
        swarm_test.add_New_Particle(qi=np.asarray([small_offset, -q_max / 2, -q_max / 2]))
        return swarm_test

    def initialize_hypercube_swarm_in_phase_space(self, q_max: np.ndarray, p_max: np.ndarray, num_grid_edge: int,
                                                  upperSymmetry: bool = False) -> Swarm:
        # create a cloud of particles in phase space at the origin. In the xy plane, the average velocity vector points
        # to the west. The transverse plane is the yz plane.
        # q_max: absolute value maximum position in the transverse direction
        # q_max: absolute value maximum position in the transverse momentum
        # num: number of samples along each axis in phase space. Total is num^4
        # upperSymmetry: if this is true, exploit the symmetry between +/-z and ignore coordinates below z=0
        q_arr = np.linspace(-q_max, q_max, num=num_grid_edge)
        p_arr = np.linspace(-p_max, p_max, num=num_grid_edge)
        args_arr = np.asarray(np.meshgrid(q_arr, q_arr, p_arr, p_arr)).T.reshape(-1, 4)
        swarm = Swarm()
        for arg in args_arr:
            qi = np.asarray([0.0, arg[0], arg[1]])
            pi = np.asarray([-self.lattice.speed_nominal, arg[2], arg[3]])
            if upperSymmetry == True:
                if qi[2] < 0:
                    pass
                else:
                    swarm.add_New_Particle(qi, pi)
            else:
                swarm.add_New_Particle(qi, pi)
        return swarm

    def initialize_Simulated_Collector_Focus_Swarm(self, num_particles: int) -> Swarm:
        """
        Initialize swarm particles with phase space coordinates from a simulation of the focus of the collector.


        :param num_particles: Number of particles to add to swarm from data file
        :return: swarm of particles
        """
        swarm_data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "particleInitialConditions.txt")
        particle_data = np.loadtxt(swarm_data_file)
        assert len(particle_data) >= num_particles and particle_data.shape[1] == 6 and len(particle_data.shape) == 2
        q_arr, p_arr = particle_data[:num_particles, :3], particle_data[:num_particles, 3:]
        swarm = Swarm()
        for qi, pi, in zip(q_arr, p_arr):
            assert np.all(np.abs(qi) < 1) and np.all(np.abs(pi) < 1000)  # avoid possible unit conversion error
            assert -250 < pi[0] < -150
            assert qi[0] <= 0.0
            if qi[0] == 0.0:
                qi[0] -= 1e-10
            swarm.add_New_Particle(qi=qi, pi=pi)
        return swarm

    def initialize_Observed_Collector_Swarm_Probability_Weighted(self, capture_diam: float, collector_output_angle: float,
                                                                 num_particles: float, gammaSpace: float = 3.5e-3,
                                                                 sameSeed: bool = False, upperSymmetry: bool = False,
                                                                 probabilityMin: float = 0.01) -> Swarm:
        # this function generates a swarm that models the observed swarm. This is done by first generating a pseudorandom
        # swarm that is well spread out in space, then weighitng each particle by it's probability according to the
        # observed data. The probability is finally rescaled
        # captureDiam: Diameter of the circle of atoms we wish to collect, meters
        # collectorOutputAngle: Maximum angle of atoms leaving the collector, radians
        # num_particles: Number of particles to sample. Will not always equal exactly this
        # gammaSpace: The FWHM of the lorentz function that models our spatial data, meters
        # temperature: The temperature of the atoms, kelvin. Decides thermal velocity spread

        assert 0.0 < capture_diam <= .1 and 0.0 < collector_output_angle <= .2 and 0.0 < gammaSpace <= .01 \
               and probabilityMin >= 0.0  # reasonable values

        p_trans_max = self.lattice.speed_nominal * np.tan(
            collector_output_angle)  # transverse velocity dominates thermal velocity,
        # ie, geometric heating
        # sigmaVelocity=np.sqrt(BOLTZMANN_CONSTANT*temperature/MASS_LITHIUM_7) #thermal velocity spread. Used for
        # longitudinal velocity only because geometric dominates thermal in transverse dimension
        p_longitudinal_min = -1e-3
        p_longitudinal_max = 1e-3
        p_long_bounds = (p_longitudinal_min, p_longitudinal_max)
        swarm_evenly_spread = self.initalize_PseudoRandom_Swarm_In_Phase_Space(capture_diam / 2.0, p_trans_max, p_long_bounds,
                                                                             num_particles, same_seed=sameSeed,
                                                                             upper_symmetry=upperSymmetry)
        probabilities = []
        for particle in swarm_evenly_spread:
            probability = 1.0
            x, y, z = particle.qi
            r = np.sqrt(y ** 2 + z ** 2)  # remember x is longitudinal
            px, py, pz = particle.pi
            probability = probability * lorentz_function(r, gammaSpace)  # spatial probability
            p_trans = np.sqrt(py ** 2 + pz ** 2)
            px = -np.sqrt(self.lattice.speed_nominal ** 2 - p_trans ** 2)
            particle.pi[0] = px
            assert probability < 1.0
            probabilities.append(probability)
        swarm_observed = Swarm()
        peak_probability = max(probabilities)  # I think this is unnesesary
        for particle, probability in zip(swarm_evenly_spread.particles, probabilities):
            particle.probability = probability / peak_probability
            if particle.probability > probabilityMin:
                swarm_observed.particles.append(particle)
        return swarm_observed

    def _make_PseudoRandom_Swarm_Bounds_List(self, qT_bounds: TupleOrNum, pT_bounds: TupleOrNum, px_bounds: TupleOrNum,
                                             use_z_symmetry: bool = False) -> list:

        if isinstance(qT_bounds, realNumbers):
            assert qT_bounds > 0.0
            y_bounds = (-qT_bounds, qT_bounds)
            z_bounds = y_bounds if use_z_symmetry is False else (0.0, qT_bounds)
            qT_bounds = [y_bounds, z_bounds]
        if isinstance(pT_bounds, realNumbers):
            assert pT_bounds > 0.0
            pT_bounds = [(-pT_bounds, pT_bounds), (-pT_bounds, pT_bounds)]
        if isinstance(px_bounds, realNumbers):
            assert px_bounds > 0.0
            px_bounds = (-px_bounds - self.lattice.speed_nominal, px_bounds - self.lattice.speed_nominal)
        else:
            px_bounds = (px_bounds[0] - self.lattice.speed_nominal, px_bounds[1] - self.lattice.speed_nominal)
        generator_bounds = [*qT_bounds, px_bounds, *pT_bounds]
        px_min, px_max = generator_bounds[2]
        assert len(generator_bounds) == 5 and px_min < -self.lattice.speed_nominal < px_max
        return generator_bounds

    def initalize_PseudoRandom_Swarm_In_Phase_Space(self, qTBounds: TupleOrNum, p_t_bounds: TupleOrNum,
                                                    px_bounds: TupleOrNum, num_particles: int,
                                                    upper_symmetry: bool = False,
                                                    same_seed: bool = False, circular: bool = True,
                                                    small_x_offset: bool = True) -> Swarm:
        # return a swarm object who position and momentum values have been randomly generated inside a phase space hypercube
        # and that is heading in the negative x direction with average velocity lattice.speed_nominal. A seed can be reused to
        # get repeatable random results. a sobol sequence is used that is then jittered. In additon points are added at
        # each corner exactly and midpoints between corners if desired
        # NOTE: it's not guaranteed that there will be exactly num particles.
        if circular:
            assert all((isinstance(_bounds, realNumbers)) and _bounds > 0.0 for _bounds in (qTBounds, p_t_bounds))
            q_trans_max = qTBounds
            p_trans_max = p_t_bounds
        generator_bounds = self._make_PseudoRandom_Swarm_Bounds_List(qTBounds, p_t_bounds, px_bounds,
                                                                    use_z_symmetry=upper_symmetry)

        if circular is True:
            num_particles_frac = 1 / (
                    (np.pi / 4) ** 2)  # the ratio of the are of the circle to the cross section. There is one
            # factor for momentum and one for position
        else:
            num_particles_frac = 1.0
        re_seed_val = np.random.get_state()[1][0]
        if same_seed:
            np.random.seed(42)
        elif type(same_seed) is int:
            np.random.seed(same_seed)
        swarm = Swarm()
        sample_seed = None if not same_seed else same_seed
        samples = low_discrepancy_sample(generator_bounds, round(num_particles * num_particles_frac), seed=sample_seed)
        np.random.shuffle(samples)

        if small_x_offset:
            x0 = -1e-10  # to push negative
        else:
            x0 = 0.0
        samples = np.column_stack((np.ones(len(samples)) * x0, samples))
        particle_count = 0  # track how many particles have been added to swarm
        for xi in samples:
            q = xi[:3]
            p = xi[3:]
            if circular:
                y, z, py, pz = xi[[1, 2, 4, 5]]
                if np.sqrt(y ** 2 + z ** 2) < q_trans_max and np.sqrt(py ** 2 + pz ** 2) < p_trans_max:
                    swarm.add_New_Particle(qi=q, pi=p)
                    particle_count += 1
                if particle_count == num_particles:
                    break
            else:
                swarm.add_New_Particle(qi=q, pi=p)
        if same_seed or isinstance(same_seed, int):
            np.random.seed(re_seed_val)  # re randomize
        return swarm

    def initialize_point_source_swarm(self, source_angle: float, num_particles: int, smallXOffset: bool = True,
                                      sameSeed: bool = False) -> Swarm:
        p0 = self.lattice.speed_nominal  # the momentum of each particle
        q_trans_bounds, px_bounds = 1e-12, 1e-12  # force to a point spatialy, and no speed spread
        p_trans_bounds = np.tan(source_angle) * p0
        swarm_pseudo_random = self.initalize_PseudoRandom_Swarm_In_Phase_Space(q_trans_bounds, p_trans_bounds, px_bounds, num_particles,
                                                                             same_seed=sameSeed, circular=True,
                                                                             small_x_offset=smallXOffset)
        for particle in swarm_pseudo_random:
            px, py, pz = particle.pi
            px = -np.sqrt(p0 ** 2 - (py ** 2 + pz ** 2))
            particle.pi = np.asarray([px, py, pz])
        return swarm_pseudo_random

    def initialize_Ring_Swarm(self, angle, num):
        assert 0.0 < angle < np.pi / 2
        pr = np.tan(angle) * self.lattice.speed_nominal
        theta_arr = np.linspace(0.0, 2 * np.pi, num + 1)[:-1]
        swarm = Swarm()
        for theta in theta_arr:
            swarm.add_New_Particle(pi=np.asarray([-self.lattice.speed_nominal, pr * np.cos(theta), pr * np.sin(theta)]))
        return swarm

    def initalize_PseudoRandom_Swarm_At_Combiner_Output(self, q_t_bounds, p_t_bounds, px_bounds, num_particles,
                                                        use_z_symmetry=False,
                                                        same_seed=False, circular=True, small_x_offset=True):
        swarm_at_origin = self.initalize_PseudoRandom_Swarm_In_Phase_Space(q_t_bounds, p_t_bounds, px_bounds, num_particles,
                                                                         upper_symmetry=use_z_symmetry,
                                                                         same_seed=same_seed, circular=circular,
                                                                         small_x_offset=small_x_offset)
        swarm_at_combiner = self.move_Swarm_To_Combiner_Output(swarm_at_origin, copy_swarm=False, scoot=True)
        return swarm_at_combiner

    def combiner_Output_Offset_Shift(self) -> np.ndarray:
        # combiner may have an output offset (ie hexapole combiner). This return the 3d vector (x,y,0) that connects the
        # geoemtric center of the output plane with the offset point, which also lies in the plane. stern gerlacht
        # style doesn't have and offset
        n2 = self.lattice.combiner.ne.copy()  # unit normal to outlet
        np2 = -np.asarray([n2[1], -n2[0], 0.0])  # unit parallel to outlet
        return np2 * self.lattice.combiner.output_offset

    def move_Swarm_To_Combiner_Output(self, swarm: Swarm, scoot: bool = False, copy_swarm: bool = True) -> Swarm:
        # take a swarm where at move it to the combiner's output. Swarm should be created such that it is centered at
        # (0,0,0) and have average negative velocity.
        # swarm: the swarm to move to output
        # scoot: if True, move the particles along a tiny amount so that they are just barely in the next element. Helpful
        # for the doing the particle tracing sometimes
        if copy_swarm == True:
            swarm = swarm.copy()

        R = self.lattice.combiner.R_In.copy()  # matrix to rotate into combiner frame
        r2 = self.lattice.combiner.r2.copy()  # position of the outlet of the combiner
        r2 += self.combiner_Output_Offset_Shift()

        for particle in swarm.particles:
            particle.qi[:2] = particle.qi[:2] @ R
            particle.qi += r2
            particle.pi[:2] = particle.pi[:2] @ R
            if scoot == True:
                tiny_time_step = 1e-9
                particle.qi += particle.pi * tiny_time_step
        return swarm

    def _super_Fast_Trace(self, swarm: Swarm, trace_Particle_Function) -> Swarm:
        # use trick of accessing only the important class variabels and passing those through to reduce pickle time
        def fastFunc(compactDict):
            particle = Particle()
            for key, val in compactDict.items():
                setattr(particle, key, val)
            particle = trace_Particle_Function(particle)
            compact_dict_traced = {}
            for key, val in vars(particle).items():
                if val is not None:
                    compact_dict_traced[key] = val
            return compact_dict_traced

        compact_dict_list = []
        for particle in swarm:
            compact_dict = {}
            for key, val in vars(particle).items():
                if val is not None:
                    if not (isinstance(val, Iterable) and len(val) == 0):
                        compact_dict[key] = val
            compact_dict_list.append(compact_dict)
        with mp.Pool(mp.cpu_count()) as Pool:
            compact_dict_traced_list = Pool.map(fastFunc, compact_dict_list)
        for particle, compact_dict in zip(swarm.particles, compact_dict_traced_list):
            for key, val in compact_dict.items():
                setattr(particle, key, val)
        return swarm

    def trace_Swarm_Through_Lattice(self, swarm: Swarm, h: float, T: float, parallel: bool = False,
                                    use_fast_mode: bool = True,
                                    copy_swarm: bool = True, accelerated: bool = False, steps_per_logging: int = 1,
                                    use_energy_correction: bool = False, use_collisions: bool = False,
                                    log_phase_space_coords: bool = False) -> Swarm:
        if copy_swarm == True:
            swarm_new = swarm.copy()
        else:
            swarm_new = swarm

        def trace_Particle(particle):
            particle_new = self.particle_tracer.trace(particle, h, T, fast_mode=use_fast_mode, accelerated=accelerated,
                                                    steps_between_logging=steps_per_logging,
                                                    use_energy_correction=use_energy_correction,
                                                    log_phase_space_coords=log_phase_space_coords,
                                                    use_collisions=use_collisions)
            return particle_new

        if parallel == 'superfast':
            # use trick of accessing only the important class variabels and passing those through. about 30%
            # faster
            swarm_new = self._super_Fast_Trace(swarm_new, trace_Particle)
            return swarm_new
        elif parallel == True:
            # more straightforward method. works for serial as well
            with mp.Pool(mp.cpu_count()) as pool:
                swarm_new.particles = pool.map(trace_Particle, swarm_new.particles)
            return swarm_new
        else:
            swarm_new.particles = [trace_Particle(particle) for particle in swarm_new]
            return swarm_new
