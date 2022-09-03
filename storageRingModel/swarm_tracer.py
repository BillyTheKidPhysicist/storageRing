import os
from typing import Union

import numpy as np

from constants import DEFAULT_ATOM_SPEED
from helper_tools import low_discrepancy_sample, parallel_evaluate, temporary_seed, arr_product
from particle import Swarm, Particle
from particle_tracer import ParticleTracer, trace_particle_periodic_linear_lattice
from particle_tracer_lattice import ParticleTracerLattice
from type_hints import RealNum


def lorentz_function(x, gamma):
    # returns a value of 1.0 for x=0
    return (gamma / 2) ** 2 / (x ** 2 + (gamma / 2) ** 2)


TupleOrNum = Union[tuple[float, float], RealNum]
real_number = (int, float)

# todo: I this shouldn't be neccesary, I can find a better solution
TINY_NEG_OFFSET_FROM_ELEMENT_EDGE = -1e-10  # to prevent the particle from starting righ on edge of element, which causes

# unnecesary evaluation time

initial_dict_key_and_index_for_dim = {'x': ('qi', 0), 'y': ('qi', 1), 'z': ('qi', 2),
                                      'px': ('pi', 0), 'py': ('pi', 1), 'pz': ('pi', 2)}


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
        small_offset = TINY_NEG_OFFSET_FROM_ELEMENT_EDGE  # this prevents setting a particle right at a boundary which is takes time to sort out
        swarm_test = Swarm()
        swarm_test.add_new_particle(qi=np.asarray([small_offset, 0.0, 0.0]))
        swarm_test.add_new_particle(qi=np.asarray([small_offset, q_max / 2, q_max / 2]))
        swarm_test.add_new_particle(qi=np.asarray([small_offset, -q_max / 2, q_max / 2]))
        swarm_test.add_new_particle(qi=np.asarray([small_offset, q_max / 2, -q_max / 2]))
        swarm_test.add_new_particle(qi=np.asarray([small_offset, -q_max / 2, -q_max / 2]))
        return swarm_test

    def initialize_hypercube_swarm_in_phase_space(self, q_max: np.ndarray, p_max: np.ndarray, num_grid_edge: int,
                                                  use_z_symmetry: bool = False) -> Swarm:
        # create a cloud of particles in phase space at the origin. In the xy plane, the average velocity vector points
        # to the west. The transverse plane is the yz plane.
        # q_max: absolute value maximum position in the transverse direction
        # q_max: absolute value maximum position in the transverse momentum
        # num: number of samples along each axis in phase space. Total is num^4
        # use_z_symmetry: if this is true, exploit the symmetry between +/-z and ignore coordinates below z=0
        q_arr = np.linspace(-q_max, q_max, num=num_grid_edge)
        p_arr = np.linspace(-p_max, p_max, num=num_grid_edge)
        phase_space_coords = np.asarray(np.meshgrid(q_arr, q_arr, p_arr, p_arr)).T_max.reshape(-1, 4)
        if use_z_symmetry:
            z_index = 1
            phase_space_coords = phase_space_coords[phase_space_coords[:, z_index] >= 0.0]
        swarm = Swarm()
        for [y, z, py, pz] in phase_space_coords:
            qi = np.asarray([0.0, y, z])
            pi = np.asarray([-self.lattice.speed_nominal, py, pz])
            swarm.add_new_particle(qi, pi)
        return swarm

    def initialize_simulated_collector_focus_swarm(self, num_particles: int) -> Swarm:
        """
        Initialize swarm particles with phase space coordinates from a simulation of the focus of the collector.


        :param num_particles: Number of particles to add to swarm from data file
        :return: swarm of particles
        """
        particle_data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "particleInitialConditions.txt")
        particle_data = np.loadtxt(particle_data_file)
        assert len(particle_data) >= num_particles and particle_data.shape[1] == 6 and len(particle_data.shape) == 2
        q_arr, p_arr = particle_data[:num_particles, :3], particle_data[:num_particles, 3:]
        swarm = Swarm()
        min_py, max_py = 150.0, 250.0
        for qi, pi, in zip(q_arr, p_arr):
            assert np.all(np.abs(qi) < 1) and np.all(np.abs(pi) < 1000)  # avoid possible unit conversion error
            assert min_py < abs(pi[0]) < max_py and pi[0] < 0.0 and qi[0] == 0.0
            qi[0] += TINY_NEG_OFFSET_FROM_ELEMENT_EDGE
            swarm.add_new_particle(qi=qi, pi=pi)
        return swarm

    def initialize_observed_collector_swarm_probability_weighted(self, capture_diam: float,
                                                                 collector_output_angle: float,
                                                                 num_particles: int, gamma_space: float = 3.5e-3,
                                                                 same_seed: bool = False, use_z_symmetry: bool = False,
                                                                 probability_min: float = 0.01) -> Swarm:

        assert 0.0 < capture_diam <= .1 and 0.0 < collector_output_angle <= .2 and 0.0 < gamma_space <= .01 \
               and probability_min >= 0.0  # reasonable values

        p_trans_max = self.lattice.speed_nominal * np.tan(
            collector_output_angle)  # transverse velocity dominates thermal velocity,
        # ie, geometric heating
        # sigmaVelocity=np.sqrt(BOLTZMANN_CONSTANT*temperature/MASS_LITHIUM_7) #thermal velocity spread. Used for
        # longitudinal velocity only because geometric dominates thermal in transverse dimension
        p_longitudinal_min = -1e-3
        p_longitudinal_max = 1e-3
        p_long_bounds = (p_longitudinal_min, p_longitudinal_max)
        swarm_evenly_spread = self.initalize_pseudorandom_swarm_in_phase_space(capture_diam / 2.0, p_trans_max,
                                                                               p_long_bounds,
                                                                               num_particles, same_seed=same_seed,
                                                                               use_z_symmetry=use_z_symmetry)
        probabilities = []
        for particle in swarm_evenly_spread:
            probability = 1.0
            x, y, z = particle.qi
            r = np.sqrt(y ** 2 + z ** 2)  # remember x is longitudinal
            px, py, pz = particle.pi
            probability = probability * lorentz_function(r, gamma_space)  # spatial probability
            p_trans = np.sqrt(py ** 2 + pz ** 2)
            px = -np.sqrt(self.lattice.speed_nominal ** 2 - p_trans ** 2)
            particle.pi[0] = px
            assert probability < 1.0
            probabilities.append(probability)
        swarm_observed = Swarm()
        peak_probability = max(probabilities)  # I think this is unnesesary
        for particle, probability in zip(swarm_evenly_spread.particles, probabilities):
            particle.probability = probability / peak_probability
            if particle.probability > probability_min:
                swarm_observed.particles.append(particle)
        return swarm_observed

    def _make_pseudorandom_swarm_bounds(self, qT_bounds: TupleOrNum, pT_bounds: TupleOrNum,
                                        px_bounds: TupleOrNum, use_z_symmetry: bool = False) -> list:

        if isinstance(qT_bounds, real_number):
            y_bounds = (-qT_bounds, qT_bounds)
            z_bounds = y_bounds if use_z_symmetry is False else (0.0, qT_bounds)
            qT_bounds = [y_bounds, z_bounds]
        if isinstance(pT_bounds, real_number):
            pT_bounds = [(-pT_bounds, pT_bounds), (-pT_bounds, pT_bounds)]
        if isinstance(px_bounds, real_number):
            px_bounds = (-px_bounds - self.lattice.speed_nominal, px_bounds - self.lattice.speed_nominal)
        else:
            px_bounds = (px_bounds[0] - self.lattice.speed_nominal, px_bounds[1] - self.lattice.speed_nominal)
        generator_bounds = [*qT_bounds, px_bounds, *pT_bounds]
        px_min, px_max = generator_bounds[2]
        assert len(generator_bounds) == 5 and px_min <= -self.lattice.speed_nominal <= px_max
        return generator_bounds

    def initalize_pseudorandom_swarm_in_phase_space(self, q_trans_bounds: TupleOrNum, p_trans_bounds: TupleOrNum,
                                                    px_bounds: TupleOrNum, num_particles: int,
                                                    use_z_symmetry: bool = False,
                                                    same_seed: bool = False, circular: bool = True,
                                                    small_x_offset: bool = True) -> Swarm:
        if circular:
            for _bounds in (q_trans_bounds, p_trans_bounds):
                assert isinstance(_bounds, real_number) and _bounds > 0.0
            q_trans_max = q_trans_bounds
            p_trans_max = p_trans_bounds
        generator_bounds = self._make_pseudorandom_swarm_bounds(q_trans_bounds, p_trans_bounds, px_bounds,
                                                                use_z_symmetry=use_z_symmetry)

        # The ratio of the are of the circle to the cross section. one factor for momentum and one for position
        num_particles_frac = 1 / ((np.pi / 4) ** 2) if circular else 1.0

        seed = 42 if same_seed else None

        samples = low_discrepancy_sample(generator_bounds, round(num_particles * num_particles_frac), seed=seed)
        with temporary_seed(seed):
            np.random.shuffle(samples)

        x0 = TINY_NEG_OFFSET_FROM_ELEMENT_EDGE if small_x_offset else 0.0
        samples = np.column_stack((np.ones(len(samples)) * x0, samples))
        particle_count = 0  # track how many particles have been added to swarm
        swarm = Swarm()
        for xi in samples:
            q = xi[:3]
            p = xi[3:]
            if circular:
                y, z, py, pz = xi[[1, 2, 4, 5]]
                if np.sqrt(y ** 2 + z ** 2) < q_trans_max and np.sqrt(py ** 2 + pz ** 2) < p_trans_max:
                    swarm.add_new_particle(qi=q, pi=p)
                    particle_count += 1
                if particle_count == num_particles:
                    break
            else:
                swarm.add_new_particle(qi=q, pi=p)
        return swarm

    def initialize_point_source_swarm(self, source_angle: float, num_particles: int, small_x_offset: bool = True,
                                      same_seed: bool = False) -> Swarm:
        p0 = self.lattice.speed_nominal  # the momentum of each particle
        q_trans_bounds, px_bounds = 1e-12, 1e-12  # force to a point spatialy, and no speed spread
        p_trans_bounds = np.tan(source_angle) * p0
        swarm_pseudo_random = self.initalize_pseudorandom_swarm_in_phase_space(q_trans_bounds, p_trans_bounds,
                                                                               px_bounds, num_particles,
                                                                               same_seed=same_seed, circular=True,
                                                                               small_x_offset=small_x_offset)
        for particle in swarm_pseudo_random:
            px, py, pz = particle.pi
            px = -np.sqrt(p0 ** 2 - (py ** 2 + pz ** 2))
            particle.pi = np.asarray([px, py, pz])
        return swarm_pseudo_random

    def initalize_pseudorandom_swarm_at_combiner_output(self, q_t_bounds, p_trans_bounds, px_bounds, num_particles,
                                                        use_z_symmetry=False,
                                                        same_seed=False, circular=True, small_x_offset=True):
        swarm_at_origin = self.initalize_pseudorandom_swarm_in_phase_space(q_t_bounds, p_trans_bounds, px_bounds,
                                                                           num_particles,
                                                                           use_z_symmetry=use_z_symmetry,
                                                                           same_seed=same_seed, circular=circular,
                                                                           small_x_offset=small_x_offset)
        swarm_at_combiner = self.move_swarm_to_combiner_output(swarm_at_origin, copy_swarm=False, scoot=True)
        return swarm_at_combiner

    def initialize_pseudorandom_1_dim_swarm(self, pos_max: RealNum, p_max: RealNum, num_particles: int,
                                            seed: int = None, px_spread: RealNum = 0.0, which_dim='y') -> Swarm:
        """Build a swarm only along the y dimension ([x,y,z]). Useful for tracing particle through matrix model
        lattice"""
        assert which_dim in ('y', 'z')
        swarm = Swarm()
        px0 = -self.lattice.speed_nominal
        bounds = [(-pos_max, pos_max), (-p_max, p_max), (px0 - px_spread, px0 + px_spread)]
        samples = low_discrepancy_sample(bounds, num_particles, seed=seed)
        for [pos, p, px] in samples:
            qi = np.array([-1e-10, 0, 0])
            pi = np.array([px, 0.0, 0.0])
            index = 1 if which_dim == 'y' else 2
            qi[index] = pos
            pi[index] = p
            swarm.add_new_particle(qi=qi, pi=pi)
        return swarm

    def initialize_grid_2_dim_swarm(self, dim1_max: RealNum, dim2_max: RealNum, num_points_per_dim: int,
                                    dim1_name: str, dim2_name: str, px=DEFAULT_ATOM_SPEED) -> Swarm:
        """Initialize a swarm in 2 dimensions along specified dimensions"""
        key1, idx1 = initial_dict_key_and_index_for_dim[dim1_name]
        key2, idx2 = initial_dict_key_and_index_for_dim[dim2_name]
        vals_dim1 = np.linspace(-dim1_max, dim1_max, num_points_per_dim)
        vals_dim2 = np.linspace(-dim2_max, dim2_max, num_points_per_dim)
        coords_vals = arr_product(vals_dim1, vals_dim2)
        swarm = Swarm()
        for coords in coords_vals:
            val1, val2 = coords
            particle = Particle(qi=np.array([-1e-10, 0, 0]), pi=np.array([-px, 0, 0]))
            particle.__dict__[key1][idx1] = val1
            particle.__dict__[key2][idx2] = val2
            swarm.add(particle)
        return swarm

    def combiner_output_offset_shift(self) -> np.ndarray:
        # combiner may have an output offset (ie hexapole combiner). This return the 3d vector (x,y,0) that connects the
        # geoemtric center of the output plane with the offset point, which also lies in the plane. stern gerlacht
        # style doesn't have and offset
        n2 = self.lattice.combiner.ne.copy()  # unit normal to outlet
        np2 = -np.asarray([n2[1], -n2[0], 0.0])  # unit parallel to outlet
        return np2 * self.lattice.combiner.output_offset

    def move_swarm_to_combiner_output(self, swarm: Swarm, scoot: bool = False, copy_swarm: bool = True) -> Swarm:
        # take a swarm where at move it to the combiner's output. Swarm should be created such that it is centered at
        # (0,0,0) and have average negative velocity.
        # swarm: the swarm to move to output
        # scoot: if True, move the particles along a tiny amount so that they are just barely in the next element. Helpful
        # for the doing the particle tracing sometimes
        if copy_swarm:
            swarm = swarm.copy()

        R = self.lattice.combiner.R_In.copy()  # matrix to rotate into combiner frame
        r2 = self.lattice.combiner.r2.copy()  # position of the outlet of the combiner
        r2 += self.combiner_output_offset_shift()

        for particle in swarm.particles:
            particle.qi[:2] = particle.qi[:2] @ R
            particle.qi += r2
            particle.pi[:2] = particle.pi[:2] @ R
            if scoot:
                tiny_time_step = 1e-9
                particle.qi += particle.pi * tiny_time_step
        return swarm

    def trace_swarm_through_lattice(self, swarm: Swarm, h: float, T_max: float, parallel: bool = False,
                                    use_fast_mode: bool = True,
                                    copy_swarm: bool = True, accelerated: bool = False, steps_per_logging: int = 1,
                                    use_energy_correction: bool = False, use_collisions: bool = False,
                                    log_el_phase_space_coords: bool = False) -> Swarm:

        def trace_particle(particle):
            particle_new = self.particle_tracer.trace(particle, h, T_max, fast_mode=use_fast_mode,
                                                      accelerated=accelerated,
                                                      steps_between_logging=steps_per_logging,
                                                      use_energy_correction=use_energy_correction,
                                                      log_el_phase_space_coords=log_el_phase_space_coords,
                                                      use_collisions=use_collisions)
            return particle_new

        swarm_traced = swarm.copy() if copy_swarm else swarm
        swarm_traced.particles = parallel_evaluate(trace_particle, swarm_traced.particles, parallel=parallel)
        return swarm_traced


def trace_swarm_periodic_linear_lattice(swarm_initial: Swarm, st: SwarmTracer, h: float,
                                        T: float, parallel: bool = False) -> Swarm:
    """Trace a swarm through a linear lattice assuming that the lattice is periodic"""
    pt = st.particle_tracer
    swarm_traced = Swarm()
    wrap = lambda particle: trace_particle_periodic_linear_lattice(particle, pt, h, T)
    swarm_traced.particles = parallel_evaluate(wrap, swarm_initial.particles, parallel=parallel)
    return swarm_traced
