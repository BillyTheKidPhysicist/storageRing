import copy
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon

from Particle_tracer_lattice import ParticleTracerLattice
from floor_plan_checker import does_fit_in_room, plot_floor_plan_in_lab
from helper_tools import full_arctan2
from kevin_bumper import swarmShift_x
from lattice_elements.elements import Element
from lattice_elements.elements import HalbachLensSim, Drift, CombinerLensSim, CombinerSim, CombinerIdeal
from lattice_elements.orbit_trajectories import make_orbit_shape
from lattice_models.lattice_model_parameters import INJECTOR_TUNABILITY_LENGTH
from lattice_models.system_model import get_optimal_ring_params, get_optimal_injector_params, make_system_model
from particle_class import Swarm, Particle
from swarm_tracer import SwarmTracer

combiners = (CombinerLensSim, CombinerSim, CombinerIdeal)
Shape = Union[LineString, Polygon]

# expected elements of injector.
# todo: This logic here could be changed. These expected elements shouldn't be hard coded here
ELEMENTS_BUMPER = (HalbachLensSim, Drift, HalbachLensSim, Drift)
ELEMENTS_MODE_MATCHER = (Drift, HalbachLensSim, Drift, Drift, HalbachLensSim, Drift, CombinerLensSim)
DEFAULT_SIMULATION_TIME = 30.0


def injector_is_expected_design(lattice_injector: ParticleTracerLattice, has_bumper: bool):
    """Check that the injector lattice has the expected ordering of elements. Some logic may break if the injector
    is not laid out as expected. """
    expected_elements = (*ELEMENTS_BUMPER, *ELEMENTS_MODE_MATCHER) if has_bumper else ELEMENTS_MODE_MATCHER
    for el, expected_type in zip(lattice_injector.el_list, expected_elements):
        if not type(el) is expected_type:
            return False
    return True


def make_shapes(lattice: ParticleTracerLattice) -> tuple[list[Shape], list[Shape], list[Shape]]:
    shapes_outer = [el.SO_outer for el in lattice]
    shapes_inner = [el.SO for el in lattice]
    shapes_trajectories = [make_orbit_shape(el) for el in lattice]
    return shapes_inner, shapes_outer, shapes_trajectories


class StorageRingModel:
    max_cost = 2.0
    max_swarm_cost = 1.0
    max_floor_plan_cost = 1.0

    def __init__(self, lattice_ring: ParticleTracerLattice, lattice_injector: ParticleTracerLattice,
                 num_particles: int = 1024, use_collisions: bool = False, use_energy_correction: bool = False,
                 use_bumper: bool = False, sim_time_max=DEFAULT_SIMULATION_TIME):
        assert lattice_ring.lattice_type == 'storage_ring' and lattice_injector.lattice_type == 'injector'
        assert injector_is_expected_design(lattice_injector, use_bumper)
        self.lattice_ring = lattice_ring
        self.lattice_injector = lattice_injector
        self.injector_lens_indices = [i for i, el in enumerate(self.lattice_injector) if type(el) is HalbachLensSim]
        self.swarm_tracer_injector = SwarmTracer(self.lattice_injector)
        self.h = 7.5e-6  # timestep size
        self.T = sim_time_max
        self.swarm_tracer_ring = SwarmTracer(self.lattice_ring)
        self.has_bumper = use_bumper
        self.use_collisions = use_collisions
        self.use_energy_correction = use_energy_correction
        self.swarm_injector_initial = self.generate_swarm(num_particles)

    def generate_swarm(self, num_particlesSwarm: int) -> Swarm:
        """Generate injector swarm. optionally shift the particles in the swarm for the bumper"""
        swarm = self.swarm_tracer_injector.initialize_simulated_collector_focus_swarm(num_particlesSwarm)
        if self.has_bumper:
            swarm = self.swarm_tracer_injector.time_step_swarm_distance_along_x(swarm, swarmShift_x,
                                                                                hold_position_in_x=True)
        return swarm

    def convert_position_injector_to_ring_frame(self, q_lab_inject: np.ndarray) -> np.ndarray:
        """Convert 3D position in injector lab frame into ring lab frame."""
        # a nice trick
        q_lab_ring = self.lattice_injector.combiner.transform_lab_coords_into_element_frame(q_lab_inject)
        q_lab_ring = self.lattice_ring.combiner.transform_element_coords_into_lab_frame(q_lab_ring)
        return q_lab_ring

    def convert_momentum_injector_to_ring_frame(self, p_lab_inject: np.ndarray) -> np.ndarray:
        """Convert 3D particle momentum in injector lab frame into ring lab frame"""
        p_lab_ring = self.lattice_injector.combiner.transform_lab_frame_vector_into_element_frame(p_lab_inject)
        p_lab_ring = self.lattice_ring.combiner.transform_element_frame_vector_into_lab_frame(p_lab_ring)
        return p_lab_ring

    def line_In_ring_frame_from_injector_particle(self, particle: Particle) -> Optional[LineString]:
        """
        Make a shapely line object from an injector particle. If the injector particle was clipped right away
        (starting outside the vacuum for example), None is returned"""
        assert particle.traced
        if len(particle.el_phase_space_log) <= 1:
            return None
        else:
            q_list = []
            for q, _ in particle.el_phase_space_log:
                q_ring_frame_xy = self.convert_position_injector_to_ring_frame(q)[:2]
                q_list.append(q_ring_frame_xy)
            line = LineString(q_list)
            return line

    def does_ring_clip_injector_particle(self, particle: Particle) -> bool:
        """
        Test if particle clipped the ring. Only certain elements are considered, as of writing this is any lens before
        the combiner
        """
        line = self.line_In_ring_frame_from_injector_particle(particle)
        if line is None:  # particle was clipped immediately, but in the injector not in the ring
            return False
        else:
            clippable_elements = self.clippable_elements_in_ring()
            return any(line.intersects(el.SO_outer) for el in clippable_elements)

    def lenses_before_ring_combiner(self) -> tuple[HalbachLensSim, ...]:
        """Get the lens before the combiner but after the bend in the ring. There should be only one lens"""
        lenses = []
        for i, el in enumerate(self.lattice_ring.el_list):
            if type(el) is HalbachLensSim:
                lenses.append(el)
            if i == self.lattice_ring.combiner.index:
                break
        assert len(lenses) > 0
        return tuple(lenses)

    def injector_shapes_in_lab_frame(self) -> tuple[list[Shape], list[Shape], list[Shape]]:
        shapes_inner, shapes_outer, shapes_trajectories = make_shapes(self.lattice_injector)
        shapes_inner = self.move_injector_shapes_to_lab_frame(shapes_inner)
        shapes_outer = self.move_injector_shapes_to_lab_frame(shapes_outer)
        shapes_trajectories = self.move_injector_shapes_to_lab_frame(shapes_trajectories)
        return shapes_inner, shapes_outer, shapes_trajectories

    def move_injector_shapes_to_lab_frame(self, shapes: list[Shape]) -> list[Shape]:
        ne_Inj, ne_Ring = self.lattice_injector.combiner.ne, self.lattice_ring.combiner.ne
        angle_injector = full_arctan2(ne_Inj[1], ne_Inj[0])
        angle_ring = full_arctan2(ne_Ring[1], ne_Ring[0])
        rotation_angle = angle_ring - angle_injector
        r2_injector = self.lattice_injector.combiner.r2
        r2_ring = self.lattice_ring.combiner.r2
        shapes_lab_frame = []
        for shape in shapes:
            shape = copy.copy(shape)
            shape = translate(shape, xoff=-r2_injector[0], yoff=-r2_injector[1])
            shape = rotate(shape, rotation_angle, use_radians=True, origin=(0, 0))
            shape = translate(shape, xoff=r2_ring[0], yoff=r2_ring[1])
            shapes_lab_frame.append(shape)
        return shapes_lab_frame

    def non_drift_elements_in_ring(self) -> list[Element]:
        return [el for el in self.lattice_ring if type(el) is not Drift]

    def clippable_elements_in_ring(self):
        return [el for el in self.lattice_ring if not (type(el) is Drift or isinstance(el, combiners))]

    def floor_plan_overlap_mm(self) -> float:
        """Find the area overlap between the elements before and including the last injector lens, and the lenses
        between combiner input and adjacent bender output. Overlap of the drift region after the last injector lens is
        handled later by clipping particles on it
        """
        overlap_elements = self.non_drift_elements_in_ring()
        _, injector_shapes_outer, _ = self.injector_shapes_in_lab_frame()
        injector_shapes_to_compare = injector_shapes_outer[:self.injector_lens_indices[-1] + 1]
        m_to_mm_area = 1e3 ** 2
        area_mm = 0.0
        for el in overlap_elements:  # count up all the area overlap
            for shape in injector_shapes_to_compare:  # don't forget to add 1
                area_mm += el.SO_outer.intersection(shape).area * m_to_mm_area
        return area_mm

    def floor_plan_shapes(self) -> tuple[list[Shape], list[Shape], list[Shape]]:
        shapes_inner_ring, shapes_outer_ring, shapes_trajectories_ring = make_shapes(self.lattice_ring)
        shapes_inner_inj, shapes_outer_inj, shapes_trajectories_inj = self.injector_shapes_in_lab_frame()
        shapes_inner = [*shapes_inner_ring, *shapes_inner_inj]
        shapes_outer = [*shapes_outer_ring, *shapes_outer_inj]
        shapes_trajectories = [*shapes_trajectories_ring, *shapes_trajectories_inj]
        return shapes_inner, shapes_outer, shapes_trajectories

    def show_floor_plan(self, defer_show=False, true_aspect_ratio=True, save_fig=None, dpi=300,
                        fig_size=None) -> None:  # todo: change this dumb name

        shapes_system = (make_shapes(self.lattice_ring), self.injector_shapes_in_lab_frame())
        if fig_size is not None:
            plt.figure(figsize=fig_size)
        for [shapes_inner, shapes_outer, shapes_trajectories] in shapes_system:
            for shape in shapes_inner:
                plt.plot(*shape.exterior.xy, c='black', linestyle=':')
            for shape in shapes_outer:
                plt.plot(*shape.exterior.xy, c='black')
            for shape in shapes_trajectories:
                plt.plot(*shape.xy, c='red', linestyle=':')

        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        if true_aspect_ratio:
            plt.gca().set_aspect('equal')
        if save_fig is not None:
            plt.savefig(save_fig, dpi=dpi)
        if not defer_show:
            plt.show()

    def show_system_floor_plan_in_room(self) -> None:
        plot_floor_plan_in_lab(self)

    def show_floor_plan_with_trajectories(self, true_aspect_ratio: bool = True, T_max=1.0, save_fig=None,
                                          dpi=300, fig_size=None, show_trace_lines=True, num_particles=100,
                                          parallel=True) -> None:
        """Trace particles through the lattices, and plot the results. Interior and exterior of element is shown"""
        self.show_floor_plan(defer_show=True, true_aspect_ratio=true_aspect_ratio, fig_size=fig_size)
        swarm = Swarm()
        swarm.particles = self.swarm_injector_initial.particles[:num_particles]
        swarm_injector_traced = self.swarm_tracer_injector.trace_swarm_through_lattice(
            swarm, self.h, 1.0, parallel=False,
            use_fast_mode=False, copy_swarm=True, accelerated=False, log_el_phase_space_coords=True,
            use_energy_correction=True,
            use_collisions=self.use_collisions)
        for particle in swarm_injector_traced:
            particle.clipped = True if self.does_ring_clip_injector_particle(particle) else particle.clipped
        swarm_ring_initial = self.transform_swarm_from_injector_to_ring_frame(swarm_injector_traced,
                                                                              copy_particles=True)
        swarm_ring_traced = self.swarm_tracer_ring.trace_swarm_through_lattice(swarm_ring_initial, self.h, T_max,
                                                                               use_fast_mode=False,
                                                                               parallel=parallel,
                                                                               use_energy_correction=True,
                                                                               steps_per_logging=4,
                                                                               use_collisions=self.use_collisions)

        for particle_injector, particle_ring in zip(swarm_injector_traced, swarm_ring_traced):
            assert not (particle_injector.clipped and not particle_ring.clipped)  # this wouldn't make sense
            color = 'r' if particle_ring.clipped else 'g'
            q_arr_injector = particle_injector.q_vals if len(particle_injector.q_vals) != 0 else \
                np.array([particle_injector.qi])
            q_arr_ring = np.array([self.convert_position_injector_to_ring_frame(q) for q in q_arr_injector])
            if show_trace_lines:
                plt.plot(q_arr_ring[:, 0], q_arr_ring[:, 1], c=color, alpha=.3)
            if particle_injector.clipped:  # if clipped in injector, plot last location
                plt.scatter(q_arr_ring[-1, 0], q_arr_ring[-1, 1], marker='x', zorder=100, c=color)
            if particle_ring.q_vals is not None and len(particle_ring.q_vals) > 1:  # if made to ring
                if show_trace_lines:
                    plt.plot(particle_ring.q_vals[:, 0], particle_ring.q_vals[:, 1], c=color, alpha=.3)
                if not particle_injector.clipped:  # if not clipped in injector plot last ring location
                    plt.scatter(particle_ring.q_vals[-1, 0], particle_ring.q_vals[-1, 1], marker='x', zorder=100,
                                c=color)
        if save_fig is not None:
            plt.savefig(save_fig, dpi=dpi)
        plt.show()

    def mode_match(self, floor_plan_cost_cutoff: float = np.inf, parallel: bool = False) -> tuple[float, float]:
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        assert floor_plan_cost_cutoff >= 0
        floor_plan_cost = self.floor_plan_cost_with_tunability()
        if self.floor_plan_cost() > floor_plan_cost_cutoff:
            cost = self.max_swarm_cost + floor_plan_cost
            flux_mult = np.nan
        else:
            swarm_traced = self.inject_swarm(parallel)
            flux_mult = swarm_traced.weighted_flux_mult()
            swarm_cost = self.swarm_cost(swarm_traced)
            cost = swarm_cost + floor_plan_cost
        assert 0.0 <= cost <= self.max_cost
        return cost, flux_mult

    def inject_swarm(self, parallel: bool = False) -> Swarm:
        swarm_initial = self.trace_through_injector_and_transform_to_ring()
        swarm_traced = self.swarm_tracer_ring.trace_swarm_through_lattice(swarm_initial, self.h, self.T,
                                                                          use_fast_mode=True, accelerated=True,
                                                                          copy_swarm=False,
                                                                          use_energy_correction=self.use_energy_correction,
                                                                          use_collisions=self.use_collisions,
                                                                          parallel=parallel)
        return swarm_traced

    def transform_swarm_from_injector_to_ring_frame(self, swarm_injector_traced: Swarm,
                                                    copy_particles: bool = False) -> Swarm:
        swarm_ring_frame = Swarm()
        for particle in swarm_injector_traced:
            clipped = particle.clipped or self.does_ring_clip_injector_particle(particle)
            qRing = self.convert_position_injector_to_ring_frame(particle.qf)
            pRing = self.convert_momentum_injector_to_ring_frame(particle.pf)
            particle_ring = particle.copy() if copy_particles else particle
            particle_ring.qi, particle_ring.pi = qRing, pRing
            particle_ring.reset()
            particle_ring.clipped = clipped
            swarm_ring_frame.add(particle_ring)
        return swarm_ring_frame

    def trace_through_injector_and_transform_to_ring(self) -> Swarm:
        swarm_injector_traced = self.swarm_tracer_injector.trace_swarm_through_lattice(
            self.swarm_injector_initial.copy(), self.h, 1.0, use_fast_mode=True, copy_swarm=False,
            log_el_phase_space_coords=True, accelerated=True, use_collisions=self.use_collisions)
        swarm_ring_initial = self.transform_swarm_from_injector_to_ring_frame(swarm_injector_traced,
                                                                              copy_particles=True)
        return swarm_ring_initial

    def swarm_flux_mult_percent_of_max(self, swarm_traced: Swarm) -> float:
        # What percent of the maximum flux multiplication is the swarm reaching? It's cruical I consider that not
        # all particles survived through the lattice.
        max_flux_mult = self.T * self.lattice_ring.speed_nominal / self.lattice_ring.total_length
        flux_mult_perc = 1e2 * swarm_traced.weighted_flux_mult() / max_flux_mult
        assert 0.0 <= flux_mult_perc <= 100.0
        return flux_mult_perc

    def floor_plan_cost(self) -> float:
        overlap = self.floor_plan_overlap_mm()  # units of mm^2
        factor = 100  # units of mm^2
        cost_overlap = 2 / (1 + np.exp(-overlap / factor)) - 1
        cost = self.max_floor_plan_cost if not does_fit_in_room(self) else cost_overlap
        assert 0.0 <= cost <= self.max_floor_plan_cost
        return cost

    def get_drift_after_last_lens_injector(self) -> Drift:
        """Get drift element which comes immediately after last lens in injector"""
        drift = self.lattice_injector.el_list[self.injector_lens_indices[-1] + 1]
        assert type(drift) is Drift
        return drift

    def floor_plan_cost_with_tunability(self) -> float:
        """Measure floor plan cost at nominal position, and at maximum spatial tuning displacement in each direction.
        Return the largest value of the three. This is used to punish the system when the injector lens is no longer
        tunable because it is so close to the ring"""
        drift_after_lens = self.get_drift_after_last_lens_injector()
        L0 = drift_after_lens.L  # value before tuning
        cost = [self.floor_plan_cost()]

        drift_after_lens.set_length(L0 + -INJECTOR_TUNABILITY_LENGTH)  # move lens away from combiner
        self.lattice_injector.build_lattice(False, False)  # don't waste time building field helpers
        cost.append(self.floor_plan_cost())

        drift_after_lens.set_length(L0)  # reset
        self.lattice_injector.build_lattice(False, False)  # don't waste time building field helpers,
        # previous helpers are still saved
        floor_plan_cost = max(cost)
        assert 0.0 <= floor_plan_cost <= 1.0
        return floor_plan_cost

    def swarm_cost(self, swarm: Swarm) -> float:
        """Cost associated with a swarm after being traced through system"""
        flux_mult_perc = self.swarm_flux_mult_percent_of_max(swarm)
        swarm_cost = (100.0 - flux_mult_perc) / 100.0
        assert 0.0 <= swarm_cost <= self.max_swarm_cost
        return swarm_cost


def build_storage_ring_model(ring_params, injector_params, ring_version, num_particles: int = 1024,
                             use_collisions: bool = False, include_mag_cross_talk: bool = False,
                             use_energy_correction: bool = False, use_mag_errors: bool = False,
                             use_solenoid_field: bool = True, use_bumper: bool = False,
                             include_misalignments: bool = False, sim_time_max=DEFAULT_SIMULATION_TIME):
    """Convenience function for building a StorageRingModel"""
    options = {'use_mag_errors': use_mag_errors, 'use_solenoid_field': use_solenoid_field, 'has_bumper': use_bumper,
               'include_mag_cross_talk_in_ring': include_mag_cross_talk, 'include_misalignments': include_misalignments}
    lattice_ring, lattice_injector = make_system_model(ring_params, injector_params, ring_version, options)
    model = StorageRingModel(lattice_ring, lattice_injector, use_energy_correction=use_energy_correction,
                             num_particles=num_particles, use_collisions=use_collisions,
                             use_bumper=use_bumper)
    return model


def make_optimal_solution_model(ring_version, use_bumper: bool = True,
                                use_solenoid_field: bool = True, use_mag_errors=False,
                                use_energy_correction: bool = False,
                                include_mag_cross_talk: bool = False,
                                include_misalignments: bool = False,
                                sim_time_max=DEFAULT_SIMULATION_TIME) -> StorageRingModel:
    """Convenience function for building the current optimal model"""
    ring_params_optimal = get_optimal_ring_params(ring_version)
    injector_params_optimal = get_optimal_injector_params()
    model = build_storage_ring_model(ring_params_optimal, injector_params_optimal, ring_version, use_bumper=use_bumper,
                                     use_solenoid_field=use_solenoid_field, use_mag_errors=use_mag_errors,
                                     use_energy_correction=use_energy_correction,
                                     include_mag_cross_talk=include_mag_cross_talk,
                                     include_misalignments=include_misalignments,
                                     sim_time_max=sim_time_max)
    return model
