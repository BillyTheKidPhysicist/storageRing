from typing import Callable, Union, Optional

import numpy as np

from async_de import solve_async
from helper_tools import shrink_bounds_around_vals
from lattice_elements.utilities import CombinerDimensionError
from lattice_elements.utilities import ElementTooShortError as ElementTooShortErrorFields
from lattice_models.system_model import get_optimal_ring_params, get_optimal_injector_params
from lattice_models.system_model import make_system_model, get_ring_bounds, get_injector_bounds, \
    make_surrogate_ring_for_injector, make_injector_lattice
from lattice_models.utilities import RingGeometryError, InjectorGeometryError, assert_combiners_are_same, LockedDict
from octopus_optimizer import octopus_optimize
from particle_tracer import ElementTooShortError as ElementTooShortErrorTimeStep
from particle_tracer_lattice import ParticleTracerLattice
from simple_line_search import line_search
from storage_ring_modeler import StorageRingModel, DEFAULT_SIMULATION_TIME
from type_hints import sequence


class Solution:
    """class to hold onto results of each solution"""

    def __init__(self, params, cost, flux_mult=None, survival=None):
        self.params, self.flux_mult, self.cost, self.survival = params, flux_mult, cost, survival

    def __str__(self) -> str:  # method that gets called when you do print(Solution())
        np.set_printoptions(precision=100)
        string = '----------Solution-----------   \n'
        string += 'parameters: ' + repr(self.params) + '\n'
        string += 'cost: ' + str(self.cost) + '\n'
        if self.flux_mult is not None:
            string += 'flux multiplication: ' + str(self.flux_mult) + '\n'
        elif self.survival is not None:
            string += 'injection survival: ' + str(self.survival) + '\n'
        string += '----------------------------'
        return string


def invalid_Solution(params):
    sol = Solution(params, StorageRingModel.max_cost)
    return sol


def injected_swarm_cost(model) -> float:
    swarm_ring_traced = model.inject_swarm()
    num_particles_initial = model.swarm_injector_initial.num_particles(weighted=True)
    num_particles_final = swarm_ring_traced.num_particles(weighted=True, unclipped_only=True)
    swarm_cost = (num_particles_initial - num_particles_final) / num_particles_initial
    assert 0.0 <= swarm_cost <= 1.0
    return swarm_cost


def build_injector_and_surrrogate(injector_params, ring_version, options: dict) -> tuple[
    ParticleTracerLattice, ParticleTracerLattice]:
    assert len(injector_params) == len(get_injector_bounds())

    lattice_injector = make_injector_lattice(injector_params, options)
    lattice_surrogate = make_surrogate_ring_for_injector(injector_params, ring_version, options)
    assert_combiners_are_same(lattice_injector, lattice_surrogate)
    return lattice_surrogate, lattice_injector


def build_lattice_params_dict(params, which: str, ring_version: str) -> dict:
    assert not isinstance(params, dict) and which in ('ring', 'injector')
    keys = get_ring_bounds(ring_version).keys() if which == 'ring' else get_injector_bounds().keys()
    assert len(keys) == len(params)
    return dict(zip(keys, params))


class Solver:
    def __init__(self, which_system, ring_version: str, ring_params=None, injector_params=None,
                 use_solenoid_field=False,
                 use_collisions=False,
                 use_energy_correction=False, num_particles=1024, use_bumper=False, use_standard_tube_OD=False,
                 use_standard_mag_size=False, sim_time_max=DEFAULT_SIMULATION_TIME, include_mag_cross_talk=False):
        assert which_system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
        self.which_system = which_system
        self.ring_version = ring_version
        self.ring_params = ring_params
        self.injector_params = injector_params
        self.use_collisions = use_collisions
        self.use_energy_correction = use_energy_correction
        self.num_particles = num_particles
        self.sim_time_max = sim_time_max
        self.storage_ring_system_options = {
            'use_solenoid_field': use_solenoid_field,
            'has_bumper': use_bumper,
            'use_standard_tube_OD': use_standard_tube_OD,
            'use_standard_mag_size': use_standard_mag_size,
            'include_mag_cross_talk_in_ring': include_mag_cross_talk,
            'build_field_helpers': False  # When the floor plan is violated, fast field helpers are not used, so don't
            # waste time computing them unless needed to trace swarm
        }

    def unpack_params(self, params):
        ring_params, injector_params = None, None
        if self.which_system == 'ring':
            ring_params = params
            injector_params = self.injector_params
        elif self.which_system == 'injector_Surrogate_Ring':
            injector_params = params
        elif self.which_system == 'injector_Actual_Ring':
            ring_params = self.ring_params
            injector_params = params
        else:
            ring_params = params[:len(get_ring_bounds(self.ring_version))]
            injector_params = params[-len(get_injector_bounds()):]
            assert len(ring_params) + len(injector_params) == len(params)
        ring_params = None if ring_params is None else build_lattice_params_dict(ring_params, 'ring', self.ring_version)
        injector_params = None if injector_params is None else build_lattice_params_dict(injector_params, 'injector',
                                                                                         self.ring_version)
        if ring_params is not None and injector_params is not None:
            ring_params['Lm_combiner'] = injector_params['Lm_combiner']
            ring_params['load_beam_offset'] = injector_params['load_beam_offset']
        return ring_params, injector_params

    def build_lattices(self, params):
        ring_params, injector_params = self.unpack_params(params)
        if self.which_system == 'injector_Surrogate_Ring':
            lattice_ring, lattice_injector = build_injector_and_surrrogate(injector_params, self.ring_version,
                                                                           self.storage_ring_system_options)
        else:
            lattice_ring, lattice_injector = make_system_model(ring_params, injector_params, self.ring_version,
                                                               self.storage_ring_system_options)
        return lattice_ring, lattice_injector

    def make_system_model(self, params) -> StorageRingModel:
        lattice_ring, lattice_injector = self.build_lattices(params)
        model = StorageRingModel(lattice_ring, lattice_injector, use_collisions=self.use_collisions,
                                 num_particles=self.num_particles, use_energy_correction=self.use_energy_correction,
                                 use_bumper=self.storage_ring_system_options['has_bumper'],
                                 sim_time_max=self.sim_time_max)
        return model

    def _solve(self, params: tuple[float, ...]) -> Solution:
        model = self.make_system_model(params)
        floor_plan_cost_cutoff = .1
        if self.which_system == 'injector_Surrogate_Ring':
            floor_plan_cost = model.floor_plan_cost_with_tunability()
            if floor_plan_cost > floor_plan_cost_cutoff:
                cost = model.max_swarm_cost + floor_plan_cost
                survival = None
            else:
                swarm_cost = injected_swarm_cost(model)
                survival = 1e2 * (1.0 - swarm_cost)
                cost = swarm_cost + floor_plan_cost
            sol = Solution(params, cost, survival=survival)
        else:
            cost, flux_mult = model.mode_match(floor_plan_cost_cutoff=floor_plan_cost_cutoff)
            sol = Solution(params, cost, flux_mult=flux_mult)
        return sol

    def solve(self, params: tuple[float, ...]) -> Solution:
        """Solve storage ring system. If using 'both' for system, params must be ring then injector params strung
        together"""
        try:
            sol = self._solve(params)
        except (RingGeometryError, InjectorGeometryError, ElementTooShortErrorFields,
                CombinerDimensionError, ElementTooShortErrorTimeStep):
            sol = invalid_Solution(params)
        except:
            print('exception with:', repr(params))
            raise Exception("unhandled exception on paramsForBuilding: ")
        return sol


def make_bounds(which_bounds, ring_version, keys_to_not_change=None, range_factor=1.0) -> tuple:
    """Take bounds for ring and injector and combine into new bounds list. Order is ring bounds then injector bounds.
    Optionally expand the range of bounds by 10%, but not those specified to ignore. If none specified, use a
    default list of values to ignore"""
    assert which_bounds in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    ring_param_bounds = get_ring_bounds(ring_version)
    injector_param_bounds = get_injector_bounds()
    bounds_ring = np.array(list(ring_param_bounds.values()))
    bounds_injector = np.array(list(injector_param_bounds.values()))
    keys_ring = list(ring_param_bounds.keys())
    keys_injector = list(injector_param_bounds.keys())
    if which_bounds == 'ring':
        bounds, keys = bounds_ring, keys_ring
    elif which_bounds in ('injector_Surrogate_Ring', 'injector_Actual_Ring'):
        bounds, keys = bounds_injector, keys_injector
    else:
        bounds, keys = np.array([*bounds_ring, *bounds_injector]), [*keys_ring, *keys_injector]
    if range_factor != 1.0:
        assert range_factor > 0.0
        keys_to_not_change = () if keys_to_not_change is None else keys_to_not_change
        for key in keys_to_not_change:
            assert key in keys
        for bound, key in zip(bounds, keys):
            if key not in keys_to_not_change:
                delta = (bound[1] - bound[0]) * range_factor
                bound[0] -= delta
                bound[1] += delta
                bound[0] = 0.0 if bound[0] < 0.0 else bound[0]
                assert bound[0] >= 0.0
    return tuple(bounds)


def get_cost_function(which_system: str, ring_version, ring_params: Optional[tuple], injector_params: Optional[tuple],
                      use_solenoid_field, use_bumper, num_particles, use_standard_tube_OD,
                      use_standard_mag_size, use_energy_correction, include_mag_cross_talk) -> Callable[[tuple], float]:
    """Return a function that gives the cost when given solution parameters such as ring and or injector parameters.
    Wraps Solver class."""
    solver = Solver(which_system, ring_version, ring_params=ring_params, use_solenoid_field=use_solenoid_field,
                    use_bumper=use_bumper,
                    num_particles=num_particles, use_standard_tube_OD=use_standard_tube_OD,
                    use_standard_mag_size=use_standard_mag_size, injector_params=injector_params,
                    use_energy_correction=use_energy_correction, include_mag_cross_talk=include_mag_cross_talk)

    def cost(params: tuple[float, ...]) -> float:
        sol = solver.solve(params)
        # if sol.flux_mult is not None and sol.flux_mult > 10:
        #     print(sol)
        return sol.cost

    return cost


def strip_combiner_params(params) -> LockedDict:
    """Given params, remove Lm_combiner and load_beam_offset params. Used to ring params"""
    new_params = {}
    for key, value in params.items():
        if key not in ('Lm_combiner', 'load_beam_offset'):
            new_params[key] = value
    return LockedDict(new_params)


def ring_params_optimal_without_combiner(ring_version) -> LockedDict:
    """Return optimal ring params with combiner params included"""
    ring_params_optimal_with_combiner = get_optimal_ring_params(ring_version)
    return strip_combiner_params(ring_params_optimal_with_combiner)


def initial_params_from_optimal(which_system, ring_version) -> tuple[float, ...]:
    """Return tuple of initial parameters from optimal parameters of lattice system"""
    if which_system == 'both':
        params_optimal = tuple([*list(ring_params_optimal_without_combiner(ring_version).values()),
                                *list(get_optimal_injector_params(ring_version).values())])
    elif which_system == 'ring':
        params_optimal = tuple(ring_params_optimal_without_combiner(ring_version).values())
    elif which_system in ('injector_Surrogate_Ring', 'injector_Actual_Ring'):
        params_optimal = tuple(get_optimal_injector_params(ring_version).values())
    else:
        raise NotImplementedError
    return params_optimal


def _global_optimize(cost_func, bounds: sequence, time_out_seconds: float, processes: int, disp: bool,
                     progress_file: Optional[str], save_population: Optional[str],
                     initial_vals, init_pop_file) -> tuple[float, float]:
    """globally optimize a storage ring model cost function"""
    member = solve_async(cost_func, bounds, workers=processes,
                         disp=disp, progress_file=progress_file,
                         initial_vals=initial_vals, save_population=save_population,
                         time_out_seconds=time_out_seconds, init_pop_file=init_pop_file)
    x_optimal, cost_min = member.DNA, member.cost
    return x_optimal, cost_min


def _local_optimize(cost_func, bounds: sequence, xi: sequence, disp: bool, processes: int,
                    local_optimizer: str, local_search_region: float) -> tuple[tuple[float, ...], float]:
    """Locally optimize a storage ring model cost function"""
    if local_optimizer == 'octopus':
        x_optimal, cost_min = octopus_optimize(cost_func, bounds, xi, disp=disp, processes=processes,
                                               num_searches_criteria=20, tentacle_length=local_search_region)
    elif local_optimizer == 'simple_line':
        x_optimal, cost_min = line_search(cost_func, xi, 1e-3, bounds, processes=processes)
    else:
        raise ValueError
    return x_optimal, cost_min


def optimize(which_system, method, ring_version, xi: Union[tuple, str] = None, ring_params: tuple = None,
             shrink_bounds_range_factor=np.inf,
             time_out_seconds=np.inf,
             disp=True, processes=-1, local_optimizer='simple_line', use_solenoid_field: bool = False,
             use_bumper: bool = False, local_search_region=.01, num_particles=1024,
             use_standard_tube_OD=False, use_standard_mag_size=False, injector_params=None,
             use_energy_correction=False, progress_file: str = None, initial_vals: sequence = None,
             save_population: str = None, include_mag_cross_talk=False, init_pop_file=None):
    """Optimize a model of the ring and injector"""
    assert which_system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    assert method in ('global', 'local')
    assert xi is not None if method == 'local' else True
    assert ring_params is not None if which_system == 'injector_Actual_Ring' else True
    assert injector_params is not None if which_system == 'ring' else True
    assert xi == 'optimal' if isinstance(xi, str) else True
    xi = initial_params_from_optimal(which_system, ring_version) if xi == 'optimal' else xi
    bounds = make_bounds(which_system, ring_version)
    cost_func = get_cost_function(which_system, ring_version, ring_params, injector_params, use_solenoid_field,
                                  use_bumper,
                                  num_particles, use_standard_tube_OD, use_standard_mag_size,
                                  use_energy_correction, include_mag_cross_talk)
    if method == 'global':
        bounds = shrink_bounds_around_vals(bounds, xi, shrink_bounds_range_factor) if xi is not None else bounds
        x_optimal, cost_min = _global_optimize(cost_func, bounds, time_out_seconds, processes, disp,
                                               progress_file, save_population, initial_vals, init_pop_file)
    else:
        x_optimal, cost_min = _local_optimize(cost_func, bounds, xi, disp, processes, local_optimizer,
                                              local_search_region)

    return x_optimal, cost_min
