from typing import Callable, Union, Optional

import numpy as np

from async_de import solve_async
from helper_tools import shrink_bounds_around_vals
from lattice_elements.utilities import CombinerDimensionError
from lattice_elements.utilities import ElementTooShortError as ElementTooShortErrorFields
from lattice_models.lattice_model_parameters import combiner_param_bounds
from lattice_models.system_model import get_optimal_ring_params, get_optimal_injector_params
from lattice_models.system_model import make_system_model, get_ring_bounds, get_injector_bounds, \
    make_surrogate_ring_for_injector, make_injector_lattice
from lattice_models.utilities import RingGeometryError, InjectorGeometryError, assert_combiners_are_valid, LockedDict
from octopus_optimizer import octopus_optimize
from particle_tracer import ElementTooShortError as ElementTooShortErrorTimeStep
from particle_tracer_lattice import ParticleTracerLattice
from simple_line_search import line_search
from storage_ring_modeler import StorageRingModel, DEFAULT_SIMULATION_TIME
from type_hints import sequence


# IMPROVEMENT: THIS IS REALLY FRAGILE. SHOULD BE MADE MORE GENERAL
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
    assert_combiners_are_valid(lattice_injector, lattice_surrogate)
    return lattice_surrogate, lattice_injector


def build_lattice_params_dict(params, which: str, ring_version: str) -> dict:
    assert not isinstance(params, dict) and which in ('ring', 'injector')
    keys = get_ring_bounds(ring_version).keys() if which == 'ring' else get_injector_bounds().keys()
    assert len(keys) == len(params)
    return dict(zip(keys, params))


def bounds_and_keys(scheme, ring_version):
    """Return the total bounds for ring,injector,combiner and corresponding keys. Bounds is a list  like
    [(lower,upp),..] and keys is a tuple of 3 lists corresponding to ring,injector,combiner. These keys are used to
     construct the dictionaries for bulding injector and ring from the raw paremeters from optimization"""
    ring_param_bounds = get_ring_bounds(ring_version)
    injector_param_bounds = get_injector_bounds()
    ring_param_keys = list(ring_param_bounds.keys())
    injector_param_keys = list(injector_param_bounds.keys())
    combiner_param_keys = list(combiner_param_bounds.keys())
    if scheme == 'ring':
        injector_param_keys = []
        bounds = [*ring_param_bounds.values(), *combiner_param_bounds.values()]
    elif scheme == 'both':
        bounds = [*ring_param_bounds.values(), *injector_param_bounds.values(), *combiner_param_bounds.values()]
    elif scheme == 'injector_with_surrogate_ring':
        ring_param_keys = []
        raise NotImplementedError
    elif scheme == 'injector_with_ring':
        ring_param_keys = []
        combiner_param_keys = []
        bounds = [*injector_param_bounds.values()]
    else:
        raise NotImplementedError
    return bounds, (ring_param_keys, injector_param_keys, combiner_param_keys)


def scheme_bounds(scheme, ring_version):
    """Return bounds for the scheme and ring_version"""
    bounds, _ = bounds_and_keys(scheme, ring_version)
    return bounds


def length_params(params):
    """Return the length of sequence 'params', return 0 if 'params' is None"""
    return 0 if params is None else len(params)


def split_param_values(param_values: sequence, scheme: str, ring_version: str) -> tuple[list, ...]:
    """Return three lists, ring params, injector params and combiner params, corresponding to the entries
    in 'param_values'"""

    _, (ring_param_keys, injector_param_keys, combiner_param_keys) = bounds_and_keys(scheme, ring_version)
    split_values = []
    idxa = 0
    for keys in (ring_param_keys, injector_param_keys, combiner_param_keys):
        num_params = len(keys)
        idxb = idxa + num_params
        values = param_values[idxa:idxb]
        idxa += len(values)
        split_values.append(values)
    return tuple(split_values)


def add_combiner_values(ring_params, injector_params, combiner_param_keys, combiner_params_values):
    """Add combiner values to ring_params and injector_params. They both must share the same combiner values"""
    for params in (ring_params, injector_params):
        for key, value in zip(combiner_param_keys, combiner_params_values):
            params[key] = value


def ring_system_params(param_values, scheme, ring_version) -> tuple[dict, dict]:
    """Return system parameters for 'ring' optimization scheme in which injector parameters are the stored optimal
    values"""
    _, (ring_param_keys, _, combiner_param_keys) = bounds_and_keys(scheme, ring_version)
    ring_params_values, _, combiner_params_values = split_param_values(param_values, scheme, ring_version)
    ring_params = dict(zip(ring_param_keys, ring_params_values))
    injector_params = dict(get_optimal_injector_params(ring_version))
    add_combiner_values(ring_params, injector_params, combiner_param_keys, combiner_params_values)
    return ring_params, injector_params


def injector_with_ring_system_params(param_values, scheme, ring_version) -> tuple[dict, dict]:
    """Return system parameters for 'injector_with_ring' optimization scheme in which
     ring parameters are the stored optimal values and injector parameters are optimized. """
    _, (_, injector_param_keys, combiner_params_keys) = bounds_and_keys(scheme, ring_version)
    _, inj_params_values, _ = split_param_values(param_values, scheme, ring_version)
    inj_params = dict(zip(injector_param_keys, inj_params_values))
    ring_params = get_optimal_ring_params(ring_version)
    for key in combiner_param_bounds.keys():
        inj_params[key] = ring_params[key]
    return ring_params, inj_params


def both_system_params(param_values, scheme, ring_version) -> tuple[dict, dict]:
    """Return system parameters for 'both' optimization scheme in which parameters for lens and ring are both
     optimized"""
    _, (ring_param_keys, inj_param_keys, combiner_param_keys) = bounds_and_keys(scheme, ring_version)
    ring_param_values, inj_params_values, combiner_params_values = split_param_values(param_values, scheme,
                                                                                      ring_version)
    ring_params = dict(zip(ring_param_keys, ring_param_values))
    inj_params = dict(zip(inj_param_keys, inj_params_values))
    add_combiner_values(ring_params, inj_params, combiner_param_keys, combiner_params_values)
    return ring_params, inj_params


def system_parameters(param_values: sequence, scheme: str, ring_version: str) -> tuple[dict, dict]:
    """Return the system parameters for ring and injector from the parameter values 'param_values' from the
    optimizer. ring and injector params needs to be a dict, while the 'param_values' is a 1D sequence"""
    if scheme == 'ring':
        return ring_system_params(param_values, scheme, ring_version)
    elif scheme == 'injector_with_ring':
        return injector_with_ring_system_params(param_values, scheme, ring_version)
    elif scheme == 'both':
        return both_system_params(param_values, scheme, ring_version)
    else:
        raise NotImplementedError


class Solver:
    def __init__(self, scheme, ring_version: str, ring_params=None, injector_params=None,
                 use_solenoid_field=False, use_collisions=False, num_particles=1024, use_bumper=False,
                 use_standard_tube_OD=False, sim_time_max=DEFAULT_SIMULATION_TIME, use_long_range_fields=False):
        assert scheme in ('ring', 'injector_with_surrogate_ring', 'injector_with_ring', 'both')
        self.scheme = scheme
        self.ring_version = ring_version
        self.ring_params = ring_params
        self.injector_params = injector_params
        self.use_collisions = use_collisions
        self.num_particles = num_particles
        self.sim_time_max = sim_time_max
        self.storage_ring_system_options = {
            'use_solenoid_field': use_solenoid_field,
            'has_bumper': use_bumper,
            'use_standard_tube_OD': use_standard_tube_OD,
            'include_mag_cross_talk_in_ring': use_long_range_fields,
            'build_field_helpers': False  # When the floor plan is violated, fast field helpers are not used, so don't
            # waste time computing them unless needed to trace swarm
        }

    def unpack_params(self, params):
        return system_parameters(params, self.scheme, self.ring_version)

    def build_lattices(self, params):
        ring_params, injector_params = self.unpack_params(params)
        if self.scheme == 'injector_with_surrogate_ring':
            lattice_ring, lattice_injector = build_injector_and_surrrogate(injector_params, self.ring_version,
                                                                           self.storage_ring_system_options)
        else:
            lattice_ring, lattice_injector = make_system_model(ring_params, injector_params, self.ring_version,
                                                               self.storage_ring_system_options)
        return lattice_ring, lattice_injector

    def make_system_model(self, params) -> StorageRingModel:
        lattice_ring, lattice_injector = self.build_lattices(params)
        model = StorageRingModel(lattice_ring, lattice_injector, use_collisions=self.use_collisions,
                                 num_particles=self.num_particles,
                                 use_bumper=self.storage_ring_system_options['has_bumper'],
                                 sim_time_max=self.sim_time_max)
        return model

    def _solve(self, params: tuple[float, ...]) -> Solution:
        model = self.make_system_model(params)
        floor_plan_cost_cutoff = .1
        if self.scheme == 'injector_with_surrogate_ring':
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


def get_cost_function(scheme: str, ring_version, ring_params: Optional[tuple], injector_params: Optional[tuple],
                      use_solenoid_field, use_bumper, num_particles, use_standard_tube_OD,
                      use_long_range_fields) -> Callable[[tuple], float]:
    """Return a function that gives the cost when given solution parameters such as ring and or injector parameters.
    Wraps Solver class."""
    solver = Solver(scheme, ring_version, ring_params=ring_params, use_solenoid_field=use_solenoid_field,
                    use_bumper=use_bumper, num_particles=num_particles, use_standard_tube_OD=use_standard_tube_OD,
                    injector_params=injector_params, use_long_range_fields=use_long_range_fields)

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
        if key not in combiner_param_bounds.keys():
            new_params[key] = value
    return LockedDict(new_params)


def ring_params_optimal_without_combiner(ring_version) -> LockedDict:
    """Return optimal ring params with combiner params included"""
    ring_params_optimal_with_combiner = get_optimal_ring_params(ring_version)
    return strip_combiner_params(ring_params_optimal_with_combiner)


def initial_params_from_optimal(scheme, ring_version) -> tuple[float, ...]:
    """Return tuple of initial parameters from optimal parameters of lattice system"""
    if scheme == 'both':
        params_optimal = tuple([*list(ring_params_optimal_without_combiner(ring_version).values()),
                                *list(get_optimal_injector_params(ring_version).values())])
    elif scheme == 'ring':
        params_optimal = tuple(get_optimal_ring_params(ring_version).values())
    elif scheme == 'injector_with_ring':
        params_optimal = get_optimal_injector_params(ring_version)
        params_optimal = strip_combiner_params(params_optimal)
        params_optimal = tuple(params_optimal.values())
    elif scheme == 'injector_with_surrogate_ring':
        raise NotImplementedError
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


def optimize(scheme, method, ring_version, xi: Union[tuple, str] = None, ring_params: tuple = None,
             shrink_bounds_range_factor=np.inf, time_out_seconds=np.inf, disp=True, processes=-1,
             local_optimizer='simple_line', use_solenoid_field: bool = False, use_bumper: bool = False,
             local_search_region=.01, num_particles=1024, use_standard_tube_OD=False, injector_params=None,
             progress_file: str = None, initial_vals: sequence = None, save_population: str = None,
             use_long_range_fields=False, init_pop_file=None):
    """Optimize a model of the ring and injector"""
    assert scheme in ('ring', 'injector_with_surrogate_ring', 'injector_with_ring', 'both')
    assert method in ('global', 'local')
    assert xi is not None if method == 'local' else True
    if injector_params is not None or ring_params is not None:
        raise NotImplementedError("These values are replaced with the optimal values for now")
    assert xi == 'optimal' if isinstance(xi, str) else True
    if xi is not None and scheme == 'injector':
        raise NotImplementedError
    xi = initial_params_from_optimal(scheme, ring_version) if xi == 'optimal' else xi
    bounds, _ = bounds_and_keys(scheme, ring_version)
    cost_func = get_cost_function(scheme, ring_version, ring_params, injector_params, use_solenoid_field,
                                  use_bumper, num_particles, use_standard_tube_OD, use_long_range_fields)
    if method == 'global':
        bounds = shrink_bounds_around_vals(bounds, xi, shrink_bounds_range_factor) if xi is not None else bounds
        x_optimal, cost_min = _global_optimize(cost_func, bounds, time_out_seconds, processes, disp,
                                               progress_file, save_population, initial_vals, init_pop_file)
    else:
        x_optimal, cost_min = _local_optimize(cost_func, bounds, xi, disp, processes, local_optimizer,
                                              local_search_region)

    return x_optimal, cost_min
