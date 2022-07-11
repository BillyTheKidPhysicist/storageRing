from typing import Callable
from typing import Optional

import numpy as np

from asyncDE import solve_async
from latticeElements.utilities import ElementTooShortError, CombinerDimensionError
from latticeModels import make_ring_and_injector, RingGeometryError, InjectorGeometryError, \
    make_injector_version_any, make_ring_surrogate_for_injection_version_1
from latticeModels_Parameters import optimizerBounds_V1_3, injectorParamsBoundsAny, injectorRingConstraintsV1, \
    LockedDict
from octopusOptimizer import octopus_optimize
from simpleLineSearch import line__search
from storageRingModeler import StorageRingModel
from typeHints import sequence


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


def build_injector_and_surrrogate(injector_params, options):
    assert len(injector_params) == len(injectorParamsBoundsAny)
    surrogate_params = LockedDict(
        {'rpLens1': injectorRingConstraintsV1['rp1LensMax'], 'L_Lens': .5})
    params_injector_dict = {}
    for key, val in zip(injectorParamsBoundsAny.keys(), injector_params):
        params_injector_dict[key] = val
    params_injector_dict = LockedDict(params_injector_dict)
    lattice_injector = make_injector_version_any(params_injector_dict, options=options)
    ptl_ring_surrogate = make_ring_surrogate_for_injection_version_1(params_injector_dict, surrogate_params,
                                                                     options=options)
    return ptl_ring_surrogate, lattice_injector


class Solver:
    def __init__(self, system, ring_params=None, injector_params=None, use_solenoid_field=False, use_collisions=False,
                 use_energy_correction=False, num_particles=1024, use_bumper=False, use_standard_tube_OD=False,
                 use_standard_mag_size=False):
        assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
        self.system = system
        self.ring_params = ring_params
        self.injector_params = injector_params
        self.use_collisions = use_collisions
        self.use_energy_correction = use_energy_correction
        self.num_particles = num_particles
        self.storage_ring_system_options = {'use_solenoid_field': use_solenoid_field, 'has_bumper': use_bumper,
                                            'use_standard_tube_OD': use_standard_tube_OD,
                                            'use_standard_mag_size': use_standard_mag_size}

    def unpack_params(self, params):
        ring_params, injector_params = None, None
        if self.system == 'ring':
            ring_params = params
            injector_params = self.injector_params
        elif self.system == 'injector_Surrogate_Ring':
            injector_params = params
        elif self.system == 'injector_Actual_Ring':
            ring_params = self.ring_params
            injector_params = params
        else:
            raise NotImplementedError  # something seems weird with the below
            ring_params = params[:len(optimizerBounds_V1_3)]
            injector_params = params[len(optimizerBounds_V1_3):]
        return ring_params, injector_params

    def build_lattices(self, params):
        ring_params, injector_params = self.unpack_params(params)
        if self.system == 'injector_Surrogate_Ring':
            lattice_ring, lattice_injector = build_injector_and_surrrogate(injector_params,
                                                                           options=self.storage_ring_system_options)
        else:
            system_params = (ring_params, injector_params)
            lattice_ring, lattice_injector = make_ring_and_injector(system_params, '3',
                                                                    options=self.storage_ring_system_options)
        return lattice_ring, lattice_injector

    def make_system_model(self, params):
        lattice_ring, lattice_injector = self.build_lattices(params)
        model = StorageRingModel(lattice_ring, lattice_injector, use_collisions=self.use_collisions,
                                 num_particles=self.num_particles, use_energy_correction=self.use_energy_correction,
                                 use_bumper=self.storage_ring_system_options['has_bumper'])
        return model

    def _solve(self, params: tuple[float, ...]) -> Solution:
        model = self.make_system_model(params)
        if self.system == 'injector_Surrogate_Ring':
            swarm_cost = injected_swarm_cost(model)
            survival = 1e2 * (1.0 - swarm_cost)
            floor_plan_cost = model.floor_plan_cost_with_tunability()
            cost = swarm_cost + floor_plan_cost
            sol = Solution(params, cost, survival=survival)
        else:
            cost, flux_mult = model.mode_match(floor_plan_cost_cutoff=.05)
            sol = Solution(params, cost, flux_mult=flux_mult)
        return sol

    def solve(self, params: tuple[float, ...]) -> Solution:
        """Solve storage ring system. If using 'both' for system, params must be ring then injector params strung
        together"""
        try:
            sol = self._solve(params)
        except (RingGeometryError, InjectorGeometryError, ElementTooShortError, CombinerDimensionError):
            sol = invalid_Solution(params)
        except:
            print('exception with:', repr(params))
            raise Exception("unhandled exception on paramsForBuilding: ")
        return sol


def make_bounds(expand, keys_to_not_change=None, which_bounds='ring') -> np.ndarray:
    """Take bounds for ring and injector and combine into new bounds list. Order is ring bounds then injector bounds.
    Optionally expand the range of bounds by 10%, but not those specified to ignore. If none specified, use a
    default list of values to ignore"""
    assert which_bounds in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    bounds_ring = np.array(list(optimizerBounds_V1_3.values()))
    bounds_injector = list(injectorParamsBoundsAny.values())
    keys_ring = list(optimizerBounds_V1_3.keys())
    keys_injector = list(injectorParamsBoundsAny.keys())
    if which_bounds == 'ring':
        bounds, keys = bounds_ring, keys_ring
    elif which_bounds in ('injector_Surrogate_Ring', 'injector_Actual_Ring'):
        bounds, keys = bounds_injector, keys_injector
    else:
        bounds, keys = np.array([*bounds_ring, *bounds_injector]), [*keys_ring, *keys_injector]
    if expand:
        keys_to_not_change = () if keys_to_not_change is None else keys_to_not_change
        for key in keys_to_not_change:
            assert key in keys
        for bound, key in zip(bounds, keys):
            if key not in keys_to_not_change:
                delta = (bound[1] - bound[0]) / 10.0
                bound[0] -= delta
                bound[1] += delta
                bound[0] = 0.0 if bound[0] < 0.0 else bound[0]
                assert bound[0] >= 0.0
    return bounds


def get_cost_function(system: str, ring_params: Optional[tuple], injector_params: Optional[tuple], use_solenoid_field,
                      use_bumper, num_particles, use_standard_tube_OD, use_standard_mag_size) -> Callable[
    [tuple], float]:
    """Return a function that gives the cost when given solution parameters such as ring and or injector parameters.
    Wraps Solver class."""
    solver = Solver(system, ring_params=ring_params, use_solenoid_field=use_solenoid_field, use_bumper=use_bumper,
                    num_particles=num_particles, use_standard_tube_OD=use_standard_tube_OD,
                    use_standard_mag_size=use_standard_mag_size, injector_params=injector_params)

    def cost(params: tuple[float, ...]) -> float:
        sol = solver.solve(params)
        if sol.flux_mult is not None and sol.flux_mult > 10:
            print(sol)
        return sol.cost

    return cost


def _global_optimize(cost_func, bounds: sequence, globalTol: float, processes: int, disp: bool) -> tuple[
    float, float]:
    """globally optimize a storage ring model cost function"""
    member = solve_async(cost_func, bounds, 15 * len(bounds), workers=processes,
                         save_data='optimizerProgress', tol=globalTol, disp=disp)
    x_optimal, cost_min = member.DNA, member.cost
    return x_optimal, cost_min


def _local_optimize(cost_func, bounds: sequence, xi: sequence, disp: bool, processes: int,
                    local_optimizer: str, local_search_region: float) -> tuple[tuple[float, ...], float]:
    """Locally optimize a storage ring model cost function"""
    if local_optimizer == 'octopus':
        x_optimal, cost_min = octopus_optimize(cost_func, bounds, xi, disp=disp, processes=processes,
                                               num_searches_criteria=20, tentacle_length=local_search_region)
    elif local_optimizer == 'simple_line':
        x_optimal, cost_min = line__search(cost_func, xi, 1e-3, bounds, processes=processes)
    else:
        raise ValueError
    return x_optimal, cost_min


def optimize(system, method, xi: tuple = None, ring_params: tuple = None, expandedBounds=False, globalTol=.005,
             disp=True, processes=-1, local_optimizer='octopus', use_solenoid_field: bool = False,
             use_bumper: bool = False, local_search_region=.01, num_particles=1024,
             use_standard_tube_OD=False, use_standard_mag_size=False, injector_params=None):
    """Optimize a model of the ring and injector"""
    assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    assert method in ('global', 'local')
    assert xi is not None if method == 'local' else True
    assert ring_params is not None if system == 'injector_Actual_Ring' else True
    assert injector_params is not None if system in ('ring', 'both') else True
    bounds = make_bounds(expandedBounds, which_bounds=system)
    cost_func = get_cost_function(system, ring_params, injector_params, use_solenoid_field, use_bumper,
                                  num_particles,
                                  use_standard_tube_OD, use_standard_mag_size)

    if method == 'global':
        x_optimal, cost_min = _global_optimize(cost_func, bounds, globalTol, processes, disp)
    else:
        x_optimal, cost_min = _local_optimize(cost_func, bounds, xi, disp, processes, local_optimizer,
                                              local_search_region)

    return x_optimal, cost_min


def main():

    optimize('injector_Surrogate_Ring', 'global')

    # print(Solver('ring',injector_params=injector_params,use_standard_tube_OD=True).solve(ring_params))

    # print(Solver('ring', injector_params=injector_params, use_standard_tube_OD=False).solve(ring_params))


if __name__ == '__main__':
    main()


"""

----------Solution-----------   
parameters: array([0.025153165715661275, 0.008921455792494704, 0.00945901807497999 ,
       0.05                , 0.46379142056301526 ])
cost: 0.7905998670566851
flux multiplication: 54.52998502532289



iter 1:

injector: 
0.15208781577932018 array([0.28552378, 0.0146052 , 0.22421804, 0.02979563, 0.16616535,
       0.00741861, 0.08236912, 0.25659989, 0.23644295])


"""