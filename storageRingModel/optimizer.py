from typing import Callable
from typing import Optional

import numpy as np

from ParticleTracerLatticeClass import ParticleTracerLattice
from asyncDE import solve_async
from latticeElements.utilities import CombinerDimensionError
from latticeElements.utilities import ElementTooShortError as ElementTooShortErrorFields
from latticeModels.injectorModel_1 import make_injector_lattice, injector_param_bounds
from latticeModels.latticeModelUtilities import RingGeometryError, InjectorGeometryError, assert_combiners_are_same
from latticeModels.ringModelSurrogate_1 import make_ring_surrogate
from latticeModels.ringModel_1 import ring_param_bounds
from latticeModels.systemModel import make_system_model
from octopusOptimizer import octopus_optimize
from simpleLineSearch import line_search
from storageRingModeler import StorageRingModel
from typeHints import sequence
from ParticleTracerClass import ElementTooShortError as ElementTooShortErrorTimeStep


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


def build_injector_and_surrrogate(injector_params, options: dict) -> tuple[
    ParticleTracerLattice, ParticleTracerLattice]:
    assert len(injector_params) == len(injector_param_bounds)

    lattice_injector = make_injector_lattice(injector_params, options=options)
    lattice_surrogate = make_ring_surrogate(injector_params, options=options)
    assert_combiners_are_same(lattice_injector, lattice_surrogate)
    return lattice_surrogate, lattice_injector


def build_lattice_params_dict(params, which: str) -> dict:
    assert not isinstance(params, dict) and which in ('ring', 'injector')
    keys = ring_param_bounds.keys() if which == 'ring' else injector_param_bounds.keys()
    assert len(keys) == len(params)
    return dict(zip(keys, params))


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
            ring_params = params[:len(ring_param_bounds)]
            injector_params = params[-len(injector_param_bounds):]
            assert len(ring_params) + len(injector_params) == len(params)
        ring_params = None if ring_params is None else build_lattice_params_dict(ring_params, 'ring')
        injector_params = None if injector_params is None else build_lattice_params_dict(injector_params, 'injector')
        if ring_params is not None and injector_params is not None:
            ring_params['Lm_combiner']=injector_params['Lm_combiner']
            ring_params['load_beam_offset']=injector_params['load_beam_offset']
        return ring_params, injector_params

    def build_lattices(self, params):
        ring_params, injector_params = self.unpack_params(params)
        if self.system == 'injector_Surrogate_Ring':
            lattice_ring, lattice_injector = build_injector_and_surrrogate(injector_params,
                                                                           options=self.storage_ring_system_options)
        else:
            lattice_ring, lattice_injector = make_system_model(ring_params, injector_params,
                                                               self.storage_ring_system_options)
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
        except (RingGeometryError, InjectorGeometryError, ElementTooShortErrorFields,
                CombinerDimensionError,ElementTooShortErrorTimeStep):
            sol = invalid_Solution(params)
        except:
            print('exception with:', repr(params))
            raise Exception("unhandled exception on paramsForBuilding: ")
        return sol


def make_bounds(expand, keys_to_not_change=None, which_bounds='ring') -> tuple:
    """Take bounds for ring and injector and combine into new bounds list. Order is ring bounds then injector bounds.
    Optionally expand the range of bounds by 10%, but not those specified to ignore. If none specified, use a
    default list of values to ignore"""
    assert which_bounds in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
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
    return tuple(bounds)


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
        x_optimal, cost_min = line_search(cost_func, xi, 1e-3, bounds, processes=processes)
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
    assert injector_params is not None if system == 'ring' else True
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
    injector_params=  (0.10617748477362449 , 0.01924168427615551 , 0.17524745026617014 ,
       0.025311623781107514, 0.1769771197959241  , 0.00573236996928707 ,
       0.0881778534132327  , 0.27215737293142345 , 0.16347379419098668 )
    optimize('ring', 'global',use_bumper=True,injector_params=injector_params)

    # print(Solver('ring',injector_params=injector_params,use_standard_tube_OD=True).solve(ring_params))

    # print(Solver('ring', injector_params=injector_params, use_standard_tube_OD=False).solve(ring_params))


if __name__ == '__main__':
    main()


"""
BEST MEMBER BELOW
---population member---- 
DNA: array([0.10617748477362449 , 0.01924168427615551 , 0.17524745026617014 ,
       0.025311623781107514, 0.1769771197959241  , 0.00573236996928707 ,
       0.0881778534132327  , 0.27215737293142345 , 0.16347379419098668 ])
cost: 0.04559906124553903
"""