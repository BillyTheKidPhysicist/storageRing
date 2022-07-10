from typing import Callable
from typing import Optional

import numpy as np

from asyncDE import solve_async
from latticeElements.utilities import ElementTooShortError, CombinerDimensionError
from latticeModels import make_Ring_And_Injector, RingGeometryError, InjectorGeometryError, \
    make_Injector_Version_Any, make_Ring_Surrogate_For_Injection_Version_1
from latticeModels_Parameters import optimizerBounds_V1_3, injectorParamsBoundsAny, injectorRingConstraintsV1,\
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


def injected_Swarm_Cost(model) -> float:
    swarmRingTraced = model.inject_swarm()
    num_particlesInitial = model.swarm_injector_initial.num_Particles(weighted=True)
    num_particlesFinal = swarmRingTraced.num_Particles(weighted=True, unClippedOnly=True)
    swarm_cost = (num_particlesInitial - num_particlesFinal) / num_particlesInitial
    assert 0.0 <= swarm_cost <= 1.0
    return swarm_cost


def build_Injector_And_Surrrogate(injector_params,options):
    assert len(injector_params) == len(injectorParamsBoundsAny)
    surrogateParams = LockedDict(
        {'rpLens1': injectorRingConstraintsV1['rp1LensMax'], 'L_Lens': .5})
    paramsInjectorDict = {}
    for key, val in zip(injectorParamsBoundsAny.keys(), injector_params):
        paramsInjectorDict[key] = val
    paramsInjectorDict = LockedDict(paramsInjectorDict)
    lattice_injector = make_Injector_Version_Any(paramsInjectorDict,options=options)
    PTL_Ring_Surrogate = make_Ring_Surrogate_For_Injection_Version_1(paramsInjectorDict, surrogateParams,options=options)
    return PTL_Ring_Surrogate, lattice_injector


class Solver:
    def __init__(self, system, ring_params=None,injector_params=None, use_solenoid_field=False, use_collisions=False,
                 use_energy_correction=False, num_particles=1024, useBumper=False,use_standard_tube_OD=False,
                 use_standard_mag_size=False):
        assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
        self.system = system
        self.ring_params = ring_params
        self.injector_params=injector_params
        self.use_collisions = use_collisions
        self.use_energy_correction = use_energy_correction
        self.num_particles = num_particles
        self.storage_ring_system_options = {'use_solenoid_field': use_solenoid_field, 'has_bumper': useBumper,
                                         'use_standard_tube_OD':use_standard_tube_OD,'use_standard_mag_size':use_standard_mag_size}

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
            raise NotImplementedError #something seems weird with the below
            ring_params = params[:len(optimizerBounds_V1_3)]
            injector_params = params[len(optimizerBounds_V1_3):]
        return ring_params, injector_params

    def build_Lattices(self, params):
        ring_params, injector_params = self.unpack_params(params)
        if self.system == 'injector_Surrogate_Ring':
            lattice_ring, lattice_injector = build_Injector_And_Surrrogate(injector_params, options=self.storage_ring_system_options)
        else:
            system_params = (ring_params, injector_params)
            lattice_ring, lattice_injector = make_Ring_And_Injector(system_params, '3', options=self.storage_ring_system_options)
        return lattice_ring, lattice_injector

    def make_System_Model(self, params):
        lattice_ring, lattice_injector = self.build_Lattices(params)
        model = StorageRingModel(lattice_ring, lattice_injector, use_collisions=self.use_collisions,
                                 num_particles_swarm=self.num_particles, use_energy_correction=self.use_energy_correction,
                                 use_bumper=self.storage_ring_system_options['has_bumper'])
        return model

    def _solve(self, params: tuple[float, ...]) -> Solution:
        model = self.make_System_Model(params)
        if self.system == 'injector_Surrogate_Ring':
            swarm_cost = injected_Swarm_Cost(model)
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


def make_Bounds(expand, keysToNotChange=None, whichBounds='ring') -> np.ndarray:
    """Take bounds for ring and injector and combine into new bounds list. Order is ring bounds then injector bounds.
    Optionally expand the range of bounds by 10%, but not those specified to ignore. If none specified, use a
    default list of values to ignore"""
    assert whichBounds in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    boundsRing = np.array(list(optimizerBounds_V1_3.values()))
    boundsInjector = list(injectorParamsBoundsAny.values())
    keysRing = list(optimizerBounds_V1_3.keys())
    keysInjector = list(injectorParamsBoundsAny.keys())
    if whichBounds == 'ring':
        bounds, keys = boundsRing, keysRing
    elif whichBounds in ('injector_Surrogate_Ring', 'injector_Actual_Ring'):
        bounds, keys = boundsInjector, keysInjector
    else:
        bounds, keys = np.array([*boundsRing, *boundsInjector]), [*keysRing, *keysInjector]
    if expand:
        keysToNotChange = () if keysToNotChange is None else keysToNotChange
        for key in keysToNotChange:
            assert key in keys
        for bound, key in zip(bounds, keys):
            if key not in keysToNotChange:
                delta = (bound[1] - bound[0]) / 10.0
                bound[0] -= delta
                bound[1] += delta
                bound[0] = 0.0 if bound[0] < 0.0 else bound[0]
                assert bound[0] >= 0.0
    return bounds


def get_cost_function(system: str, ring_params: Optional[tuple],injector_params: Optional[tuple], use_solenoid_field,
                      useBumper, num_particles,use_standard_tube_OD,use_standard_mag_size) -> Callable[[tuple], float]:
    """Return a function that gives the cost when given solution parameters such as ring and or injector parameters.
    Wraps Solver class."""
    solver = Solver(system, ring_params=ring_params, use_solenoid_field=use_solenoid_field, useBumper=useBumper,
                    num_particles=num_particles,use_standard_tube_OD=use_standard_tube_OD,
                    use_standard_mag_size=use_standard_mag_size,injector_params=injector_params)

    def cost(params: tuple[float, ...]) -> float:
        sol = solver.solve(params)
        if sol.flux_mult is not None and sol.flux_mult > 10:
            print(sol)
        return sol.cost

    return cost


def _global_optimize(cost_Function, bounds: sequence, globalTol: float, processes: int, disp: bool) -> tuple[
    float, float]:
    """globally optimize a storage ring model cost function"""
    member = solve_async(cost_Function, bounds, 15 * len(bounds), workers=processes,
                         save_data='optimizerProgress', tol=globalTol, disp=disp)
    xOptimal, costMin = member.DNA, member.cost
    return xOptimal, costMin


def _local_optimize(cost_Function, bounds: sequence, xi: sequence, disp: bool, processes: int,
                    local_optimizer: str, local_search_region: float) -> tuple[tuple[float,...], float]:
    """Locally optimize a storage ring model cost function"""
    if local_optimizer == 'octopus':
        xOptimal, costMin = octopus_optimize(cost_Function, bounds, xi, disp=disp, processes=processes,
                                             num_searches_criteria=20, tentacle_length=local_search_region)
    elif local_optimizer == 'simple_line':
        xOptimal, costMin = line__search(cost_Function, xi, 1e-3, bounds,processes=processes)
    else:
        raise ValueError
    return xOptimal, costMin


def optimize(system, method, xi: tuple = None, ring_params: tuple = None, expandedBounds=False, globalTol=.005,
             disp=True, processes=10, local_optimizer='octopus', use_solenoid_field: bool = False,
             useBumper: bool = False, local_search_region=.01, num_particles=1024,
             use_standard_tube_OD=False,use_standard_mag_size=False, injector_params=None):
    """Optimize a model of the ring and injector"""
    assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    assert method in ('global', 'local')
    assert xi is not None if method == 'local' else True
    assert ring_params is not None if system == 'injector_Actual_Ring' else True
    assert injector_params is not None if system in ('ring','both') else True
    bounds = make_Bounds(expandedBounds, whichBounds=system)
    cost_Function = get_cost_function(system, ring_params,injector_params, use_solenoid_field, useBumper, num_particles,
                                      use_standard_tube_OD,use_standard_mag_size)

    if method == 'global':
        xOptimal, costMin = _global_optimize(cost_Function, bounds, globalTol, processes, disp)
    else:
        xOptimal, costMin = _local_optimize(cost_Function, bounds, xi, disp, processes, local_optimizer,
                                            local_search_region)

    return xOptimal, costMin


def main():

    injector_params=(0.05210079         , 0.01473206         , 0.16259032         ,
       0.02387552         , 0.1480253          , 0.00820821         ,
       0.09877617999999999, 0.2650349          , 0.218033835        )

    ring_params=(0.022428779923884062, 0.009666045743955602, 0.01                ,
       0.05                , 0.5234875264044453  )

    # optimize('ring','global',injector_params=injector_params,use_standard_mag_size=True,use_standard_tube_OD=True)

    optimize('injector_Actual_Ring', 'local', ring_params=ring_params, xi=injector_params,
             use_standard_mag_size=True, use_standard_tube_OD=True)

    # xOpt,cost=optimize('injector_Surrogate_Ring','local',local_optimizer='simple_line',xi=xi)
    # print('line results:')
    # print(xOpt,cost)
    # optimize('injector_Surrogate_Ring','local',xi=tuple(xOpt),local_search_region=.02)
    # # optimize('injector_Surrogate_Ring','local')


if __name__ == '__main__':
    main()

"""

injector stuff:  
----------Solution-----------   
parameters: (0.05285079, 0.01473206, 0.16540282, 0.02387552, 0.1480253, 0.00820821, 0.09821368, 0.2650349, 0.21845571)
cost: 0.1152597007175391
injection survival: 88.4765625
----------------------------



Ring stuff: 
0.7616359151405484 0.00017797851562500002 array([0.022428779923884062, 0.009666045743955602, 0.01                ,
       0.05                , 0.5234875264044453  ])

"""
