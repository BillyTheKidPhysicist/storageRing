from typing import Callable
from typing import Optional

import numpy as np

from asyncDE import solve_Async
from latticeElements.utilities import ElementTooShortError, CombinerDimensionError
from latticeModels import make_Ring_And_Injector, RingGeometryError, InjectorGeometryError, \
    make_Injector_Version_Any, make_Ring_Surrogate_For_Injection_Version_1
from latticeModels_Parameters import optimizerBounds_V1_3, injectorParamsBoundsAny, injectorParamsOptimalAny, \
    injectorRingConstraintsV1, lockedDict
from octopusOptimizer import octopus_Optimize
from simpleLineSearch import line_Search
from storageRingModeler import StorageRingModel
from typeHints import lst_tup_arr


class Solution:
    """class to hold onto results of each solution"""

    def __init__(self, params, cost, fluxMultiplication=None, survival=None):
        self.params, self.fluxMultiplication, self.cost, self.survival = params, fluxMultiplication, cost, survival

    def __str__(self) -> str:  # method that gets called when you do print(Solution())
        np.set_printoptions(precision=100)
        string = '----------Solution-----------   \n'
        string += 'parameters: ' + repr(self.params) + '\n'
        string += 'cost: ' + str(self.cost) + '\n'
        if self.fluxMultiplication is not None:
            string += 'flux multiplication: ' + str(self.fluxMultiplication) + '\n'
        elif self.survival is not None:
            string += 'injection survival: ' + str(self.survival) + '\n'
        string += '----------------------------'
        return string


def update_Injector_Params_Dictionary(injectorParams):
    assert len(injectorParams) == len(injectorParamsOptimalAny)
    for key, value in zip(injectorParamsOptimalAny.keys(), injectorParams):
        injectorParamsOptimalAny.super_Special_Change_Item(key, value)


def invalid_Solution(params):
    sol = Solution(params, StorageRingModel.maximumCost)
    return sol


def injected_Swarm_Cost(model) -> float:
    swarmRingTraced = model.inject_And_Trace_Swarm()
    numParticlesInitial = model.swarmInjectorInitial.num_Particles(weighted=True)
    numParticlesFinal = swarmRingTraced.num_Particles(weighted=True, unClippedOnly=True)
    swarmCost = (numParticlesInitial - numParticlesFinal) / numParticlesInitial
    assert 0.0 <= swarmCost <= 1.0
    return swarmCost


def build_Injector_And_Surrrogate(injectorParams,options):
    assert len(injectorParams) == len(injectorParamsBoundsAny)
    surrogateParams = lockedDict(
        {'rpLens1': injectorRingConstraintsV1['rp1LensMax'], 'L_Lens': .5})
    paramsInjectorDict = {}
    for key, val in zip(injectorParamsBoundsAny.keys(), injectorParams):
        paramsInjectorDict[key] = val
    paramsInjectorDict = lockedDict(paramsInjectorDict)
    PTL_Injector = make_Injector_Version_Any(paramsInjectorDict,options=options)
    PTL_Ring_Surrogate = make_Ring_Surrogate_For_Injection_Version_1(paramsInjectorDict, surrogateParams,options=options)
    return PTL_Ring_Surrogate, PTL_Injector


class Solver:
    def __init__(self, system, ringParams=None, useSolenoidField=False, useCollisions=False,
                 useEnergyCorrection=False, numParticles=1024, useBumper=False,standard_tube_ODs=False,
                 standard_mag_sizes=False):
        assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
        self.system = system
        self.ringParams = ringParams
        self.useCollisions = useCollisions
        self.useEnergyCorrection = useEnergyCorrection
        self.numParticles = numParticles
        self.storageRingSystemOptions = {'useSolenoidField': useSolenoidField, 'includeBumper': useBumper,
                                         'standard_tube_ODs':standard_tube_ODs,'standard_mag_sizes':standard_mag_sizes}

    def unpack_Params(self, params):
        ringParams, injectorParams = None, None
        if self.system == 'ring':
            ringParams = params
            injectorParams = tuple(injectorParamsOptimalAny.values())
        elif self.system == 'injector_Surrogate_Ring':
            injectorParams = params
        elif self.system == 'injector_Actual_Ring':
            ringParams = self.ringParams
            injectorParams = params
        else:
            ringParams = params[:len(optimizerBounds_V1_3)]
            injectorParams = params[len(optimizerBounds_V1_3):]
        return ringParams, injectorParams

    def build_Lattices(self, params):
        ringParams, injectorParams = self.unpack_Params(params)
        if self.system == 'injector_Surrogate_Ring':
            PTL_Ring, PTL_Injector = build_Injector_And_Surrrogate(injectorParams, options=self.storageRingSystemOptions)
        else:
            systemParams = (ringParams, injectorParams)
            PTL_Ring, PTL_Injector = make_Ring_And_Injector(systemParams, '3', options=self.storageRingSystemOptions)
        return PTL_Ring, PTL_Injector

    def make_System_Model(self, params):
        PTL_Ring, PTL_Injector = self.build_Lattices(params)
        model = StorageRingModel(PTL_Ring, PTL_Injector, collisionDynamics=self.useCollisions,
                                 numParticlesSwarm=self.numParticles, energyCorrection=self.useEnergyCorrection,
                                 isBumperIncludedInInjector=self.storageRingSystemOptions['includeBumper'])
        return model

    def _solve(self, params: tuple[float, ...]) -> Solution:
        model = self.make_System_Model(params)
        if self.system == 'injector_Surrogate_Ring':
            swarmCost = injected_Swarm_Cost(model)
            survival = 1e2 * (1.0 - swarmCost)
            floorPlanCost = model.floor_Plan_Cost_With_Tunability()
            cost = swarmCost + floorPlanCost
            sol = Solution(params, cost, survival=survival)
        else:
            cost, fluxMultiplication = model.mode_Match(floorPlanCostCutoff=.05)
            sol = Solution(params, cost, fluxMultiplication=fluxMultiplication)
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
    keysInjector = list(injectorParamsOptimalAny.keys())
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


def get_Cost_Function(system: str, ringParams: Optional[tuple], useSolenoidField, useBumper, num_particles,
                      standard_tube_ODs,standard_mag_sizes) -> Callable[[tuple], float]:
    """Return a function that gives the cost when given solution parameters such as ring and or injector parameters.
    Wraps Solver class."""
    solver = Solver(system, ringParams=ringParams, useSolenoidField=useSolenoidField, useBumper=useBumper,
                    numParticles=num_particles,standard_tube_ODs=standard_tube_ODs,standard_mag_sizes=standard_mag_sizes)

    def cost(params: tuple[float, ...]) -> float:
        sol = solver.solve(params)
        if sol.fluxMultiplication is not None and sol.fluxMultiplication > 10:
            print(sol)
        return sol.cost

    return cost


def _global_Optimize(cost_Function, bounds: lst_tup_arr, globalTol: float, processes: int, disp: bool) -> tuple[
    float, float]:
    """globally optimize a storage ring model cost function"""
    member = solve_Async(cost_Function, bounds, 15 * len(bounds), workers=processes,
                         saveData='optimizerProgress', tol=globalTol, disp=disp)
    xOptimal, costMin = member.DNA, member.cost
    return xOptimal, costMin


def _local_Optimize(cost_Function, bounds: lst_tup_arr, xi: lst_tup_arr, disp: bool, processes: int,
                    local_optimizer: str, local_search_region: float) -> tuple[tuple[float,...], float]:
    """Locally optimize a storage ring model cost function"""
    if local_optimizer == 'octopus':
        xOptimal, costMin = octopus_Optimize(cost_Function, bounds, xi, disp=disp, processes=processes,
                                             numSearchesCriteria=20, tentacleLength=local_search_region)
    elif local_optimizer == 'simple_line':
        xOptimal, costMin = line_Search(cost_Function, xi, 1e-3, bounds,processes=processes)
    else:
        raise ValueError
    return xOptimal, costMin


def optimize(system, method, xi: tuple = None, ringParams: tuple = None, expandedBounds=False, globalTol=.005,
             disp=True, processes=10, local_optimizer='octopus', useSolenoidField: bool = False,
             useBumper: bool = False, local_search_region=.01, num_particles=1024,
             standard_tube_ODs=False,standard_mag_sizes=False):
    """Optimize a model of the ring and injector"""
    assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    assert method in ('global', 'local')
    assert xi is not None if method == 'local' else True
    assert ringParams is not None if system == 'injector_Actual_Ring' else True
    bounds = make_Bounds(expandedBounds, whichBounds=system)
    cost_Function = get_Cost_Function(system, ringParams, useSolenoidField, useBumper, num_particles,
                                      standard_tube_ODs,standard_mag_sizes)

    if method == 'global':
        xOptimal, costMin = _global_Optimize(cost_Function, bounds, globalTol, processes, disp)
    else:
        xOptimal, costMin = _local_Optimize(cost_Function, bounds, xi, disp, processes, local_optimizer,
                                            local_search_region)

    return xOptimal, costMin


def main():



    optimize('injector_Surrogate_Ring','global',standard_mag_sizes=True,standard_tube_ODs=True)


    # xOpt,cost=optimize('injector_Surrogate_Ring','local',local_optimizer='simple_line',xi=xi)
    # print('line results:')
    # print(xOpt,cost)
    # optimize('injector_Surrogate_Ring','local',xi=tuple(xOpt),local_search_region=.02)
    # # optimize('injector_Surrogate_Ring','local')


if __name__ == '__main__':
    main()

"""

injector stuff:  
---population member---- 
DNA: array([0.12150846527965559 , 0.026300063215229087, 0.15880857130535447 ,
       0.023840726815439268, 0.15587511504069645 , 0.008366847447454981,
       0.10861195981388541 , 0.2906100314153392  , 0.2077660647873749  ])
cost: 0.15526356500496963

---population member---- 
DNA: array([0.09572666856412555 , 0.02698268769887092 , 0.1916190281352214  ,
       0.026884058369130694, 0.14529769261910877 , 0.007595821397923533,
       0.1018789051202314  , 0.29899148256024266 , 0.2518145293100914  ])
cost: 0.1572265625


---population member---- 
DNA: array([0.12487268204491239 , 0.026452795207681238, 0.1544875779627295  ,
       0.023746964206229768, 0.14586066734676437 , 0.007358920017384266,
       0.11226613872967701 , 0.2866928018124262  , 0.22395295855036998 ])
cost: 0.1611328125


0.1494140625 [0.12150847 0.02630006 0.15880857 0.02384073 0.15787512 0.00836685
 0.10861196 0.29061003 0.20751606]



Ring stuff: 
---population member---- 
DNA: array([0.025013349356554054, 0.008001737370826989, 0.04                ,
       0.009429285103391407, 0.051762587781885874, 0.4732940268274622  ])
cost: 0.8136743583131429

"""
