import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from typing import Callable
from octopusOptimizer import octopus_Optimize
from storageRingModeler import StorageRingModel
from latticeElements.utilities import ElementTooShortError, CombinerDimensionError
from latticeModels import make_Ring_And_Injector, RingGeometryError, InjectorGeometryError, \
    make_Injector_Version_Any, make_Ring_Surrogate_For_Injection_Version_1
from latticeModels_Parameters import optimizerBounds_V1_3, injectorParamsBoundsAny, injectorParamsOptimalAny, \
    injectorRingConstraintsV1, lockedDict
from asyncDE import solve_Async


class Solution:
    """class to hold onto results of each solution"""

    def __init__(self, params, cost, fluxMultiplication=None, survival=None):
        self.params, self.fluxMultiplication, self.cost, self.survival = params, fluxMultiplication, cost, survival
        assert not (fluxMultiplication is None and survival is None)

    def __str__(self) -> str:  # method that gets called when you do print(Solution())
        string = '----------Solution-----------   \n'
        string += 'parameters: ' + repr(self.params) + '\n'
        string += 'cost: ' + str(self.cost) + '\n'
        if self.fluxMultiplication is not None:
            string += 'flux multiplication: ' + str(self.fluxMultiplication) + '\n'
        else:
            string += 'injection survival: ' + str(self.survival) + '\n'
        string += '----------------------------'
        return string


def update_Injector_Params_Dictionary(injectorParams):
    assert len(injectorParams) == len(injectorParamsOptimalAny)
    for key, value in zip(injectorParamsOptimalAny.keys(), injectorParams):
        injectorParamsOptimalAny.super_Special_Change_Item(key, value)


def invalid_Solution(params):
    sol = Solution(params, None, StorageRingModel.maximumCost)
    return sol


def injected_Swarm_Cost(model) -> float:
    swarmRingTraced = model.inject_And_Trace_Swarm()
    numParticlesInitial = model.swarmInjectorInitial.num_Particles(weighted=True)
    numParticlesFinal = swarmRingTraced.num_Particles(weighted=True, unClippedOnly=True)
    swarmCost = (numParticlesInitial - numParticlesFinal) / numParticlesInitial
    assert 0.0 <= swarmCost <= 1.0
    return swarmCost


def build_Injector_And_Surrrogate(injectorParams):
    assert len(injectorParams) == len(injectorParamsBoundsAny)
    surrogateParams = lockedDict(
        {'rpLens1': injectorRingConstraintsV1['rp1LensMax'], 'rpLens2': .025, 'L_Lens': .5})
    paramsInjectorDict = {}
    for key, val in zip(injectorParamsBoundsAny.keys(), injectorParams):
        paramsInjectorDict[key] = val
    paramsInjectorDict = lockedDict(paramsInjectorDict)
    PTL_Injector = make_Injector_Version_Any(paramsInjectorDict)
    PTL_Ring_Surrogate = make_Ring_Surrogate_For_Injection_Version_1(paramsInjectorDict, surrogateParams)
    return PTL_Ring_Surrogate, PTL_Injector


class Solver:
    def __init__(self, system, ringParams, useSolenoidField=False, useMagnetErrors=False, useCollisions=False,
                 useEnergyCorrection=False, numParticles=1024):
        assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
        self.system = system
        self.ringParams = ringParams
        self.useCollisions = useCollisions
        self.useEnergyCorrection = useEnergyCorrection
        self.numParticles = numParticles
        self.options = {'useSolenoidField': useSolenoidField, 'useMagnetErrors': useMagnetErrors}

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
            PTL_Ring, PTL_Injector = build_Injector_And_Surrrogate(injectorParams)
        else:
            systemParams = (ringParams, injectorParams)
            PTL_Ring, PTL_Injector = make_Ring_And_Injector(systemParams, '3', options=self.options)
        return PTL_Ring, PTL_Injector

    def make_System_Model(self, params):
        PTL_Ring, PTL_Injector = self.build_Lattices(params)
        model = StorageRingModel(PTL_Ring, PTL_Injector, collisionDynamics=self.useCollisions,
                                 numParticlesSwarm=self.numParticles, energyCorrection=self.useEnergyCorrection)
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
            print(repr(params))
            raise Exception("unhandled exception on paramsForBuilding: ")
        return sol


def make_Bounds(expand, keysToNotChange=None, whichBounds='ring'):
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


def get_Cost_Function(system: float, ringParams: tuple) -> Callable[[tuple], float]:
    solver = Solver(system, ringParams)

    def cost(params):
        sol = solver.solve(params)
        if sol.fluxMultiplication is not None and sol.fluxMultiplication > 10:
            print(sol)
        return sol.cost

    return cost


def optimize(system, method, xi: tuple = None, ringParams: tuple = None, expandedBounds=False, globalTol=.005,
             disp=True, processes=10):
    assert system in ('ring', 'injector_Surrogate_Ring', 'injector_Actual_Ring', 'both')
    assert method in ('global', 'local')
    assert xi is not None if method == 'local' else True
    assert ringParams is not None if system == 'injector_Actual_Ring' else True
    bounds = make_Bounds(expandedBounds, whichBounds=system)
    cost_Function = get_Cost_Function(system, ringParams)

    if method == 'global':
        member = solve_Async(cost_Function, bounds, 15 * len(bounds), workers=processes,
                             saveData='optimizerProgress', tol=globalTol, disp=disp)
        xOptimal, costMin = member.DNA, member.cost
    else:
        xOptimal, costMin = octopus_Optimize(cost_Function, bounds, xi, disp=disp, processes=processes,
                                             numSearchesCriteria=20,
                                             tentacleLength=.01)
    return xOptimal, costMin


def main():
    # xRing = (0.01232265, 0.00998983, 0.03899118,
    #             0.00796353, 0.10642821,0.4949227 )
    # solver = Solver('ring', None)
    # params=tuple(injectorParamsOptimalAny.values())
    # print(solver.solve(xRing))
    # xi = (*xRing, *list(injectorParamsOptimalAny.values()))
    # optimize('both', 'local', xi=xi)
    print(optimize('injector_Surrogate_Ring', 'global', globalTol=.01))


if __name__ == '__main__':
    main()
