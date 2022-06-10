import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
# from asyncDE import solve_Async
from storageRingModeler import StorageRingModel, Solution
from ParticleTracerLatticeClass import ParticleTracerLattice
from latticeElements.utilities import ElementTooShortError, CombinerDimensionError, CombinerIterExceededError
from latticeModels import make_Ring_And_Injector_Version3, RingGeometryError, InjectorGeometryError
from latticeModels_Parameters import optimizerBounds_V1_3, injectorParamsBoundsAny, injectorParamsOptimalAny
from asyncDE import solve_Async


def plot_Results(_params):
    ringParams = _params
    PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(ringParams)
    optimizer = StorageRingModel(PTL_Ring, PTL_Injector)
    optimizer.show_Floor_Plan_And_Trajectories(None, True)


def invalid_Solution(params):
    sol = Solution(params, None, StorageRingModel.maximumCost)
    return sol


def separate_Ring_And_Injector_Params(params):
    ringParams = params[:len(optimizerBounds_V1_3)]
    injectorParams = params[len(optimizerBounds_V1_3):]
    return ringParams, injectorParams


def update_Injector_Params_Dictionary(injectorParams):
    assert len(injectorParams) == len(injectorParamsOptimalAny)
    for key, value in zip(injectorParamsOptimalAny.keys(), injectorParams):
        injectorParamsOptimalAny.super_Special_Change_Item(key, value)


def solve(_params: tuple[float, ...]) -> Solution:
    ringParams, injectorParams = separate_Ring_And_Injector_Params(_params)
    update_Injector_Params_Dictionary(injectorParams)
    PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(ringParams)
    energyConservation = True
    collisionDynamics = False
    numParticles = 500
    optimizer = StorageRingModel(PTL_Ring, PTL_Injector, collisionDynamics=collisionDynamics,
                                 numParticlesSwarm=numParticles)
    cost, fluxMultiplication = optimizer.mode_Match(energyConservation, floorPlanCostCutoff=.05)
    sol = Solution(_params, fluxMultiplication, cost)
    return sol


def solve_For_Lattice_Params(params: tuple[float, ...]) -> Solution:
    try:
        sol = solve(params)
    except (RingGeometryError, InjectorGeometryError, ElementTooShortError):
        sol = invalid_Solution(params)
    except:
        raise Exception("unhandled exception on paramsForBuilding: ", repr(params))
    return sol


def wrapper(params):
    sol = solve_For_Lattice_Params(params)
    if sol.fluxMultiplication > 10:
        print(sol)
    cost = sol.cost
    return cost


def main():
    boundsRing = np.array(list(optimizerBounds_V1_3.values()))
    boundsInjector = list(injectorParamsBoundsAny.values())
    bounds = np.array([*boundsRing, *boundsInjector])
    for i, bound in enumerate(bounds):
        delta = bound[1] - bound[0]
        bounds[i][0] -= delta * .1
        bounds[i][1] += delta * .1
        bounds[i] = [bounds[i][j] if bounds[i][j] >= 0.0 else 0.0 for j in [0, 1]]
    #
    solve_Async(wrapper, bounds, 15 * len(bounds), timeOut_Seconds=1_000_000, disp=True, workers=10,
                saveData='optimizerProgress1')
    #
    # from octopusOptimizer import octopus_Optimize
    # octopus_Optimize(wrapper, bounds, x, disp=True, numSearchesCriteria=1000, tentacleLength=.02,
    #                  searchCutoff=.005, processes=10, maxTrainingMemory=250)
    # print(wrapper(x))
    # plot_Results(x)


if __name__ == '__main__':
    main()

"""
----------Solution-----------   
parameters: None
cost: 0.7190999733458808
flux multiplication: 88.26480620549987
----------------------------
0.7190999733458808
"""
