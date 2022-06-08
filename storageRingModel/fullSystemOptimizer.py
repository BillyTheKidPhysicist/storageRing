import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
# from asyncDE import solve_Async
from storageRingModeler import StorageRingModel, Solution
from ParticleTracerLatticeClass import ParticleTracerLattice
from elementPT import ElementTooShortError, CombinerDimensionError, CombinerIterExceededError
from latticeModels import make_Ring_And_Injector_Version3, RingGeometryError, InjectorGeometryError
from latticeModels_Parameters import optimizerBounds_V1_3, injectorParamsBoundsAny, injectorParamsOptimalAny


def plot_Results(_params):
    ringParams = _params
    PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(ringParams)
    optimizer = StorageRingModel(PTL_Ring, PTL_Injector)
    optimizer.show_Floor_Plan_And_Trajectories(None, True)


def invalid_Solution(XLattice, invalidInjector=None, invalidRing=None):
    sol = Solution()
    sol.xRing_TunedParams1 = XLattice
    sol.fluxMultiplication = 0.0
    sol.swarmCost, sol.floorPlanCost = 1.0, 1.0
    sol.cost = sol.swarmCost + sol.floorPlanCost
    sol.description = 'Pseudorandom search'
    sol.invalidRing = invalidRing
    sol.invalidInjector = invalidInjector
    return sol


def solution_From_Lattice(PTL_Ring: ParticleTracerLattice, PTL_Injector: ParticleTracerLattice) -> Solution:
    energyConservation = False
    collisionDynamics = True
    optimizer = StorageRingModel(PTL_Ring, PTL_Injector, collisionDynamics=collisionDynamics)

    sol = Solution()
    knobParams = None
    swarmCost, floorPlanCost, swarmTraced = optimizer.mode_Match_Cost(knobParams, False, energyConservation,
                                                                      floorPlanCostCutoff=.05,
                                                                      rejectUnstable=False, returnFullResults=True)
    if swarmTraced is None:  # wasn't traced because of other cutoff
        sol.floorPlanCost = floorPlanCost
        sol.cost = 1.0 + floorPlanCost
        sol.swarmCost = np.nan
        sol.fluxMultiplication = np.nan
    else:
        sol.floorPlanCost = floorPlanCost
        sol.swarmCost = swarmCost
        sol.cost = swarmCost + floorPlanCost
        sol.fluxMultiplication = optimizer.compute_Flux_Multiplication(swarmTraced)

    return sol


def separate_Ring_And_Injector_Params(params):
    ringParams = params[:len(optimizerBounds_V1_3)]
    injectorParams = params[len(optimizerBounds_V1_3):]
    return ringParams, injectorParams


def update_Injector_Params_Dictionary(injectorParams):
    assert len(injectorParams) == len(injectorParamsOptimalAny)
    for key, value in zip(injectorParamsOptimalAny.keys(), injectorParams):
        injectorParamsOptimalAny.super_Special_Change_Item(key, value)


def solve_For_Lattice_Params(params: tuple) -> Solution:
    ringParams, injectorParams = separate_Ring_And_Injector_Params(params)
    update_Injector_Params_Dictionary(injectorParams)

    # print(repr(injectorParamsOptimalAny))
    # print('-----------')
    try:
        PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(ringParams)
        sol = solution_From_Lattice(PTL_Ring, PTL_Injector)
        sol.ringParams = params
    except RingGeometryError:
        sol = invalid_Solution(params)
    except InjectorGeometryError:
        sol = invalid_Solution(params)
    except ElementTooShortError:
        sol = invalid_Solution(params)
    except CombinerDimensionError:
        sol = invalid_Solution(params)
    except CombinerIterExceededError:
        print('iter error', params)
        sol = invalid_Solution(params)
    except:
        raise Exception("unhandled exception on params: ", repr(params))
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
    # for i, bound in enumerate(bounds):
    #     delta = bound[1] - bound[0]
    #     bounds[i][0] -= delta * .1
    #     bounds[i][1] += delta * .1
    #     bounds[i]=[bounds[i][j] if bounds[i][j]>=0.0 else 0.0 for j in [0,1]]

    # solve_Async(wrapper, bounds, 15 * len(bounds), timeOut_Seconds=1_000_000, disp=True, workers=10,
    #             saveData='optimizerProgress1')

    # from octopusOptimizer import octopus_Optimize
    x = np.array([0.02283789, 0.0083024, 0.0376554, 0.00739516, 0.05774111,
                  0.49668791, 0.05136774, 0.01352533, 0.20385701, 0.02954697,
                  0.15152598, 0.03781438, 0.01858794, 0.0938976, 0.23864994,
                  0.42840668])
    # octopus_Optimize(wrapper, bounds, x, disp=True, numSearchesCriteria=1000, tentacleLength=.02,
    #                  searchCutoff=.005, processes=10, maxTrainingMemory=250)
    print(wrapper(x))
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
