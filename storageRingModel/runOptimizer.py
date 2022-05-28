import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from asyncDE import solve_Async
from storageRingOptimizer import LatticeOptimizer,Solution
from ParticleTracerLatticeClass import ParticleTracerLattice
from elementPT import ElementTooShortError
from latticeModels import make_Ring_And_Injector_Version3,RingGeometryError,InjectorGeometryError
from latticeModels_Parameters import optimizerBounds_V1_3,atomCharacteristic

def plot_Results(params):
    PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(params)
    optimizer = LatticeOptimizer(PTL_Ring, PTL_Injector)
    optimizer.show_Floor_Plan_And_Trajectories(None,True)

def invalid_Solution(XLattice,invalidInjector=None,invalidRing=None):
    sol=Solution()
    sol.xRing_TunedParams1=XLattice
    sol.fluxMultiplication=0.0
    sol.swarmCost,sol.floorPlanCost=1.0,1.0
    sol.cost=sol.swarmCost+sol.floorPlanCost
    sol.description='Pseudorandom search'
    sol.invalidRing=invalidRing
    sol.invalidInjector=invalidInjector
    return sol

def solution_From_Lattice(PTL_Ring: ParticleTracerLattice, PTL_Injector: ParticleTracerLattice)-> Solution:
    optimizer = LatticeOptimizer(PTL_Ring, PTL_Injector,collisionDynamics=True)

    sol = Solution()
    knobParams = None
    swarmCost, floorPlanCost,swarmTraced = optimizer.mode_Match_Cost(knobParams, False, False, floorPlanCostCutoff=.05,
                                                         rejectUnstable=False, returnFullResults=True)
    if swarmTraced is None: #wasn't traced because of other cutoff
        sol.floorPlanCost = floorPlanCost
        sol.cost = 1.0 + floorPlanCost
        sol.swarmCost = np.nan
        sol.fluxMultiplication=np.nan
    else:
        sol.floorPlanCost = floorPlanCost
        sol.swarmCost = swarmCost
        sol.cost = swarmCost + floorPlanCost
        sol.fluxMultiplication = optimizer.compute_Flux_Multiplication(swarmTraced)

    return sol

def solve_For_Lattice_Params(_params:tuple)-> Solution:
    paramsForBuilding=_params
    try:
        PTL_Ring, PTL_Injector=make_Ring_And_Injector_Version3(paramsForBuilding)
        sol = solution_From_Lattice(PTL_Ring, PTL_Injector)
        sol.paramsForBuilding=paramsForBuilding
    except RingGeometryError:
        sol=invalid_Solution(paramsForBuilding)
    except InjectorGeometryError:
        sol = invalid_Solution(paramsForBuilding)
    except ElementTooShortError:
        sol = invalid_Solution(paramsForBuilding)
    except:
        raise Exception("unhandled exception on paramsForBuilding: ",repr(paramsForBuilding))
    return sol

def wrapper(params):
    sol=solve_For_Lattice_Params(params)
    if sol.fluxMultiplication>10:
        print(sol)
    cost=sol.cost
    return cost

def main():
    bounds = np.array(list(optimizerBounds_V1_3.values()))

    # solve_Async(wrapper,bounds,15*len(bounds),timeOut_Seconds=100_000,disp=True,workers=10,saveData='optimizerProgress')
    #
    x = [0.02477938, 0.01079024, 0.04059919, 0.010042, 0.07175166, 0.51208528]
    # print(wrapper(x))
    def func(T):
        from temp3 import MUT
        MUT.T=T
        return wrapper(x)
    print(func(.01))
    # from helperTools import tool_Parallel_Process
    # TArr=np.logspace(-4,np.log10(20e-3),20)
    # res= tool_Parallel_Process(func,TArr)
    # print(res)
    # from helperTools import plt
    # plt.plot(TArr,res)
    # plt.show()
    # plot_Results(x)
if __name__=='__main__':
    main()
