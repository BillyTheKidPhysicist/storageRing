import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from asyncDE import solve_Async
from storageRingOptimizer import LatticeOptimizer,Solution
from ParticleTracerLatticeClass import ParticleTracerLattice
from elementPT import ElementTooShortError
from latticeModels import make_Ring_And_Injector_Version1,RingGeometryError,InjectorGeometryError
from latticeModels_Parameters import optimizerBounds_V1

def plot_Results(params):
    PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version1(params)
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
    optimizer = LatticeOptimizer(PTL_Ring, PTL_Injector)

    sol = Solution()
    knobParams = None
    swarmCost, floorPlanCost,swarmTraced = optimizer.mode_Match_Cost(knobParams, False, True, floorPlanCostCutoff=.05,
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

def solve_For_Lattice_Params(params:tuple)-> Solution:
    try:
        PTL_Ring, PTL_Injector=make_Ring_And_Injector_Version1(params)
        sol = solution_From_Lattice(PTL_Ring, PTL_Injector)
        sol.params=params
    except RingGeometryError:
        sol=invalid_Solution(params)
    except InjectorGeometryError:
        sol = invalid_Solution(params)
    except ElementTooShortError:
        sol = invalid_Solution(params)
    except:
        raise Exception("unhandled exception on params: ",repr(params))
    return sol

def wrapper(params):
    sol=solve_For_Lattice_Params(params)
    if sol.fluxMultiplication>10:
        print(sol)
    cost=sol.cost
    return cost

def main():
    bounds = list(optimizerBounds_V1.values())

    solve_Async(wrapper,bounds,15*len(bounds),timeOut_Seconds=100_000,disp=True)
    # import skopt
    # vals=skopt.sampler.Sobol().generate(bounds,1000)
    # tool_Parallel_Process(wrapper,vals)
    # print(bounds)
    # x=[0.0228824 , 0.0190287 , 0.00883217, 0.02814639, 0.03345624,
    #    0.01127   , 0.1317694 , 0.37731129, 0.18078798, 0.30192093]
    # wrapper(x)
    # plot_Results(x)
if __name__=='__main__':
    main()







