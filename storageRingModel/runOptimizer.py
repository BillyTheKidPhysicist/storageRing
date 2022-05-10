import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *
from asyncDE import solve_Async
from latticeModels_Constants import constants_Version1
from injectionOptimizer import surrogateParams
from storageRingOptimizer import LatticeOptimizer,Solution
from ParticleTracerLatticeClass import ParticleTracerLattice
from elementPT import ElementTooShortError
from latticeModels import make_Ring_And_Injector_Version1,RingGeometryError,InjectorGeometryError


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
    swarmCost, floorPlanCost,swarmTraced = optimizer.mode_Match_Cost(knobParams, False, True, rejectIllegalFloorPlan=False,
                                                         rejectUnstable=False, returnFullResults=True)
    sol.cost = swarmCost + floorPlanCost
    sol.floorPlanCost = floorPlanCost
    sol.swarmCost = swarmCost
    sol.fluxMultiplication = optimizer.compute_Flux_Multiplication(swarmTraced)

    return sol

def solve_For_Lattice_Params(params):
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

def survival_Optimize(bounds: list,workers: int):
    def wrapper(params):
        try:
            sol=solve_For_Lattice_Params(params)
            cost=sol.cost
            if sol.fluxMultiplication>10:
                print(sol)
        except:
            np.set_printoptions(precision=100)
            print('unhandled exception on args: ',repr(params))
            cost=np.inf
        return cost
    #rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens
    solve_Async(wrapper,bounds,15*len(bounds),timeOut_Seconds=100000,disp=True,workers=workers)
    # vals=skopt.sampler.Sobol().generate(bounds,1000)
    # tool_Parallel_Process(wrapper,vals)
    # args0 = np.array([0.014691009213257358, 0.0297440095905699, 0.011880244167712408, 0.011336008446642887, 0.32066412285553747])
    # wrapper(args)


def main():
    bounds = [
        (.005, .03),  # rpLens3_4
        (.005, surrogateParams['rpLens1']),  # rpLens1
        (.005, .03),  # rplens2
        (.005, constants_Version1["bendingApMax"]*1.1),  # rpBend
        (.1, .4)  # L_Lens
    ]
    # stability_And_Survival_Optimize(bounds,None,8)
    survival_Optimize(bounds,10)
if __name__=='__main__':
    main()







