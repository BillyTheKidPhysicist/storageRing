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

    x = [0.02477938, 0.01079024, 0.04059919, 0.010042, 0.07175166, 0.51208528]
    # print(wrapper(x))
    plot_Results(x)
if __name__=='__main__':
    main()


"""
from one run, some bests:

DNA: array([0.015952586890402236, 0.008921903716458061, 0.03273934110970934 ,
       0.009897323234147217, 0.23650357816406012 , 0.3948241353273637  ])
cost: 0.7785364192063521
----------Solution-----------   

DNA: array([0.013296327955384008, 0.009487958867167356, 0.03306096893997447 ,
       0.00817113543659328 , 0.2495124992966306  , 0.3948241353273637  ])
cost: 0.7774117076552659


DNA: array([0.013227196575069485, 0.009671302007693236, 0.03318583393285798 ,
       0.007936646065561416, 0.2556664486411267  , 0.38971531666963743 ])
cost: 0.755503376812761


DNA: array([0.014013686017863835, 0.01                , 0.03284383084253231 ,
       0.008888666611775698, 0.1                 , 0.4                 ])
cost: 0.7195150992618845

"""


"""

version3

BEST MEMBER BELOW
---population member---- 
DNA: array([0.011289371309399825, 0.01                , 0.03221557023564858 ,
       0.007377437528248196, 0.1                 , 0.4                 ])
cost: 0.7267619461070828
----------Solution-----------  
"""







