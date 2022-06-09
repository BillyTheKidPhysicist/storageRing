import numpy as np

from latticeElements.utilities import ElementTooShortError
from latticeModels import make_Ring_And_Injector_Version3, RingGeometryError, InjectorGeometryError
from latticeModels_Parameters import optimizerBounds_V1_3
from storageRingModeler import StorageRingModel, Solution


def plot_Results(params):
    PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(params)
    optimizer = StorageRingModel(PTL_Ring, PTL_Injector)
    optimizer.show_Floor_Plan_And_Trajectories(True)


def invalid_Solution(params):
    sol = Solution(params, None, None)
    return sol


def solve(params: tuple[float, ...]) -> Solution:
    PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(params)
    energyConservation = False
    collisionDynamics = True
    optimizer = StorageRingModel(PTL_Ring, PTL_Injector, collisionDynamics=collisionDynamics)
    cost, fluxMultiplication = optimizer.mode_Match(energyConservation, floorPlanCostCutoff=.05)
    sol = Solution(params, fluxMultiplication, cost)
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
    # from asyncDE import solve_Async
    bounds = np.array(list(optimizerBounds_V1_3.values()))

    # solve_Async(wrapper,bounds,15*len(bounds),timeOut_Seconds=100_000,disp=True,workers=10,saveData='optimizerProgress')
    x = np.array([0.023801057453580743, 0.010865155679545636, 0.039901298278481497,
                  0.010145870717811905, 0.060600536295301044, 0.4895337924060436])
    print(wrapper(x))

    raise Exception("issue with repeatability with collision physics")

    x = np.array([0.023801057453580743, 0.010865155679545636, 0.039901298278481497,
                  0.010145870717811905, 0.060600536295301044, 0.4895337924060436])
    print(wrapper(x))

    x = np.array([0.023801057453580743, 0.010865155679545636, 0.039901298278481497,
                  0.010145870717811905, 0.060600536295301044, 0.4895337924060436])
    print(wrapper(x))
    # from helperTools import tool_Parallel_Process
    # TArr=np.logspace(-4,np.log10(20e-3),20)
    # res= tool_Parallel_Process(func,TArr)
    # print(res)
    # from helperTools import plt
    # plt.plot(TArr,res)
    # plt.show()
    # plot_Results(x)


if __name__ == '__main__':
    main()

"""
------ITERATIONS:  5130
POPULATION VARIABILITY: [0.005667280347543179   0.052391376269014      0.0031954290081018908
 0.04550477868952548    0.009997735630440398   0.00012139473877598557]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.014915041197354493, 0.01097586851491024 , 0.04                ,
       0.010064624087585298, 0.06405240440024305 , 0.5                 ])
cost: 0.8034745617458244



------ITERATIONS:  5760
POPULATION VARIABILITY: [0.0070015762077465125 0.023877691379240628  0.004593778576254029
 0.004038869074356368  0.007734563201444062  0.007768864926388778 ]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.023801057453580743, 0.010865155679545636, 0.039901298278481497,
       0.010145870717811905, 0.060600536295301044, 0.4895337924060436  ])
cost: 0.7487877460123314

array([0.021301449835450233, 0.011000000000000001, 0.03734118729103023 ,
       0.008864674985895155, 0.5                 , 0.48253992279904434 ,
       0.05                , 0.03                , 0.11540376013505205 ,
       0.014076292221880626, 0.24546460746656512 , 0.021684261436978172,
       0.03                , 0.3                 , 0.1390756996742596  ,
       0.23383634865350428 ])


"""
