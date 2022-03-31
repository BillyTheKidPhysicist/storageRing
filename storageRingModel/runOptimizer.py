import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *
from profilehooks import profile
from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
from parallel_Gradient_Descent import global_Gradient_Descent,gradient_Descent
def survival_Optimize(bounds: list,tuning: Optional[str],workers: int,magnetErrors: bool):
    def wrapper(args):
        try:
            sol=solve_For_Lattice_Params(args,tuning,magnetErrors)
            if sol.fluxMultiplicationPercent>1.0:
                print(sol)
        except:
            np.set_printoptions(precision=100)
            print('assert during evaluation on args: ',repr(args))
            assert False
        return sol.cost
    #rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens
    solve_Async(wrapper,bounds,15*len(bounds),timeOut_Seconds=100000,disp=True,workers=workers)
    # args0 = np.array([0.02072277, 0.03239042, 0.01302925, 0.008     , 0.3932113 ])
    # wrapper(args0)
def stability_And_Survival_Optimize(bounds,tuning,workers):
    def get_Individual_Costs(args):
        try:
            sol=solve_For_Lattice_Params(args,tuning)
            floorPlanCost=sol.floorPlanCost
            swarmCost=sol.swarmCost
        except:
            np.set_printoptions(precision=100)
            print('assert during evaluation on args: ',args)
            assert False
        return swarmCost,floorPlanCost
    def wrapper(args0):
        stabilityTestStep=500e-6
        args0=np.asarray(args0)
        swarmCost0,floorPlanCost0=get_Individual_Costs(args0)
        nominalCost=swarmCost0+floorPlanCost0
        if nominalCost>.9: #don't bother exploring stability
            return nominalCost
        mask = np.eye(len(args0))
        mask = np.repeat(mask, axis=0, repeats=2)
        mask[1::2] *= -1
        testArr = mask * stabilityTestStep + args0
        results=np.asarray([get_Individual_Costs(arg) for arg in testArr])
        swarmCosts,floorPlanCosts=results[:,0],results[:,1]
        variability=np.std(swarmCosts)
        # print(repr(args0),nominalCost,variability)
        return nominalCost+variability
    solve_Async(wrapper, bounds, 15*len(bounds), timeOut_Seconds=100000, workers=workers)
    # args0 = np.asarray([0.01505976, 0.0388815, 0.0191172, 0.01054996, 0.1])
    # wrapper(args0)
def main():
    bounds = [
        (.005, .03),  # rpLens
        (.02, .04),  # rpLensFirst
        (.005, .02),  # rplensLast
        (.008, .012),  # rpBend
        (.1, .4)  # L_Lens
        # (.125,.2125), #L_Injector
        # (.01,.03), #rpInjector
        # (.075,.2), #LmCombiner
        # (.02,.05)  #rpCombiner
    ]
    # stability_And_Survival_Optimize(bounds,None,8)
    survival_Optimize(bounds,None,8,True)
if __name__=='__main__':
    main()



"""----------Solution-----------  

ideal case
 
------ITERATIONS:  4350
POPULATION VARIABILITY: [0.03897526 0.02361456 0.0131504  0.00323601 0.02216822]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.02072277, 0.03239042, 0.01302925, 0.008     , 0.3932113 ])
cost: 0.5838620992518562"""





