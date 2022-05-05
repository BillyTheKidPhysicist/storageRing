import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *
from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params

def survival_Optimize(bounds: list,tuning: Optional[str],workers: int,magnetErrors: bool):
    def wrapper(args):
        cost=np.inf
        try:
            sol=solve_For_Lattice_Params(args,tuning,magnetErrors)
            cost=sol.cost
            if sol.fluxMultiplication>10:
                print(sol)
        except:
            np.set_printoptions(precision=100)
            print('assert during evaluation on args: ',repr(args))
        return cost
    #rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens
    solve_Async(wrapper,bounds,15*len(bounds),timeOut_Seconds=100000,disp=True,workers=workers)
    # vals=skopt.sampler.Sobol().generate(bounds,1000)
    # tool_Parallel_Process(wrapper,vals)
    # args0 = np.array([0.014691009213257358, 0.0297440095905699, 0.011880244167712408, 0.011336008446642887, 0.32066412285553747])
    # wrapper(args)

def stability_And_Survival_Optimize(bounds,tuning,workers):
    def get_Individual_Costs(args):
        try:
            sol=solve_For_Lattice_Params(args,tuning,False)
            floorPlanCost=sol.floorPlanCost
            swarmCost=sol.swarmCost
        except:
            np.set_printoptions(precision=100)
            print('assert during evaluation on args: ',args)
            assert False
        return swarmCost,floorPlanCost
    def wrapper(args0):
        stabilityTestStep=100e-6
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
        print(swarmCosts)
        variability=np.std(swarmCosts)
        # print(repr(args0),nominalCost,variability)
        return nominalCost+variability
    # solve_Async(wrapper, bounds, 15*len(bounds), timeOut_Seconds=100000, workers=workers)
    # args0 = np.array([0.014691009213257358, 0.0297440095905699, 0.011880244167712408, 0.011336008446642887, 0.32066412285553747])
    # wrapper(args0)

def main():
    bounds = [
        (.005, .03),  # rpLens
        (.02, .04),  # rpLensFirst
        (.005, .02),  # rplensLast
        (.005, .012),  # rpBend
        (.1, .4)  # L_Lens
        # (.125,.2125), #L_Injector
        # (.01,.03), #rpInjector
        # (.075,.2), #LmCombiner
        # (.02,.05)  #rpCombiner
    ]
    # stability_And_Survival_Optimize(bounds,None,8)
    survival_Optimize(bounds,None,10,False)
if __name__=='__main__':
    main()







