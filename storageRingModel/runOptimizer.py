import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *

from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
from parallel_Gradient_Descent import global_Gradient_Descent,gradient_Descent
def survival_Optimize(bounds,tuning,workers):
    def wrapper(args):
        try:
            sol=solve_For_Lattice_Params(args,tuning)
            if sol.fluxMultiplicationPercent>1.0:
                print(sol)
        except:
            np.set_printoptions(precision=100)
            print('assert during evaluation on args: ',repr(args))
            assert False
        return sol.cost
    #rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens
    solve_Async(wrapper,bounds,15*len(bounds),timeOut_Seconds=100000,disp=True)
    # args0 = np.asarray([0.01505976, 0.0388815  ,0.0191172,  0.01054996, 0.1       ])
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
    survival_Optimize(bounds,None,8)
if __name__=='__main__':
    main()

"""
# rpLens, rpLensFirst,rplensLast, rpBend , L_Lens
----------Solution-----------   
injector element spacing optimum configuration: None
storage ring tuned params 1 optimum configuration: array([0.02126719, 0.03330273, 0.01353354, 0.008943  , 0.39831075])
storage ring tuned params 2 optimum configuration: None
cost: 0.5982274404677516
percent max flux multiplication: 40.30475837756654
scipy message: None

"""


# many particles:

"""----------Solution-----------   
injector element spacing optimum configuration: None
storage ring tuned params 1 optimum configuration: array([0.02126719, 0.03330273, 0.01353354, 0.008943  , 0.39831075])
storage ring tuned params 2 optimum configuration: None
cost: 0.5911770700478259
percent max flux multiplication: 41.00979780702859
scipy message: None
----------------------------"""

# many particles, method of moments




