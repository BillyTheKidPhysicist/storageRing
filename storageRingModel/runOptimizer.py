import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *
from asyncDE import solve_Async
from temp5 import solve_For_Lattice_Params

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
        (.005, .03),  # rpLens1
        (.005, .02),  # rplens2
        (.005, .012),  # rpBend
        (.1, .4)  # L_Lens
    ]
    # stability_And_Survival_Optimize(bounds,None,8)
    survival_Optimize(bounds,10)
if __name__=='__main__':
    main()



"""

rpLensFirst=.02

------ITERATIONS:  4200
POPULATION VARIABILITY: [0.27248292 0.10177719 0.06158428 0.11931989 0.10092491]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.03      , 0.02875096, 0.01293413, 0.00881959, 0.4       ])
cost: 0.8442073369525684

        (.005, .03),  # rpLens
        (.02, .04),  # rpLensFirst
        (.005, .02),  # rplensLast
        (.005, .012),  # rpBend
        (.1, .4)  # L_Lens
        # (.125,.2125), #L_Injector
        # (.01,.03), #rpInjector
        # (.075,.2), #LmCombiner
        # (.02,.05)  #rpCombiner


"""



