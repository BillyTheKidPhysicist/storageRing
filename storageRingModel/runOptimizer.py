import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *
from asyncDE import solve_Async
from temp5 import solve_For_Lattice_Params
from latticeModels_Constants import constants_Version1
from injectionOptimizer import surrogateParams

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







