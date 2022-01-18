import os

import numpy as np

os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
def wrapper(args):
    try:
        cost=solve_For_Lattice_Params(args,parallel=False).cost
    except:
        np.set_printoptions(precision=100)
        print('assert during evaluation on args: ',args)
        assert False
    return cost
def main():
    #rpLens,rpLensFirst,rpLensLast,LLens, injectorFactor,rpInjectorFactor,LmCombiner,rpCombiner
    bounds=[(.005,.03),(.02,.04),(.005,.03),(.2,.4),(.5,1.5),(.5,1.5),(.05,.2),(.015,.04)]
    print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=.5,timeOut_Seconds=1e12) )
if __name__=='__main__':
    main()


# ----------Solution-----------
# injector element spacing optimum configuration: [0.39952416 0.30446732]
#  storage ring tuned params 1 optimum configuration: [0.01195178 0.02872478 0.01391224 0.3030036  1.20776536 1.3854315
#  0.08843782 0.02597373]
#  storage ring tuned params 2 optimum configuration: [0.33064234 0.7194921 ]
#  cost: 0.3421403981620351
# survival: 65.78596018379649
# scipy message: Optimization terminated successfully.