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
#[1.10677162, 1.00084144, 0.11480408, 0.02832031]
def main():
    #[1.10677162, 1.00084144, 0.11480408, 0.02832031]
    #rpLens,rpLensFirst,rpLensLast,LLens, injectorFactor,rpInjectorFactor,LmCombiner,rpCombiner
    bounds=[(.005,.03),(.02,.04),(.005,.03),(.2,.4),(.5,1.5),(.5,1.5),(.05,.2),(.015,.04)]
    print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=.1,tol=.05) )
if __name__=='__main__':
    main()


# ----------Solution-----------
# injector element spacing optimum configuration: [0.10408747 0.18105588]
#  storage ring tuned params 1 optimum configuration: [0.0188699  0.02033402 0.01       0.26722868]
#  storage ring tuned params 2 optimum configuration: [0.70348256 0.20900569]
#  cost: 0.5766591698329813
# survival: 42.59585161594419