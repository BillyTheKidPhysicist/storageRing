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
    # rpLens,rpLensFirst,Lm,LLens
    # bounds=[(.005,.03),(.02,.04),(.01,.03),(.2,.4)]
    # print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=.1,tol=.05) )
    XRing = [0.0188699 , 0.02033402 ,0.01    ,   0.26722868]
    wrapper(XRing)
if __name__=='__main__':
    main()


# ----------Solution-----------
# injector element spacing optimum configuration: [0.10408747 0.18105588]
#  storage ring tuned params 1 optimum configuration: [0.0188699  0.02033402 0.01       0.26722868]
#  storage ring tuned params 2 optimum configuration: [0.70348256 0.20900569]
#  cost: 0.5766591698329813
# survival: 42.59585161594419