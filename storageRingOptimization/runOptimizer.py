import os
os.environ['OPENBLAS_NUM_THREADS']='1'

import numpy as np

from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
from profilehooks import profile
def wrapper(args):
    try:
        cost=solve_For_Lattice_Params(args).cost
    except:
        np.set_printoptions(precision=100)
        print('assert during evaluation on args: ',args)
        assert False
    return cost
def main():
    bounds=[
        (.005,.03), #rpLens
        (.01,.03), #rpLensFirst
        (.005,.02), #rplensLast
        (.0075, .0125),  # rpBend
        (.1,.4) #L_Lens
        # (.125,.2125), #L_Injector
        # (.01,.03), #rpInjector
        # (.075,.2), #LmCombiner
        # (.02,.05)  #rpCombiner
        ]
    #rpLens,rpLensFirst,rpLensLast,LLens, injectorFactor,rpInjectorFactor,LmCombiner,rpCombiner
    print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=.1,timeOut_Seconds=1e12) )
if __name__=='__main__':
    main()
