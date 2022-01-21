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
    print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=.5,timeOut_Seconds=1e12,workers=8) )
if __name__=='__main__':
    main()


'''
----------Solution-----------   
injector element spacing optimum configuration: [0.10970817 0.13830605]
 storage ring tuned params 1 optimum configuration: [0.02000886 0.03119919 0.01324413 0.36503544 1.15110092 1.00777664
 0.12208005 0.03236198]
 storage ring tuned params 2 optimum configuration: [0.66706644 0.78041371]
 cost: 0.5396119504745164
survival: 46.03880495254836
scipy message: Optimization terminated successfully.
----------------------------
'''