import os
os.environ['OPENBLAS_NUM_THREADS']='1'

import numpy as np

from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
from parallel_Gradient_Descent import global_Gradient_Descent,gradient_Descent
from profilehooks import profile
def wrapper(args):
    try:
        tuning=None
        cost=solve_For_Lattice_Params(args,tuning).cost
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
    # print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=.1,timeOut_Seconds=1e12,workers=8) )
    solve_Async(wrapper,bounds,15,timeOut_Seconds=100000,workers=8)
    # gradient_Descent(wrapper,Xi,25e-6,100,momentumFact=0.0,frictionFact=0.0,gradStepSize=25e-6,disp=True)
    # wrapper(X)
if __name__=='__main__':
    main()
'''

----------Solution-----------   
injector element spacing optimum configuration: nan
storage ring tuned params 1 optimum configuration: [0.01003313 0.01843031 0.01834712 0.00989281 0.18579258]
storage ring tuned params 2 optimum configuration: [0.37154555 0.50950135]
cost: 0.5311869592905618
percent max flux multiplication: 46.88130407094382
scipy message: Optimization terminated successfully.

'''