import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
def wrapper(args):
    cost=solve_For_Lattice_Params(args,parallel=False).cost
    return cost

def main():
    #rpLens,rpLensFirst,Lm,LLens
    bounds=[(.005,.03),(.02,.04),(.01,.03),(.2,.4)]
    print(solve_Async(wrapper,bounds,3600*24,15*len(bounds),surrogateMethodProb=.05) )
if __name__=='__main__':
    main()
'''
----------Solution-----------   
injector element spacing optimum configuration: [0.06570139 0.31885332]
 storage ring tuned params 1 optimum configuration: [0.0170735  0.02123114 0.01620267 0.26168577]
 storage ring tuned params 2 optimum configuration: [0.27015212 0.79327379]
 cost: 0.8224469763025926
survival: 17.755302369740743
'''