import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
def wrapper(args):
    cost=solve_For_Lattice_Params(args,parallel=False).cost
    return cost

def main():
    # rpLens,rpLensFirst,Lm,LLens
    bounds=[(.005,.03),(.02,.04),(.01,.03),(.2,.4)]
    print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=.1,tol=.05) )
if __name__=='__main__':
    main()