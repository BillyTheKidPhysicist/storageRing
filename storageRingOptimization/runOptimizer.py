from asyncDE import solve_Async
from optimizerHelperFunctions import solve_For_Lattice_Params
def wrapper(args):
    cost=solve_For_Lattice_Params(args,parallel=False).cost
    return cost

def main():
    #rpLens,rpLensFirst,Lm,LLens,injectorFactor,rpInjectorFactor,LmCombiner
    bounds=[(.005,.03),(.02,.04),(.01,.03),(.2,.4),(.25,1.25),(.25,1.25),(.1,.2)]
    print(solve_Async(wrapper,bounds,3600*24,15*len(bounds),surrogateMethodProb=.05) )
if __name__=='__main__':
    main()