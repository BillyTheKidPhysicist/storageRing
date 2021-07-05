import scipy.optimize as spo
import numpy as np

global limits
limits=None
global object
object=None

def solve():
    def minCost(args):
        if object.enforce_Geometry(args)==False:  #if configuration violates geometry
            return np.inf
        object.update_spheres(args)
        cost=object.cost(Print=False)
        return cost
    sol=spo.differential_evolution(minCost,limits,disp=False,popsize=32*3,polish=False,mutation=.5,maxiter=1000,workers=1)
    return sol