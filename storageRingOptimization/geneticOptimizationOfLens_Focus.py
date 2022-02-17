from asyncDE import solve_Async
import matplotlib.pyplot as plt
import time
from HalbachLensClass import GeneticLens
import numpy as np
import scipy.optimize as spo
from profilehooks import profile
from temp5 import IPeak_And_Magnification

SMALL_NUMBER=1E-10
class GeneticLens_Analyzer:
    def __init__(self,DNA_List,apLens):
        self.lens = GeneticLens(DNA_List)
        self.apLens=apLens
        assert self.apLens<self.lens.minimum_Radius()
        self._zArr=np.linspace(-self.lens.length*1.5,0.0,50)


apMin=.045
rpCompare=.05
L=.3
numSlicesTotal=15
magnetWidth=.0254
numVarsPerLayer=1
numSlicesSymmetry=int(numSlicesTotal//2 +numSlicesTotal%2)
# @profile()
def construct_Full_Args(args0):
    args=args0.copy()
    args=args.reshape(-1,numVarsPerLayer)
    if numSlicesTotal%2==1: #There is a center lens, ie the number of slices in odd
        argsBack = args[:-1]
    else: #there is no center lens, number of slices are even
        argsBack = args.copy()
    args=np.row_stack((args,np.flip(argsBack,axis=0)))
    assert np.all(args[0] == args[-1]) and len(args) == numSlicesTotal
    return args


def _initial_Focus_And_Cost(args0):
    args0 = np.asarray(args0)
    argsArr = np.asarray(construct_Full_Args(args0))
    assert len(argsArr.shape) == 2
    DNA_List = [{'rp': args[0], 'width': magnetWidth, 'length': L / len(argsArr)} for args in argsArr]
    Lens = GeneticLens(DNA_List)
    assert abs(Lens.length - L) < 1e-12
    IPeak0, m0 = IPeak_And_Magnification(Lens, apMin)
    return IPeak0,m0

Xi=[rpCompare]*numSlicesSymmetry
IPeak0,m0=_initial_Focus_And_Cost(Xi)

def cost_Function(args0):
    args0=np.asarray(args0)
    argsArr=np.asarray(construct_Full_Args(args0))
    assert len(argsArr.shape)==2
    DNA_List=[{'rp':args[0],'width':magnetWidth,'length':L/len(argsArr)} for args in argsArr]
    Lens=GeneticLens(DNA_List)
    assert abs(Lens.length-L)<1e-12
    IPeak,m=IPeak_And_Magnification(Lens,apMin)
    focusCost=IPeak0/IPeak #goal to is shrink this
    magCost=1+abs(m/m0-1) #goal is to keep this the same
    cost=focusCost*magCost
    return cost

bounds=[(apMin+1e-6,.075)]*numSlicesSymmetry
numDimensions=len(bounds)
sol=solve_Async(cost_Function,bounds,15*numDimensions,tol=.01,surrogateMethodProb=.1,workers=8)
print(sol)

"""
BEST MEMBER BELOW
---population member---- 
DNA: [0.0595685  0.05907538 0.05646126 0.05523791 0.05474563 0.05335864
 0.05261596 0.0536713 ]
cost: 0.35990839436527183
"""



# args= [0.0595685,  0.05907538, 0.05646126, 0.05523791, 0.05474563 ,0.05335864 ,0.05261596, 0.0536713 ]
# print(_cost(args))





# print(cost(args))