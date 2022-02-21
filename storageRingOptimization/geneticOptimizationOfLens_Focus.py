from asyncDE import solve_Async
import matplotlib.pyplot as plt
import time
from parallel_Gradient_Descent import gradient_Descent,batch_Gradient_Descent
from HalbachLensClass import GeneticLens
import descent
import numpy as np
import scipy.optimize as spo
from profilehooks import profile
from lensOptimizerHelperFunctions import IPeak_And_Magnification_From_Lens

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
numSlicesTotal=12
magnetWidth=.0254
numVarsPerLayer=3
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


def IPeak_And_Magnification(args0):
    args0 = np.asarray(args0)
    argsArr = np.asarray(construct_Full_Args(args0))
    assert len(argsArr.shape) == 2
    DNA_List = [{'rp': args, 'width': magnetWidth, 'length': L / len(argsArr)} for args in argsArr]
    Lens = GeneticLens(DNA_List)
    IPeak, m = IPeak_And_Magnification_From_Lens(Lens,apMin)
    return IPeak,m
Xcompare=[rpCompare]*numSlicesSymmetry*numVarsPerLayer
IPeak0,m0=IPeak_And_Magnification(Xcompare)
print(IPeak0,m0)
def cost_Function(args0,Print=False):
    IPeak,m=IPeak_And_Magnification(args0)
    if Print==True:
        print(IPeak,m)
    focusCost=IPeak0/IPeak #goal to is shrink this
    magCost=1+10*abs(m/m0-1) #goal is to keep this the same
    cost=focusCost*magCost
    return cost
bounds=[(apMin+1e-6,.075)]*numSlicesSymmetry*numVarsPerLayer
Xi=[x for x in Xcompare]
stepSize=50e-6


def _forward_Difference_And_F0(x0):
    mask=np.eye(len(x0))
    mask=np.row_stack((np.zeros(len(x0)),mask))
    x_abArr=mask*stepSize+x0  # x upper and lower values for derivative calc
    with mp.Pool() as pool:
        vals_ab=np.asarray(pool.map(cost_Function,x_abArr))
    F0=vals_ab[0]
    deltaVals=vals_ab[1:]-F0
    grad=deltaVals/(2*stepSize)
    return F0,grad
_forward_Difference_And_F0(Xi)

opt=descent.adam(stepSize)
sol=opt.minimize(_forward_Difference_And_F0,Xi,maxiter=300)
print(sol)
plt.plot(sol['obj'])
plt.xlabel("iterations")
plt.ylabel('cost')
plt.show()






# print(cost(args))