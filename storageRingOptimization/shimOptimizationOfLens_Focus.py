from asyncDE import solve_Async
import matplotlib.pyplot as plt
import multiprocess as mp
import time
from parallel_Gradient_Descent import gradient_Descent,global_Gradient_Descent
from geneticLensClass import GeneticLens
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


uniqueSphereEachEnd=False
rp=.05
L0=.23
magnetWidth=.0254
numSphere=1
assert not (numSphere%2!=1 and uniqueSphereEachEnd==True)
sphereEndSymmFact=2 if uniqueSphereEachEnd==True else 1
argsPerSphere=5
sphereRadius=.0254/2.0

DNA_List0=[{'component':'layer','rp':(rp,),'width':magnetWidth,'length':L0}]

boundsLens=np.asarray([(L0-rp,L0+rp)])
#r, phi, z (L+deltaL), theta,psi
boundsShim=[(rp,rp*2),(-np.pi/6,np.pi/6),(0.0,2*rp),(0.0,np.pi),(0.0,2*np.pi)]*sphereEndSymmFact*numSphere
bounds=np.asarray(boundsShim)
if uniqueSphereEachEnd==True:
    bounds[2+argsPerSphere::2*argsPerSphere]=-np.flip(bounds[2+argsPerSphere::2*argsPerSphere],axis=1)
bounds=np.row_stack((boundsLens,boundsShim))


def build_Lens_From_Args(args0):
    DNA_List=DNA_List0.copy()
    if args0 is not None:
        L=args0[0]
        DNA_List[0]['length']=L
        args=args0[1:].copy()
        assert len(args) == numSphere * argsPerSphere*sphereEndSymmFact
        args=args.reshape(-1,argsPerSphere)
        for arg in args:
            z=arg[2]+L
            sphereDict={'component':'shim','radius':sphereRadius,'r':arg[0],'phi':arg[1],'z':z,'theta':arg[3],'psi':arg[4]}
            DNA_List.append(sphereDict)
    lens=GeneticLens(DNA_List,sphereEndSymmetry=not uniqueSphereEachEnd)
    return lens
lens0=build_Lens_From_Args(None)
IPeak0, m0 = IPeak_And_Magnification_From_Lens(lens0,rp*.95)
print(IPeak0,m0) # 17.179046887321306 0.9741538042693159
def cost_Function(args,Print=False):
    lens=build_Lens_From_Args(args)
    if lens.is_Geometry_Valid()==False:
        return np.inf
    IPeak, m = IPeak_And_Magnification_From_Lens(lens,.95*rp)
    if Print==True:
        print(IPeak,m) #90.16348342043328 0.8080038559411498
    focusCost=IPeak0/IPeak #goal to is shrink this
    magCost=1+5.0*abs(m/m0-1) #goal is to keep this the same
    cost=focusCost*magCost
    return cost
# args=np.array([ 0.05863646, -0.05521788,  0.12974828,  1.4257796 ,  2.28718574,
#         0.05750753,  0.52228188,  0.12842825,  1.77799039,  5.31679471])
# cost_Function(args,Print=True) #62.65321350060982 0.9365678234490117
xOptimal=solve_Async(cost_Function,bounds,15*len(bounds),timeOut_Seconds=3600*4,workers=10).DNA

# sol=global_Gradient_Descent(cost_Function,bounds,500,300e-6,300,100e-6,descentMethod='adam',Plot=True)
# print(sol)
# Xi=np.array([0.06185511, 0.52359878, 0.12792018, 1.68455923, 0.08704761])
# print(gradient_Descent(cost_Function,Xi,50e-6,50,gradStepSize=10e-6,descentMethod='adam',Plot=True))
'''
1 sphere
has z symmetry

POPULATION VARIABILITY: [0.00512099 0.00334752 0.00279618 0.01889909 0.00997273]
BEST MEMBER BELOW
---population member---- 
DNA: array([ 0.06171082, -0.50545746,  0.12776681,  1.44541379,  2.55378568])
cost: 0.7060276662605259
'''

'''
2 sphere
each has z symmetry

POPULATION VARIABILITY: [0.02066468 0.03704525 0.00649431 0.03660749 0.02620414 0.00505213
 0.00366778 0.00999073 0.04695054 0.01486507]
BEST MEMBER BELOW
---population member---- 
DNA: array([ 0.05863646, -0.05521788,  0.12974828,  1.4257796 ,  2.28718574,
        0.05750753,  0.52228188,  0.12842825,  1.77799039,  5.31679471])
cost: 0.35301654421170253
'''

'''
2 sphere
one at each end, unique


'''