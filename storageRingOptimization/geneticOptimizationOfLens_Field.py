from asyncDE import solve_Async
import matplotlib.pyplot as plt
import time
from HalbachLensClass import GeneticLens
import numpy as np
import scipy.optimize as spo
from profilehooks import profile

SMALL_NUMBER=1E-10
class GeneticLens_Analyzer:
    def __init__(self,DNA_List,apLens):
        self.lens = GeneticLens(DNA_List)
        self.apLens=apLens
        assert self.apLens<self.lens.minimum_Radius()
        self._zArr=np.linspace(-self.lens.length*1.5,0.0,50)
    def BNormGrad_Along_Line(self,x0, y0):
        assert np.sqrt(x0 ** 2 + y0 ** 2) < self.lens.minimum_Radius()
        coords = np.column_stack((np.ones(len(self._zArr)) * x0, np.ones(len(self._zArr)) * y0, self._zArr))
        valList = np.linalg.norm(self.lens.BNorm_Gradient(coords)[:,:2],axis=1)
        return np.asarray(valList)
    def BNormTrans_Grad_At_Coords(self,coords):
        assert np.all(np.sqrt(coords[:,0] ** 2 + coords[:,1] ** 2) < self.lens.minimum_Radius())
        BNormTrans_Grad_List = np.linalg.norm(self.lens.BNorm_Gradient(coords)[:, :2], axis=1)
        return np.asarray(BNormTrans_Grad_List)
    def field_Integral_And_CenterValue(self,x0, y0):
        valList = self.BNormGrad_Along_Line(x0, y0)
        centerIndex = np.argmin(np.abs(self._zArr - 0))
        centerVal = valList[centerIndex]
        integral = np.trapz(valList, x=self._zArr)
        # plt.plot(self._zArr,valList)
        # plt.show()
        return centerVal,integral
    def _make_Field_Quality_Coords(self):
        rArr = np.linspace(.1*self.apLens, self.apLens, 5)
        numAnglesArr=np.linspace(3,11,len(rArr)).astype(int)
        xList = []
        yList = []

        for r, numAngles in zip(rArr, numAnglesArr):
            angleArr = np.linspace(0.0, np.pi / 6, numAngles)
            for angle in angleArr:
                xList.extend(r * np.cos(angle) * np.ones(len(self._zArr)))
                yList.extend(r * np.sin(angle) * np.ones(len(self._zArr)))
        numLines=sum(numAnglesArr)
        zArrRepeated = np.tile(self._zArr, numLines)
        coords = np.column_stack((xList, yList, zArrRepeated))
        return coords
    def field_Quality(self):
        fieldQualityCoords=self._make_Field_Quality_Coords()
        valsAtCoords=self.BNormTrans_Grad_At_Coords(fieldQualityCoords)
        valsAlongLines=valsAtCoords.reshape(-1,len(self._zArr))
        fieldIntegralArr=np.trapz(valsAlongLines,axis=1)
        xyArr=fieldQualityCoords[::len(self._zArr)][:,:2]
        rArr=np.linalg.norm(xyArr,axis=1)
        assert np.all(rArr<self.apLens+SMALL_NUMBER)==True
        m,b=np.polyfit(rArr,fieldIntegralArr,1)
        relativeResiduals=1e2*(fieldIntegralArr-(m*rArr+b))/fieldIntegralArr
        qualityFactor=np.sqrt(np.sum(relativeResiduals**2))/len(relativeResiduals)
        return qualityFactor
    def field_Gradient_Strength(self):
        xArr = np.linspace(-self.apLens, self.apLens, 10)
        coords = np.asarray(np.meshgrid(xArr, xArr)).T.reshape(-1, 2)
        coords = coords[np.linalg.norm(coords, axis=1) < .9 * self.apLens]
        coords=np.column_stack((coords,np.zeros(len(coords))))
        BGradArr=self.lens.BNorm_Gradient(coords)[:,:2]
        return np.linalg.norm(BGradArr)
#todo: dumb method with center layer
apMin=.045
rpCompare=.05
L=.1524
numSlicesTotal=6
centerLayer=False
magnetWidth=.0254
numVarsPerLayer=1
if centerLayer==False:
    numSlicesSymmetry=int(numSlicesTotal//2)
else:
    numSlicesSymmetry = int((numSlicesTotal+1) // 2)
# @profile()
def construct_Full_Args(args0):
    args=args0.copy()
    args=args.reshape(-1,numVarsPerLayer)
    if centerLayer==True:
        argsBack = args[:-1]
    else: #there is no center lens
        argsBack = args.copy()
    args=np.row_stack((args,np.flip(argsBack,axis=0)))
    assert np.all(args[0] == args[-1]) and len(args) == numSlicesTotal
    return args
def _cost(args0):
    args0=np.asarray(args0)
    argsArr=np.asarray(construct_Full_Args(args0))
    assert len(argsArr.shape)==2
    DNA_List=[]
    for args in argsArr:
        DNA_List.append({'rp':args[0],'width':magnetWidth,'length':L/len(argsArr)})
    Lens=GeneticLens_Analyzer(DNA_List,apMin)
    assert abs(Lens.lens.length-L)<1e-12
    quality=Lens.field_Quality()
    assert quality>0.0
    fieldStrength=Lens.field_Gradient_Strength()
    cost=1e2*quality/fieldStrength
    return cost

Xi=[rpCompare]*numSlicesSymmetry
cost0=_cost(Xi)
# print('------')
# Xi=[rpCompare+.02]*numSlicesSymmetry
# cost0=_cost(Xi)
# def cost(args):
#     cost=_cost(args)/cost0
#     return cost

# bounds=[(apMin+1e-6,.075)]*numSlicesSymmetry
# numDimensions=len(bounds)
# sol=solve_Async(cost,bounds,15*numDimensions,tol=.01,surrogateMethodProb=.1,workers=8)
# print(sol)
# args=[0.06433653, 0.06447622, 0.05795036, 0.05538298, 0.05306453 ,0.05131647]
