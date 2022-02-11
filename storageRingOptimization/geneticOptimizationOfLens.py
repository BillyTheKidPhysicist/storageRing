import matplotlib.pyplot as plt
import multiprocess as mp
from HalbachLensClass import GeneticLens
import numpy as np
import scipy.optimize as spo
from profilehooks import profile


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
    def field_Integral_And_CenterValue(self,x0, y0):
        valList = self.BNormGrad_Along_Line(x0, y0)
        centerIndex = np.argmin(np.abs(self._zArr - 0))
        centerVal = valList[centerIndex]
        integral = np.trapz(valList, x=self._zArr)
        # plt.plot(self._zArr,valList)
        # plt.show()
        return centerVal,integral
    def field_Quality(self):
        rArr = np.linspace(.1*self.apLens, self.apLens, 5)
        numAnglesArr=np.linspace(3,11,len(rArr)).astype(int)
        # for numAngles in numAnglesArr:
        fieldIntegralList=[]
        rList=[]
        for r,numAngles in zip(rArr,numAnglesArr):
            angleArr=np.linspace(0.0,np.pi/6,numAngles)
            for angle in angleArr:
                x=r * np.cos(angle)
                y=r*np.sin(angle)
                rList.append(r)
                centerVal,fieldIntegral=self.field_Integral_And_CenterValue(x,y)
                fieldIntegralList.append(fieldIntegral)
        rArr=np.asarray(rList)
        fieldIntegralArr=np.asarray(fieldIntegralList)
        m,b=np.polyfit(rList,fieldIntegralList,1)
        relativeResiduals=1e2*(fieldIntegralArr-(m*rArr+b))/fieldIntegralArr
        qualityFactor=np.sqrt(np.sum(relativeResiduals**2))/len(relativeResiduals)
        return qualityFactor

        # plt.scatter(rArr,fieldIntegralArr)
        # plt.plot(rArr,rArr*m+b)
        # plt.show()
        # plt.plot(rArr,relativeResiduals)
        # plt.show()
    def field_Gradient_Strength(self):
        xArr = np.linspace(-self.apLens, self.apLens, 10)
        coords = np.asarray(np.meshgrid(xArr, xArr)).T.reshape(-1, 2)
        coords = coords[np.linalg.norm(coords, axis=1) < .9 * self.apLens]
        coords=np.column_stack((coords,np.zeros(len(coords))))
        BGradArr=self.lens.BNorm_Gradient(coords)[:,:2]
        return np.linalg.norm(BGradArr)

apMin=.045
L=.1524
numSlices=21 #needs to be odd number because of how symmetry is handled
assert numSlices%2==1
# @profile()
def _cost(args0):
    args0=list(args0)
    args=args0[1:]
    args.reverse()
    args.extend(args0)
    assert args[0]==args[-1] and len(args)==numSlices
    DNA_List=[]
    for arg in args:
        DNA_List.append({'rp':arg,'width':.0254,'length':L/len(args)})
    Lens=GeneticLens_Analyzer(DNA_List,apMin)
    quality=Lens.field_Quality()
    assert quality>0.0
    fieldStrength=Lens.field_Gradient_Strength()
    cost=1e2*quality/fieldStrength
    return cost

# Xi=np.array([0.05061419, 0.05315874, 0.05806932, 0.06425857])
# _cost(Xi)
Xi=[apMin*1.1]*(numSlices//2 + 1)
cost0=_cost(Xi)
def cost(args):
    cost=_cost(args)/cost0
    print(args,cost)
    return cost


bounds=[(apMin+1e-6,.075)]*(numSlices//2 + 1)
sol=spo.differential_evolution(cost,bounds,polish=False,workers=9,disp=True,maxiter=30)
print(sol)

# X0=np.array([0.0464599 , 0.04983626, 0.05269872, 0.05300803, 0.05456089,
#        0.05668497, 0.05601397, 0.06005384, 0.06131109, 0.06931891,
#        0.06288761])
# sol=spo.minimize(cost,X0,bounds=bounds,method='Nelder-Mead')
# print(sol)