import matplotlib.pyplot as plt
from collections.abc import Iterable
import multiprocess as mp
import numpy as np
import scipy.optimize as spo
import skopt
import warnings


SMALL_NUMBER=1e-9

class GradientOptimizer:
    def __init__(self,funcObj,xi,stepSizeInitial,maxIters,momentumFact,disp,Plot,parallel,gradMethod,
                 gradStepSampSize,maxStepSize):
        if isinstance(xi, np.ndarray) == False:
            xi = np.asarray(xi)
            assert len(xi.shape) == 1
        self.funcObj=funcObj
        self.xi=xi
        self.numDim=len(self.xi)
        self.stepSizeInitial=stepSizeInitial
        self.maxStepSize=maxStepSize
        self.maxIters=maxIters
        self.momentumFact=momentumFact
        self.disp=disp
        self.Plot=Plot
        self.gradStepSampSize=gradStepSampSize
        self.gradMethod=gradMethod
        self.parallel=parallel
        self.hFrac=1.0
        self.objValHistList=[]
        self.xHistList=[]

    def _gradient_And_F0(self, X0):
        if self.gradMethod=='central':
            return self._central_Difference_And_F0(X0)
        elif self.gradMethod=='forward':
            return self._forward_Difference_And_F0(X0)
        else:
            raise ValueError
    def _central_Difference_And_F0(self,x0):
        mask = np.eye(len(x0))
        mask = np.repeat(mask, axis=0, repeats=2)
        mask[1::2] *= -1
        x_abArr = mask * self.gradStepSampSize + x0  # x upper and lower values for derivative calc
        if self.parallel == True:
            with mp.Pool() as pool:
                vals_ab = np.asarray(pool.map(self.funcObj, x_abArr))
        else:
            vals_ab = np.asarray([self.funcObj(x) for x in x_abArr])
        vals_ab = vals_ab.reshape(-1, 2)
        meanF0 = np.mean(vals_ab)
        deltaVals = vals_ab[:, 0] - vals_ab[:, 1]
        grad = deltaVals / (2 * self.gradStepSampSize)
        return grad, meanF0
    def _forward_Difference_And_F0(self,x0):
        mask = np.eye(len(x0))
        mask=np.row_stack((np.zeros(self.numDim),mask))
        x_abArr = mask * self.gradStepSampSize + x0  # x upper and lower values for derivative calc
        if self.parallel == True:
            with mp.Pool() as pool:
                vals_ab = np.asarray(pool.map(self.funcObj, x_abArr))
        else:
            vals_ab = np.asarray([self.funcObj(x) for x in x_abArr])
        F0=vals_ab[0]
        deltaVals=vals_ab[1:]-F0
        grad = deltaVals / (2 * self.gradStepSampSize)
        return grad, F0
    def _update_Speed(self,gradUnit,vPrev):
        deltaV = -gradUnit * self.stepSizeInitial
        deltaVDrag=(1.0-self.momentumFact)*vPrev*self.hFrac
        vPrev=vPrev-deltaVDrag
        v=deltaV+vPrev
        if np.linalg.norm(v)<SMALL_NUMBER:
            return np.zeros(self.numDim)
        return v
    def _update_Time_Step(self,v):
        deltaX0=np.linalg.norm(v)*1.0
        self.hFrac=self.maxStepSize/deltaX0 if deltaX0>self.maxStepSize else 1
    def solve_Grad_Descent(self):
        x = self.xi.copy()
        vPrev = np.zeros(self.numDim)
        for i in range(self.maxIters):
            grad, F0 = self._gradient_And_F0(x)
            self.objValHistList.append(F0)
            self.xHistList.append(x)
            grad0 = np.linalg.norm(grad)
            if np.abs(grad0)>SMALL_NUMBER:
                gradUnit = grad / grad0
            else:
                if i==0:
                    print('Gradient is zero on first iteration. Descent cannot proceed')
                    break
                elif self.momentumFact==0.0:
                    print('Gradient is zero, and there is no momentum. Cannot proceed')
                gradUnit=np.zeros(len(grad))
            v = self._update_Speed(gradUnit,vPrev)
            if np.all(np.abs(v)<SMALL_NUMBER):
                warnings.warn("All step sizes are very small, and possibly result largely from numeric noise ")
            vPrev = v.copy()
            if self.disp == True:
                print('-----------',
                    'iteration: '+str(i),
                    'Function value: ' + str(F0),
                    'at parameter space point: '+str(repr(x)),
                      'gradient: '+str(repr(grad)))
            self._update_Time_Step(v)
            deltaX=v*self.hFrac
            x = x + deltaX
        if self.Plot==True:
            plt.plot(self.objValHistList)
            plt.xlabel("Iteration")
            plt.ylabel("Cost (goal is to minimize)")
            plt.show()
        return self.xHistList[np.nanargmin(self.objValHistList)],np.nanmin(self.objValHistList)
def gradient_Descent(funcObj,xi,stepSizeInitial,maxIters,momentumFact=.9,gradMethod='central',disp=False,
                     parallel=True,Plot=False,gradStepSize=5e-6,maxStepSize=np.inf):
    solver=GradientOptimizer(funcObj,xi,stepSizeInitial,maxIters,momentumFact,disp,Plot,parallel,gradMethod,
                             gradStepSize,maxStepSize)
    return solver.solve_Grad_Descent()

def batch_Gradient_Descent(funcObj,bounds,numSamples,stepSizeInitial,maxIters,momentumFact=.9,disp=False,
                           gradMethod='central',maxStepSize=np.inf):
    samples=np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples))
    wrap=lambda x: gradient_Descent(funcObj,x,stepSizeInitial,maxIters,momentumFact=momentumFact,
                                        disp=disp
                                    ,parallel=False,gradMethod=gradMethod,Plot=False,maxStepSize=maxStepSize)
    with mp.Pool() as pool:
        results=pool.map(wrap,samples)
    return results
def global_Gradient_Descent(funcObj,bounds,numSamples,stepSizeInitial,maxIters,gradStepSize,momentumFact=.9,disp=False,
                           gradMethod='central',parallel=True,Plot=False):
    samples = np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples))
    if parallel==True:
        with mp.Pool() as pool:
            vals=np.asarray(pool.map(funcObj,samples))
    else:
        vals=np.asarray([funcObj(x) for x in samples])
    xOptimal=samples[np.argmin(vals)]
    result= gradient_Descent(funcObj,xOptimal,stepSizeInitial,maxIters,momentumFact=momentumFact,
                             disp=disp,parallel=parallel,gradMethod=gradMethod,Plot=Plot,gradStepSize=gradStepSize)
    return result
def _self_Test1():
    def test_Func(X):
        return np.linalg.norm(X) / 2 + np.sin(X[0]) ** 2 * np.sin(X[1] * 3) ** 2
    Xi=[8.0,8.0]
    x,f=gradient_Descent(test_Func,Xi,.05,1000,momentumFact=0.9,gradMethod='central',parallel=False,
                         Plot=True,disp=True,maxStepSize=.15)
    x0=np.asarray([9.586321188035513e-05 ,-0.0026766080917656563])
    f0=0.0013391632778202074
    assert np.all(x0==x)
    assert f==f0