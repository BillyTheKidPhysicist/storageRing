import matplotlib.pyplot as plt
from collections.abc import Iterable
import multiprocess as mp
import numpy as np
import scipy.optimize as spo
import skopt




class GradientOptimizer:
    def __init__(self,funcObj,xi,stepSize,maxIters,momentumFact,frictionFact,disp,parallel,gradStepSampSize=1e-6):
        if isinstance(xi, np.ndarray) == False:
            xi = np.asarray(xi)
            assert len(xi.shape) == 1
        assert stepSize>10*gradStepSampSize
        self.funcObj=funcObj
        self.xi=xi
        self.numDim=len(self.xi)
        self.stepSize=stepSize
        self.maxIters=maxIters
        self.momentumFact=momentumFact
        self.frictionFact=frictionFact
        self.disp=disp
        self.gradStepSampSize=gradStepSampSize
        self.parallel=parallel
        self.objValHistList=[]
        self.xHistList=[]

    def _gradient_And_F0_Parallel(self, x0):
        mask = np.eye(len(x0))
        mask = np.repeat(mask, axis=0, repeats=2)
        mask[1::2] *= -1
        x_abArr = mask * self.gradStepSampSize + x0 #x upper and lower values for derivative calc
        if self.parallel==True:
            with mp.Pool() as pool:
                vals_ab = np.asarray(pool.map(self.funcObj, x_abArr))
        else:
            vals_ab = np.asarray([self.funcObj(x) for x in x_abArr])
        vals_ab = vals_ab.reshape(-1, 2)
        meanF0 = np.mean(vals_ab)
        deltaVals = vals_ab[:, 0] - vals_ab[:, 1]
        grad = deltaVals / (2 * self.gradStepSampSize)
        return grad, meanF0
    def _update_Speed(self,gradUnit,deltaXPrev):
        deltaX = -gradUnit * self.stepSize + deltaXPrev * self.momentumFact
        deltaX = deltaX - self.frictionFact * (deltaX / np.linalg.norm(deltaX)) * np.linalg.norm(deltaX) ** 2
        return deltaX
    def solve_Grad_Descent(self):
        x = self.xi.copy()
        deltaXPrev = np.zeros(self.numDim)
        for i in range(self.maxIters):
            grad, F0 = self._gradient_And_F0_Parallel(x)
            self.objValHistList.append(F0)
            self.xHistList.append(x)
            grad0 = np.linalg.norm(grad)
            gradUnit = grad / grad0
            deltaX = self._update_Speed(gradUnit,deltaXPrev)
            deltaXPrev = deltaX.copy()
            if self.disp == True:
                print(i, F0, repr(x), repr(deltaX))
            x = x + deltaX
        # plt.plot(self.objValHistList)
        # plt.xlabel("Iteration")
        # plt.ylabel("Cost (goal is to minimize)")
        # plt.show()
        return self.xHistList[np.argmin(self.objValHistList)],np.min(self.objValHistList)
def solve_Gradient_Descent(funcObj,xi,stepSize,maxIters,momentumFact=.9,frictionFact=.01,disp=False,parallel=True):
    solver=GradientOptimizer(funcObj,xi,stepSize,maxIters,momentumFact,frictionFact,disp,parallel)
    return solver.solve_Grad_Descent()

def batch_Gradient_Descent(funcObj,bounds,numSamples,stepSize,maxIters,momentumFact=.9,frictionFact=.01,disp=False):
    samples=np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples))
    wrap=lambda x: solve_Gradient_Descent(funcObj,x,stepSize,maxIters,momentumFact=momentumFact,
                                          frictionFact=frictionFact,disp=disp
                                    ,parallel=False)
    with mp.Pool() as pool:
        results=pool.map(wrap,samples)
    return results


def _self_Test1():
    def test_Func(X):
        return np.linalg.norm(X) / 2 + np.sin(X[0]) ** 2 * np.sin(X[1] * 3) ** 2
    Xi=[8.0,8.0]

    x,f=solve_Gradient_Descent(test_Func,Xi,.05,300,.9,.01,False)
    x0=np.asarray([-0.007751966334679762 ,0.004129096462041654])
    f0=0.004391547110012801
    assert np.all(x0==x)
    assert f==f0

