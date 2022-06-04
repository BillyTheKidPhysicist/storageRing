import sys
import matplotlib.pyplot as plt
from collections.abc import Iterable
from asyncDE import solve_Async
import multiprocess as mp
import numpy as np
import descent
import scipy.optimize as spo
import skopt
import warnings

SMALL_NUMBER = 1e-9


class GradientOptimizer:
    def __init__(self, funcObj, xi, stepSizeInitial, maxIters, momentumFact, disp, Plot, parallel, gradMethod,
                 gradStepSampSize, maxStepSize, descentMethod):
        if isinstance(xi, np.ndarray) == False:
            xi = np.asarray(xi)
            assert len(xi.shape) == 1
        assert descentMethod in ('momentum', 'adam')
        assert gradMethod in ('forward', 'central')
        self.funcObj = funcObj
        self.xi = xi
        self.numDim = len(self.xi)
        self.stepSizeInitial = stepSizeInitial
        self.maxStepSize = maxStepSize
        self.maxIters = maxIters
        self.momentumFact = momentumFact
        self.disp = disp
        self.Plot = Plot
        self.gradStepSampSize = gradStepSampSize
        self.gradMethod = gradMethod
        self.parallel = parallel
        self.hFrac = 1.0
        self.descentMethod = descentMethod
        self.objValHistList = []
        self.xHistList = []

    def _gradient_And_F0(self, X0):
        if self.gradMethod == 'central':
            return self._central_Difference_And_F0(X0)
        elif self.gradMethod == 'forward':
            return self._forward_Difference_And_F0(X0)
        else:
            raise ValueError

    def _central_Difference_And_F0(self, x0):
        assert len(x0) == self.numDim
        mask = np.eye(len(x0))
        mask = np.repeat(mask, axis=0, repeats=2)
        mask[1::2] *= -1
        mask = np.row_stack((np.zeros(self.numDim), mask))
        x_abArr = mask * self.gradStepSampSize + x0  # x upper and lower values for derivative calc
        if self.parallel == True:
            with mp.Pool() as pool:
                vals_ab = np.asarray(pool.map(self.funcObj, x_abArr))
        else:
            vals_ab = np.asarray([self.funcObj(x) for x in x_abArr])
        assert len(vals_ab.shape) == 1 and len(vals_ab) == 2 * self.numDim + 1
        F0 = vals_ab[0]
        vals_ab = vals_ab[1:].reshape(-1, 2)
        gradForward = (vals_ab[:, 0] - F0) / self.gradStepSampSize
        gradBackward = (F0 - vals_ab[:, 1]) / self.gradStepSampSize
        gradMean = (gradForward + gradBackward) / 2.0
        if self.disp == True and self.descentMethod == 'adam':
            print(repr(x0), F0)
        return F0, gradMean

    def _forward_Difference_And_F0(self, x0):
        assert len(x0) == self.numDim
        mask = np.eye(len(x0))
        mask = np.row_stack((np.zeros(self.numDim), mask))
        x_abArr = mask * self.gradStepSampSize + x0  # x upper and lower values for derivative calc
        if self.parallel == True:
            with mp.Pool() as pool:
                vals_ab = np.asarray(pool.map(self.funcObj, x_abArr))
        else:
            vals_ab = np.asarray([self.funcObj(x) for x in x_abArr])
        assert len(vals_ab.shape) == 1 and len(vals_ab) == self.numDim + 1
        F0 = vals_ab[0]
        deltaVals = vals_ab[1:] - F0
        grad = deltaVals / self.gradStepSampSize
        if self.disp == True and self.descentMethod == 'adam':
            print(repr(x0), F0)
        return F0, grad

    def _update_Speed(self, gradUnit, vPrev):
        deltaV = -gradUnit * self.stepSizeInitial
        deltaVDrag = (1.0 - self.momentumFact) * vPrev * self.hFrac
        vPrev = vPrev - deltaVDrag
        v = deltaV + vPrev
        if np.linalg.norm(v) < SMALL_NUMBER:
            return np.zeros(self.numDim)
        return v

    def _update_Time_Step(self, v):
        deltaX0 = np.linalg.norm(v) * 1.0
        self.hFrac = self.maxStepSize / deltaX0 if deltaX0 > self.maxStepSize else 1

    def momentum_Method(self):
        x = self.xi.copy()
        vPrev = np.zeros(self.numDim)
        for i in range(self.maxIters):
            F0, grad = self._gradient_And_F0(x)
            assert isinstance(F0, float) and len(grad) >= 1
            self.objValHistList.append(F0)
            self.xHistList.append(x)
            grad0 = np.linalg.norm(grad)
            if np.abs(grad0) > SMALL_NUMBER:
                gradUnit = grad / grad0
            else:
                if i == 0:
                    print('Gradient is zero on first iteration. Descent cannot proceed')
                    break
                elif self.momentumFact == 0.0:
                    print('Gradient is zero, and there is no momentum. Cannot proceed')
                gradUnit = np.zeros(len(grad))
            v = self._update_Speed(gradUnit, vPrev)
            if np.all(np.abs(v) < SMALL_NUMBER):
                warnings.warn("All step sizes are very small, and possibly result largely from numeric noise ")
            vPrev = v.copy()
            if self.disp == True:
                print('-----------',
                      'iteration: ' + str(i),
                      'Function value: ' + str(F0),
                      'at parameter space point: ' + str(repr(x)),
                      'gradient: ' + str(repr(grad)))
            self._update_Time_Step(v)
            deltaX = v * self.hFrac
            x = x + deltaX
        if self.Plot == True:
            plt.plot(self.objValHistList)
            plt.xlabel("Iteration")
            plt.ylabel("Cost (goal is to minimize)")
            plt.show()
        return self.xHistList[np.argmin(self.objValHistList)], np.min(self.objValHistList)

    def adam_Method(self):
        opt = descent.adam(self.stepSizeInitial)
        func = self._forward_Difference_And_F0 if self.gradMethod == 'forward' else self._central_Difference_And_F0
        display = None if self.disp == False else sys.stdout
        sol = opt.minimize(func, self.xi, maxiter=self.maxIters, display=display)
        if self.Plot == True:
            plt.semilogy(sol['obj'])
            plt.xlabel("Iteration")
            plt.ylabel("Cost (goal is to minimize)")
            plt.tight_layout()
            plt.show()
        return sol.x, np.min(sol['obj'])

    def solve_Grad_Descent(self):
        if self.descentMethod == 'momentum':
            xOptimal, objOptimal = self.momentum_Method()
        else:
            xOptimal, objOptimal = self.adam_Method()

        return xOptimal, objOptimal


def gradient_Descent(funcObj, xi, stepSizeInitial, maxIters, momentumFact=.9, gradMethod='central', disp=False,
                     parallel=True, Plot=False, gradStepSize=5e-6, maxStepSize=np.inf, descentMethod='adam'):
    solver = GradientOptimizer(funcObj, xi, stepSizeInitial, maxIters, momentumFact, disp, Plot, parallel, gradMethod,
                               gradStepSize, maxStepSize, descentMethod)
    return solver.solve_Grad_Descent()


def batch_Gradient_Descent(funcObj, bounds, numSamples, stepSizeInitial, maxIters, momentumFact=.9, disp=False,
                           gradMethod='central', maxStepSize=np.inf, descentMethod='adam'):
    samples = np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples))
    wrap = lambda x: gradient_Descent(funcObj, x, stepSizeInitial, maxIters, momentumFact=momentumFact,
                                      disp=disp
                                      , parallel=False, gradMethod=gradMethod, Plot=False, maxStepSize=maxStepSize)
    with mp.Pool() as pool:
        results = pool.map(wrap, samples)
    return results


def global_Gradient_Descent(funcObj, bounds, numSamples, stepSizeInitial, maxIters, gradStepSize, momentumFact=.9,
                            disp=False,
                            gradMethod='central', parallel=True, Plot=False, descentMethod='adam'):
    samples = np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples))
    if parallel == True:
        with mp.Pool(maxtasksperchild=1) as pool:
            vals = np.asarray(pool.map(funcObj, samples, chunksize=1))
    else:
        vals = np.asarray([funcObj(x) for x in samples])
    xOptimal = samples[np.argmin(vals)]
    result = gradient_Descent(funcObj, xOptimal, stepSizeInitial, maxIters, momentumFact=momentumFact,
                              disp=disp, parallel=parallel, gradMethod=gradMethod, Plot=Plot, gradStepSize=gradStepSize,
                              descentMethod=descentMethod)
    return result
