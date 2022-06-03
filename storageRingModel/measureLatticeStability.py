import itertools
import os
# os.environ['OPENBLAS_NUM_THREADS']='1'

import numpy as np

from helperTools import *
from typing import Union
import skopt
import elementPT
from SwarmTracerClass import SwarmTracer
from latticeModels import make_Ring_And_Injector_Version3
from runOptimizer import solution_From_Lattice
from ParticleTracerLatticeClass import ParticleTracerLattice
from collections.abc import Sequence

combinerTypes = (elementPT.CombinerHalbachLensSim, elementPT.CombinerIdeal, elementPT.CombinerSim)


class StabilityAnalyzer:
    def __init__(self, paramsOptimal: np.ndarray, alignmentTol: float = 1e-3,
                 machineTolerance: float = 250e-6):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param alignmentTol: Maximum displacement from ideal trajectory perpindicular to vacuum tube in one direction.
        This is our accepted alignmentTol
        """

        self.paramsOptimal = paramsOptimal
        self.alignmentTol = alignmentTol
        self.machineTolerance = machineTolerance
        self.jitterableElements = (elementPT.CombinerHalbachLensSim, elementPT.LensIdeal, elementPT.HalbachLensSim)

    def generate_Ring_And_Injector_Lattice(self, useMagnetErrors: bool,
                                           combinerSeed: int = None) \
            -> tuple[ParticleTracerLattice, ParticleTracerLattice, np.ndarray]:
        # params=self.apply_Machining_Errors(self.paramsOptimal) if useMachineError==True else self.paramsOptimal
        params = self.paramsOptimal
        PTL_Ring, PTL_Injector = make_Ring_And_Injector_Version3(params, useMagnetErrors=useMagnetErrors,
                                                                 combinerSeed=combinerSeed)
        # if misalign:
        #     self.jitter_System(PTL_Ring,PTL_Injector)
        return PTL_Ring, PTL_Injector, params

    def apply_Machining_Errors(self, params: np.ndarray) -> np.ndarray:
        deltaParams = 2 * (np.random.random_sample(params.shape) - .5) * self.machineTolerance
        params_Error = params + deltaParams
        return params_Error

    def make_Jitter_Amplitudes(self, element: elementPT.Element, randomOverRide: Optional[tuple]) -> tuple[float, ...]:
        angleMax = np.arctan(self.alignmentTol / element.L)
        randomNum = np.random.random_sample() if randomOverRide is None else randomOverRide[0]
        random4Nums = np.random.random_sample(4) if randomOverRide is None else randomOverRide[1]
        fractionAngle, fractionShift = randomNum, 1 - randomNum
        angleAmp, shiftAmp = angleMax * fractionAngle, self.alignmentTol * fractionShift
        angleAmp = angleAmp / np.sqrt(2)  # consider both dimensions being misaligned
        shiftAmp = shiftAmp / np.sqrt(2)  # consider both dimensions being misaligned
        rotAngleY = 2 * (random4Nums[0] - .5) * angleAmp
        rotAngleZ = 2 * (random4Nums[1] - .5) * angleAmp
        shiftY = 2 * (random4Nums[2] - .5) * shiftAmp
        shiftZ = 2 * (random4Nums[3] - .5) * shiftAmp
        return shiftY, shiftZ, rotAngleY, rotAngleZ

    def jitter_Lattice(self, PTL, combinerRandomOverride):
        for el in PTL:
            if any(validElementType == type(el) for validElementType in self.jitterableElements):
                if any(type(el) == elType for elType in combinerTypes):
                    shiftY, shiftZ, rotY, rotZ = self.make_Jitter_Amplitudes(el, randomOverRide=combinerRandomOverride)
                else:
                    shiftY, shiftZ, rotY, rotZ = self.make_Jitter_Amplitudes(el, None)
                el.perturb_Element(shiftY, shiftZ, rotY, rotZ)

    def jitter_System(self, PTL_Ring: ParticleTracerLattice, PTL_Injector: ParticleTracerLattice) -> None:
        combinerRandomOverride = (np.random.random_sample(), np.random.random_sample(4))
        self.jitter_Lattice(PTL_Ring, combinerRandomOverride)
        self.jitter_Lattice(PTL_Injector, combinerRandomOverride)

    def dejitter_System(self, PTL_Ring, PTL_Injector):
        # todo: possibly useless
        tolerance0 = self.alignmentTol
        self.alignmentTol = 0.0
        self.jitter_Lattice(PTL_Ring, None)
        self.jitter_Lattice(PTL_Injector, None)
        self.alignmentTol = tolerance0

    def inject_And_Trace_Through_Ring(self, useMagnetErrors: bool, combinerSeed: int = None):
        PTL_Ring, PTL_Injector, params = self.generate_Ring_And_Injector_Lattice(useMagnetErrors,
                                                                                 combinerSeed=combinerSeed)
        sol = solution_From_Lattice(PTL_Ring, PTL_Injector)
        sol.params = params
        return sol

    def measure_Sensitivity(self) -> None:

        # todo: now that i'm doing much more than just jittering elements, I should refactor this. Previously
        # it worked by reusing the lattice over and over again. Maybe I still want that to be a feature? Or maybe
        # that's dumb

        # todo: I need to figure out what all this is doing

        def flux_Multiplication(i):
            np.random.seed(i)
            if i == 0:
                sol = self.inject_And_Trace_Through_Ring(False)
            else:
                sol = self.inject_And_Trace_Through_Ring(True, combinerSeed=i)
            # print('seed',i)
            print(i, sol.fluxMultiplication)
            return sol.cost, sol.fluxMultiplication

        indices = list(range(1, 17))
        results = tool_Parallel_Process(flux_Multiplication, indices, processes=8, resultsAsArray=True)
        os.system("say 'done bitch!'")
        # print('cost',np.mean(results[:,0]),np.std(results[:,0]))
        print('flux', np.mean(results[:, 1]), np.std(results[:, 1]))
        # plt.hist(results[:,1])
        # plt.show()
        print(repr(results[:, 1]))
        # np.savetxt('data',results)
        # print(repr(results))
        # _cost(1)


x = np.array([0.02477938, 0.01079024, 0.04059919, 0.010042, 0.07175166, 0.51208528])
sa = StabilityAnalyzer(x)
sa.measure_Sensitivity()
