import os

import numpy as np

os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *
from typing import Union
import skopt
import elementPT
import optimizerHelperFunctions
from SwarmTracerClass import SwarmTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from collections.abc import Sequence

combinerTypes=(elementPT.CombinerHalbachLensSim,elementPT.CombinerIdeal,elementPT.CombinerSim)



class StabilityAnalyzer:
    def __init__(self,paramsOptimal: Sequence,precision: float=1e-3,tuning: Union[str]=None):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian. Heavy lifiting is done in optimizerHelperFunctions.py

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param precision: Maximum displacement from ideal trajectory perpindicular to vacuum tube in one direction.
        This is our accepted tolerance
        :param tuning: Type of tuning for the lattice.
        """
        assert tuning in ('spacing','field',None)
        self.paramsOptimal=paramsOptimal
        self.tuning=tuning
        self.precision=precision
        self.jitterableElements=(elementPT.CombinerHalbachLensSim,elementPT.LensIdeal,elementPT.CombinerHalbachLensSim)
    def generate_Ring_And_Injector_Lattice(self,useMagnetErrors: bool,fieldDensityMul: float)-> \
                                                                    tuple[ParticleTracerLattice,ParticleTracerLattice]:
        return optimizerHelperFunctions.generate_Ring_And_Injector_Lattice(self.paramsOptimal,None,
                                jitterAmp=self.precision,fieldDensityMultiplier=fieldDensityMul,standardMagnetErrors=useMagnetErrors)
    def make_Jitter_Amplitudes(self, element: elementPT.Element,randomOverRide=None):
        L = element.L
        angleMax = np.arctan(self.precision / L)
        randomNum = np.random.random_sample() if randomOverRide is None else randomOverRide[0]
        random4Nums=np.random.random_sample(4) if randomOverRide is None else randomOverRide[1]
        angleAmp, shiftAmp = angleMax * randomNum, self.precision * (1 - randomNum)
        angleAmp = angleAmp / np.sqrt(2)  # consider both dimensions being misaligned
        shiftAmp = shiftAmp / np.sqrt(2) # consider both dimensions being misaligned
        rotAngleY = 2 * (random4Nums[0] - .5)* angleAmp
        rotAngleZ = 2 * (random4Nums[1] - .5)* angleAmp
        shiftY = 2 * (random4Nums[2] - .5) * shiftAmp
        shiftZ = 2 * (random4Nums[3] - .5) * shiftAmp
        return shiftY, shiftZ, rotAngleY,rotAngleZ

    def jitter_Lattice(self,PTL,combinerRandomOverride):
        for el in PTL:
            if any(validElementType==type(el) for validElementType in self.jitterableElements):
                if any(type(el)==elType for elType in combinerTypes):
                    print(combinerRandomOverride)
                    shiftY, shiftZ, rotY, rotZ = self.make_Jitter_Amplitudes(el,randomOverRide=combinerRandomOverride)
                else:
                    shiftY, shiftZ, rotY, rotZ = self.make_Jitter_Amplitudes(el)
                print(shiftY, shiftZ, rotY, rotZ)
                el.perturb_Element(shiftY, shiftZ, rotY, rotZ)
    def jitter_System(self,PTL_Ring,PTL_Injector):
        print('jitter')
        combinerRandomOverride=(np.random.random_sample(),np.random.random_sample(4))
        self.jitter_Lattice(PTL_Ring,combinerRandomOverride)
        self.jitter_Lattice(PTL_Injector,combinerRandomOverride)
    def dejitter_System(self,PTL_Ring,PTL_Injector):
        print('dejitter')
        precision0 = self.precision
        self.precision = 0.0
        self.jitter_Lattice(PTL_Ring,None)
        self.jitter_Lattice(PTL_Injector,None)
        self.precision=precision0
    def cost(self,PTL_Ring,PTL_Injector, misalign):
        if misalign==True:
            self.jitter_System(PTL_Ring,PTL_Injector)
        cost=optimizerHelperFunctions.solution_From_Lattice(PTL_Ring,PTL_Injector,self.paramsOptimal,self.tuning).cost
        if misalign==True:
            self.dejitter_System(PTL_Ring,PTL_Injector)

        return cost
    def measure_Sensitivity(self,useMagnetErrors,fieldDensity=1.0):
        def _cost(i):
            PTL_Ring, PTL_Injector = self.generate_Ring_And_Injector_Lattice(useMagnetErrors,fieldDensity)
            if i!=0:
                self.jitter_System(PTL_Ring, PTL_Injector)
                cost=self.cost(PTL_Ring, PTL_Injector,True)
            else:
                cost=self.cost(PTL_Ring, PTL_Injector,False)
            return cost
        results=tool_Parallel_Process(_cost,list(range(1,8)),processes=1,resultsAsArray=True)
        # np.savetxt('data',results)
        # print(repr(results))
        _cost(1)



sa=StabilityAnalyzer(np.array([0.02054458, 0.0319046 , 0.01287383, 0.008     , 0.38994521]))
sa.measure_Sensitivity(True,1.5)



exit()
'''
------ITERATIONS:  3480
POPULATION VARIABILITY: [0.01475089 0.01717158 0.01157133 0.01893284]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.02417499, 0.02112171, 0.02081137, 0.22577471])
cost: 0.7099381604306393
'''
numSamples=50
#rplens rplensfirst rplenslast rpBend LLens
Xi=np.array([0.02398725, 0.02110859, 0.02104631, 0.22405252])


# wrapper(Xi,1)
# exit()
# deltaXTest=np.ones(len(Xi))*varJitterAmp/2
# boundsUpper=Xi+deltaXTest
# boundsLower=Xi-deltaXTest
# bounds=np.row_stack((boundsLower,boundsUpper)).T
#
# samples=np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples-1))
# samples=np.row_stack((samples,Xi))
# seedArr=np.arange(numSamples)+int(time.time())
#
#
# with mp.Pool() as pool:
#     results=np.asarray(pool.starmap(wrapper,zip(samples,seedArr)))
# print(results)
# data=np.column_stack((samples-Xi,results))
# np.savetxt('stabilityData',data)
# # plt.hist(data[:,-1])
# # plt.show()
