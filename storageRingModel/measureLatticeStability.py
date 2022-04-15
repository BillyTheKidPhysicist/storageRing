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
    def __init__(self,paramsOptimal: Sequence,tolerance: float=1e-3,tuning: Union[str]=None):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian. Heavy lifiting is done in optimizerHelperFunctions.py

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param tolerance: Maximum displacement from ideal trajectory perpindicular to vacuum tube in one direction.
        This is our accepted tolerance
        :param tuning: Type of tuning for the lattice.
        """
        assert tuning in ('spacing','field',None)
        self.paramsOptimal=paramsOptimal
        self.tuning=tuning
        self.tolerance=tolerance
        self.jitterableElements=(elementPT.CombinerHalbachLensSim,elementPT.LensIdeal,elementPT.CombinerHalbachLensSim)
    def generate_Ring_And_Injector_Lattice(self,useMagnetErrors: bool,fieldDensityMul: float)-> \
                                                                    tuple[ParticleTracerLattice,ParticleTracerLattice]:
        return optimizerHelperFunctions.generate_Ring_And_Injector_Lattice(self.paramsOptimal,None,
                                jitterAmp=self.tolerance,fieldDensityMultiplier=fieldDensityMul,standardMagnetErrors=useMagnetErrors)
    def make_Jitter_Amplitudes(self, element: elementPT.Element,randomOverRide=None):
        L = element.L
        angleMax = np.arctan(self.tolerance / L)
        randomNum = np.random.random_sample() if randomOverRide is None else randomOverRide[0]
        random4Nums=np.random.random_sample(4) if randomOverRide is None else randomOverRide[1]
        angleAmp, shiftAmp = angleMax * randomNum, self.tolerance * (1 - randomNum)
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
                    shiftY, shiftZ, rotY, rotZ = self.make_Jitter_Amplitudes(el,randomOverRide=combinerRandomOverride)
                else:
                    shiftY, shiftZ, rotY, rotZ = self.make_Jitter_Amplitudes(el)
                el.perturb_Element(shiftY, shiftZ, rotY, rotZ)
    def jitter_System(self,PTL_Ring,PTL_Injector):
        combinerRandomOverride=(np.random.random_sample(),np.random.random_sample(4))
        self.jitter_Lattice(PTL_Ring,combinerRandomOverride)
        self.jitter_Lattice(PTL_Injector,combinerRandomOverride)
    def dejitter_System(self,PTL_Ring,PTL_Injector):
        tolerance0 = self.tolerance
        self.tolerance = 0.0
        self.jitter_Lattice(PTL_Ring,None)
        self.jitter_Lattice(PTL_Injector,None)
        self.tolerance=tolerance0
    def cost(self,PTL_Ring,PTL_Injector, misalign):
        if misalign==True:
            self.jitter_System(PTL_Ring,PTL_Injector)
        cost=optimizerHelperFunctions.solution_From_Lattice(PTL_Ring,PTL_Injector,self.paramsOptimal,self.tuning).cost
        if misalign==True:
            self.dejitter_System(PTL_Ring,PTL_Injector)

        return cost
    def measure_Sensitivity(self,useMagnetErrors,fieldDensity):
        # PTL_Ring, PTL_Injector = self.generate_Ring_And_Injector_Lattice(True, fieldDensity)
        # self.cost(PTL_Ring, PTL_Injector, False)
        # exit()
        def _cost(i):
            np.random.seed(i)
            PTL_Ring, PTL_Injector = self.generate_Ring_And_Injector_Lattice(useMagnetErrors,fieldDensity)
            if i!=0:
                cost=self.cost(PTL_Ring, PTL_Injector,True)
            else:
                cost=self.cost(PTL_Ring, PTL_Injector,False)
            return cost
        indices=[1,2,3]#list(range(1,8))
        results=tool_Parallel_Process(_cost,indices,processes=3,resultsAsArray=True,)
        print(np.mean(results),np.std(results))
        # np.savetxt('data',results)
        # print(repr(results))
        # _cost(1)


sa=StabilityAnalyzer(np.array([0.02054458, 0.0319046 , 0.01287383, 0.008     , 0.38994521]),tolerance=0.0)
sa.measure_Sensitivity(True,1.0)


#     perfect bender field
# dimTol!=0: 0.6911018370400578 0.038520128618614285

# angleTol!=0: 0.7854371756923504 0.07346859678625345

# normTol!=0.0: 0.7012097214125814 0.055367115646637696

#all enabled: 0.8659972015253184 0.07045670900857874


#       including bad bender field




# 7 points:  0.896675367008891 0.009885088830867797 t: ~250
# 9 points: 0.8346502377017649 0.03829933350921161, t: ~500
# 11 points: 0.8347214827543971 0.04581475138471638, t: ~800



"""
----------Solution-----------   
injector element spacing optimum configuration: None
storage ring tuned params 1 optimum configuration: None
storage ring tuned params 2 optimum configuration: None
cost: 0.5991795690377881
percent max flux multiplication: 40.15859482589723
scipy message: None
----------------------------
"""


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
