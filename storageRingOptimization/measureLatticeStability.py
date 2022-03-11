import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from typing import Union
import skopt
import numpy as np
import elementPT
import multiprocess as mp
import optimizerHelperFunctions
import matplotlib.pyplot as plt
from SwarmTracerClass import SwarmTracer
import skopt
import time
from ParticleTracerLatticeClass import ParticleTracerLattice
from collections.abc import Sequence



# def wrapper(args,seed):
#     np.random.seed(seed)
#     try:
#         tuning=None
#         sol=solve_For_Lattice_Params(args,tuning,bumpOffsetAmp)
#         cost=sol.swarmCost
#         print(cost)
#     except:
#         np.set_printoptions(precision=100)
#         print('assert during evaluation on args: ',args)
#         assert False
#     return cost



class StabilityAnalyzer:
    def __init__(self,paramsOptimal: Sequence,jitterFWHM: float=500e-6,tuning: Union[str]=None):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian. Heavy lifiting is done in optimizerHelperFunctions.py

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param jitterFWHM: FWHM of gaussian distribution from which jittering is sampled. Tilt jitter is scaled depending
            on the length of the element because long elements will have smaller angular tilts. Translational and
            rotational misalignment will be added, so total may up to twice as much.
        :param tuning: Type of tuning for the lattice.
        """
        assert tuning in ('spacing','field',None)
        self.paramsOptimal=paramsOptimal
        self.tuning=tuning
        self.jitterSigma=jitterFWHM/2.355
        self.maxJitter=3*self.jitterSigma
        self.jitterableElements=(elementPT.CombinerHexapoleSim,elementPT.LensIdeal,elementPT.CombinerHexapoleSim)
    def generate_Ring_And_Injector_Lattice(self):
        return optimizerHelperFunctions.generate_Ring_And_Injector_Lattice(self.paramsOptimal,None, jitterAmp=self.maxJitter)
    def jitter_Element(self,element:elementPT.Element):
        L=element.L
        angleSigma=self.jitterSigma/L #simple scaling
        angleMax=self.maxJitter/L #simple scaling
        rotY=np.random.normal(scale=angleSigma)
        rotZ=np.random.normal(scale=angleSigma)
        shiftY=np.random.normal(scale=self.jitterSigma)
        shiftZ=np.random.normal(scale=self.jitterSigma)
        rotY=  np.clip([rotY],-angleMax,angleMax)[0]
        rotZ=  np.clip([rotZ],-angleMax,angleMax)[0]
        shiftY=np.clip([shiftY],-self.jitterSigma,self.jitterSigma)[0]
        shiftZ=np.clip([shiftZ],-self.jitterSigma,self.jitterSigma)[0]
        # print(element,shiftY,shiftZ,rotY,rotZ)
        element.perturb_Element(shiftY,shiftZ,rotY,rotZ)
    def jitter_Lattice(self,PTL):
        for el in PTL:
            if any(validElementType==type(el) for validElementType in self.jitterableElements):
                self.jitter_Element(el)
    def cost(self,PTL_Ring,PTL_Injector):
        cost=optimizerHelperFunctions.solution_From_Lattice(PTL_Ring,PTL_Injector,self.paramsOptimal,self.tuning).cost
        return cost
    def measure_Alingment_Sensitivity(self):
        PTL_Ring,PTL_Injector=self.generate_Ring_And_Injector_Lattice()
        for _ in range(10):
            self.jitter_Lattice(PTL_Ring)
            print(self.cost(PTL_Ring,PTL_Injector))


sa=StabilityAnalyzer(np.array([0.02417499, 0.02112171, 0.02081137, 0.22577471]))
sa.measure_Alingment_Sensitivity()



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


wrapper(Xi,1)
exit()
deltaXTest=np.ones(len(Xi))*varJitterAmp/2
boundsUpper=Xi+deltaXTest
boundsLower=Xi-deltaXTest
bounds=np.row_stack((boundsLower,boundsUpper)).T

samples=np.asarray(skopt.sampler.Sobol().generate(bounds, numSamples-1))
samples=np.row_stack((samples,Xi))
seedArr=np.arange(numSamples)+int(time.time())


with mp.Pool() as pool:
    results=np.asarray(pool.starmap(wrapper,zip(samples,seedArr)))
print(results)
data=np.column_stack((samples-Xi,results))
np.savetxt('stabilityData',data)
# plt.hist(data[:,-1])
# plt.show()
