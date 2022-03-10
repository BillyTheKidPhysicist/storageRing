import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import skopt
import numpy as np
import elementPT
import multiprocess as mp
from optimizerHelperFunctions import generate_Ring_And_Injector_Lattice
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
    def __init__(self,paramsOptimal: Sequence,jitterFWHM: float=1e-3):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian.

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param jitterFWHM: FWHM of gaussian distribution from which jittering is sampled. Tilt jitter is scaled depending
            on the length of the element because long elements will have smaller angular tilts. Translational and
            rotational misalignment will be added, so total may up to twice as much.
        """
        self.PTL=self.make_Lattice(paramsOptimal)
        self.jitterSigma=jitterFWHM/2.355
        self.jitterableElements=(elementPT.LensIdeal,elementPT.HalbachLensSim,elementPT.CombinerHexapoleSim)
    def make_Lattice(self,params):
        PTL = ParticleTracerLattice(200.0, latticeType='storageRing')
        PTL.add_Lens_Ideal(.5, 1.0, .01)
        PTL.add_Drift(.1)
        PTL.add_Lens_Ideal(.5, 1.0, .01)
        PTL.add_Bender_Ideal(np.pi, 1.0, 1.0, .01)
        PTL.add_Lens_Ideal(.5, 1.0, .01)
        PTL.add_Drift(.1)
        PTL.add_Lens_Ideal(.5, 1.0, .01)
        PTL.add_Bender_Ideal(np.pi, 1.0, 1.0, .01)
        PTL.end_Lattice()
        return PTL
    def jitter_Element(self,element:elementPT.Element):
        L=element.L
        maxAngle=self.jitterSigma/L #simple scaling
        rotY=np.random.normal(scale=maxAngle)
        rotZ=np.random.normal(scale=maxAngle)
        shiftY=np.random.normal(scale=self.jitterSigma)
        shiftZ=np.random.normal(scale=self.jitterSigma)
        element.perturb_Element(shiftY,shiftZ,rotY,rotZ)
    def jitter_Lattice(self):
        for el in self.PTL:
            if any(validElementType==type(el) for validElementType in self.jitterableElements):
                self.jitter_Element(el)
    def cost(self):
        swarmTracer=SwarmTracer(self.PTL)
        swarm=swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(3e-3,5.0,1.0,1000,sameSeed=True)
        swarmTraced=swarmTracer.trace_Swarm_Through_Lattice(swarm,1e-5,.5,fastMode=True,stepsBetweenLogging=4)
        # self.PTL.show_Lattice(swarm=swarmTraced,showTraceLines=True)
        return swarmTraced.survival_Rev()
    def jitter_Cost(self):
        print(self.cost())
        costList=[]
        for _ in range(100):
            self.jitter_Lattice()
            costList.append(self.cost())
        plt.hist(costList)
        plt.show()
import dill
dill.pickle()

sa=StabilityAnalyzer([])
sa.jitter_Cost()



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
