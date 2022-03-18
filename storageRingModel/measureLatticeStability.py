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
    def __init__(self,paramsOptimal: Sequence,precision: float=1e-3,tuning: Union[str]=None):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian. Heavy lifiting is done in optimizerHelperFunctions.py

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param precision: Maximum displacement from ideal trajectory perpindicular to vacuum tube. This is out accepted
        tolerance
        :param tuning: Type of tuning for the lattice.
        """
        assert tuning in ('spacing','field',None)
        self.paramsOptimal=paramsOptimal
        self.tuning=tuning
        self.precision=precision
        self.jitterableElements=(elementPT.CombinerHalbachLensSim,elementPT.LensIdeal,elementPT.CombinerHalbachLensSim)
    def generate_Ring_And_Injector_Lattice(self):
        return optimizerHelperFunctions.generate_Ring_And_Injector_Lattice(self.paramsOptimal,None,
                                                        jitterAmp=self.precision/2.0,fieldDensityMultiplier=1.0)
    def jitter_Amplitudes(self,element: elementPT.Element):
        L = element.L
        angleMax = np.arctan(self.precision / L)
        randomNum = np.random.random_sample()
        angleAmp, shiftAmp = angleMax * randomNum, self.precision * (1 - randomNum)
        angleAmp = angleAmp / np.sqrt(2)  # consider both dimensions being misaligned
        shiftAmp = shiftAmp / np.sqrt(2) # consider both dimensions being misaligned
        rotAngleY = 2 * (np.random.random_sample() - .5)* angleAmp
        rotAngleZ = 2 * (np.random.random_sample() - .5)* angleAmp
        shiftY = 2 * (np.random.random_sample() - .5) * shiftAmp
        shiftZ = 2 * (np.random.random_sample() - .5) * shiftAmp
        return shiftY, shiftZ, rotAngleY,rotAngleZ

    def jitter_Element(self,element:elementPT.Element):
        shiftY,shiftZ,rotY,rotZ=self.jitter_Amplitudes(element)
        element.perturb_Element(shiftY,shiftZ,rotY,rotZ)
    def jitter_Lattice(self,PTL):
        for el in PTL:
            if any(validElementType==type(el) for validElementType in self.jitterableElements):
                self.jitter_Element(el)
    def cost(self,PTL_Ring,PTL_Injector):
        cost=optimizerHelperFunctions.solution_From_Lattice(PTL_Ring,PTL_Injector,self.paramsOptimal,self.tuning).cost
        return cost
    def measure_Alingment_Sensitivity(self):
        # np.random.seed(42)
        PTL_Ring,PTL_Injector=self.generate_Ring_And_Injector_Lattice()
        import dill
        # # file=open('temp1','wb')
        # # dill.dump(PTL_Ring,file)
        # file = open('temp2', 'wb')
        # dill.dump(PTL_Injector, file)
        # exit()
        #
        # import dill
        # file = open('temp1', 'rb')
        # PTL_Ring=dill.load(file)
        file = open('temp2', 'rb')
        PTL_Injector = dill.load(file)
        file.close()
        # self.jitter_Lattice(PTL_Injector)

        def _cost(i):
            print(i)
            if i!=0:
                np.random.seed(i)
                self.jitter_Lattice(PTL_Ring)
                self.jitter_Lattice(PTL_Injector)
            try:
                return self.cost(PTL_Ring, PTL_Injector)
            except:
                return np.nan
        # _cost(0)
        # [_cost(i) for i in list(range(30))]
        with mp.Pool(10) as pool:
            results=np.asarray(pool.map(_cost,list(range(30))))
        # np.savetxt('stabilityData',results)
        print(results)

"""
[0.70624971 0.7173343  0.69522913 0.7154003  0.70952638 0.71102004
 0.70953184 0.72955335 0.71696684 0.72404627 0.68602692 0.71580795
 0.70982592 0.71133951 0.71000365 0.7082258  0.75010548 0.72227082
 0.71354967 0.72761908 0.70818812 0.73874164 0.70931439 0.71204169
 0.71617644 0.70881442 0.70971138 0.71839448 0.72854086 0.71631354]
"""


sa=StabilityAnalyzer(np.array([0.00976065, 0.03458421, 0.01329697, 0.01013278, 0.39046408]))
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
