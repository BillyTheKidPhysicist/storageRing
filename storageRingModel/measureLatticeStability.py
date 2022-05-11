import itertools
import os
# os.environ['OPENBLAS_NUM_THREADS']='1'

import numpy as np

from helperTools import *
from typing import Union
import skopt
import elementPT
from SwarmTracerClass import SwarmTracer
from latticeModels import make_Ring_And_Injector_Version1
from runOptimizer import solution_From_Lattice
from ParticleTracerLatticeClass import ParticleTracerLattice
from collections.abc import Sequence


combinerTypes=(elementPT.CombinerHalbachLensSim,elementPT.CombinerIdeal,elementPT.CombinerSim)



class StabilityAnalyzer:
    def __init__(self,paramsOptimal: np.ndarray,alignmentTol: float=1e-3,
                 machineTolerance: float=250e-6):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param alignmentTol: Maximum displacement from ideal trajectory perpindicular to vacuum tube in one direction.
        This is our accepted alignmentTol
        """

        self.paramsOptimal=paramsOptimal
        self.alignmentTol=alignmentTol
        self.machineTolerance=machineTolerance
        self.jitterableElements=(elementPT.CombinerHalbachLensSim,elementPT.LensIdeal,elementPT.HalbachLensSim)

    def generate_Ring_And_Injector_Lattice(self,useMagnetErrors: bool,useMachineError: bool,misalign: bool,
                                fieldDensityMul: float,combinerSeed: int=None)\
                                -> tuple[ParticleTracerLattice,ParticleTracerLattice,np.ndarray]:
        params=self.apply_Machining_Errors(self.paramsOptimal) if useMachineError==True else self.paramsOptimal
        PTL_Ring,PTL_Injector= make_Ring_And_Injector_Version1(params,None,
                jitterAmp=self.alignmentTol,fieldDensityMultiplier=fieldDensityMul,
                standardMagnetErrors=useMagnetErrors,combinerSeed=combinerSeed)
        if misalign:
            self.jitter_System(PTL_Ring,PTL_Injector)
        return PTL_Ring,PTL_Injector,params

    def apply_Machining_Errors(self,params: np.ndarray)-> np.ndarray:
        deltaParams=2*(np.random.random_sample(params.shape)-.5)*self.machineTolerance
        params_Error=params+deltaParams
        return params_Error

    def make_Jitter_Amplitudes(self, element: elementPT.Element,randomOverRide: Optional[tuple])-> tuple[float,...]:
        angleMax = np.arctan(self.alignmentTol / element.L)
        randomNum = np.random.random_sample() if randomOverRide is None else randomOverRide[0]
        random4Nums=np.random.random_sample(4) if randomOverRide is None else randomOverRide[1]
        fractionAngle,fractionShift=randomNum,1-randomNum
        angleAmp, shiftAmp = angleMax *fractionAngle, self.alignmentTol * fractionShift
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
                    shiftY, shiftZ, rotY, rotZ = self.make_Jitter_Amplitudes(el,None)
                el.perturb_Element(shiftY, shiftZ, rotY, rotZ)

    def jitter_System(self,PTL_Ring: ParticleTracerLattice,PTL_Injector: ParticleTracerLattice)-> None:
        combinerRandomOverride=(np.random.random_sample(),np.random.random_sample(4))
        self.jitter_Lattice(PTL_Ring,combinerRandomOverride)
        self.jitter_Lattice(PTL_Injector,combinerRandomOverride)

    def dejitter_System(self,PTL_Ring,PTL_Injector):
        #todo: possibly useless
        tolerance0 = self.alignmentTol
        self.alignmentTol = 0.0
        self.jitter_Lattice(PTL_Ring,None)
        self.jitter_Lattice(PTL_Injector,None)
        self.alignmentTol=tolerance0

    def inject_And_Trace_Through_Ring(self, useMagnetErrors: bool, useMachineError: bool, fieldDensityMul: float,
                                      misalign: bool,combinerSeed: int=None):
        PTL_Ring, PTL_Injector,params = self.generate_Ring_And_Injector_Lattice(useMagnetErrors, useMachineError,
                                                                misalign,fieldDensityMul,combinerSeed=combinerSeed)
        sol=solution_From_Lattice(PTL_Ring,PTL_Injector,None)
        sol.params=params
        return sol

    def measure_Sensitivity(self)-> None:

        #todo: now that i'm doing much more than just jittering elements, I should refactor this. Previously
        #it worked by reusing the lattice over and over again. Maybe I still want that to be a feature? Or maybe
        #that's dumb

        #todo: I need to figure out what all this is doing

        def flux_Multiplication(i):
            np.random.seed(i)
            if i==0:
                sol=self.inject_And_Trace_Through_Ring(False, False, 1.0, False)
            else:
                # t=time.time()
                sol=self.inject_And_Trace_Through_Ring(True, False, 1.0, False,combinerSeed=i)
                # print(time.time()-t)

            # print('seed',i)
            # print(sol)
            return sol.cost,sol.fluxMultiplication
        indices=[0]#list(range(1,31))
        results=tool_Parallel_Process(flux_Multiplication,indices,processes=1,resultsAsArray=True)
        # os.system("say 'done bitch!'")
        print('cost',np.mean(results[:,0]),np.std(results[:,0]))
        print('flux',np.mean(results[:,1]),np.std(results[:,1]))
        print(repr(results))
        # np.savetxt('data',results)
        # print(repr(results))
        # _cost(1)


sa=StabilityAnalyzer(np.array([0.02054458, 0.0319046 , 0.01287383, 0.008     , 0.38994521]))
sa.measure_Sensitivity()


"""
cost 0.722327186619278 0.028545171039622976
flux 87.49116325974461 8.969504274976623
"""

"""
array([[  0.63446924, 115.09801186],
       [  0.65183042, 109.64275847],
       [  0.70120433,  94.12841809],
       [  0.72436033,  86.85230662],
       [  0.7330989 ,  84.10645887],
       [  0.65487233, 108.68692294],
       [  0.67829281, 101.32770684],
       [  0.67482588, 102.4170921 ],
       [  0.74577102,  80.1246072 ],
       [  0.66843761, 104.42442301],
       [  0.66743459, 104.73959441],
       [  0.65744768, 107.87769558],
       [  0.66802113, 104.55528895],
       [  0.69671807,  95.53809593],

"""


"""
cost 0.741651810121768 0.0899743429824311
flux 81.41907789070954 28.27186612050469
"""

"""


"""


#     perfect bender field
# dimTol!=0: 0.6911018370400578 0.038520128618614285

# angleTol!=0: 0.7854371756923504 0.07346859678625345

# normTol!=0.0: 0.7012097214125814 0.055367115646637696

#all enabled: 0.8659972015253184 0.07045670900857874


#       including bad bender field




"""
cost: 0.5857515739680698
cost: 0.5747971424938154
cost: 0.7570404978803216
"""


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
