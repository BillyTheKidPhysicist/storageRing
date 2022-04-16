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
    def __init__(self,paramsOptimal: np.ndarray,alignmentTol: float=1e-3,tuning: str=None,
                 machineTolerance: float=250e-6):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian. Heavy lifiting is done in optimizerHelperFunctions.py

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param alignmentTol: Maximum displacement from ideal trajectory perpindicular to vacuum tube in one direction.
        This is our accepted alignmentTol
        :param tuning: Type of tuning for the lattice.
        """

        assert tuning in ('spacing','field',None)
        self.paramsOptimal=paramsOptimal
        self.tuning=tuning
        self.alignmentTol=alignmentTol
        self.machineTolerance=machineTolerance
        self.jitterableElements=(elementPT.CombinerHalbachLensSim,elementPT.LensIdeal,elementPT.HalbachLensSim)

    def generate_Ring_And_Injector_Lattice(self,useMagnetErrors: bool,useMachineError: bool,misalign: bool,
                                fieldDensityMul: float)-> tuple[ParticleTracerLattice,ParticleTracerLattice,np.ndarray]:
        params=self.apply_Machining_Errors(self.paramsOptimal) if useMachineError==True else self.paramsOptimal
        PTL_Ring,PTL_Injector= optimizerHelperFunctions.generate_Ring_And_Injector_Lattice(params,None,
                    jitterAmp=self.alignmentTol,fieldDensityMultiplier=fieldDensityMul,standardMagnetErrors=useMagnetErrors)
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
                                      misalign: bool):
        PTL_Ring, PTL_Injector,params = self.generate_Ring_And_Injector_Lattice(useMagnetErrors, useMachineError,misalign,
                                                                         fieldDensityMul)
        sol=optimizerHelperFunctions.solution_From_Lattice(PTL_Ring,PTL_Injector,self.tuning)
        sol.params=params
        return sol

    def measure_Sensitivity(self)-> None:

        #todo: now that i'm doing much more than just jittering elements, I should refactor this. Previously
        #it worked by reusing the lattice over and over again. Maybe I still want that to be a feature? Or maybe
        #that's dumb

        #todo: I need to figure out what all this is doing

        def _cost(i):
            # print(i)
            np.random.seed(i)
            if i==0:
                sol=self.inject_And_Trace_Through_Ring(False, False, 1.0, False)
            else:
                sol=self.inject_And_Trace_Through_Ring(True, False, 1.0, False)
            print(sol)
            print('seed',i)
            return sol.cost
        indices=list(range(1,30))
        results=tool_Parallel_Process(_cost,indices,processes=2,resultsAsArray=True)
        print(np.mean(results[1:]),np.std(results[1:]))
        print(repr(results))
        # np.savetxt('data',results)
        # print(repr(results))
        # _cost(1)


sa=StabilityAnalyzer(np.array([0.02054458, 0.0319046 , 0.01287383, 0.008     , 0.38994521]))
sa.measure_Sensitivity()



"""

-------only combiner------
turn of magnet errors for everything but combiner


slices with MOM only 
turn off magnet errors
manually tune slice number
1 slices : 0.59358765
2 slices: 0.59017531
3 slices: 0.5954740379081979

slices with MOM and errors:
1 slices: about .59
2 slices: about .59


----------only short lenses----------
only in ring, starting with 10 slices. 

1 slices:  0.7268293349807858 0.06507508625763957
2 slices: 0.7157740494294191 0.04613000206199112
3 slice: 0.7119855214760513 0.04843951004834795
5 slices: 0.6869815469708638 0.043694629731551374
10 slicse: 0.6711612810962547 0.02473337107514755
20 slices: 0.642223706952908 0.012517324025955645


# only injector
no combiner
1 slice: about .59




# only bender

original z slices:
0.7296869057061335 0.07165531405069497

double z slices: 








# no magnet aspect ratio
0.8546059062068966, 0.07168749551587099
array([0.57984456, 0.78309115, 0.83268921, 0.8671091 , 0.95749779,
       0.82510633, 0.72136541, 0.85506473, 0.92286548, 0.79680241,
       0.8618697 , 0.82773377, 0.88607702, 0.91018527, 0.91809394,
       0.77758302, 0.84396167, 0.94114121, 0.8004069 , 0.85607278,
       0.77116621, 0.93620748, 0.90577709, 0.68486005, 0.87391141,
       0.75917725, 0.9085369 , 0.95586637, 0.83615051, 0.96720112])
       
#aspect ratio of 5
0.7726052520689656, 0.04102837592284272
array([0.59358765, 0.74308414, 0.80562999, 0.7679912 , 0.83342578,
       0.73266141, 0.70230252, 0.7626731 , 0.80891237, 0.76717985,
       0.84083178, 0.73039704, 0.74557384, 0.82882023, 0.76976731,
       0.76990361, 0.8202544 , 0.77057172, 0.74728221, 0.76231874,
       0.75682056, 0.75563324, 0.76214688, 0.77685373, 0.73842207,
       0.73561506, 0.70658596, 0.79591469, 0.77855412, 0.88942476])
       
#aspect ratio of 4

0.752351276551724, 0.03620937794199646
array([0.59358765, 0.73831024, 0.71412211, 0.76487371, 0.77081482,
       0.66915509, 0.73629484, 0.71798777, 0.82648226, 0.78208901,
       0.74013826, 0.7573325 , 0.76545081, 0.81434752, 0.72557242,
       0.73725031, 0.80170659, 0.76648397, 0.71676238, 0.76989566,
       0.73966908, 0.81051598, 0.72064941, 0.70312636, 0.77790341,
       0.73567753, 0.71063524, 0.78621325, 0.78050114, 0.73822535])
       
#aspect ratio of 2





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
