import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from helperTools import *
from typing import Union
import skopt
import elementPT
import optimizerHelperFunctions
from SwarmTracerClass import SwarmTracer
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
        :param precision: Maximum displacement from ideal trajectory perpindicular to vacuum tube in one direction.
        This is our accepted tolerance
        :param tuning: Type of tuning for the lattice.
        """
        assert tuning in ('spacing','field',None)
        self.paramsOptimal=paramsOptimal
        self.tuning=tuning
        self.precision=precision
        self.jitterableElements=(elementPT.CombinerHalbachLensSim,elementPT.LensIdeal,elementPT.CombinerHalbachLensSim)
    def generate_Ring_And_Injector_Lattice(self,useMagnetErrors: bool):
        return optimizerHelperFunctions.generate_Ring_And_Injector_Lattice(self.paramsOptimal,None,
                                jitterAmp=0*self.precision/2.0,fieldDensityMultiplier=1.0,standardMagnetErrors=useMagnetErrors)
    def make_Jitter_Amplitudes(self, element: elementPT.Element):
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
        shiftY,shiftZ,rotY,rotZ=self.make_Jitter_Amplitudes(element)
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
        # file = open('temp2', 'rb')
        # PTL_Injector = dill.load(file)
        # file.close()
        # self.jitter_Lattice(PTL_Injector)

        def _cost(i):
            if i!=0:
                PTL_Ring, PTL_Injector = self.generate_Ring_And_Injector_Lattice(True)
                np.random.seed(i)
                self.jitter_Lattice(PTL_Ring)
                self.jitter_Lattice(PTL_Injector)
                cost=self.cost(PTL_Ring, PTL_Injector)
            else:
                try:
                    PTL_Ring, PTL_Injector = self.generate_Ring_And_Injector_Lattice(False)
                    cost=self.cost(PTL_Ring, PTL_Injector)
                except:
                    print('issue')
                    cost= np.nan
            print(cost)
            return cost
        # _cost(0)
        # _cost(2)
        results=tool_Parallel_Process(_cost,list(range(10)),processes=5)
        # print(repr(results))




sa=StabilityAnalyzer(np.array([0.02054458, 0.0319046 , 0.01287383, 0.008     , 0.38994521]))
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
