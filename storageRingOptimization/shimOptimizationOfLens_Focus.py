from asyncDE import solve_Async
import matplotlib.pyplot as plt
import multiprocess as mp
import time
# from parallel_Gradient_Descent import gradient_Descent,global_Gradient_Descent
from geneticLensClass import GeneticLens
from parallel_Gradient_Descent import gradient_Descent,global_Gradient_Descent
import numpy as np
from profilehooks import profile
from lensOptimizerHelperFunctions import characterize_Lens_Full_Swarm,characterize_Lens_Concentric_Swarms
import copy
import skopt

class ShimHelper:
    def __init__(self, paramsBounds, paramsLocked, shimID):
        #todo: It doesn't really make sense that I attached shimID to param strings here.

        # bounds: Dict of bounds for shim arguments
        # params: Dict of bounds for shim params
        paramsBounds, paramsLocked = copy.copy(paramsBounds), copy.copy(paramsLocked)
        paramKeysNoID = ['radius', 'r', 'phi', 'deltaZ', 'theta', 'psi', 'planeSymmetry']
        assert 'planeSymmetry' in paramsLocked
        if paramsLocked['planeSymmetry']==False:
            paramKeysNoID.append('location') #users needs to specificy where the shim is
        assert all(key in paramsLocked|paramsBounds for key in paramKeysNoID)
        assert len(paramsLocked) +len(paramsBounds) == len(paramKeysNoID)
        assert all(len(value) == 2 for key, value in paramsBounds.items())
        assert all(key not in paramsLocked for key,value in paramsBounds.items())
        self.ID=shimID
        self.paramKeys = [key + shimID for key in paramKeysNoID]
        self.paramsLocked = {'component'+shimID: 'shim'}
        # self.paramsLocked['component']='shim'
        self.paramsBounds = {}
        for key, value in paramsLocked.items():
            self.paramsLocked[key + shimID] = value
        for key, value in paramsBounds.items():
            self.paramsBounds[key + shimID] = value



class LensHelper:
    def __init__(self, paramsBounds, paramsLocked):
        paramsBounds, paramsLocked = copy.copy(paramsBounds), copy.copy(paramsLocked)
        self.paramKeys = ['rp', 'width', 'length']
        assert all(key in {**paramsLocked, **paramsBounds} for key in self.paramKeys)
        assert len(paramsLocked) +len(paramsBounds) == len(self.paramKeys)
        assert all(len(value) == 2 for key, value in paramsBounds.items())
        assert all(key not in paramsLocked for key, value in paramsBounds.items())
        self.paramsLocked = paramsLocked
        self.paramsBounds = paramsBounds
        self.paramsLocked['component'] = 'layer'


class ShimOptimizer:
    def __init__(self,swarmModel):
        assert swarmModel in ('concentric', 'full')
        self.swarmModel=swarmModel
        self.lens = None #may be variable
        self.lensBaseLine=None
        self.shimsList = []
        self.bounds = []
        self.boundsKeys = []
        self.makeBoundsCalled = False
        self.addedLens=False
        self.baseLineFocusDict=None

    def set_Lens(self, paramsBounds, paramsLocked, baseLineParams):
        assert self.addedLens==False
        self.lens = LensHelper(paramsBounds, paramsLocked)
        self.lensBaseLine=LensHelper({},baseLineParams)
        self.addedLens=True

    def add_Shim(self, paramsBounds, paramsLocked):
        shimID = str(len(self.shimsList))
        self.shimsList.append(ShimHelper(paramsBounds, paramsLocked, shimID))

    def initialize_Optimization(self):
        assert self.makeBoundsCalled == False
        self.initialize_Baseline_Values()
        self.makeBoundsCalled = True
        for key, val in self.lens.paramsBounds.items():
            self.bounds.append(val)
            self.boundsKeys.append(key)
        for shim in self.shimsList:
            for key, val in shim.paramsBounds.items():
                self.bounds.append(val)
                self.boundsKeys.append(key)
        assert len(self.bounds) != 0

    def make_Lens_DNA_Dict(self, args):
        lensDNADict = self.lens.paramsLocked
        argsConsumed = 0
        for key, arg in zip(self.boundsKeys, args):
            if key in self.lens.paramKeys:
                argsConsumed += 1
                lensDNADict[key] = arg
        assert len(lensDNADict) == 4
        return lensDNADict, argsConsumed

    def make_Shim_DNA_Dict(self, shim, args, lensDNA_Dict):
        shimDNA_Dict_With_ID = shim.paramsLocked
        argsConsumed = 0 #track when args are used to make sure it adds up
        for key, arg in zip(self.boundsKeys, args): #unload relevant args into shim DNA. Params still have ID
            if key in shim.paramKeys:
                shimDNA_Dict_With_ID[key] = arg
                argsConsumed += 1
        shimDNA_Dict = {} #Genetic lens can't work with params with ID tags
        for key, val in shimDNA_Dict_With_ID.items(): #make
            keyNoID = key[:-1]
            shimDNA_Dict[keyNoID] = val
        #change deltaz to z and adjust z position of shim depending on location param
        lensZ_End = lensDNA_Dict['length'] / 2  # lens is centered with z=0
        if shim.paramsLocked['planeSymmetry'+shim.ID] == True or shim.paramsLocked['location'+shim.ID] == 'top':
            zLoc = lensZ_End + shimDNA_Dict['deltaZ']
        else:
            assert shim.paramsLocked['location' + shim.ID] == 'bottom'
            zLoc = -(lensZ_End + shimDNA_Dict['deltaZ'])  # flip shim to bottom
        shimDNA_Dict.pop('deltaZ')
        shimDNA_Dict['z']=zLoc
        if shim.paramsLocked['planeSymmetry'+shim.ID]==True: assert len(shimDNA_Dict) == 8
        else: assert len(shimDNA_Dict) == 9
        return shimDNA_Dict, argsConsumed

    def make_DNA_List_From_Args(self, args):
        assert self.makeBoundsCalled == True
        assert len(args) == len(self.boundsKeys)
        lensDNA_Dict, argsConsumedTotal = self.make_Lens_DNA_Dict(args)
        DNA_List = [lensDNA_Dict]
        # build shim dict
        for shim in self.shimsList:
            shimDNA_Dict, argsConsumed = self.make_Shim_DNA_Dict(shim, args, lensDNA_Dict)
            DNA_List.append(shimDNA_Dict)
            argsConsumedTotal += argsConsumed
        assert argsConsumedTotal == len(args)
        return DNA_List

    def make_Lens(self, args):
        DNA_List = self.make_DNA_List_From_Args(args)
        assert len(DNA_List) == 1 + len(self.shimsList)
        lens = GeneticLens(DNA_List)
        return lens

    def cost_Function(self, args,rejectOutOfRange,smoothGeometryCost):
        assert self.baseLineFocusDict is not None
        assert isinstance(rejectOutOfRange,bool) and isinstance(smoothGeometryCost,bool)
        lens = self.make_Lens(args)
        cost=0
        if lens.is_Geometry_Valid() == False and smoothGeometryCost==False:
            return np.inf
        else:
            cost+=lens.geometry_Frac_Overlap()
        if self.swarmModel=='concentric':
            results = characterize_Lens_Concentric_Swarms(lens,rejectOutOfRange)
        else:
            results=characterize_Lens_Full_Swarm(lens,rejectOutOfRange)
        if results is None:
            return np.inf
        cost+=self._peformance_Cost(results)
        return cost
    def _peformance_Cost(self,results):
        if self.swarmModel == 'full':
            cost=self._single_Swarm_Cost(results)
        elif self.swarmModel=='concentric':
            cost=self._multi_Swarm_Cost(results)
        else: raise ValueError
        return cost
    def _multi_Swarm_Cost(self,results):
        I_New=np.asarray([result['I'] for result in results])
        m_New=np.asarray([result['m'] for result in results])
        I_Baseline=np.asarray([result['I'] for result in self.baseLineFocusDict])
        m_Baseline=np.asarray([result['m'] for result in self.baseLineFocusDict])
        magSpread_Baseline=np.std(m_Baseline)
        magSpread_New=np.std(m_New)
        magSpreadCost=magSpread_New/magSpread_Baseline
        magDriftCost = 1000.0 * (np.mean(m_New) / np.mean(m_Baseline) - 1) ** 2
        focusInvarBaseline=I_Baseline*m_Baseline
        focusInvarNew=I_New*m_New
        focusCost=1/np.mean(focusInvarNew/focusInvarBaseline)
        cost=(magDriftCost+magSpreadCost+focusCost)/3.0 #normalize to one
        return cost
    def _single_Swarm_Cost(self,results):
        IPeak, m = results['I'], results['m']
        assert IPeak > 0.0
        focusCost = self.baseLineFocusDict['I'] / IPeak  # goal to is shrink this
        magCost = 1000.0 * (m / self.baseLineFocusDict['m'] - 1) ** 2  # goal is to keep magnification the same
        cost = focusCost + magCost
        return cost
    def initialize_Baseline_Values(self):
        lensBaseLine = GeneticLens([self.lensBaseLine.paramsLocked])
        if self.swarmModel=='concentric':
            self.baseLineFocusDict = characterize_Lens_Concentric_Swarms(lensBaseLine,True)
        else:
            self.baseLineFocusDict=characterize_Lens_Full_Swarm(lensBaseLine,True)
    def characterize_Results(self,args):
        self.initialize_Optimization()
        if self.swarmModel=='full':
            results=characterize_Lens_Full_Swarm(self.make_Lens(args),True)
        elif self.swarmModel=='concentric':
            results=characterize_Lens_Concentric_Swarms(self.make_Lens(args),True)
        else: raise ValueError
        print('----baseline----')
        print(self.baseLineFocusDict)
        print('----proposed lens----')
        print(results)
        performanceCost=self._peformance_Cost(results)
        lens = self.make_Lens(args)
        print('performance cost:',performanceCost)
        print('geometry cost:',lens.geometry_Frac_Overlap())
    def optimize(self,saveData=None):
        self.initialize_Optimization()
        rejectOutOfRange, smoothGeometryCost=True,False
        costFunc=lambda x: self.cost_Function(x,rejectOutOfRange,smoothGeometryCost)
        sol = solve_Async(costFunc, self.bounds, 15 * len(self.bounds),
                          tol=.03,disp=True,saveData=saveData,workers=8)
        print(sol)
    def optimize_Descent(self,Xi=None):
        self.initialize_Optimization()
        numSamples=1000
        samples = np.asarray(skopt.sampler.Sobol().generate(self.bounds, numSamples))
        costFuncGlobal = lambda x: self.cost_Function(x,True, False)
        with mp.Pool(maxtasksperchild=10) as pool:
            vals = np.asarray(pool.map(costFuncGlobal, samples))
        Xi = samples[np.argmin(vals)]
        costFunc = lambda x: self.cost_Function(x,False, True)
        return gradient_Descent(costFunc,Xi,100e-6,100,gradStepSize=10e-6,gradMethod='central',
                                descentMethod='adam',disp=True)