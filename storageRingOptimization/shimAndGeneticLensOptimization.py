from asyncDE import solve_Async
import matplotlib.pyplot as plt
from geneticLensElement_Wrapper import GeneticLens
from parallel_Gradient_Descent import gradient_Descent,global_Gradient_Descent
import numpy as np
from lensOptimizerHelper import characterize_Lens_Full_Swarm,characterize_Lens_Concentric_Swarms
import copy
import skopt

def strip_ID_From_Dict(dictWithID):
    dictNoID = {}  # Genetic lens can't work with params with ID tags
    for key, val in dictWithID.items():  # make
        keyNoID = key[:-1]
        dictNoID[keyNoID] = val
    return dictNoID

class ShimHelper:
    def __init__(self, paramsBounds, paramsLocked, shimID):
        #todo: It doesn't really make sense that I attached shimID to param strings here.
        assert int(shimID)<=9
        # bounds: Dict of bounds for shim arguments
        # params: Dict of bounds for shim params
        paramsBounds, paramsLocked = copy.deepcopy(paramsBounds), copy.deepcopy(paramsLocked)
        if paramsLocked['shape']=='cube':
            paramKeysNoID = ['diameter', 'r', 'phi', 'deltaZ', 'psi', 'planeSymmetry','shape']
        elif paramsLocked['shape']=='sphere':
            paramKeysNoID = ['diameter', 'r', 'phi', 'deltaZ', 'theta', 'psi', 'planeSymmetry','shape']
        else: raise ValueError
        assert 'planeSymmetry' in paramsLocked
        if paramsLocked['planeSymmetry']==False:
            paramKeysNoID.append('location') #users needs to specificy where the shim is
        assert all(key in paramsLocked|paramsBounds for key in paramKeysNoID)
        assert len(paramsLocked) +len(paramsBounds) == len(paramKeysNoID)
        assert all(len(value) == 2 for key, value in paramsBounds.items())
        assert all(key not in paramsLocked for key,value in paramsBounds.items())
        assert isinstance(shimID,str)
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
    def __init__(self, paramsBoundsLayers, paramsLockedLayers,longitudinalSymm):
        paramsBoundsLayers, paramsLockedLayers = copy.deepcopy(paramsBoundsLayers), copy.deepcopy(paramsLockedLayers)
        assert isinstance(paramsBoundsLayers,list) and isinstance(paramsLockedLayers,list)
        assert len(paramsBoundsLayers)==len(paramsLockedLayers)
        paramKeys = ['rp', 'width', 'length']
        self.paramsLockedLayers=[]
        self.paramsBoundsLayers=[]
        self.numLayers=2*len(paramsBoundsLayers) if longitudinalSymm==True else len(paramsBoundsLayers)
        for paramsLocked,paramsBounds,i in zip(paramsLockedLayers,paramsBoundsLayers,range(self.numLayers)):
            assert i<=9
            ID=str(i)
            assert isinstance(paramsBounds,dict) and isinstance(paramsLocked,dict)
            assert all(key in {**paramsLocked, **paramsBounds} for key in paramKeys)
            assert len(paramsLocked) +len(paramsBounds) == len(paramKeys)
            assert all(len(value) == 2 for key, value in paramsBounds.items())
            assert all(key not in paramsLocked for key, value in paramsBounds.items())
            paramsLockedNew,paramsBoundsNew={},{}
            for key, value in paramsLocked.items():
                paramsLockedNew[key + ID] = value
            for key, value in paramsBounds.items():
                paramsBoundsNew[key + ID] = value
            paramsLockedNew['component'+ID]='layer'
            self.paramsLockedLayers.append(paramsLockedNew)
            self.paramsBoundsLayers.append(paramsBoundsNew)


class ShimOptimizer:
    def __init__(self,swarmModel,lensLayerMirrorSymm):
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
        self.apBase=None
        self.lensLayerMirrorSymm=lensLayerMirrorSymm
        self.lensHomoParamBounds= {} #parameters for tuning that effect ever layer in the lens

    def set_Lens(self, paramsBounds, paramsLocked, baseLineParams,lensHomoParamBounds=None):
        assert self.addedLens==False
        assert len(baseLineParams)==1 #I asuume base lens is made of 1 layer
        if lensHomoParamBounds is not None:
            assert isinstance(lensHomoParamBounds,dict)
            viableParams=['length'] #options allowed for tuning every layer in the lens
            assert viableParams[0] in lensHomoParamBounds and len(lensHomoParamBounds)==1
            self.lensHomoParamBounds=lensHomoParamBounds
        self.apBase=baseLineParams[0]['rp']*.9
        self.lens = LensHelper(paramsBounds, paramsLocked,self.lensLayerMirrorSymm)
        self.lensBaseLine=LensHelper([{}],baseLineParams,self.lensLayerMirrorSymm)
        self.addedLens=True

    def add_Shim(self, paramsBounds, paramsLocked):
        shimID = str(len(self.shimsList))
        self.shimsList.append(ShimHelper(paramsBounds, paramsLocked, shimID))

    def initialize_Optimization(self,overRideNumBoundsError=False):
        assert self.makeBoundsCalled == False
        self.initialize_Baseline_Values()
        self.makeBoundsCalled = True
        if 'length' in self.lensHomoParamBounds:
            self.bounds.append(self.lensHomoParamBounds['length'])
            self.boundsKeys.append('length')
        for paramsBounds in self.lens.paramsBoundsLayers:
            for key, val in paramsBounds.items():
                self.bounds.append(val)
                self.boundsKeys.append(key)
        for shim in self.shimsList:
            for key, val in shim.paramsBounds.items():
                self.bounds.append(val)
                self.boundsKeys.append(key)
        if overRideNumBoundsError==False:
            assert len(self.bounds) != 0
    def make_Lens_DNA_Dict(self, args):
        lensDNAList = self.lens.paramsLockedLayers #list of dicts for each layer
        argsConsumed = 0
        for key, arg in zip(self.boundsKeys, args):
            for i in range(len(lensDNAList)):
                if key in self.lens.paramsBoundsLayers[i]:
                    argsConsumed += 1
                    lensDNAList[i][key] = arg
        lensDNAList=[strip_ID_From_Dict(lensDNAList[i]) for i in range(len(lensDNAList))]
        if self.lensLayerMirrorSymm==True:
            lensDNAListFlipped=copy.deepcopy(lensDNAList)
            lensDNAListFlipped.reverse()
            lensDNAList.extend(lensDNAListFlipped)
        for key, arg in zip(self.boundsKeys, args):
            if key=='length':
                assert arg>0.0
                for DNA in lensDNAList: DNA[key]=arg/self.lens.numLayers
                assert abs(arg-sum(DNA['length'] for DNA in lensDNAList))<1e-12
                argsConsumed+=1
        assert all(len(lensDNA_Dict)==4 for lensDNA_Dict in lensDNAList)
        return lensDNAList, argsConsumed

    def make_Shim_DNA_Dict(self, shim, args, lensDNAList):
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
        lensZ_End = sum(DNA['length'] for DNA in lensDNAList) / 2  # lens is centered with z=0
        if shim.paramsLocked['planeSymmetry'+shim.ID] == True or shim.paramsLocked['location'+shim.ID] == 'top':
            zLoc = lensZ_End + shimDNA_Dict['deltaZ']+shimDNA_Dict['diameter']/2.0
        else:
            assert shim.paramsLocked['location' + shim.ID] == 'bottom'
            zLoc = -(lensZ_End + shimDNA_Dict['deltaZ']+shimDNA_Dict['diameter']/2.0)  # flip shim to bottom
        shimDNA_Dict.pop('deltaZ')
        shimDNA_Dict['r']+=shimDNA_Dict['diameter']/2.0
        shimDNA_Dict['z']=zLoc
        numArgsWithSymmetry=9 if shim.paramsLocked['shape'+shim.ID]=='sphere' else 8
        if shim.paramsLocked['planeSymmetry'+shim.ID]==True: assert len(shimDNA_Dict) == numArgsWithSymmetry
        else: assert len(shimDNA_Dict) == numArgsWithSymmetry+1 #if not symmetry, need to specify which side
        return shimDNA_Dict, argsConsumed

    def make_DNA_List_From_Args(self, args):
        assert self.makeBoundsCalled == True
        assert len(args) == len(self.boundsKeys)
        lensDNA_List, argsConsumedTotal = self.make_Lens_DNA_Dict(args)
        DNA_List = copy.deepcopy(lensDNA_List)
        # build shim dict
        for shim in self.shimsList:
            shimDNA_Dict, argsConsumed = self.make_Shim_DNA_Dict(shim, args, lensDNA_List)
            DNA_List.append(shimDNA_Dict)
            argsConsumedTotal += argsConsumed
        assert argsConsumedTotal == len(args)
        return DNA_List

    def make_Lens(self, args):
        DNA_List = self.make_DNA_List_From_Args(args)
        assert len(DNA_List) == self.lens.numLayers+ len(self.shimsList)
        # L=sum(DNA['length'] if DNA['component']=='layer' else 0 for DNA in DNA_List)
        lens = GeneticLens(DNA_List)
        #todo: obnoxious convention with lens length has caused a collision. I need to refactor to be consisten somehow
        #I don't anticipate making individual layer lengths, so I should refaactor assuming that won't happen

        # if 'length' not in self.lensHomoParamBounds: #otherwise length is being varied
        #     assert abs(L-self.lensBaseLine.paramsLockedLayers[0]['length0'])<1e-12 #too many layers possibly
        # else:
        #     assert self.lensHomoParamBounds['length'][0]-1e-6<=L<=self.lensHomoParamBounds['length'][1]+1e-6
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
            results = characterize_Lens_Concentric_Swarms(lens,rejectOutOfRange,apMin=self.apBase)
        else:
            results=characterize_Lens_Full_Swarm(lens,rejectOutOfRange,apMin=self.apBase)
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
        # magSpread_Baseline=np.std(m_Baseline)
        # magSpread_New=np.std(m_New)
        # magSpreadCost=magSpread_New/magSpread_Baseline
        magDriftCost = 1000.0 * (np.mean(m_New) / np.mean(m_Baseline) - 1) ** 2
        focusInvarBaseline=I_Baseline*m_Baseline
        focusInvarNew=I_New*m_New
        focusCost=1/np.mean(focusInvarNew/focusInvarBaseline)
        cost=magDriftCost+focusCost
        return cost
    def _single_Swarm_Cost(self,results):
        IPeak, m = results['I'], results['m']
        assert IPeak > 0.0
        focusCost = self.baseLineFocusDict['I'] / IPeak  # goal to is shrink this
        magCost = 1000.0 * (m / self.baseLineFocusDict['m'] - 1) ** 2  # goal is to keep magnification the same
        cost = focusCost + magCost
        return cost
    def initialize_Baseline_Values(self):
        baseLine_DNA_List=[strip_ID_From_Dict(self.lensBaseLine.paramsLockedLayers[0])]
        lensBaseLine = GeneticLens(baseLine_DNA_List)
        if self.swarmModel=='concentric':
            self.baseLineFocusDict = characterize_Lens_Concentric_Swarms(lensBaseLine,True,apMin=self.apBase)
        else:
            self.baseLineFocusDict=characterize_Lens_Full_Swarm(lensBaseLine,True,apMin=self.apBase)

    def characterize_Results(self,args,display=True):
        self.initialize_Optimization(overRideNumBoundsError=True)
        if self.swarmModel=='full':
            results=characterize_Lens_Full_Swarm(self.make_Lens(args),True,apMin=self.apBase)
        elif self.swarmModel=='concentric':
            results=characterize_Lens_Concentric_Swarms(self.make_Lens(args),True,apMin=self.apBase)
        else: raise ValueError
        performanceCost = self._peformance_Cost(results)
        lens = self.make_Lens(args)
        if display==True:
            print('----baseline----')
            print(self.baseLineFocusDict)
            print('----proposed lens----')
            print(results)
            print('performance cost:',performanceCost)
            print('geometry cost:',lens.geometry_Frac_Overlap())
        return results,performanceCost
    def optimize(self,saveData=None):
        self.initialize_Optimization()
        print(self.boundsKeys)
        rejectOutOfRange, smoothGeometryCost=True,False
        costFunc=lambda x: self.cost_Function(x,rejectOutOfRange,smoothGeometryCost)
        sol = solve_Async(costFunc, self.bounds, 15 * len(self.bounds),
                          tol=.03,disp=True,saveData=saveData,workers=9)
        print(sol)
    def optimize_Descent(self,Xi):
        self.initialize_Optimization()
        # numSamples=1000
        # samples = np.asarray(skopt.sampler.Sobol().generate(self.bounds, numSamples))
        # costFuncGlobal = lambda x: self.cost_Function(x,True, False)
        # with mp.Pool(maxtasksperchild=10) as pool:
        #     vals = np.asarray(pool.map(costFuncGlobal, samples))
        # Xi = samples[np.argmin(vals)]
        costFunc = lambda x: self.cost_Function(x,False, True)
        return gradient_Descent(costFunc,Xi,50e-6,100,gradStepSize=10e-6,gradMethod='central',
                                descentMethod='adam',disp=True)
