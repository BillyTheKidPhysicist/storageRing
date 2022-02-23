from asyncDE import solve_Async
import matplotlib.pyplot as plt
import multiprocess as mp
import time
# from parallel_Gradient_Descent import gradient_Descent,global_Gradient_Descent
from geneticLensClass import GeneticLens

import numpy as np
from profilehooks import profile
from lensOptimizerHelperFunctions import IPeak_And_Magnification_From_Lens


class ShimHelper:
    def __init__(self, paramsBounds, paramsLocked, shimID):
        # bounds: Dict of bounds for shim arguments
        # params: Dict of bounds for shim params
        paramKeysNoID = ['radius', 'r', 'phi', 'z', 'theta', 'psi', 'planeSymmetry']
        assert all(key in {**paramsLocked, **paramsBounds} for key in paramKeysNoID)
        assert len(paramsLocked | paramsBounds) == len(paramKeysNoID)
        assert 'planeSymmetry' not in paramsBounds
        assert isinstance(paramsLocked['planeSymmetry'], bool)
        assert all(len(value) == 2 for key, value in paramsBounds.items())
        self.paramKeys = [key + shimID for key in paramKeysNoID]
        self.paramsLocked = {'component': 'shim'}
        # self.paramsLocked['component']='shim'
        self.paramsBounds = {}
        for key, value in paramsLocked.items():
            self.paramsLocked[key + shimID] = value
        for key, value in paramsBounds.items():
            self.paramsBounds[key + shimID] = value


class LensHelper:
    def __init__(self, paramsBounds, paramsLocked):
        self.paramKeys = ['rp', 'width', 'length']
        assert all(key in {**paramsLocked, **paramsBounds} for key in self.paramKeys)
        assert len(paramsLocked | paramsBounds) == len(self.paramKeys)
        assert all(len(value) == 2 for key, value in paramsBounds.items())
        self.paramsLocked = paramsLocked
        self.paramsBounds = paramsBounds
        self.paramsLocked['component'] = 'layer'


class ShimOptimizer:
    def __init__(self):
        self.lens = None
        self.shimsList = []
        self.bounds = []
        self.boundsKeys = []
        self.makeBoundsCalled = False
        self.IPeak0 = None
        self.m0 = None

    def set_Lens(self, paramsBounds, paramsLocked):
        self.lens = LensHelper(paramsBounds, paramsLocked)

    def add_Shim(self, paramsBounds, paramsLocked):
        shimID = str(len(self.shimsList))
        self.shimsList.append(ShimHelper(paramsBounds, paramsLocked, shimID))

    def make_Bounds(self):
        assert self.makeBoundsCalled == False
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
        shimDNA_Dict_Temp = shim.paramsLocked
        argsConsumed = 0
        for key, arg in zip(self.boundsKeys, args):
            if key in shim.paramKeys:
                shimDNA_Dict_Temp[key] = arg
                argsConsumed += 1
        shimDNA_Dict = {}
        for key, val in shimDNA_Dict_Temp.items():
            key = key[:-1] if key != 'component' else key
            if key == 'z':
                val += lensDNA_Dict['length'] / 2
            shimDNA_Dict[key] = val
        assert len(shimDNA_Dict) == 8
        return shimDNA_Dict, argsConsumed

    def make_DNA_List_From_Args(self, args):
        assert len(args) == len(self.boundsKeys)
        assert self.makeBoundsCalled == True
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

    def cost_Function(self, args, Print=False):
        lens = self.make_Lens(args)
        if lens.is_Geometry_Valid() == False:
            return np.inf
        IPeak, m = IPeak_And_Magnification_From_Lens(lens)
        if Print == True:
            print(IPeak, m)  # 29.189832542115177 0.9739048904890494
        focusCost = self.IPeak0 / IPeak  # goal to is shrink this
        magCost = 1 + 5.0 * abs(m / self.m0 - 1)  # goal is to keep this the same
        cost = focusCost * magCost
        return cost

    def initialize_Baseline_Values(self, lensBaseLineParams):
        assert len(lensBaseLineParams) == 3
        lensBaseLineParams['component'] = 'layer'
        lensBaseLine = GeneticLens([lensBaseLineParams])
        self.IPeak0, self.m0 = IPeak_And_Magnification_From_Lens(lensBaseLine)

    def optimize(self, lensBaseLineParams):
        self.make_Bounds()
        self.initialize_Baseline_Values(lensBaseLineParams)
        sol = solve_Async(self.cost_Function, self.bounds, 15 * len(self.bounds), timeOut_Seconds=3600 * 4, workers=10)
        print(sol)


def run():
    rp = .05
    L0 = .23
    magnetWidth = .0254
    lensBounds = {'length': (L0 - rp, L0)}
    lensParams = {'rp': rp, 'width': magnetWidth}
    lensBaseLineParams = {'rp': rp, 'width': magnetWidth, 'length': L0}

    shim1ParamBounds = {'r': (rp, rp + magnetWidth), 'phi':(0.0,np.pi/6),'z': (0.0, rp), 'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shim1LockedParams = {'radius': .0254 / 2, 'planeSymmetry': False}

    shim2ParamBounds = {'r': (rp, rp + magnetWidth), 'phi':(0.0,np.pi/6),'z': (-(L0/2+rp), -L0/2), 'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shim2LockedParams = {'radius': .0254 / 2, 'planeSymmetry': False}
    shimOptimizer = ShimOptimizer()
    shimOptimizer.set_Lens(lensBounds, lensParams)
    shimOptimizer.add_Shim(shim1ParamBounds, shim1LockedParams)
    shimOptimizer.add_Shim(shim2ParamBounds, shim2LockedParams)
    shimOptimizer.optimize(lensBaseLineParams)


run()

'''
1 sphere
has z symmetry

POPULATION VARIABILITY: [0.009579   0.007715   0.00096799 0.01281744 0.03542631 0.02044798]
BEST MEMBER BELOW
---population member---- 
DNA: array([ 0.2181784 ,  0.05971335, -0.52356326,  0.013647  ,  1.57591938,
        1.75151608])
cost: 0.5892802938504855
'''

'''
2 sphere
each has z symmetry
one is set to phi=0.0 and the other phi=pi/6.

------ITERATIONS:  2430
POPULATION VARIABILITY: [0.05763707 0.06745681 0.11878127 0.12529942 0.09902359 0.05341342
 0.09657387 0.10569297 0.08088745]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.20239452, 0.0595642 , 0.02210917, 1.45808684, 2.93923315,
       0.05699902, 0.01929926, 1.62861884, 5.21635232])
cost: 0.268153059157226

stopped early, but previously observed minima is ~.2
'''

'''
2 sphere
one at each end, unique



'''