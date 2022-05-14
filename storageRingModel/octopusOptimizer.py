import os
import multiprocess as mp
from skopt import optimizer

import numpy as np

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from helperTools import *
import skopt
import random

class Octopus:

    def __init__(self,func,bounds: np.ndarray):
        assert callable(func) and len(bounds.shape)==2
        self.tentacleLengthFact = 1e-2
        self.func=func
        self.bounds=bounds.astype(float)
        self.headPosition: np.ndarray=None
        self.tentaclePositions: np.ndarray=None
        self.costMin: float=None
        self.numTentacles: int=round(1.5*mp.cpu_count())
        self.memory: list=[]


    def pick_New_Tentacle_Positions(self)-> None:

        tentacleLengths = self.tentacleLengthFact * (self.bounds[:, 1] - self.bounds[:, 0])
        tentacleBounds = np.column_stack((-tentacleLengths+self.headPosition, tentacleLengths+self.headPosition))
        skopt.sampler.Sobol().generate(tentacleBounds, self.numTentacles)
        numPointsDumb=round(.5*self.numTentacles)
        numPointsSmart=self.numTentacles-numPointsDumb
        positionsDumb = self.random_Tentacle_Positions(tentacleBounds,numPointsDumb)
        positionsSmart =  self.smart_Tentacle_Positions(tentacleBounds,numPointsSmart)

        self.tentaclePositions= [*positionsDumb,*positionsSmart]

    def random_Tentacle_Positions(self,bounds: np.ndarray,numPostions: int)-> np.ndarray:
        positions=skopt.sampler.Sobol().generate(bounds, numPostions)
        for i, params in enumerate(positions):
            positions[i] = np.clip(params, self.bounds[:, 0], self.bounds[:, 1])
        return positions

    def smart_Tentacle_Positions(self,bounds: np.ndarray,numPositions: int)-> np.ndarray:
        maxMemory=100
        validMemory = [(pos,cost) for pos,cost in self.memory if
                       np.all(pos >= bounds[:, 0]) and np.all(pos <= bounds[:, 1])]
        if len(validMemory)<len(bounds):
            return self.random_Tentacle_Positions(bounds,numPositions)
        if len(validMemory)>maxMemory:
            random.shuffle(validMemory)
            validMemory=validMemory[:100]
        opt=skopt.Optimizer(bounds,n_initial_points=0,n_jobs=-1)
        x=[list(pos) for pos,cost in validMemory]
        y=[cost for pos,cost in validMemory]
        opt.tell(x,y)
        positions=np.array(opt.ask(numPositions))
        return positions
        


    def investigate_Results(self,results)-> None:
        if np.min(results)>self.costMin:
            print('didnt find food')
        else:
            print('found food')
            self.headPosition=self.tentaclePositions[np.argmin(results)]
            self.costMin=np.min(results)

    def search_For_Food(self,xi: np.ndarray, costInitial: float=None):
        self.headPosition=xi
        self.costMin = costInitial if costInitial is not None else self.func(self.headPosition)
        for i in range(100):
            print('best of iter: '+str(i), self.costMin, repr(self.headPosition))
            self.pick_New_Tentacle_Positions()
            results = tool_Parallel_Process(self.func, self.tentaclePositions, processes=len(self.tentaclePositions),
                                            resultsAsArray=True)
            self.memory.extend(list(zip(self.tentaclePositions,results)))
            self.investigate_Results(results)

        print('done', self.costMin, repr(self.headPosition))

