"""This module contains a class, Octopus, that helps with polishing the results of my asynchronous differential
evolution. It also contains a function that conveniently wraps it. The approach is inspired by how I imagine a
smart octopus might search for food."""

import random
from typing import Callable,Optional
import multiprocess as mp
import numpy as np
import skopt
from helperTools import tool_Parallel_Process

class Octopus:

    def __init__(self,func: Callable,bounds: np.ndarray, xInitial: np.ndarray,tentacleLengthFactor=1e-2):
        """
        Initialize Octopus object

        :param func: Callable to be optimized. Must accept a sequence of n numbers, and return a single numeric
            value between -inf and inf
        :param bounds: array of shape (n,2) where each row is the bounds of the nth entry in sequence of numbers
            accepted by func
        :param xInitial: initial location of search. Becomes the position of the octopus
        """

        assert callable(func) and bounds.ndim==2 and bounds.shape[1]==2
        assert all(upper>lower for lower,upper in bounds) and xInitial.ndim==1
        assert 0.0<tentacleLengthFactor<=1.0
        self.func =func
        self.bounds=bounds.astype(float)
        self.tentacleLengths=tentacleLengthFactor * (self.bounds[:, 1] - self.bounds[:, 0])
        self.octopusLocation=xInitial
        self.tentaclePositions: np.ndarray=None
        self.numTentacles: int=round(max([1.5*len(bounds),mp.cpu_count()]))
        self.memory: list=[]

    def make_Tentacle_Bounds(self)-> np.ndarray:
        """Get bounds for tentacle exploration. Tentacles reach out from locations of head, and are shorter
        than width of global bounds. Need to make sure than no tentacle reaches outside global bounds"""

        tentacleBounds = np.column_stack((-self.tentacleLengths + self.octopusLocation,
                                          self.tentacleLengths + self.octopusLocation))
        for i, bound in enumerate(tentacleBounds):
            if bound[0]<self.bounds[i, 0]:
                tentacleBounds[i]+=bound[0]-self.bounds[i, 0]
            elif bound[1]>self.bounds[i, 1]:
                tentacleBounds[i] -= bound[1] - self.bounds[i, 1]
        return tentacleBounds

    def get_Cost_Min(self)-> float:
        """Get minimum solution cost from memory"""

        return min(cost for position,cost in self.memory)

    def pick_New_Tentacle_Positions(self)-> None:
        """Determine new positions to place tentacles to search for food (Reduction in minimum cost). Half of tentacle
        positions are determine randomly, other half intelligently with gaussian process when enough historical data
        is present"""

        tentacleBounds = self.make_Tentacle_Bounds()
        numPointsDumb=round(.5*self.numTentacles)
        numPointsSmart=self.numTentacles-numPointsDumb
        positionsDumb = self.random_Tentacle_Positions(tentacleBounds,numPointsDumb)
        positionsSmart =  self.smart_Tentacle_Positions(tentacleBounds,numPointsSmart)
        self.tentaclePositions= [*positionsDumb,*positionsSmart]

    def random_Tentacle_Positions(self,bounds: np.ndarray,numPostions: int)-> np.ndarray:
        """
        Get new positions of tentacles to search for food with low discrepancy pseudorandom sampling

        :param bounds: bounds of parameter space. shape (n,2) where n is dimensionality
        :param numPostions: Number of tentacles positions to generate
        :return:
        """

        positions=skopt.sampler.Sobol().generate(bounds, numPostions)
        return positions

    def smart_Tentacle_Positions(self,bounds: np.ndarray,numPositions: int)-> np.ndarray:
        """Intelligently determine where to put tentacles to search for food. Uses gaussian process regression. Training
        data has minimum size for accuracy for maximum size for computation time considerations"""
        
        maxTrainingMemory=150 #computation grows as n^3, gets much slower for larger numbers
        validMemory = [(pos,cost) for pos,cost in self.memory if
                       np.all(pos >= bounds[:, 0]) and np.all(pos <= bounds[:, 1])]
        print('valid:',len(validMemory))
        if len(validMemory)<len(bounds):
            return self.random_Tentacle_Positions(bounds,numPositions)
        if len(validMemory)>maxTrainingMemory:
            random.shuffle(validMemory)
            validMemory=validMemory[:100]

        opt=skopt.Optimizer(bounds,n_initial_points=0,n_jobs=-1,acq_optimizer_kwargs={"n_restarts_optimizer":10,
                                                                                      "n_points":50_000})
        x=[list(pos) for pos,cost in validMemory]
        y=[cost for pos,cost in validMemory]
        opt.tell(x,y) #train model
        positions=np.array(opt.ask(numPositions)) #get new positions to test from model
        return positions
        


    def investigate_Results(self,results: np.ndarray)-> None:
        """
        Investigate results of function evaluation at tentacle positions. Check format is correct, update location
        of octopus is better results found

        :param results: array of results of shape (m,n) where m is number of results, and n is parameter space
            dimensionality
        :return: None
        """

        assert not np.any(np.isnan(results)) and not np.any(np.abs(results)==np.inf)
        if np.min(results)>self.get_Cost_Min():
            print('didnt find food')
        else:
            print('found food')
            self.octopusLocation=self.tentaclePositions[np.argmin(results)] #octopus gets moved

    def assess_Food_Quantity(self):
        """Run the function being optimized at the parameter space locations of the tentacles. """

        numProcesses=max([self.numTentacles,3*mp.cpu_count()])

        results=tool_Parallel_Process(self.func, self.tentaclePositions, processes=numProcesses,
                              resultsAsArray=True)
        return results
    def search_For_Food(self, costInitial: Optional[float],numSearchesCriteria:Optional[int], searchCutoff: float):
        """
        Send out octopus to search for food (reduction in cost)

        :param costInitial: Cost of initial location of octopus in parameter space. If None, then compute the cost
            at that value before starting
        :param numSearchesCriteria: If this number of the last search for food have not changed by a specified cutoff
            value, then the octopus is done
        :param searchCutoff: cutoff value for use with numSearchesCriteria
        :return: None
        """

        assert numSearchesCriteria is None or (numSearchesCriteria>0 and isinstance(numSearchesCriteria,int))
        assert searchCutoff>0.0

        costInitial = costInitial if costInitial is not None else self.func(self.octopusLocation)
        self.memory.append((self.octopusLocation.copy(),costInitial))
        costMinList=[]
        for i in range(1_000_000):
            print('best of iter: '+str(i), self.get_Cost_Min(), repr(self.octopusLocation))
            self.pick_New_Tentacle_Positions()
            results = self.assess_Food_Quantity()
            self.memory.extend(list(zip(self.tentaclePositions.copy(),results)))
            self.investigate_Results(results)
            costMinList.append(self.get_Cost_Min())
            if numSearchesCriteria is not None and len(costMinList)>numSearchesCriteria:
                if max(costMinList[numSearchesCriteria:])-min(costMinList[numSearchesCriteria:])<searchCutoff:
                    break

        print('done', self.get_Cost_Min(), repr(self.octopusLocation))
        return self.octopusLocation,self.get_Cost_Min()

def octopus_Optimize(func,bounds,xi,costInitial:float=None,numSearchesCriteria:int=10,
                     searchCutoff: float=.01)->tuple[np.ndarray,float]:
    """
    Minimize a scalar function within bounds by octopus optimization. An octopus searches for food
    (reduction in cost function) by a combinations of intelligently and blindly searching with her tentacles in her
    vicinity and moving to better feeding grounds.

    :param func: Function to be minimized in n dimensional parameter space. Must accept array like input
    :param bounds: bounds of parameter space, (n,2) shape.
    :param xi: Numpy array of initial optimal value. This will be the starting location of the octopus
    :param costInitial: Initial cost value at xi. If None, then it will be recalculated before proceeding
    :param numSearchesCriteria: Number of searches with results all falling within a cutoff to trigger termination. If
        None, search proceeds forever.
    :param searchCutoff: The cutoff criteria for numSearchesCriteria
    :return: Tuple as (optimal position in parameter, cost at optimal position)
    """

    octopus=Octopus(func,bounds,xi)
    posOptimal,costMin=octopus.search_For_Food(costInitial,numSearchesCriteria,searchCutoff)
    return posOptimal,costMin