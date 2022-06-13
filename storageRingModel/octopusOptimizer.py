"""This module contains a class, Octopus, that helps with polishing the results of my asynchronous differential
evolution. It also contains a function that conveniently wraps it. The approach is inspired by how I imagine a
smart octopus might search for food."""

import random
from typing import Callable, Optional

import multiprocess as mp
import numpy as np
import skopt

from helperTools import low_Discrepancy_Sample


class Octopus:

    def __init__(self, func: Callable, globalBounds: np.ndarray, xInitial: np.ndarray, tentacleLength,
                 maxTrainingMemory, processes):
        """
        Initialize Octopus object

        :param func: Callable to be optimized. Must accept a sequence of n numbers, and return a single numeric
            value between -inf and inf
        :param globalBounds: array of shape (n,2) where each row is the bounds of the nth entry in sequence of numbers
            accepted by func
        :param xInitial: initial location of search. Becomes the position of the octopus
        """
        processes = round(1.5*mp.cpu_count()) if processes == -1 else processes
        bounds = np.array(globalBounds) if isinstance(globalBounds, (list, tuple)) else globalBounds
        xInitial = np.array(xInitial) if isinstance(xInitial, (list, tuple)) else xInitial
        assert isinstance(xInitial, np.ndarray) and isinstance(bounds, np.ndarray)
        assert callable(func) and bounds.ndim == 2 and bounds.shape[1] == 2
        assert all(upper > lower for lower, upper in bounds) and xInitial.ndim == 1
        assert 0.0 < tentacleLength <= 1.0
        self.func = func
        self.globalSearchBounds = bounds.astype(float)  # if int some goofy stuff can happen
        self.tentacleLengths = tentacleLength * (self.globalSearchBounds[:, 1] - self.globalSearchBounds[:, 0])
        self.octopusLocation = xInitial
        self.tentaclePositions: np.ndarray = None
        self.numTentacles: int = round(max([1.5 * len(bounds), processes]))
        self.maxTrainingMemory = maxTrainingMemory
        self.memory: list = []

    def make_Tentacle_Bounds(self) -> np.ndarray:
        """Get bounds for tentacle exploration. Tentacles reach out from locations of head, and are shorter
        than width of global bounds. Need to make sure than no tentacle reaches outside global bounds"""

        tentacleBounds = np.column_stack((-self.tentacleLengths + self.octopusLocation,
                                          self.tentacleLengths + self.octopusLocation))
        for i, ((tentLower, tentUpper), (globLower, globUpper)) in enumerate(
                zip(tentacleBounds, self.globalSearchBounds)):
            if tentLower < globLower:
                tentacleBounds[i] += globLower - tentLower
            elif tentUpper > globUpper:
                tentacleBounds[i] -= tentUpper - globUpper
        return tentacleBounds

    def get_Cost_Min(self) -> float:
        """Get minimum solution cost from memory"""

        return min(cost for position, cost in self.memory)

    def pick_New_Tentacle_Positions(self) -> None:
        """Determine new positions to place tentacles to search for food (Reduction in minimum cost). Some fraction of
        tentacle positions are determine randomly, others intelligently with gaussian process when enough historical
        data is present"""

        tentacleBounds = self.make_Tentacle_Bounds()
        numSmartMin=5
        if self.numTentacles<numSmartMin:
            numSmart=self.numTentacles
        else:
            fractionSmart = .1
            numSmart = round(fractionSmart * (self.numTentacles-numSmartMin))+numSmartMin
        numRandom = self.numTentacles - numSmart
        randTentaclePositions = self.random_Tentacle_Positions(tentacleBounds, numRandom)
        smartTentaclePositions = self.smart_Tentacle_Positions(tentacleBounds, numSmart)
        self.tentaclePositions = np.row_stack((randTentaclePositions, *smartTentaclePositions))
        assert len(self.tentaclePositions) == self.numTentacles

    def random_Tentacle_Positions(self, bounds: np.ndarray, numPostions: int) -> np.ndarray:
        """Get new positions of tentacles to search for food with low discrepancy pseudorandom sampling"""

        positions = low_Discrepancy_Sample(bounds, numPostions)
        return positions

    def smart_Tentacle_Positions(self, bounds: np.ndarray, numPositions) -> np.ndarray:
        """Intelligently determine where to put tentacles to search for food. Uses gaussian process regression. Training
        data has minimum size for accuracy for maximum size for computation time considerations"""
        validMemory = [(pos, cost) for pos, cost in self.memory if
                       np.all(pos >= bounds[:, 0]) and np.all(pos <= bounds[:, 1])]
        if len(validMemory) < 2 * len(bounds):
            return self.random_Tentacle_Positions(bounds, numPositions)
        if len(validMemory) > self.maxTrainingMemory:
            random.shuffle(validMemory)  # so the model can change
            validMemory = validMemory[:self.maxTrainingMemory]
        # base_estimator = cook_estimator("GP", space=bounds,noise=.005)
        opt = skopt.Optimizer(bounds, n_initial_points=0, n_jobs=-1,
                              acq_optimizer_kwargs={"n_restarts_optimizer": 10, "n_points": 30_000}, acq_func="EI")

        x = [list(pos) for pos, cost in validMemory]
        y = [cost for pos, cost in validMemory]
        opt.tell(x, y)  # train model
        positions = np.array(opt.ask(numPositions))
        return positions

    def investigate_Results(self, results: np.ndarray, disp: bool) -> None:
        """
        Investigate results of function evaluation at tentacle positions. Check format is correct, update location
        of octopus is better results found

        :param results: array of results of shape (m,n) where m is number of results, and n is parameter space
            dimensionality
        :return: None
        """

        assert not np.any(np.isnan(results)) and not np.any(np.abs(results) == np.inf)
        if np.min(results) > self.get_Cost_Min():
            message='didnt find food'
        else:
            message='found food'
            self.octopusLocation = self.tentaclePositions[np.argmin(results)]  # octopus gets moved
        if disp:
            print(message)

    def assess_Food_Quantity(self, processes: int):
        """Run the function being optimized at the parameter space locations of the tentacles. """

        if processes == -1 or processes > 1:
            numProcesses = min([self.numTentacles, 3 * mp.cpu_count()]) if processes == -1 else processes
            with mp.Pool(numProcesses) as pool:  # pylint: disable=not-callable
                results = np.array(pool.map(self.func, self.tentaclePositions, chunksize=1))
        else:
            results = np.array([self.func(pos) for pos in self.tentaclePositions])
        return results

    # pylint: disable=too-many-arguments
    def search_For_Food(self, costInitial: Optional[float], numSearchesCriteria: Optional[int], searchCutoff: float,
                        processes: int, disp: bool, memory: Optional[list]):
        """ Send out octopus to search for food (reduction in cost) """

        assert numSearchesCriteria is None or (numSearchesCriteria > 0 and isinstance(numSearchesCriteria, int))
        assert searchCutoff > 0.0
        if memory is not None:
            assert all(isinstance(x, np.ndarray) and isinstance(val, float) for x, val in memory)
            self.memory = memory

        costInitial = costInitial if costInitial is not None else self.func(self.octopusLocation)
        self.memory.append((self.octopusLocation.copy(), costInitial))
        costMinList = []
        for i in range(1_000_000):
            if disp:
                print('best of iter: ' + str(i), self.get_Cost_Min(), repr(self.octopusLocation))
            self.pick_New_Tentacle_Positions()
            results = self.assess_Food_Quantity(processes)
            self.memory.extend(list(zip(self.tentaclePositions.copy(), results)))
            self.investigate_Results(results, disp)
            costMinList.append(self.get_Cost_Min())
            if numSearchesCriteria is not None and len(costMinList) > numSearchesCriteria + 1:
                if max(costMinList[-numSearchesCriteria:]) - min(costMinList[-numSearchesCriteria:]) < searchCutoff:
                    break
        if disp:
            print('done', self.get_Cost_Min(), repr(self.octopusLocation))
        return self.octopusLocation, self.get_Cost_Min()


# pylint: disable=too-many-arguments
def octopus_Optimize(func, bounds, xi, costInitial: float = None, numSearchesCriteria: int = 10,
                     searchCutoff: float = .01, processes: int = -1, disp: bool = True, tentacleLength: float = .01,
                     memory: list = None, maxTrainingMemory: int = 250) -> tuple[np.ndarray, float]:
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
    :param processes: -1 to search for results using all processors, 1 for serial search, >1 to specify number of
        processes
    :param disp: Whether to display results of solver per iteration
    :param tentacleLength: The distance that each tentacle can reach. Expressed as a fraction of the separation between
        min and max of bounds for each dimension
    :param memory: List of previous results to use for optimizer
    :param maxTrainingMemory: Maximum number of samples to use to build gaussian process fit.
    :return: Tuple as (optimal position in parameter, cost at optimal position)
    """

    octopus = Octopus(func, bounds, xi, tentacleLength, maxTrainingMemory, processes)
    posOptimal, costMin = octopus.search_For_Food(costInitial, numSearchesCriteria, searchCutoff, processes, disp,
                                                  memory)
    return posOptimal, costMin
