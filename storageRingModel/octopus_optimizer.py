"""This module contains a class, Octopus, that helps with polishing the results of my asynchronous differential
evolution. It also contains a function that conveniently wraps it. The approach is inspired by how I imagine a
smart octopus might search for food."""

import random
from typing import Callable, Optional

import multiprocess as mp
import numpy as np
import skopt

from helper_tools import low_discrepancy_sample


class Octopus:

    def __init__(self, func: Callable, global_bounds: np.ndarray, x_initial: np.ndarray, tentacle_length,
                 max_training_mem, processes):
        """
        Initialize Octopus object

        :param func: Callable to be optimized. Must accept a sequence of n numbers, and return a single numeric
            value between -inf and inf
        :param globalBounds: array of shape (n,2) where each row is the bounds of the nth entry in sequence of numbers
            accepted by func
        :param xInitial: initial location of search. Becomes the position of the octopus
        """
        processes = round(1.5 * mp.cpu_count()) if processes == -1 else processes
        bounds = np.array(global_bounds) if isinstance(global_bounds, (list, tuple)) else global_bounds
        x_initial = np.array(x_initial) if isinstance(x_initial, (list, tuple)) else x_initial
        assert isinstance(x_initial, np.ndarray) and isinstance(bounds, np.ndarray)
        assert callable(func) and bounds.ndim == 2 and bounds.shape[1] == 2
        assert all(upper > lower for lower, upper in bounds) and x_initial.ndim == 1
        assert 0.0 < tentacle_length <= 1.0
        self.func = func
        self.global_search_bounds = bounds.astype(float)  # if int some goofy stuff can happen
        self.tentacle_lengths = tentacle_length * (self.global_search_bounds[:, 1] - self.global_search_bounds[:, 0])
        self.octopus_location = x_initial
        self.tentacle_positions = None
        self.numTentacles: int = round(max([1.5 * len(bounds), processes]))
        self.max_training_mem = max_training_mem
        self.memory: list = []

    def make_tentacle_bounds(self) -> np.ndarray:
        """Get bounds for tentacle exploration. Tentacles reach out from locations of head, and are shorter
        than width of global bounds. Need to make sure than no tentacle reaches outside global bounds"""

        tentacle_bounds = np.column_stack((-self.tentacle_lengths + self.octopus_location,
                                           self.tentacle_lengths + self.octopus_location))
        for i, ((tentLower, tentUpper), (globLower, globUpper)) in enumerate(
                zip(tentacle_bounds, self.global_search_bounds)):
            if tentLower < globLower:
                tentacle_bounds[i] += globLower - tentLower
            elif tentUpper > globUpper:
                tentacle_bounds[i] -= tentUpper - globUpper
        return tentacle_bounds

    def get_cost_min(self) -> float:
        """Get minimum solution cost from memory"""

        return min(cost for position, cost in self.memory)

    def pick_new_tentacle_positions(self) -> None:
        """Determine new positions to place tentacles to search for food (Reduction in minimum cost). Some fraction of
        tentacle positions are determine randomly, others intelligently with gaussian process when enough historical
        data is present"""

        tentacle_bounds = self.make_tentacle_bounds()
        num_smart_min = 5
        if self.numTentacles < num_smart_min:
            num_smart = self.numTentacles
        else:
            fraction_smart = .1
            num_smart = round(fraction_smart * (self.numTentacles - num_smart_min)) + num_smart_min
        num_random = self.numTentacles - num_smart
        rand_tentacle_positions = self.random_tentacle_positions(tentacle_bounds, num_random)
        smart_tentacle_positions = self.smart_tentacle_positions(tentacle_bounds, num_smart)
        self.tentacle_positions = np.row_stack((rand_tentacle_positions, *smart_tentacle_positions))
        assert len(self.tentacle_positions) == self.numTentacles

    def random_tentacle_positions(self, bounds: np.ndarray, num_postions: int) -> np.ndarray:
        """Get new positions of tentacles to search for food with low discrepancy pseudorandom sampling"""

        positions = low_discrepancy_sample(bounds, num_postions)
        return positions

    def smart_tentacle_positions(self, bounds: np.ndarray, num_positions) -> np.ndarray:
        """Intelligently determine where to put tentacles to search for food. Uses gaussian process regression. Training
        data has minimum size for accuracy for maximum size for computation time considerations"""
        valid_memory = [(pos, cost) for pos, cost in self.memory if
                        np.all(pos >= bounds[:, 0]) and np.all(pos <= bounds[:, 1])]
        if len(valid_memory) < 2 * len(bounds):
            return self.random_tentacle_positions(bounds, num_positions)
        if len(valid_memory) > self.max_training_mem:
            random.shuffle(valid_memory)  # so the model can change
            valid_memory = valid_memory[:self.max_training_mem]
        # base_estimator = cook_estimator("GP", space=bounds,noise=.005)
        opt = skopt.Optimizer(bounds, n_initial_points=0, n_jobs=-1,
                              acq_optimizer_kwargs={"n_restarts_optimizer": 10, "n_points": 30_000}, acq_func="EI")

        x = [list(pos) for pos, cost in valid_memory]
        y = [cost for pos, cost in valid_memory]
        opt.tell(x, y)  # train model
        positions = np.array(opt.ask(num_positions))
        return positions

    def investigate_results(self, results: np.ndarray, disp: bool) -> None:
        """
        Investigate results of function evaluation at tentacle positions. Check format is correct, update location
        of octopus is better results found

        :param results: array of results of shape (m,n) where m is number of results, and n is parameter space
            dimensionality
        :return: None
        """

        assert not np.any(np.isnan(results)) and not np.any(np.abs(results) == np.inf)
        if np.min(results) > self.get_cost_min():
            message = 'didnt find food'
        else:
            message = 'found food'
            self.octopus_location = self.tentacle_positions[np.argmin(results)]  # octopus gets moved
        if disp:
            print(message)

    def assess_food_quantity(self, processes: int):
        """Run the function being optimized at the parameter space locations of the tentacles. """

        if processes == -1 or processes > 1:
            num_processes = min([self.numTentacles, 3 * mp.cpu_count()]) if processes == -1 else processes
            with mp.Pool(num_processes) as pool:  # pylint: disable=not-callable
                results = np.array(pool.map(self.func, self.tentacle_positions, chunksize=1))
        else:
            results = np.array([self.func(pos) for pos in self.tentacle_positions])
        return results

    # pylint: disable=too-many-arguments
    def search_for_food(self, cost_initial: Optional[float], num_searches_criteria: Optional[int], search_cutoff: float,
                        processes: int, disp: bool, memory: Optional[list]):
        """ Send out octopus to search for food (reduction in cost) """

        assert num_searches_criteria is None or (num_searches_criteria > 0 and isinstance(num_searches_criteria, int))
        assert search_cutoff > 0.0
        if memory is not None:
            assert all(isinstance(x, np.ndarray) and isinstance(val, float) for x, val in memory)
            self.memory = memory

        cost_initial = cost_initial if cost_initial is not None else self.func(self.octopus_location)
        self.memory.append((self.octopus_location.copy(), cost_initial))
        cost_min_list = []
        for i in range(1_000_000):
            if disp:
                print('best of iter: ' + str(i), self.get_cost_min(), repr(self.octopus_location))
            self.pick_new_tentacle_positions()
            results = self.assess_food_quantity(processes)
            self.memory.extend(list(zip(self.tentacle_positions.copy(), results)))
            self.investigate_results(results, disp)
            cost_min_list.append(self.get_cost_min())
            if num_searches_criteria is not None and len(cost_min_list) > num_searches_criteria + 1:
                if max(cost_min_list[-num_searches_criteria:]) - min(
                        cost_min_list[-num_searches_criteria:]) < search_cutoff:
                    break
        if disp:
            print('done', self.get_cost_min(), repr(self.octopus_location))
        return self.octopus_location, self.get_cost_min()


# pylint: disable=too-many-arguments
def octopus_optimize(func, bounds, xi, cost_initial: float = None, num_searches_criteria: int = 10,
                     search_cutoff: float = .01, processes: int = -1, disp: bool = True, tentacle_length: float = .01,
                     memory: list = None, max_training_mem: int = 250) -> tuple[np.ndarray, float]:
    """
    Minimize a scalar function within bounds by octopus optimization. An octopus searches for food
    (reduction in cost function) by a combinations of intelligently and blindly searching with her tentacles in her
    vicinity and moving to better feeding grounds.

    :param func: Function to be minimized in n dimensional parameter space. Must accept array like input
    :param bounds: bounds of parameter space, (n,2) shape.
    :param xi: Numpy array of initial optimal value. This will be the starting location of the octopus
    :param cost_initial: Initial cost value at xi. If None, then it will be recalculated before proceeding
    :param num_searches_criteria: Number of searches with results all falling within a cutoff to trigger termination. If
        None, search proceeds forever.
    :param search_cutoff: The cutoff criteria for num_searches_criteria
    :param processes: -1 to search for results using all processors, 1 for serial search, >1 to specify number of
        processes
    :param disp: Whether to display results of solver per iteration
    :param tentacle_length: The distance that each tentacle can reach. Expressed as a fraction of the separation between
        min and max of bounds for each dimension
    :param memory: List of previous results to use for optimizer
    :param max_training_mem: Maximum number of samples to use to build gaussian process fit.
    :return: Tuple as (optimal position in parameter, cost at optimal position)
    """

    octopus = Octopus(func, bounds, xi, tentacle_length, max_training_mem, processes)
    pos_optimal, cost_min = octopus.search_for_food(cost_initial, num_searches_criteria, search_cutoff, processes, disp,
                                                    memory)
    return pos_optimal, cost_min
