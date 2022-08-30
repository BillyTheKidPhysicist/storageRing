# pylint: disable= missing-module-docstring
import copy
import itertools

import numpy as np
from scipy.optimize import differential_evolution

from storage_ring_placement.shapes import Line, Bend, norm_2D  # pylint: disable= import-error
from storage_ring_placement.storage_ring_geometry import StorageRingGeometry, Shape  # pylint: disable= import-error
from type_hints import RealNumTuple, FloatTuple, sequence, IntTuple

BenderAndLensParams = tuple[tuple[RealNumTuple, ...], FloatTuple]


class NoValidRingConfiguration(RuntimeError):
    pass


class ExcessiveRingCost(ValueError):
    pass


class StorageRingGeometryConstraintsSolver:
    """
    Class to take a storage ring geometry and find a configuration of bending parameters and tunable lenses (not
    all lenses are assumed to be tunable, only user indicated lenses) that results in a valid closed storage ring
    """

    def __init__(self, storage_ring: StorageRingGeometry, radius_target: float):
        """
        :param storage_ring: storage ring to solve, which is already roughly closed. A copy is made to
                prevent modifiying original
        :param radius_target: Target bending radius of all bending segments. This is approximate because the
            constraints may not allow an exact value, and because this is actually the radius of the bending orbit
        """

        self.storage_ring = copy.deepcopy(storage_ring)
        self.radius_target = radius_target
        self.tuned_lenses = self.get_tuned_lenses()
        self.is_same_length_tuned_lenses = True

        assert all(type(shape) is not Bend for shape in self.storage_ring)  # not yet supported

    def get_tuned_lenses(self) -> tuple[Shape]:
        """Get lenses(Line objects) that have been marked to have their length tunable to satisfy geometry"""
        tuned_lenses = tuple([el for el in self.storage_ring if (type(el) is Line and el.constrained)])
        return tuned_lenses

    def separate_params(self, params: RealNumTuple) -> tuple[FloatTuple, IntTuple, FloatTuple]:
        """
        Take 1D tuple of parameters, and break apart into tuple of each paramter (bending radius, number of magnets in
        bender, and length of lens)

        :param params: storage ring params. (radius_i,num_magnets_i,...,lensLength_j,...) where i is number of benders
            and j is number of length tunable lenses
        :return: seperate tuples of params ((radius_i...),(num_magnets_i,...),(lensLenth_j,...))
        """

        index_first_radius, index_first_num_mags, num_params_per_bend, num_bends = 0, 1, 2, self.storage_ring.numBenders
        assert num_bends != 0  # does not work without benders
        radii = tuple(params[index_first_radius:num_params_per_bend * num_bends:num_params_per_bend])
        nums_of_magnets = tuple(params[index_first_num_mags:num_params_per_bend * num_bends:num_params_per_bend])
        lens_lengths = tuple(params[num_params_per_bend * num_bends:])
        return radii, nums_of_magnets, lens_lengths

    def round_integer_params(self, params: RealNumTuple) -> RealNumTuple:
        """
        differential_evolution only works on floats. So integer parameters (number of magnets) must be rounded

        :param params: storage ring params. (radius_i,num_magnets_i,...,lensLength_j,...) where i is number of benders
            and j is number modifiable lenses
        :return: params with num_magnets_i rounded to integer
        """

        radii, nums_of_magnets, lens_lengths = self.separate_params(params)
        nums_of_magnets = tuple(round(numMags) for numMags in nums_of_magnets)
        # trick to remake flattened list with integer parameters rounded
        params = [[radius, numMags] for radius, numMags in zip(radii, nums_of_magnets)]
        params.append(list(lens_lengths))
        params = tuple(itertools.chain(*params))  # flatten list
        return params

    def shape_and_round_params(self, params: FloatTuple) -> BenderAndLensParams:
        """
        Take parameters in 1D tuple, in tuple casted form that differential_evolution supplies, and round integer
        params and shape for updating elements. differential_evolution only works with float so integer values need to
        be rounded to integers from floats

        :param params: (radius_i,num_magnets_i,...,lensLength_j,...) where i is number of benders and j is number of
                    modifiable lenses
        :return: ( ((radius_i,num_magnets_i),...) , (lensLength_j,...) )
        """

        params = self.round_integer_params(params)
        radii, nums_of_magnets, lens_lengths = self.separate_params(params)
        bender_params = tuple((radius, numMags) for radius, numMags in zip(radii, nums_of_magnets))
        lens_params = lens_lengths
        return bender_params, lens_params

    def update_ring(self, params: FloatTuple) -> None:
        """Update bender and lens parameters with params in storage ring geometry"""

        bending_params, lens_params = self.shape_and_round_params(params)
        assert len(bending_params) == self.storage_ring.numBenders
        for i, bender_params in enumerate(bending_params):
            radius, num_magnets = bender_params
            self.storage_ring.benders[i].set_number_magnets(num_magnets)
            if self.storage_ring.numBenders == 2:
                self.storage_ring.benders[i].set_radius(radius)
            elif self.storage_ring.numBenders == 4:
                radius = bending_params[i // 2][0]  # both benders in an arc must have same bending radius
                self.storage_ring.benders[i].set_radius(radius)
            else:
                raise NotImplementedError

        if self.is_same_length_tuned_lenses:
            for lens in self.tuned_lenses:
                lens.set_length(lens_params[0])
        else:
            raise NotImplementedError
        self.storage_ring.build()

    def closed_ring_cost(self, params: RealNumTuple) -> float:
        """punish if the ring isn't closed"""

        self.update_ring(params)
        delta_pos, delta_normal = self.storage_ring.get_end_separation_vectors()
        closed_cost = (norm_2D(delta_pos) + norm_2D(delta_normal))
        return closed_cost

    def number_magnets_cost(self, params: FloatTuple) -> float:
        """Cost for number of magnets in each bender being different than each other"""
        bender_params, _ = self.shape_and_round_params(params)
        nums_of_magnets = [num_magnets for _, num_magnets in bender_params]
        assert all(isinstance(num, int) for num in nums_of_magnets)
        weight = 1  # different numbers of magnets isn't so bad
        cost = weight * sum([abs(a - b) for a, b in itertools.combinations(nums_of_magnets, 2)])
        return cost

    def get_radius_cost(self, params: FloatTuple) -> float:
        """Cost for when bender radii differ from target radius"""

        bender_params, _ = self.shape_and_round_params(params)
        radii = self.get_bender_params_radii(bender_params)
        weight = 100
        cost = weight * sum([abs(radius - self.radius_target) for radius in radii])
        return cost

    def get_bounds(self) -> tuple[tuple[float, float], ...]:
        """
        Get bounds for differential_evolution

        :return: bounds in shape of [(upper_i,lower_i),...] where i is a paremeter such as bending radius, number of
            magnets, lens length etc.
        """
        combiner_angle = 0 if self.storage_ring.combiner is None else self.storage_ring.combiner.kinkAngle
        angle_per_bender_approx = (2 * np.pi - combiner_angle) / self.storage_ring.numBenders
        unit_cell_angle_approx = self.storage_ring.benders[0].length_seg / self.radius_target  # each bender has
        # same segment length
        num_magnets_approx = round(angle_per_bender_approx / unit_cell_angle_approx)
        bounds = [(self.radius_target * .95, self.radius_target * 1.05),
                  (num_magnets_approx * .9, num_magnets_approx * 1.1)]
        bounds = bounds * self.storage_ring.numBenders
        if self.is_same_length_tuned_lenses:
            if len(self.tuned_lenses) >= 1:
                bounds.append((.1, 5.0))  # big overshoot
        bounds = tuple(bounds)  # i want this to be immutable
        return bounds

    def cost(self, params: sequence) -> float:
        """Get cost associated with a storage ring configuration from params. Cost comes from ring not being closed from
        end not meeting beginning and/or tangents not being colinear"""

        params = tuple(params)  # I want this to be immutable for safety
        if not self.is_within_radius_tol(params, 1e-2):
            return np.inf
        # punish if ring is not closed and aligned properly
        closed_cost = self.closed_ring_cost(params)
        # punish if magnets or radius are not desired values
        mag_cost = self.number_magnets_cost(params)
        # punish if the radius is different from target
        radius_cost = self.get_radius_cost(params)
        _cost = (1e-12 + closed_cost) * (1 + mag_cost + radius_cost)  # +1e-6*(mag_cost+radiusCost)
        assert _cost >= 0.0
        return _cost

    def get_bender_params_radii(self, bender_params: tuple) -> list[float]:
        """Get list of bender radius. For two or 4 benders this is a list of 2 radii. This is actually a hack because
        the differential evolution solver is still working with 4 radius, but here I only pick two of them. I don't
        want each bending quadrant to have it's own bending radius"""

        if self.storage_ring.numBenders == 2:
            radii = [radius for radius, _ in bender_params]
        elif self.storage_ring.numBenders == 4:
            radii = [bender_params[0][0], bender_params[2][0]]
        else:
            raise ValueError
        return radii

    def is_within_radius_tol(self, params: FloatTuple, tolerance: float) -> bool:
        """Are the params within the tolerance of bending radiuses? I don't want each bending radius to be very
        different"""

        bender_params, _ = self.shape_and_round_params(params)
        radii = self.get_bender_params_radii(bender_params)
        return all(abs(radius - self.radius_target) < tolerance for radius in radii)

    def solve(self) -> RealNumTuple:
        """
        Find parameters that give a valid storage ring configuration. This will deform self.storage_ring from its original
        configuration!

        :return: parameters as (param_i,...) where i is each parameter
        """

        bounds = self.get_bounds()
        closed_ring_cost = 1e-11
        tries, max_tries = 0, 20
        done = False
        while not done:
            try:
                seed = 42 + tries  # seed must change for each try, but this is repeatable

                def termination_criteria(x, **kwargs):
                    return self.closed_ring_cost(x) < closed_ring_cost

                sol = differential_evolution(self.cost, bounds, seed=seed, polish=False, maxiter=1_000,
                                             tol=0.0, atol=0.0, callback=termination_criteria, disp=False)

                solution_params = self.round_integer_params(sol.x)
                if self.closed_ring_cost(solution_params) > closed_ring_cost:
                    raise ExcessiveRingCost
                else:
                    done = True
            except ExcessiveRingCost:
                tries += 1
            if tries >= max_tries:
                raise NoValidRingConfiguration
        return solution_params

    def make_valid_storage_ring(self) -> StorageRingGeometry:
        """
        solve and return a valid storage ring shape. Not guaranteed to be unique

        :return: a valid storage ring. A copy of self.storage ring, which was a copy of the original.
        """

        solution_params = self.solve()
        self.update_ring(solution_params)
        return copy.deepcopy(self.storage_ring)
