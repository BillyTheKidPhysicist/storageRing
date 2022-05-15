#pylint: disable= missing-module-docstring
import itertools
import copy
from typing import Union
import numpy as np
from scipy.optimize import differential_evolution
from storageRingGeometryModules.storageRingGeometry import StorageRingGeometry #pylint: disable= import-error
from storageRingGeometryModules.shapes import Line,Bend,norm_2D #pylint: disable= import-error

realNumber = Union[float, int]
floatTuple=tuple[float,...]
intTuple=tuple[int,...]
realNumberTuple=tuple[Union[float, int],...]
benderAndLensParams=tuple[ tuple[realNumberTuple,...] ,floatTuple ]
lst_arr_tple = Union[list, np.ndarray, tuple]

class NoValidRingConfiguration(Exception):
    pass


class StorageRingGeometryConstraintsSolver:
    """
    Class to take a storage ring geometry and find a configuration of bending parameters and tunable lenses (not
    all lenses are assumed to be tunable, only user indicated lenses) that results in a valid closed storage ring
    """

    def __init__(self, storageRing: StorageRingGeometry, radiusTarget: float):
        """
        :param storageRing: storage ring to solve, which is already roughly closed. A copy is made to
                prevent modifiying original
        :param radiusTarget: Target bending radius of all bending segments. This is approximate because the
            constraints may not allow an exact value, and because this is actually the radius of the bending orbit
        """

        self.storageRing = copy.deepcopy(storageRing)
        self.targetRadius = radiusTarget
        self.tunedLenses = self.get_Tuned_Lenses()
        self.isSameLengthTuneLenses=True

        assert all(type(shape) is not Bend for shape in self.storageRing) #not yet supported

    def get_Tuned_Lenses(self) -> tuple[Line]:
        """Get lenses(Line objects) that have been marked to have their length tunable to satisfy geometry"""

        tunedLenses = []
        for element in self.storageRing:
            if type(element) == Line:
                if element.constrained:
                    tunedLenses.append(element)
        assert len(tunedLenses)!=0 #must be at least one length tunable lenses
        tunedLenses=tuple(tunedLenses)
        return tunedLenses

    def separate_Params(self,params: realNumberTuple)->tuple[floatTuple,intTuple,floatTuple]:
        """
        Take 1D tuple of parameters, and break apart into tuple of each paramter (bending radius, number of magnets in
        bender, and length of lens)

        :param params: storage ring params. (radius_i,numMagnets_i,...,lensLength_j,...) where i is number of benders
            and j is number of length tunable lenses
        :return: seperate tuples of params ((radius_i...),(numMagnets_i,...),(lensLenth_j,...))
        """

        radiusIndexFirt,numMagsIndexFirst, numParamsPerBend, numBends = 0,1, 2, self.storageRing.numBenders
        assert numBends!=0 #does not work without benders
        radiusTuple=tuple(params[radiusIndexFirt:numParamsPerBend*numBends:numParamsPerBend])
        numMagnetsTuple=tuple(params[numMagsIndexFirst:numParamsPerBend*numBends:numParamsPerBend])
        lensLengthTuple=tuple(params[numParamsPerBend*numBends:])
        return radiusTuple,numMagnetsTuple,lensLengthTuple

    def round_Integer_Params(self, params: realNumberTuple) -> realNumberTuple:
        """
        differential_evolution only works on floats. So integer parameters (number of magnets) must be rounded

        :param params: storage ring params. (radius_i,numMagnets_i,...,lensLength_j,...) where i is number of benders
            and j is number modifiable lenses
        :return: params with numMagnets_i rounded to integer
        """
        
        radiusTuple, numMagnetsTuple, lensLengthTuple=self.separate_Params(params)
        numMagnetsTuple=tuple(round(numMags) for numMags in numMagnetsTuple)
        #trick to remake flattened list with integer parameters rounded
        params=[[radius,numMags] for radius,numMags in zip(radiusTuple,numMagnetsTuple)]
        params.append(list(lensLengthTuple))
        params=tuple(itertools.chain(*params)) #flatten list
        return params

    def shape_And_Round_Params(self, params: floatTuple) -> benderAndLensParams:
        """
        Take parameters in 1D tuple, in tuple casted form that differential_evolution supplies, and round integer
        params and shape for updating elements. differential_evolution only works with float so integer values need to
        be rounded to integers from floats

        :param params: (radius_i,numMagnets_i,...,lensLength_j,...) where i is number of benders and j is number of
                    modifiable lenses
        :return: ( ((radius_i,numMagnets_i),...) , (lensLength_j,...) )
        """

        params = self.round_Integer_Params(params)
        radiusTuple, numMagnetsTuple, lensLengthTuple = self.separate_Params(params)
        benderParams = tuple((radius, numMags) for radius, numMags in zip(radiusTuple, numMagnetsTuple))
        lensParams=lensLengthTuple
        return benderParams, lensParams

    def update_Ring(self, params: floatTuple) -> None:
        """Update bender and lens parameters with params in storage ring geometry"""

        bendingParams, lensParams = self.shape_And_Round_Params(params)
        assert len(bendingParams) == self.storageRing.numBenders
        for i, benderParams in enumerate(bendingParams):
            radius, numMagnets = benderParams
            self.storageRing.benders[i].set_Number_Magnets(numMagnets)
            if self.storageRing.numBenders==2:
                self.storageRing.benders[i].set_Radius(radius)
            elif self.storageRing.numBenders==4:
                radius=bendingParams[i//2][0] #both benders in an arc must have same bending radius
                self.storageRing.benders[i].set_Radius(radius)
            else: raise ValueError

        if self.isSameLengthTuneLenses:
            assert len(lensParams)==1
            for lens in self.tunedLenses:
                lens.set_Length(lensParams[0])
        else:
            raise NotImplementedError
        self.storageRing.build()

    def closed_Ring_Cost(self, params: realNumberTuple) -> float:
        """punish if the ring isn't closed"""

        self.update_Ring(params)
        deltaPos, deltaNormal = self.storageRing.get_End_Separation_Vectors()
        closedCost = (norm_2D(deltaPos) + norm_2D(deltaNormal))
        return closedCost

    def number_Magnets_Cost(self, params: floatTuple) -> float:
        """Cost for number of magnets in each bender being different than each other"""

        benderParams, _ = self.shape_And_Round_Params(params)
        numMagnetsList = [numMagnets for _, numMagnets in benderParams]
        assert all(isinstance(num, int) for num in numMagnetsList)
        weight=1 #different numbers of magnets isn't so bad
        cost = weight*sum([abs(a - b) for a, b in itertools.combinations(numMagnetsList, 2)])
        return cost

    def get_Radius_Cost(self, params: floatTuple) -> float:
        """Cost for when bender radii differ from target radius"""

        benderParams, _ = self.shape_And_Round_Params(params)
        radii=self.get_Bender_Params_Radii(benderParams)
        weight = 100
        cost = weight*sum([abs(radius - self.targetRadius) for radius in radii])
        return cost

    def get_Bounds(self) -> tuple[tuple[float, float], ...]:
        """
        Get bounds for differential_evolution

        :return: bounds in shape of [(upper_i,lower_i),...] where i is a paremeter such as bending radius, number of
            magnets, lens length etc.
        """

        anglePerBenderApprox = (2 * np.pi - self.storageRing.combiner.kinkAngle) / self.storageRing.numBenders
        unitCellAngleApprox = self.storageRing.benders[0].lengthSegment / self.targetRadius  # each bender has
        # same segment length
        numMagnetsApprox = round(anglePerBenderApprox / unitCellAngleApprox)
        bounds = [(self.targetRadius * .95, self.targetRadius * 1.05), (numMagnetsApprox * .9, numMagnetsApprox * 1.1)]
        bounds = bounds * self.storageRing.numBenders
        if self.isSameLengthTuneLenses:
            if len(self.tunedLenses)>=1:
                bounds.append((.1, 5.0))  # big overshoot
        else:
            raise NotImplementedError
        bounds = tuple(bounds)  # i want this to be immutable
        return bounds

    def cost(self, params: lst_arr_tple) -> float:
        """Get cost associated with a storage ring configuration from params. Cost comes from ring not being closed from
        end not meeting beginning and/or tangents not being colinear"""

        if not self.is_Within_Radius_Tolerance(params,1e-2):
            return np.inf
        params = tuple(params)  # I want this to be immutable for safety
        # punish if ring is not closed and aligned properly
        closedCost = self.closed_Ring_Cost(params)
        # punish if magnets or radius are not desired values
        magCost = self.number_Magnets_Cost(params)
        # punish if the radius is different from target
        radiusCost = self.get_Radius_Cost(params)
        _cost = (1e-12 + closedCost) * (1 + magCost + radiusCost)  # +1e-6*(magCost+radiusCost)
        assert _cost >= 0.0
        return _cost

    def get_Bender_Params_Radii(self,benderParams: tuple)-> list[float]:
        """Get list of bender radius. For two or 4 benders this is a list of 2 radii. This is actually a hack because
        the differential evolution solver is still working with 4 radius, but here I only pick two of them. I don't
        want each bending quadrant to have it's own bending radius"""

        if self.storageRing.numBenders==2:
            radii=[radius for radius,_ in benderParams]
        elif self.storageRing.numBenders==4:
            radii=[benderParams[0][0],benderParams[2][0]]
        else: raise ValueError
        return radii

    def is_Within_Radius_Tolerance(self, params: floatTuple,tolerance: float ) -> bool:
        """Are the params within the tolerance of bending radiuses? I don't want each bending radius to be very
        different"""

        benderParams, _ = self.shape_And_Round_Params(params)
        radii=self.get_Bender_Params_Radii(benderParams)
        return all(abs(radius - self.targetRadius) < tolerance for radius in radii)

    def solve(self) -> realNumberTuple:
        """
        Find parameters that give a valid storage ring configuration. This will deform self.storageRing from its original
        configuration!

        :return: parametes as (param_i,...) where i is each parameter
        """

        assert self.storageRing.combiner is not None
        bounds = self.get_Bounds()
        closedRingCost = 1e-11
        maxTries=20
        i=0
        while True:
            try:
                seed=42+i #seed must change for each try, but this is repeatable
                terminationCriteria=lambda x, **kwargs:self.closed_Ring_Cost(x)<closedRingCost
                sol = differential_evolution(self.cost, bounds,seed=seed,polish=False,maxiter=5_000,
                                             tol=0.0,atol=0.0,callback=terminationCriteria,mutation=.5,disp=False)
                solutionParams = self.round_Integer_Params(sol.x)
                assert self.closed_Ring_Cost(solutionParams) < closedRingCost
                break
            except:
                i+=1
                if i==maxTries:
                    raise NoValidRingConfiguration
        return solutionParams

    def make_Valid_Storage_Ring(self)-> StorageRingGeometry:
        """
        solve and return a valid storage ring shape. Not guaranteed to be unique

        :return: a valid storage ring. A copy of self.storage ring, which was a copy of the original.
        """

        solutionParams=self.solve()
        self.update_Ring(solutionParams)
        return copy.deepcopy(self.storageRing)
