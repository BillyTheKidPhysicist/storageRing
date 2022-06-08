import itertools
import os
import elementPT
from typing import Union, Optional
import numpy as np
from asyncDE import solve_Async
from constants import DEFAULT_ATOM_SPEED, COST_PER_CUBIC_INCH_PERM_MAGNET
from storageRingModeler import StorageRingModel
from ParticleTracerLatticeClass import ElementDimensionError, ElementTooShortError, CombinerDimensionError
from latticeModels import make_Injector_Version_Any, make_Ring_Surrogate_For_Injection_Version_1, InjectorGeometryError
from latticeModels_Parameters import lockedDict, injectorRingConstraintsV1, injectorParamsBoundsAny
from scipy.special import expit as sigmoid


def is_Valid_Injector_Phase(L_InjectorMagnet, rpInjectorMagnet):
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / DEFAULT_ATOM_SPEED ** 2) * BpLens / rpInjectorMagnet ** 2) \
                        * L_InjectorMagnet
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True


CUBIC_METER_TO_INCH = 61023.7


class Injection_Model(StorageRingModel):

    def __init__(self, latticeRing, latticeInjector, tunabilityLength: float = 2e-2):
        super().__init__(latticeRing, latticeInjector)
        self.tunabilityLength = tunabilityLength
        assert len(self.injectorLensIndices) == 2  # i expect this to be two

    def cost(self) -> float:
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        swarmCost = self.injected_Swarm_Cost()
        assert 0.0 <= swarmCost <= 1.0
        floorPlanCost = self.floor_Plan_Cost_With_Tunability()
        assert 0.0 <= floorPlanCost <= 1.0
        priceCost = self.get_Rough_Material_Cost()
        cost = np.sqrt(floorPlanCost ** 2 + swarmCost ** 2 + priceCost ** 2)
        return cost

    def injected_Swarm_Cost(self) -> float:

        swarmRingTraced = self.inject_And_Trace_Swarm(None, False, False)
        numParticlesInitial = self.swarmInjectorInitial.num_Particles(weighted=True)
        numParticlesFinal = swarmRingTraced.num_Particles(weighted=True, unClippedOnly=True)
        swarmCost = (numParticlesInitial - numParticlesFinal) / numParticlesInitial

        return swarmCost

    def get_Drift_After_Second_Lens_Injector(self) -> elementPT.Drift:

        drift = self.latticeInjector.elList[self.injectorLensIndices[-1] + 1]
        assert type(drift) is elementPT.Drift
        return drift

    def floor_Plan_Cost_With_Tunability(self) -> float:
        """Measure floor plan cost at nominal position, and at maximum spatial tuning displacement in each direction.
        Return the largest value of the three"""

        driftAfterLens = self.get_Drift_After_Second_Lens_Injector()
        L0 = driftAfterLens.L  # value before tuning
        cost = [self.floor_Plan_Cost(None)]
        for separation in (-self.tunabilityLength, self.tunabilityLength):
            driftAfterLens.set_Length(L0 + separation)  # move lens away from combiner
            self.latticeInjector.build_Lattice(False)
            cost.append(self.floor_Plan_Cost(None))
        driftAfterLens.set_Length(L0)  # reset
        self.latticeInjector.build_Lattice(False)
        return max(cost)

    def get_Rough_Material_Cost(self) -> float:
        """Get a value proportional to the cost of magnetic materials. This is proportional to the volume of
        magnetic material. Sigmoid is used to scale"""

        volume = 0.0  # volume of magnetic material in cubic inches
        for el in itertools.chain(self.latticeRing.elList, self.latticeInjector):
            if type(el) in (elementPT.CombinerHalbachLensSim, elementPT.HalbachLensSim):
                volume += CUBIC_METER_TO_INCH * np.sum(el.Lm * np.array(el.magnetWidths) ** 2)
        price_USD = volume * COST_PER_CUBIC_INCH_PERM_MAGNET
        price_USD_Scale = 5_000.0
        cost = 2 * (sigmoid(price_USD / price_USD_Scale) - .5)
        return cost


maximumCost = 2.0

L_Injector_TotalMax = 2.0


def get_Model(paramsInjector: Union[np.ndarray, list, tuple]) -> Optional[Injection_Model]:
    surrogateParams = lockedDict({'rpLens1': injectorRingConstraintsV1['rp1LensMax'], 'rpLens2': .025, 'L_Lens': .5})
    paramsInjectorDict = {}
    for key, val in zip(injectorParamsBoundsAny.keys(), paramsInjector):
        paramsInjectorDict[key] = val
    paramsInjectorDict = lockedDict(paramsInjectorDict)
    PTL_I = make_Injector_Version_Any(paramsInjectorDict)
    if PTL_I.totalLength > L_Injector_TotalMax:
        return None
    PTL_R = make_Ring_Surrogate_For_Injection_Version_1(paramsInjectorDict, surrogateParams)
    if PTL_R is None:
        return None
    assert PTL_I.combiner.outputOffset == PTL_R.combiner.outputOffset
    return Injection_Model(PTL_R, PTL_I)


def plot_Results(paramsInjector: Union[np.ndarray, list, tuple], trueAspectRatio=False):
    model = get_Model(paramsInjector)
    assert model is not None
    model.show_Floor_Plan_And_Trajectories(None, trueAspectRatio)


def injector_Cost(paramsInjector: Union[np.ndarray, list, tuple]):
    model = get_Model(paramsInjector)
    if model is None:
        cost = maximumCost
    else:
        cost = model.cost()
    assert 0.0 <= cost <= maximumCost
    return cost


def wrapper(X: Union[np.ndarray, list, tuple]) -> float:
    try:
        return injector_Cost(X)
    except (ElementDimensionError, InjectorGeometryError, ElementTooShortError, CombinerDimensionError):
        return maximumCost
    except:
        np.set_printoptions(precision=100)
        print('unhandled exception on args: ', repr(X))
        raise Exception


def main():

    bounds = [vals for vals in injectorParamsBoundsAny.values()]
    #
    member = solve_Async(wrapper, bounds, 15 * len(bounds), tol=.1, disp=True)
    print('optimal',repr(member.DNA),member.cost)

    x0=member.DNA
    from octopusOptimizer import octopus_Optimize
    octopus_Optimize(wrapper,bounds,x0,tentacleLength=.02,numSearchesCriteria=20,maxTrainingMemory=200)
    # print(wrapper(X0))
    # plot_Results(X0)


if __name__ == "__main__":
    main()

"""

"""
