import itertools
from typing import Union, Optional

import numpy as np
from scipy.special import expit as sigmoid

from asyncDE import solve_Async
from constants import DEFAULT_ATOM_SPEED, COST_PER_CUBIC_INCH_PERM_MAGNET, CUBIC_METER_TO_CUBIC_INCH
from latticeElements.elements import HalbachLensSim, CombinerHalbachLensSim
from latticeElements.utilities import ElementDimensionError, ElementTooShortError, CombinerDimensionError
from latticeModels import make_Injector_Version_Any, make_Ring_Surrogate_For_Injection_Version_1, InjectorGeometryError
from latticeModels_Parameters import lockedDict, injectorRingConstraintsV1, injectorParamsBoundsAny
from octopusOptimizer import octopus_Optimize
from storageRingModeler import StorageRingModel


def is_Valid_Injector_Phase(L_InjectorMagnet, rpInjectorMagnet):
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / DEFAULT_ATOM_SPEED ** 2) * BpLens / rpInjectorMagnet ** 2) \
                        * L_InjectorMagnet
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True


class Injection_Model(StorageRingModel):
    maximumCost = 2.0
    L_Injector_TotalMax = 2.0

    def __init__(self, latticeRing, latticeInjector):
        super().__init__(latticeRing, latticeInjector)
        assert len(self.injectorLensIndices) == 2  # i expect this to be two

    def cost(self) -> float:
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        swarmCost = self.injected_Swarm_Cost()
        floorPlanCost = self.floor_Plan_Cost_With_Tunability()
        priceCost = self.get_Rough_Material_Cost()
        cost = floorPlanCost + swarmCost + priceCost
        cost = min([self.maximumCost, cost])
        return cost

    def injected_Swarm_Cost(self) -> float:

        swarmRingTraced = self.inject_And_Trace_Swarm(False)
        numParticlesInitial = self.swarmInjectorInitial.num_Particles(weighted=True)
        numParticlesFinal = swarmRingTraced.num_Particles(weighted=True, unClippedOnly=True)
        swarmCost = (numParticlesInitial - numParticlesFinal) / numParticlesInitial
        assert 0.0 <= swarmCost <= 1.0
        return swarmCost

    def get_Rough_Material_Cost(self) -> float:
        """Get a value proportional to the cost of magnetic materials. This is proportional to the volume of
        magnetic material. Sigmoid is used to scale"""

        volume = 0.0  # volume of magnetic material in cubic inches
        for el in itertools.chain(self.latticeRing.elList, self.latticeInjector):
            if type(el) in (CombinerHalbachLensSim, HalbachLensSim):
                volume += CUBIC_METER_TO_CUBIC_INCH * np.sum(el.Lm * np.array(el.magnetWidths) ** 2)
        price_USD = volume * COST_PER_CUBIC_INCH_PERM_MAGNET
        price_USD_Scale = 5_000.0
        cost = 2 * (sigmoid(price_USD / price_USD_Scale) - .5)
        return cost


def get_Model(paramsInjector: Union[np.ndarray, list, tuple]) -> Optional[Injection_Model]:
    surrogateParams = lockedDict({'rpLens1': injectorRingConstraintsV1['rp1LensMax'], 'rpLens2': .025, 'L_Lens': .5})
    paramsInjectorDict = {}
    for key, val in zip(injectorParamsBoundsAny.keys(), paramsInjector):
        paramsInjectorDict[key] = val
    paramsInjectorDict = lockedDict(paramsInjectorDict)
    PTL_I = make_Injector_Version_Any(paramsInjectorDict)
    if PTL_I.totalLength > Injection_Model.L_Injector_TotalMax:
        return None
    PTL_R = make_Ring_Surrogate_For_Injection_Version_1(paramsInjectorDict, surrogateParams)
    if PTL_R is None:
        return None
    assert PTL_I.combiner.outputOffset == PTL_R.combiner.outputOffset
    return Injection_Model(PTL_R, PTL_I)


def plot_Results(paramsInjector: Union[np.ndarray, list, tuple], trueAspectRatio=False):
    model = get_Model(paramsInjector)
    assert model is not None
    model.show_Floor_Plan_And_Trajectories(trueAspectRatio)


def injector_Cost(paramsInjector: Union[np.ndarray, list, tuple]):
    model = get_Model(paramsInjector)
    cost = Injection_Model.maximumCost if model is None else model.cost()
    assert 0.0 <= cost <= Injection_Model.maximumCost
    return cost


def wrapper(X: Union[np.ndarray, list, tuple]) -> float:
    try:
        return injector_Cost(X)
    except (ElementDimensionError, InjectorGeometryError, ElementTooShortError, CombinerDimensionError):
        return Injection_Model.maximumCost
    except:
        np.set_printoptions(precision=100)
        print('unhandled exception on args: ', repr(X))
        raise Exception


def main():
    bounds = [vals for vals in injectorParamsBoundsAny.values()]

    for _ in range(3):
        member = solve_Async(wrapper, bounds, 15 * len(bounds), tol=.05, disp=True)
        x0 = member.DNA
        posOptimal, costMin = octopus_Optimize(wrapper, bounds, x0, tentacleLength=.02,
                                               numSearchesCriteria=20, maxTrainingMemory=200, disp=True)
        print(repr(posOptimal), costMin)
    # plot_Results(x)


if __name__ == "__main__":
    main()

"""


array([0.09194512, 0.02083394, 0.13107531, 0.02206909, 0.1886514 ,
       0.04      , 0.01460026, 0.19921984, 0.23500085, 0.18705991]) 0.2617156644496601
       
array([0.29      , 0.01388565, 0.19922547, 0.02661814, 0.19020056,
       0.04      , 0.0141065 , 0.08446912, 0.25643251, 0.2207812 ]) 0.2883693354872183
       
array([0.15296214, 0.02929302, 0.12755628, 0.02196965, 0.1947876 ,
       0.04      , 0.01543   , 0.20473507, 0.23142568, 0.18207568]) 0.2634068970471264

"""
