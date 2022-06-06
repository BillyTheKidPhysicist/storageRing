import itertools
import os
import elementPT
from typing import Union, Optional
import numpy as np
import warnings
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
    # L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, LmCombiner, rpCombiner,
    # loadBeamDiam, L1, L2, L3
    # bounds = [vals for vals in injectorParamsBoundsAny.values()]
    #
    # member = solve_Async(wrapper, bounds, 15 * len(bounds), tol=.05, disp=True)
    # print(repr(member.DNA),member.cost)

    from latticeModels_Parameters import injectorParamsOptimalAny
    # # X0=np.array([0.08326160110590838 , 0.020993060372921774, 0.16088998779932584 ,
    # #        0.024763975149604798, 0.19375652148870226 , 0.0398938436893404  ,
    # #        0.018280132203330864, 0.16047790265328432 , 0.26596808943711425 ,
    # #        0.21231305487552196 ])
    X0 = np.array(list(injectorParamsOptimalAny.values()))
    print(wrapper(X0))
    plot_Results(X0)


if __name__ == "__main__":
    main()

"""

-----------------attack 1: 

coarse: 
BEST MEMBER BELOW
---population member---- 
DNA: array([0.15983706722579363 , 0.03                , 0.1318471593072328  ,
       0.02224089503466501 , 0.20020617195221901 , 0.03912975206758641 ,
       0.014889359039106131, 0.19473657568556155 , 0.2425783750828193  ,
       0.15842445390730578 ])
cost: 0.22240731523646098
finished with total evals:  7935
array([0.15983706722579363 , 0.03                , 0.1318471593072328  ,
       0.02224089503466501 , 0.20020617195221901 , 0.03912975206758641 ,
       0.014889359039106131, 0.19473657568556155 , 0.2425783750828193  ,
       0.15842445390730578 ]) 0.22240731523646098
fine: 

0.19245109021710935 array([0.153132  , 0.02963415, 0.13430734, 0.02258658, 0.19126756,
       0.0418939 , 0.01501351, 0.20163243, 0.23014572, 0.17625537])

-----------------attack 2:

coarse:

DNA: array([0.12262512509801454 , 0.03                , 0.15497825945216434 ,
       0.023588293574130267, 0.17972102518246974 , 0.04                ,
       0.014340208454092672, 0.16881858164260252 , 0.234654880587174   ,
       0.19908381690019694 ])
cost: 0.21081256710195267
finished with total evals:  7211
array([0.12262512509801454 , 0.03                , 0.15497825945216434 ,
       0.023588293574130267, 0.17972102518246974 , 0.04                ,
       0.014340208454092672, 0.16881858164260252 , 0.234654880587174   ,
       0.19908381690019694 ]) 0.21081256710195267



fine: 
38 0.19468989454681276 array([0.13862286, 0.03007318, 0.15414578, 0.02380996, 0.18852086,
       0.04141863, 0.01434852, 0.16452731, 0.23317253, 0.1854686 ])

"""
