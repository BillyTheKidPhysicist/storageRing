import itertools
import os
import elementPT
from asyncDE import solve_Async
from typing import Union,Optional
import numpy as np
import warnings
from constants import DEFAULT_ATOM_SPEED,COST_PER_CUBIC_INCH_PERM_MAGNET
from storageRingOptimizer import LatticeOptimizer
from ParticleTracerLatticeClass import ElementDimensionError,ElementTooShortError,CombinerDimensionError
from elementPT import HalbachLensSim
import matplotlib.pyplot as plt
from latticeModels import make_Injector_Version_Any,make_Ring_Surrogate_Version_1,InjectorGeometryError
from latticeModels_Parameters import constantsV1,lockedDict,injectorRingConstraintsV1
from scipy.special import expit as sigmoid
import dill



def is_Valid_Injector_Phase(L_InjectorMagnet, rpInjectorMagnet):
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / DEFAULT_ATOM_SPEED** 2) * BpLens / rpInjectorMagnet ** 2) \
                        * L_InjectorMagnet
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True

CUBIC_METER_TO_INCH=61023.7


class Injection_Model(LatticeOptimizer):

    def __init__(self, latticeRing, latticeInjector,tunabilityLength: float=2e-2):
        super().__init__(latticeRing,latticeInjector)
        self.tunabilityLength=tunabilityLength
        assert len(self.injectorLensIndices)==2 # i expect this to be two

    def cost(self)-> float:
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        swarmCost = self.injected_Swarm_Cost()
        assert 0.0<=swarmCost<=1.0
        floorPlanCost=self.floor_Plan_Cost_With_Tunability()
        assert 0.0<=floorPlanCost<=1.0
        priceCost=self.get_Rough_Material_Cost()
        cost=np.sqrt(floorPlanCost**2+swarmCost**2+priceCost**2)
        return cost

    def injected_Swarm_Cost(self)-> float:

        swarmRingTraced=self.inject_And_Trace_Swarm(None,False,False)
        numParticlesInitial=self.swarmInjectorInitial.num_Particles(weighted=True)
        numParticlesFinal=swarmRingTraced.num_Particles(weighted=True,unClippedOnly=True)
        swarmCost=(numParticlesInitial-numParticlesFinal)/numParticlesInitial

        return swarmCost

    def get_Drift_After_Second_Lens_Injector(self)-> elementPT.Drift:

        drift=self.latticeInjector.elList[self.injectorLensIndices[-1]+1]
        assert type(drift) is elementPT.Drift
        return drift

    def floor_Plan_Cost_With_Tunability(self)-> float:
        """Measure floor plan cost at nominal position, and at maximum spatial tuning displacement in each direction.
        Return the largest value of the three"""

        driftAfterLens=self.get_Drift_After_Second_Lens_Injector()
        L0=driftAfterLens.L #value before tuning
        cost=[self.floor_Plan_Cost(None)]
        for separation in (-self.tunabilityLength,self.tunabilityLength):
            driftAfterLens.set_Length(L0+separation) #move lens away from combiner
            self.latticeInjector.build_Lattice(False)
            cost.append(self.floor_Plan_Cost(None))
        driftAfterLens.set_Length(L0) #reset
        self.latticeInjector.build_Lattice(False)
        return max(cost)

    def get_Rough_Material_Cost(self)-> float:
        """Get a value proportional to the cost of magnetic materials. This is proportional to the volume of
        magnetic material. Sigmoid is used to scale"""

        volume=0.0 #volume of magnetic material in cubic inches
        for el in itertools.chain(self.latticeRing.elList,self.latticeInjector):
            if type(el) in (elementPT.CombinerHalbachLensSim,elementPT.HalbachLensSim):
                volume+=CUBIC_METER_TO_INCH*np.sum(el.Lm*np.array(el.magnetWidths)**2)
        price_USD=volume*COST_PER_CUBIC_INCH_PERM_MAGNET
        price_USD_Scale=5_000.0
        cost=2*(sigmoid(price_USD/price_USD_Scale)-.5)
        return cost



maximumCost=2.0

L_Injector_TotalMax = 2.0
surrogateParams=lockedDict({'rpLens1':injectorRingConstraintsV1['rp1LensMax'],'rpLens2':.025,'L_Lens':.5})

def get_Model(paramsInjector: Union[np.ndarray,list,tuple])-> Optional[Injection_Model]:

    PTL_I = make_Injector_Version_Any(paramsInjector)
    if PTL_I.totalLength > L_Injector_TotalMax:
        return None
    PTL_R = make_Ring_Surrogate_Version_1(paramsInjector, surrogateParams)
    if PTL_R is None:
        return None
    assert PTL_I.combiner.outputOffset == PTL_R.combiner.outputOffset
    return Injection_Model(PTL_R, PTL_I)

def plot_Results(paramsInjector: Union[np.ndarray,list,tuple], trueAspectRatio=False):
    model=get_Model(paramsInjector)
    assert model is not None
    model.show_Floor_Plan_And_Trajectories(None,trueAspectRatio)

def injector_Cost(paramsInjector: Union[np.ndarray,list,tuple]):
    model=get_Model(paramsInjector)
    if model is None:
        cost= maximumCost
    else:
        cost=model.cost()
    assert 0.0<=cost<=maximumCost
    return cost

def wrapper(X: Union[np.ndarray,list,tuple])-> float:
    try:
        return injector_Cost(X)
    except (ElementDimensionError,InjectorGeometryError,ElementTooShortError,CombinerDimensionError):
        return maximumCost
    except:
        np.set_printoptions(precision=100)
        print('unhandled exception on args: ',repr(X))
        raise Exception

def main():

    # L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, LmCombiner, rpCombiner,
    # # loadBeamDiam, L1, L2, L3
    bounds = [(.05, .3), (.01, .03),(.05, .3), (.01, .03), (.02, .25), (.005, .04),(5e-3,30e-3),(.05,.5),
    (.05,.5),(.05,.3)]


    # valveAp=[constants_Version1["lens1ToLens2_Valve_Ap"],
    #          constants_Version1["lens1ToLens2_Valve_Ap"]+.001,
    #          constants_Version1["lens1ToLens2_Valve_Ap"]+.002,
    #          constants_Version1["lens1ToLens2_Valve_Ap"]+.004]
    # deltaOPAp=[constants_Version1["OP_MagAp"]-.001]#,
                # constants_Version1["OP_MagAp"]+.001,
               # constants_Version1["OP_MagAp"]+.002,
               # constants_Version1["OP_MagAp"]+.004]
    # for val in deltaOPAp:
    #     constants_Version1["OP_MagAp"]=val
    #     print('OP is:', constants_Version1["OP_MagAp"])
    #     member = solve_Async(wrapper, bounds, 15 * len(bounds), tol=.05, disp=True, workers=9)

    #
    X0=np.array([0.29374941, 0.01467768, 0.22837003, 0.0291507 , 0.19208822,
       0.04      , 0.01462034, 0.08151122, 0.27099428, 0.26718875])
    print(wrapper(X0))
    # plot_Results(X0)
if __name__=="__main__":
    main()

"""

with op = 2.0 cm

BEST MEMBER BELOW
---population member---- 
DNA: array([0.15388971, 0.03      , 0.13261248, 0.02266874, 0.20860142,
       0.04      , 0.01729315, 0.14818731, 0.28535098, 0.20815677])
cost: 0.33858069480205766


with op =2.2 cm (default)

---population member---- 
DNA: array([0.29374941, 0.01467768, 0.22837003, 0.0291507 , 0.19208822,
       0.04      , 0.01462034, 0.08151122, 0.27099428, 0.26718875])
cost: 0.25502537159259564
finished with total evals:  10252


op =2.4 cm

BEST MEMBER BELOW
---population member---- 
DNA: array([0.11286314, 0.03      , 0.22316751, 0.03      , 0.18818945,
       0.04      , 0.0159573 , 0.11879764, 0.26109759, 0.29286133])
cost: 0.23677264350310473






"""
