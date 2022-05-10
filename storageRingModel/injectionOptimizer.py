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
from latticeModels import make_Injector_Version_1,make_Ring_Surrogate_Version_1,InjectorGeometryError
from latticeModels_Constants import constants_Version1,lockedDict
from scipy.special import expit
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
        self.injectorLensIndices=[i for i,el in enumerate(self.latticeInjector) if type(el) is HalbachLensSim]
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
        fastMode=True
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            self.swarmInjectorInitial.quick_Copy(), self.h, 1.0, parallel=False,
            fastMode=fastMode, copySwarm=False, accelerated=True,logPhaseSpaceCoords=True)
        swarmEnd = self.move_Survived_Particles_From_Injector_Combiner_To_Origin(swarmInjectorTraced, copyParticles=False)
        # print(swarmEnd.num_Particles(weighted=True))
        swarmRingInitial = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False,scoot=True)
        swarmRingTraced=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmRingInitial,self.h,1,fastMode=fastMode)
        numParticlesInitial=self.swarmInjectorInitial.num_Particles(weighted=True)
        numParticlesFinal=swarmRingTraced.num_Particles(weighted=True,unClippedOnly=True)
        swarmCost=(numParticlesInitial-numParticlesFinal)/numParticlesInitial

        return swarmCost

    def get_Drift_After_Second_Lens(self)-> elementPT.Drift:

        drift=self.latticeInjector.elList[self.injectorLensIndices[-1]+1]
        assert type(drift) is elementPT.Drift
        return drift

    def floor_Plan_Cost_With_Tunability(self)-> float:
        """Measure floor plan cost at nominal position, and at maximum spatial tuning displacement in each direction.
        Return the largest value of the three"""

        driftAfterLens=self.get_Drift_After_Second_Lens()
        L0=driftAfterLens.L #value before tuning
        cost=[self.floor_Plan_Cost()]
        for separation in (-self.tunabilityLength,self.tunabilityLength):
            driftAfterLens.set_Length(L0+separation) #move lens away from combiner
            self.latticeInjector.build_Lattice(False)
            cost.append(self.floor_Plan_Cost())
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
        cost=2*(expit(price_USD/price_USD_Scale)-.5)
        return cost

    def floor_Plan_Cost(self)-> float:
        overlap=self.floor_Plan_OverLap_mm() #units of mm^2
        factor = 300 #units of mm^2
        cost = 2 / (1 + np.exp(-overlap / factor)) - 1
        assert 0.0<=cost<=1.0
        return cost

    def show_Floor_Plan(self)-> None:
        shapelyObjectList = self.generate_Shapely_Object_List_Of_Floor_Plan()
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy,c='black')
        plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        plt.show()

    def floor_Plan_OverLap_mm(self):
        """Find the area overlap between the element before the second injector lens, and the first lens in the ring
        surrogate"""

        firstLensRing=self.latticeRing.elList[0]
        assert type(firstLensRing) is HalbachLensSim
        firstLensRingShapely=firstLensRing.SO_Outer
        injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame()
        converTo_mm = 1e3 ** 2
        area=0.0
        for i in range(self.injectorLensIndices[-1]+1): #don't forget to add 1
            area += firstLensRingShapely.intersection(injectorShapelyObjects[i]).area * converTo_mm
        return area


maximumCost=2.0

L_Injector_TotalMax = 2.0
surrogateParams=lockedDict({'rpLens1':.01,'rpLens2':.025,'L_Lens':.5})

def injector_Cost(paramsInjector: Union[np.ndarray,list,tuple]):
    PTL_I=make_Injector_Version_1(paramsInjector)
    if PTL_I.totalLength>L_Injector_TotalMax:
        return maximumCost
    PTL_R=make_Ring_Surrogate_Version_1(paramsInjector,surrogateParams)
    if PTL_R is None:
        return maximumCost
    assert PTL_I.combiner.outputOffset==PTL_R.combiner.outputOffset
    model=Injection_Model(PTL_R,PTL_I)
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
    # loadBeamDiam, L1, L2, L3
    bounds = [(.05, .3), (.01, .03),(.05, .3), (.01, .03), (.02, .25), (.005, .04),(5e-3,30e-3),(.05,.5),
    (.05,.5),(.05,.3)]



    for _ in range(1):
        member = solve_Async(wrapper, bounds, 15 * len(bounds), tol=.05, disp=True, workers=9)
        print(member.DNA, member.cost)

    # X0=np.array([0.3       , 0.03      , 0.142369  , 0.02148764, 0.25      ,
    #    0.04      , 0.03      , 0.21738463, 0.5       , 0.3       ])
    # print(wrapper(X0))
if __name__=="__main__":
    main()

"""


---population member---- 
DNA: array([0.29784632, 0.02633112, 0.24822711, 0.02495677, 0.14805594,
       0.03944398, 0.01564161, 0.32696537, 0.22914697, 0.3       ])
cost: 0.22292088281208783
finished with total evals:  19620



"""
