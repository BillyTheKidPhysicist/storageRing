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
            fastMode=fastMode, copySwarm=False, accelerated=False,logPhaseSpaceCoords=True)
        swarmEnd = self.move_Particles_From_Injector_Combiner_End_To_Origin(swarmInjectorTraced, copyParticles=False)
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
        cost=2*(sigmoid(price_USD/price_USD_Scale)-.5)
        return cost

    def floor_Plan_Cost(self)-> float:
        overlap=self.floor_Plan_OverLap_mm() #units of mm^2
        factor = 300 #units of mm^2
        cost = 2 / (1 + np.exp(-overlap / factor)) - 1
        assert 0.0<=cost<=1.0
        return cost

    def show_Floor_Plan(self, which: str= 'exterior',deferPltShow=False,trueAspect=True, linestyle: str='-',
                        color: str='black')-> None:

        shapelyObjectList = self.generate_Shapely_Object_List_Of_Floor_Plan(which)
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy,c=color,linestyle=linestyle)
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        if trueAspect:
            plt.gca().set_aspect('equal')
        if not deferPltShow:
            plt.show()

    def floor_Plan_OverLap_mm(self)-> float:
        """Find the area overlap between the element before the second injector lens, and the first lens in the ring
        surrogate"""

        firstLensRing=self.latticeRing.elList[0]
        assert type(firstLensRing) is HalbachLensSim
        firstLensRingShapely=firstLensRing.SO_Outer
        injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame('exterior')
        converTo_mm = 1e3 ** 2
        area=0.0
        for i in range(self.injectorLensIndices[-1]+1): #don't forget to add 1
            area += firstLensRingShapely.intersection(injectorShapelyObjects[i]).area * converTo_mm
        return area

    def show_Floor_Plan_And_Trajectories(self)-> None:
        """Tracer particles through the lattices, and plot the results. Interior and exterior of element is shown"""

        self.show_Floor_Plan(deferPltShow=True,trueAspect=False,color='grey')
        self.show_Floor_Plan(which='interior',deferPltShow=True,trueAspect=False,linestyle=':')
        self.swarmInjectorInitial.particles=self.swarmInjectorInitial.particles[:200]
        fastMode=False
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            self.swarmInjectorInitial.quick_Copy(), 2e-6, 1.0, parallel=False,
            fastMode=fastMode, copySwarm=False, accelerated=False,logPhaseSpaceCoords=True)
        swarmEnd = self.move_Particles_From_Injector_Combiner_End_To_Origin(swarmInjectorTraced,
                                                                            copyParticles=True,onlyUnclipped=False)
        swarmRingInitial = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False,scoot=True)
        swarmRingTraced=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmRingInitial,2e-6,1,fastMode=fastMode)

        for particleInj,particleRing in zip(swarmInjectorTraced,swarmRingTraced):
            assert not (particleInj.clipped and not particleRing.clipped) #this wouldn't make sense
            color='r' if particleRing.clipped else 'g'
            if particleInj.qArr is not None and len(particleInj.qArr)>1:
                qRingArr=np.array([self.convert_Injector_Coord_To_Ring_Frame(q) for q in particleInj.qArr])
                plt.plot(qRingArr[:,0],qRingArr[:,1],c=color,alpha=.3)
                if particleInj.clipped: #if clipped in injector, plot last location
                    plt.scatter(qRingArr[-1,0],qRingArr[-1,1],marker='x',zorder=100,c=color)
            if particleRing.qArr is not None and len(particleRing.qArr)>1:
                plt.plot(particleRing.qArr[:, 0], particleRing.qArr[:, 1], c=color,alpha=.3)
                if not particleInj.clipped: #if not clipped in injector plot last ring location
                    plt.scatter(particleRing.qArr[-1, 0], particleRing.qArr[-1, 1],marker='x',zorder=100,c=color)
        plt.show()


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
    # # loadBeamDiam, L1, L2, L3
    bounds = [(.05, .3), (.01, .03),(.05, .3), (.01, .03), (.02, .25), (.005, .04),(5e-3,30e-3),(.05,.5),
    (.05,.5),(.05,.3)]


    # valveAp=[constants_Version1["lens1ToLens2_Valve_Ap"],
    #          constants_Version1["lens1ToLens2_Valve_Ap"]+.001,
    #          constants_Version1["lens1ToLens2_Valve_Ap"]+.002,
    #          constants_Version1["lens1ToLens2_Valve_Ap"]+.004]
    # deltaOPAp=[constants_Version1["OP_MagAp"],
    #            constants_Version1["OP_MagAp"]+.001,
    #            constants_Version1["OP_MagAp"]+.002,
    #            constants_Version1["OP_MagAp"]+.004]

    #
    #
    #
    # for _ in range(3):
    #     member = solve_Async(wrapper, bounds, 15 * len(bounds), tol=.05, disp=True, workers=9)
    #     print(member.DNA, member.cost)

    X0=np.array([0.3       , 0.03      , 0.29610018, 0.0119179 , 0.1838093 ,
       0.03845377, 0.0130857 , 0.27998204, 0.45111831, 0.08009889])
    print(wrapper(X0))
if __name__=="__main__":
    main()

"""


 BEST MEMBER BELOW
---population member---- 
DNA: array([0.29088071, 0.02645018, 0.24901717, 0.02470027, 0.15714022,
       0.04      , 0.01491993, 0.35098667, 0.22986842, 0.2684908 ])
cost: 0.29012723528821527


---population member---- 
DNA: array([0.29390206, 0.02634514, 0.27490015, 0.02700695, 0.15667666,
       0.03956025, 0.01542462, 0.33072091, 0.24101027, 0.3       ])
cost: 0.28270904333730534


BEST MEMBER BELOW
---population member---- 
DNA: array([0.29509052, 0.02626593, 0.29127314, 0.02814894, 0.16666863,
       0.03926284, 0.01619822, 0.33503927, 0.23289526, 0.2913201 ])
cost: 0.28639844796818503



"""
