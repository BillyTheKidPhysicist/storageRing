import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
from typing import Union,Optional
import numpy as np
import warnings
from constants import DEFAULT_ATOM_SPEED
from storageRingOptimizer import LatticeOptimizer
from ParticleTracerLatticeClass import ParticleTracerLattice,ElementDimensionError,ElementTooShortError,\
    CombinerDimensionError
from elementPT import HalbachLensSim
import matplotlib.pyplot as plt
from latticeModels import make_Injector_Version_1,make_Ring_Surrogate_Version_1,InjectorGeometryError



def is_Valid_Injector_Phase(L_InjectorMagnet, rpInjectorMagnet):
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / DEFAULT_ATOM_SPEED** 2) * BpLens / rpInjectorMagnet ** 2) \
                        * L_InjectorMagnet
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True



class Injection_Model(LatticeOptimizer):

    def __init__(self, latticeRing, latticeInjector,tunabilityLength: float=2e-2):
        super().__init__(latticeRing,latticeInjector)
        self.tunabilityLength=tunabilityLength

    def cost(self)-> float:
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        swarmCost = self.injected_Swarm_Cost()
        assert 0.0<=swarmCost<=1.0
        floorPlanCost=self.floor_Plan_Cost_With_Tunability()
        assert 0.0<=floorPlanCost<=1.0
        cost=np.sqrt(floorPlanCost**2+swarmCost**2)
        return cost

    def injected_Swarm_Cost(self)-> float:
        fastMode=True
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            self.swarmInjectorInitial.quick_Copy()
            , self.h, 1.0, parallel=False,
            fastMode=fastMode, copySwarm=False,
            accelerated=True)
        # self.latticeInjector.show_Lattice(swarm=swarmInjectorTraced,showTraceLines=True,trueAspectRatio=False,traceLineAlpha=.25)
        swarmEnd = self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced, copyParticles=False)
        # print(swarmEnd.num_Particles(weighted=True))
        swarmRingInitial = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False,scoot=True)

        swarmRingTraced=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmRingInitial,self.h,1.0,fastMode=fastMode)
        # self.latticeRing.show_Lattice(swarm=swarmRingTraced,finalCoords=False, showTraceLines=True,
        #                               trueAspectRatio=False)

        #only count particle survived if they make it to the end of the last element of the ring surrogate. I do some
        #lazy trick here with the width of the end of the last element and the range that a particle could be in that
        #width and have made it to the end
        ne=self.latticeRing.elList[-1].ne
        endAngle=abs(ne[1]/ne[0])
        xMax=self.latticeRing.elList[-1].r2[0]
        numSurvivedWeighted=0.0
        for particle in swarmRingTraced:
            if (particle.qf is None) ==False:
                xFinal=particle.qf[0]
                slantWidth=endAngle*self.latticeRing.elList[-1].ap
                survived=abs(xFinal-xMax)<slantWidth+2*self.h*np.linalg.norm(particle.pf)
                numSurvivedWeighted+=particle.probability if survived==True else 0.0
        # print(numSurvivedWeighted)
        swarmCost = (self.swarmInjectorInitial.num_Particles(weighted=True) - numSurvivedWeighted) \
                                / self.swarmInjectorInitial.num_Particles(weighted=True)
        # self.show_Floor_Plan()
        return swarmCost

    def floor_Plan_Cost_With_Tunability(self)-> float:
        """Measure floor plan cost at nominal position, and at maximum spatial tuning displacement in each direction.
        Return the largest value of the three"""
        L0=self.latticeInjector.elList[-2].L #value before tuning
        cost0=self.floor_Plan_Cost()
        self.latticeInjector.elList[-2].set_Length(L0+self.tunabilityLength) #move lens away from combiner
        self.latticeInjector.build_Lattice(False)
        costClose = self.floor_Plan_Cost()
        self.latticeInjector.elList[-2].set_Length(L0-self.tunabilityLength) #move lens towards combiner
        self.latticeInjector.build_Lattice(False)
        costFar = self.floor_Plan_Cost()
        self.latticeInjector.elList[-2].set_Length(L0) #reset
        return max([cost0,costClose,costFar])

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
        """Not a great approach. Overlap is measured for the last lens of the injector and each element in the ring.
        A crude model of the ring blocking the injector is achieved by using a less weight ovelap of injector
        drift element right before combiner and ring lens"""
        injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame()
        injectorOverlapLensIndex=-3
        injectorLensShapely = injectorShapelyObjects[injectorOverlapLensIndex]
        injectorLastDrift=injectorShapelyObjects[4]
        assert isinstance(self.latticeInjector.elList[injectorOverlapLensIndex],HalbachLensSim)
        ringShapelyObjects = [el.SO_Outer for el in self.latticeRing.elList]
        area = 0
        converTo_mm=(1e3)**2
        for element in ringShapelyObjects:
            area += element.intersection(injectorLensShapely).area*converTo_mm
        area += injectorLastDrift.intersection(ringShapelyObjects[0]).area * converTo_mm / 10 #crude method of
        #punishing for ring lens blocking path through injector to combiner
        return area

maximumCost=2.0
surrogateParams={'rpLens1':.01,'rpLens2':.025,'L_Lens':.3}
L_Injector_TotalMax = 2.0


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


def main():
    def wrapper(X: Union[np.ndarray,list,tuple]):
        try:
            return injector_Cost(X)
        except (ElementDimensionError,InjectorGeometryError,ElementTooShortError,CombinerDimensionError):
            return maximumCost
        except:
            np.set_printoptions(precision=100)
            print('unhandled exception on args: ',repr(X))
            raise Exception
    # L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, LmCombiner, rpCombiner,
    # loadBeamDiam, L1, L2, L3
    bounds = [(.05, .3), (.01, .03),(.05, .3), (.01, .03), (.02, .25), (.005, .04),(5e-3,30e-3),(.05,.5),
    (.05,.5),(.05,.3)]
    for _ in range(3):
        print(solve_Async(wrapper, bounds, 15 * len(bounds), tol=.03, disp=True,workers=8))
    # X0=np.array([0.25060918, 0.02310519, 0.09129291, 0.01658076, 0.18996305,
    #    0.03871766, 0.0239747 , 0.05      , 0.40325806, 0.15953472])
    # print(wrapper(X0))
if __name__=="__main__":
    main()


"""
BEST MEMBER BELOW
---population member---- 
DNA: array([0.25060918, 0.02310519, 0.09129291, 0.01658076, 0.18996305,
       0.03871766, 0.0239747 , 0.05      , 0.40325806, 0.15953472])
cost: 0.10791484489389803
"""
