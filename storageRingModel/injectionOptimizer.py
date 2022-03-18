import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import time
from asyncDE import solve_Async
from typing import Union,Optional
import numpy as np
from storageRingOptimizer import LatticeOptimizer
from ParticleTracerLatticeClass import ParticleTracerLattice
from elementPT import HalbachLensSim
import matplotlib.pyplot as plt


V0=210
def is_Valid_Injector_Phase(L_InjectorMagnet, rpInjectorMagnet):
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / V0 ** 2) * BpLens / rpInjectorMagnet ** 2) * L_InjectorMagnet
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True
def generate_Injector_Lattice_Double_Lens(X) -> Optional[ParticleTracerLattice]:
    L_InjectorMagnet1, rpInjectorMagnet1,L_InjectorMagnet2, rpInjectorMagnet2, \
    LmCombiner, rpCombiner,loadBeamDiam,L1,L2,L3=X
    fringeFrac = 1.5
    LMagnet1 = L_InjectorMagnet1 - 2 * fringeFrac * rpInjectorMagnet1
    LMagnet2 = L_InjectorMagnet2 - 2 * fringeFrac * rpInjectorMagnet2
    aspect1,aspect2=LMagnet1/rpInjectorMagnet1,LMagnet2/rpInjectorMagnet2
    if aspect1<1.0 or aspect2<1.0: #horrible fringe field performance
        return None
    if LMagnet1 < 1e-3 or LMagnet2 < 1e-3:  # minimum fringe length must be respected.
        return None
    if loadBeamDiam>rpCombiner: #silly if load beam doens't fit in half of magnet
        return None
    PTL_Injector = ParticleTracerLattice(V0, latticeType='injector')
    PTL_Injector.add_Drift(L1, ap=rpInjectorMagnet1)
    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet1, L_InjectorMagnet1)
    PTL_Injector.add_Drift(L2, ap=max([rpInjectorMagnet1,rpInjectorMagnet2]))
    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet2, L_InjectorMagnet2)
    PTL_Injector.add_Drift(L3, ap=rpInjectorMagnet2)
    PTL_Injector.add_Combiner_Sim_Lens(LmCombiner, rpCombiner,loadBeamDiam=loadBeamDiam,layers=1)
    PTL_Injector.end_Lattice(constrain=False, enforceClosedLattice=False)
    assert PTL_Injector.elList[1].fringeFracOuter==fringeFrac and PTL_Injector.elList[3].fringeFracOuter==fringeFrac
    return PTL_Injector


def generate_Ring_Surrogate_Lattice(X)->ParticleTracerLattice:
    jeremyGap=5e-2
    rpLensLast=.02
    rplensFirst=.02
    lastGap = 5e-2
    L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, \
    LmCombiner, rpCombiner, loadBeamDiam, L1, L2, L3 = X
    PTL_Ring=ParticleTracerLattice(V0,latticeType='storageRing')
    PTL_Ring.add_Drift(.5,ap=rpLensLast) #models the size of the lengs
    PTL_Ring.add_Drift(lastGap)
    PTL_Ring.add_Combiner_Sim_Lens(LmCombiner, rpCombiner,loadBeamDiam=loadBeamDiam,layers=1)
    PTL_Ring.add_Drift(jeremyGap)
    PTL_Ring.add_Halbach_Lens_Sim(rplensFirst, .2)
    PTL_Ring.end_Lattice(enforceClosedLattice=False,constrain=False,surpressWarning=True)  # 17.8 % of time here
    return PTL_Ring

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
        self.latticeInjector.build_Lattice()
        costClose = self.floor_Plan_Cost()
        self.latticeInjector.elList[-2].set_Length(L0-self.tunabilityLength) #move lens towards combiner
        self.latticeInjector.build_Lattice()
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


def injector_Cost(X: Union[np.ndarray,list,tuple]):
    maximumCost=2.0
    L_Injector_TotalMax = 2.0
    try:
        PTL_I=generate_Injector_Lattice_Double_Lens(X)
    except:
        print('exception',repr(X))
        return maximumCost
    if PTL_I is None:
        return maximumCost
    if PTL_I.totalLength>L_Injector_TotalMax:
        return maximumCost
    try:
        PTL_R=generate_Ring_Surrogate_Lattice(X)
    except:
        return maximumCost
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
        except:
            np.set_printoptions(precision=100)
            print('failed with params',repr(np.asarray(X)))
            raise Exception()

    # L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, LmCombiner, rpCombiner,
    # loadBeamDiam, L1, L2, L3
    bounds = [(.05, .3), (.01, .03),(.05, .3), (.01, .03), (.02, .25), (.005, .04),(5e-3,30e-3),(.05,.5),
    (.05,.5),(.05,.3)]
    for _ in range(3):
        print(solve_Async(wrapper, bounds, 15 * len(bounds), surrogateMethodProb=0.05, tol=.03, disp=True,workers=8))
    # X0=np.array([0.15831797, 0.01681428, 0.19546755, 0.02932734, 0.15318842,
    #    0.03942919, 0.01484423, 0.01      , 0.42291596, 0.22561926])
    # injector_Cost(X0)

if __name__=="__main__":
    main()


'''
---population member---- 
DNA: array([0.12524277, 0.02191994, 0.16189484, 0.02631402, 0.16314783,
       0.03999999, 0.01706859, 0.0605528 , 0.27198268, 0.2054394 ])
cost: 0.10423455810539292
finished with total evals:  11732
---population member---- 
DNA: array([0.06567916, 0.01414217, 0.16048129, 0.0251745 , 0.15970735,
       0.03985693, 0.01875451, 0.05      , 0.26854486, 0.2264674 ])
cost: 0.09930996157904261
'''


"""
---population member---- 
DNA: array([0.14625806, 0.02415056, 0.121357  , 0.02123799, 0.19139004,
       0.04      , 0.01525237, 0.05      , 0.19573719, 0.22186834])
cost: 0.14064186273851484

------ITERATIONS:  4200
POPULATION VARIABILITY: [0.10151815 0.06409042 0.09856872 0.15354714 0.11433127 0.0416822
 0.15623459 0.13163136 0.30097587 0.14214971]
BEST MEMBER BELOW
---population member---- 
DNA: array([0.20585844, 0.03      , 0.10421198, 0.02284784, 0.21853755,
       0.04      , 0.01701195, 0.1733694 , 0.30854762, 0.22761157])
cost: 0.15890535569991435
"""