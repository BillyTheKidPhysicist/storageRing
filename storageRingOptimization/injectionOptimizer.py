import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from elementPT import HalbachLensSim
import random
import time
from asyncDE import solve_Async
import numpy as np
from latticeOptimizer import LatticeOptimizer
from ParticleTracerLatticeClass import ParticleTracerLattice
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
def generate_Injector_Lattice(L_Injector, rpInjector, LmCombiner, rpCombiner,loadBeamDiam,L1,L2)->ParticleTracerLattice:
    fringeFrac=1.5
    apFrac=.95
    L_InjectorMagnet=L_Injector-2*fringeFrac*rpInjector
    if L_InjectorMagnet<1e-9:  # minimum fringe length must be respected.
        return None
    if loadBeamDiam>rpCombiner: #don't waste time exploring silly configurations
        return None
    PTL_Injector=ParticleTracerLattice(V0,latticeType='injector')
    PTL_Injector.add_Drift(L1,ap=rpInjector)

    PTL_Injector.add_Halbach_Lens_Sim(rpInjector,L_Injector,apFrac=apFrac)
    PTL_Injector.add_Drift(L2,ap=apFrac*rpInjector)
    PTL_Injector.add_Combiner_Sim_Lens(LmCombiner,rpCombiner,loadBeamDiam=loadBeamDiam)
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)
    return PTL_Injector
def generate_Ring_Surrogate_Lattice(X)->ParticleTracerLattice:
    jeremyGap=5e-2
    rpLensLast=.02
    rplensFirst=.02
    lastGap = 5e-2
    L_InjectorMagnet, rpInjectorMagnet, LmCombiner, rpCombiner,loadBeamDiam,L1,L2=X
    PTL_Ring=ParticleTracerLattice(V0,latticeType='storageRing',parallel=False)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensLast,.5)
    PTL_Ring.add_Drift(lastGap)
    try:
        PTL_Ring.add_Combiner_Sim_Lens(LmCombiner, rpCombiner,loadBeamDiam=loadBeamDiam)
    except:
        return None
    PTL_Ring.add_Drift(jeremyGap)
    PTL_Ring.add_Halbach_Lens_Sim(rplensFirst, .1)
    PTL_Ring.end_Lattice(enforceClosedLattice=False,constrain=False,surpressWarning=True)  # 17.8 % of time here
    return PTL_Ring

class Injection_Model(LatticeOptimizer):
    def __init__(self, latticeRing, latticeInjector):
        super().__init__(latticeRing,latticeInjector)
    def cost(self):
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        swarmCost = self.injected_Swarm_Cost()
        assert 0.0<=swarmCost<=1.0
        overLapCost=self.floor_Plan_Cost()
        assert 0.0<=overLapCost<=1.0
        cost=np.sqrt(overLapCost**2+swarmCost**2)
        return cost

    def injected_Swarm_Cost(self):
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            self.swarmInjectorInitial.quick_Copy()
            , self.h, 1.0, parallel=False,
            fastMode=True, copySwarm=False,
            accelerated=True)
        # self.latticeInjector.show_Lattice(swarm=swarmInjectorTraced,showTraceLines=True,trueAspectRatio=False)
        swarmEnd = self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced, copyParticles=False)
        # print(swarmEnd.num_Particles(weighted=True))
        swarmRingInitial = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False,scoot=True)

        swarmRingTraced=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmRingInitial,self.h,1.0,fastMode=True)
        # self.latticeRing.show_Lattice(swarm=swarmRingTraced,finalCoords=False, showTraceLines=True, trueAspectRatio=False)
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
        return swarmCost
    def floor_Plan_Cost(self):
        overlap=self.floor_Plan_OverLap_mm() #units of mm^2
        factor = 300 #units of mm^2
        cost = 2 / (1 + np.exp(-overlap / factor)) - 1
        assert 0.0<=cost<=1.0
        return cost
    def show_Floor_Plan(self):
        shapelyObjectList = self.generate_Shapely_Floor_Plan()
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy,c='black')
        plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        plt.show()
    def floor_Plan_OverLap_mm(self):
        injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame()
        assert len(injectorShapelyObjects)==4
        overlapElIndex = 1
        injectorLensShapely = injectorShapelyObjects[overlapElIndex]
        assert isinstance(self.latticeInjector.elList[overlapElIndex], HalbachLensSim)
        ringShapelyObjects = [el.SO_Outer for el in self.latticeRing.elList]
        area = 0
        converTo_mm = (1e3) ** 2
        for element in ringShapelyObjects:
            area += element.intersection(injectorLensShapely).area * converTo_mm
        area+=injectorShapelyObjects[2].intersection(ringShapelyObjects[0]).area*converTo_mm/10
        return area
def injector_Cost(X):
    maximumCost=2.0
    L_Injector_TotalMax = 1.0
    PTL_I=generate_Injector_Lattice(*X)
    if PTL_I is None:
        return maximumCost
    if PTL_I.totalLength>L_Injector_TotalMax:
        return maximumCost
    PTL_R=generate_Ring_Surrogate_Lattice(X)
    if PTL_R is None:
        return maximumCost
    assert PTL_I.combiner.outputOffset==PTL_R.combiner.outputOffset
    model=Injection_Model(PTL_R,PTL_I)
    cost=model.cost()
    assert 0.0<=cost<=maximumCost
    return cost
def main():
    def wrapper(X):
        try:
            return injector_Cost(X)
        except:
            np.set_printoptions(precision=100)
            print('failed with params',X)
            raise Exception()
    # L_InjectorMagnet, rpInjectorMagnet, LmCombiner, rpCombiner,loadBeamDiam,L1,L2
    bounds = [(.05, .5), (.005, .05), (.02, .2), (.005, .05),(5e-3,30e-3),(.03,.5),(.03,.5)]
    for _ in range(1):
        print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=0.1,tol=.01,disp=True,workers=8))
    # args=[0.15630672 ,0.02294941, 0.14133239, 0.04233863, 0.01828147, 0.2481217,
 # 0.23321294]
 #    print(wrapper(args))
if __name__=="__main__":
    main()

'''
------ITERATIONS: 4410
POPULATION VARIABILITY: [0.02002418 0.00735082 0.03328811 0.03747261 0.04054063 0.01262041
 0.02359532]
BEST MEMBER BELOW
---population member---- 
DNA: [0.36759352 0.01463225 0.14569697 0.04130349 0.01488213 0.14432835
 0.12596972]
cost: 0.34753944232992556
'''
