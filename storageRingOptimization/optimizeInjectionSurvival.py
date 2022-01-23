import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import random
import time
from asyncDE import solve_Async
import numpy as np
from latticeKnobOptimizer import LatticeOptimizer
from ParticleTracerLatticeClass import ParticleTracerLattice
from SwarmTracerClass import SwarmTracer
from ParticleTracerClass import ParticleTracer
from ParticleClass import Swarm
import matplotlib.pyplot as plt

V0=210
def is_Valid_Injector_Phase(injectorFactor, rpInjectorFactor):
    LInjector = injectorFactor * .15
    rpInjector = rpInjectorFactor * .02
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / V0 ** 2) * BpLens / rpInjector ** 2) * LInjector
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True
def generate_Injector_Lattice(X) -> ParticleTracerLattice:
    parallel=False
    injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner,L1,L2=X
    assert type(parallel) == bool
    if is_Valid_Injector_Phase(injectorFactor, rpInjectorFactor) == False:
        return None
    LInjector = injectorFactor * .15
    rpInjector = rpInjectorFactor * .02
    fringeFrac = 1.5
    LMagnet = LInjector - 2 * fringeFrac * rpInjector
    if LMagnet < 1e-9:  # minimum fringe length must be respected.
        return None
    PTL_Injector = ParticleTracerLattice(V0, latticeType='injector', parallel=parallel)
    PTL_Injector.add_Drift(L1, ap=.025)
    PTL_Injector.add_Halbach_Lens_Sim(rpInjector, LInjector, apFrac=.9)
    PTL_Injector.add_Drift(L2, ap=.01)
    try:
        PTL_Injector.add_Combiner_Sim_Lens(LmCombiner, rpCombiner)
    except:
        # print('combiner error')
        return None
    PTL_Injector.end_Lattice(constrain=False, enforceClosedLattice=False)
    return PTL_Injector



def generate_Ring_Surrogate_Lattice(X,parallel=False)->ParticleTracerLattice:
    jeremyGap=2.54e-2
    rpLensLast=.015
    rplensFirst=.015
    LLens=.4
    injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner,L1,L2 = X
    assert type(parallel)==bool
    PTL_Ring=ParticleTracerLattice(V0,latticeType='storageRing',parallel=parallel)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensLast,LLens)
    try:
        PTL_Ring.add_Combiner_Sim_Lens(LmCombiner, rpCombiner)
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
        swarmEnd = self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced, copyParticles=False)
        swarmRingInitial = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False,scoot=True)

        swarmRingTraced=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmRingInitial,self.h,1.0,fastMode=False)
        ne=self.latticeRing.elList[-1].ne
        endAngle=abs(ne[1]/ne[0])
        xMax=self.latticeRing.elList[-1].r2[0]
        numSurvivedWeighted=0.0
        for particle in swarmRingTraced:
            xFinal=particle.qf[0]
            slantWidth=endAngle*self.latticeRing.elList[-1].ap
            survived=abs(xFinal-xMax)<slantWidth+2*self.h*np.linalg.norm(particle.pf)
            numSurvivedWeighted+=particle.probability if survived==True else 0.0
        swarmCost = (self.swarmInjectorInitial.num_Particles(weighted=True) - numSurvivedWeighted) \
                                / self.swarmInjectorInitial.num_Particles(weighted=True)
        return swarmCost
    def floor_Plan_Cost(self):
        overlap=self.floor_Plan_OverLap()
        factor = 1e-3
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
def injector_Cost(X):
    maximumCost=2.0
    PTL_I=generate_Injector_Lattice(X)
    if PTL_I is None:
        return maximumCost
    PTL_R=generate_Ring_Surrogate_Lattice(X)
    if PTL_R is None:
        return maximumCost
    model=Injection_Model(PTL_R,PTL_I)
    cost=model.cost()
    assert cost<maximumCost
    return cost
def main():
    # injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner,rpFirst,L1,L2
    bounds = [(.2, 2), (.2, 2), (.02, .2), (.005, .05),(.05,.3),(.05,.3)]
    print(solve_Async(injector_Cost,bounds,15*len(bounds),surrogateMethodProb=0.1,timeOut_Seconds=1e12))
if __name__=="__main__":
    main()
'''
BEST MEMBER BELOW
---population member---- 
DNA: [0.65197469 0.68726815 0.11550685 0.03046567 0.13747125 0.07512276]
cost: 0.26565685183227844
'''
