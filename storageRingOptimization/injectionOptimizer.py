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
def is_Valid_Injector_Phase(L_InjectorMagnet, rpInjectorMagnet):
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / V0 ** 2) * BpLens / rpInjectorMagnet ** 2) * L_InjectorMagnet
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True
def generate_Injector_Lattice(X) -> ParticleTracerLattice:
    L_InjectorMagnet, rpInjectorMagnet, LmCombiner, rpCombiner,loadBeamDiam,L1,L2=X
    fringeFrac = 1.5
    maximumMagnetWidth = rpInjectorMagnet * np.tan(2 * np.pi / 24) * 2
    rpInjectorLayers=(rpInjectorMagnet,rpInjectorMagnet+maximumMagnetWidth)
    LMagnet = L_InjectorMagnet - 2 * fringeFrac * max(rpInjectorLayers)
    if LMagnet < 1e-9:  # minimum fringe length must be respected.
        return None
    PTL_Injector = ParticleTracerLattice(V0, latticeType='injector', parallel=False)
    PTL_Injector.add_Drift(L1, ap=.03)
    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorLayers, L_InjectorMagnet, apFrac=.9)
    PTL_Injector.add_Drift(L2, ap=.03)
    try:
        PTL_Injector.add_Combiner_Sim_Lens(LmCombiner, rpCombiner,loadBeamDiam=loadBeamDiam)
    except:
        # print('combiner error')
        return None
    PTL_Injector.end_Lattice(constrain=False, enforceClosedLattice=False)
    return PTL_Injector



def generate_Ring_Surrogate_Lattice(X)->ParticleTracerLattice:
    jeremyGap=2.54e-2
    rpLensLast=.015
    rplensFirst=.015
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
def injector_Cost(X):
    maximumCost=2.0
    L_Injector_TotalMax = 1.0
    PTL_I=generate_Injector_Lattice(X)
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
    assert 0.0<cost<maximumCost
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
    print(solve_Async(wrapper,bounds,15*len(bounds),surrogateMethodProb=0.1,timeOut_Seconds=99000,workers=8))
    # args=[0.09720613, 0.01065874 ,0.09999615 ,0.01263289, 0.0090281,  0.2057647,
 # 0.38057085]
 #    wrapper(args)
if __name__=="__main__":
    main()
'''
[0.16251646 ,0.02351541, 0.13552071 ,0.04318609, 0.01744993 ,0.26296971,
 0.2057939 ]
'''
