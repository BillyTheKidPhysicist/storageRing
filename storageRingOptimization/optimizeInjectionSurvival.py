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


def is_Valid_Injector_Phase(injectorFactor, rpInjectorFactor):
    LInjector = injectorFactor * .15
    rpInjector = rpInjectorFactor * .02
    BpLens = .7
    injectorLensPhase = np.sqrt((2 * 800.0 / 200 ** 2) * BpLens / rpInjector ** 2) * LInjector
    if np.pi < injectorLensPhase or injectorLensPhase < np.pi / 10:
        # print('bad lens phase')
        return False
    else:
        return True
def generate_Injector_Lattice(X) -> ParticleTracerLattice:
    parallel=False
    injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner=X
    assert type(parallel) == bool
    if is_Valid_Injector_Phase(injectorFactor, rpInjectorFactor) == False:
        return None
    LInjector = injectorFactor * .15
    rpInjector = rpInjectorFactor * .02
    fringeFrac = 1.5
    LMagnet = LInjector - 2 * fringeFrac * rpInjector
    if LMagnet < 1e-9:  # minimum fringe length must be respected.
        return None
    PTL_Injector = ParticleTracerLattice(200.0, latticeType='injector', parallel=parallel)
    PTL_Injector.add_Drift(.1, ap=.025)
    PTL_Injector.add_Halbach_Lens_Sim(rpInjector, LInjector, apFrac=.9)
    PTL_Injector.add_Drift(.2, ap=.01)
    try:
        PTL_Injector.add_Combiner_Sim_Lens(LmCombiner, rpCombiner)
    except:
        # print('combiner error')
        return None
    PTL_Injector.end_Lattice(constrain=False, enforceClosedLattice=False)
    return PTL_Injector



def generate_Ring_Surrogate_Lattice(X,parallel=False)->ParticleTracerLattice:
    tunableDriftGap=2.54e-2
    rpLens=.015
    LLens=.4
    injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner = X
    assert type(parallel)==bool
    PTL_Ring=ParticleTracerLattice(200.0,latticeType='storageRing',parallel=parallel)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
    try:
        PTL_Ring.add_Combiner_Sim_Lens(LmCombiner, rpCombiner)
    except:
        return None
    PTL_Ring.end_Lattice(enforceClosedLattice=False,constrain=False,surpressWarning=True)  # 17.8 % of time here
    return PTL_Ring

class Injection_Model(LatticeOptimizer):
    def __init__(self, latticeRing, latticeInjector):
        self.latticeRing = latticeRing
        self.latticeInjector = latticeInjector
        self.i = 0  # simple variable to track solution status
        self.swarmTracerInjector = SwarmTracer(self.latticeInjector)
        self.h = 1e-5  # timestep size
        self.T = 10.0
        self.swarmTracerRing = SwarmTracer(self.latticeRing)
        self.phaseSpaceFunc = None  # function that returns the number of revolutions of a particle at a given
        # point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.tunedElementList = None
        self.tuningChoice = None  # what type of tuning will occur
        self.useLatticeUpperSymmetry = True  # exploit the fact that the lattice has symmetry in the z axis to use half
        # the number of particles. Symmetry is broken if including gravity
        self.sameSeedForSwarm = True  # generate the same swarms every time by seeding the random generator during swarm
        # generation with the same number, 42
        self.sameSeedForSearch = True  # wether to use the same seed, 42, for the search process
        self.postCombinerAperture = .015
        self.spotCaptureDiam = 5e-3
        self.collectorAngleMax = .06
        self.temperature = 3e-3
        numParticlesFullSwarm=50
        self.swarmInjectorInitial = self.swarmTracerInjector.initialize_Observed_Collector_Swarm_Probability_Weighted(
            self.spotCaptureDiam, self.collectorAngleMax, numParticlesFullSwarm, temperature=self.temperature,
            sameSeed=self.sameSeedForSwarm, upperSymmetry=self.useLatticeUpperSymmetry)
    def cost(self):
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        swarmCost = self.injected_Swarm_Cost()
        assert 0.0<=swarmCost<=1.0
        overLapCost=self.floor_Plan_Cost()
        assert 0.0<=overLapCost<=1.0
        cost=np.sqrt(overLapCost**2+swarmCost**2)
        self.show_Floor_Plan()
        return cost

    def injected_Swarm_Cost(self):
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            self.swarmInjectorInitial
            , self.h, 1.0, parallel=False,
            fastMode=True, copySwarm=False,
            accelerated=True)
        # self.latticeInjector.show_Lattice(swarm=swarmInjectorTraced,showTraceLines=True,trueAspectRatio=False)
        swarmSurvived = self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced,
                                                                                 copyParticles=False)
        swarmCost = (self.swarmInjectorInitial.num_Particles() - swarmSurvived.num_Particles()) \
                                / self.swarmInjectorInitial.num_Particles()
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
    def move_Survived_Particles_In_Injector_Swarm_To_Origin(self, swarmInjectorTraced,copyParticles=False):
        #fidentify particles that survived to combiner end, walk them right up to the end, exclude any particles that
        #are now clipping the combiner and any that would clip the next element
        #NOTE: The particles offset is taken from the origin of the orbit output of the combiner, not the 0,0 output
        swarmSurvived = Swarm()
        for particle in swarmInjectorTraced:
            outputCenter=self.latticeInjector.combiner.r2+self.swarmTracerInjector.combiner_Output_Offset_Shift()
            qf = particle.qf - outputCenter
            qf[:2] = self.latticeInjector.combiner.RIn @ qf[:2]
            if qf[0] <= self.h * self.latticeRing.v0Nominal:  # if the particle is within a timestep of the end,
                # assume it's at the end
                pf = particle.pf.copy()
                pf[:2] = self.latticeInjector.combiner.RIn @ particle.pf[:2]
                qf = qf + pf * np.abs(qf[0] / pf[0]) #walk particle up to the end of the combiner
                qf[0]=0.0 #no rounding error
                clipsNextElement=np.sqrt(qf[1] ** 2 + qf[2] ** 2) > self.postCombinerAperture
                if clipsNextElement==False:  # test that particle survives through next aperture
                    if copyParticles == False:
                        particleEnd = particle
                    else:
                        particleEnd = particle.copy()
                    particleEnd.qi = qf
                    particleEnd.pi = pf
                    particleEnd.reset()
                    swarmSurvived.particles.append(particleEnd)
        return swarmSurvived
def injector_Cost(X):
    maximumCost=2.0
    PTL_I=generate_Injector_Lattice(X)
    if PTL_I is None:
        return maximumCost
    PTL_R=generate_Ring_Surrogate_Lattice(X)
    if PTL_R is None:
        return maximumCost
    test=Injection_Model(PTL_R,PTL_I)
    cost=test.cost()
    assert cost<maximumCost
    return cost
def main():
    bounds = [(.5, 2), (.5, 2), (.05, .5), (.015, .1)]
    print(solve_Async(injector_Cost,bounds,15*len(bounds),tol=.05,surrogateMethodProb=0.0))
if __name__=="__main__":
    main()
