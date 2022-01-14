import os
import random

os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
import numpy as np

from ParticleTracerLatticeClass import ParticleTracerLattice
from SwarmTracerClass import SwarmTracer
import scipy.optimize as spo

h=5e-6

def is_Valid_Injector_Phase(injectorFactor,rpInjectorFactor):
    LInjector=injectorFactor*.15
    rpInjector=rpInjectorFactor*.02
    BpLens=.7
    injectorLensPhase=np.sqrt((2*800.0/200**2)*BpLens/rpInjector**2)*LInjector
    if np.pi<injectorLensPhase or injectorLensPhase<np.pi/10:
        # print('bad lens phase')
        return False
    else:
        return True

def generate_Injector_Lattice(injectorFactor,rpInjectorFactor,LmCombiner,rpCombiner,parallel=False)->ParticleTracerLattice:
    assert type(parallel)==bool
    if is_Valid_Injector_Phase(injectorFactor,rpInjectorFactor)==False:
        return None
    LInjector=injectorFactor*.15
    rpInjector=rpInjectorFactor*.02
    fringeFrac=1.5
    LMagnet=LInjector-2*fringeFrac*rpInjector
    if LMagnet<1e-9:  # minimum fringe length must be respected.
        return None
    PTL_Injector=ParticleTracerLattice(200.0,latticeType='injector',parallel=parallel)
    PTL_Injector.add_Drift(.1,ap=.025)

    PTL_Injector.add_Halbach_Lens_Sim(rpInjector,LInjector,apFrac=.9)
    PTL_Injector.add_Drift(.2,ap=.01)
    try:
        PTL_Injector.add_Combiner_Sim_Lens(LmCombiner,rpCombiner)
    except:
        # print('combiner error')
        return None
    PTL_Injector.add_Halbach_Lens_Sim(.02,.2)
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)



    return PTL_Injector


def number_Survived_Particles(swarmTraced,PTL,swarmTracer):
    # fidentify particles that survived to combiner end, walk them right up to the end, exclude any particles that
    # are now clipping the combiner and any that would clip the next element
    # NOTE: The particles offset is taken from the origin of the orbit output of the combiner, not the 0,0 output
    assert PTL.latticeType=='injector'
    apNextElement=1e-2
    numSurvived=0
    for particle in swarmTraced:
        outputCenter = PTL.combiner.r2 + swarmTracer.combiner_Output_Offset_Shift()
        qf = particle.qf - outputCenter
        qf[:2] = PTL.combiner.RIn @ qf[:2]
        if qf[0] <= h * PTL.v0Nominal:  # if the particle is within a timestep of the end,
            # assume it's at the end
            pf = particle.pf.copy()
            pf[:2] = PTL.combiner.RIn @ particle.pf[:2]
            qf = qf + pf * np.abs(qf[0] / pf[0])  # walk particle up to the end of the combiner
            qf[0] = 0.0  # no rounding error
            clipsNextElement = np.sqrt(qf[1] ** 2 + qf[2] ** 2) > apNextElement
            if clipsNextElement == False:  # test that particle survives through next aperture
                numSurvived+=1
    return numSurvived
    
def calculate_Cost(swarmTraced,swarmTracer,PTL_Injector,X,targetMinBend):
    injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner = X
    swarmCost = (swarmTraced.num_Particles() - number_Survived_Particles(swarmTraced, PTL_Injector, swarmTracer))\
                / swarmTraced.num_Particles()

    PTL_Ring = ParticleTracerLattice(200.0)
    try:
        PTL_Ring.add_Combiner_Sim_Lens(LmCombiner,rpCombiner)
    except:
        return None
    totalBend=abs(PTL_Ring.combiner.ang)+abs(PTL_Injector.combiner.ang)
    bendingCost=0.0
    if totalBend<targetMinBend: #severly punish for bending too little
        bendingCost += 10.0 * abs((totalBend - targetMinBend) / targetMinBend)
    cost = np.sqrt(swarmCost ** 2 + bendingCost ** 2)
    return cost
def lattice_Cost(X,bendingAngle):
    assert len(X)==4
    # injectorFactor,rpInjectorFactor,LmCombiner,rpCombiner=X
    PTL=generate_Injector_Lattice(*X)
    if PTL is None:
        return 1.0
    swarmTracer=SwarmTracer(PTL)

    
    sameSeedForSearch = True  # wether to use the same seed, 42, for the search process
    numParticles = 300
    spotCaptureDiam = 5e-3
    collectorAngleMax = .06
    temperature = 3e-3
    swarmInitial = swarmTracer.initialize_Observed_Collector_Swarm_Probability_Weighted(
        spotCaptureDiam, collectorAngleMax, numParticles, temperature=temperature,
        sameSeed=42, upperSymmetry=sameSeedForSearch)
    swarmTraced=swarmTracer.trace_Swarm_Through_Lattice(swarmInitial,5e-6,1.0,parallel=False,fastMode=True,
                                                        accelerated=True)
    cost=calculate_Cost(swarmTraced,swarmTracer,PTL,X,bendingAngle)
    # PTL.show_Lattice(swarm=swarmTraced,showTraceLines=True,trueAspectRatio=False)
    return cost
def main():
    import skopt
    #injectorFactor,rpInjectorFactor,LmCombiner,rpCombiner
    angle=.18
    bounds=[(1.0,4.0),(1,3),(.05,.5),(.015,.1)]
    wrapper=lambda x: lattice_Cost(x,angle)
    print(solve_Async(wrapper,bounds,3600,15*len(bounds),surrogateMethodProb=.1))
if __name__=="__main__":
    main()

'''
RESULTS:

Min bending angle |  cost | X optimal
.16 |  |  
.18 | 0.08053691275167785 |  [1.57234466 1.24405822 0.09132377 0.0226607 ]
.2| |  |

'''