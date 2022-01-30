import os
import random

os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
import numpy as np
from ParticleTracerClass import ParticleTracer
from SwarmTracerClass import SwarmTracer
from latticeKnobOptimizer import LatticeOptimizer,Solution
from ParticleTracerLatticeClass import ParticleTracerLattice

# XInjector=[1.10677162, 1.00084144, 0.11480408, 0.02832031]
V0=210
def invalid_Solution(XLattice,invalidInjector=None,invalidRing=None):
    assert len(XLattice)==5,"must be lattice paramters"
    sol=Solution()
    sol.xRing_TunedParams1=XLattice
    sol.fluxMultiplicationPercent=0.0
    sol.cost=1.0
    sol.description='Pseudorandom search'
    sol.invalidRing=invalidRing
    sol.invalidInjector=invalidInjector
    return sol




def generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens,
                          LmCombiner, rpCombiner,loadBeamDiam)->ParticleTracerLattice:
    tunableDriftGap=2.54e-2
    jeremyGap=.05
    Lm=.0254/2.0
    lastGap=5e-2
    fringeFrac=1.5
    if L_Lens-2*rpLens*fringeFrac<0 or L_Lens-2*rpLensFirst*fringeFrac<0 or L_Lens-2*rpLensLast*fringeFrac<0:
        # minimum fringe length must be respected
        return None
    PTL_Ring=ParticleTracerLattice(V0,latticeType='storageRing')
    rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm)
    if rOffsetFact is None:
        return None
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,L_Lens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensLast,L_Lens)
    PTL_Ring.add_Drift(lastGap)
    PTL_Ring.add_Combiner_Sim_Lens(LmCombiner,rpCombiner,loadBeamDiam=loadBeamDiam)
    PTL_Ring.add_Drift(jeremyGap)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensFirst,L_Lens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,L_Lens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  # 17.8 % of time here
    return PTL_Ring


def generate_Injector_Lattice(L_Injector, rpInjector, LmCombiner, rpCombiner,loadBeamDiam,L1,L2)->ParticleTracerLattice:


    fringeFrac=1.5
    L_InjectorMagnet=L_Injector-2*fringeFrac*rpInjector
    if L_InjectorMagnet<1e-9:  # minimum fringe length must be respected.
        return None
    PTL_Injector=ParticleTracerLattice(V0,latticeType='injector')
    PTL_Injector.add_Drift(L1,ap=.025)

    PTL_Injector.add_Halbach_Lens_Sim(rpInjector,L_Injector,apFrac=.98)
    PTL_Injector.add_Drift(L2,ap=.01)
    PTL_Injector.add_Combiner_Sim_Lens(LmCombiner,rpCombiner,loadBeamDiam=loadBeamDiam)
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)
    return PTL_Injector


def solve_For_Lattice_Params(X):

    rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens=X
    #value2 from seperate optimizer
    L_Injector, rpInjector, LmCombiner, rpCombiner=0.16063242, 0.0233469 , 0.13838299, 0.04630577
    loadBeamDiam, L1, L2=0.02055515 ,0.24656484 ,0.208979
    PTL_Ring=generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens, LmCombiner, rpCombiner,loadBeamDiam)
    if PTL_Ring is None:
        print('invalid ring')
        sol=invalid_Solution(X,invalidRing=True)
        return sol
    PTL_Injector=generate_Injector_Lattice(L_Injector, rpInjector, LmCombiner, rpCombiner,loadBeamDiam, L1, L2)
    if PTL_Injector is None:
        print('invalid injector')
        sol=invalid_Solution(X,invalidInjector=True)
        return sol
    assert PTL_Ring.combiner.outputOffset == PTL_Injector.combiner.outputOffset
    optimizer=LatticeOptimizer(PTL_Ring,PTL_Injector)
    sol=optimizer.optimize((1,9),whichKnobs='ring')
    sol.xRing_TunedParams1=X
    sol.description='Pseudorandom search'
    if sol.fluxMultiplicationPercent>1.0:
        print(sol)
    return sol
