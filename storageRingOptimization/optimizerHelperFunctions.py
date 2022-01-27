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
    assert len(XLattice)==8,"must be lattice paramters"
    sol=Solution()
    sol.xRing_TunedParams1=XLattice
    sol.fluxMultiplicationPercent=0.0
    sol.cost=1.0
    sol.description='Pseudorandom search'
    sol.invalidRing=invalidRing
    sol.invalidInjector=invalidInjector
    return sol




def generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,L_Lens,L_Injector, rpInjector, LmCombiner, rpCombiner
                          )->ParticleTracerLattice:
    tunableDriftGap=2.54e-2
    jeremyGap=.05
    rpBend=.01
    Lm=.0254/2.0
    lastGap=5e-2
    fringeFrac=1.5
    loadBeamDiameter=.017
    if L_Lens-2*rpLens*fringeFrac<0 or L_Lens-2*rpLensFirst*fringeFrac<0:  # minimum fringe length must be respected
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
    PTL_Ring.add_Combiner_Sim_Lens(LmCombiner,rpCombiner,loadBeamDiam=loadBeamDiameter)
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


def generate_Injector_Lattice(L_Injector, rpInjector, LmCombiner, rpCombiner)->ParticleTracerLattice:


    L1_Tunable, L2_Tunable = [.1,.2] #pretty much dummy arguments
    fringeFrac=1.5
    loadBeamDiameter=.017
    L_InjectorMagnet=L_Injector-2*fringeFrac*rpInjector
    if L_InjectorMagnet<1e-9:  # minimum fringe length must be respected.
        return None
    PTL_Injector=ParticleTracerLattice(V0,latticeType='injector')
    PTL_Injector.add_Drift(L1_Tunable,ap=.025)

    PTL_Injector.add_Halbach_Lens_Sim(rpInjector,L_Injector,apFrac=.9)
    PTL_Injector.add_Drift(L2_Tunable,ap=.01)
    PTL_Injector.add_Combiner_Sim_Lens(LmCombiner,rpCombiner,loadBeamDiam=loadBeamDiameter)
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)
    return PTL_Injector


def solve_For_Lattice_Params(X,useSurrogateMethod=False):

    rpLens,rpLensFirst,rpLensLast,L_Lens,L_Injector, rpInjector, LmCombiner, rpCombiner=X

    PTL_Ring=generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,L_Lens,L_Injector, rpInjector,
                                   LmCombiner, rpCombiner)
    if PTL_Ring is None:
        print('invalid ring')
        sol=invalid_Solution(X,invalidRing=True)
        return sol
    PTL_Injector=generate_Injector_Lattice(L_Injector, rpInjector, LmCombiner, rpCombiner)
    if PTL_Injector is None:
        print('invalid injector')
        sol=invalid_Solution(X,invalidInjector=True)
        return sol
    assert PTL_Ring.combiner.outputOffset == PTL_Injector.combiner.outputOffset
    # import dill
    # with open('temp','wb') as file:
    #     dill.dump(PTL_Ring,file)
    # with open('temp','rb') as file:
    #     PTL_Ring=dill.load(file)
    # with open('temp1', 'wb') as file:
    #     dill.dump(PTL_Injector, file)
    # with open('injectorFile','rb') as file:
    #     PTL_Injector=dill.load(file)

    # PTL_Injector.show_Lattice()
    # PTL_Ring.show_Lattice()
    optimizer=LatticeOptimizer(PTL_Ring,PTL_Injector)
    sol=optimizer.optimize((1,9),fastSolver=useSurrogateMethod)
    sol.xRing_TunedParams1=X
    sol.description='Pseudorandom search'
    if sol.fluxMultiplicationPercent>.1:
        print(sol)
    return sol
