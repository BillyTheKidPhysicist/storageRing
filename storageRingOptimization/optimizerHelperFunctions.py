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


def is_Valid_Injector_Phase(injectorFactor,rpInjectorFactor):
    LInjector=injectorFactor*.15
    rpInjector=rpInjectorFactor*.02
    BpLens=.7
    injectorLensPhase=np.sqrt((2*800.0/200**2)*BpLens/rpInjector**2)*LInjector
    if np.pi<injectorLensPhase or injectorLensPhase<np.pi/10:
        print('bad lens phase')
        return False
    else:
        return True


def generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,LLens,injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner
                          ,parallel=False)->ParticleTracerLattice:
    tunableDriftGap=2.54e-2
    jeremyGap=.05
    rpBend=.01
    Lm=.0254/2.0
    assert type(parallel)==bool
    fringeFrac=1.5
    if LLens-2*rpLens*fringeFrac<0 or LLens-2*rpLensFirst*fringeFrac<0:  # minimum fringe length must be respected
        return None
    PTL_Ring=ParticleTracerLattice(200.0,latticeType='storageRing',parallel=parallel)
    rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm,
                                                    parallel=parallel)  # 25% of time here, 1.0138513851385138
    if rOffsetFact is None:
        return None
    PTL_Ring.add_Drift(tunableDriftGap/2,ap=rpLens)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2,ap=rpLens)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensLast,LLens)
    PTL_Ring.add_Combiner_Sim_Lens(LmCombiner, rpCombiner)
    PTL_Ring.add_Drift(jeremyGap)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensFirst,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  # 17.8 % of time here
    return PTL_Ring


def generate_Injector_Lattice(injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner,parallel=False)->ParticleTracerLattice:

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
    PTL_Injector.add_Combiner_Sim_Lens(LmCombiner,rpCombiner)
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)
    return PTL_Injector


def solve_For_Lattice_Params(X,parallel=False,useSurrogateMethod=False):

    rpLens,rpLensFirst,rpLensLast,LLens,injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner=X

    PTL_Ring=generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,LLens,injectorFactor, rpInjectorFactor,
                                   LmCombiner, rpCombiner,parallel=parallel)
    if PTL_Ring is None:
        print('invalid ring')
        sol=invalid_Solution(X,invalidRing=True)
        return sol
    PTL_Injector=generate_Injector_Lattice(injectorFactor, rpInjectorFactor, LmCombiner, rpCombiner,parallel=parallel)
    if PTL_Injector is None:
        print('invalid injector')
        sol=invalid_Solution(X,invalidInjector=True)
        return sol
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
    sol=optimizer.optimize((1,8),parallel=parallel,fastSolver=useSurrogateMethod)
    sol.xRing_TunedParams1=X
    sol.description='Pseudorandom search'
    if sol.fluxMultiplicationPercent>.1:
        print(sol)
    return sol
