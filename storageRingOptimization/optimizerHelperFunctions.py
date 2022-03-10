import os
import random

os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
import numpy as np
from ParticleTracerClass import ParticleTracer
from SwarmTracerClass import SwarmTracer
from latticeOptimizer import LatticeOptimizer,Solution
from ParticleTracerLatticeClass import ParticleTracerLattice
SMALL_NUMBER=1E-9
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
                          LmCombiner, rpCombiner,loadBeamDiam,tuning)->ParticleTracerLattice:
    assert tuning in (None, 'field', 'spacing')
    tunableDriftGap=2.54e-2
    jeremyGap=.05
    Lm=.0254/2.0
    lastGap=5e-2
    fringeFrac=1.5
    if L_Lens-2*rpLens*fringeFrac<0 or L_Lens-2*rpLensFirst*fringeFrac<0 or L_Lens-2*rpLensLast*fringeFrac<0:
        # minimum fringe length must be respected
        return None
    PTL_Ring=ParticleTracerLattice(V0,latticeType='storageRing')
    if abs(rpBend-1e-2)<SMALL_NUMBER and abs(Lm-.0254/2.0)<SMALL_NUMBER:
        rOffsetFact = 1.0106
    else:
        # rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm)
        raise ValueError
    if rOffsetFact is None:
        return None
    if tuning=='spacing':
        PTL_Ring.add_Drift(tunableDriftGap/2)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,L_Lens)
        PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensLast,L_Lens)
    PTL_Ring.add_Drift(lastGap)
    PTL_Ring.add_Combiner_Sim_Lens(LmCombiner,rpCombiner,loadBeamDiam=loadBeamDiam)
    PTL_Ring.add_Drift(jeremyGap)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensFirst,L_Lens)
    if tuning=='spacing':
        PTL_Ring.add_Drift(tunableDriftGap/2)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,L_Lens)
        PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  # 17.8 % of time here
    return PTL_Ring


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

def generate_Injector_Lattice_Double_Magnet(L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2,
    LmCombiner, rpCombiner, loadBeamDiam, L1, L2, L3,bumpOffsetAmp=0.0)->ParticleTracerLattice:
    fringeFrac = 1.5
    apFrac = .95
    LMagnet1 = L_InjectorMagnet1 - 2 * fringeFrac * rpInjectorMagnet1
    LMagnet2 = L_InjectorMagnet2 - 2 * fringeFrac * rpInjectorMagnet2
    if LMagnet1 < 1e-9 or LMagnet2 < 1e-9:  # minimum fringe length must be respected.
        return None
    if loadBeamDiam > apFrac * rpCombiner:  # silly if load beam doens't fit in half of magnet
        return None
    PTL_Injector = ParticleTracerLattice(V0, latticeType='injector', parallel=False)
    PTL_Injector.add_Drift(L1, ap=apFrac * rpInjectorMagnet1)
    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet1, L_InjectorMagnet1, apFrac=apFrac)
    PTL_Injector.add_Drift(L2, ap=apFrac * max([rpInjectorMagnet1, rpInjectorMagnet2]))
    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet2, L_InjectorMagnet2, apFrac=apFrac)
    PTL_Injector.add_Drift(L3, ap=apFrac * rpInjectorMagnet2)

    try:  # even with guards it can still derp out
        PTL_Injector.add_Combiner_Sim_Lens(LmCombiner, rpCombiner, loadBeamDiam=loadBeamDiam)
    except:
        return None
    PTL_Injector.end_Lattice(constrain=False, enforceClosedLattice=False)
    return PTL_Injector

def generate_Ring_And_Injector_Lattice(X,tuning):
    rpBend=1e-2
    XInjector=np.array([0.05       ,0.01056943 ,0.17291778 ,0.0256151  ,0.18110825 ,0.04915702
        ,0.01790981 ,0.01645214 ,0.27854378 ,0.19162297])

    L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, LmCombiner, rpCombiner, \
    loadBeamDiam, L1, L2, L3=XInjector

    rpLens,rpLensFirst,rpLensLast,L_Lens=X
    #value2 from seperate optimizer
    PTL_Ring=generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens, LmCombiner, rpCombiner,loadBeamDiam,
                                   tuning)
    if PTL_Ring is None:
        print('invalid ring')
        sol=invalid_Solution(X,invalidRing=True)
        return sol
    # PTL_Injector=generate_Injector_Lattice(L_Injector, rpInjector, LmCombiner, rpCombiner,loadBeamDiam, L1, L2)
    PTL_Injector=generate_Injector_Lattice_Double_Magnet(L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2,
                                rpInjectorMagnet2, LmCombiner, rpCombiner, loadBeamDiam, L1, L2, L3)
    if PTL_Injector is None:
        print('invalid injector')
        sol=invalid_Solution(X,invalidInjector=True)
        return sol
    assert PTL_Ring.combiner.outputOffset == PTL_Injector.combiner.outputOffset
    return PTL_Ring,PTL_Injector
def solution_From_Lattice(PTL_Ring, PTL_Injector,X,tuning):
    optimizer = LatticeOptimizer(PTL_Ring, PTL_Injector)
    if tuning is None:
        sol = Solution()
        knobParams = None
        swarmCost, floorPlanCost = optimizer.mode_Match_Cost(knobParams, False, True, rejectIllegalFloorPlan=False,
                                                             rejectUnstable=False, returnCostsSeperate=True)
        sol.cost = swarmCost + floorPlanCost
        sol.floorPlanCost = floorPlanCost
        sol.swarmCost = swarmCost
        sol.fluxMultiplicationPercent = optimizer.flux_Percent_From_Cost(sol.cost, knobParams)
    elif tuning == 'field':
        sol = optimizer.optimize((0, 4), whichKnobs='ring', tuningChoice=tuning, ringTuningBounds=[(.25, 1.75)] * 2)
    else:
        sol = optimizer.optimize((1, 9), whichKnobs='ring', tuningChoice=tuning)
    sol.xRing_TunedParams1 = X
    # if sol.fluxMultiplicationPercent>10.0:
    #     print(sol)
    return sol
def solve_For_Lattice_Params(X,tuning):
    assert tuning in (None,'field','spacing')
    PTL_Ring, PTL_Injector=generate_Ring_And_Injector_Lattice(X,tuning)
    return solution_From_Lattice(PTL_Ring,PTL_Injector,X,tuning)
