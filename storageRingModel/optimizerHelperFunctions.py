import time
import numpy as np
from typing import Optional
from storageRingOptimizer import LatticeOptimizer,Solution
from ParticleTracerLatticeClass import ParticleTracerLattice
SMALL_NUMBER=1E-9
# XInjector=[1.10677162, 1.00084144, 0.11480408, 0.02832031]
V0=210
def invalid_Solution(XLattice,invalidInjector=None,invalidRing=None):
    assert len(XLattice)==5,"must be lattice paramters"
    sol=Solution()
    sol.xRing_TunedParams1=XLattice
    sol.fluxMultiplication=0.0
    sol.swarmCost,sol.floorPlanCost=1.0,1.0
    sol.cost=sol.swarmCost+sol.floorPlanCost
    sol.description='Pseudorandom search'
    sol.invalidRing=invalidRing
    sol.invalidInjector=invalidInjector
    return sol




def generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens, LmCombiner, rpCombiner,loadBeamDiam,
                          tuning,jitterAmp=0.0,fieldDensityMultiplier: float =1.0,
                          standardMagnetErrors: bool=False,combinerSeed: int=None)->Optional[ParticleTracerLattice]:
    assert tuning in (None, 'field', 'spacing')
    tunableDriftGap=2.54e-2
    jeremyGap=.05
    Lm=.0254/2.0
    lastGap=5e-2
    fringeFrac=1.5
    if L_Lens-2*rpLens*fringeFrac<rpLens/2 or L_Lens-2*rpLensFirst*fringeFrac<rpLensFirst/2 or \
            L_Lens-2*rpLensLast*fringeFrac<rpLensLast/2:
        # minimum fringe length must be respected
        return None
    PTL_Ring=ParticleTracerLattice(V0,latticeType='storageRing',jitterAmp=jitterAmp,fieldDensityMultiplier
                    =fieldDensityMultiplier,standardMagnetErrors=standardMagnetErrors)
    rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm)
    if rOffsetFact is None:
        return None
    if tuning=='spacing':
        PTL_Ring.add_Drift(tunableDriftGap/2)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,L_Lens)
        PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensLast,L_Lens)
    PTL_Ring.add_Drift(lastGap)
    if combinerSeed is not None:
        np.random.seed(combinerSeed)
    PTL_Ring.add_Combiner_Sim_Lens(LmCombiner,rpCombiner,loadBeamDiam=loadBeamDiam,layers=1)
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


def generate_Injector_Lattice_Double_Lens(X: tuple,jitterAmp: float =0.0,fieldDensityMultiplier: float=1.0,
                        standardMagnetErrors: bool=False,combinerSeed: int=None) -> Optional[ParticleTracerLattice]:
    L_InjectorMagnet1, rpInjectorMagnet1,L_InjectorMagnet2, rpInjectorMagnet2, \
    LmCombiner, rpCombiner,loadBeamDiam,L1,L2,L3=X
    fringeFrac = 1.5
    LMagnet1 = L_InjectorMagnet1 - 2 * fringeFrac * rpInjectorMagnet1
    LMagnet2 = L_InjectorMagnet2 - 2 * fringeFrac * rpInjectorMagnet2
    aspect1,aspect2=LMagnet1/rpInjectorMagnet1,LMagnet2/rpInjectorMagnet2
    if aspect1<1.0 or aspect2<1.0: #horrible fringe field performance
        return None
    if LMagnet1 < rpInjectorMagnet1/2 or LMagnet2 < rpInjectorMagnet2/2:  # minimum fringe length must be respected.
        return None
    if loadBeamDiam>rpCombiner: #silly if load beam doens't fit in half of magnet
        return None
    PTL_Injector = ParticleTracerLattice(V0, latticeType='injector',jitterAmp=jitterAmp,
                    fieldDensityMultiplier=fieldDensityMultiplier,standardMagnetErrors=standardMagnetErrors)
    PTL_Injector.add_Drift(L1, ap=rpInjectorMagnet1)
    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet1, L_InjectorMagnet1)
    PTL_Injector.add_Drift(L2, ap=max([rpInjectorMagnet1,rpInjectorMagnet2]))
    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet2, L_InjectorMagnet2)
    PTL_Injector.add_Drift(L3, ap=rpInjectorMagnet2)
    if combinerSeed is not None:
        np.random.seed(combinerSeed)
    PTL_Injector.add_Combiner_Sim_Lens(LmCombiner, rpCombiner,loadBeamDiam=loadBeamDiam,layers=1)
    PTL_Injector.end_Lattice(constrain=False, enforceClosedLattice=False)
    assert PTL_Injector.elList[1].fringeFracOuter==fringeFrac and PTL_Injector.elList[3].fringeFracOuter==fringeFrac
    return PTL_Injector

def generate_Ring_And_Injector_Lattice(X,tuning,jitterAmp=0.0,fieldDensityMultiplier=1.0,
                                       standardMagnetErrors:bool =False, combinerSeed: int=None):
    # rpBend=1e-2
    XInjector=(0.14625806, 0.02415056, 0.121357  , 0.02123799, 0.19139004,
       0.04      , 0.01525237, 0.05      , 0.19573719, 0.22186834)
    L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, LmCombiner, rpCombiner, \
    loadBeamDiam, L1, L2, L3=XInjector

    rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens=X
    combinerSeed = None if standardMagnetErrors==False else combinerSeed
    PTL_Ring=generate_Ring_Lattice(rpLens,rpLensFirst,rpLensLast,rpBend,L_Lens, LmCombiner, rpCombiner,loadBeamDiam,
                                   tuning,jitterAmp=jitterAmp,fieldDensityMultiplier=fieldDensityMultiplier,
                                   standardMagnetErrors=standardMagnetErrors,combinerSeed=combinerSeed)
    if PTL_Ring is None:
        return None,None
    PTL_Injector=generate_Injector_Lattice_Double_Lens(XInjector,jitterAmp=jitterAmp,fieldDensityMultiplier
                    =fieldDensityMultiplier,standardMagnetErrors=standardMagnetErrors,combinerSeed=combinerSeed)
    if PTL_Injector is None:
        return None,None
    assert PTL_Ring.combiner.outputOffset == PTL_Injector.combiner.outputOffset
    return PTL_Ring,PTL_Injector
def solution_From_Lattice(PTL_Ring: ParticleTracerLattice, PTL_Injector: ParticleTracerLattice,tuning: Optional[str])\
        -> Solution:
    optimizer = LatticeOptimizer(PTL_Ring, PTL_Injector)
    if tuning is None:
        sol = Solution()
        knobParams = None
        swarmCost, floorPlanCost,swarmTraced = optimizer.mode_Match_Cost(knobParams, False, True, rejectIllegalFloorPlan=False,
                                                             rejectUnstable=False, returnFullResults=True)
        sol.cost = swarmCost + floorPlanCost
        sol.floorPlanCost = floorPlanCost
        sol.swarmCost = swarmCost
        sol.fluxMultiplication = optimizer.compute_Flux_Multiplication(swarmTraced)
    elif tuning == 'field':
        sol = optimizer.optimize((0, 4), whichKnobs='ring', tuningChoice=tuning, ringTuningBounds=[(.25, 1.75)] * 2)
    elif tuning=='spacing':
        sol = optimizer.optimize((1, 9), whichKnobs='ring', tuningChoice=tuning)
    else: raise ValueError
    return sol
def solve_For_Lattice_Params(X,tuning,magnetErrors):
    assert tuning in (None,'field','spacing')
    PTL_Ring, PTL_Injector=generate_Ring_And_Injector_Lattice(X,tuning,standardMagnetErrors=magnetErrors)
    if PTL_Ring is None or PTL_Injector is None:
        sol = invalid_Solution(X)
    else:
        sol=solution_From_Lattice(PTL_Ring,PTL_Injector,tuning)
    return sol