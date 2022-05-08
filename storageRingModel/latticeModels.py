import time
import numpy as np
from constants import DEFAULT_ATOM_SPEED
import elementPT
from typing import Optional,Union
from storageRingOptimizer import LatticeOptimizer,Solution
from latticeModels_Constants import constants_Version1
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleTracerClass import ParticleTracer

class RingGeometryError(Exception):
    pass

class InjectorGeometryError(Exception):
    pass

lst_arr_tple = Union[list, np.ndarray, tuple]


def el_Fringe_Space(elementName: str, rp: float)-> float:
    assert rp>0
    fringeFracs={"combiner":elementPT.CombinerHalbachLensSim.outerFringeFrac,
                 "lens":elementPT.HalbachLensSim.fringeFracOuter,
                 "bender":elementPT.HalbachBenderSimSegmented.fringeFracOuter}
    return fringeFracs[elementName]*rp

def clip_If_Below_Min_Time_Step_Gap(length: float)-> float:
    h = 1e-5  # timestep, s. Assumed to be no larger than this
    minTimeStepGap=h*DEFAULT_ATOM_SPEED*ParticleTracer.minTimeStepsPerElement
    if length< minTimeStepGap:
        return minTimeStepGap
    else:
        return length

def add_Drift_If_Needed(PTL: ParticleTracerLattice,minSpace: float,elBeforeName: str,
                        elAfterName: str,elBefore_rp: float,elAfter_rp: float, ap:float=None)-> None:
    assert minSpace>=0 and elAfter_rp>0 and elBefore_rp>0
    extraSpace=minSpace-(el_Fringe_Space(elBeforeName,elBefore_rp)+el_Fringe_Space(elAfterName,elAfter_rp))
    if extraSpace>0:
        ap=min([elBefore_rp,elAfter_rp]) if ap is None else ap
        PTL.add_Drift(clip_If_Below_Min_Time_Step_Gap(extraSpace),ap=ap)

def add_Bend_Adjacent_Gap(PTL: ParticleTracerLattice,rpLens: float,gapMin: float,rpBend: float)-> None:
    add_Drift_If_Needed(PTL,gapMin,'lens','bender',rpLens,rpBend)

def add_First_Racetrack(PTL,rpLens1,rpLens2,L_Lens,rpCombiner,LmCombiner,loadBeamDiam,rpBend,consts: dict):
    # ------gap 1--------  bender-> lens
    # there is none here because of strong adjacent pumping

    # --------lens 1---------

    PTL.add_Halbach_Lens_Sim(rpLens1, L_Lens)

    # --------gap 2-------- lens-> combiner

    add_Drift_If_Needed(PTL, consts["gap2Min"], 'lens', 'combiner', rpLens1, rpCombiner)

    # -------combiner-------

    PTL.add_Combiner_Sim_Lens(LmCombiner, rpCombiner, loadBeamDiam=loadBeamDiam, layers=1)

    # ------gap 3--------- combiner-> lens
    # there must be a drift here to account for the optical pumping aperture limit
    gap3ExtraSpace = consts["OP_MagWidth"] - (el_Fringe_Space('combiner', rpCombiner)
                                              + el_Fringe_Space('lens', rpLens2))
    PTL.add_Drift(clip_If_Below_Min_Time_Step_Gap(gap3ExtraSpace), ap=consts["OP_MagAp"])

    # -------lens 2-------
    PTL.add_Halbach_Lens_Sim(rpLens2, L_Lens)

    # ---------gap 4----- lens-> bender

    add_Bend_Adjacent_Gap(PTL, rpLens2, consts["lensToBendGap"], rpBend)

def make_Ring_And_Injector_Version1(params: lst_arr_tple)-> tuple[ParticleTracerLattice, ParticleTracerLattice]:

    injectorParams=(0.14625806, 0.02415056, 0.121357  , 0.02123799, 0.19139004,
       0.04      , 0.01525237, 0.05      , 0.19573719, 0.22186834)
    L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, LmCombiner, rpCombiner, \
    loadBeamDiam, gap1, gap2, gap3=injectorParams

    rpLens3_4, rpLens2, rpLens1, rpBend, L_Lens = params

    ringParams=[rpLens3_4, rpLens2, rpLens1, rpBend, L_Lens, LmCombiner, rpCombiner,
                                     loadBeamDiam]


    PTL_Ring=make_Ring_Version_1(ringParams)
    PTL_Injector=make_Injector_Version_1(injectorParams)
    assert PTL_Injector.combiner.outputOffset==PTL_Ring.combiner.outputOffset
    return PTL_Ring,PTL_Injector

def make_Ring_Version_1(params: lst_arr_tple)-> ParticleTracerLattice:

    rpLens3_4, rpLens1, rpLens2, rpBend, L_Lens, LmCombiner, rpCombiner, loadBeamDiam = params
    consts=constants_Version1

    if rpBend>consts["rbTarget"]:
        raise RingGeometryError


    PTL = ParticleTracerLattice(v0Nominal=DEFAULT_ATOM_SPEED, latticeType='storageRing')

    #------starting at gap 1 through lenses and gaps and combiner to gap4

    add_First_Racetrack(PTL,rpLens1,rpLens2,L_Lens,rpCombiner,LmCombiner,loadBeamDiam,rpBend,consts)

    #-------bender 1------
    PTL.add_Halbach_Bender_Sim_Segmented(consts['Lm'], rpBend, None, consts['rbTarget'])

    # ---------gap 5-----  bender->lens

    add_Bend_Adjacent_Gap(PTL, rpLens3_4, consts["lensToBendGap"], rpBend)

    #-------lens 3-----------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    # ---------gap 6------ lens -> lens
    add_Drift_If_Needed(PTL,consts["observationGap"],'lens','lens',rpLens3_4,rpLens3_4)

    # ---------lens 4-------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    #------gap 7------

    add_Bend_Adjacent_Gap(PTL, rpLens3_4, consts["observationGap"], rpBend)

    #------bender 2--------
    PTL.add_Halbach_Bender_Sim_Segmented(consts['Lm'], rpBend, None, consts['rbTarget'])

    #----done---
    PTL.end_Lattice(enforceClosedLattice=True, constrain=True)

    return PTL

def make_Injector_Version_1(injectorParams)-> ParticleTracerLattice:

    L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, \
    LmCombiner, rpCombiner, loadBeamDiam, gap1, gap2, gap3 = injectorParams
    consts = constants_Version1

    if gap2+el_Fringe_Space('lens',rpInjectorMagnet1)+el_Fringe_Space('lens',rpInjectorMagnet2)<\
            consts["lens1ToLens2_Inject_Gap"]:
        raise InjectorGeometryError


    PTL_Injector = ParticleTracerLattice(DEFAULT_ATOM_SPEED, latticeType='injector')

    PTL_Injector.add_Drift(gap1, ap=rpInjectorMagnet1)

    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet1, L_InjectorMagnet1)

    PTL_Injector.add_Drift(gap2, ap=consts["lens1ToLens2_Inject_Ap"])

    PTL_Injector.add_Halbach_Lens_Sim(rpInjectorMagnet2, L_InjectorMagnet2)

    PTL_Injector.add_Drift(gap3, ap=rpInjectorMagnet2)

    PTL_Injector.add_Combiner_Sim_Lens(LmCombiner, rpCombiner,loadBeamDiam=loadBeamDiam,layers=1)

    PTL_Injector.end_Lattice(constrain=False, enforceClosedLattice=False)

    return PTL_Injector

def make_Ring_Surrogate_Version_1(injectionParams,surrogateParamsDict: dict):
    L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, \
    LmCombiner, rpCombiner, loadBeamDiam, gap1, gap2, gap3 = injectionParams

    consts = constants_Version1
    rpBend=1.0

    rpLens1=surrogateParamsDict['rpLens1']
    rpLens2=surrogateParamsDict['rpLens2']
    L_Lens=surrogateParamsDict['L_Lens']


    PTL = ParticleTracerLattice(v0Nominal=DEFAULT_ATOM_SPEED, latticeType='storageRing')

    add_First_Racetrack(PTL, rpLens1, rpLens2, L_Lens, rpCombiner, LmCombiner, loadBeamDiam, rpBend, consts)

    PTL.end_Lattice( constrain=False)
    return PTL