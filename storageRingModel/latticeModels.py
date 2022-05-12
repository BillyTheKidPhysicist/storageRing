from typing import  Union
import numpy as np
from constants import DEFAULT_ATOM_SPEED
import elementPT
from latticeModels_Parameters import constantsV1,constantsV3, injectorParamsOptimalV1,optimizerBounds_V1,\
        lockedDict
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleTracerClass import ParticleTracer


class RingGeometryError(Exception):
    pass


class InjectorGeometryError(Exception):
    pass


lst_arr_tple = Union[list, np.ndarray, tuple]

h: float = 1e-5  # timestep, s. Assumed to be no larger than this
minTimeStepGap = h * DEFAULT_ATOM_SPEED * ParticleTracer.minTimeStepsPerElement
InjectorModel=RingModel=ParticleTracerLattice


def el_Fringe_Space(elementName: str, elementBoreRadius: float) -> float:
    """Return the gap between hard edge of element (magnetic material) and end of element model. This gap exists
    to allow the field values to fall to negligeable amounts"""

    assert elementBoreRadius > 0
    if elementName == 'none':
        return 0.0
    fringeFracs = {"combiner": elementPT.CombinerHalbachLensSim.outerFringeFrac,
                   "lens": elementPT.HalbachLensSim.fringeFracOuter,
                   "bender": elementPT.HalbachBenderSimSegmented.fringeFracOuter}
    return fringeFracs[elementName] * elementBoreRadius

def round_Up_If_Below_Min_Time_Step_Gap(proposedLength: float) -> float:
    """Elements have a minimum length dictated by ParticleTracerClass for time stepping considerations. A  reasonable
    value for the time stepping is assumed. If wrong, an error will be thrown in ParticleTracerClass"""

    if proposedLength < minTimeStepGap:
        return minTimeStepGap
    else:
        return proposedLength

def add_Drift_If_Needed(PTL: ParticleTracerLattice, gapLength: float, elBeforeName: str,
                        elAfterName: str, elBefore_rp: float, elAfter_rp: float, ap: float = None) -> None:
    """Sometimes the fringe field gap is enough to accomodate the minimum desired separation between elements.
    Otherwise a gap needs to be added. The drift will have a minimum length, so the total gap may be larger in some
    cases"""

    assert gapLength >= 0 and elAfter_rp > 0 and elBefore_rp > 0
    extraSpace = gapLength - (el_Fringe_Space(elBeforeName, elBefore_rp) + el_Fringe_Space(elAfterName, elAfter_rp))
    if extraSpace > 0:
        ap = min([elBefore_rp, elAfter_rp]) if ap is None else ap
        PTL.add_Drift(round_Up_If_Below_Min_Time_Step_Gap(extraSpace), ap=ap)

def add_Bend_Version_1(PTL: ParticleTracerLattice,rpBend: float)-> None:
    """Single bender element"""

    PTL.add_Halbach_Bender_Sim_Segmented(constantsV1['Lm'], rpBend, None, constantsV1['rbTarget'])

def add_Bend_Version_3(PTL: ParticleTracerLattice,rpBend: float)-> None:
    """Two bender elements possibly separated by a drift region for pumping between end/beginning of benders. If fringe
    field region is long enough, no drift region is add"""

    PTL.add_Halbach_Bender_Sim_Segmented(constantsV3['Lm'], rpBend, None, constantsV3['rbTarget'])
    add_Drift_If_Needed(PTL,constantsV3['bendApexGap'],'bender','bender',rpBend,rpBend)
    PTL.add_Halbach_Bender_Sim_Segmented(constantsV3['Lm'], rpBend, None, constantsV3['rbTarget'])

def add_Bender(PTL: ParticleTracerLattice,rpBend: float,whichVersion: str)-> None:
    """Add bender section to storage ring. Racetrack design requires two "benders", though each bender may actually be
    composed of more than 1 bender element and/or other elements"""

    assert whichVersion in ('1','3')
    if whichVersion=='1': #single bender element
        add_Bend_Version_1(PTL,rpBend)
    elif whichVersion=='2':#two bender elements, possibly a drift element
        raise NotImplementedError
    else:
        add_Bend_Version_3(PTL,rpBend)

def add_Combiner_And_OP(PTL,rpCombiner,LmCombiner,loadBeamDiam,rpLensBefore,rpLensAfter)-> None:
    """Add gap for vacuum + combiner + gap for optical pumping. Elements before and after must be a lens. """

    #gap between combiner and previous lens
    add_Drift_If_Needed(PTL, constantsV1["gap2Min"], 'lens', 'combiner', rpLensBefore, rpCombiner)

    # -------combiner-------

    PTL.add_Combiner_Sim_Lens(LmCombiner, rpCombiner, loadBeamDiam=loadBeamDiam, layers=1)

    # ------gap 3--------- combiner-> lens, Optical Pumping (OP) region
    # there must be a drift here to account for the optical pumping aperture limit. It must also be at least as long
    # as optical pumping region. I am doing it like this because I don't have it coded up yet to include an aperture
    # without it being a new drift region
    OP_Gap = constantsV1["OP_MagWidth"] - (el_Fringe_Space('combiner', rpCombiner)
                                           + el_Fringe_Space('lens', rpLensAfter))
    OP_Gap = round_Up_If_Below_Min_Time_Step_Gap(OP_Gap)
    OP_Gap = OP_Gap if OP_Gap > constantsV1["OP_PumpingRegionLength"] else \
        constantsV1["OP_PumpingRegionLength"]
    PTL.add_Drift(OP_Gap, ap=constantsV1["OP_MagAp"])

def add_First_Racetrack_Straight_Version1(PTL, ringParams: lockedDict)-> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
     Not actually straight because of combiner. Two lenses and a combiner, with supporting drift regions if
     neccesary for gap spacing"""

    # ------gap 1--------  bender-> lens
    # there is none here because of strong adjacent vacuum pumping

    # --------lens 1---------

    PTL.add_Halbach_Lens_Sim(ringParams['rpLens1'], ringParams['L_Lens1'])

    #---gap 2 + combiner + OP magnet-----

    add_Combiner_And_OP(PTL,ringParams['rpCombiner'],ringParams['LmCombiner'],ringParams['loadBeamDiam'],
                        ringParams['rpLens1'],ringParams['rpLens2'])

    # -------lens 2-------
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens2'], ringParams['L_Lens2'])

    # ---------gap 4----- lens-> bender
    add_Drift_If_Needed(PTL, constantsV1["lensToBendGap"], 'lens', 'bender', ringParams['rpLens2'],
                        ringParams['rpBend'])

def add_Second_Racetrack_Straight_Version_Any(PTL: ParticleTracerLattice,rpLens3_4: float,rpBend: float)-> None:
    """Going from output of bender in +x direction to input of next bender. Two lenses with supporting drift regions if
     neccesary for gap spacing"""

    # ---------gap 5-----  bender->lens

    add_Drift_If_Needed(PTL, constantsV1["lensToBendGap"], 'lens', 'bender', rpLens3_4, rpBend)

    # -------lens 3-----------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    # ---------gap 6------ lens -> lens
    add_Drift_If_Needed(PTL, constantsV1["observationGap"], 'lens', 'lens', rpLens3_4, rpLens3_4)

    # ---------lens 4-------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    # ------gap 7------ lens-> bender

    add_Drift_If_Needed(PTL, constantsV1["lensToBendGap"], 'lens', 'bender', rpLens3_4, rpBend)

def make_Ring(ringParams: lockedDict, whichVersion: str) -> RingModel:
    """Make ParticleTraceLattice object that represents storage ring. This does not include the injector components.
    Several versions available

    Version 1: in clockwise order starting at 0,0 and going in -x direction elements are [lens, combiner, lens, bender,
    lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation

    Version 3: Same as Version 1, but instead with the bending sections split into to bending elements so pumping
    can be applied at apex
    """

    assert whichVersion in ('1','3')

    PTL = ParticleTracerLattice(v0Nominal=DEFAULT_ATOM_SPEED, latticeType='storageRing')

    # ------starting at gap 1 through lenses and gaps and combiner to gap4

    add_First_Racetrack_Straight_Version1(PTL, ringParams)

    # -------bender 1------
    add_Bender(PTL,ringParams['rpBend'],whichVersion)

    # -----starting at gap 5r through gap 7r-----

    add_Second_Racetrack_Straight_Version_Any(PTL, ringParams['rpLens3_4'], ringParams['rpBend'])

    # ------bender 2--------
    add_Bender(PTL,ringParams['rpBend'],whichVersion)

    # ----done---
    PTL.end_Lattice(constrain=True)

    ringParams.assert_All_Entries_Accessed_And_Reset_Counter()

    return PTL

def make_Injector_Version_Any(injectorParams: lockedDict) -> InjectorModel:
    """Make ParticleTraceLattice object that represents injector. Injector is a double lens design. """

    gap1=injectorParams["gap1"]
    gap2=injectorParams["gap2"]
    gap3=injectorParams["gap3"]

    gap1 -= el_Fringe_Space('lens', injectorParams["rp1"])
    gap2 -= el_Fringe_Space('lens', injectorParams["rp1"]) + el_Fringe_Space('lens', injectorParams["rp2"])
    if gap2 < constantsV1["lens1ToLens2_Inject_Gap"]:
        raise InjectorGeometryError
    if gap1 < constantsV1["sourceToLens1_Inject_Gap"]:
        raise InjectorGeometryError

    PTL = ParticleTracerLattice(DEFAULT_ATOM_SPEED, latticeType='injector')

    # -----gap between source and first lens-----

    add_Drift_If_Needed(PTL, gap1, 'none', 'lens', np.inf, injectorParams["rp1"])  # hacky

    # ---- first lens------

    PTL.add_Halbach_Lens_Sim(injectorParams["rp1"], injectorParams["L1"])

    # -----gap with valve--------------

    gapValve = constantsV1["lens1ToLens2_Valve_Length"]
    gap2 = gap2 - gapValve
    if gap2 < 0:  # this is approximately true. I am ignoring that there is space in the fringe fields
        raise InjectorGeometryError
    PTL.add_Drift(gap2, ap=injectorParams["rp1"])
    PTL.add_Drift(gapValve, ap=constantsV1["lens1ToLens2_Valve_Ap"],
                  outerHalfWidth=constantsV1["lens1ToLens2_Inject_Valve_OD"] / 2)

    # ---------------------

    PTL.add_Halbach_Lens_Sim(injectorParams["rp2"], injectorParams["L2"])

    PTL.add_Drift(gap3, ap=injectorParams["rp2"])

    PTL.add_Combiner_Sim_Lens(injectorParams["LmCombiner"],
                              injectorParams["rpCombiner"],
                              loadBeamDiam=injectorParams["loadBeamDiam"], layers=1)

    PTL.end_Lattice(constrain=False)

    injectorParams.assert_All_Entries_Accessed_And_Reset_Counter()

    return PTL

def make_Ring_Surrogate_Version_1(injectorParams: tuple[float,...], surrogateParamsDict: dict)-> RingModel:
    """Surrogate model of storage to aid in realism of independent injector optimizing. Benders are exluded. Model
    serves to represent geometric constraints between injector and ring, and to test that if particle that travel
    into combiner can make it to the next element. Since injector is optimized independent of ring, the parameters
    for the ring surrogate are typically educated guesses"""

    raise NotImplementedError #need to fix

    assert all(val > 0 for val in injectorParams)
    L_InjectorMagnet1, rpInjectorMagnet1, L_InjectorMagnet2, rpInjectorMagnet2, \
    LmCombiner, rpCombiner, loadBeamDiam, gap1, gap2, gap3 = injectorParams

    rpBend = constantsV1["rbTarget"]
    rpLens1 = surrogateParamsDict['rpLens1']
    rpLens2 = surrogateParamsDict['rpLens2']
    L_Lens = surrogateParamsDict['L_Lens']
    raceTrackParams = rpLens1, rpLens2, L_Lens, rpCombiner, LmCombiner, loadBeamDiam, rpBend

    PTL = ParticleTracerLattice(v0Nominal=DEFAULT_ATOM_SPEED, latticeType='storageRing')

    add_First_Racetrack_Straight_Version1(PTL, raceTrackParams)

    PTL.end_Lattice(constrain=False)
    return PTL

def make_ringParams_Dict_Version1(variableParams: list[float])-> lockedDict:
    """Take parameters values list and construct dictionary of variable ring parameters. For version1, all tunable
    variables (lens length, radius, etc) describe the ring only. Injector is optimized entirely independenly before"""

    assert len(variableParams)==6

    ringParams={"LmCombiner":injectorParamsOptimalV1["LmCombiner"],
                "rpCombiner":injectorParamsOptimalV1["rpCombiner"],
                "loadBeamDiam":injectorParamsOptimalV1["loadBeamDiam"]}

    for variableKey, value in zip(optimizerBounds_V1.keys(),variableParams):
        ringParams[variableKey]=value
    ringParams=lockedDict(ringParams)
    return ringParams

def make_injectorParams_Dict_Version_Any(variableParams: list[float])-> lockedDict:
    """For now, all parameters of injector are constant"""

    return injectorParamsOptimalV1

def _make_Ring_And_Injector(variableParams: lst_arr_tple, whichVersion: str) -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTracerLattice models of ring and injector system. Combiner must be the same, though in low
    field seeking configuration for ring, and high field seeking for injector

    :param variableParams: Non constant parameters to construct lens.
    :param whichVersion:
    :return:
    """

    assert whichVersion in ('1','3')
    assert all(val > 0 for val in variableParams)

    ringParams=make_ringParams_Dict_Version1(variableParams)
    injectorParams=make_injectorParams_Dict_Version_Any(variableParams)

    PTL_Ring = make_Ring(ringParams,whichVersion)
    PTL_Injector = make_Injector_Version_Any(injectorParams)
    assert PTL_Injector.combiner.outputOffset == PTL_Ring.combiner.outputOffset
    assert PTL_Injector.combiner.ang < 0 <PTL_Ring.combiner.ang
    return PTL_Ring, PTL_Injector

def make_Ring_And_Injector_Version1(variableParams: lst_arr_tple) -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTraceLattice objects that represents storage ring and injector systems

    Version 1: in clockwise order starting at 0,0 and going in -x direction elements are [lens, combiner, lens, bender,
    lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation

    :param variableParams: Non constant parameters of the ring/injector system.
    :return: Ring and injector model
    """

    version='1'
    return _make_Ring_And_Injector(variableParams,version)

def make_Ring_And_Injector_Version3(variableParams: lst_arr_tple) -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTraceLattice objects that represents storage ring and injector systems

    Version 3: Same as Version 1, but instead with the bending sections split into to bending elements so pumping
    can be applied at apex

    :param variableParams: Non constant parameters of the ring/injector system.
    :return: Ring and injector model
    """

    version='3'
    return _make_Ring_And_Injector(variableParams,version)
