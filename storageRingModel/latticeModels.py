from typing import Union, Optional
import numpy as np
from constants import DEFAULT_ATOM_SPEED
import elementPT
from latticeModels_Parameters import constantsV1_2, constantsV3, injectorParamsOptimalAny, optimizerBounds_V1_3, \
    optimizerBounds_V2, lockedDict, atomCharacteristic
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleTracerClass import ParticleTracer


class RingGeometryError(Exception):
    pass


class InjectorGeometryError(Exception):
    pass


lst_arr_tple = Union[list, np.ndarray, tuple]

h: float = 1e-5  # timestep, s. Assumed to be no larger than this
minTimeStepGap = 1.1 * h * DEFAULT_ATOM_SPEED * ParticleTracer.minTimeStepsPerElement
InjectorModel = RingModel = ParticleTracerLattice


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


def add_Bend_Version_1_2(PTL: ParticleTracerLattice, rpBend: float) -> None:
    """Single bender element"""

    PTL.add_Halbach_Bender_Sim_Segmented(constantsV1_2['Lm'], rpBend, None, constantsV1_2['rbTarget'])


def add_Bend_Version_3(PTL: ParticleTracerLattice, rpBend: float) -> None:
    """Two bender elements possibly separated by a drift region for pumping between end/beginning of benders. If fringe
    field region is long enough, no drift region is add"""

    PTL.add_Halbach_Bender_Sim_Segmented(constantsV3['Lm'], rpBend, None, constantsV3['rbTarget'])
    add_Drift_If_Needed(PTL, constantsV3['bendApexGap'], 'bender', 'bender', rpBend, rpBend)
    PTL.add_Halbach_Bender_Sim_Segmented(constantsV3['Lm'], rpBend, None, constantsV3['rbTarget'])


def add_Bender(PTL: ParticleTracerLattice, rpBend: float, whichVersion: str) -> None:
    """Add bender section to storage ring. Racetrack design requires two "benders", though each bender may actually be
    composed of more than 1 bender element and/or other elements"""

    assert whichVersion in ('1', '2', '3')
    if whichVersion in ('1', '2'):  # single bender element
        add_Bend_Version_1_2(PTL, rpBend)
    else:
        add_Bend_Version_3(PTL, rpBend)


def add_Combiner(PTL: ParticleTracerLattice, LmCombiner: float, rpCombiner: float, loadBeamDiam: float,
                 combinerSeed: Optional[int]):
    """add combiner element to PTL. Set random state for reproducible results if seed is not None, then reset the state
    """

    if combinerSeed is not None:
        state = np.random.get_state()
        np.random.seed(combinerSeed)
    PTL.add_Combiner_Sim_Lens(LmCombiner, rpCombiner, loadBeamDiam=loadBeamDiam, layers=1)
    if combinerSeed is not None:
        np.random.set_state(state)


def add_Combiner_And_OP(PTL, rpCombiner, LmCombiner, loadBeamDiam, rpLensBefore, rpLensAfter,
                        combinerSeed: Optional[int],
                        whichOP_Ap: str = "Circulating") -> None:
    """Add gap for vacuum + combiner + gap for optical pumping. Elements before and after must be a lens. """

    # gap between combiner and previous lens
    add_Drift_If_Needed(PTL, constantsV1_2["gap2Min"], 'lens', 'combiner', rpLensBefore, rpCombiner)

    # -------combiner-------

    add_Combiner(PTL, LmCombiner, rpCombiner, loadBeamDiam, combinerSeed)

    # ------gap 3--------- combiner-> lens, Optical Pumping (OP) region
    # there must be a drift here to account for the optical pumping aperture limit. It must also be at least as long
    # as optical pumping region. I am doing it like this because I don't have it coded up yet to include an aperture
    # without it being a new drift region
    OP_Gap = constantsV1_2["OP_MagWidth"] - (el_Fringe_Space('combiner', rpCombiner)
                                             + el_Fringe_Space('lens', rpLensAfter))
    OP_Gap = round_Up_If_Below_Min_Time_Step_Gap(OP_Gap)
    OP_Gap = OP_Gap if OP_Gap > constantsV1_2["OP_PumpingRegionLength"] else \
        constantsV1_2["OP_PumpingRegionLength"]
    PTL.add_Drift(OP_Gap, ap=constantsV1_2["OP_MagAp_" + whichOP_Ap])


def add_First_RaceTrack_Straight_Version1_3(PTL: ParticleTracerLattice, ringParams: lockedDict,
                                            combinerSeed: Optional[int], whichOP_Ap: str = "Circulating") -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
    elements are [lens,combiner,lens] with gaps as needed for vacuum Not actually straight because of combiner.
     """

    # ----from bender output to combiner
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens1'], ringParams['L_Lens1'])

    # ---combiner + OP magnet-----
    add_Combiner_And_OP(PTL, ringParams['rpCombiner'], ringParams['LmCombiner'], ringParams['loadBeamDiam'],
                        ringParams['rpLens1'], ringParams['rpLens2'], combinerSeed, whichOP_Ap=whichOP_Ap)

    # ---from OP to bender input---
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens2'], ringParams['L_Lens2'])


def add_First_RaceTrack_Straight_Version2(PTL: ParticleTracerLattice, ringParams: lockedDict,
                                          combinerSeed: Optional[int]) -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
    elements are [lens,lens,combiner,len,lens] with gaps as needed for vacuum Not actually straight because of combiner.
     """

    # ----from bender output to combiner
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens1'], ringParams['L_Lens1'])
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens2'], ringParams['L_Lens2'])

    # ---combiner + OP magnet-----
    add_Combiner_And_OP(PTL, ringParams['rpCombiner'], ringParams['LmCombiner'], ringParams['loadBeamDiam'],
                        ringParams['rpLens2'], ringParams['rpLens3'], combinerSeed)

    # ---from OP to bender input---
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens3'], ringParams['L_Lens3'])
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens4'], ringParams['L_Lens4'])


def add_First_Racetrack_Straight(PTL: ParticleTracerLattice, ringParams: lockedDict, whichVersion: str,
                                 combinerSeed: Optional[int]) -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
     Not actually straight because of combiner. Two lenses and a combiner, with supporting drift regions if
     neccesary for gap spacing"""

    # No need for gap before first lens
    assert whichVersion in ('1', '2', '3')

    if whichVersion in ('1', '3'):
        add_First_RaceTrack_Straight_Version1_3(PTL, ringParams, combinerSeed)
    else:
        add_First_RaceTrack_Straight_Version2(PTL, ringParams, combinerSeed)


def add_Second_Racetrack_Straight_Version_Any(PTL: ParticleTracerLattice, rpLens3_4: float, rpBend: float) -> None:
    """Going from output of bender in +x direction to input of next bender. Two lenses with supporting drift regions if
     neccesary for gap spacing"""

    # ---------gap 5-----  bender->lens

    add_Drift_If_Needed(PTL, constantsV1_2["lensToBendGap"], 'lens', 'bender', rpLens3_4, rpBend)

    # -------lens 3-----------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    # ---------gap 6------ lens -> lens
    add_Drift_If_Needed(PTL, constantsV1_2["observationGap"], 'lens', 'lens', rpLens3_4, rpLens3_4)

    # ---------lens 4-------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    # ------gap 7------ lens-> bender

    add_Drift_If_Needed(PTL, constantsV1_2["lensToBendGap"], 'lens', 'bender', rpLens3_4, rpBend)


def make_Ring(ringParams: lockedDict, whichVersion: str, useMagnetErrors: bool,
              combinerSeed: Optional[int]) -> RingModel:
    """Make ParticleTraceLattice object that represents storage ring. This does not include the injector components.
    Several versions available

    Version 1: in clockwise order starting at 0,0 and going in -x direction elements are [lens, combiner, lens, bender,
    lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation

    Version 3: Same as Version 1, but instead with the bending sections split into to bending elements so pumping
    can be applied at apex
    """

    assert whichVersion in ('1', '2', '3')

    PTL = ParticleTracerLattice(v0Nominal=atomCharacteristic["nominalDesignSpeed"], latticeType='storageRing',
                                standardMagnetErrors=useMagnetErrors,useSolenoidField=True)

    # ------starting at gap 1 through lenses and gaps and combiner to gap4

    add_First_Racetrack_Straight(PTL, ringParams, whichVersion, combinerSeed)

    # -------bender 1------
    add_Bender(PTL, ringParams['rpBend'], whichVersion)

    # -----starting at gap 5r through gap 7r-----

    add_Second_Racetrack_Straight_Version_Any(PTL, ringParams['rpLens3_4'], ringParams['rpBend'])

    # ------bender 2--------
    add_Bender(PTL, ringParams['rpBend'], whichVersion)

    # ----done---
    PTL.end_Lattice(constrain=True)

    ringParams.assert_All_Entries_Accessed_And_Reset_Counter()

    return PTL


def make_Injector_Version_Any(injectorParams: lockedDict, useMagnetError: bool = False, combinerSeed: int = None) \
        -> InjectorModel:
    """Make ParticleTraceLattice object that represents injector. Injector is a double lens design. """

    gap1 = injectorParams["gap1"]
    gap2 = injectorParams["gap2"]
    gap3 = injectorParams["gap3"]

    gap1 -= el_Fringe_Space('lens', injectorParams["rp1"])
    gap2 -= el_Fringe_Space('lens', injectorParams["rp1"]) + el_Fringe_Space('lens', injectorParams["rp2"])
    if gap2 < constantsV1_2["lens1ToLens2_Inject_Gap"]:
        raise InjectorGeometryError
    if gap1 < constantsV1_2["sourceToLens1_Inject_Gap"]:
        raise InjectorGeometryError

    PTL = ParticleTracerLattice(atomCharacteristic["nominalDesignSpeed"], latticeType='injector',
                                standardMagnetErrors=useMagnetError,useSolenoidField=True)

    # -----gap between source and first lens-----

    add_Drift_If_Needed(PTL, gap1, 'none', 'lens', np.inf, injectorParams["rp1"])  # hacky

    # ---- first lens------

    PTL.add_Halbach_Lens_Sim(injectorParams["rp1"], injectorParams["L1"])

    # -----gap with valve--------------

    gapValve = constantsV1_2["lens1ToLens2_Valve_Length"]
    gap2 = gap2 - gapValve
    if gap2 < 0:  # this is approximately true. I am ignoring that there is space in the fringe fields
        raise InjectorGeometryError
    PTL.add_Drift(gap2, ap=injectorParams["rp1"])
    PTL.add_Drift(gapValve, ap=constantsV1_2["lens1ToLens2_Valve_Ap"],
                  outerHalfWidth=constantsV1_2["lens1ToLens2_Inject_Valve_OD"] / 2)

    # ---------------------

    PTL.add_Halbach_Lens_Sim(injectorParams["rp2"], injectorParams["L2"])

    PTL.add_Drift(gap3, ap=injectorParams["rp2"])

    add_Combiner(PTL, injectorParams["LmCombiner"], injectorParams["rpCombiner"], injectorParams["loadBeamDiam"],
                 combinerSeed)

    PTL.end_Lattice(constrain=False)

    injectorParams.assert_All_Entries_Accessed_And_Reset_Counter()

    return PTL


def make_Ring_Surrogate_For_Injection_Version_1(injectorParams: lockedDict,
                                                surrogateParamsDict: lockedDict) -> RingModel:
    """Surrogate model of storage to aid in realism of independent injector optimizing. Benders are exluded. Model
    serves to represent geometric constraints between injector and ring, and to test that if particle that travel
    into combiner can make it to the next element. Since injector is optimized independent of ring, the parameters
    for the ring surrogate are typically educated guesses"""

    raceTrackParams = lockedDict({'rpCombiner': injectorParams['rpCombiner'],
                                  'LmCombiner': injectorParams['LmCombiner'],
                                  'loadBeamDiam': injectorParams['loadBeamDiam'],
                                  'rpLens1': surrogateParamsDict['rpLens1'],
                                  'L_Lens1': surrogateParamsDict['L_Lens'],
                                  'rpLens2': surrogateParamsDict['rpLens2'],
                                  'L_Lens2': surrogateParamsDict['L_Lens']})

    PTL = ParticleTracerLattice(v0Nominal=atomCharacteristic["nominalDesignSpeed"], latticeType='storageRing',useSolenoidField=True)

    add_First_RaceTrack_Straight_Version1_3(PTL, raceTrackParams, None, whichOP_Ap='Injection')
    raceTrackParams.assert_All_Entries_Accessed_And_Reset_Counter()
    PTL.end_Lattice(constrain=False)
    return PTL


def make_ringParams_Dict(variableParams: list[float], whichVersion: str) -> lockedDict:
    """Take parameters values list and construct dictionary of variable ring parameters. For version1, all tunable
    variables (lens length, radius, etc) describe the ring only. Injector is optimized entirely independenly before"""

    assert whichVersion in ('1', '2', '3')

    ringParams = {"LmCombiner": injectorParamsOptimalAny["LmCombiner"],
                  "rpCombiner": injectorParamsOptimalAny["rpCombiner"],
                  "loadBeamDiam": injectorParamsOptimalAny["loadBeamDiam"]}

    if whichVersion in ('1', '3'):
        assert len(variableParams) == 6
        for variableKey, value in zip(optimizerBounds_V1_3.keys(), variableParams):
            ringParams[variableKey] = value
    else:
        assert len(variableParams) == 10
        for variableKey, value in zip(optimizerBounds_V2.keys(), variableParams):
            ringParams[variableKey] = value
    ringParams = lockedDict(ringParams)
    return ringParams


def make_injectorParams_Dict_Version_Any(variableParams: list[float]) -> lockedDict:
    """For now, all parameters of injector are constant"""

    return injectorParamsOptimalAny


def assert_Combiners_Are_Same(PTL_Injector: ParticleTracerLattice, PTL_Ring: ParticleTracerLattice) -> None:
    """Combiner from injector and ring must have the same shared characteristics, as well as have the expected
    parameters"""

    assert PTL_Injector.combiner.outputOffset == PTL_Ring.combiner.outputOffset
    assert PTL_Injector.combiner.ang < 0 < PTL_Ring.combiner.ang


def _make_Ring_And_Injector(variableParams: lst_arr_tple, whichVersion: str, useMagnetErrors: bool,
                            combinerSeed: Optional[int]) -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTracerLattice models of ring and injector system. Combiner must be the same, though in low
    field seeking configuration for ring, and high field seeking for injector

    :param variableParams: Non constant parameters to construct lens.
    :param whichVersion: which version of the ring/injector system to use
    :param useMagnetErrors: Wether to apply neodymium individual cubiod magnet material/alignment errors
    :param combinerSeed: random number seed for combiner element. WIthout this, using magnet errors will produce
        different combiners because their magnet will be perturbed differently. This is unphysical
    :return:
    """

    assert whichVersion in ('1', '2', '3')
    assert all(val > 0 for val in variableParams)
    assert isinstance(combinerSeed, int) if combinerSeed is not None else True

    ringParams = make_ringParams_Dict(variableParams, whichVersion)
    injectorParams = make_injectorParams_Dict_Version_Any(variableParams)

    PTL_Ring = make_Ring(ringParams, whichVersion, useMagnetErrors, combinerSeed)
    PTL_Injector = make_Injector_Version_Any(injectorParams, useMagnetError=useMagnetErrors, combinerSeed=combinerSeed)
    assert_Combiners_Are_Same(PTL_Injector, PTL_Ring)
    return PTL_Ring, PTL_Injector


def make_Ring_And_Injector_Version1(variableParams: lst_arr_tple) -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTraceLattice objects that represents storage ring and injector systems

    Version 1: in clockwise order starting at 0,0 and going in -x direction elements are [lens, combiner, lens, bender,
    lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation. This is a simple
    design
    """

    version = '1'
    return _make_Ring_And_Injector(variableParams, version, False, None)


def make_Ring_And_Injector_Version2(variableParams: lst_arr_tple) -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTraceLattice objects that represents storage ring and injector systems

    Version 2: in clockwise order starting at 0,0 and going in -x direction elements are [lens,lens, combiner,lens, 
    lens, bender, lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation. The
    idea of this version is that having double lenses flanking the combiner can improve mode matching
    """

    version = '2'
    return _make_Ring_And_Injector(variableParams, version, False, None)


def make_Ring_And_Injector_Version3(variableParams: lst_arr_tple, useMagnetErrors: bool = False,
                                    combinerSeed: int = None) \
        -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTraceLattice objects that represents storage ring and injector systems

    Version 3: Same as Version 1, but instead with the bending sections split into to bending elements so pumping
    can be applied at apex
    """

    version = '3'
    return _make_Ring_And_Injector(variableParams, version, useMagnetErrors, combinerSeed)
