from typing import Union, Optional

import numpy as np

from KevinBumperClass import add_Kevin_Bumper_Elements
from ParticleTracerClass import ParticleTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from constants import DEFAULT_ATOM_SPEED
from latticeElements.elements import CombinerHalbachLensSim, HalbachLensSim, HalbachBenderSimSegmented
from latticeModels_Parameters import system_constants, optimizerBounds_V1_3, \
    optimizerBounds_V2, lockedDict, atomCharacteristic, injectorParamsBoundsAny
from latticeElements.utilities import min_Bore_Radius_From_Tube_OD


class RingGeometryError(Exception):
    pass


class InjectorGeometryError(Exception):
    pass


lst_arr_tple = Union[list, np.ndarray, tuple]

h: float = 1e-5  # timestep, s. Assumed to be no larger than this
minTimeStepGap = 1.1 * h * DEFAULT_ATOM_SPEED * ParticleTracer.minTimeStepsPerElement
InjectorModel = RingModel = ParticleTracerLattice
DEFAULT_SYSTEM_OPTIONS = lockedDict({'useMagnetErrors': False, 'combinerSeed': None, 'useSolenoidField': False,
                                     'includeBumper': False})


def check_And_Add_Default_Values(options: Optional[dict]) -> lockedDict:
    """Check that there are only the allowed keys in options, and any missing keys add them with the default
    value. If options is None, use the default dictionary"""

    if options is not None:
        numValidKeysInDict = sum([1 for key in DEFAULT_SYSTEM_OPTIONS if key in options])
        assert numValidKeysInDict == len(options)  # can't be unexpected stuff
        for key, val in DEFAULT_SYSTEM_OPTIONS.items():
            if key not in options:
                options[key] = val
        options = lockedDict(options)
    else:
        options = DEFAULT_SYSTEM_OPTIONS
    return options


def el_Fringe_Space(elementName: str, elementBoreRadius: float) -> float:
    """Return the gap between hard edge of element (magnetic material) and end of element model. This gap exists
    to allow the field values to fall to negligeable amounts"""

    assert elementBoreRadius > 0
    if elementName == 'none':
        return 0.0
    fringeFracs = {"combiner": CombinerHalbachLensSim.outerFringeFrac,
                   "lens": HalbachLensSim.fringeFracOuter,
                   "bender": HalbachBenderSimSegmented.fringeFracOuter}
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

    PTL.add_Halbach_Bender_Sim_Segmented(system_constants['Lm'], rpBend, None, system_constants['rbTarget'])


def add_Bend_Version_3(PTL: ParticleTracerLattice, rpBend: float) -> None:
    """Two bender elements possibly separated by a drift region for pumping between end/beginning of benders. If fringe
    field region is long enough, no drift region is add"""

    PTL.add_Halbach_Bender_Sim_Segmented(system_constants['Lm'], rpBend, None, system_constants['rbTarget'])
    add_Drift_If_Needed(PTL, system_constants['bendApexGap'], 'bender', 'bender', rpBend, rpBend)
    PTL.add_Halbach_Bender_Sim_Segmented(system_constants['Lm'], rpBend, None, system_constants['rbTarget'])


def add_Bender(PTL: ParticleTracerLattice, rpBend: float, whichVersion: str) -> None:
    """Add bender section to storage ring. Racetrack design requires two "benders", though each bender may actually be
    composed of more than 1 bender element and/or other elements"""

    assert whichVersion in ('1', '2', '3')
    if whichVersion in ('1', '2'):  # single bender element
        add_Bend_Version_1_2(PTL, rpBend)
    else:
        add_Bend_Version_3(PTL, rpBend)


def add_Combiner(PTL: ParticleTracerLattice, LmCombiner: float, rpCombiner: float, loadBeamOffset: float,
                 combinerSeed: Optional[int]):
    """add combiner element to PTL. Set random state for reproducible results if seed is not None, then reset the state
    """

    if combinerSeed is not None:
        state = np.random.get_state()
        np.random.seed(combinerSeed)
    PTL.add_Combiner_Sim_Lens(LmCombiner, rpCombiner, loadBeamOffset=loadBeamOffset, layers=1)
    if combinerSeed is not None:
        np.random.set_state(state)


def add_Combiner_And_OP(PTL, rpCombiner, LmCombiner, loadBeamOffset, rpLensBefore, rpLensAfter,
                        options: Optional[dict], whichOP_Ap: str = "Circulating") -> None:
    """Add gap for vacuum + combiner + gap for optical pumping. Elements before and after must be a lens. """

    # gap between combiner and previous lens
    add_Drift_If_Needed(PTL, system_constants["gap2Min"], 'lens', 'combiner', rpLensBefore, rpCombiner)

    # -------combiner-------

    add_Combiner(PTL, LmCombiner, rpCombiner, loadBeamOffset, options['combinerSeed'])

    # ------gap 3--------- combiner-> lens, Optical Pumping (OP) region
    # there must be a drift here to account for the optical pumping aperture limit. It must also be at least as long
    # as optical pumping region. I am doing it like this because I don't have it coded up yet to include an aperture
    # without it being a new drift region
    OP_Gap = system_constants["OP_MagWidth"] - (el_Fringe_Space('combiner', rpCombiner)
                                                + el_Fringe_Space('lens', rpLensAfter))
    OP_Gap = round_Up_If_Below_Min_Time_Step_Gap(OP_Gap)
    OP_Gap = OP_Gap if OP_Gap > system_constants["OP_PumpingRegionLength"] else \
        system_constants["OP_PumpingRegionLength"]
    PTL.add_Drift(OP_Gap, ap=system_constants["OP_MagAp_" + whichOP_Ap])


def add_First_RaceTrack_Straight_Version1_3(PTL: ParticleTracerLattice, ringParams: lockedDict,
                                            options: Optional[dict], whichOP_Ap: str = "Circulating") -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
    elements are [lens,combiner,lens] with gaps as needed for vacuum Not actually straight because of combiner.
     """

    # ----from bender output to combiner
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens1'], ringParams['L_Lens1'])

    # ---combiner + OP magnet-----
    add_Combiner_And_OP(PTL, system_constants['rpCombiner'], ringParams['LmCombiner'], ringParams['loadBeamOffset'],
                        ringParams['rpLens1'], ringParams['rpLens2'], options, whichOP_Ap=whichOP_Ap)

    # ---from OP to bender input---
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens2'], ringParams['L_Lens2'])


def add_First_RaceTrack_Straight_Version2(PTL: ParticleTracerLattice, ringParams: lockedDict,
                                          options: Optional[dict]) -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
    elements are [lens,lens,combiner,len,lens] with gaps as needed for vacuum Not actually straight because of combiner.
     """

    # ----from bender output to combiner
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens1'], ringParams['L_Lens1'])
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens2'], ringParams['L_Lens2'])

    # ---combiner + OP magnet-----
    add_Combiner_And_OP(PTL, ringParams['rpCombiner'], ringParams['LmCombiner'], ringParams['loadBeamOffset'],
                        ringParams['rpLens2'], ringParams['rpLens3'], options['combinerSeed'])

    # ---from OP to bender input---
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens3'], ringParams['L_Lens3'])
    PTL.add_Halbach_Lens_Sim(ringParams['rpLens4'], ringParams['L_Lens4'])


def add_First_Racetrack_Straight(PTL: ParticleTracerLattice, ringParams: lockedDict, whichVersion: str,
                                 options: Optional[dict]) -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
     Not actually straight because of combiner. Two lenses and a combiner, with supporting drift regions if
     neccesary for gap spacing"""

    # No need for gap before first lens
    assert whichVersion in ('1', '2', '3')

    if whichVersion in ('1', '3'):
        add_First_RaceTrack_Straight_Version1_3(PTL, ringParams, options)
    else:
        add_First_RaceTrack_Straight_Version2(PTL, ringParams, options)


def add_Second_Racetrack_Straight_Version_Any(PTL: ParticleTracerLattice, rpLens3_4: float, rpBend: float) -> None:
    """Going from output of bender in +x direction to input of next bender. Two lenses with supporting drift regions if
     neccesary for gap spacing"""

    # ---------gap 5-----  bender->lens

    add_Drift_If_Needed(PTL, system_constants["lensToBendGap"], 'lens', 'bender', rpLens3_4, rpBend)

    # -------lens 3-----------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    # ---------gap 6------ lens -> lens
    add_Drift_If_Needed(PTL, system_constants["observationGap"], 'lens', 'lens', rpLens3_4, rpLens3_4)

    # ---------lens 4-------
    PTL.add_Halbach_Lens_Sim(rpLens3_4, None, constrain=True)

    # ------gap 7------ lens-> bender

    add_Drift_If_Needed(PTL, system_constants["lensToBendGap"], 'lens', 'bender', rpLens3_4, rpBend)


def make_Ring(ringParams: lockedDict, whichVersion: str, options: Optional[dict]) -> RingModel:
    """Make ParticleTraceLattice object that represents storage ring. This does not include the injector components.
    Several versions available

    Version 1: in clockwise order starting at 0,0 and going in -x direction elements are [lens, combiner, lens, bender,
    lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation

    Version 3: Same as Version 1, but instead with the bending sections split into to bending elements so pumping
    can be applied at apex
    """
    options = check_And_Add_Default_Values(options)
    assert whichVersion in ('1', '2', '3')

    PTL = ParticleTracerLattice(v0Nominal=atomCharacteristic["nominalDesignSpeed"], latticeType='storageRing',
                        standardMagnetErrors=options['useMagnetErrors'],useSolenoidField=options['useSolenoidField'],
                                standard_tube_ODs=True,standard_mag_sizes=True)

    # ------starting at gap 1 through lenses and gaps and combiner to gap4

    add_First_Racetrack_Straight(PTL, ringParams, whichVersion, options)

    rpBend=ringParams['rpBend']

    # -------bender 1------
    add_Bender(PTL,rpBend, whichVersion)

    # -----starting at gap 5r through gap 7r-----

    add_Second_Racetrack_Straight_Version_Any(PTL, ringParams['rpLens3_4'], rpBend)

    # ------bender 2--------
    add_Bender(PTL, rpBend, whichVersion)

    # ----done---
    PTL.end_Lattice(constrain=True)

    ringParams.assert_All_Entries_Accessed_And_Reset_Counter()

    return PTL


def make_Injector_Version_Any(injectorParams: lockedDict, options: dict = None) -> InjectorModel:
    """Make ParticleTraceLattice object that represents injector. Injector is a double lens design. """

    options = check_And_Add_Default_Values(options)
    gap1 = injectorParams["gap1"]
    gap2 = injectorParams["gap2"]
    gap3 = injectorParams["gap3"]
    gap1 -= el_Fringe_Space('lens', injectorParams["rp1"])
    gap2 -= el_Fringe_Space('lens', injectorParams["rp1"]) + el_Fringe_Space('lens', injectorParams["rp2"])
    if gap2 < system_constants["lens1ToLens2_Inject_Gap"]:
        raise InjectorGeometryError
    if gap1 < system_constants["sourceToLens1_Inject_Gap"]:
        raise InjectorGeometryError
    PTL = ParticleTracerLattice(atomCharacteristic["nominalDesignSpeed"], latticeType='injector',
                        standardMagnetErrors=options['useMagnetErrors'],useSolenoidField=options['useSolenoidField'],
                        standard_tube_ODs=True,standard_mag_sizes=True)
    if options['includeBumper']:
        add_Kevin_Bumper_Elements(PTL)

    # -----gap between source and first lens-----

    add_Drift_If_Needed(PTL, gap1, 'none', 'lens', np.inf, injectorParams["rp1"])  # hacky

    # ---- first lens------

    PTL.add_Halbach_Lens_Sim(injectorParams["rp1"], injectorParams["L1"])

    # -----gap with valve--------------

    gapValve = system_constants["lens1ToLens2_Valve_Length"]
    gap2 = gap2 - gapValve
    if gap2 < 0:  # this is approximately true. I am ignoring that there is space in the fringe fields
        raise InjectorGeometryError
    PTL.add_Drift(gap2, ap=injectorParams["rp1"])
    PTL.add_Drift(gapValve, ap=system_constants["lens1ToLens2_Valve_Ap"],
                  outerHalfWidth=system_constants["lens1ToLens2_Inject_Valve_OD"] / 2)

    # ---------------------

    PTL.add_Halbach_Lens_Sim(injectorParams["rp2"], injectorParams["L2"])

    PTL.add_Drift(gap3, ap=injectorParams["rp2"])

    add_Combiner(PTL, injectorParams["LmCombiner"], system_constants["rpCombiner"], injectorParams["loadBeamOffset"],
                 options['combinerSeed'])

    PTL.end_Lattice(constrain=False)

    injectorParams.assert_All_Entries_Accessed_And_Reset_Counter()

    return PTL


def make_Ring_Surrogate_For_Injection_Version_1(injectorParams: lockedDict,
                                                surrogateParamsDict: lockedDict, options: dict = None) -> RingModel:
    """Surrogate model of storage to aid in realism of independent injector optimizing. Benders are exluded. Model
    serves to represent geometric constraints between injector and ring, and to test that if particle that travel
    into combiner can make it to the next element. Since injector is optimized independent of ring, the parameters
    for the ring surrogate are typically educated guesses"""

    options = check_And_Add_Default_Values(options)

    raceTrackParams = lockedDict({'LmCombiner': injectorParams['LmCombiner'],
                                  'loadBeamOffset': injectorParams['loadBeamOffset'],
                                  'rpLens1': surrogateParamsDict['rpLens1'],
                                  'L_Lens1': surrogateParamsDict['L_Lens'],
                                  'rpLens2': surrogateParamsDict['rpLens2'],
                                  'L_Lens2': surrogateParamsDict['L_Lens']})

    PTL = ParticleTracerLattice(v0Nominal=atomCharacteristic["nominalDesignSpeed"], latticeType='storageRing',
                                useSolenoidField=options['useSolenoidField'],
                                standard_tube_ODs=True,standard_mag_sizes=True)

    add_First_RaceTrack_Straight_Version1_3(PTL, raceTrackParams, options, whichOP_Ap='Injection')
    raceTrackParams.assert_All_Entries_Accessed_And_Reset_Counter()
    PTL.end_Lattice(constrain=False)
    return PTL


def make_ringParams_Dict(ringParams_tuple: tuple, injectorParams: dict, whichVersion: str) -> dict:
    """Take parameters values list and construct dictionary of variable ring parameters. For version1, all tunable
    variables (lens length, radius, etc) describe the ring only. Injector is optimized entirely independenly before"""

    assert whichVersion in ('1', '2', '3')

    ringParamsDict = {"LmCombiner": injectorParams["LmCombiner"],
                      "loadBeamOffset": injectorParams["loadBeamOffset"]}

    if whichVersion in ('1', '3'):
        assert len(ringParams_tuple) == 6
        for variableKey, value in zip(optimizerBounds_V1_3.keys(), ringParams_tuple):
            ringParamsDict[variableKey] = value
    else:
        assert len(ringParams_tuple) == 10
        for variableKey, value in zip(optimizerBounds_V2.keys(), ringParams_tuple):
            ringParamsDict[variableKey] = value
    return ringParamsDict


def make_injectorParams_Dict_Version_Any(injectorParams_tuple: tuple) -> dict:
    injectorParamsDict = dict(zip(injectorParamsBoundsAny.keys(), injectorParams_tuple))
    return injectorParamsDict


def assert_Combiners_Are_Same(PTL_Injector: ParticleTracerLattice, PTL_Ring: ParticleTracerLattice) -> None:
    """Combiner from injector and ring must have the same shared characteristics, as well as have the expected
    parameters"""

    assert PTL_Injector.combiner.outputOffset == PTL_Ring.combiner.outputOffset
    assert PTL_Injector.combiner.ang < 0 < PTL_Ring.combiner.ang


def _make_Ring_And_Injector_Params_Locked_Dicts(systemParams: tuple[tuple, tuple], whichVersion) \
        -> tuple[lockedDict, lockedDict]:
    """Given system parameters of (ring_params,injector_params) construct the lockedDicts for each"""
    ringParams_tuple, injectorParams_tuple = systemParams
    injectorParams = make_injectorParams_Dict_Version_Any(injectorParams_tuple)
    ringParams = make_ringParams_Dict(ringParams_tuple, injectorParams, whichVersion)
    # make locked dicts after so the item access counters start at 0
    return lockedDict(ringParams), lockedDict(injectorParams)


def _make_Ring_And_Injector(systemParams: tuple[tuple, tuple], whichVersion: str, options: Optional[dict]) \
        -> tuple[RingModel, InjectorModel]:
    """
    Make ParticleTracerLattice models of ring and injector system. Combiner must be the same, though in low
    field seeking configuration for ring, and high field seeking for injector

    :param systemParams: Non constant parameters to construct lens.
    :param whichVersion: which version of the ring/injector system to use
    :param options: dictionary of parameters that affect the full system such as wether to use magnet errors, solenoid
        field, etc.
    :return:
    """

    assert whichVersion in ('1', '2', '3')
    ringParams, injectorParams = _make_Ring_And_Injector_Params_Locked_Dicts(systemParams, whichVersion)
    PTL_Ring = make_Ring(ringParams, whichVersion, options)
    PTL_Injector = make_Injector_Version_Any(injectorParams, options=options)
    assert_Combiners_Are_Same(PTL_Injector, PTL_Ring)
    return PTL_Ring, PTL_Injector


def make_Ring_And_Injector(systemParams: tuple[tuple, tuple], version, options: dict = None) -> tuple[
    RingModel, InjectorModel]:
    """
    Make ParticleTraceLattice objects that represents storage ring and injector systems

    Version 1: in clockwise order starting at 0,0 and going in -x direction elements are [lens, combiner, lens, bender,
    lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation. This is a simple
    design

    Version 2: in clockwise order starting at 0,0 and going in -x direction elements are [lens,lens, combiner,lens,
    lens, bender, lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation. The
    idea of this version is that having double lenses flanking the combiner can improve mode matching

    Version 3: Same as Version 1, but instead with the bending sections split into to bending elements so pumping
    can be applied at apex
    """

    return _make_Ring_And_Injector(systemParams, version, options)
