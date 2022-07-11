from typing import Union, Optional

import numpy as np

from KevinBumperClass import add_Kevin_Bumper_Elements
from ParticleTracerClass import ParticleTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from constants import DEFAULT_ATOM_SPEED
from latticeElements.elements import CombinerHalbachLensSim, HalbachLensSim, HalbachBenderSimSegmented
from latticeModels_Parameters import system_constants, optimizerBounds_V1_3, \
    optimizerBounds_V2, LockedDict, atomCharacteristic, injectorParamsBoundsAny, constants_V1_3


class RingGeometryError(Exception):
    pass


class InjectorGeometryError(Exception):
    pass


lst_arr_tple = Union[list, np.ndarray, tuple]

h: float = 1e-5  # timestep, s. Assumed to be no larger than this
min_time_step_gap = 1.1 * h * DEFAULT_ATOM_SPEED * ParticleTracer.minTimeStepsPerElement
InjectorLattice = RingLattice = ParticleTracerLattice
DEFAULT_SYSTEM_OPTIONS = LockedDict({'use_mag_errors': False, 'combiner_seed': None, 'use_solenoid_field': False,
                                     'has_bumper': False, 'use_standard_tube_OD': False,
                                     'use_standard_mag_size': False})


def check_and_add_default_values(options: Optional[dict]) -> LockedDict:
    """Check that there are only the allowed keys in options, and any missing keys add them with the default
    value. If options is None, use the default dictionary"""

    if options is not None:
        num_valid_keys_in_dict = sum([1 for key in DEFAULT_SYSTEM_OPTIONS if key in options])
        assert num_valid_keys_in_dict == len(options)  # can't be unexpected stuff
        for key, val in DEFAULT_SYSTEM_OPTIONS.items():
            if key not in options:
                options[key] = val
        options = LockedDict(options)
    else:
        options = DEFAULT_SYSTEM_OPTIONS
    return options


def el_Fringe_Space(elementName: str, elementBoreRadius: float) -> float:
    """Return the gap between hard edge of element (magnetic material) and end of element model. This gap exists
    to allow the field values to fall to negligeable amounts"""

    assert elementBoreRadius > 0
    if elementName == 'none':
        return 0.0
    fringe_fracs = {"combiner": CombinerHalbachLensSim.outerFringeFrac,
                    "lens": HalbachLensSim.fringe_frac_outer,
                    "bender": HalbachBenderSimSegmented.fringe_frac_outer}
    return fringe_fracs[elementName] * elementBoreRadius


def round_Up_If_Below_Min_Time_Step_Gap(proposedLength: float) -> float:
    """Elements have a minimum length dictated by ParticleTracerClass for time stepping considerations. A  reasonable
    value for the time stepping is assumed. If wrong, an error will be thrown in ParticleTracerClass"""

    if proposedLength < min_time_step_gap:
        return min_time_step_gap
    else:
        return proposedLength


def add_drift_if_needed(lattice: ParticleTracerLattice, gap_length: float, el_before_name: str,
                        el_after_name: str, el_before_rp: float, el_after_rp: float, ap: float = None) -> None:
    """Sometimes the fringe field gap is enough to accomodate the minimum desired separation between elements.
    Otherwise a gap needs to be added. The drift will have a minimum length, so the total gap may be larger in some
    cases"""

    assert gap_length >= 0 and el_after_rp > 0 and el_before_rp > 0
    extra_space = gap_length - (
            el_Fringe_Space(el_before_name, el_before_rp) + el_Fringe_Space(el_after_name, el_after_rp))
    if extra_space > 0:
        ap = min([el_before_rp, el_after_rp]) if ap is None else ap
        lattice.add_drift(round_Up_If_Below_Min_Time_Step_Gap(extra_space), ap=ap)


def add_bend_version_1_2(lattice: ParticleTracerLattice, rp_bend: float) -> None:
    """Single bender element"""

    lattice.add_segmented_halbach_bender(system_constants['Lm'], rp_bend, None, system_constants['rbTarget'])


def add_bend_version_3(lattice: ParticleTracerLattice, rp_bend: float) -> None:
    """Two bender elements possibly separated by a drift region for pumping between end/beginning of benders. If fringe
    field region is long enough, no drift region is add"""

    lattice.add_segmented_halbach_bender(system_constants['Lm'], rp_bend, None, system_constants['rbTarget'])
    add_drift_if_needed(lattice, system_constants['bendApexGap'], 'bender', 'bender', rp_bend, rp_bend)
    lattice.add_segmented_halbach_bender(system_constants['Lm'], rp_bend, None, system_constants['rbTarget'])


def add_bender(lattice: ParticleTracerLattice, rp_bend: float, which_version: str) -> None:
    """Add bender section to storage ring. Racetrack design requires two "benders", though each bender may actually be
    composed of more than 1 bender element and/or other elements"""

    assert which_version in ('1', '2', '3')
    if which_version in ('1', '2'):  # single bender element
        add_bend_version_1_2(lattice, rp_bend)
    else:
        add_bend_version_3(lattice, rp_bend)


def add_combiner(lattice: ParticleTracerLattice, Lm_combiner: float, rp_combiner: float, load_beam_offset: float,
                 seed: Optional[int]):
    """add combiner element to lattice. Set random state for reproducible results if seed is not None, then reset the state
    """

    lattice.add_combiner_sim_lens(Lm_combiner, rp_combiner, load_beam_offset=load_beam_offset, layers=1, seed=seed)


def add_combiner_and_OP(lattice, rp_combiner, Lm_combiner, load_beam_offset, rp_lens_before, rp_lens_after,
                        options: Optional[dict], which_OP_ap: str = "Circulating") -> None:
    """Add gap for vacuum + combiner + gap for optical pumping. Elements before and after must be a lens. """

    # gap between combiner and previous lens
    add_drift_if_needed(lattice, system_constants["gap2Min"], 'lens', 'combiner', rp_lens_before, rp_combiner)

    # -------combiner-------

    add_combiner(lattice, Lm_combiner, rp_combiner, load_beam_offset, options['combiner_seed'])

    # ------gap 3--------- combiner-> lens, Optical Pumping (OP) region
    # there must be a drift here to account for the optical pumping aperture limit. It must also be at least as long
    # as optical pumping region. I am doing it like this because I don't have it coded up yet to include an aperture
    # without it being a new drift region
    OP_gap = system_constants["OP_MagWidth"] - (el_Fringe_Space('combiner', rp_combiner)
                                                + el_Fringe_Space('lens', rp_lens_after))
    OP_gap = round_Up_If_Below_Min_Time_Step_Gap(OP_gap)
    OP_gap = OP_gap if OP_gap > system_constants["OP_PumpingRegionLength"] else \
        system_constants["OP_PumpingRegionLength"]
    lattice.add_drift(OP_gap, ap=system_constants["OP_MagAp_" + which_OP_ap])


def add_first_racetrack_straight_version1_3(lattice: ParticleTracerLattice, ring_params: LockedDict,
                                            options: Optional[dict], which_OP_ap: str = "Circulating") -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
    elements are [lens,combiner,lens] with gaps as needed for vacuum Not actually straight because of combiner.
     """
    rp_lens2 = constants_V1_3['rpLens2']
    rp_lens1 = ring_params['rpLens1']

    # ----from bender output to combiner
    lattice.add_halbach_lens_sim(rp_lens1, ring_params['L_Lens1'])

    # ---combiner + OP magnet-----
    add_combiner_and_OP(lattice, system_constants['rp_combiner'], ring_params['Lm_combiner'],
                        ring_params['load_beam_offset'],
                        rp_lens1, rp_lens2, options, which_OP_ap=which_OP_ap)

    # ---from OP to bender input---

    lattice.add_halbach_lens_sim(rp_lens2, ring_params['L_Lens2'])


def add_first_raceTrack_straight_version2(lattice: ParticleTracerLattice, ring_params: LockedDict,
                                          options: Optional[dict]) -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
    elements are [lens,lens,combiner,len,lens] with gaps as needed for vacuum Not actually straight because of combiner.
     """

    # ----from bender output to combiner
    lattice.add_halbach_lens_sim(ring_params['rpLens1'], ring_params['L_Lens1'])
    lattice.add_halbach_lens_sim(ring_params['rpLens2'], ring_params['L_Lens2'])

    # ---combiner + OP magnet-----
    add_combiner_and_OP(lattice, ring_params['rp_combiner'], ring_params['Lm_combiner'],
                        ring_params['load_beam_offset'],
                        ring_params['rpLens2'], ring_params['rpLens3'], options['combiner_seed'])

    # ---from OP to bender input---
    lattice.add_halbach_lens_sim(ring_params['rpLens3'], ring_params['L_Lens3'])
    lattice.add_halbach_lens_sim(ring_params['rpLens4'], ring_params['L_Lens4'])


def add_first_racetrack_straight(lattice: ParticleTracerLattice, ring_params: LockedDict, which_version: str,
                                 options: Optional[dict]) -> None:
    """Starting from a bender output at 0,0 and going in -x direction to a bender input is the first "straight" section.
     Not actually straight because of combiner. Two lenses and a combiner, with supporting drift regions if
     neccesary for gap spacing"""

    # No need for gap before first lens
    assert which_version in ('1', '2', '3')

    if which_version in ('1', '3'):
        add_first_racetrack_straight_version1_3(lattice, ring_params, options)
    else:
        add_first_raceTrack_straight_version2(lattice, ring_params, options)


def add_second_racetrack_straight_version_any(lattice: ParticleTracerLattice, rp_lens3_4: float,
                                              rp_bend: float) -> None:
    """Going from output of bender in +x direction to input of next bender. Two lenses with supporting drift regions if
     neccesary for gap spacing"""

    # ---------gap 5-----  bender->lens

    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens3_4, rp_bend)

    # -------lens 3-----------
    lattice.add_halbach_lens_sim(rp_lens3_4, None, constrain=True)

    # ---------gap 6------ lens -> lens
    add_drift_if_needed(lattice, system_constants["observationGap"], 'lens', 'lens', rp_lens3_4, rp_lens3_4)

    # ---------lens 4-------
    lattice.add_halbach_lens_sim(rp_lens3_4, None, constrain=True)

    # ------gap 7------ lens-> bender

    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens3_4, rp_bend)


def make_ring(ring_params: LockedDict, which_version: str, options: Optional[dict]) -> RingLattice:
    """Make ParticleTraceLattice object that represents storage ring. This does not include the injector components.
    Several versions available

    Version 1: in clockwise order starting at 0,0 and going in -x direction elements are [lens, combiner, lens, bender,
    lens,lens,bender] with drift regions as neccesary to fill in gaps for pumping and obsercation

    Version 3: Same as Version 1, but instead with the bending sections split into to bending elements so pumping
    can be applied at apex
    """
    options = check_and_add_default_values(options)
    assert which_version in ('1', '2', '3')

    lattice = ParticleTracerLattice(speed_nominal=atomCharacteristic["nominalDesignSpeed"], lattice_type='storage_ring',
                                    use_mag_errors=options['use_mag_errors'],
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'],
                                    use_standard_mag_size=options['use_standard_mag_size'])

    # ------starting at gap 1 through lenses and gaps and combiner to gap4

    add_first_racetrack_straight(lattice, ring_params, which_version, options)

    rp_bend = ring_params['rp_bend']

    # -------bender 1------
    add_bender(lattice, rp_bend, which_version)

    # -----starting at gap 5r through gap 7r-----

    add_second_racetrack_straight_version_any(lattice, ring_params['rp_lens3_4'], rp_bend)

    # ------bender 2--------
    add_bender(lattice, rp_bend, which_version)

    # ----done---
    lattice.end_lattice(constrain=True)

    ring_params.assert_All_Entries_Accessed_And_Reset_Counter()

    return lattice


def make_injector_version_any(injector_params: LockedDict, options: dict = None) -> InjectorLattice:
    """Make ParticleTraceLattice object that represents injector. Injector is a double lens design. """

    options = check_and_add_default_values(options)
    gap1 = injector_params["gap1"]
    gap2 = injector_params["gap2"]
    gap3 = injector_params["gap3"]
    gap1 -= el_Fringe_Space('lens', injector_params["rp1"])
    gap2 -= el_Fringe_Space('lens', injector_params["rp1"]) + el_Fringe_Space('lens', injector_params["rp2"])
    if gap2 < system_constants["lens1ToLens2_Inject_Gap"]:
        raise InjectorGeometryError
    if gap1 < system_constants["sourceToLens1_Inject_Gap"]:
        raise InjectorGeometryError
    lattice = ParticleTracerLattice(atomCharacteristic["nominalDesignSpeed"], lattice_type='injector',
                                    use_mag_errors=options['use_mag_errors'],
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'],
                                    use_standard_mag_size=options['use_standard_mag_size'])
    if options['has_bumper']:
        add_Kevin_Bumper_Elements(lattice)

    # -----gap between source and first lens-----

    add_drift_if_needed(lattice, gap1, 'none', 'lens', np.inf, injector_params["rp1"])  # hacky

    # ---- first lens------

    lattice.add_halbach_lens_sim(injector_params["rp1"], injector_params["L1"])

    # -----gap with valve--------------

    gap_valve = system_constants["lens1ToLens2_Valve_Length"]
    gap2 = gap2 - gap_valve
    if gap2 < 0:  # this is approximately true. I am ignoring that there is space in the fringe fields
        raise InjectorGeometryError
    lattice.add_drift(gap2, ap=injector_params["rp1"])
    lattice.add_drift(gap_valve, ap=system_constants["lens1ToLens2_Valve_Ap"],
                      outer_half_width=system_constants["lens1ToLens2_Inject_Valve_OD"] / 2)

    # ---------------------

    lattice.add_halbach_lens_sim(injector_params["rp2"], injector_params["L2"])

    lattice.add_drift(gap3, ap=injector_params["rp2"])

    add_combiner(lattice, injector_params["Lm_combiner"], system_constants["rp_combiner"],
                 injector_params["load_beam_offset"],
                 options['combiner_seed'])

    lattice.end_lattice(constrain=False)

    injector_params.assert_All_Entries_Accessed_And_Reset_Counter()

    return lattice


def make_ring_surrogate_for_injection_version_1(injector_params: LockedDict,
                                                surrogate_params_dict: LockedDict, options: dict = None) -> RingLattice:
    """Surrogate model of storage to aid in realism of independent injector optimizing. Benders are exluded. Model
    serves to represent geometric constraints between injector and ring, and to test that if particle that travel
    into combiner can make it to the next element. Since injector is optimized independent of ring, the parameters
    for the ring surrogate are typically educated guesses"""

    options = check_and_add_default_values(options)

    race_track_params = LockedDict({'Lm_combiner': injector_params['Lm_combiner'],
                                    'load_beam_offset': injector_params['load_beam_offset'],
                                    'rpLens1': surrogate_params_dict['rpLens1'],
                                    'L_Lens1': surrogate_params_dict['L_Lens'],
                                    'L_Lens2': surrogate_params_dict['L_Lens']})

    lattice = ParticleTracerLattice(speed_nominal=atomCharacteristic["nominalDesignSpeed"], lattice_type='storage_ring',
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'],
                                    use_standard_mag_size=options['use_standard_mag_size'])

    add_first_racetrack_straight_version1_3(lattice, race_track_params, options, which_OP_ap='Injection')
    race_track_params.assert_All_Entries_Accessed_And_Reset_Counter()
    lattice.end_lattice(constrain=False)
    return lattice


def make_ring_params_dict(ring_params_tuple: tuple, injector_params: dict, which_version: str) -> dict:
    """Take parameters values list and construct dictionary of variable ring parameters. For version1, all tunable
    variables (lens length, radius, etc) describe the ring only. Injector is optimized entirely independenly before"""

    assert which_version in ('1', '2', '3')

    ring_params_dict = {"Lm_combiner": injector_params["Lm_combiner"],
                        "load_beam_offset": injector_params["load_beam_offset"]}

    if which_version in ('1', '3'):
        assert len(ring_params_tuple) == 5
        for variableKey, value in zip(optimizerBounds_V1_3.keys(), ring_params_tuple):
            ring_params_dict[variableKey] = value
    else:
        assert len(ring_params_tuple) == 10
        for variableKey, value in zip(optimizerBounds_V2.keys(), ring_params_tuple):
            ring_params_dict[variableKey] = value
    return ring_params_dict


def make_injector_params_dict_version_any(injector_params_tuple: tuple) -> dict:
    injectorParamsDict = dict(zip(injectorParamsBoundsAny.keys(), injector_params_tuple))
    return injectorParamsDict


def assert_combiners_are_same(lattice_injector: ParticleTracerLattice, lattice_ring: ParticleTracerLattice) -> None:
    """Combiner from injector and ring must have the same shared characteristics, as well as have the expected
    parameters"""

    assert lattice_injector.combiner.output_offset == lattice_ring.combiner.output_offset
    assert lattice_injector.combiner.ang < 0 < lattice_ring.combiner.ang


def _make_ring_and_injector_params_locked_dicts(system_params: tuple[tuple, tuple], which_version) \
        -> tuple[LockedDict, LockedDict]:
    """Given system parameters of (ring_params,injector_params) construct the lockedDicts for each"""
    ring_params_tuple, injector_params_tuple = system_params
    injector_params = make_injector_params_dict_version_any(injector_params_tuple)
    ring_params = make_ring_params_dict(ring_params_tuple, injector_params, which_version)
    # make locked dicts after so the item access counters start at 0
    return LockedDict(ring_params), LockedDict(injector_params)


def _make_ring_and_injector(system_params: tuple[tuple, tuple], which_version: str, options: Optional[dict]) \
        -> tuple[RingLattice, InjectorLattice]:
    """
    Make ParticleTracerLattice models of ring and injector system. Combiner must be the same, though in low
    field seeking configuration for ring, and high field seeking for injector

    :param system_params: Non constant parameters to construct lens.
    :param which_version: which version of the ring/injector system to use
    :param options: dictionary of parameters that affect the full system such as wether to use magnet errors, solenoid
        field, etc.
    :return:
    """

    assert which_version in ('1', '2', '3')
    ring_params, injector_params = _make_ring_and_injector_params_locked_dicts(system_params, which_version)
    lattice_ring = make_ring(ring_params, which_version, options)
    lattice_injector = make_injector_version_any(injector_params, options=options)
    assert_combiners_are_same(lattice_injector, lattice_ring)
    return lattice_ring, lattice_injector


def make_ring_and_injector(system_params: tuple[tuple, tuple], version, options: dict = None) \
        -> tuple[RingLattice, InjectorLattice]:
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

    return _make_ring_and_injector(system_params, version, options)
