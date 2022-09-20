"""
Module contains functions for use in building injector and ring models.
"""
from typing import Optional

from constants import DEFAULT_ATOM_SPEED
from helper_tools import random_num_for_seeding
from lattice_elements.elements import CombinerLensSim, HalbachLensSim, BenderSim
from lattice_models.lattice_model_parameters import system_constants, DEFAULT_SYSTEM_OPTIONS, atom_characteristics
from lattice_models.utilities import LockedDict
from particle_tracer import ParticleTracer
from particle_tracer_lattice import ParticleTracerLattice

h: float = 1e-5  # timestep, s. Assumed to be no larger than this
min_time_step_gap = 1.1 * h * DEFAULT_ATOM_SPEED * ParticleTracer.minTimeStepsPerElement


def set_cominer_seed_if_unset(options: LockedDict):
    """If combiner seed is None, set to an integer value. This ensures both combiners have the same random behaviour"""
    if options['combiner_seed'] is None:
        options.super_special_change_item('combiner_seed', random_num_for_seeding())


def check_and_add_default_values(options: Optional[dict]) -> LockedDict:
    """Check that there are only the allowed keys in options, and any missing keys add them with the default
    value. If options is None, use the default dictionary"""

    if options is not None:
        assert all(key in DEFAULT_SYSTEM_OPTIONS for key in options.keys())
        for key, val in DEFAULT_SYSTEM_OPTIONS.items():
            if key not in options:
                options[key] = val
        options = LockedDict(options)
    else:
        options = DEFAULT_SYSTEM_OPTIONS
    set_cominer_seed_if_unset(options)
    return options


def add_drift_if_needed(lattice: ParticleTracerLattice, gap_length: float, el_before_name: str,
                        el_after_name: str, el_before_rp: float, el_after_rp: float, ap: float = None) -> None:
    """Sometimes the fringe field gap is enough to accomodate the minimum desired separation between elements.
    Otherwise a gap needs to be added. The drift will have a minimum length, so the total gap may be larger in some
    cases"""

    assert gap_length >= 0 and el_after_rp > 0 and el_before_rp > 0
    extra_space = gap_length - (
            el_fringe_space(el_before_name, el_before_rp) + el_fringe_space(el_after_name, el_after_rp))
    if extra_space > 0:
        ap = min([el_before_rp, el_after_rp]) if ap is None else ap
        lattice.add_drift(round_up_if_below_min_time_step_gap(extra_space), ap=ap)


def el_fringe_space(elementName: str, elementBoreRadius: float) -> float:
    """Return the gap between hard edge of element (magnetic material) and end of element model. This gap exists
    to allow the field values to fall to negligeable amounts"""

    assert elementBoreRadius > 0
    if elementName == 'none':
        return 0.0
    fringe_fracs = {"combiner": CombinerLensSim.fringe_frac_outer,
                    "lens": HalbachLensSim.fringe_frac_outer,
                    "bender": BenderSim.fringe_frac_outer}
    return fringe_fracs[elementName] * elementBoreRadius


def check_and_format_params(params: dict, expected_num_params: int) -> LockedDict:
    """Check that params is the expected length, and cast to LockedDict."""
    params = LockedDict(params)
    assert len(params) == expected_num_params
    return params


def initialize_ring_lattice(options: dict) -> ParticleTracerLattice:
    lattice = ParticleTracerLattice(design_speed=atom_characteristics["nominalDesignSpeed"],
                                    include_mag_errors=options['include_mag_errors'],
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'],
                                    use_long_range_fields=options['include_mag_cross_talk_in_ring'],
                                    include_misalignments=options['include_misalignments'])
    return lattice


def finish_ring_lattice(lattice: ParticleTracerLattice, ring_params: LockedDict,
                        options: LockedDict) -> ParticleTracerLattice:
    ring_params.assert_all_entries_accesed()
    lattice.end_lattice(constrain=True, build_field_helpers=options['build_field_helpers'])
    return lattice


def round_up_if_below_min_time_step_gap(proposedLength: float) -> float:
    """Elements have a minimum length dictated by particle_tracer for time stepping considerations. A  reasonable
    value for the time stepping is assumed. If wrong, an error will be thrown in particle_tracer"""

    if proposedLength < min_time_step_gap:
        return min_time_step_gap
    else:
        return proposedLength


def add_split_bend(lattice: ParticleTracerLattice, rp_bend):
    lattice.add_segmented_halbach_bender(system_constants['Lm'], rp_bend, None, system_constants['rbTarget'])
    add_drift_if_needed(lattice, system_constants['bendApexGap'], 'bender', 'bender', rp_bend, rp_bend)
    lattice.add_segmented_halbach_bender(system_constants['Lm'], rp_bend, None, system_constants['rbTarget'])


def add_split_bend_with_lens(lattice: ParticleTracerLattice, rp_bend, rp_lens, L_lens):
    lattice.add_segmented_halbach_bender(system_constants['Lm'], rp_bend, None, system_constants['rbTarget'])
    add_drift_if_needed(lattice, system_constants['bendApexGap'], 'bender', 'lens', rp_bend, rp_lens)
    lattice.add_halbach_lens_sim(rp_lens, L_lens)
    add_drift_if_needed(lattice, system_constants['bendApexGap'], 'lens', 'bender', rp_lens, rp_bend)
    lattice.add_segmented_halbach_bender(system_constants['Lm'], rp_bend, None, system_constants['rbTarget'])


def add_combiner_and_OP_ring(lattice, rp_combiner, Lm_combiner, load_beam_offset, rp_lens_after,
                             options: Optional[dict], which_OP_ap: str) -> None:
    """Add combiner + gap for optical pumping. Element after must be a lens. """

    # -------combiner-------

    lattice.add_combiner_sim_lens(Lm_combiner, rp_combiner, load_beam_offset=load_beam_offset, layers=1,
                                  seed=options['combiner_seed'], atom_state='LOW_SEEK')

    # ------gap 3--------- combiner-> lens, Optical Pumping (OP) region
    # there must be a drift here to account for the optical pumping aperture limit. It must also be at least as long
    # as optical pumping region. I am doing it like this because I don't have it coded up yet to include an aperture
    # without it being a new drift region
    OP_gap = system_constants["OP_mag_space"] - (el_fringe_space('combiner', rp_combiner)
                                                 + el_fringe_space('lens', rp_lens_after))
    OP_gap = round_up_if_below_min_time_step_gap(OP_gap)
    # this is to enforce atoms clipping on op magnet. Not ideal solution
    OP_gap = OP_gap if OP_gap > system_constants["OP_PumpingRegionLength"] else \
        system_constants["OP_PumpingRegionLength"]
    lattice.add_drift(OP_gap, ap=system_constants["OP_MagAp_" + which_OP_ap])
