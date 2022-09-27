"""
Module contains functions and parameters to produce ring model version 3. Also contains optimal parameters found so
far. Also contains functions for automatic vacuum system analysis, but this has not been implemented fully.

Sequence of elements:
- Lens
- Lens
- Combiner
- Lens
- Lens
- Bender
- Lens
- Bender
- Lens
- Lens
- Bender
- Lens
- Bender

With drift regions in between as needed for spacing requirements
"""

from lattice_models.lattice_model_functions import (add_drift_if_needed, check_and_format_params,
                                                    add_split_bend_with_lens, add_combiner_and_OP_ring,
                                                    initialize_ring_lattice, add_lens_if_long_enough)
from lattice_models.lattice_model_parameters import system_constants
from lattice_models.utilities import LockedDict
from particle_tracer_lattice import ParticleTracerLattice

ring_param_bounds: LockedDict = LockedDict({
    'rp_lens1': (.005, .03),
    'rp_lens2': (.005, .03),
    'rp_lens3': (.005, .03),
    'rp_lens4': (.005, .03),
    'rp_lens5_8': (.005, .03),
    'rp_lens6_7': (.005, .03),
    'rp_bend': (.005, .012),
    'rp_apex_lens': (.005, .012),
    'L_apex_lens': (.001, .2),
    'L_Lens1': (0, .5),
    'L_Lens2': (0, .5),
    'L_Lens3': (.1, .5),
    'L_Lens4': (.1, .5)
})

num_ring_params = 15


def make_ring_lattice(ring_params: dict, options: dict = None) -> ParticleTracerLattice:
    ring_params = check_and_format_params(ring_params, num_ring_params)
    lattice = initialize_ring_lattice(options)
    rp_lens1 = ring_params['rp_lens1']
    rp_lens2 = ring_params['rp_lens2']
    rp_lens3 = ring_params['rp_lens3']
    rp_lens4 = ring_params['rp_lens4']
    rp_lens5_8 = ring_params['rp_lens5_8']
    rp_lens6_7 = ring_params['rp_lens6_7']
    rp_bend = ring_params['rp_bend']
    rp_apex_lens = ring_params['rp_apex_lens']
    L_apex_lens = ring_params['L_apex_lens']
    L_lens1 = ring_params['L_Lens1']
    L_lens2 = ring_params['L_Lens2']
    L_lens3 = ring_params['L_Lens3']
    L_lens4 = ring_params['L_Lens4']

    # ---------gap between bender output and first lens -------
    # add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens1, rp_bend)

    # ----two lensese before combiner
    add_lens_if_long_enough(lattice, rp_lens1, L_lens1)
    add_lens_if_long_enough(lattice, rp_lens2, L_lens2)

    el_before_name = 'bender' if len(lattice) == 0 else 'lens'
    el_before_rp = rp_bend if len(lattice) == 0 else lattice[-1].rp

    # ----gap before combiner
    add_drift_if_needed(lattice, system_constants["pre_combiner_gap"], el_before_name, 'combiner', el_before_rp,
                        system_constants['rp_combiner'])

    # ---combiner + OP magnet-----
    add_combiner_and_OP_ring(lattice, system_constants['rp_combiner'], ring_params['Lm_combiner'],
                             ring_params['load_beam_offset'], rp_lens2, options, 'Circulating')

    # ---two lenses after combiner---

    lattice.add_halbach_lens_sim(rp_lens3, L_lens3)
    lattice.add_halbach_lens_sim(rp_lens4, L_lens4)

    # -----gap between lens and bender input

    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens4, rp_bend)

    # -----first split bender---------

    add_split_bend_with_lens(lattice, rp_bend, rp_apex_lens, L_apex_lens)

    # ------gap between bender output and lens-----
    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens5_8, rp_bend)

    # -------lens after bender-----------
    lattice.add_halbach_lens_sim(rp_lens5_8, None, constrain=True)
    lattice.add_halbach_lens_sim(rp_lens6_7, None, constrain=True)

    # ---------observation gap----------
    add_drift_if_needed(lattice, system_constants["observationGap"], 'lens', 'lens', rp_lens6_7, rp_lens6_7)

    # -------lens before bender-------
    lattice.add_halbach_lens_sim(rp_lens6_7, None, constrain=True)
    lattice.add_halbach_lens_sim(rp_lens5_8, None, constrain=True)
    # ---------gap between lens and bender input-------
    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens5_8, rp_bend)

    # --------last split bender--------------

    add_split_bend_with_lens(lattice, rp_bend, rp_apex_lens, L_apex_lens)

    ring_params.assert_all_entries_accesed()
    lattice.end_lattice(constrain=True, build_field_helpers=options['build_field_helpers'])
    return lattice
