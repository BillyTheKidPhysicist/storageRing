"""
Model of ring with mode match lenses at the apex of the benders, and mode matching lenses into and out of the
combiner
"""

from lattice_models.lattice_model_functions import add_drift_if_needed, \
    add_split_bend_with_lens, add_combiner_and_OP, initialize_ring_lattice
from lattice_models.lattice_model_parameters import system_constants
from lattice_models.utilities import LockedDict
from particle_tracer_lattice import ParticleTracerLattice

ring_param_bounds: LockedDict = LockedDict({
    'rp_lens1': (.005, .04),
    'rp_lens2': (.005, .04),
    'rp_lens3': (.005, .04),
    'rp_lens4': (.005, .04),
    'rp_lens5_6': (.005, .03),
    'rp_bend': (.005, .012),
    'rp_apex_lens': (.005, .012),
    'L_apex_lens': (.001, .2),
    'L_Lens1': (.1, .6),
    'L_Lens2': (.1, .6),
    'L_Lens3': (.1, .6),
    'L_Lens4': (.1, .6)
})

num_ring_params = 14


def make_ring_lattice(ring_params: dict, options: dict = None) -> ParticleTracerLattice:
    lattice, ring_params, options = initialize_ring_lattice(ring_params, options, num_ring_params)
    rp_lens1 = ring_params['rp_lens1']
    rp_lens2 = ring_params['rp_lens2']
    rp_lens3 = ring_params['rp_lens3']
    rp_lens4 = ring_params['rp_lens4']
    rp_lens5_6 = ring_params['rp_lens5_6']
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
    lattice.add_halbach_lens_sim(rp_lens1, L_lens1)
    lattice.add_halbach_lens_sim(rp_lens2, L_lens2)

    # ----gap before combiner
    add_drift_if_needed(lattice, system_constants["pre_combiner_gap"], 'lens', 'combiner', rp_lens1,
                        system_constants['rp_combiner'])

    # ---combiner + OP magnet-----
    add_combiner_and_OP(lattice, system_constants['rp_combiner'], ring_params['Lm_combiner'],
                        ring_params['load_beam_offset'], rp_lens2, options, 'Circulating')

    # ---two lenses after combiner---

    lattice.add_halbach_lens_sim(rp_lens3, L_lens3)
    lattice.add_halbach_lens_sim(rp_lens4, L_lens4)

    # -----gap between lens and bender input

    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens2, rp_bend)

    # -----first split bender---------

    add_split_bend_with_lens(lattice, rp_bend, rp_apex_lens, L_apex_lens)

    # ------gap between bender output and lens-----
    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens5_6, rp_bend)

    # -------lens after bender-----------
    lattice.add_halbach_lens_sim(rp_lens5_6, None, constrain=True)

    # ---------observation gap----------
    add_drift_if_needed(lattice, system_constants["observationGap"], 'lens', 'lens', rp_lens5_6, rp_lens5_6)

    # -------lens before bender-------
    lattice.add_halbach_lens_sim(rp_lens5_6, None, constrain=True)
    # ---------gap between lens and bender input-------
    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens5_6, rp_bend)

    # --------last split bender--------------

    add_split_bend_with_lens(lattice, rp_bend, rp_apex_lens, L_apex_lens)

    ring_params.assert_all_entries_accesed()
    lattice.end_lattice(constrain=True, build_field_helpers=options['build_field_helpers'])
    return lattice
