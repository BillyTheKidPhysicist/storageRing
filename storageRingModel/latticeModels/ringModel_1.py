from ParticleTracerLatticeClass import ParticleTracerLattice
from latticeModels.latticeModelFunctions import check_and_add_default_values, add_drift_if_needed, add_split_bend, \
    add_combiner_and_OP
from latticeModels.latticeModelParameters import atom_characteristics, system_constants
from latticeModels.latticeModelUtilities import LockedDict
from latticeModels.ringModelSurrogate_1 import surrogate_params

ring_param_bounds: LockedDict = LockedDict({
    'rp_lens3_4': (.005, .03),
    'rp_lens1': (.005, surrogate_params['rp_lens1']),
    # 'rpLens2': (.02, .04),
    'rp_bend': (.005, .01),
    'L_Lens1': (.1, .6),
    'L_Lens2': (.1, .7)
})

ring_params_optimal = {
    'rp_lens3_4': 0.02102425839849725,
    'rp_lens1': .01,
    'rp_lens2': 0.038773002120334694,
    'rp_bend': 0.00759624174653381,
    'L_Lens1': 0.441164241347491,
    'L_Lens2': 0.46839105549798354,
    "Lm_combiner": 0.17047182734978528,
    "load_beam_offset": 0.007187101087953732
}


def make_ring_lattice(ring_params: dict, options: dict=None) -> ParticleTracerLattice:
    ring_params = LockedDict(ring_params)
    options = check_and_add_default_values(options)

    lattice = ParticleTracerLattice(speed_nominal=atom_characteristics["nominalDesignSpeed"],
                                    lattice_type='storage_ring',
                                    use_mag_errors=options['use_mag_errors'],
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'],
                                    use_standard_mag_size=options['use_standard_mag_size'])
    rp_lens2 = ring_params['rp_lens2']
    rp_lens1 = ring_params['rp_lens1']
    rp_lens3_4 = ring_params['rp_lens3_4']
    rp_bend = ring_params['rp_bend']

    # ---------gap between bender output and first lens -------
    # add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens1, rp_bend)

    # ----lens before combiner
    lattice.add_halbach_lens_sim(rp_lens1, ring_params['L_Lens1'])

    # ---combiner + OP magnet-----
    add_combiner_and_OP(lattice, system_constants['rp_combiner'], ring_params['Lm_combiner'],
                        ring_params['load_beam_offset'], rp_lens1, rp_lens2, options)

    # ---lens after combiner---

    lattice.add_halbach_lens_sim(rp_lens2, ring_params['L_Lens2'])

    # -----gap between lens and bender input

    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens2, rp_bend)

    # -----first split bender---------

    add_split_bend(lattice, rp_bend)

    # ------gap between bender output and lens-----
    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens3_4, rp_bend)

    # -------lens after bender-----------
    lattice.add_halbach_lens_sim(rp_lens3_4, None, constrain=True)

    # ---------observation gap----------
    add_drift_if_needed(lattice, system_constants["observationGap"], 'lens', 'lens', rp_lens3_4, rp_lens3_4)

    # -------lens before bender-------
    lattice.add_halbach_lens_sim(rp_lens3_4, None, constrain=True)
    # ---------gap between lens and bender input-------
    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens3_4, rp_bend)

    # --------last split bender--------------

    add_split_bend(lattice, rp_bend)

    ring_params.assert_all_entries_accesed()
    lattice.end_lattice(constrain=True)
    return lattice
