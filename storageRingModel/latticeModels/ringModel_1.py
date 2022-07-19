from math import pi

from ParticleTracerLatticeClass import ParticleTracerLattice
from constants import TUBE_WALL_THICKNESS
from constants import gas_masses
from helperTools import meter_to_cm
from latticeModels.latticeModelFunctions import check_and_add_default_values, add_drift_if_needed, add_split_bend, \
    add_combiner_and_OP
from latticeModels.latticeModelParameters import atom_characteristics, system_constants
from latticeModels.latticeModelUtilities import LockedDict
from latticeModels.ringModelSurrogate_1 import surrogate_params
from vacuumanalysis.vacuumanalyzer import VacuumSystem, solve_vac_system
from vacuumanalysis.vacuumconstants import outgassing_rates, big_ion_pump_speed, small_ion_pump_speed

ring_param_bounds: LockedDict = LockedDict({
    'rp_lens3_4': (.005, .03),
    'rp_lens1': (.005, surrogate_params['rp_lens1']),
    'rp_bend': (.005, .012),
    'L_Lens1': (.1, .6),
    'L_Lens2': (.1, .7)
})

ring_params_optimal = {
    'rp_lens3_4': 0.01868819246427924,
    'rp_lens1': .01,
    'rp_bend': 0.00943026879837814,
    'L_Lens1': 0.25977731202276544,
    'L_Lens2': 0.48987106310146267,
    "Lm_combiner": 0.1775396197959241,  # hard edge length of combiner
    "load_beam_offset": 0.00573236996928707,  # offset of incoming beam into combiner
}

ring_constants = LockedDict({
    'rp_lens2': .04
})


def make_ring_lattice(ring_params: dict, options: dict = None) -> ParticleTracerLattice:
    ring_params = LockedDict(ring_params)
    options = check_and_add_default_values(options)

    lattice = ParticleTracerLattice(speed_nominal=atom_characteristics["nominalDesignSpeed"],
                                    lattice_type='storage_ring',
                                    use_mag_errors=options['use_mag_errors'],
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'],
                                    use_standard_mag_size=options['use_standard_mag_size'])
    rp_lens2 = ring_constants['rp_lens2']
    rp_lens1 = ring_params['rp_lens1']
    rp_lens3_4 = ring_params['rp_lens3_4']
    rp_bend = ring_params['rp_bend']
    L_lens1 = ring_params['L_Lens1']
    L_lens2 = ring_params['L_Lens2']

    # ---------gap between bender output and first lens -------
    # add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens1, rp_bend)

    # ----lens before combiner
    lattice.add_halbach_lens_sim(rp_lens1, L_lens1)

    # ---combiner + OP magnet-----
    add_combiner_and_OP(lattice, system_constants['rp_combiner'], ring_params['Lm_combiner'],
                        ring_params['load_beam_offset'], rp_lens1, rp_lens2, options)

    # ---lens after combiner---

    lattice.add_halbach_lens_sim(rp_lens2, L_lens2)

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


def inside_diam(bore_radius):
    return 2 * (bore_radius - TUBE_WALL_THICKNESS)


def add_split_bend_vacuum(vac_sys, rp_bend, L_split_bend, S_small_pumps, q):
    vac_sys.add_chamber(S=S_small_pumps, name='bend')
    vac_sys.add_tube(L_split_bend, inside_diam(rp_bend), q=q)
    vac_sys.add_chamber(S=S_small_pumps, name='bend')
    vac_sys.add_tube(L_split_bend, inside_diam(rp_bend), q=q)
    vac_sys.add_chamber(S=S_small_pumps, name='bend')


def make_vacuum_model(ring_params: dict) -> VacuumSystem:
    # todo: I can make this more accurate accounting for chamber width. It's not a single point
    gas_species = 'H2'
    ring_params = LockedDict(ring_params)
    rp_lens1 = meter_to_cm(ring_params['rp_lens1'])
    L_lens1 = meter_to_cm(ring_params['L_Lens1'])
    rp_lens2 = meter_to_cm(ring_constants['rp_lens2'])
    rp_lens3_4 = meter_to_cm(ring_params['rp_lens3_4'])
    rp_bend = meter_to_cm(ring_params['rp_bend'])
    L_lens2 = meter_to_cm(ring_params['L_Lens2'])
    rp_combiner = meter_to_cm(system_constants['rp_combiner'])
    Lm_combiner = meter_to_cm(ring_params['Lm_combiner'])
    rb = meter_to_cm(system_constants['rbTarget'])
    L_split_bend = pi * rb / 2.0
    L_tube_after_combiner = system_constants["OP_mag_space"] + L_lens2
    L_long_connecting_tube = L_lens1 + L_lens2 + Lm_combiner + system_constants["OP_mag_space"]

    q = outgassing_rates[gas_species]

    # pumps can be brough up right to the fact of the tube it appears, so I will not reduce the pumping speed
    # because of some geometric concerns

    S_small_pump = small_ion_pump_speed[gas_species]
    S_big_pump = big_ion_pump_speed[gas_species]

    vac_sys = VacuumSystem(is_circular=True, gas_mass_Daltons=gas_masses[gas_species])
    vac_sys.add_tube(L_lens1, inside_diam(rp_lens1), q=q)
    vac_sys.add_chamber(S=S_big_pump, name='combiner')
    vac_sys.add_tube(Lm_combiner, inside_diam(rp_combiner), q=q)
    vac_sys.add_tube(L_tube_after_combiner, inside_diam(rp_lens2), q=q)
    add_split_bend_vacuum(vac_sys, rp_bend, L_split_bend, S_small_pump, q)
    vac_sys.add_tube(L_long_connecting_tube, inside_diam(rp_lens3_4), q=q)
    add_split_bend_vacuum(vac_sys, rp_bend, L_split_bend, S_small_pump, q)

    solve_vac_system(vac_sys)
    return vac_sys
