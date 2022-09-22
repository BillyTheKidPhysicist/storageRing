"""
Module contains functions and parameters to produce ring model version 2. Also contains optimal parameters found so
far. Also contains functions for automatic vacuum system analysis, but this has not been implemented fully.

Sequence of elements:
- Lens
- Combiner
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
from math import pi

from constants import TUBE_WALL_THICKNESS
from constants import gas_masses
from helper_tools import meter_to_cm
from lattice_elements.lens_sim import HalbachLensSim
from lattice_models.lattice_model_functions import (add_drift_if_needed, check_and_format_params,
                                                    add_split_bend_with_lens, add_combiner_and_OP_ring,
                                                    initialize_ring_lattice, finish_ring_lattice)
from lattice_models.lattice_model_parameters import system_constants,DEFAULT_SYSTEM_OPTIONS
from lattice_models.utilities import LockedDict
from particle_tracer_lattice import ParticleTracerLattice
from vacuum_modeling.vacuum_analyzer import VacuumSystem, solve_vac_system
from vacuum_modeling.vacuum_constants import outgassing_rates, big_ion_pump_speed, small_ion_pump_speed

ring_param_bounds: LockedDict = LockedDict({
    'rp_lens3_4': (.005, .03),
    'rp_bend': (.005, .012),
    'rp_apex_lens': (.005, .012),
    'L_apex_lens': (.001, .1),
    'L_Lens1': (1e-6, .3),
    'L_Lens2': (.1, .7)
})

ring_params_optimal = {
    'rp_lens3_4': 0.012593597021671735,
    'rp_bend': 0.010115712864579277,
    'rp_apex_lens': 0.007415429587324836,
    'L_apex_lens': 0.04513305464223805,
    'L_Lens1': .1,  # length of lens before combiner
    'L_Lens2': 0.49472608069737817,  # length of lens after combiner
    "Lm_combiner": 0.17709193919623706,  # hard edge length of combiner
    "load_beam_offset": 0.009704452870607685,  # offset of incoming beam into combiner
}

injector_params_optimal = LockedDict({
    "L1": .298,  # length of first lens
    "rp1": 0.01824404317562657,  # bore radius of first lens
    "L2": 0.23788459956313238,  # length of first lens
    "rp2": 0.03,  # bore radius of second lens
    "Lm_combiner": 0.17709193919623706,  # hard edge length of combiner
    "load_beam_offset": 0.009704452870607685,  # offset of incoming beam into combiner
    "gap1": 0.10615316973237765,  # separation between source and first lens
    "gap2": 0.22492222955994753,  # separation between two lenses
    "gap3": 0.22148833301792942  ##separation between final lens and input to combiner
})

ring_constants = LockedDict({
    'rp_lens2': .04,
    'rp_lens1': .01
})

num_ring_params = 8


def make_ring_lattice(ring_params: dict, options: LockedDict = None) -> ParticleTracerLattice:
    if options is None: #IMPROVEMENT: IMPLEMENT THIS BETTER AND ADD EVERYWHERE
        options=DEFAULT_SYSTEM_OPTIONS
    ring_params = check_and_format_params(ring_params, num_ring_params)
    lattice = initialize_ring_lattice(options)
    rp_lens2 = ring_constants['rp_lens2']
    rp_lens1 = ring_constants['rp_lens1']
    rp_lens3_4 = ring_params['rp_lens3_4']
    rp_bend = ring_params['rp_bend']
    rp_apex_lens = ring_params['rp_apex_lens']
    L_apex_lens = ring_params['L_apex_lens']
    L_lens1 = ring_params['L_Lens1']
    L_lens2 = ring_params['L_Lens2']

    # ---------gap between bender output and first lens -------
    # add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens1, rp_bend)

    # ----lens before combiner
    if not HalbachLensSim.is_lens_too_short(L_lens1, rp_lens1):
        lattice.add_halbach_lens_sim(rp_lens1, L_lens1)

    # ----gap before combiner
    add_drift_if_needed(lattice, system_constants["pre_combiner_gap"], 'lens', 'combiner', rp_lens1,
                        system_constants['rp_combiner'])

    # ---combiner + OP magnet-----
    add_combiner_and_OP_ring(lattice, system_constants['rp_combiner'], ring_params['Lm_combiner'],
                             ring_params['load_beam_offset'], rp_lens2, options, 'Circulating')

    # ---lens after combiner---

    lattice.add_halbach_lens_sim(rp_lens2, L_lens2)

    # -----gap between lens and bender input

    add_drift_if_needed(lattice, system_constants["lensToBendGap"], 'lens', 'bender', rp_lens2, rp_bend)

    # -----first split bender---------

    add_split_bend_with_lens(lattice, rp_bend, rp_apex_lens, L_apex_lens)

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

    add_split_bend_with_lens(lattice, rp_bend, rp_apex_lens, L_apex_lens)

    lattice = finish_ring_lattice(lattice, ring_params, options)
    return lattice


def inside_diam(bore_radius):
    return 2 * (bore_radius - TUBE_WALL_THICKNESS)


def add_split_bend_vacuum(vac_sys, rp_bend, L_split_bend, S_small_pumps, q, exclude_end_pump=False):
    vac_sys.add_chamber(S=S_small_pumps, name='bend')
    vac_sys.add_tube(L_split_bend, inside_diam(rp_bend), q=q)
    vac_sys.add_chamber(S=S_small_pumps, name='bend')
    vac_sys.add_tube(L_split_bend, inside_diam(rp_bend), q=q)
    if not exclude_end_pump:
        vac_sys.add_chamber(S=S_small_pumps, name='bend')


def make_vacuum_model(ring_params: dict) -> VacuumSystem:
    raise NotImplementedError  # need to rework this
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
    # L_tube_after_combiner = system_constants["OP_mag_space"] + L_lens2
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
    vac_sys.add_tube(L_lens2, inside_diam(rp_lens2), q=q)
    add_split_bend_vacuum(vac_sys, rp_bend, L_split_bend, S_small_pump, q)
    vac_sys.add_tube(L_long_connecting_tube, inside_diam(rp_lens3_4), q=q)
    add_split_bend_vacuum(vac_sys, rp_bend, L_split_bend, S_small_pump, q, exclude_end_pump=True)

    solve_vac_system(vac_sys)
    return vac_sys
