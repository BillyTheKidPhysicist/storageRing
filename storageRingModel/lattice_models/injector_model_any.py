"""
Module contains functions and parameters to produce an injector model. There is only 1 injector model currently. Also
contains optimal parameters found so far. Also contains functions for automatic vacuum system analysis, but this has
not been implemented fully.

Sequence of elements:

- Optional if bumper is included
    - Shifted lens
    - Drift
    - lens
    - Drift
- Drift
- Lens
- Drift
- Lens
- Drift
- Combiner


"""

import numpy as np

import kevin_bumper as bumper
from constants import TUBE_WALL_THICKNESS
from constants import gas_masses
from helper_tools import inch_to_meter
from helper_tools import meter_to_cm
from kevin_bumper import add_Kevin_Bumper_Elements
from lattice_models.lattice_model_functions import el_fringe_space, add_drift_if_needed
from lattice_models.lattice_model_parameters import system_constants, atom_characteristics, flange_OD
from lattice_models.utilities import LockedDict, InjectorGeometryError
from particle_tracer_lattice import ParticleTracerLattice
from vacuum_modeling.vacuum_analyzer import VacuumSystem, solve_vac_system
from vacuum_modeling.vacuum_constants import (turbo_pump_speed, diffusion_pump_speed, big_chamber_pressure,
                                              big_ion_pump_speed)

injector_constants = LockedDict({
    "pre_combiner_gap": inch_to_meter(3.5),  # from lens to combiner
    "OP_mag_space": .065 + 2 * .035,  # account for fringe fields with .02
    "OP_MagAp_Injection": .022 / 2.0,
    "OP_MagAp_Circulating": .035 / 2.0,
    "OP_PumpingRegionLength": .01,  # distance for effective optical pumping
    # "bendTubeODMax": inch_to_meter(3 / 4) ,
    "sourceToLens1_Inject_Gap": .05,  # gap between source and first lens. Shouldn't have first lens on top of source
    "lens1ToLens2_Inject_Gap": inch_to_meter(5.9),  # pumps and valve
    "lens1ToLens2_Valve_Ap": inch_to_meter(.75),  # aperture (ID/2) for valve #2 3/4
    "lens1ToLens2_Valve_Length": inch_to_meter(3.25),  # includes flanges and screws
    "lens1ToLens2_Inject_Valve_OD": flange_OD['2-3/4']  # outside diameter of valve
})

injector_param_bounds: LockedDict = LockedDict({
    "L1": (.05, .4),  # length of first lens
    "rp1": (.01, .03),  # bore radius of first lens
    "L2": (.05, .3),  # length of second lens
    "rp2": (.01, .03),  # bore radius of second lens
    "Lm_combiner": (.05, .25),  # hard edge length of combiner
    "load_beam_offset": (1e-3, 30e-3),  # assumed diameter of incoming beam
    "gap1": (.05, .4),  # separation between source and first lens
    "gap2": (.05, .4),  # separation between two lenses
    "gap3": (.05, .4)  ##separation between final lens and input to combnier
})


def make_injector_lattice(injector_params: dict, options: dict = None) -> ParticleTracerLattice:
    injector_params = LockedDict(injector_params)

    gap1 = injector_params["gap1"]
    gap2 = injector_params["gap2"]
    gap3 = injector_params["gap3"]
    gap1 -= el_fringe_space('lens', injector_params["rp1"])
    gap2 -= el_fringe_space('lens', injector_params["rp1"]) + el_fringe_space('lens', injector_params["rp2"])
    if gap2 < injector_constants["lens1ToLens2_Inject_Gap"]:
        raise InjectorGeometryError
    if gap1 < injector_constants["sourceToLens1_Inject_Gap"]:
        raise InjectorGeometryError
    lattice = ParticleTracerLattice(atom_characteristics["nominalDesignSpeed"],
                                    include_mag_errors=options['include_mag_errors'],
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'],
                                    include_misalignments=options['include_misalignments'], )
    if options['has_bumper']:
        add_Kevin_Bumper_Elements(lattice)

    # -----gap between source and first lens-----

    add_drift_if_needed(lattice, gap1, 'none', 'lens', np.inf, injector_params["rp1"])  # hacky

    # ---- first lens------

    lattice.add_halbach_lens_sim(injector_params["rp1"], injector_params["L1"])

    # -----gap with valve--------------

    gap_valve = injector_constants["lens1ToLens2_Valve_Length"]
    gap2 = gap2 - gap_valve
    if gap2 < 0:  # this is approximately true. I am ignoring that there is space in the fringe fields
        raise InjectorGeometryError
    lattice.add_drift(gap2, ap=injector_params["rp1"])
    lattice.add_drift(gap_valve, ap=injector_constants["lens1ToLens2_Valve_Ap"],
                      outer_half_width=injector_constants["lens1ToLens2_Inject_Valve_OD"] / 2)

    lattice.add_halbach_lens_sim(injector_params["rp2"], injector_params["L2"])

    lattice.add_drift(gap3, ap=injector_params["rp2"])
    lattice.add_combiner_sim_lens(injector_params["Lm_combiner"], system_constants["rp_combiner"],
                                  load_beam_offset=injector_params["load_beam_offset"], layers=1,
                                  seed=options['combiner_seed'], atom_state='HIGH_SEEK')

    lattice.end_lattice(constrain=False, build_field_helpers=options['build_field_helpers'])

    injector_params.assert_All_Entries_Accessed_And_Reset_Counter()

    return lattice


def inside_diam(bore_radius):
    return 2 * (bore_radius - TUBE_WALL_THICKNESS)


def helium_dump_back_gas_load():
    Q = 3.6e-5  # assuming 1 cm hole located 2 meter from nozzle,60 sccm, and dump chamber is 3 meter from nozzle
    dummy_speed = 1e9  # assume that none of the helium that wanders back through the tube returns
    vac_sys = VacuumSystem(gas_mass_Daltons=gas_masses['He'])
    vac_sys.add_chamber(Q=Q, S=1000)
    vac_sys.add_tube(50, 2)
    vac_sys.add_chamber(S=dummy_speed)
    solve_vac_system(vac_sys)
    Q_back = vac_sys.components[-1].P * vac_sys.components[-1].S()
    return Q_back


def make_vacuum_model(injector_params: dict) -> VacuumSystem:
    raise NotImplementedError  # need to rework this
    gas = 'He'
    gap1 = meter_to_cm(injector_params["gap1"])
    gap2 = meter_to_cm(injector_params["gap2"])
    gap3 = meter_to_cm(injector_params["gap3"])
    rp1, L1 = meter_to_cm(injector_params["rp1"]), meter_to_cm(injector_params["L1"])
    rp2, L2 = meter_to_cm(injector_params["rp2"]), meter_to_cm(injector_params["L2"])
    rp1_bumper = meter_to_cm(bumper.rp1)
    rp2_bumper = meter_to_cm(bumper.rp2)

    first_tube_length_bumper = meter_to_cm(bumper.L_lens1 + bumper.L_gap / 2.0)
    second_tube_length_bumper = meter_to_cm(bumper.L_lens2 + bumper.L_gap / 2.0)

    first_tube_length_mode_match = L1 + gap1 + gap2 / 2.0

    second_tube_length_mode_match = gap2 / 2.0 + L2

    vac_sys = VacuumSystem(gas_mass_Daltons=gas_masses[gas])
    vac_sys.add_chamber(P=big_chamber_pressure)
    # ------bumper section----
    vac_sys.add_tube(first_tube_length_bumper, inside_diam(rp1_bumper))
    vac_sys.add_chamber(S=diffusion_pump_speed, Q=helium_dump_back_gas_load())
    vac_sys.add_tube(second_tube_length_bumper, inside_diam(rp2_bumper))

    # -----mode match section----
    S_turbo = turbo_pump_speed
    vac_sys.add_tube(first_tube_length_mode_match, inside_diam(rp1))
    vac_sys.add_chamber(S=S_turbo)
    vac_sys.add_tube(second_tube_length_mode_match, inside_diam(rp2))
    vac_sys.add_chamber(S=S_turbo)
    vac_sys.add_tube(gap3, 3.0)  # estimate
    vac_sys.add_chamber(S=big_ion_pump_speed[gas])

    solve_vac_system(vac_sys)
    return vac_sys
