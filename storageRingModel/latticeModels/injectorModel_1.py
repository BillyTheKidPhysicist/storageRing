import numpy as np

from KevinBumperClass import add_Kevin_Bumper_Elements
from ParticleTracerLatticeClass import ParticleTracerLattice
from helperTools import inch_to_meter
from latticeModels.latticeModelFunctions import el_fringe_space, add_drift_if_needed, check_and_add_default_values
from latticeModels.latticeModelParameters import system_constants, atom_characteristics, flange_OD
from latticeModels.latticeModelUtilities import LockedDict, InjectorGeometryError

injector_constants = LockedDict({
    "pre_combiner_gap": inch_to_meter(3.5),  # from lens to combiner
    "OP_MagWidth": .065 + 2 * .035,  # account for fringe fields with .02
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
    "L1": (.05, .3),  # length of first lens
    "rp1": (.01, .03),  # bore radius of first lens
    "L2": (.05, .3),  # length of second lens
    "rp2": (.01, .03),  # bore radius of second lens
    "Lm_combiner": (.05, .25),  # hard edge length of combiner
    "load_beam_offset": (5e-3, 30e-3),  # assumed diameter of incoming beam
    "gap1": (.05, .3),  # separation between source and first lens
    "gap2": (.05, .3),  # separation between two lenses
    "gap3": (.05, .3)  ##separation between final lens and input to combnier
})

injector_params_optimal = LockedDict({
    "L1": 0.09622282605012868,  # length of first lens
    "rp1": 0.01,  # bore radius of first lens
    "L2": 0.24155344028683254,  # length of first lens
    "rp2": 0.03,  # bore radius of first lens
    "Lm_combiner": 0.17047182734978528,  # hard edge length of combiner
    "load_beam_offset": 0.007187101087953732,  # offset of incoming beam into combiner
    "gap1": 0.06778754563396035,  # separation between source and first lens
    "gap2": 0.2709792520743716,  # separation between two lenses
    "gap3": 0.2293989213563593  ##separation between final lens and input to combiner
})


def make_injector_lattice(injector_params: dict, options: dict = None) -> ParticleTracerLattice:
    injector_params = LockedDict(injector_params)
    options = check_and_add_default_values(options)

    options = check_and_add_default_values(options)
    gap1 = injector_params["gap1"]
    gap2 = injector_params["gap2"]
    gap3 = injector_params["gap3"]
    gap1 -= el_fringe_space('lens', injector_params["rp1"])
    gap2 -= el_fringe_space('lens', injector_params["rp1"]) + el_fringe_space('lens', injector_params["rp2"])
    if gap2 < injector_constants["lens1ToLens2_Inject_Gap"]:
        raise InjectorGeometryError
    if gap1 < injector_constants["sourceToLens1_Inject_Gap"]:
        raise InjectorGeometryError
    lattice = ParticleTracerLattice(atom_characteristics["nominalDesignSpeed"], lattice_type='injector',
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

    gap_valve = injector_constants["lens1ToLens2_Valve_Length"]
    gap2 = gap2 - gap_valve
    if gap2 < 0:  # this is approximately true. I am ignoring that there is space in the fringe fields
        raise InjectorGeometryError
    lattice.add_drift(gap2, ap=injector_params["rp1"])
    lattice.add_drift(gap_valve, ap=injector_constants["lens1ToLens2_Valve_Ap"],
                      outer_half_width=injector_constants["lens1ToLens2_Inject_Valve_OD"] / 2)

    # ---------------------

    lattice.add_halbach_lens_sim(injector_params["rp2"], injector_params["L2"])

    lattice.add_drift(gap3, ap=injector_params["rp2"])
    lattice.add_combiner_sim_lens(injector_params["Lm_combiner"], system_constants["rp_combiner"],
                                  load_beam_offset=injector_params["load_beam_offset"], layers=1,
                                  seed=options['combiner_seed'])

    lattice.end_lattice(constrain=False)

    injector_params.assert_All_Entries_Accessed_And_Reset_Counter()

    return lattice
