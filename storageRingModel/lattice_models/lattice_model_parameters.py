"""
Contains parameters used in building ring and injector such as system wide constants or default values
"""

import numpy as np

from constants import DEFAULT_ATOM_SPEED
from helper_tools import inch_to_meter
from lattice_models.utilities import LockedDict

# IMPROVEMENT: rename to system
realNumber = (float, int, np.float64, np.int64)

INJECTOR_TUNABILITY_LENGTH = 2e-2

DEFAULT_SYSTEM_OPTIONS = LockedDict({'include_mag_errors': False, 'combiner_seed': None, 'use_solenoid_field': False,
                                     'has_bumper': False, 'use_standard_tube_OD': False,
                                     'include_mag_cross_talk_in_ring': False,
                                     'include_misalignments': False, 'build_field_helpers': True,
                                     'field_dens_mult':1.0})

combiner_param_bounds: LockedDict = LockedDict({
    "Lm_combiner": (.05, .25),  # hard edge length of combiner
    "load_beam_offset": (1e-3, 30e-3),  # assumed diameter of incoming beam
})

# flange outside diameters
flange_OD: LockedDict = LockedDict({
    '1-1/3': 34e-3,
    '2-3/4': 70e-3,
    '4-1/2': 114e-3
})

atom_characteristics = LockedDict({"nominalDesignSpeed": DEFAULT_ATOM_SPEED})  # Elements are placed assuming this is
# the nominal speed in the ring.


system_constants = LockedDict({
    "Lm": .0254 / 2.0,  # length of individual magnets in bender
    "OP_mag_space": .065 + 2 * .035,  # account for fringe fields with .02
    "OP_MagAp_Injection": .022 / 2.0,
    "OP_MagAp_Circulating": .035 / 2.0,
    "OP_PumpingRegionLength": .01,  # distance for effective optical pumping
    # "bendTubeODMax": inch_to_meter(3 / 4) ,
    "rp_combiner": 0.04,
    'bendApexGap': inch_to_meter(1.0),
    "lensToBendGap": inch_to_meter(2),  # same at each bend to lens joint. Vacuum tube limited
    "observationGap": inch_to_meter(2),  # gap required for observing atoms
    "rbTarget": 1.0,  # target bending radius
    "pre_combiner_gap": inch_to_meter(3.5)
})
