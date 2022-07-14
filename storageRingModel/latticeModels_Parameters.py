import numpy as np

from constants import DEFAULT_ATOM_SPEED
from helperTools import inch_to_meter

realNumber = (float, int, np.float64, np.int64)


class LockedDict(dict):

    def __init__(self, dictionary: dict):
        super().__init__(dictionary)
        self._isKeyUsed = {}
        self._reset_Use_Counter()

    def _reset_Use_Counter(self):
        """Reset dictionary that records if a parameter was used"""
        for key in super().keys():
            self._isKeyUsed[key] = False

    def __setitem__(self, key, item):

        raise Exception("this dictionary cannot have new items added")

    def pop(self, *args):
        raise Exception("entries cannot be removed from dictionary")

    def clear(self):
        raise Exception("dictionary cannot be cleared")

    def __delete__(self, instance):
        raise Exception("dictionary cannot be deleted except by garbage collector")

    def __getitem__(self, key):
        """Get key, and record that it was accesed to later it can be checked wether every value was accessed"""

        assert key in self._isKeyUsed.keys()
        self._isKeyUsed[key] = True
        return super().__getitem__(key)

    def super_Special_Change_Item(self, key, item):

        assert key in super().keys()
        assert type(item) in realNumber and item >= 0.0
        super().__setitem__(key, item)

    def assert_All_Entries_Accessed_And_Reset_Counter(self):
        """Check that every value in the dictionary was accesed, and reset counter"""

        for value in self._isKeyUsed.values():
            assert value  # value must have been used
        self._reset_Use_Counter()


INJECTOR_TUNABILITY_LENGTH = 2e-2

# optimal injector parameters
injectorParamsOptimalAny: LockedDict = LockedDict({
    "L1": 0.05285079,  # length of first lens
    "rp1": 0.01473206,  # bore radius of first lens
    "L2": 0.16540282,  # length of first lens
    "rp2": 0.02387552,  # bore radius of first lens
    "Lm_combiner": 0.1480253,  # hard edge length of combiner
    "load_beam_offset": 0.00820821,  # offset of incoming beam into combiner
    "gap1": 0.09821368,  # separation between source and first lens
    "gap2": 0.2650349,  # separation between two lenses
    "gap3": 0.21845571  ##separation between final lens and input to combiner
})

injectorParamsBoundsAny: LockedDict = LockedDict({
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

injectorRingConstraintsV1: LockedDict = LockedDict({
    'rp1LensMax': .01
})

# flange outside diameters
flange_OD: LockedDict = LockedDict({
    '1-1/3': 34e-3,
    '2-3/4': 70e-3,
    '4-1/2': 114e-3
})

standard_Tubing_OD = (3 / 16, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 1.0)

atomCharacteristic = LockedDict(
    {"nominalDesignSpeed": DEFAULT_ATOM_SPEED})  # Elements are placed assuming this is the nominal
# speed in the ring.

# constraints and parameters of version 1 storage ring/injector
_system_constants = {
    "Lm": .0254 / 2.0,  # length of individual magnets in bender
    "gap2Min": inch_to_meter(3.5),  # from lens to combiner
    "OP_MagWidth": .065 + 2 * .035,  # account for fringe fields with .02
    "OP_MagAp_Injection": .022 / 2.0,
    "OP_MagAp_Circulating": .035 / 2.0,
    "OP_PumpingRegionLength": .01,  # distance for effective optical pumping
    # "bendTubeODMax": inch_to_meter(3 / 4) ,
    "rp_combiner": 0.04,
    'bendApexGap': inch_to_meter(1.0),
    "lensToBendGap": inch_to_meter(2),  # same at each bend to lens joint. Vacuum tube limited
    "observationGap": inch_to_meter(2),  # gap required for observing atoms
    "rbTarget": 1.0,  # target bending radius
    "sourceToLens1_Inject_Gap": .05,  # gap between source and first lens. Shouldn't have first lens on top of source
    "lens1ToLens2_Inject_Gap": inch_to_meter(5.9),  # pumps and valve
    "lens1ToLens2_Valve_Ap": inch_to_meter(.75),  # aperture (ID/2) for valve #2 3/4
    "lens1ToLens2_Valve_Length": inch_to_meter(3.25),  # includes flanges and screws
    "lens1ToLens2_Inject_Valve_OD": flange_OD['2-3/4']  # outside diameter of valve
}
system_constants: LockedDict = LockedDict(_system_constants)

# --------optimizer bounds-------

# bounds for optimization for ring injector system. These are also used to extract keys that
# correspond to variables

# version 1 bounds
# simple model with [lens,combiner,lens,bender,lens,lens,bender]
optimizerBounds_V1_3: LockedDict = LockedDict({
    'rp_lens3_4': (.005, .03),
    'rpLens1': (.005, injectorRingConstraintsV1['rp1LensMax']),
    'rpLens2': (.02, .04),
    'rp_bend': (.005, .01),
    'L_Lens1': (.1, .6),
    'L_Lens2': (.1, .7)
})

# version 2 bounds
# more complicated model with 2 lens on either side of combiner
optimizerBounds_V2: LockedDict = LockedDict({
    'rp_lens3_4': (.005, .03),
    'rpLens1': (.005, injectorRingConstraintsV1['rp1LensMax'] * 2),
    'rpLens2': (.005, injectorRingConstraintsV1['rp1LensMax']),
    'rpLens3': (.01, .04),
    'rpLens4': (.01, .04),
    'rp_bend': (.005, .01),
    'L_Lens1': (.1, .4),
    'L_Lens2': (.1, .4),
    'L_Lens3': (.1, .4),
    'L_Lens4': (.1, .4)
})

# as of June 23, solution 6
ringParamsOptimal_V3 = LockedDict({
    'rp_lens3_4': 0.02102425839849725,
    'rpLens1': .01,
    'rp_bend': 0.00759624174653381,
    'L_Lens1': 0.441164241347491,
    'L_Lens2': 0.46839105549798354
})

constants_V1_3 = LockedDict({
    "rpLens2": .04
})
