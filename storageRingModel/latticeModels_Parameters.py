import copy
from helperTools import inch_To_Meter
from constants import VACUUM_TUBE_THICKNESS, DEFAULT_ATOM_SPEED
import numpy as np

realNumber = (float, int, np.float64, np.int64)


class lockedDict(dict):

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

    def __getitem__(self, key) -> float:
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


# optimal injector parameters
injectorParamsOptimalAny: lockedDict = lockedDict({
    "L1": 0.153132,  # length of first lens
    "rp1": 0.02963415,  # bore radius of first lens
    "L2": 0.13430734,  # length of first lens
    "rp2": 0.02258658,  # bore radius of first lens
    "LmCombiner": 0.19126756,  # hard edge length of combiner
    "rpCombiner": 0.0418939,  # bore radius of combiner
    "loadBeamDiam": 0.01501351,  # assumed diameter of incoming beam
    "gap1": 0.20163243,  # separation between source and first lens
    "gap2": 0.23014572,  # separation between two lenses
    "gap3": 0.17625537  ##separation between final lens and input to combiner
})

injectorParamsBoundsAny: lockedDict = lockedDict({
    "L1": (.05, .3),  # length of first lens
    "rp1": (.01, .03),  # bore radius of first lens
    "L2": (.05, .3),  # length of first lens
    "rp2": (.01, .03),  # bore radius of first lens
    "LmCombiner": (.02, .25),  # hard edge length of combiner
    "rpCombiner": (.005, .04),  # bore radius of combiner
    "loadBeamDiam": (5e-3, 30e-3),  # assumed diameter of incoming beam
    "gap1": (.05, .3),  # separation between source and first lens
    "gap2": (.05, .3),  # separation between two lenses
    "gap3": (.05, .3)  ##separation between final lens and input to combnier
})

injectorRingConstraintsV1: lockedDict = lockedDict({
    'rp1LensMax': .01
})

# flange outside diameters
flange_OD: lockedDict = lockedDict({
    '1-1/3': 34e-3,
    '2-3/4': 70e-3,
    '4-1/2': 114e-3
})

standard_Tubing_OD = (3 / 16, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 1.0)

atomCharacteristic = lockedDict(
    {"nominalDesignSpeed": DEFAULT_ATOM_SPEED})  # Elements are placed assuming this is the nominal
# speed in the ring.

# constraints and parameters of version 1 storage ring/injector
_constants_Version1_2 = {
    "Lm": .0254 / 2.0,  # length of individual magnets in bender
    "gap2Min": inch_To_Meter(3.5),  # from lens to combiner
    "OP_MagWidth": .065 + 2 * .035,  # account for fringe fields with .02
    "OP_MagAp_Injection": .022 / 2.0,
    "OP_MagAp_Circulating": .035 / 2.0,
    "OP_PumpingRegionLength": .01,  # distance for effective optical pumping
    "bendingApMax": .01,  # maximum from 1.33 flange limit (ID/2). I rounded up
    "lensToBendGap": inch_To_Meter(2),  # same at each bend to lens joint. Vacuum tube limited
    "observationGap": inch_To_Meter(2),  # gap required for observing atoms
    "rbTarget": 1.0,  # target bending radius
    "sourceToLens1_Inject_Gap": .05,  # gap between source and first lens. Shouldn't have first lens on top of source
    "lens1ToLens2_Inject_Gap": inch_To_Meter(5.9),  # pumps and valve
    "lens1ToLens2_Valve_Ap": inch_To_Meter(.75),  # aperture (ID/2) for valve #2 3/4
    "lens1ToLens2_Valve_Length": inch_To_Meter(3.25),  # includes flanges and screws
    "lens1ToLens2_Inject_Valve_OD": flange_OD['2-3/4']  # outside diameter of valve
}
constantsV1_2: lockedDict = lockedDict(_constants_Version1_2)

_constants_Version3 = copy.deepcopy(_constants_Version1_2)
_constants_Version3['bendApexGap'] = inch_To_Meter(1)
constantsV3: lockedDict = lockedDict(_constants_Version3)

# --------optimizer bounds-------

# bounds for optimization for ring injector system. These are also used to extract keys that
# correspond to variables

# version 1 bounds
# simple model with [lens,combiner,lens,bender,lens,lens,bender]
optimizerBounds_V1_3: lockedDict = lockedDict({
    'rpLens3_4': (.005, .03),
    'rpLens1': (.005, injectorRingConstraintsV1['rp1LensMax'] * 1.1),
    'rpLens2': (.01, .04),
    'rpBend': (.005, constantsV1_2["bendingApMax"] + VACUUM_TUBE_THICKNESS),
    'L_Lens1': (.05, .5),
    'L_Lens2': (.05, .5)
})

# version 2 bounds
# more complicated model with 2 lens on either side of combiner
optimizerBounds_V2: lockedDict = lockedDict({
    'rpLens3_4': (.005, .03),
    'rpLens1': (.005, injectorRingConstraintsV1['rp1LensMax'] * 2),
    'rpLens2': (.005, injectorRingConstraintsV1['rp1LensMax'] * 1.1),
    'rpLens3': (.01, .04),
    'rpLens4': (.01, .04),
    'rpBend': (.005, constantsV1_2["bendingApMax"] + VACUUM_TUBE_THICKNESS),
    'L_Lens1': (.1, .4),
    'L_Lens2': (.1, .4),
    'L_Lens3': (.1, .4),
    'L_Lens4': (.1, .4)
})
