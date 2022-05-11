import copy
from helperTools import inch_To_Meter

class lockedDict(dict):
    def __init__(self,dictionary):
        super().__init__(dictionary)

    def __setitem__(self, key, item):
        if key not in self.keys():
            raise Exception("this dictionary cannot have new items added")
        else:
            super().__setitem__(key,item)

    def pop(self,*args):
        raise Exception("entries cannot be removed from dictionary")

    def clear(self):
        raise Exception("dictionary cannot be cleared")

    def __delete__(self, instance):
        raise Exception("dictionary cannot be deleted except by garbage collector")

#optimal injector parameters
injectorParamsOptimal_Version1: dict = lockedDict({
                                "L1":.29374941, #length of first lens
                                "rp1":0.01467768, #bore radius of first lens
                                "L2":0.22837003, #length of first lens
                                "rp2":0.0291507, #bore radius of first lens
                                "LmCombiner":0.19208822, #hard edge length of combiner
                                "rpCombiner":0.04, #bore radius of combiner
                                "loadBeamDiam":0.01462034, #assumed diameter of incoming beam
                                "gap1":0.08151122, #separation between source and first lens
                                "gap2":0.27099428, #separation between two lenses
                                "gap3":0.26718875 ##separation between final lens and input to combnier
})



#flange outside diameters
flange_OD: dict=lockedDict({
    '1-1/3': 34e-3,
    '2-3/4':70e-3,
    '4-1/2': 114e-3
})


#constraints and parameters of version 1 storage ring/injector
_constants_Version1={
    "Lm": .0254 / 2.0, #length of individual magnets in bender
    "gap2Min": inch_To_Meter(3.5), #from lens to combiner
    "OP_MagWidth": .065+2*.035, #account for fringe fields with .02
    "OP_MagAp":.022/2.0,
    "OP_PumpingRegionLength":.01, #distance for effective optical pumping
    "bendingApMax":.01, #maximum from 1.33 flange limit (ID/2). I rounded up
    "lensToBendGap":inch_To_Meter(2), #same at each joint. Vacuum tube limited
    "observationGap": inch_To_Meter(2), #gap required for observing atoms
    "rbTarget": 1.0, #target bending radius
    "sourceToLens1_Inject_Gap":.05, #gap between source and first lens. Shouldn't have first lens on top of source
    "lens1ToLens2_Inject_Gap":inch_To_Meter(5.9), #pumps and valve
    "lens1ToLens2_Valve_Ap":inch_To_Meter(.75),   #aperture (ID/2) for valve #2 3/4
    "lens1ToLens2_Valve_Length":inch_To_Meter(3.25), #includes flanges and screws
    "lens1ToLens2_Inject_Valve_OD": flange_OD['2-3/4'] #outside diameter of valve
}
constantsV1: dict=lockedDict(_constants_Version1)




_constants_Version3=copy.deepcopy(_constants_Version1)
_constants_Version3['bendApexGap']=inch_To_Meter(1)
constantsV3: dict=lockedDict(_constants_Version3)