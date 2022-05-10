
class lockedDict(dict):
    def __init__(self,dictionary):
        super().__init__(dictionary)

    def __setitem__(self, key, item):
        if key not in self.keys():
            raise Exception("this dictionary cannot have new items added")
        else:
            super().__setitem__(key,item)

#flange outside diameters in mm
flange_OD=lockedDict({
    '1-1/3': 34e-3,
    '2-3/4':70e-3,
    '4-1/2': 114
})



constants_Version1=lockedDict({
    "Lm": .0254 / 2.0, #length of individual magnets in bender
    "gap2Min": 3.5*.0254, #from lens to combiner
    "OP_MagWidth": .07+.04, #account for fringe fields with .02
    "OP_MagAp":.022/2.0,
    "OP_PumpingRegionLength":.01, #distance for effective optical pumping
    "bendingApMax":.01, #maximum from 1.33 flange limit. I rounded up
    "lensToBendGap":2*.0254, #same at each joint. Vacuum tube limited
    "observationGap": 2*.0254, #gap required for observing atoms
    "rbTarget": 1.0, #target bending radius
    "sourceToLens1_Inject_Gap":.05, #gap between source and first lens. Shouldn't have first lens on top of source
    "lens1ToLens2_Inject_Gap":5.9*.0254, #pumps and valve
    "lens1ToLens2_Inject_Ap":.75*.0254/2,   #aperture for valve #2 3/4
    "lens1ToLens2_Valve_Length":2*.0254,
    "lens1ToLens2_Inject_Valve_OD": flange_OD['2-3/4'] #outside diameter of valve
})