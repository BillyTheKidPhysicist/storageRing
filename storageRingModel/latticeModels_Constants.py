
class lockedDict(dict):
    def __init__(self,dictionary):
        super().__init__(dictionary)

    def __setitem__(self, key, item):
        if key not in self.keys():
            raise Exception("this dictionary cannot have new items added")
        else:
            super().__setitem__(key,item)





constants_Version1=lockedDict({
    "Lm": .0254 / 2.0, #length of individual magnets in bender
    "gap2Min": 3.5*.0254, #from lens to combiner
    "OP_MagWidth": .05+.02, #account for fringe fields with .02
    "OP_MagAp":.012/2.0,
    "OP_PumpingRegionLength":.01, #distance for effective optical pumping
    "bendingApMax":.01, #maximum from 1.33 flange limit. I rounded up
    "lensToBendGap":2*.0254, #same at each joint. Vacuum tube limited
    "observationGap": 2*.0254, #gap required for observing atoms
    "rbTarget": 1.0, #target bending radius
    "lens1ToLens2_Inject_Gap":5.9*.0254, #pumps and valve
    "lens1ToLens2_Inject_Ap":.75*.0254 #aperture for valve
})