




# gap3Min=opticalPumpingMagnetWidth+.02

constants_Version1={
    "Lm": .0254 / 2.0, #length of individual magnets in bender
    "gap2Min": 3.5*.0254,
    "OP_MagWidth": .05+.02, #account for fringe fields
    "OP_MagAp":.012/2.0,
    "bendingApMax":.01, #maximum from 1.33 flange limit. I rounded up
    "lensToBendGap":2*.0254, #same at each joint. Vacuum tube limited
    "observationGap": 2*.0254, #gap required for observing atoms
    "rbTarget": 1.0,
    "lens1ToLens2_Inject_Gap":5.9*.0254, #pumps and valve
    "lens1ToLens2_Inject_Ap":.75*.0254 #aperture for valve
}