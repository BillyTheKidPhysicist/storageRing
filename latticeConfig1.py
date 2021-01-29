#original configuration before the bender is moved closer to the combiner. uses the larger magnet choice, and
#real field values from actual combiner

from particleTracerLattice import ParticleTracerLattice

lattice = ParticleTracerLattice(200.0)

directory='latticeConfig1_Files\\'
fileBend1 = directory+'benderSeg1.txt'
fileBend2 = directory+'benderSeg2.txt'
fileBender1Fringe = directory+'benderFringeCap1.txt'
fileBenderInternalFringe1 = directory+'benderFringeInternal1.txt'
fileBender2Fringe = directory+'benderFringeCap2.txt'
fileBenderInternalFringe2 = directory+'benderFringeInternal2.txt'
file2DLens = directory+'lens2D.txt'
file3DLens = directory+'lens3D.txt'
fileCombiner = directory+'combinerData.txt'
yokeWidth = .0254 * 5 / 8
numMagnets = 110
extraSpace = 1e-3
Lm = .0254
rp = .0125
Llens1 = .3
Llens2 = .3
rb = 1.0
Lcap=0.01875
K0 = 12037000


lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
lattice.add_Combiner_Sim(fileCombiner)
lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens2)
lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1, fileBender1Fringe, fileBenderInternalFringe1, Lm,Lcap, None, rb,
                                             extraSpace, yokeWidth)
lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, None)
lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2, fileBender2Fringe, fileBenderInternalFringe2, Lm,Lcap, None, rb,
                                             extraSpace, yokeWidth)
lattice.end_Lattice()
lattice.show_Lattice()
