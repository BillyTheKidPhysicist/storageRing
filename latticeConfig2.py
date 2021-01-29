#config with bigger magnets (1/4), and larger combiner. Lens before combiner is a reasonable distance from the combiner

import matplotlib.pyplot as plt
import numpy as np
from particleTracerLattice import ParticleTracerLattice
from ParticleClass import Particle
from ParticleTracer import ParticleTracer

lattice = ParticleTracerLattice(200.0)

directory='latticeConfig2_Files\\'
fileBend1 = None#directory+'benderSeg1.txt'
fileBend2 = None#directory+'benderSeg2.txt'
fileBender1Fringe = None#directory+'benderFringeCap1.txt'
fileBenderInternalFringe1 = None#directory+'benderFringeInternal1.txt'
fileBender2Fringe = None#directory+'benderFringeCap2.txt'
fileBenderInternalFringe2 = None#directory+'benderFringeInternal2.txt'
file2DLens = directory+'lens2D.txt'
file3DLens = directory+'lens3D.txt'
fileCombiner = directory+'combinerData.txt'
yokeWidth = .0254 * 5 / 8
numMagnets = 110
extraSpace = 1e-3 #extra space on each ender between bender segments
Lm = .0254 #hard edge length of segmented bender
rp = .0125
Llens1 = .2 #lens length before drift before combiner inlet
Llens2 = .3
Llens3=0.9194834317505234
LDrift=.2 #drift length before inlet to combiner
rb = 1.0
Lcap=0.01875
K0 = 12000000 #'spring' constant of field within 1%
rb1=1.008616613394432
rb2=1.0177466037053944
numMagnets1=110
numMagnets2=113

lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
lattice.add_Drift(LDrift)
lattice.add_Combiner_Ideal()
lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens2)
#lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1, fileBender1Fringe, fileBenderInternalFringe1, Lm,Lcap,rp,K0,
#                                              numMagnets1, rb1,extraSpace, yokeWidth)
lattice.add_Bender_Ideal(None,1.0,1.0,rp)
lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens3)
#lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2, fileBender2Fringe, fileBenderInternalFringe2, Lm,Lcap,rp, K0,
#                                              numMagnets2, rb2,extraSpace, yokeWidth)
lattice.add_Bender_Ideal(None,1.0,1.0,rp)
lattice.end_Lattice()
lattice.show_Lattice()
#print(lattice.solve_Combiner_Constraints())

#lattice.end_Lattice()
#lattice.show_Lattice()
#
#particleTracer=ParticleTracer(lattice)
#qi=np.asarray([-1e-10,0,0])
#pi=np.asarray([-200.0,0,0])
#particle=Particle(qi,pi)
#h=1e-5
#T=1.0/200
#particleTracer.trace(particle,h,T)
#
#qoArr=particle.qoArr
#plt.plot(qoArr[:,0],qoArr[:,1])
#plt.show()