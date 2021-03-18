#config with bigger magnets (1/4), and small combiner. Lens before combiner is a reasonable distance from the combiner
import numpy as np
import matplotlib.pyplot as plt
from OptimizerClass import LatticeOptimizer
from particleTracerLattice import ParticleTracerLattice
from ParticleTracer import ParticleTracer
from ParticleClass import Particle


def get_Lattice(trackPotential=True):
    lattice = ParticleTracerLattice(200.0)
    directory = 'smallCombinerBigMagnets_Files/'
    fileBend1 = directory + 'benderSeg1.txt'
    fileBend2 = directory + 'benderSeg2.txt'
    fileBender1Fringe = directory + 'benderFringeCap1.txt'
    fileBenderInternalFringe1 = directory + 'benderFringeInternal1.txt'
    fileBender2Fringe = directory + 'benderFringeCap2.txt'
    fileBenderInternalFringe2 = directory + 'benderFringeInternal2.txt'
    file2DLens = directory + 'lens2D.txt'
    file3DLens = directory + 'lens3D.txt'
    fileCombiner = directory + 'combinerV2.txt'
    yokeWidth = .0254 * 5 / 8
    extraSpace = 1e-3  # extra space on each ender between bender segments
    Lm = .0254  # hard edge length of segmented bender
    rp = .0125
    Llens1 = .15  # lens length before drift before combiner inlet
    Llens2 = .3
    Llens3 =0.6660736971880906
    Lcap = 0.01875
    K0 = 12000000  # 'spring' constant of field within 1%
    rb1 = 0.9991839560717193
    rb2 = 1.0011135734421053
    numMagnets1 = 110
    numMagnets2 = 110
    rOffsetFact = 1.00125
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
    lattice.add_Combiner_Sim(fileCombiner, sizeScale=1.0)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens2)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1, fileBender1Fringe, fileBenderInternalFringe1, Lm, Lcap, rp,
                                                   K0, numMagnets1, rb1, extraSpace, yokeWidth, rOffsetFact)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens3)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2, fileBender2Fringe, fileBenderInternalFringe2, Lm, Lcap, rp,
                                                   K0,
                                                   numMagnets2, rb2, extraSpace, yokeWidth, rOffsetFact)
    lattice.end_Lattice(trackPotential=trackPotential,enforceClosedLattice=False,buildLattice=True)
    #print(lattice.solve_Combiner_Constraints())
    lattice.show_Lattice()
    return lattice

#optimizer=None
def compute_Sol(h,Revs,maxEvals):
    print('here')
    #global optimizer
    lattice=get_Lattice(trackPotential=True)
    #lattice.show_Lattice()
    T=Revs*lattice.totalLength/lattice.v0Nominal
    qi=np.asarray([-.1-1e-15,0.0,0.0])
    particle=Particle(qi=qi)
    particleTracer=ParticleTracer(lattice)
    print('----------trace--------------')
    particleTracer.trace(particle,1e-7,.002)
    #particle.plot_Energies()
    particle.plot_Position()
    #1e-6: -5e-7
    #5e-7: -4.3e-7
    #3e-7
    #1e-7


    lattice.show_Lattice(particle=particle,showTraceLines=True)
    # optimizer=LatticeOptimizer(lattice)
    # sol=optimizer.maximize_Suvival_Through_Lattice(h,T,maxEvals=maxEvals)
    #return sol
compute_Sol(1e-5,100.0,20)