#config with smaller magnets (1/8), and smaller combiner.

import sys
import matplotlib.pyplot as plt
import numpy as np
from particleTracerLattice import ParticleTracerLattice
from ParticleClass import Particle
from ParticleTracer import ParticleTracer
from OptimizerClass import LatticeOptimizer

def get_Lattice(trackPotential=False):

    lattice = ParticleTracerLattice(200.0)
    directory='bigCombinerSmallMagnets_Files/'
    fileBend1 = directory+'benderSeg1.txt'
    fileBend2 = directory+'benderSeg2.txt'
    fileBender1Fringe =directory+'benderFringeCap1.txt'
    fileBenderInternalFringe1 =directory+'benderFringeInternal1.txt'
    fileBender2Fringe = directory+'benderFringeCap2.txt'
    fileBenderInternalFringe2 = directory+'benderFringeInternal2.txt'
    file2DLens = directory+'lens2D.txt'
    file3DLens = directory+'lens3D.txt'
    fileCombiner = directory+'combiner.txt'
    yokeWidth = (.0254 * 5 / 8)/2.0
    extraSpace = 1e-3 #extra space on each ender between bender segments
    Lm = .0254/2.0 #hard edge length of segmented bender
    rp = .0125/2.0
    Llens1 = .15 #lens length before drift before combiner inlet
    Llens2 = .3
    Llens3=0.9211288392979132
    Lcap=0.01875/2.0
    K0 = 47600000 #'spring' constant of field within 1%
    rb1=0.9992389431496942
    rb2=1.00092917243572
    numMagnets1=208
    numMagnets2=208
    rOffsetFact=1.0005


    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
    #lattice.add_Drift(LDrift,ap=.015)
    #lattice.add_Combiner_Ideal(sizeScale=2.0)
    lattice.add_Combiner_Sim(fileCombiner,sizeScale=2.0)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens2)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1, fileBender1Fringe, fileBenderInternalFringe1, Lm,Lcap,rp,K0,
                                                  numMagnets1, rb1,extraSpace, yokeWidth,rOffsetFact)
    #lattice.add_Bender_Ideal(None,1.0,1.0,rp)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens3)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2, fileBender2Fringe, fileBenderInternalFringe2, Lm,Lcap,rp, K0,
                                                  numMagnets2, rb2,extraSpace, yokeWidth,rOffsetFact)
    #lattice.add_Bender_Ideal(None,1.0,1.0,rp)
    lattice.end_Lattice(trackPotential=trackPotential)
    return lattice

def compute_Sol(h,Revs,numParticles,maxEvals,bounds=None):
    lattice=get_Lattice()
    T=Revs*lattice.totalLength/lattice.v0Nominal
    optimizer=LatticeOptimizer(lattice)


    #name='smallCombinerSmallMagnets_sub'
    #optimizer.plot_Stability(bounds=[(0.0, 0.3), (0.2, 0.5)], gridPoints=20, savePlot=False, plotName=name)
    sol=optimizer.maximize_Suvival_Through_Lattice(h,T,numParticles=numParticles,maxEvals=maxEvals,bounds=bounds)
    return sol


    #sol=optimizer.maximize_Suvival_Through_Lattice(h,T,numParticles=numParticles,maxEvals=maxEvals)
    #return sol
    # particle=Particle()
    # particleTracer=ParticleTracer(lattice)
    #
    #
    # particle=particleTracer.trace(particle,h,T,fastMode=False)
    # qoArr=particle.qoArr
    # EArr=particle.EArr
    # #lattice.show_Lattice(particleCoords=particle.qArr[-1])
