#config with bigger magnets (1/4), and small combiner. Lens before combiner is a reasonable distance from the combiner
import numpy as np
import matplotlib.pyplot as plt
from OptimizerClass import LatticeOptimizer
from particleTracerLattice import ParticleTracerLattice
from ParticleTracer import ParticleTracer
from ParticleClass import Particle

def get_Lattice(trackPotential=False):
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
    fileCombiner = directory + 'combinerData.txt'
    yokeWidth = .0254 * 5 / 8
    extraSpace = 1e-3  # extra space on each ender between bender segments
    Lm = .0254  # hard edge length of segmented bender
    rp = .0125
    Llens1 = .15  # lens length before drift before combiner inlet
    Llens2 = .3
    Llens3 = 0.648514129576195
    Lcap = 0.01875
    K0 = 12000000  # 'spring' constant of field within 1%
    rb1 = 0.9987085104809762
    rb2 = 1.0004783330494842
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
    lattice.end_Lattice(trackPotential=trackPotential,enforceClosedLattice=False)
    #lattice.show_Lattice()
    return lattice

def compute_Sol(h,Revs,numParticles,maxEvals):
    lattice=get_Lattice(trackPotential=True)
    #lattice.show_Lattice()
    #T=Revs*lattice.totalLength/lattice.v0Nominal
    #optimizer=LatticeOptimizer(lattice)
    # from SwarmTracer import SwarmTracer
    # swarmTracer = SwarmTracer(lattice)
    # swarmNew = swarmTracer.initialize_Swarm_At_Combiner_Output(.15, 1.0, 0.0, labFrame=False,numPhaseSpace=10)
    #name='smallCombinerBigMagnets_Plot_100Grid'
    #optimizer.plot_Unstable_Regions(gridPoints=100,savePlot=True,plotName=name)

    #name='smallCombinerBigMagnets'
    #optimizer.plot_Stability(bounds=[(0.0,0.5),(0.0,0.5)],gridPoints=100,savePlot=True,plotName=name)
    #func=optimizer.maximize_Suvival_Through_Lattice(h,T,numParticles=numParticles,maxEvals=maxEvals)
    #return func,lattice
    #return sol
    #X=[.1,.28]
    #lattice.elList[2].forceFact = X[0]
    #lattice.elList[4].forceFact = X[1]

    particle=Particle(qi=np.asarray([-1e-3,1.5e-20,1e-10]))
    particleTracer=ParticleTracer(lattice)
    T = .2 * lattice.totalLength / lattice.v0Nominal
    h=1e-6
    particle=particleTracer.trace(particle,h,T,fastMode=False)
    lattice.show_Lattice(particle=particle)
    #np.savetxt('poopaids',particle.q)
    #qTest=np.loadtxt('poopaids')
    #print(particle.q))
    plt.plot(particleTracer.test)
    plt.show()
    particle.plot_Energies()
    #particle.plot_Position()
    #lattice.show_Lattice(particleCoords=particle.qArr[-1])
#if __name__=='__main__':
#    main()
compute_Sol(1e-5,20,500,50)

