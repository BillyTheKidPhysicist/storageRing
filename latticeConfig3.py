#config with bigger magnets (1/4), and big combiner. Lens before combiner is a reasonable distance from the combiner

import matplotlib.pyplot as plt
import numpy as np
from particleTracerLattice import ParticleTracerLattice
from ParticleClass import Particle
from ParticleTracer import ParticleTracer
from OptimizerClass import Optimizer

def main():

    lattice = ParticleTracerLattice(200.0)

    directory='latticeConfig3_Files/'
    fileBend1 = directory+'benderSeg1.txt'
    fileBend2 = directory+'benderSeg2.txt'
    fileBender1Fringe = directory+'benderFringeCap1.txt'
    fileBenderInternalFringe1 = directory+'benderFringeInternal1.txt'
    fileBender2Fringe = directory+'benderFringeCap2.txt'
    fileBenderInternalFringe2 = directory+'benderFringeInternal2.txt'
    file2DLens = directory+'lens2D.txt'
    file3DLens = directory+'lens3D.txt'
    fileCombiner = directory+'combiner.txt'
    yokeWidth = .0254 * 5 / 8
    extraSpace = 1e-3 #extra space on each ender between bender segments
    Lm = .0254 #hard edge length of segmented bender
    rp = .0125
    Llens1 = .15 #lens length before drift before combiner inlet
    Llens2 = .3
    Llens3=0.9208923085059385
    Lcap=0.01875
    K0 = 12000000 #'spring' constant of field within 1%
    rb1=0.9993327541716397
    rb2=1.0010007387862925
    numMagnets1=110
    numMagnets2=110
    rOffsetFact = 1.00125

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
    lattice.end_Lattice(buildLattice=True)
    # #
    # #lattice.show_Lattice()
    # X=[0.09615384615384616 ,0.24615384615384617]
    # lattice.elList[2].forceFact = X[0]
    # lattice.elList[4].forceFact = X[1]
    #
    # particleTracer=ParticleTracer(lattice)
    # qi=np.asarray([-1e-10,0e-3,0.0])
    # pi=np.asarray([-200.0,0,0])
    # particle=Particle(qi,pi)
    # h=1e-6
    # T=(2 * 3.14 + 2) * 500 / 200.0
    # particleTracer.trace(particle,h,T)
    # print(particle.revolutions,particle.clipped)
    # qoArr=particle.qoArr
    # plt.plot(qoArr[:,0],1e6*qoArr[:,1])
    # plt.show()
    # plt.plot(qoArr[:, 0], 1e6 * qoArr[:, 2])
    # plt.show()
    # plt.plot(particleTracer.test)
    # plt.show()
    # pArr=particle.pArr
    # vs=np.sqrt(np.sum(pArr**2,axis=1))
    # plt.plot(vs)
    # plt.show()
    #
    # lattice.show_Lattice(particleCoords=particle.q)

    optimizer = Optimizer(lattice)
    optimizer.maximize_Suvival_Through_Lattice()
    #optimizer.optimize_Swarm_Survival_Through_Lattice_Brute([(0.05, .5), (0.05, .5)], 40, (2 * 3.14 + 2) * 250 / 200.0,h=5e-6)
if __name__=='__main__':
    main()