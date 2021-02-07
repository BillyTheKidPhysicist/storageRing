#config with smaller magnets (1/8), and smaller combiner. Lens before combiner is a reasonable distance from the combiner

import matplotlib.pyplot as plt
import numpy as np
from particleTracerLattice import ParticleTracerLattice
from ParticleClass import Particle
from ParticleTracer import ParticleTracer
from OptimizerClass import Optimizer

def main():

    lattice = ParticleTracerLattice(200.0)

    directory='latticeConfig5_Files/'
    fileBend1 = directory+'benderSeg1.txt'
    fileBend2 = directory+'benderSeg2.txt'
    fileBender1Fringe = directory+'benderFringeCap1.txt'
    fileBenderInternalFringe1 = directory+'benderFringeInternal1.txt'
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
    Llens3=0.6450476031362976
    Lcap=0.01875/2.0
    K0 = 48300000 #'spring' constant of field within 1%
    rb1=0.9991031328651591
    rb2=1.0010620406109336
    numMagnets1=208
    numMagnets2=208
    rOffsetFact=1.0005


    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
    #lattice.add_Drift(LDrift,ap=.015)
    #lattice.add_Combiner_Ideal(sizeScale=2.0)
    lattice.add_Combiner_Sim(fileCombiner,sizeScale=1.0)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens2)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1, fileBender1Fringe, fileBenderInternalFringe1, Lm,Lcap,rp,K0,
                                                  numMagnets1, rb1,extraSpace, yokeWidth,rOffsetFact)
    #lattice.add_Bender_Ideal(None,1.0,1.0,rp)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens3)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2, fileBender2Fringe, fileBenderInternalFringe2, Lm,Lcap,rp, K0,
                                                  numMagnets2, rb2,extraSpace, yokeWidth,rOffsetFact)
    #lattice.add_Bender_Ideal(None,1.0,1.0,rp)
    lattice.end_Lattice(buildLattice=False)
    #lattice.show_Lattice()
    print(lattice.solve_Combiner_Constraints())
    sys.exit()
    X=[0.21842105,0.79157895]
    lattice.elList[2].forceFact = X[0]
    lattice.elList[4].forceFact = X[1]

    particleTracer=ParticleTracer(lattice)
    qi=np.asarray([-1e-10,0e-3,0.0])
    pi=np.asarray([-200.0,0,0])
    particle=Particle(qi,pi)
    h=1e-5
    T=100.0/200
    particle=particleTracer.trace(particle,h,T)
    print(particle.revolutions)
    qoArr=particle.qoArr
    plt.plot(qoArr[:,0],1e6*qoArr[:,1])
    plt.show()
    plt.plot(qoArr[:, 0], 1e6 * qoArr[:, 2])
    plt.show()
    plt.plot(particleTracer.test)
    plt.show()
    pArr=particle.pArr
    vs=np.sqrt(np.sum(pArr**2,axis=1))
    plt.plot(vs)
    plt.show()

    lattice.show_Lattice(particleCoords=particle.q)
    T = 100 * lattice.totalLength / 200.0

    #X=[0.3696  ,1.93282 ,0.09977 ,0.11088, 0.15343,1e-5,T]
    #print(optimizer.compute_Survival_Through_Lattice(*X))
    #optimizer = Optimizer(lattice)
    #optimizer.optimize_Swarm_Survival_Through_Lattice_Brute([(.01, 1.0), (.01, 1.0)], 20, (2 * 3.14 + 2) * 100 / 200.0)
if __name__=='__main__':
    main()