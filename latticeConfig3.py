#config with bigger magnets (1/4), and big combiner. Lens before combiner is a reasonable distance from the combiner

import matplotlib.pyplot as plt
import numpy as np
from particleTracerLattice import ParticleTracerLattice
from ParticleClass import Particle
from ParticleTracer import ParticleTracer
from OptimizerClass import Optimizer
import time

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

    h = 5e-6
    T = 100 * lattice.totalLength / lattice.v0Nominal
    optimizer = Optimizer(lattice)
    t = time.time()
    sol = optimizer.maximize_Suvival_Through_Lattice(h, T, numParticles=500, qMax=3e-3, pMax=5.0, returnBestSwarm=True,
                                                     maxEvals=500)
    print(time.time() - t)  # 94
    print(sol)
    print(sol[1][0], sol[1][1])
    #(7.579945305938145, array([0.15809508, 0.1527832]), < ParticleClass.Swarm object at 0x7f2715ccdfd0 >)
    #0.15809507948483545 0.15278320000328438
    # TODO: PARTICLE IS OUTSIDE ELELMENT WARNING
if __name__=='__main__':
    main()