#config with bigger magnets (1/4), and small combiner. Lens before combiner is a reasonable distance from the combiner

from profilehooks import profile
import matplotlib.pyplot as plt
import numpy as np
from particleTracerLattice import ParticleTracerLattice
from ParticleClass import Particle
from ParticleTracer import ParticleTracer
from OptimizerClass import Optimizer
import time

def main():

    lattice = ParticleTracerLattice(200.0)

    directory='latticeConfig2_Files/'
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
    Llens3=0.6447075630948967
    Lcap=0.01875
    K0 = 12000000 #'spring' constant of field within 1%
    rb1=0.992877733644296
    rb2=1.0072957153331232
    numMagnets1=108
    numMagnets2=112
    rOffsetFact = 1.00125

    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
    # lattice.add_Drift(LDrift,ap=.015)
    # lattice.add_Combiner_Ideal(sizeScale=2.0)
    lattice.add_Combiner_Sim(fileCombiner, sizeScale=1.0)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens2)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1, fileBender1Fringe, fileBenderInternalFringe1, Lm, Lcap, rp,
                                                  K0,
                                                  numMagnets1, rb1, extraSpace, yokeWidth, rOffsetFact)
    # lattice.add_Bender_Ideal(None,1.0,1.0,rp)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens3)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2, fileBender2Fringe, fileBenderInternalFringe2, Lm, Lcap, rp,
                                                  K0,
                                                  numMagnets2, rb2, extraSpace, yokeWidth, rOffsetFact)
    # lattice.add_Bender_Ideal(None,1.0,1.0,rp)
    lattice.end_Lattice(buildLattice=True)

    #lattice.show_Lattice()
    #print(lattice.solve_Combiner_Constraints())

    X=[0.11666667,0.21666667]
    lattice.elList[2].forceFact = X[0]
    lattice.elList[4].forceFact = X[1]
    particleTracer=ParticleTracer(lattice)
    qi=np.asarray([-1e-10,1e-10,0.0])
    pi=np.asarray([-200.0,0,0])
    particle=Particle(qi,pi)
    h=5e-6
    T=25.0*lattice.totalLength/200 #25
    #partricle = particleTracer.trace(particle, h, .5*lattice.totalLength/200.0, fastMode=True)
    #@profile()
    def func():
        partricle=particleTracer.trace(particle,h,T,fastMode=True) #8.36, 4.617
        #T1:
        #t2:
        #t3:

        print(particle.q,particle.clipped,particle.revolutions) #[ 6.41003729e-01  2.14272318e-01 -7.68673173e-08] False: for 25 revs
    func()
    #t=time.time()
    #for i in range(5):
    #    func()
    #print((time.time()-t)/5) #4.0177, 4.19,4.15


    #particleTracer.trace(particle, h, T, fastMode=False)
    #qoArr=particle.qoArr
    ##plt.plot(qoArr[:,0],qoArr[:,1])
    #plt.plot(particleTracer.test)
    #plt.show()
    ##lattice.show_Lattice(particleCoords=particle.q)

    #optimizer = Optimizer(lattice)
    #optimizer.optimize_Swarm_Survival_Through_Lattice_Brute([(.1,.5),(.1,.5)],25,T,5e-6)
    #optimizer.maximize_Suvival_Through_Lattice()
    #t=time.time()
    #@profile()
    #def func():
    #    optimizer.initialize_Swarm_At_Combiner_Output(.2,.5,0.0,1e-4,1e-1,9)
    #func()
    #print(time.time() - t)

if __name__=='__main__':
    main()