import skopt
import numba
# from profilehooks import profile
from ParticleTracerClass import ParticleTracer
#import black_box as bb
import numpy.linalg as npl
#import matplotlib.pyplot as plt
import sys
import multiprocess as mp
from ParticleTracerLatticeClass import ParticleTracerLattice
import numpy as np
from ParticleClass import Swarm
import scipy.optimize as spo
import time
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from ParaWell import ParaWell
import skopt










class SwarmTracer:
    def __init__(self,lattice):
        self.lattice=lattice
        self.particleTracer = ParticleTracer(self.lattice)
        self.helper = ParaWell()  # custom class to help with parallelization

    def Rd_Sample(self,n,d=1,seed=.5):
        # copied and modified from: https://github.com/arvoelke/nengolib
        #who themselves got it from:  http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
        def gamma(d,n_iter=20):
            """Newton-Raphson-Method to calculate g = phi_d."""
            x=1.0
            for _ in range(n_iter):
                x-=(x**(d+1)-x-1)/((d+1)*x**d-1)
            return x

        g=gamma(d)
        alpha=np.zeros(d)
        for j in range(d):
            alpha[j]=(1/g)**(j+1)%1

        z=np.zeros((n,d))
        z[0]=(seed+alpha)%1
        for i in range(1,n):
            z[i]=(z[i-1]+alpha)%1

        return z
    def initialize_HyperCube_Swarm_In_Phase_Space(self, qMax, pMax, num, upperSymmetry=False):
        # create a cloud of particles in phase space at the origin. In the xy plane, the average velocity vector points
        # to the west. The transverse plane is the yz plane.
        # qMax: absolute value maximum position in the transverse direction
        # qMax: absolute value maximum position in the transverse momentum
        # num: number of samples along each axis in phase space. Total is num^4
        # upperSymmetry: if this is true, exploit the symmetry between +/-z and ignore coordinates below z=0
        qArr = np.linspace(-qMax, qMax, num=num)
        pArr = np.linspace(-pMax, pMax, num=num)
        argsArr = np.asarray(np.meshgrid(qArr, qArr, pArr, pArr)).T.reshape(-1, 4)
        swarm = Swarm()
        for arg in argsArr:
            qi = np.asarray([0.0, arg[0], arg[1]])
            pi = np.asarray([-self.lattice.v0Nominal, arg[2], arg[3]])
            if upperSymmetry == True:
                if qi[2] < 0:
                    pass
                else:
                    swarm.add_Particle(qi, pi)
            else:
                swarm.add_Particle(qi, pi)
        return swarm
    def initialize_Observed_Swarm_In_Phase_Space(self,numParticles=None,sameSeed=True):
        #generate a cloud in phase space from observed data from the experiment. This loads saved data because it takes
        #a while to generate these values. The returned particle are randomly sampled from the full cloud to achieve
        #the desired number of particles
        #numParticles: Number of particles sampled. Maximum of about 20000 (depends on the files exact amount). If None
        #use all
        #sameSeed: Wether to reuse the same seed for reproducible results
        #returns an array of particle coordinates as [[x,y,pz,px,py],....]

        fileName= 'phaseSpaceCloud.dat'
        data=np.loadtxt(fileName)
        maxParticles=data.shape[0]
        if numParticles==None:
            numParticles=maxParticles
        elif numParticles>maxParticles:
            raise Exception('You have requested more particles than are available in the file')
        if sameSeed==True: #use the same seed to get repeatable results
            np.random.seed(42)
        np.random.shuffle(data)
        indexArr=np.random.choice(np.arange(data.shape[0]),size=numParticles)
        particleCoords=data[indexArr]
        swarm=Swarm()
        for coords in particleCoords:
            swarm.add_Particle(qi=coords[:3],pi=coords[3:])
        if sameSeed==True:
            np.random.seed(int(time.time())) #rerandomize the seed (kind of random)
        return swarm
    def initalize_PseudoRandom_Swarm_In_Phase_Space(self,qTMax,pTMax,pxMax,numParticles,upperSymmetry=False,sameSeed=False,
                                                    circular=True,smallXOffset=True):
        #return a swarm object who position and momentum values have been randomly generated inside a phase space hypercube
        #and that is heading in the negative x direction with average velocity lattice.v0Nominal. A seed can be reused to
        #get repeatable random results. a sobol sequence is used that is then jittered. In additon points are added at
        #each corner exactly and midpoints between corners if desired
        #NOTE: it's not guaranteed that there will be exactly num particles.


        # qxMax and qyMax: x and y maximum dimension in position space of hypercube
        # pxMax and pyMax: px and py maximum dimension in momentum space of hupercube
        # pxMax: what value to use for longitudinal momentum spread. if None use the nominal value
        # num: number of particle in phase space
        # upperSymmetry: wether to exploit lattice symmetry by only using particles in upper half plane
        # sameSeed: wether to use the same seed eveythime in the nump random generator, the number 42, or a new one
        # cornerPonints: wether to add points at the corner of the hypercube
        # circular: Wether to model the transvers components as circular, which they are


        bounds=np.asarray([[-qTMax,qTMax],[-qTMax,qTMax],[-self.lattice.v0Nominal-pxMax,-self.lattice.v0Nominal+pxMax],
                           [-pTMax,pTMax],[-pTMax,pTMax]])


        if circular is True:
            numParticlesfrac=1/((np.pi/4)**2) #the ratio of the are of the circle to the cross section. There is one
            #factor for momentum and one for position
        else:
            numParticlesfrac=1.0
        if sameSeed==True:
            np.random.seed(42)
        if type(sameSeed) == int:
            np.random.seed(sameSeed)
        if upperSymmetry==True:
            bounds[1][0]=0.0 #don't let point be generarted below z=0

        swarm = Swarm()

        sampler=skopt.sampler.Sobol()
        samples=np.asarray(sampler.generate(bounds,int(numParticles*numParticlesfrac)))
        np.random.shuffle(samples)

        if smallXOffset==True:
            x0=-1e-10 #to push negative
        else:
            x0=0.0

        particleCount=0 #track how many particles have been added to swarm
        for Xi in samples:
            q = np.append(x0, Xi[:2])
            p = Xi[2:]
            if circular==True:
                y,z,py,pz=Xi[[0,1,3,4]]
                if np.sqrt(y**2+z**2)<qTMax and np.sqrt(py**2+pz**2)<pTMax:
                    swarm.add_Particle(qi=q, pi=p)
                    particleCount+=1
                if particleCount==numParticles:
                    break
            else:
                swarm.add_Particle(qi=q,pi=p)
        if sameSeed==True or type(sameSeed)==int:
            np.random.seed(int(time.time()))  # re randomize
        return swarm

    def initialize_Random_Swarm_At_Combiner_Output(self, qMax, pMax, num, upperSymmetry=False, sameSeed=True,pxMax=None):

        #return a swarm object who position and momentum values have been randomly generated inside a phase space hypercube.
        #A seed can be reused to getrepeatable random results
        #pxMax: what value to use for longitudinal momentum spread. if None use the nominal value
        #qMax: maximum dimension in position space of hypercube
        #pMax: maximum dimension in momentum space of hupercube
        #num: number of particle in phase space
        #upperSymmetry: wether to exploit lattice symmetry by only using particles in upper half plane
        #sameSeed: wether to use the same seed eveythime in the nump random generator, the number 42, or a new one
        #return: a swarm object
        swarm = self.initalize_Random_Swarm_In_Phase_Space(qMax, pMax, num, upperSymmetry=upperSymmetry,
                                                           sameSeed=sameSeed)
        R = self.lattice.combiner.RIn
        r2 = self.lattice.combiner.r2
        for particle in swarm.particles:
            particle.q[:2] = particle.q[:2] @ R
            particle.q += r2
            particle.p[:2] = particle.p[:2] @ R
            particle.q = particle.q + particle.p * 1e-12  # scoot particle into next element
        return swarm

    def generate_Probe_Sample(self,v0,seed=None,rpPoints=3,rqPoints=3):
        # value of 3 for points is a good value from testing to give qualitative results
        if type(seed)==int:
            np.random.seed(seed)
        pMax=10.0
        numParticlesArr=np.arange(4,4*(rpPoints+1),4)
        coords=np.asarray([[0,0]])
        for numParticles in numParticlesArr:
            r=pMax*numParticles/numParticlesArr.max()
            phiArr=np.linspace(0,2*np.pi,numParticles,endpoint=False)
            tempCoords=np.column_stack((r*np.cos(phiArr),r*np.sin(phiArr)))
            coords=np.row_stack((coords,tempCoords))
        pSamples=np.column_stack((-np.ones(coords.shape[0])*v0,coords))
        # plt.scatter(pSamples[:,1],pSamples[:,2])
        # plt.show()

        # create position samples

        qMax=2.5e-3
        numParticlesArr=np.arange(4,4*(rqPoints+1),4)
        coords=np.asarray([[0,0]])
        for numParticles in numParticlesArr:
            r=qMax*numParticles/numParticlesArr.max()
            phiArr=np.linspace(0,np.pi,numParticles,endpoint=True)
            tempCoords=np.column_stack((r*np.cos(phiArr),r*np.sin(phiArr)))
            coords=np.row_stack((coords,tempCoords))
        qSamples=np.column_stack((-np.zeros(coords.shape[0]),coords))
        # plt.scatter(qSamples[:,1],qSamples[:,2])
        # plt.show()

        swarm=Swarm()
        for qCoord in qSamples:
            for pCoord in pSamples:
                if qCoord[2]==0 and pCoord[2]<0:  # exploit symmetry along z=0 for momentum by exluding downard
                    # directed points
                    pass
                else:
                    swarm.add_Particle(qi=qCoord.copy(),pi=pCoord.copy())
        if type(seed)==int:
            np.random.seed(int(time.time()))
        # print('Number of particles generated: ', swarm.num_Particles())
        return swarm
    def move_Swarm_To_Combiner_Output(self,swarm,scoot=True,copySwarm=True):
        #take a swarm where at move it to the combiner's output. Swarm should be created such that it is centered at
        #(0,0,0) and have average negative velocity. Any swarm can work however, but the previous condition is assumed.
        #swarm: the swarm to move to output
        #scoot: if True, move the particles along a tiny amount so that they are just barely in the next element. Helpful
        #for the doing the particle tracing sometimes
        if copySwarm==True:
            swarm=swarm.copy()

        R = self.lattice.combiner.RIn #matrix to rotate into combiner frame
        r2 = self.lattice.combiner.r2 #position of the outlet of the combiner
        for particle in swarm.particles:
            particle.q[:2] = particle.q[:2] @ R
            particle.q += r2
            particle.p[:2] = particle.p[:2] @ R
            if scoot==True:
                particle.q+=particle.p*1e-10
        return swarm

    def catch_Injection_Errors(self, Li, LOffset):
        deltaL=Li+LOffset-self.lattice.combiner.Lo #distance of magnet from beginning
        #of combiner. Must be greater than at least zero
        if deltaL<.1: #if the magnet is closer than 10 cm
            print(Li,LOffset,deltaL,self.lattice.combiner.Lo)
            raise Exception('Magnet is too close to combiner')

    def trace_Swarm_Through_Lattice(self,swarm,h,T,parallel=True,fastMode=True,copySwarm=True):
        #trace a swarm through the lattice
        if copySwarm==True:
            swarmNew=swarm.copy()
        else:
            swarmNew=swarm
        if parallel==True:
            def func(particle):
                return self.particleTracer.trace(particle, h, T,fastMode=fastMode)
            results = self.helper.parallel_Chunk_Problem(func, swarmNew.particles)
            for i in range(len(results)): #replaced the particles in the swarm with the new traced particles. Order
                #is not important
                swarmNew.particles[i]=results[i][1]
        else:
            for i in range(swarmNew.num_Particles()):
                swarmNew.particles[i]=self.particleTracer.trace(swarmNew.particles[i],h,T,fastMode=fastMode)


        return swarmNew