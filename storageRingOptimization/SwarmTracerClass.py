import joblib
from joblib import Parallel,delayed
from ParticleTracerClass import ParticleTracer
import numpy as np
from ParticleClass import Swarm
import time
from ParaWell import ParaWell
import skopt
import multiprocess



def lorentz_Function(x,gamma):
    #returns a value of 1.0 for x=0
    return (gamma/2)**2/(x**2+(gamma/2)**2)
def normal(v,sigma,v0=0.0):
    return np.exp(-.5*((v-v0)/sigma)**2)




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
    def initialize_Stablity_Testing_Swarm(self,qMax):
        swarmTest = Swarm()
        swarmTest.add_Particle(qi=np.asarray([-1e-10,0.0,0.0]))
        swarmTest.add_Particle(qi=np.asarray([-1e-10, qMax/2, qMax/2]))
        swarmTest.add_Particle(qi=np.asarray([-1e-10, -qMax/2, qMax/2]))
        swarmTest.add_Particle(qi=np.asarray([-1e-10, qMax/2, -qMax/2]))
        swarmTest.add_Particle(qi=np.asarray([-1e-10, -qMax/2, -qMax/2]))
        return swarmTest
    def initialize_HyperCube_Swarm_In_Phase_Space(self, qMax, pMax, numGridEdge, upperSymmetry=False):
        # create a cloud of particles in phase space at the origin. In the xy plane, the average velocity vector points
        # to the west. The transverse plane is the yz plane.
        # qMax: absolute value maximum position in the transverse direction
        # qMax: absolute value maximum position in the transverse momentum
        # num: number of samples along each axis in phase space. Total is num^4
        # upperSymmetry: if this is true, exploit the symmetry between +/-z and ignore coordinates below z=0
        qArr = np.linspace(-qMax, qMax, num=numGridEdge)
        pArr = np.linspace(-pMax, pMax, num=numGridEdge)
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
    def initialize_Observed_Collector_Swarm_Probability_Weighted(self,captureDiam,collectorOutputAngle,numParticles,
                    gammaSpace=3.5e-3,temperature=.003,sameSeed=False,upperSymmetry=False,equalProbability=False):
        #this function generates a swarm that models the observed swarm. This is done by first generating a pseudorandom
        #swarm that is well spread out in space, then weighitng each particle by it's probability according to the
        #observed data. The probability is finally rescaled
        #captureDiam: Diameter of the circle of atoms we wish to collect, meters
        #collectorOutputAngle: Maximum angle of atoms leaving the collector, radians
        #numParticles: Number of particles to sample. Will not always equal exactly this
        #gammaSpace: The FWHM of the lorentz function that models our spatial data, meters
        #temperature: The temperature of the atoms, kelvin. Decides thermal velocity spread

        assert 0.0<captureDiam<=.1 and 0.0<collectorOutputAngle<=.2 and 0.0<gammaSpace<=.01\
               and 0.0<temperature<=.1 #reasonable values

        pTransMax=self.lattice.v0Nominal*np.tan(collectorOutputAngle) #transverse velocity dominates thermal velocity,
        #ie, geometric heating
        sigmaVelocity=np.sqrt(self.lattice.kb*temperature/self.lattice.mass_Li7) #thermal velocity spread. Used for
        #longitudinal velocity only because geometric dominates thermal in transverse dimension
        pLongitudinalMax=2*sigmaVelocity+(1-np.cos(collectorOutputAngle))*self.lattice.v0Nominal #include 2 sigma of
        #velocity, and geometric spread
        swarmEvenlySpread=self.initalize_PseudoRandom_Swarm_In_Phase_Space(captureDiam/2.0,pTransMax,pLongitudinalMax,
                                                            numParticles,sameSeed=sameSeed,upperSymmetry=upperSymmetry)
        if equalProbability==True: #don't apply the observed swarm characteristicssss
            return swarmEvenlySpread
        probabilityList=[]
        for particle in swarmEvenlySpread:
            probability=1.0
            x,y,z=particle.qi
            r=np.sqrt(y**2+z**2) #remember x is longitudinal
            px,py,pz=particle.pi
            probability=probability*lorentz_Function(r,gammaSpace) #spatial probability
            pTrans=np.sqrt(py**2+pz**2)
            pxMean=-np.sqrt(self.lattice.v0Nominal**2-pTrans**2)
            probability=probability*normal(px,sigmaVelocity,v0=pxMean)
            probabilityList.append(probability)
        peakProbability=max(probabilityList)
        for particle,probability in zip(swarmEvenlySpread.particles,probabilityList):
            particle.probability=probability/peakProbability
            assert 0.0<particle.probability<=1.0
        return swarmEvenlySpread
    def _make_PseudoRandom_Swarm_Bounds_List(self,qTBounds,pTBounds,pxBounds,upperSymmetry=False):
        if isinstance(qTBounds,float):
            assert qTBounds>0.0
            if upperSymmetry==False: qTBounds=[(-qTBounds,qTBounds),(-qTBounds,qTBounds)]
            else: qTBounds=[(-qTBounds,qTBounds),(0.0,qTBounds)] #restrict to top half
        else: assert len(qTBounds)==2 and len(qTBounds[0])==2 and len(qTBounds[1])==2
        if isinstance(pTBounds,float):
            assert pTBounds>0.0
            pTBounds=[(-pTBounds,pTBounds),(-pTBounds,pTBounds)]
        else: assert len(pTBounds)==2 and len(pTBounds[0])==2 and len(pTBounds[1])==2
        if isinstance(pxBounds,float):
            assert pxBounds>0.0
            pxBounds=(-pxBounds-self.lattice.v0Nominal,pxBounds-self.lattice.v0Nominal)
        else:
            assert len(pxBounds)==2
            pxBounds=(pxBounds[0]-self.lattice.v0Nominal,pxBounds[1]-self.lattice.v0Nominal)
        generatorBounds=qTBounds.copy()
        generatorBounds.append(pxBounds)
        generatorBounds.extend(pTBounds)
        pxMin,pxMax=generatorBounds[2]
        assert len(generatorBounds)==5 and pxMin<-self.lattice.v0Nominal<pxMax
        return generatorBounds

    def initalize_PseudoRandom_Swarm_In_Phase_Space(self,qTBounds,pTBounds,pxBounds,numParticles,upperSymmetry=False,
                                                    sameSeed=False,circular=True,smallXOffset=True):
        #return a swarm object who position and momentum values have been randomly generated inside a phase space hypercube
        #and that is heading in the negative x direction with average velocity lattice.v0Nominal. A seed can be reused to
        #get repeatable random results. a sobol sequence is used that is then jittered. In additon points are added at
        #each corner exactly and midpoints between corners if desired
        #NOTE: it's not guaranteed that there will be exactly num particles.
        if circular==True:
            assert isinstance(qTBounds,float) and qTBounds>0.0 and isinstance(pTBounds,float) and pTBounds>0.0
            qTransMax=qTBounds
            pTransMax=pTBounds
        generatorBounds=self._make_PseudoRandom_Swarm_Bounds_List(qTBounds,pTBounds,pxBounds,upperSymmetry=upperSymmetry)


        if circular is True:
            numParticlesfrac=1/((np.pi/4)**2) #the ratio of the are of the circle to the cross section. There is one
            #factor for momentum and one for position
        else:
            numParticlesfrac=1.0
        if sameSeed==True:
            np.random.seed(42)
        if type(sameSeed) == int:
            np.random.seed(sameSeed)

        swarm = Swarm()
        sampler=skopt.sampler.Sobol()
        samples=np.asarray(sampler.generate(generatorBounds,int(numParticles*numParticlesfrac)))
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
                if np.sqrt(y**2+z**2)<qTransMax and np.sqrt(py**2+pz**2)<pTransMax:
                    swarm.add_Particle(qi=q, pi=p)
                    particleCount+=1
                if particleCount==numParticles:
                    break
            else:
                swarm.add_Particle(qi=q,pi=p)
        if sameSeed==True or type(sameSeed)==int:
            np.random.seed(int(time.time()))  # re randomize
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
        return swarm
    def initalize_PseudoRandom_Swarm_At_Combiner_Output(self,qTBounds,pTBounds,pxBounds,numParticles,upperSymmetry=False,
                                                        sameSeed=False,circular=True,smallXOffset=True):
        swarmAtOrigin=self.initalize_PseudoRandom_Swarm_In_Phase_Space(qTBounds,pTBounds,pxBounds,numParticles,upperSymmetry=upperSymmetry,
                                                               sameSeed=sameSeed,circular=circular,smallXOffset=smallXOffset)
        swarmAtCombiner=self.move_Swarm_To_Combiner_Output(swarmAtOrigin,copySwarm=False,scoot=True)
        return swarmAtCombiner

    def move_Swarm_To_Combiner_Output(self,swarm,scoot=False,copySwarm=True):
        #take a swarm where at move it to the combiner's output. Swarm should be created such that it is centered at
        #(0,0,0) and have average negative velocity.
        #swarm: the swarm to move to output
        #scoot: if True, move the particles along a tiny amount so that they are just barely in the next element. Helpful
        #for the doing the particle tracing sometimes
        if copySwarm==True:
            swarm=swarm.copy()

        R = self.lattice.combiner.RIn #matrix to rotate into combiner frame
        r2 = self.lattice.combiner.r2 #position of the outlet of the combiner
        for particle in swarm.particles:
            assert np.abs(particle.qi[0])<1e-12 and particle.pi[0]<0.0 and particle.traced==False
            particle.qi[:2] = particle.qi[:2] @ R
            particle.qi += r2
            particle.pi[:2] = particle.pi[:2] @ R
            if scoot==True:
                tinyTimeStep=1e-9
                particle.qi+=particle.pi*tinyTimeStep
        return swarm

    def trace_Swarm_Through_Lattice(self,swarm,h,T,parallel=True,fastMode=True,copySwarm=True,accelerated=False):
        #trace a swarm through the lattice
        if copySwarm==True:
            swarmNew=swarm.copy()
        else:
            swarmNew=swarm
        if parallel==True:
            def func(particle):
                return self.particleTracer.trace(particle, h, T,fastMode=fastMode,accelerated=accelerated)
            results = self.helper.parallel_Chunk_Problem(func, swarmNew.particles)
            for i in range(len(results)): #replaced the particles in the swarm with the new traced particles. Order
                #is not important
                swarmNew.particles[i]=results[i][1]
        else:
            for i in range(swarmNew.num_Particles()):
                swarmNew.particles[i]=self.particleTracer.trace(swarmNew.particles[i],h,T,fastMode=fastMode
                                                                ,accelerated=accelerated)
        return swarmNew