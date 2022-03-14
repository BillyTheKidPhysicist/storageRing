from collections.abc import Iterable
from ParticleTracerClass import ParticleTracer
import numpy as np
from ParticleClass import Swarm,Particle
import time
import skopt
import multiprocess as mp
from constants import BOLTZMANN_CONSTANT,MASS_LITHIUM_7
from ParticleTracerLatticeClass import ParticleTracerLattice
from typing import Union,Optional


def lorentz_Function(x,gamma):
    #returns a value of 1.0 for x=0
    return (gamma/2)**2/(x**2+(gamma/2)**2)
def normal(v,sigma,v0=0.0):
    return np.exp(-.5*((v-v0)/sigma)**2)




class SwarmTracer:

    def __init__(self,lattice:ParticleTracerLattice):
        self.lattice=lattice
        self.particleTracer = ParticleTracer(self.lattice)

    def Rd_Sample(self,n: int,d: int=1,seed: float=.5)-> np.ndarray:
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
    def initialize_Stablity_Testing_Swarm(self,qMax: float)-> Swarm:
        smallOffset=-1e-10 #this prevents setting a particle right at a boundary which is takes time to sort out
        swarmTest = Swarm()
        swarmTest.add_Particle(qi=np.asarray([smallOffset,0.0,0.0]))
        swarmTest.add_Particle(qi=np.asarray([smallOffset, qMax/2, qMax/2]))
        swarmTest.add_Particle(qi=np.asarray([smallOffset, -qMax/2, qMax/2]))
        swarmTest.add_Particle(qi=np.asarray([smallOffset, qMax/2, -qMax/2]))
        swarmTest.add_Particle(qi=np.asarray([smallOffset, -qMax/2, -qMax/2]))
        return swarmTest
    def initialize_HyperCube_Swarm_In_Phase_Space(self, qMax: np.ndarray, pMax: np.ndarray, numGridEdge: int,
                                                  upperSymmetry: bool=False)-> Swarm:
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
    def initialize_Observed_Collector_Swarm_Probability_Weighted(self,captureDiam: float,collectorOutputAngle: float,
                                    numParticles: float,gammaSpace: float=3.5e-3,temperature: float=.003,
                                    sameSeed: bool=False,upperSymmetry: bool=False,probabilityMin: float=0.01)-> Swarm:
        #this function generates a swarm that models the observed swarm. This is done by first generating a pseudorandom
        #swarm that is well spread out in space, then weighitng each particle by it's probability according to the
        #observed data. The probability is finally rescaled
        #captureDiam: Diameter of the circle of atoms we wish to collect, meters
        #collectorOutputAngle: Maximum angle of atoms leaving the collector, radians
        #numParticles: Number of particles to sample. Will not always equal exactly this
        #gammaSpace: The FWHM of the lorentz function that models our spatial data, meters
        #temperature: The temperature of the atoms, kelvin. Decides thermal velocity spread

        assert 0.0<captureDiam<=.1 and 0.0<collectorOutputAngle<=.2 and 0.0<gammaSpace<=.01\
               and 0.0<temperature<=.1 and probabilityMin>=0.0#reasonable values

        pTransMax=self.lattice.v0Nominal*np.tan(collectorOutputAngle) #transverse velocity dominates thermal velocity,
        #ie, geometric heating
        sigmaVelocity=np.sqrt(BOLTZMANN_CONSTANT*temperature/MASS_LITHIUM_7) #thermal velocity spread. Used for
        #longitudinal velocity only because geometric dominates thermal in transverse dimension
        pLongitudinalMin=-1e-3
        pLongitudinalMax=1e-3
        pLongBounds=(pLongitudinalMin,pLongitudinalMax)
        swarmEvenlySpread=self.initalize_PseudoRandom_Swarm_In_Phase_Space(captureDiam/2.0,pTransMax,pLongBounds,
                                                            numParticles,sameSeed=sameSeed,upperSymmetry=upperSymmetry)
        probabilityList=[]
        for particle in swarmEvenlySpread:
            probability=1.0
            x,y,z=particle.qi
            r=np.sqrt(y**2+z**2) #remember x is longitudinal
            px,py,pz=particle.pi
            probability=probability*lorentz_Function(r,gammaSpace) #spatial probability
            pTrans=np.sqrt(py**2+pz**2)
            px=-np.sqrt(self.lattice.v0Nominal**2-pTrans**2)
            particle.pi[0]=px
            assert probability<1.0
            probabilityList.append(probability)
        swarmObserved = Swarm()
        peakProbability=max(probabilityList) # I think this is unnesesary
        for particle,probability in zip(swarmEvenlySpread.particles,probabilityList):
            particle.probability=probability/peakProbability
            if particle.probability>probabilityMin:
                swarmObserved.particles.append(particle)
        return swarmObserved

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
        reSeedVal=np.random.get_state()[1][0]
        if sameSeed==True:
            np.random.seed(42)
        elif type(sameSeed) == int:
            np.random.seed(sameSeed)
        else: raise ValueError

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
            np.random.seed(reSeedVal)  # re randomize
        return swarm

    def initialize_Point_Source_Swarm(self,sourceAngle: float,numParticles: int,smallXOffset: bool=True,
                                      sameSeed:bool =False)-> Swarm:
        p0=self.lattice.v0Nominal #the momentum of each particle
        qTBounds,pxBounds=1e-12,1e-12 #force to a point spatialy, and no speed spread
        pTBounds=np.tan(sourceAngle)*p0
        swarmPseudoRandom=self.initalize_PseudoRandom_Swarm_In_Phase_Space(qTBounds, pTBounds, pxBounds, numParticles,
                                sameSeed=sameSeed, circular=True, smallXOffset=smallXOffset)
        for particle in swarmPseudoRandom:
            px,py,pz=particle.pi
            px=-np.sqrt(p0**2-(py**2+pz**2))
            particle.pi=np.asarray([px,py,pz])
        return swarmPseudoRandom
    def initialize_Ring_Swarm(self,angle,num):
        assert 0.0<angle<np.pi/2
        pr=np.tan(angle)*self.lattice.v0Nominal
        thetaArr=np.linspace(0.0,2*np.pi,num+1)[:-1]
        swarm=Swarm()
        for theta in thetaArr:
            swarm.add_Particle(pi=np.asarray([-self.lattice.v0Nominal,pr*np.cos(theta),pr*np.sin(theta)]))
        return swarm

    def initalize_PseudoRandom_Swarm_At_Combiner_Output(self,qTBounds,pTBounds,pxBounds,numParticles,upperSymmetry=False,
                                                        sameSeed=False,circular=True,smallXOffset=True):
        swarmAtOrigin=self.initalize_PseudoRandom_Swarm_In_Phase_Space(qTBounds,pTBounds,pxBounds,numParticles,upperSymmetry=upperSymmetry,
                                                               sameSeed=sameSeed,circular=circular,smallXOffset=smallXOffset)
        swarmAtCombiner=self.move_Swarm_To_Combiner_Output(swarmAtOrigin,copySwarm=False,scoot=True)
        return swarmAtCombiner

    def combiner_Output_Offset_Shift(self)-> np.ndarray:
        #combiner may have an output offset (ie hexapole combiner). This return the 3d vector (x,y,0) that connects the
        #geoemtric center of the output plane with the offset point, which also lies in the plane. stern gerlacht
        #style doesn't have and offset
        n2=self.lattice.combiner.ne.copy() #unit normal to outlet
        np2=-np.asarray([n2[1],-n2[0],0.0]) # unit parallel to outlet
        return np2*self.lattice.combiner.outputOffset

    def move_Swarm_To_Combiner_Output(self,swarm: Swarm,scoot: bool=False,copySwarm: bool=True)-> Swarm:
        #take a swarm where at move it to the combiner's output. Swarm should be created such that it is centered at
        #(0,0,0) and have average negative velocity.
        #swarm: the swarm to move to output
        #scoot: if True, move the particles along a tiny amount so that they are just barely in the next element. Helpful
        #for the doing the particle tracing sometimes
        if copySwarm==True:
            swarm=swarm.copy()

        R = self.lattice.combiner.RIn.copy() #matrix to rotate into combiner frame
        r2 = self.lattice.combiner.r2.copy() #position of the outlet of the combiner
        r2+=self.combiner_Output_Offset_Shift()

        for particle in swarm.particles:
            particle.qi[:2] = particle.qi[:2] @ R
            particle.qi += r2
            particle.pi[:2] = particle.pi[:2] @ R
            if scoot==True:
                tinyTimeStep=1e-9
                particle.qi+=particle.pi*tinyTimeStep
        return swarm

    def _super_Fast_Trace(self,swarm: Swarm,trace_Particle_Function)-> Swarm:
        # use trick of accessing only the important class variabels and passing those through to reduce pickle time
        def fastFunc(compactDict):
            particle = Particle()
            for key, val in compactDict.items():
                setattr(particle, key, val)
            particle = trace_Particle_Function(particle)
            compactDictTraced = {}
            for key, val in vars(particle).items():
                if val is not None:
                    compactDictTraced[key] = val
            return compactDictTraced
        compactDictList = []
        for particle in swarm:
            compactDict = {}
            for key, val in vars(particle).items():
                if val is not None:
                    if not (isinstance(val, Iterable) and len(val) == 0):
                        compactDict[key] = val
            compactDictList.append(compactDict)
        with mp.Pool(mp.cpu_count()) as Pool:
            compactDictTracedList = Pool.map(fastFunc, compactDictList)
        for particle, compactDict in zip(swarm.particles, compactDictTracedList):
            for key, val in compactDict.items():
                setattr(particle, key, val)
        return swarm

    def trace_Swarm_Through_Lattice(self, swarm: Swarm, h: float, T: float, parallel: bool=False, fastMode: bool=True,
                                    copySwarm: bool=True, accelerated: bool=False,stepsBetweenLogging: int=1,
                                    energyCorrection: bool=False,tau_Collision: Optional[float]=None)-> Swarm:
        if copySwarm == True:
            swarmNew = swarm.copy()
        else:
            swarmNew = swarm
        def trace_Particle(particle):
            particleNew = self.particleTracer.trace(particle, h, T, fastMode=fastMode, accelerated=accelerated,
            stepsBetweenLogging=stepsBetweenLogging,energyCorrection=energyCorrection,tau_Collision=tau_Collision)
            return particleNew
        if parallel=='superfast':
            #use trick of accessing only the important class variabels and passing those through. about 30%
            #faster
            swarmNew=self._super_Fast_Trace(swarmNew,trace_Particle)
            return swarmNew
        elif parallel==True:
            #more straightforward method. works for serial as well
            with mp.Pool(mp.cpu_count()) as pool:
                swarmNew.particles = pool.map(trace_Particle, swarmNew.particles)
            return swarmNew
        else:
            swarmNew.particles=[trace_Particle(particle) for particle in swarmNew]
            return swarmNew