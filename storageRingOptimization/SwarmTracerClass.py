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
                                                    cornerPoints=False,circular=True):
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
        if self.lattice.latticeType=='injector': #lattice is towards positive x instead of towards negative x so need
            #to flip sign and order of bounds
            a=-bounds[2][0]
            b=-bounds[2][1]
            bounds[2][0]=b
            bounds[2][1]=a


        if circular is True:
            frac=(np.pi/4)**2 #the ratio of the are of the circle to the cross section. There is one
            #factor for momentum and one for position
            numParticles=int(numParticles/frac)


        if sameSeed==True:
            np.random.seed(42)
        if type(sameSeed) == int:
            np.random.seed(sameSeed)
        if upperSymmetry==True:
            bounds[1][0]=0.0 #don't let point be generarted below z=0

        x=0.0 #all particles have same x position of 0,0
        swarm = Swarm()
        if cornerPoints==True: #placing points at corner of hypercube
            XiArr=np.asarray(np.meshgrid(*bounds)).T.reshape(-1,bounds.shape[0])
            for Xi in XiArr:
                q = np.append(x, Xi[:2])
                p = Xi[2:]
                swarm.add_Particle(q, p)
            numParticles-=swarm.num_Particles()
            if numParticles<=0:
                raise Exception('adding particles at the corners of the hypercube exceeded total amount of allowed'
                                'particles')


        # samples=self.Rd_Sample(numParticles,len(bounds),seed=np.random.random()) #generate low noise sample
        # #now need to scale correctly, because returned values are from 0 to 1
        # for i in range(len(bounds)):
        #     upper=bounds[i][1]
        #     lower=bounds[i][0]
        #     samples[:,i]=samples[:,i]*(upper-lower)+lower #rescale and shift

        sampler=skopt.sampler.Sobol()
        samples=np.asarray(sampler.generate(bounds,numParticles))

        for Xi in samples:
            q = np.append(x, Xi[:2])
            p = Xi[2:]
            if circular==True:
                y,z,py,pz=Xi[[0,1,3,4]]
                if np.sqrt(y**2+z**2)<qTMax and np.sqrt(py**2+pz**2)<pTMax:
                    swarm.add_Particle(qi=q, pi=p)
                else:
                    pass
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


    def inject_Swarm(self,swarm,Lo,Li,LOffset,labFrame=True,h=1e-5,parallel=False,fastMode=True,copySwarm=True):
        # this traces a swarm through the shaper and into the output of the combiner. The swarm is traced
        # throguh the combiner assuming it is a swarm being loaded, not already circulating. The returned particles are in
        # the lab or combiner frame
        # Here the output refers to the origin in the combiner's reference frame.
        # Li: image length, ie distance from end of lens to focus
        # Lo: object distance
        # LOffset: length that the image distance is offset from the combiner output. Positive
        # value corresponds coming to focus before the output (ie, inside the combiner), negative
        # is after the output, ie outside after the output
        # qMax: absolute vale of maximum value in space dimensions
        # pMax: absolute vale of maximum value in momentum dimensions
        # numParticles: number of particles along each axis, total number is axis**n where n is dimensions
        # upperSymmetry: wether to exploit the upper symmetry of the lattice and ignore particles with z<0
        # labFrame: wether to return particles in the labframe, or in the element frame. True for lab, False for element
        # fastMode: wether to log particle paramters while tracing
        #copySwarm: if True, then make a copy of the swarm to avoid editing the original
        if copySwarm==True:
            swarm=swarm.copy()
        swarm = self.send_Swarm_Through_Shaper(swarm, Lo, Li, copySwarm=False)
        swarm=self.aim_Swarm_At_Combiner(swarm,Li,LOffset,copySwarm=False)

        # now I need to trace and position the swarm at the output. This is done by moving the swarm along with combiner to the
        # combiner's outlet in it's final position.
        r0 = self.lattice.combiner.r2
        R = self.lattice.combiner.ROut
        def func(particle):
            particle = self.step_Particle_Through_Combiner(particle,h=h,fastMode=fastMode) #particle has now been traced
            if labFrame==True:
               particle.q[:2] = R@particle.q[:2] + r0[:2] #now transform the lab frame
               particle.p[:2]=R@particle.p[:2]
               if fastMode==False: #transform tracked coordinates as well
                   for j in range(particle.qArr.shape[0]):
                        particle.qArr[j,:2] = R @ particle.qArr[j,:2] + r0[:2]  # now transform the lab frame
                        particle.pArr[j,:2] = R @ particle.pArr[j,:2]
            return particle
        if parallel == False:
            for i in range(swarm.num_Particles()):
                swarm.particles[i] = func(swarm.particles[i])
        elif parallel == True:
            results = self.helper.parallel_Chunk_Problem(func, swarm.particles)
            for i in range(len(results)):  # replaced the particles in the swarm with the new traced particles. Order
                # is not important
                swarm.particles[i] = results[i][1]

        return swarm
    def initialize_Swarm_At_Combiner_Output(self, Lo, Li, LOffset,  numParticles=1000, parallel=False,labFrame=True,
                                            h=1e-5,clipForNextApeture=False,fastMode=True):
        # this method generates a cloud of particles in phase space at the output of the combiner. The cloud is traced
        #throguh the combiner assuming it is a cloud being loaded, not already circulating. The returned particles are in
        #the lab or combiner frame
        # Here the output refers to the origin in the combiner's reference frame.
        # Li: image length, ie distance from end of lens to focus
        # Lo: object distance
        # LOffset: length that the image distance is offset from the combiner output. Positive
        # value corresponds coming to focus before the output (ie, inside the combiner), negative
        # is after the output.
        # numParticles: number of particles along each axis, total number is axis**n where n is dimensions
        # labFrame: wether to return particles in the labframe, or in the element frame. True for lab, False for element
        # clipForNextApeture: If true, clip particles if they are not respecting the apeture imediately following the combiner
        #fastMode: wehter to log particle parameters while tracing
        self.catch_Injection_Errors(Li, LOffset)
        swarm=self.initialize_Observed_Swarm_In_Phase_Space(numParticles=numParticles)
        swarm=self.inject_Swarm(swarm,Lo,Li,LOffset,labFrame=labFrame,h=h,parallel=parallel,fastMode=fastMode,copySwarm=False)
        if clipForNextApeture==True: #if checking to see if particle passes the next apeture
            ap=self.lattice.elList[self.lattice.combinerIndex+1].ap#get the next element's apeture. assumed to be
            #cylinderival/square
            r0 = self.lattice.combiner.r2
            R = self.lattice.combiner.RIn
            for particle in swarm:
                if particle.clipped==False: #only bother if the particle isn't already clipped
                    q = particle.q
                    if labFrame==True:
                        q=q.copy() # to avoid editing the original
                        q[:2]=R@(q[:2]-r0[:2]) #transform to element frame to make checking easy
                    if np.sqrt(q[1]**2+q[2]**2)>ap: #if particle is NOT inside the apeture
                        particle.clipped=True
        return swarm

    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:], numba.float64))
    def fast_qNew(q, F, p, h):
        #this is a numbafied version. It makes an appreciable differenence in performance
        #q: position, n
        #F: force, n
        #p: momentum, n
        #h: timestep
        #return q_n+1
        return q + p * h + .5 * F * h ** 2

    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:], numba.float64))
    def fast_pNew(p, F, F_New, h):
        # this is a numbafied version. It makes an appreciable differenence if performance
        # q: position, n
        # F: force, n
        #F_New: force, n+1
        # h: timestep
        # return p_n+1
        return p + .5 * (F + F_New) * h
    def step_Particle_Through_Combiner(self, particle, h=1e-5,fastMode=False):
        #this method traces the particle through the combiner. First the particle is moved up the combiner input. Then
        #it is checked if it clipping in which case the algorithm is stopped. The particle is then traced through the
        #combiner checking for clipping at each point.
        particle.currentEl = self.lattice.combiner
        q = particle.q
        p = particle.p
        if fastMode==False:
            particle.log_Params()

        #There is a line parallel to the input of the cominer on the loading side. Walk the particles up to that point.
        m=-1/np.tan(self.lattice.combiner.angLoad)
        xi=(self.lattice.combiner.space * 2 + self.lattice.combiner.Lm)
        yi=self.lattice.combiner.inputOffsetLoad
        b=yi-m*xi
        dt=(m*q[0]+b-q[1])/(p[1]-m*p[0])
        q=q+dt*p #particle is walked up to the line now
        particle.q = q
        if fastMode==False:
            particle.log_Params()


        apL = self.lattice.combiner.apL
        apR = self.lattice.combiner.apR
        apz = self.lattice.combiner.apz
        def is_Clipped(q):
            #return True if the particle is clipped. remember combiner apeture is not cylinderical
            if not -apz<q[2]<apz:  # if clipping in z direction
                return True
            elif q[0]<self.lattice.combiner.Lm+self.lattice.combiner.space: #only check if the particle is in the straight
                #section
                if not -apL<q[1]<apR:
                    return True
            else:
                return False

        if is_Clipped(q) == True:
            particle.traced=True
            particle.clipped = True
            particle.finished()
            return particle
        forcePrev=None
        while (True):
            if forcePrev is not None:
                F = forcePrev
            else:
                F = -self.lattice.combiner.force(q)  # force is negative for high field seeking
            q_n = self.fast_qNew(q, F, p, h)  # q + p * h + .5 * F * h ** 2
            if q_n[0] < 0:  # if overshot, go back and walk up to the edge assuming no force
                dr = - q[0]
                dt = dr / p[0]
                q = q + p * dt
                particle.q = q
                particle.p = p
                break
            F_n = -self.lattice.combiner.force(q_n)  # force is negative for high field seeking
            p_n = self.fast_pNew(p, F, F_n, h)  # p + .5 * (F + F_n) * h
            q = q_n
            p = p_n
            forcePrev = F_n
            particle.q = q
            particle.p = p
            if fastMode==False:
                particle.log_Params()

            if is_Clipped(q) == True:
                particle.clipped = True
                break
        if particle.clipped is None:
            particle.clipped = False
        particle.traced = True
        particle.finished()
        return particle

    def aim_Swarm_At_Combiner(self, swarm, seedingDist, LOffset,copySwarm=True):
        # This method takes a swarm in phase space, located at the origin, and moves it in phase space
        # so momentum vectors point at the combiner input. This is done IN THE COMBINER REFERENCE FRAME
        # swarm: swarm of particles in phase space. must be centered at the origin in space
        # seedingDist: Distance between swarm and loading point of combiner. keep in mind seeding point can move by LOffset
        # LiOffset: distance from combiner output of seeding point. see initialize_Swarm_At_Combiner_Output
        # copySwarm: if True, copy the swarm to prevent modifying the original one
        if copySwarm==True:
            swarm = swarm.copy()  # don't change the original swarm
        inputOffsetLoad = self.lattice.combiner.inputOffsetLoad
        inputAngleLoad = self.lattice.combiner.angLoad
        dL = seedingDist - self.lattice.combiner.Lo + LOffset #distance from combiner input
        dx = self.lattice.combiner.space * 2 + self.lattice.combiner.Lm
        dy = inputOffsetLoad
        dx += dL * np.cos(inputAngleLoad)
        dy += dL * np.sin(inputAngleLoad)
        dR = np.asarray([dx, dy])
        rotMat = np.asarray([[np.cos(inputAngleLoad), -np.sin(inputAngleLoad)], [np.sin(inputAngleLoad), np.cos(inputAngleLoad)]])
        for particle in swarm.particles:
            q = particle.q
            p = particle.p
            q[:2] = rotMat @ q[:2]+dR
            p[:2] = rotMat @ p[:2]
            particle.q = q
            particle.p = p
        return swarm
    def compute_Survival_Through_Lattice(self,Lo,Li,LOffset,F1,F2,h,T,parallel=False):
        qMax=2.5e-3
        pMax=3e1
        numParticles=9
        swarm=self.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, qMax, pMax, numParticles=numParticles,parallel=parallel,upperSymmetry=True)
        self.lattice.elList[2].fieldFact=F1
        self.lattice.elList[4].fieldFact=F2

        swarm=self.trace_Swarm_Through_Lattice(swarm,h,T,parallel=parallel)
        return swarm.survival_Rev()
    def optimize_Swarm_Survival_Through_Lattice_Brute(self,bounds,numPoints,T,h=1e-5):
        #bounds: list of tuples for bounds on F1 and F2
        def func(X):
            F1,F2=X
            swarm = Swarm()
            qi = np.asarray([-1e-10, 0.0, 0.0])
            pi = np.asarray([-200.0, 0, 0])
            swarm.add_Particle(qi, pi)
            self.lattice.elList[2].fieldFact = F1
            self.lattice.elList[4].fieldFact = F2
            swarm = self.trace_Swarm_Through_Lattice(swarm, h, T, parallel=False)
            swarm.survival_Rev()
            return swarm.survival_Rev()#1/(1+swarm.survival_Rev())


        #spo.differential_evolution(func,bounds,workers=-1,popsize=5,maxiter=10)
        F1Arr=np.linspace(bounds[0][0],bounds[0][1],num=numPoints)
        F2Arr = np.linspace(bounds[1][0], bounds[1][1], num=numPoints)
        argsArr = np.asarray(np.meshgrid(F1Arr, F2Arr)).T.reshape(-1, 2)
        results=self.helper.parallel_Chunk_Problem(func,argsArr)
        survivalList=[]
        argList=[]
        for results in results:
            survivalList.append(results[1])
            argList.append(results[0])
        argArr=np.asarray(argList)
        survivalArr=np.asarray(survivalList)
    def compute_Survival_Through_Injector(self, Lo,Li, LOffset,testNextElement=True,parallel=True):
        qMax, pMax, numParticles=3e-3,10.0,9
        swarm = self.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, qMax, pMax, numParticles,parallel=parallel)
        parallel=False
        if testNextElement==True:
            elNext=self.lattice.elList[self.lattice.combinerIndex+1]
            #need to scoot particle along
            h=2.5e-5
            T=elNext.Lo/self.lattice.v0Nominal
            if parallel==False:
                for i in range(swarm.num_Particles()):
                    particle=swarm.particles[i]
                    particle.q=particle.q+particle.p*1e-10 #scoot particle into next element
                    swarm.particles[i]=self.particleTracer.trace(particle,h,T,fastMode=True)
            else:
                def wrapper(particle):
                    return self.particleTracer.trace(particle, h, T, fastMode=True)
                results=self.helper.parallel_Chunk_Problem(wrapper,swarm.particles)
                for i in range(len(results)):
                    swarm.particles[i]=results[i][1]
                    swarm.particles[i].plot_Energies()


        return swarm.survival_Bool()
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