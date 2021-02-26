import skopt
import numba
from profilehooks import profile
from ParticleTracer import ParticleTracer
#import black_box as bb
import numpy.linalg as npl
#import matplotlib.pyplot as plt
import sys
import multiprocess as mp
from particleTracerLattice import ParticleTracerLattice
import numpy as np
from ParticleClass import Swarm
import scipy.optimize as spo
import time
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import pySOT as ps
from ParaWell import ParaWell
import poap.controller as pc


class skoptOptimizer:
    def __init__(self):
        pass



class LatticeOptimizer:
    def __init__(self, lattice):
        self.lattice = lattice
        self.helper=ParaWell() #custom class to help with parallelization
        self.i=0 #simple variable to track solution status
        self.particleTracer = ParticleTracer(lattice)
        self.swarmStability=None #particle swarm used to test for lattice stability

    def initialize_HyperCube_Swarm_In_Phase_Space(self, qMax, pMax, num, upperSymmetry=False):
        # create a cloud of particles in phase space at the origin. In the xy plane, the average velocity vector points
        # to the west. The transverse plane is the yz plane.
        #qMax: absolute value maximum position in the transverse direction
        #qMax: absolute value maximum position in the transverse momentum
        #num: number of samples along each axis in phase space
        #upperSymmetry: if this is true, exploit the symmetry between +/-z and ignore coordinates below z=0
        qArr = np.linspace(-qMax, qMax, num=num)
        pArr = np.linspace(-pMax, pMax, num=num)
        argsArr = np.asarray(np.meshgrid(qArr, qArr, pArr, pArr)).T.reshape(-1, 4)
        swarm = Swarm()
        for arg in argsArr:
            qi = np.asarray([0.0, arg[0], arg[1]])
            pi = np.asarray([-self.lattice.v0Nominal, arg[2], arg[3]])
            if upperSymmetry==True:
                if qi[2]<0:
                    pass
                else:
                    swarm.add_Particle(qi, pi)
            else:
                swarm.add_Particle(qi, pi)
        return swarm
    def initalize_Random_Swarm_In_Phase_Space(self, qMax, pMax, num, upperSymmetry=False,sameSeed=True):
        """
        return a swarm object who position and momentum values have been randomly generated inside a phase space hypercube
        and that is heading in the negative x direction with average velocity lattice.v0Nominal. A seed can be reused to
        get repeatable random results.

        :param qMax: maximum dimension in position space of hypercube
        :param pMax: maximum dimension in momentum space of hupercube
        :param num: number of particle in phase space
        :param upperSymmetry: wether to exploit lattice symmetry by only using particles in upper half plane
        :param sameSeed: wether to use the same seed eveythime in the nump random generator, the number 42, or a new one
        :return: swarm: a populated swarm object
        """
        if sameSeed==True: #if i want repetable results to compare to
            np.random.seed(42) #seed the generator
        swarm=Swarm()
        i=0
        while (i<num):
            #randomly assign values
            x=0.0
            y=qMax*(np.random.rand()-.5)
            z=qMax*(np.random.rand()-.5)
            px=-self.lattice.v0Nominal #Swarm is initialized heading in the negative x direction,
            py=pMax*(np.random.rand()-.5)
            pz=pMax*(np.random.rand()-.5)
            if upperSymmetry==True: #if only using particles in the upper plane
                if y<0:
                    pass
                else:
                    q=np.asarray([x,y,z])
                    p=np.asarray([px,py,pz])
                    swarm.add_Particle(q,p)
                    i+=1
            else:
                q = np.asarray([x, y, z])
                p = np.asarray([px, py, pz])
                swarm.add_Particle(q, p)
                i += 1
        np.random.seed(int(time.time())) #re randomize
        return swarm
    def initialize_Random_Swarm_At_Combiner_Output(self,qMax, pMax, num, upperSymmetry=False,sameSeed=True):
        """
        return a swarm object who position and momentum values have been randomly generated inside a phase space hypercube.
        A seed can be reused to getrepeatable random results
        :param qMax: maximum dimension in position space of hypercube
        :param pMax: maximum dimension in momentum space of hupercube
        :param num: number of particle in phase space
        :param upperSymmetry: wether to exploit lattice symmetry by only using particles in upper half plane
        :param sameSeed: wether to use the same seed eveythime in the nump random generator, the number 42, or a new one
        :return: swarm: a populated swarm object
        """
        swarm=self.initalize_Random_Swarm_In_Phase_Space(qMax, pMax, num, upperSymmetry=upperSymmetry,sameSeed=sameSeed)
        R=self.lattice.combiner.RIn
        r2=self.lattice.combiner.r2
        for particle in swarm.particles:
            particle.q[:2]=particle.q[:2]@R
            particle.q+=r2
            particle.p[:2]=particle.p[:2]@R
            particle.q = particle.q + particle.p * 1e-10 #scoot particle into lens
        return swarm
    def send_Swarm_Through_Shaper(self, swarm, Lo, Li, Bp=.5, rp=.03,copySwarm=True):
        # models particles traveling through an injecting element, which is a simple ideal magnet. This model
        # has the object at the origin and the lens at y=0 and x>0. Particles end up on the output of the lens
        # for now a thin lens
        # swarm: swarm to transform through injector
        # Lo: object distance for injector
        # Li: nominal image distance
        if copySwarm==True:
            swarmNew=swarm.copy()
        else:
            swarmNew=swarm
        K = 2 * self.lattice.u0 * Bp / (rp ** 2 * self.lattice.v0Nominal ** 2)
        # now find the magnet length that gives Li. Need to parametarize each entry of the transfer matrix.
        # The transfer matrix is for angles, not velocity
        CFunc = lambda x: np.cos(np.sqrt(K) * x)
        SFunc = lambda x: np.sin(np.sqrt(K) * x) / np.sqrt(K)
        CdFunc = lambda x: -np.sqrt(K) * np.sin(np.sqrt(K) * x)
        SdFunc = lambda x: np.cos(np.sqrt(K) * x)
        LiFunc = lambda x: -(CFunc(x) * Lo + SFunc(x)) / (CdFunc(x) * Lo + SdFunc(x))
        minFunc = lambda x: (LiFunc(x) - Li) ** 2
        sol = spo.minimize_scalar(minFunc, method='bounded', bounds=(.1, .5))
        Lm = sol.x
        MLens = np.asarray([[CFunc(Lm), SFunc(Lm)], [CdFunc(Lm), SdFunc(Lm)]])
        MLo = np.asarray([[1, Lo], [0, 1]])
        MTot = MLens @ MLo
        for particle in swarmNew.particles:
            qNew = particle.q.copy()
            pNew = particle.p.copy()
            # the v0Nominal is present because the matrix is for angles, not velocities
            qNew[1] = MTot[0, 0] * particle.q[1] + MTot[0, 1] * particle.p[1] / self.lattice.v0Nominal
            pNew[1] = MTot[1, 0] * particle.q[1] * self.lattice.v0Nominal + MTot[1, 1] * particle.p[1]
            qNew[2] = MTot[0, 0] * particle.q[2] + MTot[0, 1] * particle.p[2] / self.lattice.v0Nominal
            pNew[2] = MTot[1, 0] * particle.q[2] * self.lattice.v0Nominal + MTot[1, 1] * particle.p[2]
            particle.q = qNew
            particle.p = pNew
        return swarmNew
    def catch_Injection_Errors(self,Li,LOffset):

        if Li<self.lattice.combiner.Lo : #image length must be larger than combiner length
            raise Exception("IMAGE LENGTH IS TOO SHORT")
        if LOffset > self.lattice.combiner.Lo / 2:
            raise Exception("OFFSET IS TOO DEEP INTO THE COMBINER WITH THE CURRENT ALGORITHM")
    def initialize_Swarm_At_Combiner_Output(self, Lo, Li, LOffset, qMax, pMax, numPhaseSpace,parallel=False,upperSymmetry=False):
        # this method generates a cloud of particles in phase space at the output of the combiner.
        # Here the output refers to the origin in the combiner's reference frame.
        #Li: image length, ie distance from end of lens to focus
        #Lo: object distance
        # LOffset: length that the image distance is offset from the combiner output. Positive
        # value corresponds coming to focus before the output (ie, inside the combiner), negative
        # is after the output.
        #qMax: absolute vale of maximum value in space dimensions
        #pMax: absolute vale of maximum value in momentum dimensions
        #num: number of particles along each axis, total number is axis**n where n is dimensions
        self.catch_Injection_Errors(Li,LOffset)

        swarm = self.initialize_HyperCube_Swarm_In_Phase_Space(qMax, pMax, numPhaseSpace, upperSymmetry=upperSymmetry)
        swarm = self.send_Swarm_Through_Shaper(swarm, Lo, Li,copySwarm=False)
        swarm = self.aim_Swarm_At_Combiner(swarm, Li, LOffset)
        #now I need to trace and position the swarm at the output. This is done by moving the swarm along with combiner to the
        #combiner's outlet in it's final position.
        r0 = self.lattice.combiner.r2
        Mrot=self.lattice.combiner.RIn
        def func(particle):
            particle=self.step_Particle_Through_Combiner(particle)
            particle.q[:2] = particle.q[:2] @ Mrot + r0[:2]
            return particle
        if parallel==False:
            for i in range(swarm.num_Particles()):
                swarm.particles[i]=func(swarm.particles[i])
        elif parallel==True:
            results=self.helper.parallel_Chunk_Problem(func,swarm.particles,numWorkers=32)
            for i in range(len(results)): #replaced the particles in the swarm with the new traced particles. Order
                #is not important
                swarm.particles[i]=results[i][1]
        return swarm
    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
    def fast_qNew(q,F,p,h):
        return q+p*h+.5*F*h**2

    @staticmethod
    @numba.njit(numba.float64[:](numba.float64[:],numba.float64[:],numba.float64[:],numba.float64))
    def fast_pNew(p,F,F_n,h):
        return p+.5*(F+F_n)*h
    def step_Particle_Through_Combiner(self, particle, h=1e-5):
        raise Exception('NEED TO BETTER DEAL WITH PARTICLES BEING TRACED MULTIPLE TIMES. how to handle particle.force? '
                        'maybe dont use it')
        particle.currentEl=self.lattice.combiner
        q = particle.q.copy()
        p = particle.p.copy()
        #particle.log_Params()
        dx = (self.lattice.combiner.space * 2 + self.lattice.combiner.Lm) - q[0]
        dt = dx / p[0]
        q += dt * p
        particle.q=q

        force = self.lattice.combiner.force
        ap = self.lattice.combiner.ap
        apz = self.lattice.combiner.apz
        def is_Clipped(q):
            if not -apz<q[2]<apz: #if clipping in z direction
                return True
            elif q[0] < self.lattice.combiner.space + self.lattice.combiner.Lm and np.abs(q[1]) > ap: #Only consider
                #clipping in the y direction if the particle is insides the straight segment of the combiner
                return True
            else:
                return False
        if is_Clipped(q) == True:
            #particle.log_Params()
            particle.finished()
            particle.clipped = True
            return particle
        while (True):
            if particle.force is not None:
                F = particle.force
            else:
                F = -force(q)  # force is negatice for high field seeking
            q_n =self.fast_qNew(q,F,p,h)# q + p * h + .5 * F * h ** 2
            if q_n[0] < 0:  # if overshot, go back and walk up to the edge assuming no force
                dr = - q[0]
                dt = dr / p[0]
                q = q + p * dt
                particle.q = q
                particle.p = p
                #particle.log_Params()
                break
            F_n = -force(q_n)  # force is negative for high field seeking
            particle.force=F_n
            p_n = self.fast_pNew(p,F,F_n,h)#p + .5 * (F + F_n) * h
            q = q_n
            p = p_n
            particle.force = F_n
            particle.q = q
            particle.p = p
            #particle.log_Params()
            if is_Clipped(q) == True:
                particle.clipped=True
                break
        particle.finished()
        if particle.clipped is None:
            particle.clipped=False
        return particle
    def aim_Swarm_At_Combiner(self, swarm, Li, LOffset):
        # This method takes a swarm in phase space, located at the origin, and moves it in phase space
        # so momentum vectors point at the combiner input
        # swarm: swarm of particles in phase space. must be centered at the origin in space
        # Li: the image length, only makes sense for hard edge model
        # LiOffset: image offset in combiner. see initialize_Swarm_At_Combiner_Output
        swarmNew=swarm.copy() #don't change the original swarm
        inputOffset = self.lattice.combiner.inputOffsetLoad
        inputAngle = self.lattice.combiner.angLoad
        dL = Li - self.lattice.combiner.Lo + LOffset  # TODO: SHITTY ALGORITHM THAT NEEDS TO CHANGE. doesn't account
        # for curvature of trajectory. This is the length outside of the combiner
        dx = self.lattice.combiner.space * 2 + self.lattice.combiner.Lm
        dy = inputOffset
        dx += dL * np.cos(inputAngle)
        dy += dL * np.sin(inputAngle)
        dR = np.asarray([dx, dy])
        rotAng = -inputAngle
        rotMat = np.asarray([[np.cos(rotAng), np.sin(rotAng)], [-np.sin(rotAng), np.cos(rotAng)]])
        for particle in swarmNew.particles:
            q = particle.q
            p = particle.p
            q[:2] += dR
            p[:2] = rotMat @ p[:2]
            particle.q = q
            particle.p = p
        return swarmNew
    def optimize_Swarm_Survival_Through_Lattice_Brute(self,bounds,numPoints,T,h=1e-5):
        #bounds: list of tuples for bounds on F1 and F2
        def func(X):
            F1,F2=X
            swarm = Swarm()
            qi = np.asarray([-1e-10, 0.0, 0.0])
            pi = np.asarray([-200.0, 0, 0])
            swarm.add_Particle(qi, pi)
            self.lattice.elList[2].forceFact = F1
            self.lattice.elList[4].forceFact = F2
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
        qMax, pMax, numPhaseSpace=3e-3,10.0,9
        swarm = self.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, qMax, pMax, numPhaseSpace,parallel=parallel)
        particleTracer=ParticleTracer(self.lattice)

        if testNextElement==True:
            elNext=self.lattice.elList[self.lattice.combinerIndex+1]
            #need to scoot particle along
            h=2.5e-5
            T=elNext.Lo/self.lattice.v0Nominal
            if parallel==False:
                for i in range(swarm.num_Particles()):
                    particle=swarm.particles[i]
                    particle.q=particle.q+particle.p*1e-10 #scoot particle into next element
                    swarm.particles[i]=particleTracer.trace(particle,h,T,fastMode=True)
            else:
                def wrapper(particle):
                    return particleTracer.trace(particle, h, T, fastMode=True)
                results=self.helper.parallel_Chunk_Problem(wrapper,swarm.particles)
                for i in range(len(results)):
                    swarm.particles[i]=results[i][1]


        return swarm.survival_Bool()

    def trace_Swarm_Through_Lattice(self,swarm,h,T,parallel=False,fastMode=True):

        #trace a swarm through the lattice
        swarmNew=swarm.copy()
        particleTracer = ParticleTracer(self.lattice)
        if parallel==True:
            def func(particle):
                return particleTracer.trace(particle, h, T,fastMode=fastMode)


            results = self.helper.parallel_Problem(func, swarmNew.particles)
            for i in range(len(results)): #replaced the particles in the swarm with the new traced particles. Order
                #is not important
                swarmNew.particles[i]=results[i][1]
        else:
            for i in range(swarmNew.num_Particles()):
                swarmNew.particles[i]=particleTracer.trace(swarmNew.particles[i],h,T,fastMode=fastMode)

        return swarmNew
    def compute_Survival_Through_Lattice(self,Lo,Li,LOffset,F1,F2,h,T,parallel=False):
        qMax=2.5e-3
        pMax=3e1
        numPhaseSpace=9
        swarm=self.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, qMax, pMax, numPhaseSpace,parallel=parallel,upperSymmetry=True)
        self.lattice.elList[2].forceFact=F1
        self.lattice.elList[4].forceFact=F2

        swarm=self.trace_Swarm_Through_Lattice(swarm,h,T,parallel=parallel)
        print('done, survival is: ',np.round(swarm.survival_Rev(),3))
        return swarm.survival_Rev()

    def compute_Max_Revs(self,X,h=5e-6,cutoff=8.0):
        # for the given configuraiton X and particle(s) return the maximum number of revolutions
        T=(cutoff+.25)*self.lattice.totalLength/self.lattice.v0Nominal #number of revolutions can be a bit fuzzy so I add
        # a little extra to compensate
        self.lattice.elList[2].forceFact = X[0]
        self.lattice.elList[4].forceFact = X[1]
        revolutionsList = []
        for particle in self.swarmStability:
            particle = self.particleTracer.trace(particle.copy(), h, T)
            revolutionsList.append(particle.revolutions)
            if revolutionsList[-1] > cutoff:
                break
        return max(revolutionsList)
    def get_Stability_Function(self,qMax=250e-6,numParticlesPerDim=2,h=5e-6,cutoff=8.0,funcType='bool'):
        #TODO:Make this parallel! FREE LUNCH! BUT TEST IT
        T=(cutoff+.25)*self.lattice.totalLength/self.lattice.v0Nominal #number of revolutions can be a bit fuzzy so I add
        # a little extra to compensate
        if numParticlesPerDim==1:
            qInitialArr=np.asarray([np.asarray([-1e-10,0.0,0.0])])
        else:
            qTempArr=np.linspace(-qMax,qMax,numParticlesPerDim)
            qInitialArr_yz=np.asarray(np.meshgrid(qTempArr,qTempArr)).T.reshape(-1,2)
            qInitialArr=np.column_stack((np.zeros(qInitialArr_yz.shape[0])-1e-10,qInitialArr_yz))
        pi=np.asarray([-self.lattice.v0Nominal,0.0,0.0])
        swarm=Swarm()
        for qi in qInitialArr:
            swarm.add_Particle(qi,pi)
        if numParticlesPerDim%2==0: #if even number then add one more particle at the center
            swarm.add_Particle()
        def compute_Stability(X):
            #X lattice arguments
            #for the given configuraiton X and particle(s) return the maximum number of revolutions, or True for stable
            #or False for unstable
            self.lattice.elList[2].forceFact=X[0]
            self.lattice.elList[4].forceFact = X[1]
            revolutionsList=[]
            for particle in swarm:
                particle=self.particleTracer.trace(particle.copy(),h,T)
                revolutionsList.append(particle.revolutions)
                if revolutionsList[-1]>cutoff:
                    if funcType=='bool':
                        return True #stable
                    elif funcType=='rev':
                        break
            if funcType=='bool':
                return False #unstable
            elif funcType=='rev':
                return max(revolutionsList)
        return compute_Stability
    def compute_Revolution_Func_Over_Grid(self,bounds=None,qMax=1e-4,numParticlesPerDim=2,gridPoints=40,h=5e-6,cutoff=8.0):
        #this method loops over a grid and logs the numbre of revolutions up to cutoff for the particle with the
        #maximum number of revolutions
        #bounds: region to search over for instability
        #qMax: maximum dimension in transverse directions for initialized particles
        #numParticlesPerDim: Number of particles along y and z axis so total is numParticlesPerDim**2. when 1 a single
        #particle is initialized at [1e-10,0,0]
        #gridPoints: number of points per axis to test stability. Total is gridPoints**2
        #cutoff: Maximum revolution number
        #returns: a function that evaluates to the maximum number of revolutions of the particles

        revFunc=self.get_Stability_Function(qMax=qMax,numParticlesPerDim=numParticlesPerDim,h=h,cutoff=cutoff,funcType='rev')
        boundAxis1Arr=np.linspace(bounds[0][0],bounds[0][1],num=gridPoints)
        boundAxis2Arr = np.linspace(bounds[1][0], bounds[1][1], num=gridPoints)
        testPointsArr=np.asarray(np.meshgrid(boundAxis1Arr,boundAxis2Arr)).T.reshape(-1,2)
        results=self.helper.parallel_Problem(revFunc,testPointsArr)
        coordList=[]
        valList=[]
        for result in results:
            coordList.append(result[0])
            valList.append(result[1])
        revolutionFunc=spi.LinearNDInterpolator(coordList,valList)
        return revolutionFunc



    def plot_Stability(self,bounds=None,qMax=1e-4,numParticlesPerDim=2,gridPoints=40,savePlot=False,
                              plotName='stabilityPlot',cutoff=8.0,h=5e-6,showPlot=True):
        #bounds: region to search over for instability
        #qMax: maximum dimension in transverse directions for initialized particles
        #numParticlesPerDim: Number of particles along y and z axis so total is numParticlesPerDim**2.
        #gridPoints: number of points per axis to test stability. Total is gridPoints**2
        #cutoff: Maximum revolutions below this value are considered unstable

        if bounds is None:
            bounds = [(0.0, .5), (0.0, .5)]

        stabilityFunc=self.compute_Revolution_Func_Over_Grid(bounds=bounds,qMax=qMax,
            numParticlesPerDim=numParticlesPerDim, gridPoints=gridPoints,cutoff=cutoff,h=h)
        plotxArr=np.linspace(bounds[0][0],bounds[0][1],num=250)
        plotyArr = np.linspace(bounds[1][0], bounds[1][1], num=250)
        image=np.empty((plotxArr.shape[0],plotyArr.shape[0]))
        for i in range(plotxArr.shape[0]):
            for j in range(plotyArr.shape[0]):
                image[j,i]=stabilityFunc(plotxArr[i],plotyArr[j])

        image=np.flip(image,axis=0)
        extent=[bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]]
        plt.imshow(image,extent=extent)

        plt.suptitle('Stability regions')
        plt.title('Yellow indicates more stability, purple indicates more unstable')
        plt.xlabel('Field strength multiplier')
        plt.ylabel('Field strength multiplier')
        if savePlot==True:
            plt.savefig(plotName)
        if showPlot==True:
            plt.show()

    def maximize_Suvival_Through_Lattice(self,h,T,numParticles=1000,qMax=3e-3,pMax=5e0,returnBestSwarm=False,parallel=False,
                                         maxEvals=100,bounds=None,precision=5e-3):
        #todo: THis is very poorly organized! needs to be changed into its own class
        if bounds is None:
            bounds=[(0.0, .5), (0.0, .5)]
        class Solution:
            #because I renormalize bounds and function values, I used this solution class to easily access the more
            #familiar values that I am interested in
            def __init__(self):
                self.skoptSol=None #to hold the skopt solutiob object
                self.x=None #list for real paremters values
                self.fun=None #for real solution value
        swarm = self.initialize_Random_Swarm_At_Combiner_Output(qMax, pMax, numParticles)


        stepsX=int((bounds[0][1]-bounds[0][0])/precision)
        stepsY = int((bounds[1][1] - bounds[1][0]) / precision)
        if stepsX+stepsY<=1:
            raise Exception('THERE ARE NOT ENOUGH POINTS IN SPACE TO EXPLORE, MUST BE MORE THAN 1')
        boundsNorm = [(0, stepsX), (0, stepsY)]
        print(boundsNorm)

        def min_Func(X):
            XNew = X.copy()
            for i in range(len(X)):  # change normalized bounds to actual
                XNew[i] = ((bounds[i][1] - bounds[i][0]) * float(X[i])/float(boundsNorm[i][1]-boundsNorm[i][0]) + bounds[i][0])
            self.lattice.elList[2].forceFact = XNew[0]
            self.lattice.elList[4].forceFact = XNew[1]
            swarmNew = self.trace_Swarm_Through_Lattice(swarm, h, T, parallel=True, fastMode=True)
            self.i += 1
            survival = swarmNew.survival_Rev()
            print(XNew,X, survival, swarmNew.longest_Particle_Life_Revolutions())
            Tsurvival = survival * self.lattice.totalLength / self.lattice.v0Nominal
            cost = -Tsurvival / T  # cost scales from 0 to -1.0
            return cost

        stabilityFunc = self.get_Stability_Function(numParticlesPerDim=1, cutoff=8.0,h=5e-6)

        def stability_Func_Wrapper(X):
            XNew = X.copy()
            for i in range(len(X)):  # change normalized bounds to actual
                XNew[i] = ((bounds[i][1] - bounds[i][0]) * float(X[i])/float(boundsNorm[i][1]-boundsNorm[i][0]) + bounds[i][0])
            return stabilityFunc(XNew)

        unstableCost = -1.5 * (self.lattice.totalLength / self.lattice.v0Nominal) / T  # typically unstable regions return an average
        # of 1-2 revolution
        numInit = int(maxEvals * .5)  # 50% is just random
        xiRevs = .25  # search for the next points that returns an imporvement of at least this many revs
        xi = (xiRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T
        noiseRevs =1e-2 #small amount of noise to account for variability of results and encourage a smooth fit
        noise = (noiseRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T


        model = skopt.Optimizer(boundsNorm, n_initial_points=numInit, acq_func='EI', acq_optimizer='sampling',
                                acq_func_kwargs={"xi": xi, 'noise': noise}, n_jobs=-1)
        self.resetXiCounts=0
        self.countXi=False
        def generate_Next_Point():
            if evals <= numInit-1:  # if still initializing the model
                x1 = int(np.random.rand() * stepsX)
                x2 = int(np.random.rand() * stepsY)
                XSample = [x1, x2]
            else:
                XSample = model.ask()
            if len(model.Xi) > 1 and evals > numInit-1:  # if the optimizer is suggesting duplicate points
                loops = 0
                while (loops < 10):  # try to force the optimizer to pick another point
                    if np.any(np.sum((np.asarray(model.Xi) - np.asarray(XSample)) ** 2, axis=1) == 0):
                        print('DUPLICATE POINTS',XSample)
                        model.acq_func_kwargs['xi'] = model.acq_func_kwargs['xi'] * 2
                        model.update_next()
                        XSample = model.ask()
                        self.countXi=True
                    else:
                        break
                    loops += 1
                if loops == -9:
                    raise Exception('COULD NOT STEER MODEL TO A NEW POINT')
                if self.countXi==True:
                    self.resetXiCounts+=1
                    if self.resetXiCounts==5:
                        model.acq_func_kwargs['xi']=xi
                        self.countXi=False
                        self.resetXiCounts=0
                        print('search reset!')
            return XSample



        evals = 0
        t = time.time()

        print('starting')
        while (evals < maxEvals): #TODO: REMOVE DUPLICATE CODE
            print(evals)

            XSample=generate_Next_Point()
            print(XSample)
            if stability_Func_Wrapper(XSample) == True:  # possible solution
                cost = min_Func(XSample)
                model.tell(XSample, cost)
                evals += 1

            else:  # not possible solution
                model.tell(XSample, unstableCost+np.random.rand()*1e-10) #add a little random noise to help
                #with stability. Doesn't work well when all the points are the same sometimes


        print(time.time() - t)
        sol = model.get_result()
        solution = Solution()
        solution.skoptSol = sol
        x = [0, 0]
        for i in range(len(sol.x)):  # change normalized bounds to actual
            x[i] = ((bounds[i][1] - bounds[i][0]) * float(sol.x[i]) / float(boundsNorm[i][1] - boundsNorm[i][0]) +
                       bounds[i][0])
        solution.x = x
        solution.fun = -sol.fun * T * self.lattice.v0Nominal / self.lattice.totalLength
        return solution

        # swarm = self.initialize_Random_Swarm_At_Combiner_Output(qMax,pMax,numParticles)
        # boundsNorm = [(0.0, 1.0), (0.0, 1.0)]
        # self.i = 0
        # def min_Func(X):
        #     XNew = X.copy()
        #     for i in range(len(X)): #change normalized bounds to actual
        #         XNew[i] = (bounds[i][1] - bounds[i][0]) * X[i] + bounds[i][0]
        #
        #     self.lattice.elList[2].forceFact = XNew[0]
        #     self.lattice.elList[4].forceFact = XNew[1]
        #     swarmNew = self.trace_Swarm_Through_Lattice(swarm, h, T, parallel=True, fastMode=True)
        #     self.i += 1
        #     survival = swarmNew.survival_Rev()
        #     print(self.i, X, survival, swarmNew.longest_Particle_Life_Revolutions())
        #     Tsurvival = survival * self.lattice.totalLength / self.lattice.v0Nominal
        #     cost = -Tsurvival / T  # cost scales from 0 to -1.0
        #     return cost
        #
        # stabilityFunc = self.get_Stability_Function(numParticlesPerDim=1,cutoff=5.0)
        # def stability_Func_Wrapper(X):
        #     XNew = X.copy()
        #     for i in range(len(X)):  # change normalized bounds to actual
        #         XNew[i] = (bounds[i][1] - bounds[i][0]) * X[i] + bounds[i][0]
        #     return stabilityFunc(XNew)
        #
        #
        # unstableCost=-1.5*(self.lattice.totalLength / self.lattice.v0Nominal)/T  #typically unstable regions return an average
        # #of 1-2 revolution
        # numInit=int(maxEvals*.5) #50% is just random
        # xiRevs=.5 #search for the next points that returns an imporvement of at least this many revs
        # xi=(xiRevs*(self.lattice.totalLength/self.lattice.v0Nominal))/T
        # noiseRevs=.1
        # noise=(noiseRevs*(self.lattice.totalLength/self.lattice.v0Nominal))/T
        # print(xi)
        #
        # model = skopt.Optimizer(boundsNorm, n_initial_points=numInit, acq_func='EI', acq_optimizer='sampling',
        #                         acq_func_kwargs={"xi":xi,'noise':noise},n_jobs=-1)
        # evals=0
        # t=time.time()
        # print('starting')
        # while(evals<maxEvals):
        #     print(evals)
        #     if evals<numInit: #if still initializing the model
        #         x1=np.random.rand()
        #         x2=np.random.rand()
        #         XSample=[x1,x2]
        #         if stability_Func_Wrapper(XSample)==True: #possible solution
        #             cost=min_Func(XSample)
        #             model.tell(XSample,cost)
        #             evals+=1
        #         else: #not possible solution
        #             model.tell(XSample,unstableCost)
        #
        #     else: #search for minimum now
        #
        #         XSample=model.ask()
        #         if stability_Func_Wrapper(XSample)==True: #possible solution
        #             cost=min_Func(XSample)
        #             model.tell(XSample,cost)
        #             evals+=1
        #         else: #not possible solution
        #             model.tell(XSample, unstableCost)
        # print(time.time() - t)
        # sol=model.get_result()
        # solution=Solution()
        # solution.skoptSol=sol
        # x=[0,0]
        # for i in range(len(bounds)):
        #     x[i]=(bounds[i][1] - bounds[i][0]) * sol.x[i] + bounds[i][0]
        # solution.x=x
        # solution.fun=-sol.fun*T*self.lattice.v0Nominal/self.lattice.totalLength
        # return solution
