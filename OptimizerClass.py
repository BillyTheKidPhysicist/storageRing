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

import pySOT as ps
from ParaWell import ParaWell
import poap.controller as pc






class Optimizer:
    def __init__(self, lattice):
        self.lattice = lattice
        self.helper=ParaWell() #custom class to help with parallelization

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
            results=self.helper.parallel_Chunk_Problem(func,swarmNew.particles)
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
    def maximize_Suvival_Through_Lattice(self,h,T,numParticles=1000,qMax=3e-3,pMax=5e0,returnBestSwarm=False,parallel=False,
                                         maxEvals=100,bounds=None):

        if bounds is None:
            bounds=[(0.0, .5), (0.0, .5)]
        swarm = self.initialize_Random_Swarm_At_Combiner_Output(qMax,pMax,numParticles)
        self.i=0
        def min_Func(X):
            self.lattice.elList[2].forceFact=X[0]
            self.lattice.elList[4].forceFact = X[1]
            swarmNew = self.trace_Swarm_Through_Lattice(swarm, h, T, parallel=True, fastMode=True)
            self.i+=1
            print(self.i,X,swarmNew.survival_Rev(),swarmNew.longest_Particle_Life())
            return -swarmNew.survival_Rev()


        t=time.time()
        numInit=int(maxEvals*.5) #50% is just random\
        print('starting')
        sol=skopt.gp_minimize(min_Func,bounds,n_calls=maxEvals,n_initial_points=numInit,initial_point_generator='lhs'
                              ,noise=.25)
        print(time.time()-t)
        return sol
        time.sleep(10.0)
        func=sol.models
        valBest=sol.fun
        argBest=sol.x
        # argBest,func = bb.search_min(min_Func,bounds,maxEvals,1,'output.csv')  # text file where results will be saved
        # X=argBest
        # self.lattice.elList[2].forceFact = X[0]
        # self.lattice.elList[4].forceFact = X[1]
        # swarmBest = self.trace_Swarm_Through_Lattice(swarm, h, T, parallel=True, fastMode=True)
        # valBest=swarmBest.survival_Rev()
        #



        #
        #
        # FArr=np.linspace(0.05,.5,num=20)
        # argsArr=np.asarray(np.meshgrid(FArr,FArr)).T.reshape(-1,2)
        # if parallel==True:
        #     sol=self.helper.parallel_Chunk_Problem(min_Func,argsArr)
        #     valArr = np.zeros(len(sol))
        #     argArr = np.zeros((len(sol), 2))
        #     for i in range(len(sol)):
        #         argArr[i] = sol[i][0]
        #         valArr[i] = sol[i][1]
        #     valBest = 1 / np.min(valArr) - 1
        #     argBest = argArr[np.argmin(valArr)]
        # else:
        #     valList=[]
        #     for arg in argsArr:
        #         valList.append(min_Func(arg))
        #         print('done',arg,1/valList[-1]-1)
        #     valArr=np.asarray(valList)
        #     valBest = 1 / np.min(valArr) - 1
        #     argBest = argsArr[np.argmin(valArr)]

        # bounds=[(0.0,.5),(0.0,.5)]
        # sol=spo.differential_evolution(min_Func,bounds,workers=32,disp=True,popsize=32,polish=False,maxiter=4)
        # argBest=sol.x
        # valBest=1/sol.fun-1.0

        if returnBestSwarm==True:
            self.lattice.elList[2].forceFact=argBest[0]
            self.lattice.elList[4].forceFact=argBest[1]
            swarmBest = self.trace_Swarm_Through_Lattice(swarm, h, T, parallel=True, fastMode=True)
            return valBest,argBest,func,swarmBest
        else:
            return valBest,argBest,func
