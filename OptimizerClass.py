from profilehooks import profile
from ParticleTracer import ParticleTracer
import numpy.linalg as npl
import matplotlib.pyplot as plt
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

    def initialize_Swarm_In_Phase_Space(self, qMax, pMax,num,upperSymmetry=False):
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

    def send_Swarm_Through_Shaper(self, swarm, Lo, Li, Bp=.5, rp=.03):
        # models particles traveling through an injecting element, which is a simple ideal magnet. This model
        # has the object at the origin and the lens at y=0 and x>0. Particles end up on the output of the lens
        # for now a thin lens
        # swarm: swarm to transform through injector
        # Lo: object distance for injector
        # Li: nominal image distance

        swarm=swarm.copy()
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
        for particle in swarm.particles:
            qNew = particle.q.copy()
            pNew = particle.p.copy()
            # the v0Nominal is present because the matrix is for angles, not velocities
            qNew[1] = MTot[0, 0] * particle.q[1] + MTot[0, 1] * particle.p[1] / self.lattice.v0Nominal
            pNew[1] = MTot[1, 0] * particle.q[1] * self.lattice.v0Nominal + MTot[1, 1] * particle.p[1]
            qNew[2] = MTot[0, 0] * particle.q[2] + MTot[0, 1] * particle.p[2] / self.lattice.v0Nominal
            pNew[2] = MTot[1, 0] * particle.q[2] * self.lattice.v0Nominal + MTot[1, 1] * particle.p[2]
            particle.q = qNew
            particle.p = pNew
        return swarm
    def catch_Injection_Errors(self,Li,LOffset):

        if Li<self.lattice.combiner.Lo : #image length must be larger than combiner length
            raise Exception("IMAGE LENGTH IS TOO SHORT")
        if LOffset > self.lattice.combiner.Lo / 2:
            raise Exception("OFFSET IS TOO DEEP INTO THE COMBINER WITH THE CURRENT ALGORITHM")

    def initialize_Swarm_At_Combiner_Output(self, Lo, Li, LOffset, qMax, pMax, numPhaseSpace):
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
        swarm = self.initialize_Swarm_In_Phase_Space(qMax, pMax, numPhaseSpace)
        swarm = self.send_Swarm_Through_Shaper(swarm, Lo, Li)
        swarm = self.aim_Swarm_At_Combiner(swarm, Li, LOffset)

        #now I need to trace and position the swarm at the output. This is done by moving the swarm along with combiner to the
        #combiner's outlet in it's final position
        r0 = self.lattice.combiner.r2
        Mrot=self.lattice.combiner.RIn
        for particle in swarm.particles:
            self.step_Particle_Through_Combiner(particle)
            particle.q[:2]=particle.q[:2]@Mrot+r0[:2]
        return swarm

    def step_Particle_Through_Combiner(self, particle, h=1e-5):
        q = particle.q.copy()
        p = particle.p.copy()
        particle.log_Params()
        dx = (self.lattice.combiner.space * 2 + self.lattice.combiner.Lm) - q[0]
        dt = dx / p[0]
        q += dt * p

        force = self.lattice.combiner.force
        ap = self.lattice.combiner.ap
        apz = self.lattice.combiner.apz
        def is_Clipped(q):
            if np.abs(q[2]) > apz:
                return True
            elif q[0] < self.lattice.combiner.space + self.lattice.combiner.Lm and np.abs(q[1]) > ap:
                return True

        if is_Clipped(q) == True:
            particle.finished()
            particle.clipped = True
            return
        while (True):
            if particle.force is not None:
                F = particle.force
            else:
                F = -force(q)  # force is negatice for high field seeking
            q_n = q + p * h + .5 * F * h ** 2
            if q_n[0] < 0:  # if overshot, go back and walk up to the edge assuming no force
                dr = - q[0]
                dt = dr / p[0]
                q = q + p * dt
                particle.q = q
                particle.p = p
                #particle.log_Params()
                break
            F_n = -force(q_n)  # force is negative for high field seeking
            p_n = p + .5 * (F + F_n) * h
            q = q_n
            p = p_n
            particle.force = F_n
            #particle.q = q
            #particle.p = p
            #particle.log_Params()
            if is_Clipped(q) == True:
                particle.finished()
                particle.clipped = True
                return
        particle.finished()
        particle.clipped = False

        #print(particle.qi,particle.pi)
        #plt.plot(particle.qArr[:,0],particle.qArr[:,2])
        #plt.grid()
        #plt.show()

    def aim_Swarm_At_Combiner(self, swarm, Li, LOffset):
        # This method takes a swarm in phase space, located at the origin, and moves it in phase space
        # so momentum vectors point at the combiner input
        # swarm: swarm of particles in phase space. must be centered at the origin in space
        # Li: the image length, only makes sense for hard edge model
        # LiOffset: image offset in combiner. see initialize_Swarm_At_Combiner_Output
        swarm=swarm.copy() #don't change the original swarm
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
        for particle in swarm.particles:
            q = particle.q
            p = particle.p
            q[:2] += dR
            p[:2] = rotMat @ p[:2]
            particle.q = q
            particle.p = p

        return swarm
    def compute_Survival_Through_Injector(self, Lo,Li, LOffset, qMax, pMax, numPhaseSpace,testNextElement=True):
        """
        return fraction
        :param Lo:
        :param Li:
        :param LOffset:
        :param qMax:
        :param pMax:
        :param numPhaseSpace:
        :param testNextElement:
        :return:
        """
        swarm = self.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, qMax, pMax, numPhaseSpace)
        particleTracer=ParticleTracer(self.lattice)
        if testNextElement==True:
            elNext=self.lattice.elList[self.lattice.combinerIndex+1]
            #need to scoot particle along
            h=2.5e-5
            T=elNext.Lo/self.lattice.v0Nominal
            pre=0
            post=0
            for particle in swarm.particles:
                particle.q=particle.q+particle.p*1e-10 #scoot particle into next element
                #pre+=int(particle.clipped)
                particleTracer.trace(particle,h,T,fastMode=True)
                #post += int(particle.clipped)
        return swarm.survival()

    # def maximize_Survival_Through_Injector_Brute(self, LiMin,LiMax,LoMin,LoMax,qMax, pMax, numPhaseSpace, numParams=50):
    #     LOffset = 0.0
    #
    #     swarmInitial = self.initialize_Swarm_In_Phase_Space(qMax, pMax, numPhaseSpace)
    #     def func(X):
    #         Li, Lo = X
    #         self.catch_Injection_Errors(Li, LOffset)
    #         swarm = self.send_Swarm_Through_Shaper(swarmInitial, Lo, Li)
    #         swarm = self.aim_Swarm_At_Combiner(swarm, Li, LOffset)
    #         for particle in swarm.particles:
    #             self.step_Particle_Through_Combiner(particle)
    #         #print(swarm.survival())
    #         return swarm.survival()
    #
    #     x1Arr = np.linspace(LiMin, LiMax, num=numParams)
    #     x2Arr = np.linspace(LoMin, LoMax, num=numParams)
    #     inputArgsArr = np.asarray(np.meshgrid(x1Arr, x2Arr)).T.reshape(-1, 2)
    #     results=self.helper.parallel_Chunk_Problem(func, inputArgsArr)
    #     argList=[]
    #     survivalList=[]
    #     for result in results:
    #         argList.append(result[0])
    #         survivalList.append(result[1])
    #     argArr=np.asarray(argList)
    #     survivalArr=np.asarray(survivalList)
    #     return np.max(survivalArr),argArr[np.argmax(survivalArr)]

    def maximize_Survival_Through_Injector_BB(self, generations,bounds,qMax, pMax, numPhaseSpace,upperSymmetry=False):
        #bounds: a list of tuples of the bounds for Lo,Li,LOffset.
        #if len(bounds[1])!=len(bounds[0]) or len(bounds[0])!=3:
        #    raise Exception("BOUNDS ARE NOT IN RIGHT SHAPE")
        def func(X):
            survival=self.compute_Survival_Through_Injector(*X,qMax,pMax,numPhaseSpace)
            return 1/(1+survival)
        numWorkers=mp.cpu_count()
        res=bb.search_min(f=func, domain=bounds, budget=generations*numWorkers, batch=numWorkers, resfile='test.csv')
        print(res,func(res))
    def trace_Swarm_Through_Lattice(self,swarm,h,T,parallel=False):
        #trace a swarm through the lattice
        swarmNew=swarm.copy()
        particleTracer = ParticleTracer(self.lattice)
        if parallel==True:
            def func(particle):
                particleTracer.trace(particle, h, T)
            self.helper.parallel_Chunk_Problem(func,swarmNew.particles)
        else:
            for particle in swarmNew.particles:
                particleTracer.trace(particle,h,T)
        return swarmNew

    def maximize_Suvival_Through_Lattice(self):
        qMax=2.5e-4
        pMax=3e-1
        numPhaseSpace=3
        T0=10*self.lattice.totalLength/200.0
        h=1e-5
        def func(X):
            F1,F2=X
            Lo=.25
            Li=1.0
            LOffset=0.0
            swarm=self.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, qMax, pMax, numPhaseSpace)#
            self.lattice.elList[3].forceFact=F1
            self.lattice.elList[5].forceFact=F2
            swarm=self.trace_Swarm_Through_Lattice(swarm,h,T0,parallel=True)
            return 1/(1+swarm.survival())
        class problem(ps.optimization_problems.OptimizationProblem):
            def __init__(self,function):
                self.func=function
                self.lb = np.asarray([.1,.1])
                self.ub = np.asarray([.5,.5])
                self.dim = 2
                self.int_var = []
                self.cont_var = np.asarray([0,1])
                self.helper = ParaWell()
            def eval(self, args):
                print('here')
                return self.func(args)

        prob=problem(func)
        rbf = ps.surrogate.RBFInterpolant(dim=prob.dim, lb=prob.lb, ub=prob.ub)
        init = ps.experimental_design.LatinHypercube(dim=prob.dim, num_pts=5*(prob.dim+1))
        controller = pc.ThreadController()
        controller.strategy = ps.strategy.SRBFStrategy(max_evals=25, opt_prob=prob, exp_design=init, surrogate=rbf)
        for i in range(4):
            worker = pc.BasicWorkerThread(controller, prob.eval)
            # worker=pc.ProcessWorkerThread(controller,prob.eval)
            controller.launch_worker(worker)
        print('threads started')
        result = controller.run()
        print(result.value)
        print(result.params)


        #bounds=[[.1,.5],[.1,.5]]
        #num=2
        #t=time.time()
        #spo.brute(func1,bounds,Ns=num)
        #print((time.time()-t)/num**2)