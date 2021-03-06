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

        swarmNew=swarm.copy()
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

        swarm = self.initialize_Swarm_In_Phase_Space(qMax, pMax, numPhaseSpace,upperSymmetry=upperSymmetry)
        swarm = self.send_Swarm_Through_Shaper(swarm, Lo, Li)
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
                # print(npl.norm(swarm.particles[i].p))
                #
                #
                # #print(swarm.particles[i].qi,swarm.particles[i].pi)
                # fig, axs = plt.subplots(2)
                # #plot z
                # L=self.lattice.combiner.Lm+self.lattice.combiner.space
                # apz=self.lattice.combiner.apz
                # x=[0,L,L,0]
                # y=[apz,apz,-apz,-apz]
                # axs[0].plot(x,y)
                # axs[0].plot(swarm.particles[i].qArr[:,0],swarm.particles[i].qArr[:,2])
                # axs[0].scatter(swarm.particles[i].qArr[:, 0], swarm.particles[i].qArr[:, 2],c='r')
                # axs[0].grid()
                # # plot y
                # ap = self.lattice.combiner.ap
                # x = [0, L, L, 0]
                # y = [ap, ap, -ap, -ap]
                # axs[1].plot(x, y)
                # axs[1].plot(swarm.particles[i].qArr[:, 0], swarm.particles[i].qArr[:, 1])
                # axs[1].scatter(swarm.particles[i].qArr[:, 0], swarm.particles[i].qArr[:, 1],c='r')
                # axs[1].grid()
                #
                # plt.show()
                #
                # p=swarm.particles[i].pArr
                # x=swarm.particles[i].qArr[:,0]
                # y=npl.norm(p,axis=1)
                # plt.plot(x,y)
                # plt.show()
        elif parallel==True:
            results=self.helper.parallel_Chunk_Problem(func,swarm.particles)
            for i in range(len(results)): #replaced the particles in the swarm with the new traced particles. Order
                #is not important
                swarm.particles[i]=results[i][1]
        return swarm

    def step_Particle_Through_Combiner(self, particle, h=1e-5):
        particle.currentEl=self.lattice.combiner
        particle=particle.copy()
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
            if np.abs(q[2]) > apz: #if clipping in z direction
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
            particle.q = q
            particle.p = p
            #particle.log_Params()
            if is_Clipped(q) == True:
                break
        particle.finished()
        particle.clipped = False
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
            for i in range(swarm.num_Particles()):
                particle=swarm.particles[i]
                particle.q=particle.q+particle.p*1e-10 #scoot particle into next element
                #pre+=int(particle.clipped)
                swarm.particles[i]=particleTracer.trace(particle,h,T,fastMode=True)
                #post += int(particle.clipped)
        return swarm.survival_Rev()


    def trace_Swarm_Through_Lattice(self,swarm,h,T,parallel=False):
        #trace a swarm through the lattice
        swarmNew=swarm.copy()
        particleTracer = ParticleTracer(self.lattice)
        if parallel==True:
            def func(particle):
                return particleTracer.trace(particle, h, T)
            results=self.helper.parallel_Chunk_Problem(func,swarmNew.particles)
            for i in range(len(results)): #replaced the particles in the swarm with the new traced particles. Order
                #is not important
                swarmNew.particles[i]=results[i][1]
        else:
            for i in range(swarmNew.num_Particles()):
                swarmNew.particles[i]=particleTracer.trace(swarmNew.particles[i],h,T,fastMode=True)
        return swarmNew
    def compute_Survival_Through_Lattice(self,Lo,Li,LOffset,F1,F2,h,T,parallel=True):
        qMax=2.5e-4
        pMax=3e-1
        numPhaseSpace=5
        swarm=self.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, qMax, pMax, numPhaseSpace,parallel=parallel,upperSymmetry=True)
        self.lattice.elList[3].forceFact=F1
        self.lattice.elList[5].forceFact=F2
        swarm=self.trace_Swarm_Through_Lattice(swarm,h,T,parallel=parallel)
        return swarm.survival_Rev()
    def maximize_Suvival_Through_Lattice(self):
        from poap.controller import BasicWorkerThread, ThreadController
        from pySOT.experimental_design import SymmetricLatinHypercube
        from pySOT.strategy import SRBFStrategy
        from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant


        T=100*self.lattice.totalLength/200.0
        h=2.5e-5
        def func(X):
            Lo,Li,LOffset,F1,F2=X
            survival=self.compute_Survival_Through_Lattice(Lo,Li,LOffset,F1,F2,h,T)
            return 1/(1+survival)
        class problem(ps.optimization_problems.OptimizationProblem):
            def __init__(self,function):
                self.func=function
                self.lb = np.asarray([.15,.75,-.1,.1,.1])
                self.ub = np.asarray([.4,2.0,.1,.5,.5])
                self.dim = len(self.lb)
                self.int_var = []
                self.cont_var = np.asarray([0,1,2,3,4])
                self.helper = ParaWell()
                self.funcEval=0
            def eval(self, args):
                self.funcEval+=1
                val=self.func(args)
                return val
        def run_Model():
            num_threads = 4
            max_evals = 10
            prob =problem(func)#Ackley(dim=1)
            #X=[0.37727273 ,0.08181818 ,1.02272727 ,0.20909091, 0.5       ]
            #prob.eval(X)
            rbf = RBFInterpolant(dim=prob.dim, lb=prob.lb, ub=prob.ub, kernel=CubicKernel(),
                                 tail=LinearTail(prob.dim))
            slhd = SymmetricLatinHypercube(dim=prob.dim, num_pts=2 * (prob.dim + 1))

            # Create a strategy and a controller
            controller = ThreadController()
            controller.strategy = SRBFStrategy(
                max_evals=max_evals, opt_prob=prob, exp_design=slhd, surrogate=rbf, asynchronous=True
            )

            print("Number of threads: {}".format(num_threads))
            print("Maximum number of evaluations: {}".format(max_evals))
            print("Strategy: {}".format(controller.strategy.__class__.__name__))
            print("Experimental design: {}".format(slhd.__class__.__name__))
            print("Surrogate: {}".format(rbf.__class__.__name__))

            # Launch the threads and give them access to the objective function
            t=time.time()
            for _ in range(num_threads):
                worker = BasicWorkerThread(controller, prob.eval)
                controller.launch_worker(worker)

            # Run the optimization strategy
            result = controller.run()
            print(time.time()-t)  #time to beat, 312
            print("Best value found: {0}".format(1/result.value - 1.0))
            print(
                "Best solution found: {0}\n".format(
                    np.array_str(result.params[0], max_line_width=np.inf, precision=5, suppress_small=True)
                )
            )
        run_Model()
        run_Model()
        run_Model()




        #config2
        #Best
        #value
        #found: 0.269166285627191
        #Best
        #solution
        #found: [0.30332 1.74953 0.09113 0.23182 0.20247]
