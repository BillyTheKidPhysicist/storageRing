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
from SwarmTracer import SwarmTracer


class PhaseSpaceInterpolater:
    def __init__(self,X,Y):
        #X: evaluation points array with shape n,m where n is number of points, and m is dimension
        #Y: values
        self.X=X
        self.Y=Y
        self.xBounds=[] #list of tuples (min,max) for x
        self.xNorm=np.zeros(X.shape)
        for i in range(X.shape[1]): #loop through dimensions
            self.xBounds.append((X[:,i].min(),X[:,i].max()))
            xRenormed=(X[:,i]-self.xBounds[i][0])/(self.xBounds[i][1]-self.xBounds[i][0])
            self.xNorm[:,i]=xRenormed
        self.interpolater=spi.NearestNDInterpolator(self.xNorm,self.Y)
    def normalize(self,args):
        #x array to be normalized
        argsNew=[]
        i=0
        for arg in args:
            argNorm=(arg-self.xBounds[i][0])/(self.xBounds[i][1]-self.xBounds[i][0])
            if argNorm>1 or argNorm<0:
                print('input',args)
                print('bounds',self.xBounds)
                raise Exception('OUT OF BOUNDS')
            argsNew.append(argNorm)
            i+=1
        return argsNew
    def __call__(self,*args):
        return self.interpolater(self.normalize(args))


class LatticeOptimizer:
    def __init__(self, lattice):
        self.lattice = lattice
        self.helper=ParaWell() #custom class to help with parallelization
        self.i=0 #simple variable to track solution status
        self.particleTracer = ParticleTracer(lattice)
        self.h=None #timestep size
        self.T=None #maximum particle tracing time
        self.swarmTracer=SwarmTracer(self.lattice)

    def update_Lattice(self,X):
        #Update the various paremters in the lattice and injections that are variable
        self.lattice.elList[2].forceFact = X[0]
        self.lattice.elList[4].forceFact = X[1]
    def compute_Phase_Space_Map_Function(self,X,swarmInitial):
        #return a function that returns a value for mnumber of revolutions at a given point in phase space. The phase
        #space is in the combiner's reference frame with the x component zero so the coordinates looke like (y,x,px,py,pz)
        #so the could is centered at the origin in the combiner
        #X: arguments to parametarize lattice
        #swarmInitial: swarm to trace through lattice that is initialized in phase space and centered at (0,0,0)
        self.update_Lattice(X) 
        phaseSpacePoints=[] #holds the coordinates of the particles in phase space at the combiner output
        revolutions=[] #values of revolution for each particle in swarm

        swarmCombiner=self.swarmTracer.move_Swarm_To_Combiner_Output(swarmInitial)
        swarmTraced=self.swarmTracer.trace_Swarm_Through_Lattice(swarmCombiner,self.h,self.T,parallel=True)
        R = self.lattice.combiner.RIn #matrix to rotate into combiner frame
        r2 = self.lattice.combiner.r2 #position of the outlet of the combiner
        for i in range(swarmInitial.num_Particles()):
            q = swarmInitial.particles[i].q.copy()[1:]
            #q[:2]=R@(q-r2)[:2] #move to the combiner frame
            #q=q[1:] #drop the x component that is zero (or very small) for all of them anyways
            p = swarmInitial.particles[i].p[:].copy()
            #p[:2]=R@p[:2]
            #print(p)
            Xi = np.append(q, p)  # phase space coords are 2 position values and 3 momentum0
            phaseSpacePoints.append(Xi)
            revolutions.append(swarmTraced.particles[i].revolutions)
        fitFunc = PhaseSpaceInterpolater(np.asarray(phaseSpacePoints), revolutions)
        return fitFunc





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

    def maximize_Suvival_Through_Lattice(self,h,T,numParticles=1000,qMax=12e-3,pMax=5e0,returnBestSwarm=False,parallel=False,
                                         maxEvals=100,bounds=None,precision=5e-3):
        self.h=h
        self.T=T
        swarmInitial=self.swarmTracer.initalize_Random_Swarm_In_Phase_Space(qMax, pMax, numParticles,pxMax=1.0)

        func=self.compute_Phase_Space_Map_Function([.2,.33],swarmInitial)
        return func

        #
        # #todo: THis is very poorly organized! needs to be changed into its own class
        # if bounds is None:
        #     bounds=[(0.0, .5), (0.0, .5)]
        # class Solution:
        #     #because I renormalize bounds and function values, I used this solution class to easily access the more
        #     #familiar values that I am interested in
        #     def __init__(self):
        #         self.skoptSol=None #to hold the skopt solutiob object
        #         self.x=None #list for real paremters values
        #         self.fun=None #for real solution value
        # swarm = self.initialize_Random_Swarm_At_Combiner_Output(qMax, pMax, numParticles)
        #
        #
        # stepsX=int((bounds[0][1]-bounds[0][0])/precision)
        # stepsY = int((bounds[1][1] - bounds[1][0]) / precision)
        # if stepsX+stepsY<=1:
        #     raise Exception('THERE ARE NOT ENOUGH POINTS IN SPACE TO EXPLORE, MUST BE MORE THAN 1')
        # boundsNorm = [(0, stepsX), (0, stepsY)]
        # print(boundsNorm)
        #
        # def min_Func(X):
        #     XNew = X.copy()
        #     for i in range(len(X)):  # change normalized bounds to actual
        #         XNew[i] = ((bounds[i][1] - bounds[i][0]) * float(X[i])/float(boundsNorm[i][1]-boundsNorm[i][0]) + bounds[i][0])
        #     self.lattice.elList[2].forceFact = XNew[0]
        #     self.lattice.elList[4].forceFact = XNew[1]
        #     swarmNew = self.trace_Swarm_Through_Lattice(swarm, h, T, parallel=True, fastMode=True)
        #     self.i += 1
        #     survival = swarmNew.survival_Rev()
        #     print(XNew,X, survival, swarmNew.longest_Particle_Life_Revolutions())
        #     Tsurvival = survival * self.lattice.totalLength / self.lattice.v0Nominal
        #     cost = -Tsurvival / T  # cost scales from 0 to -1.0
        #     return cost
        #
        # stabilityFunc = self.get_Stability_Function(numParticlesPerDim=1, cutoff=8.0,h=5e-6)
        #
        # def stability_Func_Wrapper(X):
        #     XNew = X.copy()
        #     for i in range(len(X)):  # change normalized bounds to actual
        #         XNew[i] = ((bounds[i][1] - bounds[i][0]) * float(X[i])/float(boundsNorm[i][1]-boundsNorm[i][0]) + bounds[i][0])
        #     return stabilityFunc(XNew)
        #
        # unstableCost = -1.5 * (self.lattice.totalLength / self.lattice.v0Nominal) / T  # typically unstable regions return an average
        # # of 1-2 revolution
        # numInit = int(maxEvals * .5)  # 50% is just random
        # xiRevs = .25  # search for the next points that returns an imporvement of at least this many revs
        # xi = (xiRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T
        # noiseRevs =1e-2 #small amount of noise to account for variability of results and encourage a smooth fit
        # noise = (noiseRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T
        #
        #
        # model = skopt.Optimizer(boundsNorm, n_initial_points=numInit, acq_func='EI', acq_optimizer='sampling',
        #                         acq_func_kwargs={"xi": xi, 'noise': noise}, n_jobs=-1)
        # self.resetXiCounts=0
        # self.countXi=False
        # def generate_Next_Point():
        #     if evals <= numInit-1:  # if still initializing the model
        #         x1 = int(np.random.rand() * stepsX)
        #         x2 = int(np.random.rand() * stepsY)
        #         XSample = [x1, x2]
        #     else:
        #         XSample = model.ask()
        #     if len(model.Xi) > 1 and evals > numInit-1:  # if the optimizer is suggesting duplicate points
        #         loops = 0
        #         while (loops < 10):  # try to force the optimizer to pick another point
        #             if np.any(np.sum((np.asarray(model.Xi) - np.asarray(XSample)) ** 2, axis=1) == 0):
        #                 print('DUPLICATE POINTS',XSample)
        #                 model.acq_func_kwargs['xi'] = model.acq_func_kwargs['xi'] * 2
        #                 model.update_next()
        #                 XSample = model.ask()
        #                 self.countXi=True
        #             else:
        #                 break
        #             loops += 1
        #         if loops == -9:
        #             raise Exception('COULD NOT STEER MODEL TO A NEW POINT')
        #         if self.countXi==True:
        #             self.resetXiCounts+=1
        #             if self.resetXiCounts==5:
        #                 model.acq_func_kwargs['xi']=xi
        #                 self.countXi=False
        #                 self.resetXiCounts=0
        #                 print('search reset!')
        #     return XSample
        #
        #
        #
        # evals = 0
        # t = time.time()
        #
        # print('starting')
        # while (evals < maxEvals): #TODO: REMOVE DUPLICATE CODE
        #     print(evals)
        #
        #     XSample=generate_Next_Point()
        #     print(XSample)
        #     if stability_Func_Wrapper(XSample) == True:  # possible solution
        #         cost = min_Func(XSample)
        #         model.tell(XSample, cost)
        #         evals += 1
        #
        #     else:  # not possible solution
        #         model.tell(XSample, unstableCost+np.random.rand()*1e-10) #add a little random noise to help
        #         #with stability. Doesn't work well when all the points are the same sometimes
        #
        #
        # print(time.time() - t)
        # sol = model.get_result()
        # solution = Solution()
        # solution.skoptSol = sol
        # x = [0, 0]
        # for i in range(len(sol.x)):  # change normalized bounds to actual
        #     x[i] = ((bounds[i][1] - bounds[i][0]) * float(sol.x[i]) / float(boundsNorm[i][1] - boundsNorm[i][0]) +
        #                bounds[i][0])
        # solution.x = x
        # solution.fun = -sol.fun * T * self.lattice.v0Nominal / self.lattice.totalLength
        # return solution
