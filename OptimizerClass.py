import poisson_disc
import copy
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
import scipy.spatial as sps
import matplotlib.pyplot as plt
import pySOT as ps
from ParaWell import ParaWell
from SwarmTracer import SwarmTracer
import globalMethods as gm




class phaseSpaceInterpolater:
    #simple class to help with using LinearNDInterpolater or NearestNDINterpolator. Keeps track of bounds violation
    #and apeture violations. The assumption is that the first two coordinates correspond to positions (y,z) where (y,z)
    #are positions relative to an apture centered at y=0,z=0.
    def __init__(self,X,Y,ap):
        #X: evaluation points array with shape n,m where n is number of points, and m is dimension
        #Y: values
        #apeture: apeture in the (x,y) plane to consider
        self.X=X
        self.Y=Y
        self.ap=ap
        self.xBounds=[] #list of tuples (min,max) for X. A hypercube
        for i in range(X.shape[1]): #loop through dimensions
             self.xBounds.append([self.X[:,i].min(),self.X[:,i].max()])
        self.xBounds=np.asarray(self.xBounds)
        self.interpolater=spi.NearestNDInterpolator(self.X, self.Y,rescale=True,tree_options={'copy_data':True})#spi.LinearNDInterpolator(self.X,self.Y,rescale=True)###
    def build(self):
        #some method, mostly the ones that use qhull, require the function to be called to build, which can take a while.
        #I would rather do that here
        testPoint=[]
        for bound in self.xBounds:
            testPoint.append((bound[1]+bound[0])) #add a point in the middle of a bound
        self.__call__(*testPoint)

    def __call__(self,*args):
        args=np.asarray(args)
        if np.any(args>self.xBounds[:,1])==True or np.any(args<self.xBounds[:,0])==True: #if any argument is out of bounds
            #return a nan
           return np.nan
        else: #if inside bounds, can still be outside apeture
            y,z=args[:2]
            if np.sqrt(y**2+z**2)<self.ap:
                return self.interpolater(*args)
            else:
                return np.nan

class LatticeOptimizer:
    def __init__(self, lattice):

        self.lattice = lattice
        self.helper=ParaWell() #custom class to help with parallelization
        self.i=0 #simple variable to track solution status
        self.particleTracer = ParticleTracer(lattice)
        self.h=None #timestep size
        self.T=None #maximum particle tracing time
        self.swarmTracer=SwarmTracer(self.lattice)
        self.latticePhaseSpaceFunc=None #function that returns the number of revolutions of a particle at a given
        #point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.stepsLens1=None #number of steps between min and max value to explore in parameter space
        self.stepsLens2=None # for lens 2
        self.evals=0 #number of evaluations of the lattice particle tracing function so far
        self.skoptModel=None #skopt model for use gaussian process minimization
        self.numInit=None #initial number of lattice tracing evaluations before asking the model for points
        self.xi=None #the value for skoptmodel to use to choose the next point. The next point must give this much
        #improvement
        self.xiResetCounter=0 #if the value xi was changed to search for new points, this is used to count to
        #resetting back the original value to promote for exploration after getting bogged down
        self.randomSampleList=[] ##list of random samples generated with low discrepancy method for searching paramter
        #space
    def update_Lattice(self,X):
        #Update the various paremters in the lattice and injections that are variable
        self.lattice.elList[2].fieldFact = X[0]
        self.lattice.elList[4].fieldFact = X[1]
    def compute_Phase_Space_Map_Function(self,X,swarmInitial,swarmCombiner):
        #return a function that returns a value for mnumber of revolutions at a given point in phase space. The phase
        #space is in the combiner's reference frame with the x component zero so the coordinates looke like (y,x,px,py,pz)
        #so the swarm is centered at the origin in the combiner
        #X: arguments to parametarize lattice
        #swarmInitial: swarm to trace through lattice that is initialized in phase space and centered at (0,0,0). This is
        #used as the coordinates for creating the phase space func. You would think to use the swarmCombiner, but I want
        #the coordinates centered at (0,0,0) and swarmInitial is identical to swarmCombiner transformed to (0,0,0)!

        self.update_Lattice(X)
        phaseSpacePoints=[] #holds the coordinates of the particles in phase space at the combiner output
        revolutions=[] #values of revolution for each particle in swarm


        swarmTraced=self.swarmTracer.trace_Swarm_Through_Lattice(swarmCombiner,self.h,self.T,fastMode=False)

        for i in range(swarmInitial.num_Particles()):
            q = swarmInitial.particles[i].qi.copy()[1:] #only y and z. x is the same
            p = swarmInitial.particles[i].pi.copy()

            Xi = np.append(q, p)  # phase space coords are 2 position values and 3 momentum

            phaseSpacePoints.append(Xi)
            revolutions.append(swarmTraced.particles[i].revolutions)

        print(swarmTraced.survival_Rev(),swarmTraced.longest_Particle_Life_Revolutions())

        apeture=self.lattice.elList[self.lattice.combinerIndex+1].ap
        latticePhaseSpaceFunc=phaseSpaceInterpolater(np.asarray(phaseSpacePoints), revolutions,apeture)

        return latticePhaseSpaceFunc



    def get_Stability_Function(self,qMax=250e-6,numParticlesPerDim=2,h=5e-6,cutoff=8.0,funcType='bool'):
        #return a function that can evaluated at the lattice parameters, X, and return wether True for stable
        #and False for unstable
        #qMax: #if using multiple particles this defineds the bounds of the initial positions square,meters
        #numParticlesPerDim: Number of particles on each grid edge. Total number is this squared.
        #h: timestep size, seconds
        #cutoff: Number of revolutions of survival to be considered True for survival
        #funcType: wether to return True/False for survival or number of revolutions.
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
            self.lattice.elList[2].fieldFact=X[0]
            self.lattice.elList[4].fieldFact = X[1]
            revolutionsList=[]
            for particle in swarm:
                particle=self.particleTracer.trace(particle.copy(),h,T,fastMode=True)
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

    def generate_Next_Point(self):
        #generate the next point to test. This generator first iterates over the initial random values, and then uses
        #the model to propose new values. It uses a low discrepancy poission disc sampler to evenly explore the space.
        # this is done by first initializing points, shuffling them, and workig through them. When the list is depleted
        #more are generated. Has to be generated in a batch though, and the amount generated is variable, it is not a
        #repeatable process. self.evals is used to keep track of number of expensive evaluations. An expensize evaluation
        #is an evaluation in a stable region, these take far longer than evaluation the stability. This method also prevents
        #duplicate points from being suggested (seems like scikit-optimize should do this, but it doesn't). When a duplicate
        #point is suggested this usually indicates the model is getting bogged down so I change the selection criteria to
        #search more then after xiReset evaluation it goes back to the orignial value


        #todo: catch duplicating random points
        xiReset=5 #this many evaluations after a duplicate point before resetting to a lower value
        if self.evals <= self.numInit - 1:  # if still initializing the model
            if len(self.randomSampleList)==0: #either the first evaluation or the list has been exhausted
                points=5*self.numInit #from experience I know that at least this amount is requied based on how sparse
                #expensive evaluations are
                r = np.sqrt(2 * (1.0/points) / np.pi) #this is used to get near the required amount of points with
                #poission disc method, which does not give an exact number of points
                samples = poisson_disc.Bridson_sampling(dims=np.asarray([1.0,1.0]), k=100, radius=r)
                np.random.shuffle(samples) #shuffle array in place
                # convert to the integer value that the skopt model requires
                samples[:,0]=samples[:,0]*self.stepsLens1
                samples[:, 1] = samples[:, 1] * self.stepsLens2
                samples=samples.astype(int)
                self.randomSampleList=list(samples)
            XSample=list(self.randomSampleList.pop()) #pop a value off the list. The value is removed and the list is
            # shorter. model.tell requires a list so need to cast
        else:
            XSample = self.skoptModel.ask()
        if self.evals > self.numInit - 1:  # check if the optimizer is suggesting
            # duplicate points, but only if have asked it, not a random value
            loops = 0
            while (loops < 10):  # try to force the optimizer to pick another point if there are duplicates
                if np.any(np.sum((np.asarray(self.skoptModel.Xi) - np.asarray(XSample)) ** 2, axis=1) == 0):  # if any duplicate
                    # points
                    self.skoptModel.acq_func_kwargs['xi'] = self.skoptModel.acq_func_kwargs['xi'] * 2 #increase the required
                    #improvement. Eventually this will force the model to make a more random guess
                    self.skoptModel.update_next()
                    XSample = self.skoptModel.ask()
                    loops += 1
                    self.xiResetCounter=0 #set the counter that will count up to xiReset before resetting xi
                    print('DUPLICATE POINTS', XSample,'NEW POINTS',XSample)
                else: #if no duplicates move on
                    break
            if loops == 9: #failed
                raise Exception('COULD NOT STEER MODEL TO A NEW POINT')
            if self.skoptModel.acq_func_kwargs['xi'] != self.xi: #only increment if the value is different (has been
                #changed)
                self.xiResetCounter+=1
                if self.xiResetCounter==xiReset:
                    print('xi was reset')
                    self.skoptModel.acq_func_kwargs['xi'] = self.xi
                    self.xiResetCounter=0
            print(self.skoptModel.acq_func_kwargs['xi'] )
        return XSample


    def maximize_Suvival_Through_Lattice(self,h,T,numParticles=5000,pMax=5e0,returnBestSwarm=False,parallel=False,
                                         maxEvals=100,bounds=None,precision=10e-3):
        self.h=h
        self.T=T
        #make a swarm that whos position spans the next element's input
        qyMax=self.lattice.elList[self.lattice.combinerIndex+1].ap
        qzMax=qyMax
        pxMax=1.1
        #now estimate the maximum possible transverse velocity
        deltav=(self.lattice.v0Nominal*qyMax/self.lattice.combiner.Lo)
        pyMax=deltav
        pzMax=deltav


        #TODO: INITIALIZE A CIRCULAR SWARM
        swarmInitial=self.swarmTracer.initalize_Random_Swarm_In_Phase_Space(qyMax,qzMax,pxMax,pyMax,pzMax, numParticles,
                                                                            cylinderivalApeture=qyMax)

        swarmCombiner=self.swarmTracer.move_Swarm_To_Combiner_Output(swarmInitial)


        if bounds is None:
            bounds=[(0.0, .5), (0.0, .5)]
        class Solution:
            #because I renormalize bounds and function values, I used this solution class to easily access the more
            #familiar values that I am interested in
            def __init__(self):
                self.skoptSol=None #to hold the skopt solutiob object
                self.x=None #list for real paremters values
                self.fun=None #for real solution value

        self.stepsLens1=int((bounds[0][1]-bounds[0][0])/precision)
        self.stepsLens2 = int((bounds[1][1] - bounds[1][0]) / precision)
        if self.stepsLens1+self.stepsLens2<=1:
            raise Exception('THERE ARE NOT ENOUGH POINTS IN SPACE TO EXPLORE, MUST BE MORE THAN 1')
        boundsNorm = [(0, self.stepsLens1), (0, self.stepsLens2)]
        print('bounds',boundsNorm)
        boundsInjector = [(.15, .25), (.5, 1.5), (-.1, .1)] # (LoMin,LoMax),(LiMin,LiMax),(LOffsetMin,LOffsetMax)

        def min_Func(X):
            XNew = X.copy()
            for i in range(len(X)):  # change normalized bounds to actual
                XNew[i] = ((bounds[i][1] - bounds[i][0]) * float(X[i])/float(boundsNorm[i][1]-boundsNorm[i][0]) + bounds[i][0])
            print('start lattice tracing')
            t=time.time()
            func = self.compute_Phase_Space_Map_Function(XNew, swarmInitial, swarmCombiner)
            print('done mode matching',time.time()-t)
            gm.lattice=self.lattice
            gm.func=func
            t=time.time()
            print('start mode match')
            survival = -gm.solve()
            print('done mode match',time.time()-t)
            print(survival)
            # self.update_Lattice(XNew)
            # swarm=self.swarmTracer.trace_Swarm_Through_Lattice(swarmCombiner, self.h, self.T, fastMode=True)
            # survival=swarm.survival_Rev()
            print(survival)
            Tsurvival = survival * self.lattice.totalLength / self.lattice.v0Nominal
            cost = -Tsurvival / T  # cost scales from 0 to -1.0
            print(cost)
            return cost

        stabilityFunc = self.get_Stability_Function(numParticlesPerDim=1, cutoff=8.0,h=5e-6)

        def stability_Func_Wrapper(X):
            XNew = X.copy()
            for i in range(len(X)):  # change normalized bounds to actual
                XNew[i] = ((bounds[i][1] - bounds[i][0]) * float(X[i])/float(boundsNorm[i][1]-boundsNorm[i][0]) + bounds[i][0])
            return stabilityFunc(XNew)

        unstableCost = -1.5 * (self.lattice.totalLength / self.lattice.v0Nominal) / T  # typically unstable regions return an average
        # of 1-2 revolution
        self.numInit = int(maxEvals * .5)  # 50% is just random
        xiRevs = .25  # search for the next points that returns an imporvement of at least this many revs
        self.xi = (xiRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T
        noiseRevs =1e-2 #small amount of noise to account for variability of results and encourage a smooth fit
        noise = (noiseRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T


        self.skoptModel = skopt.Optimizer(boundsNorm, n_initial_points=self.numInit, acq_func='EI', acq_optimizer='sampling',
                                          acq_func_kwargs={"xi": self.xi, 'noise': noise}, n_jobs=-1)



        self.evals = 0
        t = time.time()

        print('starting')
        while (self.evals < maxEvals): #TODO: REMOVE DUPLICATE CODE
            print(self.evals)
            XSample=self.generate_Next_Point()
            print(XSample)
            if stability_Func_Wrapper(XSample) == True:  # possible solution
                print('stable')
                cost = min_Func(XSample)
                self.skoptModel.tell(XSample, cost)
                self.evals += 1

            else:  # not possible solution
                self.skoptModel.tell(XSample, unstableCost+np.random.rand()*noiseRevs) #add a little random noise to help
                #with stability. Doesn't work well when all the points are the same sometimes


        print(time.time() - t)
        sol = self.skoptModel.get_result()
        solution = Solution()
        solution.skoptSol = sol
        x = [0, 0]
        for i in range(len(sol.x)):  # change normalized bounds to actual
            x[i] = ((bounds[i][1] - bounds[i][0]) * float(sol.x[i]) / float(boundsNorm[i][1] - boundsNorm[i][0]) +
                    bounds[i][0])
        solution.x = x
        solution.fun = -sol.fun * T * self.lattice.v0Nominal / self.lattice.totalLength
        return solution
