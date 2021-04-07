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
        self.interpolater=spi.NearestNDInterpolator(self.X, self.Y,rescale=True,tree_options={'copy_data':True})#spi.LinearNDInterpolator(self.X,self.Y,rescale=True)####
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
        self.hardEvals=0 #number of evaluations of the lattice particle tracing function so far
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
        #X: lattice parameters in the form of an iterable
        self.lattice.elList[0].fieldFact = X[0]
        self.lattice.elList[2].fieldFact = X[1]
    def compute_Phase_Space_Map_Function(self,X,swarmCombiner):
        #return a function that returns a value for number of revolutions at a given point in phase space. The phase
        #space is in the combiner's reference frame with the x component zero so the coordinates looke like (y,x,px,py,pz)
        #so the swarm is centered at the origin in the combiner
        #X: arguments to parametarize lattice
        #swarmInitial: swarm to trace through lattice that is initialized in phase space and centered at (0,0,0). This is
        #used as the coordinates for creating the phase space func. You would think to use the swarmCombiner, but I want
        #the coordinates centered at (0,0,0) and swarmInitial is identical to swarmCombiner transformed to (0,0,0)!

        self.update_Lattice(X)
        phaseSpacePoints=[] #holds the coordinates of the particles in phase space at the combiner output
        revolutions=[] #values of revolution for each particle in swarm
        swarmTraced=self.swarmTracer.trace_Swarm_Through_Lattice(swarmCombiner,self.h,self.T,fastMode=True)
        for i in range(swarmTraced.num_Particles()):
            q = swarmTraced.particles[i].qi.copy()[1:] #only y and z. x is the same all (0)
            p = swarmTraced.particles[i].pi.copy()
            Xi = np.append(q, p)  # phase space coords are 2 position values and 3 momentum (y,z,px,py,pz)
            phaseSpacePoints.append(Xi)
            revolutions.append(swarmTraced.particles[i].revolutions)
        print(swarmTraced.survival_Rev(),swarmTraced.longest_Particle_Life_Revolutions())
        apeture=self.lattice.elList[self.lattice.combinerIndex+1].ap #apeture of next element
        latticePhaseSpaceFunc=phaseSpaceInterpolater(np.asarray(phaseSpacePoints), revolutions,apeture)
        return latticePhaseSpaceFunc


    def get_Stability_Function(self,qMax=.5e-3,numParticlesPerDim=2,h=5e-6,cutoff=8.0,funcType='bool'):
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
            self.update_Lattice(X)
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

        revolutionFunc=self.compute_Revolution_Func_Over_Grid(bounds=bounds,qMax=qMax,
                                                             numParticlesPerDim=numParticlesPerDim, gridPoints=gridPoints,cutoff=cutoff,h=h)
        plotxArr=np.linspace(bounds[0][0],bounds[0][1],num=250)
        plotyArr = np.linspace(bounds[1][0], bounds[1][1], num=250)
        image=np.empty((plotxArr.shape[0],plotyArr.shape[0]))
        for i in range(plotxArr.shape[0]):
            for j in range(plotyArr.shape[0]):
                image[j,i]=revolutionFunc(plotxArr[i],plotyArr[j])

        image=np.flip(image,axis=0)
        extent=[bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]]
        plt.imshow(image,extent=extent)

        plt.suptitle(plotName)
        plt.title('Yellow indicates more stability, purple indicates more unstable')
        plt.xlabel('Field strength of tunable 1')
        plt.ylabel('Field strength of tunable 2')
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
        if self.hardEvals <= self.numInit - 1:  # if still initializing the model
            if len(self.randomSampleList)==0: #if the low discrepancy list has been exhausted
                print("LOW DISCREPANCY LIST EXHAUSTED")
                XSample=np.random.rand(2)
                XSample[0]=XSample[0]*self.stepsLens1
                XSample[1] = XSample[1]*self.stepsLens2
            else:
                XSample=list(self.randomSampleList.pop()) #pop a value off the list. The value is removed and the list is
                # shorter. model.tell requires a list so need to cast
        else:
            XSample = self.skoptModel.ask()
        if self.hardEvals > self.numInit - 1:  # check if the optimizer is suggesting
            # duplicate points, but only if have asked it, not a random value
            loops = 0
            while (loops < 10):  # try to force the optimizer to pick another point if there are duplicates
                if np.any(np.sum((np.asarray(self.skoptModel.Xi) - np.asarray(XSample)) ** 2, axis=1) == 0):  # if any duplicate
                    # points
                    print('DUPLICATE POINT', XSample)
                    self.skoptModel.acq_func_kwargs['xi'] = self.skoptModel.acq_func_kwargs['xi'] * 2 #increase the required
                    #improvement. Eventually this will force the model to make a more random guess
                    self.skoptModel.update_next()
                    XSample = self.skoptModel.ask()
                    loops += 1
                    self.xiResetCounter=0 #set the counter that will count up to xiReset before resetting xi
                    print('NEW POINT',XSample)
                else: #if no duplicates move on
                    break
            if loops == 9: #failed
                raise Exception('COULD NOT STEER MODEL TO A NEW POINT')
            if self.skoptModel.acq_func_kwargs['xi'] != self.xi: #only increment if the value is different (has been
                #changed)
                self.xiResetCounter+=1
                if self.xiResetCounter==xiReset:
                    self.skoptModel.acq_func_kwargs['xi'] = self.xi
                    self.xiResetCounter=0
        return XSample
    def generate_Monte_Carlo_Bounds(self,numPoints=250,numParticles=5000,survivalFrac=.95,sameSeed=True):
        #this function generates the 5 bounds for the monte carlo integration of tracing particles through the lattice.
        #it is crucial to have the bounds of the integration encompass the particl'es that will be mode matched. However,
        #having too much integration volume rapidly becomes computationally expensive. Thus, this method minimizes the required
        #limits. This is done by finding the limits of numPoints of configurations of the injection system with a cloud
        #of numParticles. To avoid wasing resources on outlier, the integreation bound that captures survivalFrac of particles
        #is used. The limit of integration is calculated for each configuration and that largest is used.
        #the position values are given by simple geometry, however the velocity limits are more complicated and require
        #this method. Remember that the monte carlo integration has square limits
        #numPonts: Number of configurations to test
        #numParticles: Number of particles to test each configuration with
        #survivalFrac: Fraction of particles that corresponds to the limit. This is to prevent wasted resources with outliers
        #sameSeed: Wether the use the same seed to get repeatable results. Otherwise the sobol seuqnece uses the numpy random
        #generator
        if sameSeed==True:
            np.random.seed(42)

        def wrapper(args):
            #this does all the wrangling of unpacking the results and finding the survivalFrac cutoff values
            Lo, Li, LOffset = args
            swarm = self.swarmTracer.initialize_Swarm_At_Combiner_Output(Lo, Li, LOffset, labFrame=False,
                                                                    clipForNextApeture=True,
                                                                    numParticles=numParticles, fastMode=True)  # 5000
            qVec, pVec = swarm.vectorize(onlyUnclipped=True)
            pxArr = pVec[:, 0] + self.lattice.v0Nominal #set the pxArr to an offset from the nominal value (not necesarily
            #the mean though)
            pyArr = pVec[:, 1]
            pzArr = pVec[:, 2]
            #I'm avoiding using 3 different loops here. Kind of iffy if this actuall saves space honestly.
            #The value that allows survivalFrac is found by creating an array of limits then looping through the array
            #and counting the number of particles with absolute value less than that limit for each coordinate.
            valMaxList = [pxArr.max(), pyArr.max(), pzArr.max()]
            arrList = [pxArr, pyArr, pzArr]
            integrationMaxList = [] #the result that corresponds to 95% survival
            for i in range(len(valMaxList)): #loop over the three dimensions we're testing (px,py,pz)
                arr = arrList[i]
                valMax = valMaxList[i]
                valArr = np.linspace(valMax, 0)
                for val in valArr: #loop through and test the cutoff
                    frac = np.sum(np.abs(arr) < val) / arr.shape[0]
                    if frac <= survivalFrac:
                        integrationMaxList.append(val)
                        break #move onto the next dimension
            return integrationMaxList

        bounds = [(.15, .25), (.5, 1.5), (-.1, .1)] #bounds of the injector system. Li,Lo,LOffset, meters
        sampler = skopt.sampler.Sobol() #low discrepancy sampling
        samples = sampler.generate(bounds, numPoints)
        argList = samples
        results = self.helper.parallel_Problem(wrapper, argList, onlyReturnResults=True)
        pxList = []
        pyList = []
        pzList = []
        for result in results:
            pxList.append(result[0])
            pyList.append(result[1])
            pzList.append(result[2])

        pxLimit=np.max(np.asarray(pxList)) #the momentum limits of integration
        pyLimit=np.max(np.asarray(pyList))#the momentum limits of integration
        pzLimit=np.max(np.asarray(pzList))#the momentum limits of integration
        #use the limits of the combiner and the next element as the limits of the integration
        #remember there is no x limit. All particles are assumed to start at x=0
        qyLimit=self.lattice.elList[self.lattice.combinerIndex+1].ap*1.01 #the position limits of integration
        qzLimit=self.lattice.combiner.apz*1.01#the position limits of integration
        if sameSeed==True:
            np.random.seed(int(time.time()))#reset the seed, kind of
        return qyLimit,qzLimit,pxLimit,pyLimit,pzLimit
    def maximize_Suvival_Through_Lattice(self, h, T, numParticles=30000, maxHardsEvals=100, bounds=None, precision=10e-3):
        self.h=h
        self.T=T

        print('generating monte carlo bounds')
        qyLimit,qzLimit,pxLimit,pyLimit,pzLimit=self.generate_Monte_Carlo_Bounds()
        print('done generating monte carlo bounds')
        swarmInitial=self.swarmTracer.initalize_Random_Swarm_In_Phase_Space(qyLimit, qzLimit, pxLimit, pyLimit, pzLimit, numParticles)
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

        def min_Func(X):
            XNew = X.copy()
            for i in range(len(X)):  # change normalized bounds to actual
                XNew[i] = ((bounds[i][1] - bounds[i][0]) * float(X[i])/float(boundsNorm[i][1]-boundsNorm[i][0]) + bounds[i][0])
            print(XNew)
            print('start lattice tracing')
            t=time.time()
            func = self.compute_Phase_Space_Map_Function(XNew,swarmCombiner)
            print('done lattice tracing',time.time()-t)
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

        self.numInit = int(maxHardsEvals * .5)  # 50% is just random
        #generate random sample list. This really needs to be improved, I liked the feature where I added one point
        #at a time but that requires using random numbers when this is exhausted
        points = 5 * self.numInit  # from experience I know that at least this amount is requied based on how sparse
        # expensive evaluations are
        r = np.sqrt(2 * (1.0 / points) / np.pi)  # this is used to get near the required amount of points with
        # poission disc method, which does not give an exact number of points
        samples = poisson_disc.Bridson_sampling(dims=np.asarray([1.0, 1.0]), k=100, radius=r)
        np.random.shuffle(samples)  # shuffle array in place
        # convert to the integer value that the skopt model requires
        samples[:, 0] = samples[:, 0] * self.stepsLens1
        samples[:, 1] = samples[:, 1] * self.stepsLens2
        samples = samples.astype(int)
        self.randomSampleList = list(samples)

        unstableCost = -1.5 * (self.lattice.totalLength / self.lattice.v0Nominal) / T  # typically unstable regions return an average
        # of 1 to 2 revolution
        xiRevs = .25  # search for the next points that returns an imporvement of at least this many revs
        self.xi = (xiRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T
        noiseRevs =1e-2 #small amount of noise to account for variability of results and encourage a smooth fit
        noise = (noiseRevs * (self.lattice.totalLength / self.lattice.v0Nominal)) / T


        self.skoptModel = skopt.Optimizer(boundsNorm, n_initial_points=self.numInit, acq_func='EI', acq_optimizer='sampling',
                                          acq_func_kwargs={"xi": self.xi, 'noise': noise}, n_jobs=-1)



        self.hardEvals = 0
        t = time.time()

        print('starting')
        while (self.hardEvals < maxHardsEvals):
            print(self.hardEvals)
            XSample=self.generate_Next_Point()
            print(XSample)
            if stability_Func_Wrapper(XSample) == True:  # possible solution
                print('stable')
                cost = min_Func(XSample)
                self.skoptModel.tell(XSample, cost)
                self.hardEvals += 1

            else:  # not possible solution
                self.skoptModel.tell(XSample, unstableCost+np.random.rand()*noiseRevs) #add a little random noise to help
                #with stability. Doesn't work well when all the points are the same sometimes


        print(time.time() - t)
        sol = self.skoptModel.get_result()
        print(sol)
        solution = Solution()
        solution.skoptSol = sol
        x = [0, 0]
        for i in range(len(sol.x)):  # change normalized bounds to actual
            x[i] = ((bounds[i][1] - bounds[i][0]) * float(sol.x[i]) / float(boundsNorm[i][1] - boundsNorm[i][0]) +
                    bounds[i][0])
        solution.x = x
        solution.fun = -sol.fun * T * self.lattice.v0Nominal / self.lattice.totalLength
        return solution
