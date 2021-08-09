import skopt
from ParticleTracerClass import ParticleTracer
import numpy as np
from ParticleClass import Swarm
import scipy.optimize as spo
import time
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from ParaWell import ParaWell
from SwarmTracerClass import SwarmTracer
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
        self.elementIndices=None
        # self.minCombinerShaperDistance=.1 #minimum distance between the combiner and the shaper magnet
        # self.injectorBounds=[] #list of bounds for the injection system, Lo, Li, LOffset
        # self.injectorBounds.append((.1,.25)) #Object distance, Lo
        # LiMin=self.lattice.combiner.Lo*1.5+self.minCombinerShaperDistance
        # self.injectorBounds.append((LiMin,LiMin+1.0)) #Image distance, Li
        # self.injectorBounds.append((-self.lattice.combiner.Lo/2,self.lattice.combiner.Lo/2)) #focus offset inside the combiner, LOffset. Keep in mind
        #the signs!!!
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
                       plotName='stabilityPlot',cutoff=8.0,h=5e-6,showPlot=True,modulation='01'):
        #bounds: region to search over for instability
        #qMax: maximum dimension in transverse directions for initialized particles
        #numParticlesPerDim: Number of particles along y and z axis so total is numParticlesPerDim**2.
        #gridPoints: number of points per axis to test stability. Total is gridPoints**2
        #cutoff: Maximum revolutions below this value are considered unstable
        #modulation: Which elements in the lattice to modulate the field of
        self.modulation=modulation

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

    def optimize_Grid(self,coords0Arr,val0Arr,bounds,maxIter):
        # np.random.seed(int.from_bytes(os.urandom(4),byteorder='little'))
        newCoords=coords0Arr.copy()
        newVals=val0Arr.copy()
        plt.scatter(newCoords[:, 0], newCoords[:, 1])
        fitFunc=None
        jitterAmplitude=[]
        for bound in bounds:
            jitterAmplitude.append(1e-3*(bound[1]-bound[0]))
        jitterAmplitude=np.asarray(jitterAmplitude)
        for i in range(maxIter):
            print('------',i)
            fitFunc=spi.RBFInterpolator(newCoords,newVals,smoothing=1e-6) #a little bit of smooth helps
            def minFunc(X):
                return 1/np.abs(fitFunc([X])[0])
            probeVals=[]
            probeCoords=[]
            for j in range(10):
                sol=spo.differential_evolution(minFunc,bounds,mutation=1.5)
                probeVals.append(1/sol.fun)
                probeCoords.append(sol.x)
            bestProbeCoords=probeCoords[np.argmax(np.asarray(probeVals))]
            if i != maxIter-1: #if the last one don't jitter
                bestProbeCoords=bestProbeCoords+ jitterAmplitude*2*(np.random.rand(len(bounds))-.5) #add a small amount
                #of jitter to overcome small chance of getting stuck at a peak. Also, if the coords aren't improving
                #more than this jitter, that's a good sign we're done
            newVal=self.revFunc(bestProbeCoords,parallel=True)
            print(newVal,bestProbeCoords)
            newCoords=np.row_stack((newCoords,bestProbeCoords))
            plt.scatter(*bestProbeCoords,marker='x',c='r')
            newVals=np.append(newVals,newVal)
        print('initial value:',np.round(np.max(val0Arr),2),'new value: ',np.round(np.max(newVals),2))
        num=250
        xPlot=np.linspace(0,1,num=num)
        coords=np.asarray(np.meshgrid(xPlot,xPlot)).T.reshape(-1,2)
        image=fitFunc(coords).reshape(num,num)

        plt.imshow(np.flip(image,axis=0),extent=[0,1.0,0,1.0])
        plt.show()
        return newCoords[np.argmin(newVals)]
    def project_Swarm(self,swarm0,h,T,checkStability,copy=True):
        #project a swarm through the lattice.
        pass
    def revFunc(self,X,h=1e-5,T=1.0,numParticles=1000, parallel=False):
        self.update_Lattice(X)
        minRevs = 5.0  # test particle must achieve at least this many revolutions
        swarmTest=self.swarmTracer.initialize_Stablity_Testing_Swarm(1e-3)
        swarmTestNew = self.swarmTracer.trace_Swarm_Through_Lattice(swarmTest, h,
                                                               1.5 * minRevs * self.lattice.totalLength / 200.0,
                                                               parallel=parallel)
        t=time.time()
        for i in range(10):
            swarm0=self.swarmTracer.initalize_PseudoRandom_Swarm_In_Phase_Space(2.5e-3,5.0,1.0,numParticles,sameSeed=True)
            swarm0=self.swarmTracer.move_Swarm_To_Combiner_Output(swarm0)
        print((time.time()-t)/10)
        stable = False
        for particle in swarmTestNew:
            if particle.revolutions > minRevs:
                stable = True
        if stable == False:
            return 0.0
        else:
            swarm = self.swarmTracer.trace_Swarm_Through_Lattice(swarm0, h, T, parallel=parallel, fastMode=True)
            # self.lattice.show_Lattice(swarm=swarm,trueAspectRatio=False,showTraceLines=True)
            return swarm.survival_Rev()
    def update_Lattice(self,X):
        for i in range(len(X)):
            self.lattice.elList[self.elementIndices[i]].fieldFact=X[i]
    def optimize_Magnetic_Field(self,elementIndices,bounds,num0,maxIter=10):
        # optimize magnetic field of the lattice by tuning element field strengths. This is done by first evaluating the
        #system over a grid, then using a non parametric model to find the optimum.
        #elementIndices: tuple of indices of elements to tune the field strength
        #bounds: list of tuples of (min,max) for tuning
        #maxIter: maximum number of optimization iterations with non parametric optimizer
        assert np.unique(np.asarray(elementIndices)).shape[0]==len(elementIndices) #ensure no duplicates
        assert len(bounds)==len(elementIndices) #ensure bounds for each element being swept
        print('here')
        self.elementIndices=elementIndices
        BArrList=[]
        for bound in bounds:
            BArrList.append(np.linspace(bound[0], bound[1], num0))
        coordsArr = np.asarray(np.meshgrid(*BArrList)).T.reshape(-1, len(elementIndices))
        # gridResults=[]
        # for coord in coordsArr:
        #     gridResults.append(self.revFunc(coord))
        # gridResults=np.asarray(gridResults)
        # print(np.max(gridResults))
        # plt.plot(tem)
        # plt.show()
        gridResults = np.asarray(self.helper.parallel_Problem(self.revFunc, coordsArr, onlyReturnResults=True))
        return self.optimize_Grid(coordsArr,gridResults,bounds,maxIter)

#
# test=LatticeOptimizer(None)
# test.optimize_Magnetic_Field((1,2,3),[(0,1),(0,1),(0,1)],30,10)