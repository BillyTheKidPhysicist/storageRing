import sys
import numpy.linalg as npl
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
    def __init__(self,swarm):
        #X: evaluation points array with shape n,m where n is number of points, and m is dimension
        #Y: values
        #apeture: apeture in the (x,y) plane to consider
        xList=[]
        yList=[]
        for particle in swarm:
            q=particle.qi[1:]
            p=particle.p
            X=np.append(q,p)
            xList.append(X)
            yList.append(particle.revolutions)
        self.interpolater=spi.NearestNDInterpolator(xList, yList,rescale=True,tree_options={'copy_data':True})#spi.LinearNDInterpolator(self.X,self.Y,rescale=True)####
    def __call__(self,swarm):
        totalRevs=0
        for particle in swarm:
            q=particle.qi[1:]
            p=particle.p
            X=np.append(q,p)
            totalRevs+=self.interpolater(X)[0]
        return totalRevs
class LatticeOptimizer:
    def __init__(self, latticeRing,latticeInjector):

        self.latticeRing = latticeRing
        self.latticeInjector=latticeInjector
        self.helper=ParaWell() #custom class to help with parallelization
        self.i=0 #simple variable to track solution status
        self.particleTracerRing = ParticleTracer(latticeRing)
        self.particleTracerInjector=ParticleTracer(latticeInjector)
        self.swarmTracerInjector=SwarmTracer(self.latticeInjector)
        self.swarmInjectorInitial=None #object to hold the injector swarm object
        self.swarmRingInitial=None #object to hold the ring swarm object that will generate the mode match function
        self.h=None #timestep size
        self.T=None #maximum particle tracing time
        self.swarmTracerRing=SwarmTracer(self.latticeRing)
        self.phaseSpaceFunc=None #function that returns the number of revolutions of a particle at a given
        #point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.elementIndices=None

        self.generate_Swarms()

    def generate_Swarms(self):
        qMaxInjector=2.5e-3
        pTransMaxInjector=10.0
        numParticlesInjector=1000
        PxMaxInjector=.5
        firstApertureRing=self.latticeRing.elList[self.latticeRing.combinerIndex+1].ap

        self.swarmInjectorInitial=self.swarmTracerInjector.initalize_PseudoRandom_Swarm_In_Phase_Space(qMaxInjector,
                                                                pTransMaxInjector,PxMaxInjector,numParticlesInjector,sameSeed=True)
        # self.find_Injector_Mode_Match_Bounds()
        injectorBounds=[(-0.0077, 0.007642), (-0.005975, 0.005904), (-0.681542, 0.568391), (-9.577788, 9.41937), (-7.757492, 7.658181)]
        numParticlesRing=30000
        pxMaxRing=1.1*max(np.abs(injectorBounds[2][1]),np.abs(injectorBounds[2][0]))
        pTransMaxRing=1.1*max(np.abs(injectorBounds[3][1]),np.abs(injectorBounds[3][0]))

        self.swarmRingInitial=self.swarmTracerRing.initalize_PseudoRandom_Swarm_In_Phase_Space(firstApertureRing,pTransMaxRing,
                                                                                        pxMaxRing,numParticlesRing,sameSeed=True)


    def find_Injector_Mode_Match_Bounds(self):
        #todo: use this to set hard limits on output of combiner?
        injectorParamsBounds=(.05,1.0)
        numGridPointsPerDim = 30
        xArr = np.linspace(injectorParamsBounds[0], injectorParamsBounds[1], numGridPointsPerDim)
        coords = np.asarray(np.meshgrid(xArr, xArr)).T.reshape(-1, 2)
        fracCutOff=.98 # for any given extrema, chose the value that bounds this fraction of particles to avoid wasting
        #resources on outliers
        def wrapper(X): # need a wrapper to update lattice before tracing
            self.update_Injector_Lattice(X)
            return self.project_Injector_Swarm_To_Combiner_End()
        projectedSwarmsList=self.helper.parallel_Problem(wrapper,coords,onlyReturnResults=True)
        phaseSpaceExtremaList=[]
        # temp=[]
        for swarm in projectedSwarmsList:
            # temp.append(swarm.num_Particles())
            numParticlesSurvived=swarm.num_Particles()
            if numParticlesSurvived>self.swarmInjectorInitial.num_Particles()//10: #Too few particles is not worth analyzing
                qVec, pVec = swarm.vectorize()
                pVec[:,0]+=self.latticeInjector.v0Nominal #remove the nominal velocity from px
                phaseSpaceVec=np.column_stack((qVec,pVec))
                swarmExtrema=[] #to hold ymin,ymax,zmin,zmax,pxmin,pxmax,pymin,pymax,pzmin,pzmax
                for i in range(1,6): #start at 1 to exclude x position because we know it's zero
                    variableArrSorted=np.sort(phaseSpaceVec[:,i]) #sorted from smallest to largest, incldues negative
                    variableMin=variableArrSorted[int(numParticlesSurvived*(1-fracCutOff))]
                    variableMax=variableArrSorted[int(numParticlesSurvived*fracCutOff)-1]
                    swarmExtrema.extend([variableMin,variableMax])
                phaseSpaceExtremaList.append(swarmExtrema)
        # temp=np.asarray(temp)
        # plt.imshow(temp.reshape(numGridPointsPerDim,numGridPointsPerDim))
        # plt.show()
        phaseSpaceExtremaArr=np.asarray(phaseSpaceExtremaList)
        boundsList=[] #list to hold bounds of swarm to trace in storage ring
        for i in range(phaseSpaceExtremaArr.shape[1]//2): #loop over each coordinate, ie y,z,px etc
            boundsList.append((np.round(phaseSpaceExtremaArr[:,2*i].min(),6),np.round(phaseSpaceExtremaArr[:,2*i+1].max(),6))) #need to double
            #i because there are 2 columns per variable, a min and a max
        print('y','z','px','py','pz')
        print(boundsList)
    def compute_Phase_Space_Map_Function(self,X,swarmCombiner):
        #return a function that returns a value for number of revolutions at a given point in phase space. The phase
        #space is in the combiner's reference frame with the x component zero so the coordinates looke like (y,x,px,py,pz)
        #so the swarm is centered at the origin in the combiner
        #X: arguments to parametarize lattice
        #swarmInitial: swarm to trace through lattice that is initialized in phase space and centered at (0,0,0). This is
        #used as the coordinates for creating the phase space func. You would think to use the swarmCombiner, but I want
        #the coordinates centered at (0,0,0) and swarmInitial is identical to swarmCombiner transformed to (0,0,0)!

        self.update_Ring_Lattice(X)
        phaseSpacePoints=[] #holds the coordinates of the particles in phase space at the combiner output
        revolutions=[] #values of revolution for each particle in swarm
        swarmTraced=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmCombiner,self.h,self.T,fastMode=True)
        for i in range(swarmTraced.num_Particles()):
            q = swarmTraced.particles[i].qi.copy()[1:] #only y and z. x is the same all (0)
            p = swarmTraced.particles[i].pi.copy()
            Xi = np.append(q, p)  # phase space coords are 2 position values and 3 momentum (y,z,px,py,pz)
            phaseSpacePoints.append(Xi)
            revolutions.append(swarmTraced.particles[i].revolutions)
        apeture=self.latticeRing.elList[self.latticeRing.combinerIndex+1].ap #apeture of next element
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
        T=(cutoff+.25)*self.latticeRing.totalLength/self.latticeRing.v0Nominal #number of revolutions can be a bit fuzzy so I add
        # a little extra to compensate
        if numParticlesPerDim==1:
            qInitialArr=np.asarray([np.asarray([-1e-10,0.0,0.0])])
        else:
            qTempArr=np.linspace(-qMax,qMax,numParticlesPerDim)
            qInitialArr_yz=np.asarray(np.meshgrid(qTempArr,qTempArr)).T.reshape(-1,2)
            qInitialArr=np.column_stack((np.zeros(qInitialArr_yz.shape[0])-1e-10,qInitialArr_yz))
        pi=np.asarray([-self.latticeRing.v0Nominal,0.0,0.0])
        swarm=Swarm()
        for qi in qInitialArr:
            swarm.add_Particle(qi,pi)
        if numParticlesPerDim%2==0: #if even number then add one more particle at the center
            swarm.add_Particle()

        def compute_Stability(X):
            #X lattice arguments
            #for the given configuraiton X and particle(s) return the maximum number of revolutions, or True for stable
            #or False for unstable
            self.update_Ring_Lattice(X)
            revolutionsList=[]
            for particle in swarm:
                particle=self.particleTracerRing.trace(particle.copy(),h,T,fastMode=True)
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
        # plt.scatter(newCoords[:, 0], newCoords[:, 1])
        fitFunc=None
        def cost(X):
            return self.mode_Match(X,parallel=True)
        jitterAmplitude=[]
        for bound in bounds:
            jitterAmplitude.append(1e-3*(bound[1]-bound[0]))
        jitterAmplitude=np.asarray(jitterAmplitude)
        for i in range(maxIter):
            print('------',i)
            fitFunc=spi.RBFInterpolator(newCoords,newVals,smoothing=0.0)
            def minFunc(X):
                survival=fitFunc([X])[0]
                if survival<0:
                    return np.inf
                else:
                    return 1/survival
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
            newVal=cost(bestProbeCoords)
            newCoords=np.row_stack((newCoords,bestProbeCoords))
            plt.scatter(*bestProbeCoords,marker='x',c='r')
            newVals=np.append(newVals,newVal)
        print('initial value:',np.round(np.max(val0Arr),2),'new value: ',np.round(np.max(newVals),2))
        fitFunc=spi.RBFInterpolator(newCoords,newVals,smoothing=0.0)
        num=250
        xPlot=np.linspace(0,1,num=num)
        coords=np.asarray(np.meshgrid(xPlot,xPlot)).T.reshape(-1,2)
        image=fitFunc(coords).reshape(num,num)

        plt.imshow(np.flip(np.transpose(image),axis=0),extent=[0,1.0,0,1.0])
        plt.show()
        return newCoords[np.argmin(newVals)]

    def mode_Match(self,XRing,parallel=False):
        #project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        #configuration
        #todo: continue research in here

        self.update_Ring_Lattice(XRing)
        swarmRingTraced=self.revFunc(parallel=parallel)
        if swarmRingTraced is None: #unstable orbit
            return 0.0
        print('mode matching')
        modeMatchFunc=phaseSpaceInterpolater(swarmRingTraced)

        def cost(XInjector):
            print(XInjector)
            self.update_Injector_Lattice(XInjector)
            swarm=self.project_Injector_Swarm_To_Combiner_End()
            totalRevs=modeMatchFunc(swarm)
            meanRevs=totalRevs/self.swarmInjectorInitial.num_Particles()
            return 1/(meanRevs+1e-10)
        if parallel==True:
            num=30
            xArr=np.linspace(0.05,.5,num)
            coordsArr=np.asarray(np.meshgrid(xArr,xArr)).T.reshape(-1,2)
            results=np.asarray(self.helper.parallel_Problem(cost,coordsArr,onlyReturnResults=True))
            survival=1/results.min()
        else:
            import skopt

            t=time.time()
            sol=skopt.gp_minimize(cost,[(0.05,.5),(0.05,.5)],n_points=50,initial_point_generator='sobol')
            print(time.time()-t)
            print(sol.fun,sol.x)


            print('------------diff eveoltuion-----------------------')
            t=time.time()
            sol=spo.differential_evolution(cost,[(0.05,.5),(0.05,.5)],disp=False,polish=False)
            print(time.time()-t)
            print(sol.fun,sol.x)
            sys.exit()

            print(sol)
            survival=1/sol.fun
        print(XRing,survival)
        return survival

    def update_Injector_Lattice(self,X):
        #modify lengths of drift regions in injector
        LDrift1,LDrift2=X
        self.latticeInjector.elList[0].set_Length(LDrift1)
        self.latticeInjector.elList[2].set_Length(LDrift2)
        self.latticeInjector.set_Element_Coordinates()
        self.latticeInjector.make_Geometry()
    def project_Injector_Swarm_To_Combiner_End(self,h=1e-5):
        swarm=self.swarmTracerInjector.trace_Swarm_Through_Lattice(self.swarmInjectorInitial.quick_Copy(),h,1.0,parallel=False,
                                                                   fastMode=True,copySwarm=False)
        r0=self.latticeInjector.combiner.r2
        R=self.latticeInjector.combiner.RIn
        v0=self.latticeRing.v0Nominal
        apNextElement=self.latticeRing.elList[self.latticeRing.combinerIndex+1].ap
        swarmEnd=Swarm()
        for particle in swarm:
            q=particle.q-r0
            q[:2]=R@q[:2]
            if q[0]<h*v0: #if the particle is within a timestep of the end,
                # assume it's at the end
                p=particle.p.copy()
                p[:2]=R@p[:2]
                q=q+p*np.abs(q[0]/p[0])
                if np.sqrt(q[1] ** 2 + q[2] ** 2) < apNextElement:
                    swarmEnd.add_Particle(qi=q,pi=p)
        return swarmEnd
    def test_Stability(self,minRevs=5.0):
        swarmTest=self.swarmTracerRing.initialize_Stablity_Testing_Swarm(1e-3)
        swarmTest=self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmTest)
        swarmTest=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmTest,1e-5,
                                                                   1.5*minRevs*self.latticeRing.totalLength/200.0,
                                                                   parallel=False)
        stable=False
        for particle in swarmTest:
            if particle.revolutions>minRevs:
                stable=True
        return stable
    def revFunc(self,h=1e-5,T=1.0,numParticles=10000, parallel=False):
        stable=self.test_Stability()
        if stable == False:
            return None
        else:
            swarm0=self.swarmTracerRing.move_Swarm_To_Combiner_Output(self.swarmRingInitial.quick_Copy())
            swarm = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarm0, h, T, parallel=parallel, fastMode=True,copySwarm=False)
            # self.latticeRing.show_Lattice(swarm=swarm,trueAspectRatio=False,showTraceLines=True)
            return swarm
    def update_Ring_Lattice(self,X):
        for i in range(len(X)):
            self.latticeRing.elList[self.elementIndices[i]].fieldFact=X[i]
    def optimize_Magnetic_Field(self,elementIndices,bounds,num0,maxIter=10):
        # optimize magnetic field of the lattice by tuning element field strengths. This is done by first evaluating the
        #system over a grid, then using a non parametric model to find the optimum.
        #elementIndices: tuple of indices of elements to tune the field strength
        #bounds: list of tuples of (min,max) for tuning
        #maxIter: maximum number of optimization iterations with non parametric optimizer
        #num0: number of points in grid of magnetic fields
        assert np.unique(np.asarray(elementIndices)).shape[0]==len(elementIndices) #ensure no duplicates
        assert len(bounds)==len(elementIndices) #ensure bounds for each element being swept
        self.elementIndices=elementIndices
        BArrList=[]
        for bound in bounds:
            BArrList.append(np.linspace(bound[0], bound[1], num0))
        coordsArr = np.asarray(np.meshgrid(*BArrList)).T.reshape(-1, len(elementIndices))
        for coord in coordsArr:
            self.mode_Match(coord)
        gridResults = np.asarray(self.helper.parallel_Problem(self.mode_Match, coordsArr, onlyReturnResults=True,numWorkers=1))
        return self.optimize_Grid(coordsArr,gridResults,bounds,maxIter)

#
# test=LatticeOptimizer(None)
# test.optimize_Magnetic_Field((1,2,3),[(0,1),(0,1),(0,1)],30,10)