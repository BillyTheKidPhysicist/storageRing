from shapely.affinity import rotate,translate
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

class Solution:
    #class to hold onto results of each solution
    def __init__(self):
        self.xInjector=np.nan
        self.xRing=np.nan
        self.func=np.nan
    def __str__(self): #method that gets called when you do print(Solution())
        string='----------Solution-----------   \n'
        string+='injector element spacing optimum configuration: '+str(self.xInjector)+'\n '
        string+='storage ring magnetic field optimum configuration: '+str(self.xRing)+'\n '
        string+='optmum result: '+str(self.func)
        return string


class phaseSpaceInterpolater:
    #simple class to help with using LinearNDInterpolater or NearestNDINterpolator. Keeps track of bounds violation
    #and apeture violations. The assumption is that the first two coordinates correspond to positions (y,z) where (y,z)
    #are positions relative to an apture centered at y=0,z=0.
    def __init__(self,swarmRingTraced):
        XiList=[] #list initial coordinates in phase space at the combiner output
        revolutionList=[] #list of total revolutions before striking aperture
        for particle in swarmRingTraced:
            qi=particle.qi
            pi=particle.pi
            Xi=np.append(qi,pi)
            XiList.append(Xi)
            revolutionList.append(particle.revolutions)
        self.interpolater=spi.NearestNDInterpolator(XiList, revolutionList,rescale=True,tree_options={'copy_data':True})
    def __call__(self,swarmInjector,useUpperSymmetry):
        totalRevs=0
        zIndex=2
        for particle in swarmInjector:
            q=particle.q.copy()
            p=particle.p.copy()
            if useUpperSymmetry==True and q[zIndex]<0:
                p[zIndex]=-p[zIndex] 
                q[zIndex]=-q[zIndex]
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
        self.swarmRingInitialAtCombinerOutput=None #object to hold the ring swarm object that will generate the mode match function
        self.h=1e-5 #timestep size
        self.T=10.0
        self.swarmTracerRing=SwarmTracer(self.latticeRing)
        self.phaseSpaceFunc=None #function that returns the number of revolutions of a particle at a given
        #point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.solutionList=[] #list to hold solution objects the track coordsinates and function values for injector
        #and ring paramters
        self.elementIndices=None
        self.useLatticeUpperSymmetry=True #exploit the fact that the lattice has symmetry in the z axis to use half
        #the number of particles. Symmetry is broken if including gravity
        self.sameSeedForSwarm=True #generate the same swarms every time by seeding the random generator during swarm
        #generation with the same number, 42
        self.sameSeedForSearch=True #wether to use the same seed, 42, for the search process
        self.numParticlesInjector=500
        self.numParticlesRing=30000

        self.qMaxInjector=2.5e-3
        self.pTransMaxInjector=10.0
        self.pxMaxInjector=1.0

    def generate_Swarms(self):
        firstApertureRing=self.latticeRing.elList[self.latticeRing.combinerIndex+1].ap

        self.swarmInjectorInitial=self.swarmTracerInjector.initalize_PseudoRandom_Swarm_In_Phase_Space(self.qMaxInjector,
                                                self.pTransMaxInjector,self.pxMaxInjector,self.numParticlesInjector,
                                                sameSeed=self.sameSeedForSwarm,upperSymmetry=self.useLatticeUpperSymmetry)

        injectorBounds=self.get_Injector_Swarm_Bounds()
        pxMaxRing=1.1*max(np.abs(injectorBounds[2][1]),np.abs(injectorBounds[2][0]))
        pTransMaxRing=1.1*max(np.abs(injectorBounds[3][1]),np.abs(injectorBounds[3][0]))
        qMax=1.1*firstApertureRing

        self.swarmRingInitialAtCombinerOutput=self.swarmTracerRing.initalize_PseudoRandom_Swarm_At_Combiner_Output(qMax,pTransMaxRing,
                                            pxMaxRing,self.numParticlesRing,
                                            sameSeed=self.sameSeedForSwarm,upperSymmetry=self.useLatticeUpperSymmetry)


    def get_Injector_Swarm_Bounds(self):
        #finding injector bounds is expensive. This function attempts to reuse previously computed bounds.
        #for now, it simply uses bounds saved below, unless something has changed in which case it throws an
        #exception so I don't accidently change something and not update them
        injectorLensLength=.2
        injectorLenBoreRadius=.025
        numberInjectorElements=4
        injectorBounds=[(-0.007727, 0.007647), (-0.006008, 0.005877), (-1.146266, 1.023455), (-9.265196, 8.804026),
                        (-8.080922, 7.712291)]

        if self.latticeInjector.elList[1].L!= injectorLensLength: raise Exception('lens has changed')
        elif self.latticeInjector.elList[1].rp!=injectorLenBoreRadius:  raise Exception('lens has changed')
        elif len(self.latticeInjector.elList)!=numberInjectorElements:  raise Exception('number of element have changed')
        elif self.pxMaxInjector!=1.0 or self.pTransMaxInjector!=10.0 or self.qMaxInjector!=2.5e-3:
            raise Exception('Swarm paramters have changed')
        else: return injectorBounds
    def find_Injector_Mode_Match_Bounds(self,parallel):
        #todo: use this to set hard limits on output of combiner?
        injectorParamsBounds=(.05,1.0)
        numGridPointsPerDim = 50
        xArr = np.linspace(injectorParamsBounds[0], injectorParamsBounds[1], numGridPointsPerDim)
        coords = np.asarray(np.meshgrid(xArr, xArr)).T.reshape(-1, 2)
        fracCutOff=.98 # for any given extrema, chose the value that bounds this fraction of particles to avoid wasting
        #resources on outliers
        def wrapper(X): # need a wrapper to update lattice before tracing
            self.update_Injector_Lattice(X)
            swarmInjectorTraced=self.swarmTracerInjector.trace_Swarm_Through_Lattice(
                self.swarmInjectorInitial.quick_Copy(),self.h,1.0,
                parallel=False,fastMode=True,copySwarm=False,accelerated=False)
            swarmEnd=self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced)
            return swarmEnd
        if parallel==True:
            projectedSwarmsList=self.helper.parallel_Problem(wrapper,coords,onlyReturnResults=True)
        else:
            projectedSwarmsList=[wrapper(coord) for coord in coords]
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
        return boundsList
    def move_Injector_Shapely_Objects_To_Lab_Frame(self):
        newShapelyObjectList=[]
        rotationAngle=self.latticeInjector.combiner.ang+-self.latticeRing.combiner.ang
        r2Injector=self.latticeInjector.combiner.r2
        r2Ring=self.latticeRing.combiner.r2
        for el in self.latticeInjector.elList:
            SO=el.SO_Outer
            SO=translate(SO,xoff=-r2Injector[0],yoff=-r2Injector[1])
            SO=rotate(SO,rotationAngle,use_radians=True,origin=(0,0))
            SO=translate(SO,xoff=r2Ring[0],yoff=r2Ring[1])
            newShapelyObjectList.append(SO)
        return newShapelyObjectList
    def generate_Shapely_Floor_Plan(self):
        shapelyObjectList=[]
        shapelyObjectList.extend([el.SO_Outer for el in self.latticeRing.elList])
        shapelyObjectList.extend(self.move_Injector_Shapely_Objects_To_Lab_Frame())
        return shapelyObjectList
    def show_Floor_Plan(self):
        shapelyObjectList=self.generate_Shapely_Floor_Plan()
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.show()

    def get_Stability_Function(self,qMax=.5e-3,numParticlesPerDim=2,cutoff=8.0,funcType='bool'):
        #return a function that can evaluated at the lattice parameters, X, and return wether True for stable
        #and False for unstable
        #qMax: #if using multiple particles this defineds the bounds of the initial positions square,meters
        #numParticlesPerDim: Number of particles on each grid edge. Total number is this squared.
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
                particle=self.particleTracerRing.trace(particle.copy(),self.h,T,fastMode=True)
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
    def compute_Revolution_Func_Over_Grid(self,bounds=None,qMax=1e-4,numParticlesPerDim=2,gridPoints=40,cutoff=8.0):
        #this method loops over a grid and logs the numbre of revolutions up to cutoff for the particle with the
        #maximum number of revolutions
        #bounds: region to search over for instability
        #qMax: maximum dimension in transverse directions for initialized particles
        #numParticlesPerDim: Number of particles along y and z axis so total is numParticlesPerDim**2. when 1 a single
        #particle is initialized at [1e-10,0,0]
        #gridPoints: number of points per axis to test stability. Total is gridPoints**2
        #cutoff: Maximum revolution number
        #returns: a function that evaluates to the maximum number of revolutions of the particles

        revFunc=self.get_Stability_Function(qMax=qMax,numParticlesPerDim=numParticlesPerDim,cutoff=cutoff,funcType='rev')
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
                       plotName='stabilityPlot',cutoff=8.0,showPlot=True,modulation='01'):
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
                                                             numParticlesPerDim=numParticlesPerDim,
                                                              gridPoints=gridPoints,cutoff=cutoff)
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

    def mode_Match(self,XRing,parallel=False):
        #project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        #configuration

        #todo: add list that follows along and records results
        solution=Solution()
        solution.xRing=XRing
        self.update_Ring_Lattice(XRing)
        swarmRingTraced=self.revFunc(parallel=parallel)
        if swarmRingTraced is None: #unstable orbit
            solution.func=0.0
            return solution
        modeMatchFunc=phaseSpaceInterpolater(swarmRingTraced)
        def cost_To_Minimize(XInjector):
            self.update_Injector_Lattice(XInjector)
            swarm=self.trace_And_Project_Injector_Swarm_To_Combiner_End()
            survival=self.compute_Swarm_Survival(swarm,modeMatchFunc)
            return 1/(survival+1e-10)

        bounds=[(0.1,.5),(0.05,.5)]
        num=10
        coordsForGrid=[]
        for bound in bounds:
            dx=(bound[1]-bound[0])/num
            xArr=np.linspace(bound[0]+dx,bound[1]-dx,num)
            coordsForGrid.append(xArr)
        coordsArr=np.asarray(np.meshgrid(*coordsForGrid)).T.reshape(-1,len(bounds))
        if parallel==True:
            costArr=np.asarray(self.helper.parallel_Problem(cost_To_Minimize,coordsArr,onlyReturnResults=True))
            solverCoordInitial=coordsArr[np.argmin(costArr)]
        else:
            costList=[]
            for coord in coordsArr:
                costList.append(cost_To_Minimize(coord))
            costArr=np.asarray(costList)
            solverCoordInitial=coordsArr[np.argmin(costArr)] #start solver at minimum value
        sol=spo.minimize(cost_To_Minimize,solverCoordInitial,bounds=bounds,method='Nelder-Mead',options={'xatol':.0001})
        survival=1/sol.fun
        solution.xInjector=sol.x
        solution.func=survival
        return solution

    def update_Injector_Lattice(self,X):
        #modify lengths of drift regions in injector
        LDrift1,LDrift2=X
        self.latticeInjector.elList[0].set_Length(LDrift1)
        self.latticeInjector.elList[2].set_Length(LDrift2)
        self.latticeInjector.set_Element_Coordinates()
        self.latticeInjector.make_Geometry()
    def move_Survived_Particles_In_Injector_Swarm_To_Origin(self,swarmInjectorTraced):
        apNextElement=self.latticeRing.elList[self.latticeRing.combinerIndex+1].ap
        swarmEnd=Swarm()
        for particle in swarmInjectorTraced:
            q=particle.q-self.latticeInjector.combiner.r2
            q[:2]=self.latticeInjector.combiner.RIn@q[:2]
            if q[0]<self.h*self.latticeRing.v0Nominal:  #if the particle is within a timestep of the end,
                # assume it's at the end
                p=particle.p.copy()
                p[:2]=self.latticeInjector.combiner.RIn@p[:2]
                q=q+p*np.abs(q[0]/p[0])
                if np.sqrt(q[1]**2+q[2]**2)<apNextElement:
                    swarmEnd.add_Particle(qi=q,pi=p)
        return swarmEnd
    def trace_And_Project_Injector_Swarm_To_Combiner_End(self):
        swarmInjectorTraced=self.swarmTracerInjector.trace_Swarm_Through_Lattice(self.swarmInjectorInitial.quick_Copy(),self.h,np.inf,
                                                    parallel=False,fastMode=True,copySwarm=False,accelerated=True)
        swarmEnd=self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced)
        swarmEnd=self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd,copySwarm=False)
        return swarmEnd
    def test_Stability(self,minRevs=5.0):
        swarmTest=self.swarmTracerRing.initialize_Stablity_Testing_Swarm(1e-3)
        swarmTest=self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmTest)
        swarmTest=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmTest,1e-5,
                                                                   1.5*minRevs*self.latticeRing.totalLength/200.0,
                                                                   parallel=False,accelerated=True)
        stable=False
        for particle in swarmTest:
            if particle.revolutions>minRevs:
                stable=True
        return stable
    def revFunc(self, parallel=False):
        stable=self.test_Stability()
        if stable == False:
            return None
        else:
            swarm = self.swarmTracerRing.trace_Swarm_Through_Lattice(self.swarmRingInitialAtCombinerOutput.quick_Copy(), self.h, self.T,
                                                    parallel=parallel, fastMode=True,copySwarm=False,accelerated=True)
            # self.latticeRing.show_Lattice(swarm=swarm,trueAspectRatio=False,showTraceLines=True)
            return swarm
    def update_Ring_Lattice(self,X):
        for i in range(len(X)):
            self.latticeRing.elList[self.elementIndices[i]].fieldFact=X[i]
    def compute_Swarm_Survival(self,swarmTraced,modeMatchFunction):
        totalRevolutions=modeMatchFunction(swarmTraced,self.useLatticeUpperSymmetry)
        maximumRevs=self.swarmInjectorInitial.num_Particles()*self.T*self.latticeRing.v0Nominal/self.latticeRing.totalLength
        # fluxMultiplication=totalRevolutions/self.swarmInjectorInitial.num_Particles()
        survival=1e2*totalRevolutions/maximumRevs
        return survival
    def best_Solution(self):
        return self.solutionList[np.nanargmax(np.asarray([sol.func for sol in self.solutionList]))]
    def optimize_Magnetic_Field(self,elementIndices,bounds,num0,maxIter=10,parallel=True):
        # optimize magnetic field of the lattice by tuning element field strengths. This is done by first evaluating the
        #system over a grid, then using a non parametric model to find the optimum.
        #elementIndices: tuple of indices of elements to tune the field strength
        #bounds: list of tuples of (min,max) for tuning
        #maxIter: maximum number of optimization iterations with non parametric optimizer
        #num0: number of points in grid of magnetic fields
        assert np.unique(np.asarray(elementIndices)).shape[0]==len(elementIndices) #ensure no duplicates
        assert len(bounds)==len(elementIndices) #ensure bounds for each element being swept
        if self.sameSeedForSearch==True:
            np.random.seed(42)
        self.generate_Swarms()
        self.elementIndices=elementIndices
        BArrList=[]
        for bound in bounds:
            BArrList.append(np.linspace(bound[0], bound[1], num0))
        coordsArr = np.asarray(np.meshgrid(*BArrList)).T.reshape(-1, len(elementIndices))
        print('beginning grid search')
        if parallel==True:
            self.solutionList=self.helper.parallel_Problem(self.mode_Match, coordsArr, onlyReturnResults=True)
        else:
            self.solutionList=[self.mode_Match(coord) for coord in coordsArr]
        gridResults=np.asarray([solution.func for solution in self.solutionList])

        print('grid optimum over: ', np.max(gridResults),'Number valid',np.sum(gridResults!=0))
        def skopt_Cost(XRing):
            solution=self.mode_Match(XRing,parallel=parallel)
            self.solutionList.append(solution)
            survival=solution.func
            return 1/(survival+1e-10)

        minimizedGridResults=list(1/(gridResults+1))
        averageRevolutionsForRelevance=1e-3
        if np.max(gridResults)-np.min(gridResults)<averageRevolutionsForRelevance:
            print('no viable solution')
            return Solution()
        coordsList=[]
        for coord in coordsArr:
            coordsList.append(list(coord))
        # import skopt
        print('beginning gaussian process')
        skopt.gp_minimize(skopt_Cost,bounds,n_initial_points=0,x0=coordsList,y0=minimizedGridResults
                          ,n_calls=maxIter)
        return self.best_Solution()