from skopt.plots import plot_objective
import warnings
from shapely.affinity import rotate,translate
import black_box as bb
import sys
import numpy.linalg as npl
from profilehooks import profile
import copy
import skopt
from ParticleTracerClass import ParticleTracer
import numpy as np
from ParticleClass import Swarm,Particle
import scipy.optimize as spo
import time
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from ParaWell import ParaWell
from SwarmTracerClass import SwarmTracer
from elementPT import HalbachLensSim, LensIdeal,Drift
import globalMethods as gm

class Solution:
    #class to hold onto results of each solution
    def __init__(self):
        self.xInjector_TunedParams=np.nan
        self.xRing_TunedParams1=np.nan #paramters tuned by the 'outer' gp minimize
        self.xRing_TunedParams2=np.nan #paramters tuned by the 'inner' gp minimize
        self.survival=np.nan
        self.description=None
        self.bumpParams=None # List of parameters used in the misalignment testing
    def __str__(self): #method that gets called when you do print(Solution())
        string='----------Solution-----------   \n'
        string+='injector element spacing optimum configuration: '+str(self.xInjector_TunedParams)+'\n '
        string+='storage ring tuned params 1 optimum configuration: '+str(self.xRing_TunedParams1)+'\n '
        string+='storage ring tuned params 2 optimum configuration: '+str(self.xRing_TunedParams2)+'\n '
        string+='bump params: '+str(self.bumpParams)+'\n'
        string+='optimum result: '+str(self.survival)+'\n'
        string+='----------------------------'
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
            assert 0.0<=particle.probability<=1.0 and particle.traced==True
            q=particle.q.copy()
            p=particle.p.copy()
            if useUpperSymmetry==True and q[zIndex]<0:
                p[zIndex]=-p[zIndex]
                q[zIndex]=-q[zIndex]
            X=np.append(q,p)
            weightedRevolutions=self.interpolater(X)[0]*particle.probability
            totalRevs+=weightedRevolutions
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
        self.h=5e-6 #timestep size
        self.T=1.0
        self.swarmTracerRing=SwarmTracer(self.latticeRing)
        self.phaseSpaceFunc=None #function that returns the number of revolutions of a particle at a given
        #point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.solutionList=[] #list to hold solution objects the track coordsinates and function values for injector
        #and ring paramters
        self.tunedElementList=None
        self.tuningChoice=None #what type of tuning will occur
        self.useLatticeUpperSymmetry=True #exploit the fact that the lattice has symmetry in the z axis to use half
        #the number of particles. Symmetry is broken if including gravity
        self.sameSeedForSwarm=True #generate the same swarms every time by seeding the random generator during swarm
        #generation with the same number, 42
        self.sameSeedForSearch=True #wether to use the same seed, 42, for the search process
        self.numParticlesInjector=500
        self.numParticlesRing=50000

        self.spotCaptureDiam=5e-3
        self.collectorAngleMax=.06
        self.temperature=3e-3
        fractionalMarginOfError=1.25
        self.minElementLength=fractionalMarginOfError*self.particleTracerRing.minTimeStepsPerElement*\
                              self.latticeRing.v0Nominal*self.h
        self.tunableTotalLengthList=[] #list to hold initial tunable lengths for when optimizing by tuning element
        # length. this to prevent any numerical round issues causing the tunable length to change from initial value
        # if I do many iterations
    def fill_Swarms_And_Test_For_Feasible_Injector(self,parallel):
        firstApertureRing=self.latticeRing.elList[self.latticeRing.combinerIndex+1].ap

        self.swarmInjectorInitial=self.swarmTracerInjector.initialize_Observed_Collector_Swarm_Probability_Weighted(
            self.spotCaptureDiam, self.collectorAngleMax,self.numParticlesInjector,temperature=self.temperature,
                                            sameSeed=self.sameSeedForSwarm,upperSymmetry=self.useLatticeUpperSymmetry)
        injectorBounds=self.find_Injector_Mode_Match_Bounds(parallel)
        if injectorBounds is None:
            print('not a feasible injector configuration')
            return False
        pxMaxRing=1.1*max(np.abs(injectorBounds[2][1]),np.abs(injectorBounds[2][0]))
        pTransMaxRing=1.1*max(np.abs(injectorBounds[3][1]),np.abs(injectorBounds[3][0]))
        qMax=1.1*firstApertureRing
        self.swarmRingInitialAtCombinerOutput=self.swarmTracerRing.initalize_PseudoRandom_Swarm_At_Combiner_Output(qMax,pTransMaxRing,
                                            pxMaxRing,self.numParticlesRing,
                                            sameSeed=self.sameSeedForSwarm,upperSymmetry=self.useLatticeUpperSymmetry)

        return True
    def get_Injector_Swarm_Bounds(self):
        #finding injector bounds is expensive. This function attempts to reuse previously computed bounds.
        #for now, it simply uses bounds saved below, unless something has changed in which case it throws an
        #exception so I don't accidently change something and not update them
        injectorLensLength=.2
        injectorLenBoreRadius=.025
        numberInjectorElements=4
        injectorBounds=[(-0.015, 0.018), (-0.00625, 0.006), (-4.5, 4), (-12, 12),
                        (-8, 8)]#10 m/s
        if self.latticeInjector.elList[1].L!= injectorLensLength: raise Exception('lens has changed')
        elif self.latticeInjector.elList[1].rp!=injectorLenBoreRadius:  raise Exception('lens has changed')
        elif len(self.latticeInjector.elList)!=numberInjectorElements:  raise Exception('number of element have changed')
        elif self.temperature!=3e-3 or self.collectorAngleMax!=.06 or self.spotCaptureDiam!=5e-3:
            raise Exception('Swarm paramters have changed')
        else: return injectorBounds
    def find_Injector_Mode_Match_Bounds(self,parallel):
        injectorParamsBounds=(.05,.5)
        numGridPointsPerDim = 30
        xArr = np.linspace(injectorParamsBounds[0], injectorParamsBounds[1], numGridPointsPerDim)
        coords = np.asarray(np.meshgrid(xArr, xArr)).T.reshape(-1, 2)
        fracCutOff=.95 # for any given extrema, chose the value that bounds this fraction of particles to avoid wasting
        #resources on outliers
        def wrapper(X): # need a wrapper to update lattice before tracing
            self.update_Injector_Lattice(X)
            swarmInjectorTraced=self.swarmTracerInjector.trace_Swarm_Through_Lattice(
                self.swarmInjectorInitial.quick_Copy(),self.h,1.0,
                parallel=False,fastMode=True,copySwarm=False,accelerated=True)
            swarmEnd=self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced)
            return swarmEnd
        if parallel==True:
            projectedSwarmsList=self.helper.parallel_Problem(wrapper,coords,onlyReturnResults=True)
        else:
            projectedSwarmsList=[wrapper(coord) for coord in coords]
        phaseSpaceExtremaList=[]
        for swarm in projectedSwarmsList:
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
        phaseSpaceExtremaArr=np.asarray(phaseSpaceExtremaList)
        percentStableSolutions=100*phaseSpaceExtremaArr.shape[0]/self.swarmInjectorInitial.num_Particles()
        if percentStableSolutions<1: #practically no viable solution
            return None
        boundsList=[] #list to hold bounds of swarm to trace in storage ring
        for i in range(phaseSpaceExtremaArr.shape[1]//2): #loop over each coordinate, ie y,z,px etc
            boundsList.append((np.round(phaseSpaceExtremaArr[:,2*i].min(),6),np.round(phaseSpaceExtremaArr[:,2*i+1].max(),6))) #need to double
            #i because there are 2 columns per variable, a min and a max
        return boundsList
    def get_Injector_Shapely_Objects_In_Lab_Frame(self):
        newShapelyObjectList=[]
        rotationAngle=self.latticeInjector.combiner.ang+-self.latticeRing.combiner.ang
        r2Injector=self.latticeInjector.combiner.r2
        r2Ring=self.latticeRing.combiner.r2
        for el in self.latticeInjector.elList:
            SO=copy.copy(el.SO_Outer)
            SO=translate(SO,xoff=-r2Injector[0],yoff=-r2Injector[1])
            SO=rotate(SO,rotationAngle,use_radians=True,origin=(0,0))
            SO=translate(SO,xoff=r2Ring[0],yoff=r2Ring[1])
            newShapelyObjectList.append(SO)
        return newShapelyObjectList
    def generate_Shapely_Floor_Plan(self):
        shapelyObjectList=[]
        shapelyObjectList.extend([el.SO_Outer for el in self.latticeRing.elList])
        shapelyObjectList.extend(self.get_Injector_Shapely_Objects_In_Lab_Frame())
        return shapelyObjectList
    def is_Floor_Plan_Valid(self):
        injectorShapelyObjects=self.get_Injector_Shapely_Objects_In_Lab_Frame()
        ringShapelyObjects=[el.SO_Outer for el in self.latticeRing.elList]
        injectorLens=injectorShapelyObjects[1]
        for element in ringShapelyObjects:
            if element.intersection(injectorLens).area !=0.0:
                return False
        return True
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

        solution=Solution()
        solution.xRing_TunedParams2=XRing
        self.update_Ring_Lattice(XRing)
        swarmRingTraced=self.revFunc(parallel=parallel)
        if swarmRingTraced is None: #unstable orbit
            solution.survival=0.0
            return solution
        modeMatchFunc=phaseSpaceInterpolater(swarmRingTraced)
        def cost_To_Minimize(XInjector):
            self.update_Injector_Lattice(XInjector)
            if self.is_Floor_Plan_Valid()==False:
                return 1.0
            else:
                swarm=self.trace_And_Project_Injector_Swarm_To_Combiner_End()
                survival=self.compute_Swarm_Survival(swarm,modeMatchFunc)
                cost=self.cost_Function(survival)
                return cost
        bounds=[(0.1,.5),(0.05,.5)]
        gridEdgeNum=5
        coordsForGrid=[]
        for bound in bounds:
            dx=(bound[1]-bound[0])/gridEdgeNum
            xArr=np.linspace(bound[0]+dx,bound[1]-dx,gridEdgeNum)
            coordsForGrid.append(xArr)
        coordsArr=np.asarray(np.meshgrid(*coordsForGrid)).T.reshape(-1,len(bounds))
        coordsArr=np.row_stack((coordsArr,np.asarray([.15,.15])))
        if parallel==True:
            costArr=np.asarray(self.helper.parallel_Problem(cost_To_Minimize,coordsArr,onlyReturnResults=True))
            solverCoordInitial=coordsArr[np.argmin(costArr)]
        else:
            costArr=np.asarray([cost_To_Minimize(coord) for coord in coordsArr])
            solverCoordInitial=coordsArr[np.argmin(costArr)] #start solver at minimum value

        useScipy=False
        if useScipy==True:
            minimizerSol=spo.minimize(cost_To_Minimize,solverCoordInitial,bounds=bounds,method='Nelder-Mead',options={'xatol':.0001})
        else:
            coordsListForSkopt=coordsArr.tolist()
            costListForSkopt=costArr.tolist()
            minimizerSol=skopt.gp_minimize(cost_To_Minimize,bounds,n_calls=20,n_initial_points=0,x0=coordsListForSkopt,
                                       y0=costListForSkopt,acq_optimizer='sampling',
                                       n_points=3000,noise=.05)
        solution.xInjector_TunedParams=minimizerSol.x
        solution.survival=self.inverse_Cost_Function(minimizerSol.fun)
        return solution

    def move_Survived_Particles_In_Injector_Swarm_To_Origin(self,swarmInjectorTraced,copyParticles=False):
        apNextElement=self.latticeRing.elList[self.latticeRing.combinerIndex+1].ap
        swarmEnd=Swarm()
        for particle in swarmInjectorTraced:
            q=particle.q.copy()-self.latticeInjector.combiner.r2
            q[:2]=self.latticeInjector.combiner.RIn@q[:2]
            if q[0]<=self.h*self.latticeRing.v0Nominal:  #if the particle is within a timestep of the end,
                # assume it's at the end
                p=particle.p.copy()
                p[:2]=self.latticeInjector.combiner.RIn@p[:2]
                q=q+p*np.abs(q[0]/p[0])
                if np.sqrt(q[1]**2+q[2]**2)<apNextElement:
                    if copyParticles==False: particleEnd=particle
                    else: particleEnd=particle.copy()
                    particleEnd.q=q
                    particleEnd.p=p
                    swarmEnd.particles.append(particleEnd)
        return swarmEnd
    def trace_And_Project_Injector_Swarm_To_Combiner_End(self):
        swarmInjectorTraced=self.swarmTracerInjector.trace_Swarm_Through_Lattice(self.swarmInjectorInitial.quick_Copy(),self.h,np.inf,
                                                    parallel=False,fastMode=True,copySwarm=False,accelerated=True)
        swarmEnd=self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced,copyParticles=False)
        swarmEnd=self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd,copySwarm=False)
        return swarmEnd
    def test_Stability(self,minRevs=5.0):
        swarmTest=self.swarmTracerRing.initialize_Stablity_Testing_Swarm(1e-3)
        swarmTest=self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmTest)
        swarmTest=self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmTest,self.h,
                                                                   1.5*minRevs*self.latticeRing.totalLength/200.0,
                                                                   parallel=False,accelerated=True,fastMode=True)
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
            return swarm

    def update_Injector_Lattice(self,X):
        #modify lengths of drift regions in injector
        self.latticeInjector.elList[0].set_Length(X[0])
        self.latticeInjector.elList[2].set_Length(X[1])
        self.latticeInjector.build_Lattice()
    def update_Ring_Lattice(self,X):
        if self.tuningChoice=='field':
            self.update_Ring_Field_Values(X)
        elif self.tuningChoice=='spacing':
            self.update_Ring_Spacing(X)
        else: raise Exception('wrong tuning choice')
    def update_Ring_Field_Values(self,X):
        for el,arg in zip(self.tunedElementList,X):
            el.fieldFact=arg
    def update_Ring_Spacing(self,X):
        for elCenter,spaceFracElBefore,totalLength in zip(self.tunedElementList,X,self.tunableTotalLengthList):
            elBefore,elAfter=self.latticeRing.get_Element_Before_And_After(elCenter)
            tunableLength=(elBefore.L+elAfter.L)-2*self.minElementLength
            LBefore=spaceFracElBefore*tunableLength+self.minElementLength
            LAfter=totalLength-LBefore
            elBefore.set_Length(LBefore)
            elAfter.set_Length(LAfter)
        self.latticeRing.build_Lattice()
    def compute_Swarm_Survival(self,swarmTraced,modeMatchFunction):
        #survival as percent of particle surviving till the maximum time
        totalWeightedRevolutions=modeMatchFunction(swarmTraced,self.useLatticeUpperSymmetry)
        numWeightedInitialParticles=sum([particle.probability for particle in self.swarmInjectorInitial])
        maximumRevs=numWeightedInitialParticles*self.T*self.latticeRing.v0Nominal/self.latticeRing.totalLength
        survival=1e2*totalWeightedRevolutions/maximumRevs
        assert 0.0<=survival<=100.0
        return survival
    @staticmethod
    def cost_Function(survival):
        assert 0.0<=survival<=100.0
        return np.exp(-survival/10)
    @staticmethod
    def inverse_Cost_Function(cost):
        #returns survival
        assert 0.0<=cost<=1.0
        return -10*np.log(cost)
    def best_Solution(self):
        return self.solutionList[np.nanargmax(np.asarray([sol.survival for sol in self.solutionList]))]
    def catch_Optimizer_Errors(self,tuningBounds,elementIndices,tuningChoice):
        if len(self.solutionList)!=0: raise Exception("You cannot call optimize twice")
        if max(elementIndices)>=len(self.latticeRing.elList)-1: raise Exception("element indices out of bounds")
        if len(tuningBounds)!=len(elementIndices): raise Exception("Bounds do not match number of tuned elements")
        if tuningChoice=='field':
            for el in self.tunedElementList:
                if (isinstance(el,LensIdeal) and isinstance(el,HalbachLensSim)) != True:
                    raise Exception("For field tuning elements must be LensIdeal or HalbachLensSim")
        elif tuningChoice=='spacing':
            for elIndex in elementIndices:
                elBefore,elAfter=self.latticeRing.get_Element_Before_And_After(self.latticeRing.elList[elIndex])
                tunableLength=(elBefore.L+elAfter.L)-2*self.minElementLength
                if (isinstance(elBefore,Drift) and isinstance(elAfter,Drift))!=True:
                    raise Exception("For spacing tuning neighboring elements must be Drift elements")
                if tunableLength<0.0:
                    raise Exception("Tunable elements are too short for length tuning. Min total length is "
                                    +str(2*self.minElementLength))
        else: raise  Exception('No proper tuning choice provided')
    def make_Tuning_Coords_List(self,tuningBounds,numGridEdge):
        #gp_minimize requires a list of lists, I make that here
        meshgridArraysList=[np.linspace(bound[0],bound[1],numGridEdge) for bound in tuningBounds]
        tuningCoordsArr=np.asarray(np.meshgrid(*meshgridArraysList)).T.reshape(-1,len(self.tunedElementList))
        tuningCoordsList=tuningCoordsArr.tolist()  #must in list format
        return tuningCoordsList
    def fill_Initial_Total_Tuning_Elements_Length_List(self):
        for elCenter in self.tunedElementList:
            elBefore,elAfter=self.latticeRing.get_Element_Before_And_After(elCenter)
            self.tunableTotalLengthList.append(elBefore.L+elAfter.L)
    def initialize_Optimizer(self,elementIndices,tuningChoice):
        self.tunedElementList=[self.latticeRing.elList[index] for index in elementIndices]
        self.tuningChoice=tuningChoice
        if tuningChoice=='spacing':
            self.fill_Initial_Total_Tuning_Elements_Length_List()
        if self.sameSeedForSearch==True:
            np.random.seed(42)

    def optimize_Magnetic_Field(self,elementIndices,tuningBounds,numGridEdge,tuningChoice,maxIter=10,parallel=True):
        # optimize magnetic field of the lattice by tuning element field strengths. This is done by first evaluating the
        #system over a grid, then using a non parametric model to find the optimum.
        #elementIndices: tuple of indices of elements to tune the field strength
        #bounds: list of tuples of (min,max) for tuning
        #maxIter: maximum number of optimization iterations with non parametric optimizer
        #num0: number of points in grid of magnetic fields
        feasibleInjector=self.fill_Swarms_And_Test_For_Feasible_Injector(parallel)
        if feasibleInjector==False:
            solution=Solution()
            solution.survival=0.0
            return solution
        self.catch_Optimizer_Errors(tuningBounds,elementIndices,tuningChoice)
        self.initialize_Optimizer(elementIndices,tuningChoice)
        tuningCoordsList=self.make_Tuning_Coords_List(tuningBounds,numGridEdge)
        if parallel==True:
            self.solutionList=self.helper.parallel_Problem(self.mode_Match, tuningCoordsList, onlyReturnResults=True)
        else:
            self.solutionList=[self.mode_Match(tuneCoord) for tuneCoord in tuningCoordsList]
        gridSearchSurvivalResults=[solution.survival for solution in self.solutionList]


        survivalForRelevance=1.0 #1% survival for relevance
        if max(gridSearchSurvivalResults)<survivalForRelevance:
            print('no viable solution')
            solution=Solution()
            solution.survival=0.0
            return solution
        print('yes viable solution')


        def skopt_Cost(XRing):
            solution=self.mode_Match(XRing,parallel=parallel)
            self.solutionList.append(solution)
            cost=self.cost_Function(solution.survival)
            return cost
        gridSearchCostResults=[self.cost_Function(survival) for survival in gridSearchSurvivalResults]
        skoptMimizerJobs=-1 if parallel==True else 1
        skoptSol=skopt.gp_minimize(skopt_Cost,tuningBounds,n_initial_points=0,x0=tuningCoordsList,y0=gridSearchCostResults
                                   ,n_calls=maxIter,model_queue_size=1,n_jobs=skoptMimizerJobs,n_restarts_optimizer=32,
                                   n_points=10000,noise=1e-6,acq_optimizer='lbfgs')
        return self.best_Solution()