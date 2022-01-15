import time
import skopt
from shapely.affinity import rotate, translate
import copy
from ParticleTracerClass import ParticleTracer
import numpy as np
from ParticleClass import Swarm, Particle
import scipy.optimize as spo
import matplotlib.pyplot as plt
from SwarmTracerClass import SwarmTracer
from elementPT import HalbachLensSim, LensIdeal, Drift
import multiprocess as mp
from profilehooks import profile
'''
This file contains a class used to optimize the tunable knobs we can expect to have. These are either field values
or element spacing

'''


class Solution:
    # class to hold onto results of each solution
    def __init__(self):
        self.xInjector_TunedParams = np.nan
        self.xRing_TunedParams1 = np.nan  # paramters tuned by the 'outer' gp minimize
        self.xRing_TunedParams2 = np.nan  # paramters tuned by the 'inner' gp minimize
        self.survival = np.nan
        self.cost=np.nan
        self.stable=None
        self.invalidInjector=None
        self.invalidRing=None
        self.description = None
        self.bumpParams = None  # List of parameters used in the misalignment testing

    def __str__(self):  # method that gets called when you do print(Solution())
        string = '----------Solution-----------   \n'
        string += 'injector element spacing optimum configuration: ' + str(self.xInjector_TunedParams) + '\n '
        string += 'storage ring tuned params 1 optimum configuration: ' + str(self.xRing_TunedParams1) + '\n '
        string += 'storage ring tuned params 2 optimum configuration: ' + str(self.xRing_TunedParams2) + '\n '
        # string+='stable configuration:'+str(self.stable)+'\n'
        # string += 'bump params: ' + str(self.bumpParams) + '\n'
        string+='cost: '+str(self.cost)+'\n'
        string += 'survival: ' + str(self.survival) + '\n'
        # string += '----------------------------'
        return string


class LatticeOptimizer:
    def __init__(self, latticeRing, latticeInjector):

        self.latticeRing = latticeRing
        self.latticeInjector = latticeInjector
        self.i = 0  # simple variable to track solution status
        self.particleTracerRing = ParticleTracer(latticeRing)
        self.particleTracerInjector = ParticleTracer(latticeInjector)
        self.swarmTracerInjector = SwarmTracer(self.latticeInjector)
        self.h = 1e-5  # timestep size
        self.T = 10.0
        self.swarmTracerRing = SwarmTracer(self.latticeRing)
        self.phaseSpaceFunc = None  # function that returns the number of revolutions of a particle at a given
        # point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.tunedElementList = None
        self.tuningChoice = None  # what type of tuning will occur
        self.useLatticeUpperSymmetry = True  # exploit the fact that the lattice has symmetry in the z axis to use half
        # the number of particles. Symmetry is broken if including gravity
        self.sameSeedForSwarm = True  # generate the same swarms every time by seeding the random generator during swarm
        # generation with the same number, 42
        self.sameSeedForSearch = True  # wether to use the same seed, 42, for the search process
        self.fastSolver=None #wether to use the faster, but less accurate solver. See oneNote
        self.optimalPopSize=5 #for scipy differential solver. This was carefully chosen
        self.tolerance=.01 #for scipy differential evolution. This is the maximum accuravy roughly speaking
        self.maxEvals=500 #for scipy differntial evolution. Shouldn't be more than this
        self.spotCaptureDiam = 5e-3
        self.collectorAngleMax = .06
        self.temperature = 3e-3
        fractionalMarginOfError = 1.25
        self.minElementLength = fractionalMarginOfError * self.particleTracerRing.minTimeStepsPerElement * \
                                self.latticeRing.v0Nominal * self.h
        self.tunableTotalLengthList = []  # list to hold initial tunable lengths for when optimizing by tuning element
        # length. this to prevent any numerical round issues causing the tunable length to change from initial value
        # if I do many iterations
        self.tuningBounds = None

        numParticlesFullSwarm=300
        numParticlesSurrogate=50
        self.swarmInjectorInitial = self.swarmTracerInjector.initialize_Observed_Collector_Swarm_Probability_Weighted(
            self.spotCaptureDiam, self.collectorAngleMax, numParticlesFullSwarm, temperature=self.temperature,
            sameSeed=self.sameSeedForSwarm, upperSymmetry=self.useLatticeUpperSymmetry)
        self.swarmInjectorInitial_Surrogate = self.swarmTracerInjector.initialize_Observed_Collector_Swarm_Probability_Weighted(
            self.spotCaptureDiam, self.collectorAngleMax, numParticlesSurrogate, temperature=self.temperature,
            sameSeed=self.sameSeedForSwarm, upperSymmetry=self.useLatticeUpperSymmetry)
    def get_Injector_Shapely_Objects_In_Lab_Frame(self):
        newShapelyObjectList = []
        rotationAngle = self.latticeInjector.combiner.ang + -self.latticeRing.combiner.ang
        r2Injector = self.latticeInjector.combiner.r2
        r2Ring = self.latticeRing.combiner.r2
        for el in self.latticeInjector.elList:
            SO = copy.copy(el.SO_Outer)
            SO = translate(SO, xoff=-r2Injector[0], yoff=-r2Injector[1])
            SO = rotate(SO, rotationAngle, use_radians=True, origin=(0, 0))
            SO = translate(SO, xoff=r2Ring[0], yoff=r2Ring[1])
            newShapelyObjectList.append(SO)
        return newShapelyObjectList

    def generate_Shapely_Floor_Plan(self):
        shapelyObjectList = []
        shapelyObjectList.extend([el.SO_Outer for el in self.latticeRing.elList])
        shapelyObjectList.extend(self.get_Injector_Shapely_Objects_In_Lab_Frame())
        return shapelyObjectList

    def floor_Plan_OverLap(self):
        injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame()
        ringShapelyObjects = [el.SO_Outer for el in self.latticeRing.elList]
        injectorLens = injectorShapelyObjects[1]
        area = 0
        for element in ringShapelyObjects:
            area += element.intersection(injectorLens).area
        return area

    def is_Floor_Plan_Valid(self):
        if self.floor_Plan_OverLap() > 0.0:
            return False
        else:
            return True

    def show_Floor_Plan(self,X):
        self.update_Ring_And_Injector(X)
        shapelyObjectList = self.generate_Shapely_Floor_Plan()
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy,c='black')
        plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        plt.show()

    def mode_Match_Cost(self, X,useSurrogate,parallel):
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        for val, bounds in zip(X, self.tuningBounds):
            assert bounds[0] <= val <= bounds[1]
        if self.test_Stability(X) == False:
            swarmTraced=Swarm() #empty swarm
        else:
            swarmTraced=self.inject_Swarm(X,useSurrogate,parallel)
        cost = self.cost_Function(swarmTraced,X)
        return cost
    def inject_Swarm(self,X,useSurrogate,parallel):
        self.update_Ring_And_Injector(X)
        swarmInitial = self.trace_And_Project_Injector_Swarm_To_Combiner_End(useSurrogate)
        swarmTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmInitial, self.h, self.T, parallel=parallel,
                                                                       fastMode=True, accelerated=True, copySwarm=False,)
        return swarmTraced
    def move_Survived_Particles_In_Injector_Swarm_To_Origin(self, swarmInjectorTraced, copyParticles=False):
        #fidentify particles that survived to combiner end, walk them right up to the end, exclude any particles that
        #are now clipping the combiner and any that would clip the next element
        #NOTE: The particles offset is taken from the origin of the orbit output of the combiner, not the 0,0 output
        apNextElement = self.latticeRing.elList[self.latticeRing.combinerIndex + 1].ap
        swarmSurvived = Swarm()
        for particle in swarmInjectorTraced:
            outputCenter=self.latticeInjector.combiner.r2+self.swarmTracerInjector.combiner_Output_Offset_Shift()
            qf = particle.qf - outputCenter
            qf[:2] = self.latticeInjector.combiner.RIn @ qf[:2]
            if qf[0] <= self.h * self.latticeRing.v0Nominal:  # if the particle is within a timestep of the end,
                # assume it's at the end
                pf = particle.pf.copy()
                pf[:2] = self.latticeInjector.combiner.RIn @ particle.pf[:2]
                qf = qf + pf * np.abs(qf[0] / pf[0]) #walk particle up to the end of the combiner
                qf[0]=0.0 #no rounding error
                clipsNextElement=np.sqrt(qf[1] ** 2 + qf[2] ** 2) > apNextElement
                if clipsNextElement==False:  # test that particle survives through next aperture
                    if copyParticles == False:
                        particleEnd = particle
                    else:
                        particleEnd = particle.copy()
                    particleEnd.qi = qf
                    particleEnd.pi = pf
                    particleEnd.reset()
                    swarmSurvived.particles.append(particleEnd)
        return swarmSurvived

    def trace_And_Project_Injector_Swarm_To_Combiner_End(self,useSurrogate) -> Swarm:
        if useSurrogate==True:
            swarm=self.swarmInjectorInitial_Surrogate
        else:
            swarm=self.swarmInjectorInitial
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            swarm.quick_Copy()
            , self.h, 1.0, parallel=False,
            fastMode=True, copySwarm=False,
            accelerated=True)
        swarmEnd = self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced, copyParticles=False)
        swarmEnd = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False,scoot=True)
        return swarmEnd

    def test_Stability(self, X,minRevsToTest=5.0):
        self.update_Ring_And_Injector(X)
        maxInitialTransversePos=1e-3
        swarmTestInitial = self.swarmTracerRing.initialize_Stablity_Testing_Swarm(maxInitialTransversePos)
        swarmTestAtCombiner = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmTestInitial)
        swarmTestTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmTestAtCombiner, self.h,
                                1.1 * minRevsToTest * self.latticeRing.totalLength / self.latticeRing.v0Nominal,
                                                            parallel=False, accelerated=False,
                                                            fastMode=True)
        stable = False
        for particle in swarmTestTraced:
            if particle.revolutions > minRevsToTest: #any stable particle, consider lattice stable
                stable = True
        return stable

    def update_Ring_And_Injector(self,X):
        assert len(X)==4
        XRing = X[:2]
        XInjector = X[2:]
        self.update_Injector_Lattice(XInjector)
        self.update_Ring_Lattice(XRing)
    def update_Injector_Lattice(self, X):
        # modify lengths of drift regions in injector
        assert len(X)==2
        self.latticeInjector.elList[0].set_Length(X[0])
        self.latticeInjector.elList[2].set_Length(X[1])
        self.latticeInjector.build_Lattice()

    def update_Ring_Lattice(self, X):
        assert len(X)==2
        if self.tuningChoice == 'field':
            self.update_Ring_Field_Values(X)
        elif self.tuningChoice == 'spacing':
            self.update_Ring_Spacing(X)
        else:
            raise Exception('wrong tuning choice')

    def update_Ring_Field_Values(self, X):
        for el, arg in zip(self.tunedElementList, X):
            el.fieldFact = arg

    def update_Ring_Spacing(self, X):
        for elCenter, spaceFracElBefore, totalLength in zip(self.tunedElementList, X, self.tunableTotalLengthList):
            elBefore, elAfter = self.latticeRing.get_Element_Before_And_After(elCenter)
            assert isinstance(elBefore,Drift) and isinstance(elAfter,Drift)
            tunableLength = (elBefore.L + elAfter.L) - 2 * self.minElementLength
            LBefore = spaceFracElBefore * tunableLength + self.minElementLength
            LAfter = totalLength - LBefore
            elBefore.set_Length(LBefore)
            elAfter.set_Length(LAfter)
        self.latticeRing.build_Lattice()

    def compute_Swarm_Survival(self, swarmTraced:Swarm):
        # A reasonable metric is the total flux multiplication of the beam. If the average particle survives n
        # revolutions then the beam has been multiplied n times. I want this to be constrained between 0 and 1 however,
        #so I will use the baseline as the smallest possible lattice. The smallest is defined as a lattice with a
        # bending radius of 1 meter and the combiner. This makes the results better for smaller actual lattices
        if swarmTraced.num_Particles()==0:
            return 0.0
        fluxMult = sum([p.revolutions * p.probability for p in swarmTraced.particles])
        rBendNominal=1.0
        Lcombiner=self.latticeRing.combiner.Lo
        minLatticeLength=2*(np.pi*rBendNominal+Lcombiner)
        maxRevs=self.T * self.latticeRing.v0Nominal / minLatticeLength #approximately correct
        maxFluxMult=sum([maxRevs*p.probability for p in self.swarmInjectorInitial])
        survival = 1e2 * fluxMult/maxFluxMult
        assert 0.0 <= survival <= 100.0
        return survival
    def floor_Plan_Cost(self,X):
        self.update_Ring_And_Injector(X)
        overlap=self.floor_Plan_OverLap()
        factor = 1e-3
        cost = 2 / (1 + np.exp(-overlap / factor)) - 1
        assert 0.0<=cost<=1.0
        return cost
    def cost_Function(self, swarm,X):
        survival=self.compute_Swarm_Survival(swarm)
        survivalCost=(100.0-survival)/100.0
        cost=survivalCost+self.floor_Plan_Cost(X)
        assert 0.0<=cost<=2.0
        return cost
    def survival_From_Cost(self, cost,X):
        # returns survival
        survivalCost=cost-self.floor_Plan_Cost(X)
        survival=100.0*(1.0-survivalCost)
        assert 0.0<=survival<=100.0
        return survival
    def catch_Optimizer_Errors(self, tuningBounds, elementIndices, tuningChoice):
        if max(elementIndices) >= len(self.latticeRing.elList) - 1: raise Exception("element indices out of bounds")
        if len(tuningBounds) != len(elementIndices): raise Exception("Bounds do not match number of tuned elements")
        if self.latticeRing.combiner.L!=self.latticeInjector.combiner.L and \
            self.latticeRing.combiner.ap!=self.latticeInjector.combiner.ap:
            raise Exception('Combiners are different between the two lattices')
        if tuningChoice == 'field':
            for el in self.tunedElementList:
                if (isinstance(el, LensIdeal) and isinstance(el, HalbachLensSim)) != True:
                    raise Exception("For field tuning elements must be LensIdeal or HalbachLensSim")
        elif tuningChoice == 'spacing':
            for elIndex in elementIndices:
                elBefore, elAfter = self.latticeRing.get_Element_Before_And_After(self.latticeRing.elList[elIndex])
                tunableLength = (elBefore.L + elAfter.L) - 2 * self.minElementLength
                if (isinstance(elBefore, Drift) and isinstance(elAfter, Drift)) != True:
                    raise Exception("For spacing tuning neighboring elements must be Drift elements")
                if tunableLength < 0.0:
                    raise Exception("Tunable elements are too short for length tuning. Min total length is "
                                    + str(2 * self.minElementLength))
        else:
            raise Exception('No proper tuning choice provided')

    def make_Initial_Coords_List(self):
        # gp_minimize requires a list of lists, I make that here
        numGridEdge = 5
        meshgridArraysList = [np.linspace(bound[0], bound[1], numGridEdge) for bound in self.tuningBounds]
        tuningCoordsArr = np.asarray(np.meshgrid(*meshgridArraysList)).T.reshape(-1, len(self.tuningBounds))
        tuningCoordsList = tuningCoordsArr.tolist()  # must in list format
        return tuningCoordsList

    def fill_Initial_Total_Tuning_Elements_Length_List(self):
        for elCenter in self.tunedElementList:
            elBefore, elAfter = self.latticeRing.get_Element_Before_And_After(elCenter)
            self.tunableTotalLengthList.append(elBefore.L + elAfter.L)

    def initialize_Optimizer(self, elementIndices, tuningChoice, ringTuningBounds, injectorTuningBounds):
        self.tuningBounds = ringTuningBounds.copy()
        self.tuningBounds.extend(injectorTuningBounds)
        self.tunedElementList = [self.latticeRing.elList[index] for index in elementIndices]
        self.tuningChoice = tuningChoice
        if tuningChoice == 'spacing':
            self.fill_Initial_Total_Tuning_Elements_Length_List()
        if self.sameSeedForSearch == True:
            np.random.seed(42)

    def test_Lattice_Stability(self, ringTuningBounds,injectorTuningBounds, numEdgePoints=30, parallel=False):
        assert len(ringTuningBounds) == 2 and len(injectorTuningBounds)==2
        x1Arr = np.linspace(ringTuningBounds[0][0], ringTuningBounds[0][1], numEdgePoints)
        x2Arr = np.linspace(ringTuningBounds[1][0], ringTuningBounds[1][1], numEdgePoints)
        x3Arr = (ringTuningBounds[0][1]-injectorTuningBounds[0][0])*np.ones(numEdgePoints**2)
        x4Arr = (ringTuningBounds[1][1]-injectorTuningBounds[1][0])*np.ones(numEdgePoints**2)
        testCoords = np.asarray(np.meshgrid(x1Arr, x2Arr)).T.reshape(-1, 2)
        testCoords=np.column_stack((testCoords,x3Arr,x4Arr))
        if parallel == False:
            stabilityList = [self.test_Stability(coords) for coords in testCoords]
        else:
            with mp.Pool(mp.cpu_count()) as pool:
                stabilityList=pool.map(self.test_Stability,testCoords)
        assert len(stabilityList)==numEdgePoints**2
        if sum(stabilityList) == 0:
            return False
        else:
            return True
    def _fast_Minimize(self):
        #less accurate method that minimizes with a smaller surrogate swarm then uses the full swarm for the final
        #value
        sol_Surrogate = spo.differential_evolution(self.mode_Match_Cost, self.tuningBounds, tol=self.tolerance,
                                                   polish=False,args=(True,False),
                                                   maxiter=self.maxEvals//(self.optimalPopSize*len(self.tuningBounds)),
                                                   popsize=self.optimalPopSize)
        cost_Real=self.mode_Match_Cost(sol_Surrogate.x,False,False)
        return cost_Real,sol_Surrogate.x
    def _accurate_Minimize(self):
        #start first by quickly randomly searching with a surrogate swarm.
        randomSamplePoints=128

        samples = skopt.sampler.Sobol().generate(self.tuningBounds, randomSamplePoints)
        vals=[self.mode_Match_Cost(sample,True,False) for sample in samples]
        XInitial=samples[np.argmin(vals)]
        sol=spo.differential_evolution(self.mode_Match_Cost,self.tuningBounds,polish=False,x0=XInitial,tol=self.tolerance,
                                       maxiter=self.maxEvals//(self.optimalPopSize*len(self.tuningBounds)),args=(False,False))
        return sol.cost,sol.x

    def _minimize(self)->Solution:
        if self.fastSolver==True:
            cost,xOptimal=self._fast_Minimize()
        else:
            cost,xOptimal=self._accurate_Minimize()
        sol = Solution()
        sol.cost = cost
        optimalConfig = xOptimal
        sol.survival = self.survival_From_Cost(cost, optimalConfig)
        sol.stable = True
        sol.xRing_TunedParams2 = optimalConfig[:2]
        sol.xInjector_TunedParams = optimalConfig[2:]
        sol.invalidInjector = False
        sol.invalidRing = False
        return sol
    def optimize(self, elementIndices, ringTuningBounds=None, injectorTuningBounds=None, tuningChoice='spacing',
                 parallel=True,fastSolver=True)->Solution:
        self.fastSolver=fastSolver
        if ringTuningBounds is None:
            ringTuningBounds = [(.2, .8), (.2, .8)]
        if injectorTuningBounds is None:
            injectorTuningBounds = [(.05, .4), (.05, .4)]
        self.catch_Optimizer_Errors(ringTuningBounds, elementIndices, tuningChoice)
        self.initialize_Optimizer(elementIndices, tuningChoice, ringTuningBounds, injectorTuningBounds)
        if self.test_Lattice_Stability(ringTuningBounds,injectorTuningBounds, parallel=parallel) == False:
            sol = Solution()
            sol.survival = 0.0
            sol.cost=1.0
            sol.stable=False
            return sol
        sol=self._minimize()
        return sol