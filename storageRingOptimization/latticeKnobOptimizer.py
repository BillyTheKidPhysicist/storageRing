import random
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
        self.fluxMultiplicationPercent = np.nan
        self.cost=np.nan
        self.scipyMessage=None
        self.stable=None
        self.invalidInjector=None
        self.invalidRing=None
        self.description = None
        self.bumpParams = None  # List of parameters used in the misalignment testing

    def __str__(self):  # method that gets called when you do print(Solution())
        string = '----------Solution-----------   \n'
        string += 'injector element spacing optimum configuration: ' + str(self.xInjector_TunedParams) + '\n'
        string += 'storage ring tuned params 1 optimum configuration: ' + str(self.xRing_TunedParams1) + '\n'
        string += 'storage ring tuned params 2 optimum configuration: ' + str(self.xRing_TunedParams2) + '\n'
        # string+='stable configuration:'+str(self.stable)+'\n'
        # string += 'bump params: ' + str(self.bumpParams) + '\n'
        string+='cost: '+str(self.cost)+'\n'
        string += 'percent max flux multiplication: ' + str(self.fluxMultiplicationPercent) + '\n'
        string += 'scipy message: '+ str(self.scipyMessage)+'\n'
        # string+='flux multiplication: '+str(int(self.fluxMultiplication)) +'\n'
        string += '----------------------------'
        return string


class LatticeOptimizer:
    def __init__(self, latticeRing, latticeInjector):

        self.latticeRing = latticeRing
        self.latticeInjector = latticeInjector
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
        self.tolerance=.03 #for scipy differential evolution. This is the maximum accuravy roughly speaking
        self.maxEvals=300 #for scipy differntial evolution. Shouldn't be more than this
        self.spotCaptureDiam = 1e-2
        self.collectorAngleMax = .06
        self.temperature = 1e-3
        self.gamma_Space=3.5e-3
        fractionalMarginOfError = 1.25
        self.minElementLength = fractionalMarginOfError * self.particleTracerRing.minTimeStepsPerElement * \
                                self.latticeRing.v0Nominal * self.h
        self.tunableTotalLengthList = []  # list to hold initial tunable lengths for when optimizing by tuning element
        # length. this to prevent any numerical round issues causing the tunable length to change from initial value
        # if I do many iterations
        self.tuningBounds = None
        self.numParticlesFullSwarm=1000
        self.numParticlesSurrogate=50
        self.generate_Swarms()

    def generate_Swarms(self):
        self.swarmInjectorInitial = self.swarmTracerInjector.initialize_Observed_Collector_Swarm_Probability_Weighted(
            self.spotCaptureDiam, self.collectorAngleMax, self.numParticlesFullSwarm, temperature=self.temperature,
            sameSeed=self.sameSeedForSwarm, upperSymmetry=self.useLatticeUpperSymmetry,gammaSpace=self.gamma_Space,
            probabilityMin=.05)
        np.random.shuffle(self.swarmInjectorInitial.particles)
        self.swarmInjectorInitial_Surrogate=Swarm()
        self.swarmInjectorInitial_Surrogate.particles=self.swarmInjectorInitial.particles[:self.numParticlesSurrogate]
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

    def floor_Plan_OverLap_mm(self):
        injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame()
        assert len(injectorShapelyObjects)==4
        assert isinstance(self.latticeInjector.elList[1],HalbachLensSim)
        ringShapelyObjects = [el.SO_Outer for el in self.latticeRing.elList]
        injectorLens = injectorShapelyObjects[1]
        area = 0
        converTo_mm=(1e3)**2
        for element in ringShapelyObjects:
            area += element.intersection(injectorLens).area*converTo_mm
        return area


    def show_Floor_Plan(self,X):
        self.update_Ring_And_Injector(X)
        shapelyObjectList = self.generate_Shapely_Floor_Plan()
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy,c='black')
        plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        plt.show()

    def mode_Match_Cost(self, X,useSurrogate,energyCorrection):
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        for val, bounds in zip(X, self.tuningBounds):
            assert bounds[0] <= val <= bounds[1]
        if self.floor_Plan_Cost(X)>0.0: #only permit small floor plan costs
            return 1.0+self.floor_Plan_Cost(X)
        if self.test_Stability(X) == False:
            swarmTraced=Swarm() #empty swarm
        else:
            swarmTraced=self.inject_Swarm(X,useSurrogate,energyCorrection)
        cost = self.cost_Function(swarmTraced,X)
        return cost
    def inject_Swarm(self,X,useSurrogate,energyCorrection):
        self.update_Ring_And_Injector(X)
        swarmInitial = self.trace_And_Project_Injector_Swarm_To_Combiner_End(useSurrogate)
        swarmTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmInitial, self.h, self.T,
                                    fastMode=True, accelerated=True, copySwarm=False,energyCorrection=energyCorrection)
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
            , self.h, 1.0, fastMode=True, copySwarm=False, accelerated=True)
        swarmEnd = self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced, copyParticles=False)
        swarmEnd = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False,scoot=True)
        return swarmEnd

    def test_Stability(self, X,minRevsToTest=5.0):
        self.update_Ring_And_Injector(X)
        maxInitialTransversePos=1e-3
        T_Max=1.1 * minRevsToTest * self.latticeRing.totalLength / self.latticeRing.v0Nominal
        swarmTestInitial = self.swarmTracerRing.initialize_Stablity_Testing_Swarm(maxInitialTransversePos)
        swarmTestAtCombiner = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmTestInitial)
        swarmTestTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmTestAtCombiner, self.h,
                                T_Max, accelerated=False, fastMode=True)
        stable = False
        for particle in swarmTestTraced:
            if particle.revolutions > minRevsToTest: #any stable particle, consider lattice stable
                stable = True
        return stable

    def update_Ring_And_Injector(self,X):
        assert len(X)==4
        XRing = X[:2]
        XInjector = X[2:]
        self.update_Ring_Lattice(XRing)
        self.update_Injector_Lattice(XInjector)
    def update_Injector_Lattice(self, X):
        # modify lengths of drift regions in injector
        assert len(X)==2
        assert X[0]>0.0 and X[1]>0.0
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
        raise Exception('Not currently supported')
        for el, arg in zip(self.tunedElementList, X):
            el.fieldFact = arg

    def update_Ring_Spacing(self, X):
        for elCenter, spaceFracElBefore, totalLength in zip(self.tunedElementList, X, self.tunableTotalLengthList):
            assert 0<=spaceFracElBefore<=1.0
            elBefore, elAfter = self.latticeRing.get_Element_Before_And_After(elCenter)
            assert isinstance(elBefore,Drift) and isinstance(elAfter,Drift)
            tunableLength = (elBefore.L + elAfter.L) - 2 * self.minElementLength
            LBefore = spaceFracElBefore * tunableLength + self.minElementLength
            LAfter = totalLength - LBefore
            elBefore.set_Length(LBefore)
            elAfter.set_Length(LAfter)
        self.latticeRing.build_Lattice()

    def compute_Swarm_Flux_Mult_Percent(self, swarmTraced:Swarm):
        # What percent of the maximum flux multiplication is the swarm reaching? It's cruical I consider that not
        #all particles survived through the lattice.
        assert all([particle.traced == True for particle in swarmTraced.particles])
        if swarmTraced.num_Particles()==0: return 0.0
        weightedFluxMult=swarmTraced.weighted_Flux_Multiplication()
        weightedFluxMultMax=self.maximum_Weighted_Flux_Multiplication()
        injectionSurvivalFrac=swarmTraced.num_Particles(weighted=True)/\
                              self.swarmInjectorInitial.num_Particles(weighted=True) #particle may be lost
        #during injection
        fluxMultPerc=1e2*(weightedFluxMult/weightedFluxMultMax)*injectionSurvivalFrac
        assert 0.0 <= fluxMultPerc <= 100.0
        return fluxMultPerc
    def maximum_Weighted_Flux_Multiplication(self):
        #unrealistic maximum flux of lattice
        rBendNominal=1.0
        LCombinerNominal=.2
        minLatticeLength=2*(np.pi*rBendNominal+LCombinerNominal)
        maxFluxMult=self.T * self.latticeRing.v0Nominal / minLatticeLength #the aboslute max
        return maxFluxMult
    def floor_Plan_Cost(self,X):
        self.update_Ring_And_Injector(X)
        overlap=self.floor_Plan_OverLap_mm() #units of mm^2
        factor = 300 #units of mm^2
        cost = 2 / (1 + np.exp(-overlap / factor)) - 1
        assert 0.0<=cost<=1.0
        return cost
    def cost_Function(self, swarm,X):
        fluxMultPerc=self.compute_Swarm_Flux_Mult_Percent(swarm)
        fluxCost=(100.0-fluxMultPerc)/100.0
        cost=fluxCost+self.floor_Plan_Cost(X)
        assert 0.0<=cost<=2.0
        return cost
    def flux_Percent_From_Cost(self, cost,X):
        fluxCost=cost-self.floor_Plan_Cost(X)
        fluxMulPerc=100.0*(1.0-fluxCost)
        assert 0.0<=fluxMulPerc<=100.0
        return fluxMulPerc
    def catch_Optimizer_Errors(self, tuningBounds, tuningElementIndices, tuningChoice):
        if max(tuningElementIndices) >= len(self.latticeRing.elList) - 1: raise Exception("element indices out of bounds")
        if len(tuningBounds) != len(tuningElementIndices): raise Exception("Bounds do not match number of tuned elements")
        if self.latticeRing.combiner.L!=self.latticeInjector.combiner.L and \
            self.latticeRing.combiner.ap!=self.latticeInjector.combiner.ap:
            raise Exception('Combiners are different between the two lattices')
        if tuningChoice == 'field':
            for el in self.tunedElementList:
                if (isinstance(el, LensIdeal) and isinstance(el, HalbachLensSim)) != True:
                    raise Exception("For field tuning elements must be LensIdeal or HalbachLensSim")
        elif tuningChoice == 'spacing':
            for elIndex in tuningElementIndices:
                elBefore, elAfter = self.latticeRing.get_Element_Before_And_After(self.latticeRing.elList[elIndex])
                tunableLength = (elBefore.L + elAfter.L) - 2 * self.minElementLength
                if (isinstance(elBefore, Drift) and isinstance(elAfter, Drift)) != True:
                    raise Exception("For spacing tuning neighboring elements must be Drift elements")
                if tunableLength < 0.0:
                    raise Exception("Tunable elements are too short for length tuning. Min total length is "
                                    + str(2 * self.minElementLength))
        else: raise Exception('No proper tuning choice provided')

    def fill_Initial_Total_Tuning_Elements_Length_List(self):
        for elCenter in self.tunedElementList:
            elBefore, elAfter = self.latticeRing.get_Element_Before_And_After(elCenter)
            self.tunableTotalLengthList.append(elBefore.L + elAfter.L)

    def initialize_Optimizer(self, tuningElementIndices, tuningChoice, ringTuningBounds, injectorTuningBounds):
        self.tuningBounds = ringTuningBounds.copy()
        self.tuningBounds.extend(injectorTuningBounds)
        self.tunedElementList = [self.latticeRing.elList[index] for index in tuningElementIndices]
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
                                                   popsize=self.optimalPopSize,init='halton')
        return sol_Surrogate
    def _accurate_Minimize(self):
        #start first by quickly randomly searching with a surrogate swarm.
        randomSamplePoints=128
        samples = skopt.sampler.Sobol().generate(self.tuningBounds, randomSamplePoints)
        vals=[self.mode_Match_Cost(sample,True,False) for sample in samples]
        XInitial=samples[np.argmin(vals)]
        sol=spo.differential_evolution(self.mode_Match_Cost,self.tuningBounds,polish=False,x0=XInitial,tol=self.tolerance,
                                       maxiter=self.maxEvals//(self.optimalPopSize*len(self.tuningBounds)),
                                       args=(False,False),popsize=self.optimalPopSize,init='halton')
        return sol

    def _minimize(self)->Solution:
        if self.fastSolver==True:
            scipySol=self._fast_Minimize()
        else:
            scipySol=self._accurate_Minimize()
        cost_Most_Accurate = self.mode_Match_Cost(scipySol.x, False, True)
        sol = Solution()
        sol.scipyMessage=scipySol.message
        sol.cost = cost_Most_Accurate
        optimalConfig = scipySol.x
        sol.fluxMultiplicationPercent = self.flux_Percent_From_Cost(cost_Most_Accurate, optimalConfig)
        sol.stable = True
        sol.xRing_TunedParams2 = optimalConfig[:2]
        sol.xInjector_TunedParams = optimalConfig[2:]
        sol.invalidInjector = False
        sol.invalidRing = False
        return sol
    def optimize(self, tuningElementIndices, ringTuningBounds=None, injectorTuningBounds=None, tuningChoice='spacing',
                 parallel=False,fastSolver=True)->Solution:
        self.fastSolver=fastSolver
        if ringTuningBounds is None:
            ringTuningBounds = [(.2, .8), (.2, .8)]
        if injectorTuningBounds is None:
            injectorTuningBounds = [(.1, .4), (.1, .4)]
        self.catch_Optimizer_Errors(ringTuningBounds, tuningElementIndices, tuningChoice)
        self.initialize_Optimizer(tuningElementIndices, tuningChoice, ringTuningBounds, injectorTuningBounds)
        if self.test_Lattice_Stability(ringTuningBounds,injectorTuningBounds, parallel=parallel) == False:
            sol = Solution()
            sol.fluxMultiplicationPercent = 0.0
            sol.cost=1.0
            sol.stable=False
            return sol
        sol=self._minimize()
        return sol