from skopt.plots import plot_objective
import warnings
from shapely.affinity import rotate, translate
import black_box as bb
import sys
import numpy.linalg as npl
from profilehooks import profile
import copy
import skopt
from ParticleTracerClass import ParticleTracer
import numpy as np
from ParticleClass import Swarm, Particle
import scipy.optimize as spo
import time
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from ParaWell import ParaWell
from SwarmTracerClass import SwarmTracer
from elementPT import HalbachLensSim, LensIdeal, Drift
import globalMethods as gm


class Solution:
    # class to hold onto results of each solution
    def __init__(self):
        self.xInjector_TunedParams = np.nan
        self.xRing_TunedParams1 = np.nan  # paramters tuned by the 'outer' gp minimize
        self.xRing_TunedParams2 = np.nan  # paramters tuned by the 'inner' gp minimize
        self.survival = np.nan
        self.cost=np.nan
        self.description = None
        self.bumpParams = None  # List of parameters used in the misalignment testing

    def __str__(self):  # method that gets called when you do print(Solution())
        string = '----------Solution-----------   \n'
        string += 'injector element spacing optimum configuration: ' + str(self.xInjector_TunedParams) + '\n '
        string += 'storage ring tuned params 1 optimum configuration: ' + str(self.xRing_TunedParams1) + '\n '
        string += 'storage ring tuned params 2 optimum configuration: ' + str(self.xRing_TunedParams2) + '\n '
        string += 'bump params: ' + str(self.bumpParams) + '\n'
        string+='cost: '+str(self.cost)+'\n'
        string += 'survival: ' + str(self.survival) + '\n'
        string += '----------------------------'
        return string


class LatticeOptimizer:
    def __init__(self, latticeRing, latticeInjector):

        self.latticeRing = latticeRing
        self.latticeInjector = latticeInjector
        self.helper = ParaWell()  # custom class to help with parallelization
        self.i = 0  # simple variable to track solution status
        self.particleTracerRing = ParticleTracer(latticeRing)
        self.particleTracerInjector = ParticleTracer(latticeInjector)
        self.swarmTracerInjector = SwarmTracer(self.latticeInjector)
        self.h = 5e-6  # timestep size
        self.T = 10.0
        self.swarmTracerRing = SwarmTracer(self.latticeRing)
        self.phaseSpaceFunc = None  # function that returns the number of revolutions of a particle at a given
        # point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.solutionList = []  # list to hold solution objects the track coordsinates and function values for injector
        # and ring paramters
        self.tunedElementList = None
        self.tuningChoice = None  # what type of tuning will occur
        self.useLatticeUpperSymmetry = True  # exploit the fact that the lattice has symmetry in the z axis to use half
        # the number of particles. Symmetry is broken if including gravity
        self.sameSeedForSwarm = True  # generate the same swarms every time by seeding the random generator during swarm
        # generation with the same number, 42
        self.sameSeedForSearch = True  # wether to use the same seed, 42, for the search process
        self.numParticlesInjector = 500
        self.postCombinerAperture = self.latticeRing.elList[self.latticeRing.combinerIndex + 1].ap  # radius
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
        self.swarmInjectorInitial = self.swarmTracerInjector.initialize_Observed_Collector_Swarm_Probability_Weighted(
            self.spotCaptureDiam, self.collectorAngleMax, self.numParticlesInjector, temperature=self.temperature,
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

    def show_Floor_Plan(self):
        shapelyObjectList = self.generate_Shapely_Floor_Plan()
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy)
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.show()

    def mode_Match_Cost(self, X, parallel=False):
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        for val, bounds in zip(X, self.tuningBounds):
            assert bounds[0] <= val <= bounds[1]
        if self.test_Stability(X) == False:
            return 1.0
        swarmTraced=self.inject_Swarm(X,parallel=parallel)
        cost = self.cost_Function(swarmTraced,X)
        return cost
    def inject_Swarm(self,X,parallel=False):
        self.update_Ring_And_Injector(X)
        swarmInitial = self.trace_And_Project_Injector_Swarm_To_Combiner_End()
        swarmInitial.reset()
        swarmTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmInitial, self.h, self.T, parallel=parallel,
                                                                       fastMode=True, accelerated=True, copySwarm=False)
        return swarmTraced
    def move_Survived_Particles_In_Injector_Swarm_To_Origin(self, swarmInjectorTraced, copyParticles=False):
        apNextElement = self.latticeRing.elList[self.latticeRing.combinerIndex + 1].ap
        swarmEnd = Swarm()
        for particle in swarmInjectorTraced:
            q = particle.q.copy() - self.latticeInjector.combiner.r2
            q[:2] = self.latticeInjector.combiner.RIn @ q[:2]
            if q[0] <= self.h * self.latticeRing.v0Nominal:  # if the particle is within a timestep of the end,
                # assume it's at the end
                p = particle.p.copy()
                p[:2] = self.latticeInjector.combiner.RIn @ p[:2]
                q = q + p * np.abs(q[0] / p[0])
                if np.sqrt(q[1] ** 2 + q[2] ** 2) < apNextElement:  # test that particle survives through next aperture
                    if copyParticles == False:
                        particleEnd = particle
                    else:
                        particleEnd = particle.copy()
                    particleEnd.q = q
                    particleEnd.p = p
                    swarmEnd.particles.append(particleEnd)
        return swarmEnd

    def trace_And_Project_Injector_Swarm_To_Combiner_End(self) -> Swarm:
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            self.swarmInjectorInitial.quick_Copy()
            , self.h, 1.0, parallel=False,
            fastMode=True, copySwarm=False,
            accelerated=True)
        swarmEnd = self.move_Survived_Particles_In_Injector_Swarm_To_Origin(swarmInjectorTraced, copyParticles=False)
        swarmEnd = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmEnd, copySwarm=False)
        return swarmEnd

    def test_Stability(self, X,minRevs=5.0):
        self.update_Ring_And_Injector(X)
        swarmTest = self.swarmTracerRing.initialize_Stablity_Testing_Swarm(1e-3)
        swarmTest = self.swarmTracerRing.move_Swarm_To_Combiner_Output(swarmTest)
        swarmTest = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmTest, self.h,
                                                                     1.5 * minRevs * self.latticeRing.totalLength / 200.0,
                                                                     parallel=False, accelerated=True, fastMode=True)
        stable = False
        for particle in swarmTest:
            if particle.revolutions > minRevs:
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
            tunableLength = (elBefore.L + elAfter.L) - 2 * self.minElementLength
            LBefore = spaceFracElBefore * tunableLength + self.minElementLength
            LAfter = totalLength - LBefore
            elBefore.set_Length(LBefore)
            elAfter.set_Length(LAfter)
        self.latticeRing.build_Lattice()

    def compute_Swarm_Survival(self, swarmTraced):
        # A reasonable metric is the total flux multiplication of the beam. If the average particle survives n
        # revolutions then the beam has been multiplied n times. I want this to be constrained between 0 and 1 however,
        #so I will use the baseline as the smallest possible lattice. The smallest is defined as a lattice with a
        # bending radius of 1 meter and the combiner. This makes the results better for smaller actual lattices
        fluxMult = np.mean(np.asarray([p.revolutions * p.probability for p in swarmTraced.particles]))
        rBend=1.0
        Lcombiner=self.latticeRing.combiner.Lo
        minLatticeLength=2*(np.pi*rBend+Lcombiner)
        maxFluxMult=self.T * self.latticeRing.v0Nominal / minLatticeLength #approximately
        #correct
        survival = 1e2 * fluxMult/maxFluxMult
        assert 0.0 <= survival <= 100.0
        return survival
    def floor_Plan_Cost(self,X):
        self.update_Ring_And_Injector(X)
        return self.floor_Plan_OverLap()/1e-3
    def cost_Function(self, swarm,X):
        survival=self.compute_Swarm_Survival(swarm)
        survivalCost=(100.0-survival)/100.0
        cost=survivalCost+self.floor_Plan_Cost(X)
        assert cost>=0.0
        return cost
    def survival_From_Cost(self, cost,X):
        # returns survival
        survivalCost=cost-self.floor_Plan_Cost(X)
        survival=100.0*(1.0-survivalCost)
        return survival

    def catch_Optimizer_Errors(self, tuningBounds, elementIndices, tuningChoice):
        if len(self.solutionList) != 0: raise Exception("You cannot call optimize twice")
        if max(elementIndices) >= len(self.latticeRing.elList) - 1: raise Exception("element indices out of bounds")
        if len(tuningBounds) != len(elementIndices): raise Exception("Bounds do not match number of tuned elements")
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

    def test_Lattice_Stability(self, ringTuningBounds, numEdgePoints=30, parallel=False):
        assert len(ringTuningBounds) == 2
        x1Arr = np.linspace(ringTuningBounds[0][0], ringTuningBounds[0][1], numEdgePoints)
        x2Arr = np.linspace(ringTuningBounds[1][0], ringTuningBounds[1][1], numEdgePoints)
        x34Val=.1
        testCoords = np.asarray(np.meshgrid(x1Arr, x2Arr)).T.reshape(-1, 2)
        testCoords=np.column_stack((testCoords,np.zeros(testCoords.shape)))
        if parallel == False:
            stablilityList = [self.test_Stability(coords) for coords in testCoords]
        else:
            stablilityList = self.helper.parallel_Problem(self.test_Stability, argsList=testCoords,
                                                          onlyReturnResults=True)
        if sum(stablilityList) == 0:
            return False
        else:
            return True
    def _minimize(self,parallel)->Solution:
        if parallel == True:
            numWorkers = -1
        else:
            numWorkers = 1
        spoSol = spo.differential_evolution(self.mode_Match_Cost, self.tuningBounds, tol=.01, polish=False,
                                            workers=numWorkers)

        spoSol = spo.minimize(self.mode_Match_Cost, spoSol.x, args=(parallel), bounds=self.tuningBounds,
                              method='Nelder-Mead',options={'ftol':.001})
        sol=Solution()
        cost=spoSol.fun
        costMax=1.0
        if spoSol.fun>costMax:
            sol.cost=costMax
        else:
            sol.cost=self.survival_From_Cost(cost,spoSol.x)
        sol.xRing_TunedParams2 = spoSol.x[:2]
        sol.xInjector_TunedParams = spoSol.x[2:]
        return sol
    def optimize(self, elementIndices, ringTuningBounds=None, injectorTuningBounds=None, tuningChoice='spacing',
                 parallel=True)->Solution:
        # optimize magnetic field of the lattice by tuning element field strengths. This is done by first evaluating the
        # system over a grid, then using a non parametric model to find the optimum.
        # elementIndices: tuple of indices of elements to tune the field strength
        # bounds: list of tuples of (min,max) for tuning
        # maxIter: maximum number of optimization iterations with non parametric optimizer
        # num0: number of points in grid of magnetic fields
        if ringTuningBounds is None:
            ringTuningBounds = [(.2, .8), (.2, .8)]
        if injectorTuningBounds is None:
            injectorTuningBounds = [(.05, .3), (.05, .3)]
        self.catch_Optimizer_Errors(ringTuningBounds, elementIndices, tuningChoice)
        self.initialize_Optimizer(elementIndices, tuningChoice, ringTuningBounds, injectorTuningBounds)
        if self.test_Lattice_Stability(ringTuningBounds, parallel=parallel) == False:
            sol = Solution()
            sol.survival = 0.0
            sol.cost=1.0
            return sol
        print('stable')
        sol=self._minimize(parallel)
        print('res',sol.survival,sol.cost)
        return sol