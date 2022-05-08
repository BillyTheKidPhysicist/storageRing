import random
import time
import skopt
from shapely.affinity import rotate, translate
import copy

import elementPT
from ParticleTracerClass import ParticleTracer
import numpy as np
from ParticleClass import Swarm, Particle
import scipy.optimize as spo
import matplotlib.pyplot as plt
from SwarmTracerClass import SwarmTracer
from elementPT import HalbachLensSim, LensIdeal, Drift
import multiprocess as mp
from collections.abc import Iterable
from ParticleTracerLatticeClass import ParticleTracerLattice
from typing import Union,Optional
from shapely.geometry import Polygon
list_array_tuple=Union[np.ndarray,tuple,list]

class Solution:
    # class to hold onto results of each solution

    def __init__(self):
        self.params = None  # paramters tuned by the 'outer' gp minimize
        self.fluxMultiplication=None
        self.cost=None
        self.swarmCost=None
        self.floorPlanCost=None
        self.scipyMessage=None
        self.stable=None
        self.invalidInjector=None
        self.invalidRing=None
        self.description = None
        self.bumpParams = None  # List of parameters used in the misalignment testing

    def __str__(self):  # method that gets called when you do print(Solution())
        string = '----------Solution-----------   \n'
        string += 'parameters: ' + repr(self.params) + '\n'
        # string+='stable configuration:'+str(self.stable)+'\n'
        # string += 'bump params: ' + str(self.bumpParams) + '\n'
        string+='cost: '+str(self.cost)+'\n'
        string += 'flux multiplication: ' + str(self.fluxMultiplication) + '\n'
        string += 'scipy message: '+ str(self.scipyMessage)+'\n'
        string += '----------------------------'
        return string


class LatticeOptimizer:
    
    def __init__(self, latticeRing: ParticleTracerLattice, latticeInjector: ParticleTracerLattice):
        assert latticeRing.latticeType=='storageRing' and latticeInjector.latticeType=='injector'
        self.latticeRing = latticeRing
        self.latticeInjector = latticeInjector
        self.particleTracerRing = ParticleTracer(latticeRing)
        self.particleTracerInjector = ParticleTracer(latticeInjector)
        self.swarmTracerInjector = SwarmTracer(self.latticeInjector)
        self.whichKnobs=None #injector or lattice and injector knobs may be tuned
        self.h = 7.5e-6  # timestep size
        self.T = 10.0
        self.injectTuneElIndices=(0,2)
        self.swarmTracerRing = SwarmTracer(self.latticeRing)
        self.phaseSpaceFunc = None  # function that returns the number of revolutions of a particle at a given
        # point in 5d phase space (y,z,px,py,pz). Linear interpolation is done between points
        self.tunedElementList = None
        self.tuningChoice = None  # what type of tuning will occur
        self.useLatticeUpperSymmetry = False  # exploit the fact that the lattice has symmetry in the z axis to use half
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
        self.tuningBounds = None
        self.numParticlesFullSwarm=1000
        self.numParticlesSurrogate=50
        self.swarmInjectorInitial=None
        self.swarmInjectorInitial_Surrogate=None
        self.generate_Swarms()

    def generate_Swarms(self)-> None:
        self.swarmInjectorInitial = self.swarmTracerInjector.initialize_Observed_Collector_Swarm_Probability_Weighted(
            self.spotCaptureDiam, self.collectorAngleMax, self.numParticlesFullSwarm, temperature=self.temperature,
            sameSeed=self.sameSeedForSwarm, upperSymmetry=self.useLatticeUpperSymmetry,gammaSpace=self.gamma_Space,
            probabilityMin=.05)
        self.swarmInjectorInitial_Surrogate=Swarm()
        self.swarmInjectorInitial_Surrogate.particles=self.swarmInjectorInitial.particles[:self.numParticlesSurrogate]
        
    def get_Injector_Shapely_Objects_In_Lab_Frame(self)-> list[Polygon]:
        shapelyObjectLabFrameList = []
        rotationAngle = self.latticeInjector.combiner.ang + -self.latticeRing.combiner.ang
        r2Injector = self.latticeInjector.combiner.r2
        r2Ring = self.latticeRing.combiner.r2
        for el in self.latticeInjector:
            SO = copy.copy(el.SO_Outer)
            SO = translate(SO, xoff=-r2Injector[0], yoff=-r2Injector[1])
            SO = rotate(SO, rotationAngle, use_radians=True, origin=(0, 0))
            SO = translate(SO, xoff=r2Ring[0], yoff=r2Ring[1])
            shapelyObjectLabFrameList.append(SO)
        return shapelyObjectLabFrameList

    def generate_Shapely_Object_List_Of_Floor_Plan(self)-> list[Polygon]:
        shapelyObjectList = []
        shapelyObjectList.extend([el.SO_Outer for el in self.latticeRing])
        shapelyObjectList.extend(self.get_Injector_Shapely_Objects_In_Lab_Frame())
        return shapelyObjectList

    def floor_Plan_OverLap_mm(self)-> float:
        injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame()
        assert len(injectorShapelyObjects)==6
        overlapElIndex = 3
        injectorLensShapely = injectorShapelyObjects[overlapElIndex]
        assert isinstance(self.latticeInjector.elList[overlapElIndex], HalbachLensSim)
        ringShapelyObjects = [el.SO_Outer for el in self.latticeRing]
        area = 0
        converTo_mm = (1e3) ** 2
        for element in ringShapelyObjects:
            area += element.intersection(injectorLensShapely).area * converTo_mm
        return area

    def show_Floor_Plan(self,X: Optional[list_array_tuple])-> None:
        self.update_Ring_And_Injector(X)
        shapelyObjectList = self.generate_Shapely_Object_List_Of_Floor_Plan()
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy,c='black')
        plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        plt.show()

    def mode_Match_Cost(self, X: Optional[list_array_tuple],useSurrogate: bool,energyCorrection: bool,
                rejectUnstable: bool=True,rejectIllegalFloorPlan: bool=True,
                returnFullResults: bool=False)-> Union[float,tuple[Optional[float],Optional[float],Optional[Swarm]]]:
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        swarmCost,floorPlanCost,swarmTraced=None,None,None
        if X is not None:
            for val, bounds in zip(X, self.tuningBounds):
                assert bounds[0] <= val <= bounds[1]
        if rejectIllegalFloorPlan==True and self.floor_Plan_Cost(X)>0.0:
            floorPlanCost=self.floor_Plan_Cost(X)
            cost= 1.0+floorPlanCost
        elif rejectUnstable==True and self.is_Stable(X) == False:
            cost=2.0
        else:
            swarmTraced=self.inject_And_Trace_Swarm(X,useSurrogate,energyCorrection)
            swarmCost = self.swarm_Cost(swarmTraced)
            floorPlanCost = self.floor_Plan_Cost(X)
            cost = swarmCost + floorPlanCost
        assert 0.0 <= cost <= 2.0
        if returnFullResults: return swarmCost,floorPlanCost,swarmTraced
        else: return cost
        
    def inject_And_Trace_Swarm(self,X: list_array_tuple,useSurrogate: bool,energyCorrection: bool)-> Swarm:
        self.update_Ring_And_Injector(X)
        swarmInitial = self.trace_And_Project_Injector_Swarm_To_Combiner_End(useSurrogate)
        swarmTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmInitial, self.h, self.T,
                                    fastMode=True, accelerated=True, copySwarm=False,energyCorrection=energyCorrection)
        return swarmTraced
    
    def move_Survived_Particles_In_Injector_Swarm_To_Origin(self, swarmInjectorTraced: Swarm, 
                                                            copyParticles: bool=False,clipNextAp: bool=True)-> Swarm:
        #fidentify particles that survived to combiner end, walk them right up to the end, exclude any particles that
        #are now clipping the combiner and any that would clip the next element
        #NOTE: The particles offset is taken from the origin of the orbit output of the combiner, not the 0,0 output
        apNextElement = self.latticeRing.elList[self.latticeRing.combinerIndex + 1].ap if clipNextAp else np.inf
        swarmSurvived = Swarm()
        for particle in swarmInjectorTraced:
            if particle.qf is not None:
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

    def trace_And_Project_Injector_Swarm_To_Combiner_End(self,useSurrogate: bool) -> Swarm:
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

    def is_Stable(self, X: list_array_tuple,minRevsToTest=5.0)-> bool:
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

    def update_Ring_And_Injector(self,X: Optional[list_array_tuple]):
        if X is None:
            pass
        elif self.whichKnobs=='all':
            assert len(X) == 4
            XRing,XInjector=X[:2],X[2:]
            self.update_Ring_Lattice(XRing)
            self.update_Injector_Lattice(XInjector)
        elif self.whichKnobs=='ring':
            assert len(X) == 2
            self.update_Ring_Lattice(X)
        else: raise ValueError
        
    def update_Injector_Lattice(self, X: list_array_tuple):
        # modify lengths of drift regions in injector
        assert len(X)==2
        assert X[0]>0.0 and X[1]>0.0
        self.latticeInjector.elList[self.injectTuneElIndices[0]].set_Length(X[0])
        self.latticeInjector.elList[self.injectTuneElIndices[1]].set_Length(X[1])
        self.latticeInjector.build_Lattice()

    def update_Ring_Lattice(self, X: list_array_tuple)-> None:
        assert len(X)==2
        if self.tuningChoice == 'field':
            self.update_Ring_Field_Values(X)
        elif self.tuningChoice == 'spacing':
            self.update_Ring_Spacing(X)
        else: raise ValueError

    def update_Ring_Field_Values(self, X: list_array_tuple)-> None:
        for el, arg in zip(self.tunedElementList, X):
            el.fieldFact = arg

    def update_Ring_Spacing(self, X: list_array_tuple)-> None:
        for elCenter, spaceFracElBefore in zip(self.tunedElementList, X):
            self.move_Element_Longitudinally(elCenter,spaceFracElBefore)
        self.latticeRing.build_Lattice()

    def move_Element_Longitudinally(self,elCenter: elementPT.Element,spaceFracElBefore: float)-> None:
        assert 0 <= spaceFracElBefore <= 1.0
        elBefore, elAfter = self.latticeRing.get_Element_Before_And_After(elCenter)
        assert isinstance(elBefore, Drift) and isinstance(elAfter, Drift)
        totalBorderingElLength=elBefore.L + elAfter.L
        tunableLength = (elBefore.L + elAfter.L) - 2 * self.minElementLength
        LBefore = spaceFracElBefore * tunableLength + self.minElementLength
        LAfter = totalBorderingElLength - LBefore
        elBefore.set_Length(LBefore)
        elAfter.set_Length(LAfter)

    def compute_Flux_Multiplication(self,swarmTraced:Swarm)->float:
        """Return the multiplcation of flux expected in the ring. """

        assert all([particle.traced == True for particle in swarmTraced.particles])
        if swarmTraced.num_Particles()==0: return 0.0
        weightedFluxMultInjectedSwarm=swarmTraced.weighted_Flux_Multiplication()
        injectionSurvivalFrac=swarmTraced.num_Particles(weighted=True)/\
                              self.swarmInjectorInitial.num_Particles(weighted=True) #particles may be lost
        totalFluxMult=injectionSurvivalFrac*weightedFluxMultInjectedSwarm
        return totalFluxMult


    def compute_Swarm_Flux_Mult_Percent(self, swarmTraced:Swarm)-> float:
        # What percent of the maximum flux multiplication is the swarm reaching? It's cruical I consider that not
        #all particles survived through the lattice.
        totalFluxMult=self.compute_Flux_Multiplication(swarmTraced)
        weightedFluxMultMax = self.maximum_Weighted_Flux_Multiplication()
        fluxMultPerc=1e2*totalFluxMult/weightedFluxMultMax
        assert 0.0 <= fluxMultPerc <= 100.0
        return fluxMultPerc
    
    def maximum_Weighted_Flux_Multiplication(self)-> float:
        #unrealistic maximum flux of lattice
        rBendNominal=1.0
        LCombinerNominal=.2
        minLatticeLength=2*(np.pi*rBendNominal+LCombinerNominal)
        maxFluxMult=self.T * self.latticeRing.v0Nominal / minLatticeLength #the aboslute max
        return maxFluxMult
    
    def floor_Plan_Cost(self,X: list_array_tuple)-> float:
        self.update_Ring_And_Injector(X)
        overlap=self.floor_Plan_OverLap_mm() #units of mm^2
        factor = 300 #units of mm^2
        cost = 2 / (1 + np.exp(-overlap / factor)) - 1
        assert 0.0<=cost<=1.0
        return cost
    
    def swarm_Cost(self, swarm: Swarm)-> float:
        fluxMultPerc=self.compute_Swarm_Flux_Mult_Percent(swarm)
        swarmCost=(100.0-fluxMultPerc)/100.0
        assert 0.0<=swarmCost<=1.0
        return swarmCost
    
    def flux_Mult_Percent_From_Cost(self, cost,X)-> float:
        fluxCost=cost-self.floor_Plan_Cost(X)
        fluxMulPerc=100.0*(1.0-fluxCost)
        assert 0.0<=fluxMulPerc<=100.0
        return fluxMulPerc

    def catch_Optimizer_Errors(self, tuningBounds: list_array_tuple, tuningElementIndices: list_array_tuple,
                               tuningChoice: str,whichKnobs: str)-> None:
        if max(tuningElementIndices) >= len(self.latticeRing.elList) - 1: raise Exception("element indices out of bounds")
        if len(tuningBounds) != len(tuningElementIndices): raise Exception("Bounds do not match number of tuned elements")
        combinerRing,combinerLat=self.latticeRing.combiner,self.latticeInjector.combiner
        if not (combinerRing.Lm==combinerLat.Lm and combinerRing.ap==combinerLat.ap and combinerRing.outputOffset==
                combinerLat.outputOffset):
            raise Exception('Combiners are different between the two lattices')
        injectorTuningElements=[self.latticeInjector.elList[index] for index in self.injectTuneElIndices]
        if not all(isinstance(el,Drift) for el in injectorTuningElements):
            raise Exception("injector tuning elements must be drift region")
        if tuningChoice == 'field':
            for elIndex in tuningElementIndices:
                el=self.latticeRing.elList[elIndex]
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
        if whichKnobs not in ('all','ring'):
            raise Exception('Knobs must be either \'all\' (full system) or \'ring\' (only storage ring)')


    def initialize_Optimizer(self, tuningElementIndices: list_array_tuple, tuningChoice: str,whichKnobs: str,
                                ringTuningBounds: list_array_tuple, injectorTuningBounds: list_array_tuple)-> None:
        assert tuningChoice in ('spacing','field') and whichKnobs in ('all','ring')
        assert all(isinstance(arg,Iterable) for arg in (tuningElementIndices,ringTuningBounds,injectorTuningBounds))
        self.whichKnobs=whichKnobs
        self.tuningBounds = ringTuningBounds.copy()
        if self.whichKnobs =='all':
            self.tuningBounds.extend(injectorTuningBounds)
        self.tunedElementList = [self.latticeRing.elList[index] for index in tuningElementIndices]
        self.tuningChoice = tuningChoice
        if self.sameSeedForSearch == True:
            np.random.seed(42)

    def test_Lattice_Stability(self, ringTuningBounds: list_array_tuple,injectorTuningBounds: list_array_tuple,
                               numEdgePoints: int=30, parallel: bool=False)-> bool:
        assert len(ringTuningBounds) == 2 and len(injectorTuningBounds)==2
        ringKnob1Arr = np.linspace(ringTuningBounds[0][0], ringTuningBounds[0][1], numEdgePoints)
        ringKnob2Arr = np.linspace(ringTuningBounds[1][0], ringTuningBounds[1][1], numEdgePoints)
        injectorKnob1Arr_Constant = ringTuningBounds[0][1]*np.ones(numEdgePoints**2)
        injectorKnobA2rr_Constant = ringTuningBounds[1][1]*np.ones(numEdgePoints**2)
        testCoords = np.asarray(np.meshgrid(ringKnob1Arr, ringKnob2Arr)).T.reshape(-1, 2)
        if self.whichKnobs=='all':
            testCoords=np.column_stack((testCoords,injectorKnob1Arr_Constant,injectorKnobA2rr_Constant))
        if parallel == False:
            stabilityList = [self.is_Stable(coords) for coords in testCoords]
        else:
            with mp.Pool(mp.cpu_count()) as pool:
                stabilityList=pool.map(self.is_Stable,testCoords)
        assert len(stabilityList)==numEdgePoints**2
        if sum(stabilityList) == 0:
            return False
        else:
            return True

    def _fast_Minimize(self):
        #less accurate method that minimizes with a smaller surrogate swarm then uses the full swarm for the final
        #value
        useSurrogate,energyCorrection=[True,False]
        sol_Surrogate = spo.differential_evolution(self.mode_Match_Cost, self.tuningBounds, tol=self.tolerance,
                                                   polish=False,args=(useSurrogate,energyCorrection),
                                                   maxiter=self.maxEvals//(self.optimalPopSize*len(self.tuningBounds)),
                                                   popsize=self.optimalPopSize,init='halton')
        return sol_Surrogate

    def _accurate_Minimize(self):
        #start first by quickly randomly searching with a surrogate swarm.
        randomSamplePoints=128
        energyCorrection=False #leave this off here, apply later once
        useSurrogateRoughPass=True
        samples = skopt.sampler.Sobol().generate(self.tuningBounds, randomSamplePoints)
        vals=[self.mode_Match_Cost(sample,useSurrogateRoughPass,energyCorrection) for sample in samples]
        XInitial=samples[np.argmin(vals)]
        useSurrogateScipyOptimer=False
        sol=spo.differential_evolution(self.mode_Match_Cost,self.tuningBounds,polish=False,x0=XInitial,tol=self.tolerance,
                                       maxiter=self.maxEvals//(self.optimalPopSize*len(self.tuningBounds)),
                                       args=(useSurrogateScipyOptimer,energyCorrection),popsize=self.optimalPopSize,init='halton')
        return sol

    def _minimize(self)->Solution:
        if self.fastSolver==True:
            scipySol=self._fast_Minimize()
        else:
            scipySol=self._accurate_Minimize()
        useSurrogate,energyCorrection=[False,True]
        cost_Most_Accurate = self.mode_Match_Cost(scipySol.x, useSurrogate,energyCorrection)
        sol = Solution()
        sol.scipyMessage=scipySol.message
        sol.cost = cost_Most_Accurate
        optimalConfig = scipySol.x
        sol.stable = True
        sol.xRing_TunedParams2 = optimalConfig[:2]
        if self.whichKnobs=='all':
            sol.xInjector_TunedParams = optimalConfig[2:]
        sol.invalidInjector = False
        sol.invalidRing = False
        return sol

    def optimize(self, tuningElementIndices, ringTuningBounds=None, injectorTuningBounds=None, tuningChoice='spacing'
                 ,whichKnobs='all',parallel=False,fastSolver=False)->Solution:
        self.fastSolver=fastSolver
        if ringTuningBounds is None:
            ringTuningBounds = [(.2, .8), (.2, .8)]
        if injectorTuningBounds is None:
            injectorTuningBounds = [(.1, .4), (.1, .4)]
        self.catch_Optimizer_Errors(ringTuningBounds, tuningElementIndices, tuningChoice,whichKnobs)
        self.initialize_Optimizer(tuningElementIndices, tuningChoice,whichKnobs, ringTuningBounds, injectorTuningBounds)
        if self.test_Lattice_Stability(ringTuningBounds,injectorTuningBounds, parallel=parallel) == False:
            sol = Solution()
            sol.fluxMultiplication = 0.0
            sol.cost=1.0
            sol.stable=False
            return sol
        sol=self._minimize()
        return sol