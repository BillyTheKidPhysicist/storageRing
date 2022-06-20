import copy
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, LineString

from KevinBumperClass import swarmShift_x
from ParticleClass import Swarm, Particle
from ParticleTracerClass import ParticleTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from SwarmTracerClass import SwarmTracer
from floorPlanCheckerFunctions import does_Fit_In_Room, plot_Floor_Plan_In_Lab
from helperTools import full_Arctan
from latticeElements.elements import HalbachLensSim, Drift, CombinerHalbachLensSim
from latticeModels import make_Ring_And_Injector

list_array_tuple = Union[np.ndarray, tuple, list]

Element = None

ELEMENTS_BUMPER = (HalbachLensSim, Drift, HalbachLensSim, Drift)
ELEMENTS_MODE_MATCHER = (Drift, HalbachLensSim, Drift, Drift, HalbachLensSim, Drift, CombinerHalbachLensSim)


def injector_Is_Expected_Design(latticeInjector, isBumperIncluded):
    expectedElements = (*ELEMENTS_BUMPER, *ELEMENTS_MODE_MATCHER) if isBumperIncluded else ELEMENTS_MODE_MATCHER
    for el, elExpectedType in zip(latticeInjector.elList, expectedElements):
        if not type(el) is elExpectedType:
            return False
    return True


def build_StorageRingModel(params, version: str, numParticlesSwarm: int = 1024, collisionDynamics: bool = False,
                           energyCorrection: bool = False, useMagnetErrors: bool = False,
                           useSolenoidField: bool = True, includeBumper: bool = False):
    """Convenience function for building a StorageRingModel"""
    options = {'useMagnetErrors': useMagnetErrors, 'useSolenoidField': useSolenoidField, 'includeBumper': includeBumper}
    PTL_Ring, PTL_Injector = make_Ring_And_Injector(params, version, options=options)
    model = StorageRingModel(PTL_Ring, PTL_Injector, energyCorrection=energyCorrection,
                             numParticlesSwarm=numParticlesSwarm, collisionDynamics=collisionDynamics,
                             isBumperIncludedInInjector=includeBumper)
    return model


class StorageRingModel:
    maximumCost = 2.0
    maximumSwarmCost = 1.0
    maximumFloorPlanCost = 1.0

    def __init__(self, latticeRing: ParticleTracerLattice, latticeInjector: ParticleTracerLattice,
                 numParticlesSwarm: int = 1024, collisionDynamics: bool = False, energyCorrection: bool = False,
                 isBumperIncludedInInjector: bool = False):
        assert latticeRing.latticeType == 'storageRing' and latticeInjector.latticeType == 'injector'
        assert injector_Is_Expected_Design(latticeInjector, isBumperIncludedInInjector)
        self.latticeRing = latticeRing
        self.latticeInjector = latticeInjector
        self.injectorLensIndices = [i for i, el in enumerate(self.latticeInjector) if type(el) is HalbachLensSim]
        self.particleTracerRing = ParticleTracer(latticeRing)
        self.particleTracerInjector = ParticleTracer(latticeInjector)
        self.swarmTracerInjector = SwarmTracer(self.latticeInjector)
        self.h = 7.5e-6  # timestep size
        self.T = 10.0
        self.swarmTracerRing = SwarmTracer(self.latticeRing)

        self.sameSeedForSwarm = True  # generate the same swarms every time by seeding the random generator during swarm
        # generation with the same number, 42

        self.minElementLength = 1.1 * self.particleTracerRing.minTimeStepsPerElement * \
                                self.latticeRing.v0Nominal * self.h
        self.injectorTunabilityLength = 2e-2  # longitudinal range of tunability for injector system, for and aft

        self.swarmInjectorInitial = None

        self.isBumperIncluded = isBumperIncludedInInjector

        self.collisionDynamics = collisionDynamics
        self.energyCorrection = energyCorrection
        self.swarmInjectorInitial = self.generate_Swarm(numParticlesSwarm)

    def generate_Swarm(self, numParticlesSwarm) -> Swarm:
        """Generate injector swarm. optionally shift the particles in the swarm for the bumper"""
        swarm = self.swarmTracerInjector.initialize_Simulated_Collector_Focus_Swarm(numParticlesSwarm)
        if self.isBumperIncluded:
            swarm = self.swarmTracerInjector.time_Step_Swarm_Distance_Along_x(swarm, swarmShift_x, holdPositionInX=True)
        return swarm

    def convert_Pos_Injector_Frame_To_Ring_Frame(self, qLabInject: np.ndarray) -> np.ndarray:
        """
        Convert particle position in injector lab frame into ring lab frame.

        :param qLabInject: particle coords in injector lab frame. 3D position vector
        :return: 3D position vector
        """
        # a nice trick
        qLabRing = self.latticeInjector.combiner.transform_Lab_Coords_Into_Element_Frame(qLabInject)
        qLabRing = self.latticeRing.combiner.transform_Element_Coords_Into_Lab_Frame(qLabRing)
        return qLabRing

    def convert_Moment_Injector_Frame_To_Ring_Frame(self, pLabInject: np.ndarray) -> np.ndarray:
        """
        Convert particle momentum in injector lab frame into ring lab frame.

        :param pLabInject: particle momentum in injector lab frame. 3D position vector
        :return: 3D momentum vector
        """
        pLabRing = self.latticeInjector.combiner.transform_Lab_Frame_Vector_Into_Element_Frame(pLabInject)
        pLabRing = self.latticeRing.combiner.transform_Element_Frame_Vector_Into_Lab_Frame(pLabRing)
        return pLabRing

    def make_Shapely_Line_In_Ring_Frame_From_Injector_Particle(self, particle: Particle) -> Optional[LineString]:
        """
        Make a shapely line object from an injector particle. If the injector particle was clipped right away
        (starting outside the vacuum for example), None is returned

        :param particle: particle that was traced through injector
        :return: None if the particle has no logged coords, or a shapely line object in ring frame
        """
        assert particle.traced
        if len(particle.elPhaseSpaceLog) <= 1:
            return None
        else:
            qList = []
            for q, _ in particle.elPhaseSpaceLog:
                qRingFrame_xy = self.convert_Pos_Injector_Frame_To_Ring_Frame(q)[:2]
                qList.append(qRingFrame_xy)
            line = LineString(qList)
            return line

    def does_Injector_Particle_Clip_On_Ring(self, particle: Particle) -> bool:
        """
        Test if particle clipped the ring. Only certain elements are considered, as of writing this only the first lens
        in the ring surrogate

        :param particle: particle that was traced through injector
        :return: True if particle clipped ring, False if it didn't
        """

        line = self.make_Shapely_Line_In_Ring_Frame_From_Injector_Particle(particle)
        if line is None:  # particle was clipped immediately, but in the injector not in the ring
            return False
        lensesBeforeCombiner = self.get_Lenses_Before_Combiner_Ring()
        assert all(type(lens) is HalbachLensSim for lens in lensesBeforeCombiner)
        return any(line.intersects(lens.SO_Outer) for lens in lensesBeforeCombiner)

    def get_Lenses_Before_Combiner_Ring(self) -> tuple[HalbachLensSim, ...]:
        """Get the lens before the combiner but after the bend in the ring. There should be only one lens"""
        lenses = []
        for i, el in enumerate(self.latticeRing.elList):
            if type(el) is HalbachLensSim:
                lenses.append(el)
            if i == self.latticeRing.combiner.index:
                break
        assert len(lenses) > 0
        return tuple(lenses)

    def get_Injector_Shapely_Objects_In_Lab_Frame(self, which: str) -> list[Polygon]:
        assert which in ('interior', 'exterior')
        shapelyObjectLabFrameList = []
        ne_Inj, ne_Ring = self.latticeInjector.combiner.ne, self.latticeRing.combiner.ne
        angleInj = full_Arctan(ne_Inj[1], ne_Inj[0])
        angleRing = full_Arctan(ne_Ring[1], ne_Ring[0])
        rotationAngle = angleRing - angleInj
        r2Injector = self.latticeInjector.combiner.r2
        r2Ring = self.latticeRing.combiner.r2
        for el in self.latticeInjector:
            SO = copy.copy(el.SO_Outer if which == 'exterior' else el.SO)
            SO = translate(SO, xoff=-r2Injector[0], yoff=-r2Injector[1])
            SO = rotate(SO, rotationAngle, use_radians=True, origin=(0, 0))
            SO = translate(SO, xoff=r2Ring[0], yoff=r2Ring[1])
            shapelyObjectLabFrameList.append(SO)
        return shapelyObjectLabFrameList

    def generate_Shapely_Object_List_Of_Floor_Plan(self, whichSide: str) -> list[Polygon]:
        assert whichSide in ('exterior', 'interior')
        shapelyObjectList = []
        shapelyObjectList.extend([el.SO_Outer if whichSide == 'exterior' else el.SO for el in self.latticeRing])
        shapelyObjectList.extend(self.get_Injector_Shapely_Objects_In_Lab_Frame(whichSide))
        return shapelyObjectList

    def floor_Plan_OverLap_mm(self) -> float:
        """Find the area overlap between the element before the second injector lens, and the lenses between combiner
        input and adjacent bender output
        """
        lensesBeforeCombiner = self.get_Lenses_Before_Combiner_Ring()
        assert all(type(lens) is HalbachLensSim for lens in lensesBeforeCombiner)
        convertAreaTo_mm = 1e3 ** 2
        area = 0.0
        for lens in lensesBeforeCombiner:
            firstLensRingShapely = lens.SO_Outer
            injectorShapelyObjects = self.get_Injector_Shapely_Objects_In_Lab_Frame('exterior')
            for i in range(self.injectorLensIndices[-1] + 1):  # don't forget to add 1
                area += firstLensRingShapely.intersection(injectorShapelyObjects[i]).area * convertAreaTo_mm
        return area

    def show_Floor_Plan(self, which: str = 'exterior', deferPltShow=False, trueAspect=True,
                        linestyle: str = '-', color: str = 'black') -> None:
        shapelyObjectList = self.generate_Shapely_Object_List_Of_Floor_Plan(which)
        for shapelyObject in shapelyObjectList: plt.plot(*shapelyObject.exterior.xy, c=color, linestyle=linestyle)
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.grid()
        if trueAspect:
            plt.gca().set_aspect('equal')
        if not deferPltShow:
            plt.show()

    def show_System_Floor_Plan_In_Lab(self):
        plot_Floor_Plan_In_Lab(self)

    def show_Floor_Plan_And_Trajectories(self, trueAspectRatio: bool = True, Tmax=1.0) -> None:
        """Trace particles through the lattices, and plot the results. Interior and exterior of element is shown"""

        self.show_Floor_Plan(deferPltShow=True, trueAspect=trueAspectRatio, color='grey')
        self.show_Floor_Plan(which='interior', deferPltShow=True, trueAspect=trueAspectRatio, linestyle=':')
        swarm = Swarm()
        swarm.particles = self.swarmInjectorInitial.particles[:100]
        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            swarm, self.h, 1.0, parallel=False,
            fastMode=False, copySwarm=True, accelerated=False, logPhaseSpaceCoords=True, energyCorrection=True,
            collisionDynamics=self.collisionDynamics)
        for particle in swarmInjectorTraced:
            particle.clipped = True if self.does_Injector_Particle_Clip_On_Ring(particle) else particle.clipped
        swarmRingInitial = self.transform_Swarm_From_Injector_Frame_To_Ring_Frame(swarmInjectorTraced,
                                                                                  copyParticles=True,
                                                                                  onlyUnclipped=False)
        swarmRingTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmRingInitial, self.h, Tmax,
                                                                           fastMode=False,
                                                                           parallel=False, energyCorrection=True,
                                                                           stepsBetweenLogging=4,
                                                                           collisionDynamics=self.collisionDynamics)

        for particleInj, particleRing in zip(swarmInjectorTraced, swarmRingTraced):
            assert not (particleInj.clipped and not particleRing.clipped)  # this wouldn't make sense
            color = 'r' if particleRing.clipped else 'g'
            inj_qarr = particleInj.qArr if len(particleInj.qArr) != 0 else np.array([particleInj.qi])
            qRingArr = np.array([self.convert_Pos_Injector_Frame_To_Ring_Frame(q) for q in inj_qarr])
            plt.plot(qRingArr[:, 0], qRingArr[:, 1], c=color, alpha=.3)
            if particleInj.clipped:  # if clipped in injector, plot last location
                plt.scatter(qRingArr[-1, 0], qRingArr[-1, 1], marker='x', zorder=100, c=color)
            if particleRing.qArr is not None and len(particleRing.qArr) > 1:  # if made to ring
                plt.plot(particleRing.qArr[:, 0], particleRing.qArr[:, 1], c=color, alpha=.3)
                if not particleInj.clipped:  # if not clipped in injector plot last ring location
                    plt.scatter(particleRing.qArr[-1, 0], particleRing.qArr[-1, 1], marker='x', zorder=100, c=color)
        plt.show()

    def mode_Match(self, floorPlanCostCutoff: float = np.inf, parallel: bool = False) -> tuple[float, float]:
        # project a swarm through the lattice. Return the average number of revolutions, or return None if an unstable
        # configuration
        assert floorPlanCostCutoff >= 0
        floorPlanCost = self.floor_Plan_Cost_With_Tunability()
        if self.floor_Plan_Cost() > floorPlanCostCutoff:
            cost = self.maximumSwarmCost + floorPlanCost
            fluxMultiplication = np.nan
        else:
            swarmTraced = self.inject_And_Trace_Swarm(parallel)
            fluxMultiplication = self.compute_Flux_Multiplication(swarmTraced)
            swarmCost = self.swarm_Cost(swarmTraced)
            cost = swarmCost + floorPlanCost
        assert 0.0 <= cost <= self.maximumCost
        return cost, fluxMultiplication

    def inject_And_Trace_Swarm(self, parallel: bool = False) -> Swarm:

        swarmInitial = self.trace_Through_Injector_And_Transform_To_Ring()
        swarmTraced = self.swarmTracerRing.trace_Swarm_Through_Lattice(swarmInitial, self.h, self.T,
                                                                       fastMode=True, accelerated=True, copySwarm=False,
                                                                       energyCorrection=self.energyCorrection,
                                                                       collisionDynamics=self.collisionDynamics,
                                                                       parallel=parallel)
        return swarmTraced

    def transform_Swarm_From_Injector_Frame_To_Ring_Frame(self, swarmInjectorTraced: Swarm,
                                                          copyParticles: bool = False,
                                                          onlyUnclipped: bool = True) -> Swarm:
        # identify particles that survived to combiner end and move them assuming the combiner's output (r2)
        # was moved to the origin

        swarmRing = Swarm()
        for particle in swarmInjectorTraced:
            clipped = particle.clipped or self.does_Injector_Particle_Clip_On_Ring(particle)
            if not onlyUnclipped or not clipped:
                qRing = self.convert_Pos_Injector_Frame_To_Ring_Frame(particle.qf)
                pRing = self.convert_Moment_Injector_Frame_To_Ring_Frame(particle.pf)
                particleRing = particle.copy() if copyParticles else particle
                particleRing.qi, particleRing.pi = qRing, pRing
                particleRing.reset()
                particleRing.clipped = clipped
                swarmRing.add(particleRing)
        return swarmRing

    def trace_Through_Injector_And_Transform_To_Ring(self) -> Swarm:

        swarmInjectorTraced = self.swarmTracerInjector.trace_Swarm_Through_Lattice(
            self.swarmInjectorInitial.copy(), self.h, 1.0, fastMode=True, copySwarm=False,
            logPhaseSpaceCoords=True, accelerated=True, collisionDynamics=self.collisionDynamics)
        swarmRingInitial = self.transform_Swarm_From_Injector_Frame_To_Ring_Frame(swarmInjectorTraced,
                                                                                  copyParticles=True)
        return swarmRingInitial

    def compute_Flux_Multiplication(self, swarmTraced: Swarm) -> float:
        """Return the multiplcation of flux expected in the ring. """

        assert all([particle.traced for particle in swarmTraced.particles])
        if swarmTraced.num_Particles() == 0:
            return 0.0
        else:
            weightedFluxMultInjectedSwarm = swarmTraced.weighted_Flux_Multiplication()
            injectionSurvivalFrac = swarmTraced.num_Particles(weighted=True) / \
                                    self.swarmInjectorInitial.num_Particles(weighted=True)
            totalFluxMult = injectionSurvivalFrac * weightedFluxMultInjectedSwarm
            return totalFluxMult

    def compute_Swarm_Flux_Mult_Percent(self, swarmTraced: Swarm) -> float:
        # What percent of the maximum flux multiplication is the swarm reaching? It's cruical I consider that not
        # all particles survived through the lattice.
        totalFluxMult = self.compute_Flux_Multiplication(swarmTraced)
        weightedFluxMultMax = self.maximum_Weighted_Flux_Multiplication()
        fluxMultPerc = 1e2 * totalFluxMult / weightedFluxMultMax
        assert 0.0 <= fluxMultPerc <= 100.0
        return fluxMultPerc

    def maximum_Weighted_Flux_Multiplication(self) -> float:
        # unrealistic maximum flux of lattice
        rBendNominal = 1.0
        LCombinerNominal = .2
        minLatticeLength = 2 * (np.pi * rBendNominal + LCombinerNominal)
        maxFluxMult = self.T * self.latticeRing.v0Nominal / minLatticeLength  # the aboslute max
        return maxFluxMult

    def floor_Plan_Cost(self) -> float:
        overlap = self.floor_Plan_OverLap_mm()  # units of mm^2
        factor = 100  # units of mm^2
        costOverlap = 2 / (1 + np.exp(-overlap / factor)) - 1
        cost = self.maximumFloorPlanCost if not does_Fit_In_Room(self) else costOverlap
        assert 0.0 <= cost <= self.maximumFloorPlanCost
        return cost

    def get_Drift_After_Second_Lens_Injector(self) -> Drift:
        """Get drift element which comes immediately after second lens in injector"""

        drift = self.latticeInjector.elList[self.injectorLensIndices[-1] + 1]
        assert type(drift) is Drift
        return drift

    def floor_Plan_Cost_With_Tunability(self) -> float:
        """Measure floor plan cost at nominal position, and at maximum spatial tuning displacement in each direction.
        Return the largest value of the three. This is used to punish the system when the injector lens is no longer
        tunable because it is so close to the ring"""

        driftAfterLens = self.get_Drift_After_Second_Lens_Injector()
        L0 = driftAfterLens.L  # value before tuning
        cost = [self.floor_Plan_Cost()]

        driftAfterLens.set_Length(L0 + -self.injectorTunabilityLength)  # move lens away from combiner
        self.latticeInjector.build_Lattice(False, buildFieldHelper=False) #don't waste time building field helpers
        cost.append(self.floor_Plan_Cost())

        driftAfterLens.set_Length(L0)  # reset
        self.latticeInjector.build_Lattice(False,buildFieldHelper=False) #don't waste time building field helpers,
            #previous helpers are still saved
        floorPlanCost = max(cost)
        assert 0.0 <= floorPlanCost <= 1.0
        return floorPlanCost

    def swarm_Cost(self, swarm: Swarm) -> float:
        """Cost associated with a swarm after being traced through system"""
        fluxMultPerc = self.compute_Swarm_Flux_Mult_Percent(swarm)
        swarmCost = (100.0 - fluxMultPerc) / 100.0
        assert 0.0 <= swarmCost <= self.maximumSwarmCost
        return swarmCost
