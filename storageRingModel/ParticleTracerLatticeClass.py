# from geneticLensElement_Wrapper import GeneticLens
from typing import Iterable, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi

from constants import DEFAULT_ATOM_SPEED
from latticeElements.elements import BenderIdeal, HalbachBenderSimSegmented, LensIdeal, CombinerIdeal, \
    CombinerSim, CombinerHalbachLensSim, HalbachLensSim, Drift, ELEMENT_PLOT_COLORS
from latticeElements.elements import Element
from shapelyObjectBuilder import build_Shapely_Objects
from storageRingConstraintSolver import is_Particle_Tracer_Lattice_Closed
from storageRingConstraintSolver import solve_Floor_Plan, update_And_Place_Elements_From_Floor_Plan

# todo: There is a ridiculous naming convention here with r0 r1 and r2. If I ever hope for this to be helpful to other
# people, I need to change that. This was before my cleaner code approach


benderTypes = Union[BenderIdeal, HalbachBenderSimSegmented]


class ParticleTracerLattice:

    def __init__(self, speed_nominal: float = DEFAULT_ATOM_SPEED, lattice_type: str = 'storageRing',
                 jitter_amp: float = 0.0, fieldDensityMultiplier: float = 1.0, use_mag_errors: bool = False,
                 use_solenoid_field: bool = False, initialLocation: tuple[float, float] = None, initialAngle=None,
                 magnet_grade: str = 'N52', use_standard_mag_size: bool=False, use_standard_tube_OD: bool =False):
        assert fieldDensityMultiplier > 0.0
        if lattice_type != 'storageRing' and lattice_type != 'injector':
            raise Exception('invalid lattice type provided')
        if jitter_amp > 5e-3:
            raise Exception("Jitter values greater than 5 mm may begin to have unexpected results. Several parameters"
                            "depend on this value, and relatively large values were not planned for")
        self.lattice_type = lattice_type  # options are 'storageRing' or 'injector'. If storageRing, the geometry is the the first element's
        # input at the origin and succeeding elements in a counterclockwise fashion. If injector, then first element's input
        # is also at the origin, but seceeding elements follow along the positive x axis
        self.speed_nominal = speed_nominal  # Design particle speed
        self.benderIndices: list[int] = []  # list that holds index values of benders. First bender is the
        # first one that the particle sees
        # if it started from beginning of the lattice. Remember that lattice cannot begin with a bender
        self.initialLocation = (0.0, 0.0) if initialLocation is None else initialLocation
        self.initialAngle = -np.pi if initialAngle is None else initialAngle
        self.combinerIndex: Optional[int] = None  # the index in the lattice where the combiner is
        self.totalLength: Optional[float] = None  # total length of lattice, m
        self.jitter_amp = jitter_amp
        self.fieldDensityMultiplier = fieldDensityMultiplier
        self.use_mag_errors = use_mag_errors
        self.use_standard_tube_OD=use_standard_tube_OD
        self.use_standard_mag_size=use_standard_mag_size

        self.combiner: Optional[Element] = None  # combiner element object
        self.linearElementsToConstraint: list[HalbachLensSim] = []  # elements whos length will be changed when the
        # lattice is constrained to satisfy geometry. Must be inside bending region

        self.isClosed = None  # is the lattice closed, ie end and beginning are smoothly connected?
        self.magnet_grade = magnet_grade
        self.use_solenoid_field = use_solenoid_field

        self.elList: list[Element] = []  # to hold all the lattice elements

    def __iter__(self) -> Iterable[Element]:
        return (element for element in self.elList)

    def _find_rOptimal(self, output_offsetFactArr: np.ndarray, errorArr: np.ndarray) -> Optional[float]:
        test = errorArr.copy()[1:]
        test = np.append(test, errorArr[0])
        numValidSolutions = np.sum(~np.isnan(errorArr))
        numNanInitial = np.sum(np.isnan(errorArr))
        numNanAfter = np.sum(np.isnan(test + errorArr))
        valid = True
        if numNanAfter - numNanInitial > 1:
            valid = False
        elif numValidSolutions < 4:
            valid = False
        elif numNanInitial > 0:
            if (np.isnan(errorArr[0]) == False and np.isnan(errorArr[-1]) == False):
                valid = False
        if valid == False:
            return None
        # trim out invalid points
        output_offsetFactArr = output_offsetFactArr[~np.isnan(errorArr)]
        errorArr = errorArr[~np.isnan(errorArr)]
        fit = spi.RBFInterpolator(output_offsetFactArr[:, np.newaxis], errorArr)
        output_offsetFactArrDense = np.linspace(output_offsetFactArr[0], output_offsetFactArr[-1], 10_000)
        errorArrDense = fit(output_offsetFactArrDense[:, np.newaxis])
        rOptimal = output_offsetFactArrDense[np.argmin(errorArrDense)]
        rMinDistFromEdge = np.min(output_offsetFactArr[1:] - output_offsetFactArr[:-1]) / 4
        if rOptimal > output_offsetFactArr[-1] - rMinDistFromEdge or rOptimal < output_offsetFactArr[
            0] + rMinDistFromEdge:
            # print('Invalid solution, rMin very near edge. ')
            return None
        return rOptimal

    def set_Constrained_Linear_Element(self, el: Element) -> None:
        if len(self.linearElementsToConstraint) > 1: raise Exception("there can only be 2 constrained linear elements")
        self.linearElementsToConstraint.append(el)

    def add_Combiner_Sim(self, sizeScale: float = 1.0) -> None:
        """
        Add model of our combiner from COMSOL. rarely used

        :param sizeScale: How much to scale up or down dimensions of combiner
        :return: None
        """

        file = 'combinerV3.txt'
        el = CombinerSim(self, file, self.lattice_type, sizeScale=sizeScale)
        el.index = len(self.elList)  # where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner = el
        self.combinerIndex = el.index
        self.elList.append(el)  # add element to the list holding lattice elements in order

    def add_Combiner_Sim_Lens(self, Lm: float, rp: float, load_beam_offset: float = 5e-3, layers: int = 1,
                              ap: float = None,
                              seed: int = None) -> None:

        """
        Add halbach hexapole lens combiner element.

        The edge of a hexapole lens is used to deflect high and weak field seeking states. Transvers dimension of
        magnets are the maximum that can be used to for a halbach sextupole of given radius.

        :param Lm: Hard edge length of magnet, m. Total length of element depends on degree of deflection of nominal
        trajectory
        :param rp: Bore radius of hexapole lens, m
        :param loadBeamOffset: Maximum desired acceptance diameter of load beam, m. Circulating beam is not specified
        :param layers: Number of concentric layers of magnets
        :return: None
        """

        el = CombinerHalbachLensSim(self, Lm, rp, load_beam_offset, layers, ap, seed)
        el.index = len(self.elList)  # where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner = el
        self.combinerIndex = el.index
        self.elList.append(el)  # add element to the list holding lattice elements in order

    def add_Halbach_Lens_Sim(self, rp: Union[float, tuple], L: Optional[float], ap: Optional[float] = None,
                             constrain: bool = False, magnetWidth: Union[float, tuple] = None) -> None:
        """
        Add simulated halbach sextupole element to lattice.

        Combinations of rp and magnetWidth specify how to handle multiple layers, according to:

        rp    | magnetWidth | Explanation
        float | None        | Single layer with radius rp and magnet widths maximum possible
        float | float       | Single layer with radius rp and magnet widths of magnetWidth
        tuple | tuple       | Number of layers is len(rp). Each layer has radius corresponding value in rp, such that
                            | rp[0] is radius of first layer. Same logic for magnet widths. rp and magnetWidth must
                            | be same length.

        Configuration must be realistic.

        :param rp: Bore radius, m.
        :param L: Length of element, m. This includes fringe fields, actual magnet length will be smaller
        :param ap: Size of aperture
        :param constrain: Wether element is being used as part of a constraint. If so, fields construction will be
            deferred
        :param magnetWidth: Width of cuboid magnets in polar plane of lens, m. Magnets length is L minus
            fringe fields.
        :return: None
        """
        rp_layers = rp if isinstance(rp, tuple) else (rp,)
        magnetWidth = (magnetWidth,) if isinstance(magnetWidth, float) else magnetWidth
        el = HalbachLensSim(self, rp_layers, L, ap, magnetWidth)
        el.index = len(self.elList)  # where the element is in the lattice
        self.elList.append(el)  # add element to the list holding lattice elements in order
        if constrain == True: self.set_Constrained_Linear_Element(el)

    # def add_Genetic_lens(self,lens: GeneticLens,ap: float)-> None:
    #     """
    #     Add genetic lens used for minimizing focus size. This is part of an idea to make a low aberration lens
    #
    #     :param lens: GeneticLens object that returns field values. This sextupole lens can be shimmed, and have bizarre
    #     :param ap: Aperture of genetic lens, m
    #     :return: None
    #     """
    #     el=geneticLens(self,lens,ap)
    #     el.index = len(self.elList) #where the element is in the lattice
    #     self.elList.append(el) #add element to the list holding lattice elements in order

    def add_Lens_Ideal(self, L: float, Bp: float, rp: float, constrain: bool = False, ap: float = None) -> None:
        """
        Simple model of an ideal lens. Field norm goes as B0=Bp*r^2/rp^2

        :param L: Length of element, m. Lens hard edge length is this as well
        :param Bp: Field at bore/pole radius of lens
        :param rp: Bore/pole radius of lens
        :param ap: aperture of vacuum tube in magnet
        :param constrain:
        :return:
        """

        el = LensIdeal(self, L, Bp, rp, ap)  # create a lens element object
        el.index = len(self.elList)  # where the element is in the lattice
        self.elList.append(el)  # add element to the list holding lattice elements in order
        if constrain:
            self.set_Constrained_Linear_Element(el)
            print('not fully supported feature')

    def add_Drift(self, L: float, ap: float = .03, inputTiltAngle: float = 0.0, outputTiltAngle: float = 0.0,
                  outer_half_width: float = None) -> None:
        """
        Add drift region. This is simply a vacuum tube.

        The general shape is a trapezoid in the xy lab/element frame, and a circle in the zx,zy element frame. In the
        element frame in the xy plane the two bases are parallel with \vec{x}, and the input output can be at saome
        angle relative to \vec{y}. Positive angles are counterclockwise notation. The length of the drift region is the
        same no matter the input/output tilt because the tilt is pinned at the centerline of the two bases of the
        trapezoid.

        :param L: Length of drift region, m
        :param ap: Aperture of drift region, m
        :param inputTiltAngle: Tilt angle of the input plane to the drift region.
        :param outputTiltAngle: Tilt angle of the output to the drift region.
        :param outer_half_width: Outer half width of drift region. For example, a valve.
        :return:
        """

        el = Drift(self, L, ap, outer_half_width, inputTiltAngle, outputTiltAngle)  # create a drift element object
        el.index = len(self.elList)  # where the element is in the lattice
        self.elList.append(el)  # add element to the list holding lattice elements in order

    def add_Halbach_Bender_Sim_Segmented(self, Lm: float, rp: float, numMagnets: Optional[int], rb: float,
                                         rOffsetFact: float = 1.0, ap: float = None) -> None:
        # Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        # Lcap: Length of element on the end/input of bender
        # output_offsetFact: factor to multply the theoretical offset by to minimize oscillations in the bending segment.
        # modeling shows that ~.675 is ideal
        el = HalbachBenderSimSegmented(self, Lm, rp, numMagnets, rb, ap, rOffsetFact)
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)

    def add_Bender_Ideal(self, ang: float, Bp: float, rb: float, rp: float, ap: float = None) -> None:
        # Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        # ang: Bending angle of bender, radians
        # rb: nominal bending radius of element's centerline. Actual radius is larger because particle 'rides' a little
        # outside this, m
        # Bp: field strength at pole face of lens, T
        # rp: bore radius of element, m
        # ap: size of apeture. If none then a fraction of the bore radius. Can't be bigger than bore radius, unitless

        el = BenderIdeal(self, ang, Bp, rp, rb, ap)  # create a bender element object
        el.index = len(self.elList)  # where the element is in the lattice
        self.benderIndices.append(el.index)
        self.elList.append(el)  # add element to the list holding lattice elements in order

    def add_Combiner_Ideal(self, Lm: float = .2, c1: float = 1, c2: float = 20, ap: float = .015,
                           sizeScale: float = 1.0) -> None:
        # Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        # add combiner (stern gerlacht) element to lattice
        # La: input length of combiner. The bent portion outside of combiner
        # Lm:  hard edge length of the magnet, which is the same as the vacuum tube
        # ang: angle that particle enters the combiner at
        # offset: particle enters inner section with some offset
        # c1: dipole component of combiner
        # c2: quadrupole component of bender
        # check to see if inlet length is too short. The minimum length is a function of apeture and angle
        # minLa=ap*np.sin(ang)
        # if La<minLa:
        #    raise Exception('INLET LENGTH IS SHORTER THAN MINIMUM')

        el = CombinerIdeal(self, Lm, c1, c2, ap, ap, ap / 2, sizeScale)  # create a combiner element object
        el.index = len(self.elList)  # where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner = el
        self.combinerIndex = el.index
        self.elList.append(el)  # add element to the list holding lattice elements in order

    def build_lattice(self, constrain: bool, build_field_helpers: bool = True):
        """Build the specified lattice. This includes:
        - Fill pre constrained parameters derive from simple inputs of length, field strength etc of each element.
        - Solve the floor plan layout. If constrained, solve for bumber of magnets and lengths of bending segment and
            lenses to find a valid configuration. 
        - Use the floor plan layout to update and place elementPT elements in the lab frame.
        - Use the results from the previous step to finish filling values of the element
        - Build shapely object for elementPT
        """

        for el in self.elList:
            el.fill_pre_constrained_parameters()

        floorPlan = solve_Floor_Plan(self, constrain)
        update_And_Place_Elements_From_Floor_Plan(self, floorPlan)
        for el in self.elList:
            el.fill_post_constrained_parameters()
            if build_field_helpers:
                el.build_fast_field_helper()

        self.isClosed = is_Particle_Tracer_Lattice_Closed(self)  # lattice may not have been constrained, but could
        # still be closed
        if self.lattice_type == 'storageRing' and constrain:  # double check
            assert is_Particle_Tracer_Lattice_Closed(self)
        build_Shapely_Objects(self.elList)
        self.totalLength = 0
        for el in self.elList:  # total length of particle's orbit in an element
            self.totalLength += el.Lo

    def end_Lattice(self, constrain: bool = False, buildLattice: bool = True) -> None:
        # for element in self.elList:
        #     element.build()
        self.catch_Errors(constrain)
        if buildLattice:
            self.build_lattice(constrain)

    def catch_Errors(self, constrain: bool) -> None:
        # catch any preliminary errors. Alot of error handling happens in other methods. This is a catch all for other
        # kinds. This class is not meant to have tons of error handling, so user must be cautious
        if isinstance(self.elList[0], BenderIdeal):  # first element can't be a bending element
            raise Exception('FIRST ELEMENT CANT BE A BENDER')
        if isinstance(self.elList[0], CombinerIdeal):  # first element can't be a combiner element
            raise Exception('FIRST ELEMENT CANT BE A COMBINER')
        if len(self.benderIndices) >= 2:  # if there are two benders they must be the same.
            bender1 = self.elList[self.benderIndices[0]]
            for i in self.benderIndices:
                if not type(bender1) is type(self.elList[i]):
                    raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
        if constrain:
            if self.lattice_type != 'storageRing':
                raise Exception('Constrained lattice must be storage ring type')
            if not len(self.benderIndices) >= 2:
                raise Exception('THERE MUST BE AT LEAST TWO BENDERS')
            for i in self.benderIndices:
                bender1, benderi = self.elList[self.benderIndices[0]], self.elList[i]
                if not type(bender1) is type(benderi):
                    raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
                if not bender1.Lm == benderi.Lm or bender1.magnetWidth != benderi.magnetWidth:
                    raise Exception('SEGMENT LENGTHS AND MAGNET WIDTHS MUST BE EQUAL BETWEEN BENDERS')
            if self.combiner is None:
                raise Exception('COMBINER MUST BE PRESENT')

    def get_Element_Before_And_After(self, elCenter: Element) -> tuple[Element, Element]:
        if (elCenter.index == len(self.elList) - 1 or elCenter.index == 0) and self.lattice_type == 'injector':
            raise Exception('Element cannot be first or last if lattice is injector type')
        elBeforeIndex = elCenter.index - 1 if elCenter.index != 0 else len(self.elList) - 1
        elAfterIndex = elCenter.index + 1 if elCenter.index < len(self.elList) - 1 else 0
        elBefore = self.elList[elBeforeIndex]
        elAfter = self.elList[elAfterIndex]
        return elBefore, elAfter

    def get_Lab_Coords_From_Orbit_Distance(self, xPos: np.ndarray) -> tuple[float, float]:
        # xPos: distance along ideal orbit
        assert xPos >= 0.0
        xPos = xPos % self.totalLength  # xpos without multiple revolutions
        xInOrbitFrame = None
        element = None
        cumulativeLen = 0.0
        for latticeElement in self.elList:
            if cumulativeLen + latticeElement.Lo > xPos:
                element = latticeElement
                xInOrbitFrame = xPos - cumulativeLen
                break
            cumulativeLen += latticeElement.Lo
        xLab, yLab, zLab = element.transform_orbit_frame_into_lab_frame(np.asarray([xInOrbitFrame, 0, 0]))
        return xLab, yLab

    def show_Lattice(self, particleCoords=None, particle=None, swarm=None, showRelativeSurvival=True,
                     showTraceLines=True,showImmediately=True,
                     showMarkers=True, traceLineAlpha=1.0, true_aspect_ratio=True, extraObjects=None, finalCoords=True,
                     saveTitle=None, dpi=150, defaultMarkerSize=1000, plotOuter: bool = False, plotInner: bool = True):
        # plot the lattice using shapely. if user provides particleCoords plot that on the graph. If users provides particle
        # or swarm then plot the last position of the particle/particles. If particles have not been traced, ie no
        # revolutions, then the x marker is not shown
        # particleCoords: Array or list holding particle coordinate such as [x,y]
        # particle: particle object
        # swarm: swarm of particles to plot.
        # showRelativeSurvival: when plotting swarm indicate relative particle survival by varying size of marker
        # showMarkers: Wether to plot a marker at the position of the particle
        # traceLineAlpha: Darkness of the trace line
        # true_aspect_ratio: Wether to plot the width and height to respect the actual width and height of the plot dimensions
        # it can make things hard to see
        # extraObjects: List of shapely objects to add to the plot. Used for adding things like apetures. Limited
        # functionality right now
        plt.close('all')

        def plot_Particle(particle, xMarkerSize=defaultMarkerSize):
            if particle.color is None:  # use default plotting behaviour
                color = 'red' if particle.clipped else 'green'
            else:  # else use the specified color
                color = particle.color
            if showMarkers:
                if particle.qf is not None:
                    xy = particle.qf[:2] if finalCoords else particle.qi[:2]
                    plt.scatter(*xy, marker='x', s=xMarkerSize, c=color)
                    plt.scatter(*xy, marker='o', s=10, c=color)
                else:  # no final coords, then plot initial coords with different marker
                    xy = particle.qi[:2]
                    plt.scatter(*xy, marker='^', s=30, c='blue')
            if showTraceLines:
                if particle.qArr is not None and len(particle.qArr) > 0:  # if there are lines to show
                    plt.plot(particle.qArr[:, 0], particle.qArr[:, 1], c=color, alpha=traceLineAlpha)

        for el in self.elList:
            if plotInner:
                elPlotPoints = el.SO.exterior.xy
                linestyle = ':' if plotOuter else '-'  # dashed inner if plotting iner
                plt.plot(*elPlotPoints, c=ELEMENT_PLOT_COLORS[type(el)], linestyle=linestyle)
            if plotOuter:
                elPlotPoints = el.SO_outer.exterior.xy
                plt.plot(*elPlotPoints, c=ELEMENT_PLOT_COLORS[type(el)])

        if particleCoords is not None:  # plot from the provided particle coordinate
            if len(particleCoords) == 3:  # if the 3d value is provided trim it to 2D
                particleCoords = particleCoords[:2]
            # plot the particle as both a dot and a X
            if showMarkers:
                plt.scatter(*particleCoords, marker='x', s=1000, c='r')
                plt.scatter(*particleCoords, marker='o', s=50, c='r')
        elif particle is not None:  # instead plot from provided particle
            plot_Particle(particle)
        if swarm is not None:
            maxRevs = swarm.longest_Particle_Life_Revolutions()
            if maxRevs == 0.0:  # if it hasn't been traced
                maxRevs = 1.0
            for particle in swarm:
                revs = particle.revolutions
                if revs is None:
                    revs = 0
                if showRelativeSurvival:
                    plot_Particle(particle, xMarkerSize=1000 * revs / maxRevs)
                else:
                    plot_Particle(particle)

        if extraObjects is not None:  # plot shapely objects that the used passed through. SO far this has limited
            # functionality
            for object in extraObjects:
                plt.plot(*object.coords.xy, linewidth=1, c='black')

        plt.grid()
        if true_aspect_ratio:
            plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        if saveTitle is not None:
            plt.savefig(saveTitle, dpi=dpi)
        if showImmediately:
            plt.show()
