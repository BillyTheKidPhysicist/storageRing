# from geneticLensElement_Wrapper import GeneticLens
from typing import Iterable, Union, Optional

import matplotlib.pyplot as plt
import numpy as np

from constants import DEFAULT_ATOM_SPEED
from latticeElements.arrangeMagnets import collect_valid_neighboring_magpylib_magnets
from latticeElements.elements import BenderIdeal, HalbachBenderSimSegmented, LensIdeal, CombinerIdeal, \
    CombinerSim, CombinerHalbachLensSim, HalbachLensSim, Drift, ELEMENT_PLOT_COLORS
from latticeElements.elements import Element
from shapelyObjectBuilder import build_shapely_objects
from storageRingConstraintSolver import is_particle_tracer_lattice_closed
from storageRingConstraintSolver import solve_Floor_Plan, update_and_place_elements_from_floor_plan

# todo: There is a ridiculous naming convention here with r0 r1 and r2. If I ever hope for this to be helpful to other
# people, I need to change that. This was before my cleaner code approach


benderTypes = Union[BenderIdeal, HalbachBenderSimSegmented]


class ParticleTracerLattice:

    def __init__(self, speed_nominal: float = DEFAULT_ATOM_SPEED, lattice_type: str = 'storage_ring',
                 jitter_amp: float = 0.0, field_dens_mult: float = 1.0, use_mag_errors: bool = False,
                 use_solenoid_field: bool = False, initial_location: tuple[float, float] = None, initial_ang=None,
                 magnet_grade: str = 'N52', use_standard_mag_size: bool = False, use_standard_tube_OD: bool = False,
                 include_mag_cross_talk: bool = False, include_misalignments: bool = False):
        assert field_dens_mult > 0.0
        if lattice_type != 'storage_ring' and lattice_type != 'injector':
            raise Exception('invalid lattice type provided')
        if jitter_amp > 5e-3:
            raise Exception("Jitter values greater than 5 mm may begin to have unexpected results. Several parameters"
                            "depend on this value, and relatively large values were not planned for")
        if include_mag_cross_talk and use_mag_errors:
            raise NotImplementedError  # not sure this works.
        self.lattice_type = lattice_type  # options are 'storage_ring' or 'injector'. If storage_ring, the geometry is the the first element's
        # input at the origin and succeeding elements in a counterclockwise fashion. If injector, then first element's input
        # is also at the origin, but seceeding elements follow along the positive x axis
        self.speed_nominal = speed_nominal  # Design particle speed
        self.bender_indices: list[int] = []  # list that holds index values of benders. First bender is the
        # first one that the particle sees
        # if it started from beginning of the lattice. Remember that lattice cannot begin with a bender
        self.initial_location = (0.0, 0.0) if initial_location is None else initial_location
        self.initial_ang = -np.pi if initial_ang is None else initial_ang
        self.combiner_index: Optional[int] = None  # the index in the lattice where the combiner is
        self.total_length: Optional[float] = None  # total length of lattice, m
        self.jitter_amp = jitter_amp
        self.field_dens_mult = field_dens_mult
        self.use_mag_errors = use_mag_errors
        self.use_standard_tube_OD = use_standard_tube_OD
        self.use_standard_mag_size = use_standard_mag_size
        self.include_mag_cross_talk = include_mag_cross_talk
        self.include_misalignments = include_misalignments

        self.combiner: Optional[Element] = None  # combiner element object
        self.linear_elements_to_constrain: list[HalbachLensSim] = []  # elements whos length will be changed when the
        # lattice is constrained to satisfy geometry. Must be inside bending region

        self.is_closed = None  # is the lattice closed, ie end and beginning are smoothly connected?
        self.magnet_grade = magnet_grade
        self.use_solenoid_field = use_solenoid_field

        self.el_list: list[Element] = []  # to hold all the lattice elements

    def __iter__(self) -> Iterable[Element]:
        return (element for element in self.el_list)

    def __len__(self):
        return len(self.el_list)

    def __getitem__(self, index):
        return self.el_list[index]

    def set_constrained_linear_element(self, el: Element) -> None:
        self.linear_elements_to_constrain.append(el)
        if len(self.linear_elements_to_constrain) > 2:
            raise ValueError("there can only be 2 constrained linear elements")

    def add_combiner_sim(self, size_scale: float = 1.0) -> None:
        """
        Add model of our combiner from COMSOL. rarely used

        :param size_scale: How much to scale up or down dimensions of combiner
        :return: None
        """

        file = 'combinerV3.txt'
        el = CombinerSim(self, file, self.lattice_type, size_scale=size_scale)
        el.index = len(self.el_list)  # where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner = el
        self.combiner_index = el.index
        self.el_list.append(el)  # add element to the list holding lattice elements in order

    def add_combiner_sim_lens(self, Lm: float, rp: float, load_beam_offset: float = 5e-3, layers: int = 1,
                              ap: float = None,
                              seed: int = None) -> None:

        """
        Add halbach hexapole lens combiner element.

        The edge of a hexapole lens is used to deflect high and weak field seeking states. Transvers dimension of
        magnets are the maximum that can be used to for a halbach sextupole of given radius.

        :param Lm: Hard edge length of magnet, m. Total length of element depends on degree of deflection of nominal
        trajectory
        :param rp: Bore radius of hexapole lens, m
        :param load_beam_offset: Maximum desired acceptance diameter of load beam, m. Circulating beam is not specified
        :param layers: Number of concentric layers of magnets
        :return: None
        """

        el = CombinerHalbachLensSim(self, Lm, rp, load_beam_offset, layers, ap, seed)
        el.index = len(self.el_list)  # where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner = el
        self.combiner_index = el.index
        self.el_list.append(el)  # add element to the list holding lattice elements in order

    def add_halbach_lens_sim(self, rp: Union[float, tuple], L: Optional[float], ap: Optional[float] = None,
                             constrain: bool = False, magnet_width: Union[float, tuple] = None) -> None:
        """
        Add simulated halbach sextupole element to lattice.

        Combinations of rp and magnet_width specify how to handle multiple layers, according to:

        rp    | magnet_width | Explanation
        float | None        | Single layer with radius rp and magnet widths maximum possible
        float | float       | Single layer with radius rp and magnet widths of magnet_width
        tuple | tuple       | Number of layers is len(rp). Each layer has radius corresponding value in rp, such that
                            | rp[0] is radius of first layer. Same logic for magnet widths. rp and magnet_width must
                            | be same length.

        Configuration must be realistic.

        :param rp: Bore radius, m.
        :param L: Length of element, m. This includes fringe fields, actual magnet length will be smaller
        :param ap: Size of aperture
        :param constrain: Wether element is being used as part of a constraint. If so, fields construction will be
            deferred
        :param magnet_width: Width of cuboid magnets in polar plane of lens, m. Magnets length is L minus
            fringe fields.
        :return: None
        """
        rp_layers = rp if isinstance(rp, tuple) else (rp,)
        magnet_width = (magnet_width,) if isinstance(magnet_width, float) else magnet_width
        el = HalbachLensSim(self, rp_layers, L, ap, magnet_width)
        el.index = len(self.el_list)  # where the element is in the lattice
        self.el_list.append(el)  # add element to the list holding lattice elements in order
        if constrain:
            self.set_constrained_linear_element(el)

    # def add_Genetic_lens(self,lens: GeneticLens,ap: float)-> None:
    #     """
    #     Add genetic lens used for minimizing focus size. This is part of an idea to make a low aberration lens
    #
    #     :param lens: GeneticLens object that returns field values. This sextupole lens can be shimmed, and have bizarre
    #     :param ap: Aperture of genetic lens, m
    #     :return: None
    #     """
    #     el=geneticLens(self,lens,ap)
    #     el.index = len(self.el_list) #where the element is in the lattice
    #     self.el_list.append(el) #add element to the list holding lattice elements in order

    def add_lens_ideal(self, L: float, Bp: float, rp: float, constrain: bool = False, ap: float = None) -> None:
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
        el.index = len(self.el_list)  # where the element is in the lattice
        self.el_list.append(el)  # add element to the list holding lattice elements in order
        if constrain:
            self.set_constrained_linear_element(el)
            print('not fully supported feature')

    def add_drift(self, L: float, ap: float = .03, input_tilt_angle: float = 0.0, output_tilt_angle: float = 0.0,
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
        :param input_tilt_angle: Tilt angle of the input plane to the drift region.
        :param output_tilt_angle: Tilt angle of the output to the drift region.
        :param outer_half_width: Outer half width of drift region. For example, a valve.
        :return:
        """

        el = Drift(self, L, ap, outer_half_width, input_tilt_angle, output_tilt_angle)  # create a drift element object
        el.index = len(self.el_list)  # where the element is in the lattice
        self.el_list.append(el)  # add element to the list holding lattice elements in order

    def add_segmented_halbach_bender(self, Lm: float, rp: float, num_magnets: Optional[int], rb: float,
                                     r_offset_fact: float = 1.0, ap: float = None) -> None:
        # Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        # L_cap: Length of element on the end/input of bender
        # output_offsetFact: factor to multply the theoretical offset by to minimize oscillations in the bending segment.
        # modeling shows that ~.675 is ideal
        el = HalbachBenderSimSegmented(self, Lm, rp, num_magnets, rb, ap, r_offset_fact)
        el.index = len(self.el_list)  # where the element is in the lattice
        self.bender_indices.append(el.index)
        self.el_list.append(el)

    def add_bender_ideal(self, ang: float, Bp: float, rb: float, rp: float, ap: float = None) -> None:
        # Add element to the lattice. see elementPTPreFactor.py for more details on specific element
        # ang: Bending angle of bender, radians
        # rb: nominal bending radius of element's centerline. Actual radius is larger because particle 'rides' a little
        # outside this, m
        # Bp: field strength at pole face of lens, T
        # rp: bore radius of element, m
        # ap: size of apeture. If none then a fraction of the bore radius. Can't be bigger than bore radius, unitless

        el = BenderIdeal(self, ang, Bp, rp, rb, ap)  # create a bender element object
        el.index = len(self.el_list)  # where the element is in the lattice
        self.bender_indices.append(el.index)
        self.el_list.append(el)  # add element to the list holding lattice elements in order

    def add_combiner_ideal(self, Lm: float = .2, c1: float = 1, c2: float = 20, ap: float = .015,
                           size_scale: float = 1.0) -> None:
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

        el = CombinerIdeal(self, Lm, c1, c2, ap, ap, ap / 2, size_scale)  # create a combiner element object
        el.index = len(self.el_list)  # where the element is in the lattice
        assert self.combiner is None  # there can be only one!
        self.combiner = el
        self.combiner_index = el.index
        self.el_list.append(el)  # add element to the list holding lattice elements in order

    def build_lattice(self, constrain: bool, build_field_helpers: bool):
        """Build the specified lattice. This includes:
        - Fill pre constrained parameters derive from simple inputs of length, field strength etc of each element.
        - Solve the floor plan layout. If constrained, solve for bumber of magnets and lengths of bending segment and
            lenses to find a valid configuration. 
        - Use the floor plan layout to update and place elementPT elements in the lab frame.
        - Use the results from the previous step to finish filling values of the element
        - Build shapely object for elementPT
        """

        for el in self.el_list:
            el.fill_pre_constrained_parameters()
        floor_plan = solve_Floor_Plan(self, constrain)
        update_and_place_elements_from_floor_plan(self, floor_plan)

        for el in self.el_list:
            el.fill_post_constrained_parameters()
        if build_field_helpers:
            for el in self.el_list:
                magnets = collect_valid_neighboring_magpylib_magnets(el, self) if self.include_mag_cross_talk else None
                el.build_fast_field_helper(extra_magnets=magnets)

        self.is_closed = is_particle_tracer_lattice_closed(self)  # lattice may not have been constrained, but could
        # still be closed
        if self.lattice_type == 'storage_ring' and constrain:  # double check
            assert is_particle_tracer_lattice_closed(self)
        build_shapely_objects(self.el_list)
        self.total_length = 0
        for el in self.el_list:  # total length of particle's orbit in an element
            self.total_length += el.Lo

    def end_lattice(self, constrain: bool = False, build_lattice: bool = True,
                    build_field_helpers: bool = True) -> None:
        # for element in self.el_list:
        #     element.build()
        assert len(self) > 0
        self.catch_errors(constrain)
        if build_lattice:
            self.build_lattice(constrain, build_field_helpers)

    def catch_errors(self, constrain: bool) -> None:
        # catch any preliminary errors. Alot of error handling happens in other methods. This is a catch all for other
        # kinds. This class is not meant to have tons of error handling, so user must be cautious
        if isinstance(self.el_list[0], BenderIdeal):  # first element can't be a bending element
            raise Exception('FIRST ELEMENT CANT BE A BENDER')
        if isinstance(self.el_list[0], CombinerIdeal):  # first element can't be a combiner element
            raise Exception('FIRST ELEMENT CANT BE A COMBINER')
        if len(self.bender_indices) >= 2:  # if there are two benders they must be the same.
            bender1 = self.el_list[self.bender_indices[0]]
            for i in self.bender_indices:
                if not type(bender1) is type(self.el_list[i]):
                    raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
        if constrain:
            if self.lattice_type != 'storage_ring':
                raise Exception('Constrained lattice must be storage ring type')
            if not len(self.bender_indices) >= 2:
                raise Exception('THERE MUST BE AT LEAST TWO BENDERS')
            for i in self.bender_indices:
                bender1, benderi = self.el_list[self.bender_indices[0]], self.el_list[i]
                if not type(bender1) is type(benderi):
                    raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
                if not bender1.Lm == benderi.Lm or bender1.magnet_width != benderi.magnet_width:
                    raise Exception('SEGMENT LENGTHS AND MAGNET WIDTHS MUST BE EQUAL BETWEEN BENDERS')
            if self.combiner is None:
                raise Exception('COMBINER MUST BE PRESENT')

    def get_element_before_and_after(self, el_center: Element) -> tuple[Element, Element]:
        if (el_center.index == len(self.el_list) - 1 or el_center.index == 0) and self.lattice_type == 'injector':
            raise Exception('Element cannot be first or last if lattice is injector type')
        el_before_index = el_center.index - 1 if el_center.index != 0 else len(self.el_list) - 1
        el_after_index = el_center.index + 1 if el_center.index < len(self.el_list) - 1 else 0
        el_before = self.el_list[el_before_index]
        el_after = self.el_list[el_after_index]
        return el_before, el_after

    def get_lab_coords_from_orbit_distance(self, x_pos: np.ndarray) -> tuple[float, float]:
        # x_pos: distance along ideal orbit
        assert x_pos >= 0.0
        x_pos = x_pos % self.total_length  # xpos without multiple revolutions
        x_in_orbit_frame = None
        element = None
        cumulative_length = 0.0
        for latticeElement in self.el_list:
            if cumulative_length + latticeElement.Lo > x_pos:
                element = latticeElement
                x_in_orbit_frame = x_pos - cumulative_length
                break
            cumulative_length += latticeElement.Lo
        x_lab, y_lab, z_lab = element.transform_orbit_frame_into_lab_frame(np.asarray([x_in_orbit_frame, 0, 0]))
        return x_lab, y_lab

    def show_lattice(self, particle_coords=None, particle=None, swarm=None, show_Rel_Survival=True,
                     show_trace_lines=True, show_immediately=True,
                     show_markers=True, trace_line_alpha=1.0, true_aspect_ratio=True, extra_objects=None,
                     final_coords=True,
                     save_title=None, dpi=150, default_marker_size=1000, plot_outer: bool = False,
                     plot_inner: bool = True):
        # plot the lattice using shapely. if user provides particle_coords plot that on the graph. If users provides particle
        # or swarm then plot the last position of the particle/particles. If particles have not been traced, ie no
        # revolutions, then the x marker is not shown
        # particle_coords: Array or list holding particle coordinate such as [x,y]
        # particle: particle object
        # swarm: swarm of particles to plot.
        # show_Rel_Survival: when plotting swarm indicate relative particle survival by varying size of marker
        # show_markers: Wether to plot a marker at the position of the particle
        # trace_line_alpha: Darkness of the trace line
        # true_aspect_ratio: Wether to plot the width and height to respect the actual width and height of the plot dimensions
        # it can make things hard to see
        # extra_objects: List of shapely objects to add to the plot. Used for adding things like apetures. Limited
        # functionality right now
        plt.close('all')

        def plot_Particle(particle, xMarkerSize=default_marker_size):
            color = 'red' if particle.clipped else 'green'
            if show_markers:
                if particle.qf is not None:
                    xy = particle.qf[:2] if final_coords else particle.qi[:2]
                    plt.scatter(*xy, marker='x', s=xMarkerSize, c=color)
                    plt.scatter(*xy, marker='o', s=10, c=color)
                else:  # no final coords, then plot initial coords with different marker
                    xy = particle.qi[:2]
                    plt.scatter(*xy, marker='^', s=30, c='blue')
            if show_trace_lines:
                if particle.q_vals is not None and len(particle.q_vals) > 0:  # if there are lines to show
                    plt.plot(particle.q_vals[:, 0], particle.q_vals[:, 1], c=color, alpha=trace_line_alpha)

        for el in self.el_list:
            if plot_inner:
                el_plot_points = el.SO.exterior.xy
                linestyle = ':' if plot_outer else '-'  # dashed inner if plotting iner
                plt.plot(*el_plot_points, c=ELEMENT_PLOT_COLORS[type(el)], linestyle=linestyle)
            if plot_outer:
                el_plot_points = el.SO_outer.exterior.xy
                plt.plot(*el_plot_points, c=ELEMENT_PLOT_COLORS[type(el)])

        if particle_coords is not None:  # plot from the provided particle coordinate
            if len(particle_coords) == 3:  # if the 3d value is provided trim it to 2D
                particle_coords = particle_coords[:2]
            # plot the particle as both a dot and a X
            if show_markers:
                plt.scatter(*particle_coords, marker='x', s=1000, c='r')
                plt.scatter(*particle_coords, marker='o', s=50, c='r')
        elif particle is not None:  # instead plot from provided particle
            plot_Particle(particle)
        if swarm is not None:
            max_revs = swarm.longest_particle_life_revolutions()
            if max_revs == 0.0:  # if it hasn't been traced
                max_revs = 1.0
            for particle in swarm:
                revs = particle.revolutions
                if revs is None:
                    revs = 0
                if show_Rel_Survival:
                    plot_Particle(particle, xMarkerSize=1000 * revs / max_revs)
                else:
                    plot_Particle(particle)

        if extra_objects is not None:  # plot shapely objects that the used passed through. SO far this has limited
            # functionality
            for plot_object in extra_objects:
                plt.plot(*plot_object.coords.xy, linewidth=1, c='black')

        plt.grid()
        if true_aspect_ratio:
            plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        if save_title is not None:
            plt.savefig(save_title, dpi=dpi)
        if show_immediately:
            plt.show()
