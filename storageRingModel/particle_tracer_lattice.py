import warnings
from math import isclose, pi
from typing import Iterable, Union, Optional

import matplotlib.pyplot as plt
import numpy as np

from constants import DEFAULT_ATOM_SPEED
from helper_tools import parallel_evaluate
from lattice_elements.arrange_magnets import collect_valid_neighboring_magpylib_magnets
from lattice_elements.elements import BenderIdeal, BenderSim, LensIdeal, CombinerIdeal, \
    CombinerSim, CombinerLensSim, HalbachLensSim, Drift, ELEMENT_PLOT_COLORS
from lattice_elements.elements import Element
from shapely_object_builder import build_shapely_objects
from storage_ring_constraint_solver import is_particle_tracer_lattice_closed
from storage_ring_constraint_solver import solve_Floor_Plan, update_and_place_elements_from_floor_plan
from type_hints import ndarray, RealNum

# IMPROVEMENT: There is a ridiculous naming convention here with r0 r1 and r2


benderTypes = Union[BenderIdeal, BenderSim]
number = (int, float)


class FirstElementCombinerError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ParticleTracerLattice:

    def __init__(self, design_speed: RealNum = DEFAULT_ATOM_SPEED,
                 field_dens_mult: RealNum = 1.0, include_mag_errors: bool = False,
                 use_solenoid_field: bool = False, initial_location: tuple[RealNum, RealNum] = None,
                 initial_ang: RealNum = -pi, magnet_grade: str = 'N52', use_standard_tube_OD: bool = False,
                 use_long_range_fields: bool = False, include_misalignments: bool = False):
        """
        :param design_speed: Speed of the ideal particle traveling through the lattice
        :param field_dens_mult: Multiplier to modulate density of interpolation in elements. Carefull, computation time
            will scale as third power
        :param include_mag_errors: Whether to include the affects of magnet errors. See constants.py for the magnitude
        :param use_solenoid_field: Whether to apply a solenoid field through the center of each magnetic component
            to account for Majorana spin flips
        :param initial_location: The initial location of the input of the lattice, (x,y), m.
        :param initial_ang: Initial angle of the first element relative to the x axis, radians.
        :param magnet_grade: Grade of magnets in the lattice. See constants.py for options
        :param use_standard_tube_OD: Whether to only use off the shel tubing outside diameters
        :param use_long_range_fields: Whether to allow magnetic fields to reach into neighboring elements, including
            drift regions, and be included in interpolation. Can significantly increase computation time
        :param include_misalignments: Wether to allow magnets to be misaligned in the form of tilts and translations
            based on tolerances from constants.py .
        """
        assert field_dens_mult > 0.0
        self.design_speed = design_speed  # Design particle speed
        self.bender_indices: list[int] = []  # list that holds index values of benders. First bender is the
        # first one that the particle sees
        # if it started from beginning of the lattice. Remember that lattice cannot begin with a bender
        self.initial_location = (0.0, 0.0) if initial_location is None else initial_location
        self.initial_ang = initial_ang
        self.total_length = None  # total length of lattice, m
        self.field_dens_mult = field_dens_mult
        self.include_mag_errors = include_mag_errors
        self.use_standard_tube_OD = use_standard_tube_OD
        self.use_long_range_fields = use_long_range_fields
        self.include_misalignments = include_misalignments
        self.are_fast_field_helpers_built = False

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

    def add_element(self, el: Element, constrain=False):
        """Add an element to the lattice"""
        el.index = len(self.el_list)  # where the element is in the lattice
        self.el_list.append(el)  # add element to the list holding lattice elements in order
        if type(el) in (CombinerIdeal, CombinerLensSim, CombinerSim):
            assert self.combiner is None  # there can be only one!
            self.combiner = el
        if type(el) in (LensIdeal, HalbachLensSim):
            if constrain:
                self.set_constrained_linear_element(el)
                if type(el) is LensIdeal:
                    raise NotImplementedError('Behaviour has not been checked, though it should work')
        if type(el) in (BenderIdeal, BenderSim):
            self.bender_indices.append(el.index)

    def add_drift(self, L: RealNum, ap: RealNum = .05, input_tilt_angle: RealNum = 0.0,
                  output_tilt_angle: RealNum = 0.0, outer_half_width: RealNum = None) -> None:
        """
        Add drift region, which is field free region, to the lattice

        The general shape is a trapezoid in the xy lab frame, and a circle in the yz element frame. In the
        element frame in the xy plane the two bases are parallel with x-axis, and the input output can be at same
        angle relative to y-axis. Positive angles are counterclockwise notation. The length of the drift region is the
        same no matter the input/output tilt because the tilt is pinned at the centerline of the two bases of the
        trapezoid.

        :param L: Length of drift region, m
        :param ap: Aperture of drift region, m
        :param input_tilt_angle: Tilt angle of the input plane to the drift region, radians
        :param output_tilt_angle: Tilt angle of the output to the drift region, radians
        :param outer_half_width: Outer half width of drift region. For example, a valve body
        :return: None
        """

        self.add_element(Drift(self, L, ap, outer_half_width, input_tilt_angle, output_tilt_angle))

    def add_lens_ideal(self, L: RealNum, Bp: RealNum, rp: RealNum, constrain: bool = False, ap: RealNum = None) -> None:
        """
        Add an ideal magnetic hexapole lens to the lattice. Field norm goes as B0=Bp*r^2/rp^2

        :param L: Hard edge length of element, m.
        :param Bp: Field at bore face of lens,T.
        :param rp: Bore/pole radius of lens, m.
        :param ap: Aperture of lens, possibly a limit set by a vacuum tube, m.
        :param constrain: To use the element as a constraint, under construction
        :return: None
        """
        self.add_element(LensIdeal(self, L, Bp, rp, ap), constrain=constrain)

    def add_halbach_lens_sim(self, rp: Union[RealNum, tuple], L: Optional[RealNum], ap: RealNum = None,
                             constrain: bool = False, magnet_width: Union[RealNum, tuple] = None,
                             magnet_grade=None) -> None:
        """
        Add simulated halbach sextupole element to lattice.

        Combinations of rp and magnet_width specify how to handle multiple concentric layers, according to:

        rp    | magnet_width | Explanation
        RealNum | None        | Single layer with radius rp and magnet widths maximum possible
        RealNum | RealNum       | Single layer with radius rp and magnet widths of magnet_width
        tuple | tuple       | Number of layers is len(rp). Each layer has radius corresponding value in rp, such that
                            | rp[0] is radius of first layer. Same logic for magnet widths. rp and magnet_width must
                            | be same length.

        Configuration must be realistic.

        :param rp: Bore radius, m.
        :param L: Length of element, m. This includes fringe fields, actual magnet length will be smaller
        :param ap: Size of aperture
        :param constrain: Whether element is being used as part of a constraint. If so, fields construction will be
            deferred
        :param magnet_width: Width of cuboid magnets in polar plane of lens, m. Magnets length is L minus
            fringe fields.
        :param magnet_grade: Material grade of the mangets. If none use the default of the lattice
        :return: None
        """
        # IMPROVEMENT: IMPLEMENT magnet_grade uniformly
        magnet_grade = self.magnet_grade if magnet_grade is None else magnet_grade
        rp_layers = rp if isinstance(rp, tuple) else (rp,)
        magnet_width = (magnet_width,) if isinstance(magnet_width, number) else magnet_width
        self.add_element(HalbachLensSim(self, rp_layers, L, ap, magnet_width, magnet_grade), constrain=constrain)

    def add_combiner_ideal(self, Lm: RealNum = .2, c1: RealNum = 1, c2: RealNum = 20, ap: RealNum = .015,
                           size_scale: RealNum = 1.0, atom_state='LOW_SEEK') -> None:

        self.add_element(CombinerIdeal(self, Lm, c1, c2, ap, ap, ap / 2, size_scale, atom_state))

    def add_combiner_sim(self, size_scale: RealNum = 1.0, file: str = None, atom_state='LOW_SEEK') -> None:
        """
        Add model of our combiner from COMSOL. rarely used

        :param size_scale: How much to scale up or down dimensions of combiner
        :return: None
        """

        file = 'combinerV3.txt' if file is None else file
        self.add_element(CombinerSim(self, file, size_scale, atom_state))

    def add_combiner_sim_lens(self, Lm: RealNum, rp: RealNum, load_beam_offset: RealNum = 5e-3, layers: int = 1,
                              ap: RealNum = None, seed: int = None, atom_state='LOW_SEEK') -> None:

        """
        Add halbach hexapole lens combiner element to lattice.

        The edge of a hexapole lens is used to deflect high and weak field seeking states.
        :param Lm: Hard edge length of magnet, m. Total length of element depends on degree of deflection of nominal
        trajectory.
        :param rp: Bore radius of hexapole lens, m.
        :param load_beam_offset: Maximum desired acceptance diameter of load beam, m. Circulating beam is not specified
        :param layers: Number of concentric layers of magnets.
        :param seed: value that can be used as seed for numpy and python for reproducible results. This is provided
        because the SAME combiner must be in the ring and injector, and that sameness is enforced by the same seed
        between the two when using magnet imperfections and such
        :return: None
        """

        self.add_element(CombinerLensSim(self, Lm, rp, load_beam_offset, layers, ap, seed, atom_state))

    def add_bender_ideal(self, ang: RealNum, Bp: RealNum, rb: RealNum, rp: RealNum, ap: RealNum = None) -> None:
        """
        Add an ideal bender element, ie a revolved lens, to the lattice. Field norm goes as B0=Bp*r^2/rp^2


        :param ang: Bending angle, radians.
        :param Bp: Field strength at face of lenses, Tesla.
        :param rb: Bending radius of bender, the radius of curvature or the center-line of the bore, m.
        :param rp: Bore radius.
        :param ap:
        :return:
        """

        self.add_element(BenderIdeal(self, ang, Bp, rp, rb, ap))

    def add_segmented_halbach_bender(self, L_lens: RealNum, rp: RealNum, num_lenses: Optional[int], rb: RealNum,
                                     ap: RealNum = None) -> None:
        """
        Add a bender of simulated hexapole lenses to the lattice

        The bender is a series of hexapole lenses. They are cylinderical, not wedge shaped. The beginning and end
        of the bender are capped with half of a lens.

        :param L_lens: The length of an individual lens segment, m
        :param rp: Bore radius of each lens
        :param num_lenses: Number of lenses in the arc. Two of them are half lenses
        :param rb: Bending radius of bender, the radius of curvature or the center-line of the bore, m.
        :param ap: Aperture of the bender. Must be smaller than the limit set by interpolation grid sizing
        :return:
        """
        self.add_element(BenderSim(self, L_lens, rp, num_lenses, rb, ap))

    def build_lattice(self, constrain: bool, build_field_helpers: bool, parallel: bool):
        """Build the specified lattice. This includes:
        - Fill pre constrained parameters derive from simple inputs of length, field strength etc of each element.
        - Solve the floor plan layout. If constrained, solve for bumber of magnets and lengths of bending segment and
            lenses to find a valid configuration. 
        - Use the floor plan layout to update and place elementPT elements in the lab frame.
        - Use the results from the previous step to finish filling values of the element
        - Build shapely object for elementPT
        """

        self.fill_pre_constrained_parameters()
        self.place_elements(constrain)
        self.fill_post_constrained_parameters()

        if build_field_helpers:
            self.build_fast_field_helpers(parallel)

        self.is_closed = is_particle_tracer_lattice_closed(self)  # lattice may not have been constrained, but could
        # still be closed
        build_shapely_objects(self.el_list)
        self.total_length = 0
        for el in self.el_list:  # total length of particle's orbit in an element
            self.total_length += el.Lo

    def _build_fast_field_helper(self, el: Element) -> Element:
        magnets = collect_valid_neighboring_magpylib_magnets(el, self) if self.use_long_range_fields else None
        el.build_fast_field_helper(extra_magnets=magnets)
        return el

    def build_fast_field_helpers(self, parallel: bool, processes=-1) -> None:
        if self.include_mag_errors and parallel:
            warnings.warn(
                "Using parallel==True with magnet errors will not produce the same results as with parallel==False")
        built_els = parallel_evaluate(self._build_fast_field_helper, self.el_list,
                                      parallel=parallel, re_randomize=False, processes=processes)
        for i, el in enumerate(built_els):
            self.el_list[i] = el
        self.are_fast_field_helpers_built = True

    def fill_pre_constrained_parameters(self) -> None:
        for el in self.el_list:
            el.fill_pre_constrained_parameters()

    def fill_post_constrained_parameters(self) -> None:
        for el in self.el_list:
            el.fill_post_constrained_parameters()

    def place_elements(self, constrain):
        floor_plan = solve_Floor_Plan(self, constrain)
        update_and_place_elements_from_floor_plan(self, floor_plan)

    def is_lattice_straight(self):
        """Return True if the lattice is straight, False if there is some bend to it"""
        total_angle = sum([el.ang for el in self])
        return isclose(total_angle, 0.0)

    def end_lattice(self, constrain: bool = False, build_lattice: bool = True,
                    build_field_helpers: bool = True, parallel: bool = False) -> None:
        # for element in self.el_list:
        #     element.build()
        assert len(self) > 0
        self.catch_errors(constrain)
        if build_lattice:
            self.build_lattice(constrain, build_field_helpers, parallel)

    def catch_errors(self, constrain: bool) -> None:
        # catch any preliminary errors. Alot of error handling happens in other methods. This is a catch all for other
        # kinds. This class is not meant to have tons of error handling, so user must be cautious
        if isinstance(self.el_list[0], CombinerIdeal):  # first element can't be a combiner element
            raise FirstElementCombinerError('FIRST ELEMENT CANT BE A COMBINER')
        if len(self.bender_indices) >= 2:  # if there are two benders they must be the same.
            bender1 = self.el_list[self.bender_indices[0]]
            for i in self.bender_indices:
                if not type(bender1) is type(self.el_list[i]):
                    raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
        if constrain:
            if not len(self.bender_indices) >= 2:
                raise Exception('THERE MUST BE AT LEAST TWO BENDERS')
            for i in self.bender_indices:
                bender1, benderi = self.el_list[self.bender_indices[0]], self.el_list[i]
                if not type(bender1) is type(benderi):
                    raise Exception('BOTH BENDERS MUST BE THE SAME KIND')
                if not bender1.Lm == benderi.Lm or bender1.magnet_width != benderi.magnet_width:
                    raise Exception('SEGMENT LENGTHS AND MAGNET WIDTHS MUST BE EQUAL BETWEEN BENDERS')

    def get_element_before_and_after(self, el_center: Element) -> tuple[Element, Element]:
        if (el_center.index == len(self.el_list) - 1 or el_center.index == 0) and not self.is_closed:
            raise Exception('Element cannot be first or last if lattice is injector type')
        el_before_index = el_center.index - 1 if el_center.index != 0 else len(self.el_list) - 1
        el_after_index = el_center.index + 1 if el_center.index < len(self.el_list) - 1 else 0
        el_before = self.el_list[el_before_index]
        el_after = self.el_list[el_after_index]
        return el_before, el_after

    def get_lab_coords_from_orbit_distance(self, x_pos: ndarray) -> tuple[float, float]:
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

    def show(self, particle_coords=None, particle=None, swarm=None, show_Rel_Survival=True,
             show_trace_lines=True, show_immediately=True,
             show_markers=True, trace_line_alpha=1.0, true_aspect_ratio=True, extra_objects=None,
             final_coords=True, fig_size=None,
             save_title=None, dpi=150, default_marker_size=1000, plot_outer: bool = False,
             plot_inner: bool = True, show_grid=True):
        if fig_size is not None:
            plt.figure(figsize=fig_size)

        def plot_particle(particle, xMarkerSize=default_marker_size):
            color = 'red' if particle.clipped else 'green'
            if show_markers:
                if particle.qf is not None:
                    xy = particle.qf[:2] if final_coords else particle.qi[:2]
                    plt.scatter(*xy, marker='x', s=xMarkerSize, c=color, zorder=100)
                    plt.scatter(*xy, marker='o', s=10, c=color, zorder=100)
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
            plot_particle(particle)
        if swarm is not None:
            max_revs = swarm.longest_particle_life_revolutions()
            if max_revs == 0.0:  # if it hasn't been traced
                max_revs = 1.0
            for particle in swarm:
                revs = particle.revolutions
                if revs is None:
                    revs = 0
                if show_Rel_Survival:
                    plot_particle(particle, xMarkerSize=1000 * revs / max_revs)
                else:
                    plot_particle(particle)

        if extra_objects is not None:  # plot shapely objects that the used passed through. SO far this has limited
            # functionality
            for plot_object in extra_objects:
                plt.plot(*plot_object.coords.xy, linewidth=1, c='black')
        if show_grid:
            plt.grid()
        if true_aspect_ratio:
            plt.gca().set_aspect('equal')
        plt.xlabel('meters')
        plt.ylabel('meters')
        plt.tight_layout()
        if save_title is not None:
            plt.savefig(save_title, dpi=dpi)
        if show_immediately:
            plt.show()
