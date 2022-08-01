"""
Functions to construct shapely geometry objects for elements in ElementPT.py. These shapely objects are primarily
used for:
- enforcing spatial constraints such as elements not overlapping and the ring fitting in the room
- visualizing the floorplan layout
- a secondary method of testing if particles are in a specific element
"""
import copy
from math import pi, tan
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from shapely.geometry import Polygon

from constants import FLAT_WALL_VACUUM_THICKNESS, TUBE_WALL_THICKNESS
from latticeElements.elements import BenderIdeal, LensIdeal, CombinerIdeal, HalbachLensSim, Drift, \
    HalbachBenderSimSegmented, CombinerHalbachLensSim, CombinerSim
from latticeElements.elements import Element
from typeHints import RealNum

BENDER_POINTS = 250  # how many points to represent the bender with along each curve
SMALL_NUMBER = 1e-16


def make_halbach_lens_outer_points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a halbach lens. Overall shape is
    a rectangle overlayd on a shorter but wider rectangle. This represents the width of the magnets"""
    assert type(el) is HalbachLensSim
    half_width = el.outer_half_width
    vacuum_tube_outer_width = el.ap + TUBE_WALL_THICKNESS
    fringe_length = el.fringe_field_length
    point1 = np.asarray([0.0, vacuum_tube_outer_width])
    point2 = np.asarray([fringe_length, vacuum_tube_outer_width])
    point3 = np.asarray([fringe_length, half_width])
    point4 = np.asarray([el.L - fringe_length, half_width])
    point5 = np.asarray([el.L - fringe_length, vacuum_tube_outer_width])
    point6 = np.asarray([el.L, vacuum_tube_outer_width])
    top_points = [point1, point2, point3, point4, point5, point6]

    bottom_points = np.flip(np.row_stack(top_points), axis=0)  # points need to go clockwise
    bottom_points[:, -1] *= -1
    points_outer = [*top_points, *bottom_points]
    return points_outer


def make_hexapole_bender_caps_outer_points(el: Element) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Make points that describe the shape of the input and outputs of the hexapole bender. They have
    a stepped shape from the width of the magnets. Output cap points along -y and is easy to specify the
    coordinates and is then mirrored to produce the input cap as well."""

    vacuum_tube_outer_width = el.ap + TUBE_WALL_THICKNESS
    points_cap_start = [np.array([el.rb - el.outer_half_width, -el.Lm / 2.0]),
                        np.array([el.rb - vacuum_tube_outer_width, -el.Lm / 2.0]),
                        np.array([el.rb - vacuum_tube_outer_width, -el.L_cap]),
                        np.array([el.rb + vacuum_tube_outer_width, -el.L_cap]),
                        np.array([el.rb + vacuum_tube_outer_width, -el.Lm / 2.0]),
                        np.array([el.rb + el.outer_half_width, -el.Lm / 2.0])]

    points_cap_end = []
    m = np.tan(el.ang / 2.0)
    for point in points_cap_start:
        x_start, y_start = point
        d = (x_start + y_start * m) / (1 + m ** 2)
        point_end = np.array([2 * d - x_start, 2 * d * m - y_start])
        points_cap_end.append(point_end)
    return points_cap_start, points_cap_end


def make_hexapole_bender_outer_points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a hexapole bending section.
    Shape is a toroid with short straight section at input/ouput, with another wider but shorter toroid ontop """

    assert type(el) is HalbachBenderSimSegmented
    phi_arr = np.linspace(el.ang, 0.0, BENDER_POINTS)  # + el.theta + np.pi / 2  # angles swept out

    x_inner = (el.rb - el.outer_half_width) * np.cos(phi_arr)  # x values for inner bend
    y_inner = (el.rb - el.outer_half_width) * np.sin(phi_arr)  # y values for inner bend
    x_outer = np.flip((el.rb + el.outer_half_width) * np.cos(phi_arr))  # x values for outer bend
    y_outer = np.flip((el.rb + el.outer_half_width) * np.sin(phi_arr))  # y values for outer bend

    points_cap_start, points_cap_end = make_hexapole_bender_caps_outer_points(el)
    points_cap_start, points_cap_end = np.array(points_cap_start), np.array(points_cap_end)
    x_inner = np.append(np.flip(points_cap_end[:, 0]), x_inner)
    y_inner = np.append(np.flip(points_cap_end[:, 1]), y_inner)
    x_inner = np.append(x_inner, points_cap_start[:, 0])
    y_inner = np.append(y_inner, points_cap_start[:, 1])
    x = np.append(x_inner, x_outer)  # list of x values in order
    y = np.append(y_inner, y_outer)  # list of y values in order
    points_outer = np.column_stack((x, y))  # shape the coordinates and make the object
    rot_mat_2D = Rot.from_rotvec([0, 0, el.theta - el.ang + pi / 2]).as_matrix()[:2, :2]
    for i, point in enumerate(points_outer):
        points_outer[i] = rot_mat_2D @ point
    points_outer += el.r0[:2]
    return points_outer


def make_hexapole_combiner_outer_points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a halbach combiner. Very similiar
    to halbach lens geometry, but with tiled input and enlarged input"""

    # todo: the tube wall stuff here is wrong, need to use combienr tube wall stuff

    # todo: these equations should be condensed into a single function because they are used so much. It has caused
    # alot of trouble. I'm raelly violating DRY here and abstraction here

    # pylint: disable=too-many-locals
    assert type(el) is CombinerHalbachLensSim
    ap_right, ap_left = el.ap, el.ap
    extra_fact = el.acceptance_width / el.ap
    half_width = el.outer_half_width
    point1 = np.array([0, ap_right + TUBE_WALL_THICKNESS])  # top left ( in standard xy plane) when theta=0
    point2 = np.array([el.space, ap_right + TUBE_WALL_THICKNESS])  # top left ( in standard xy plane) when theta=0
    point3 = np.array([el.space, half_width])  # top left ( in standard xy plane) when theta=0
    point4 = np.array([el.Lb, half_width])  # top middle when theta=0
    point5 = np.array([el.Lb, extra_fact * ap_right + TUBE_WALL_THICKNESS])  # top middle when theta=0
    point6 = np.array(
        [el.Lb + (el.La - (extra_fact * ap_right + TUBE_WALL_THICKNESS) * np.sin(el.ang)) * np.cos(el.ang),
         (extra_fact * ap_right + TUBE_WALL_THICKNESS) + (
                 el.La - (extra_fact * ap_right + TUBE_WALL_THICKNESS) * np.sin(el.ang)) * np.sin(el.ang)])
    point7 = np.array([el.Lb + (el.La + (extra_fact * ap_left + TUBE_WALL_THICKNESS) * np.sin(el.ang)) * np.cos(el.ang),
                       -(extra_fact * ap_left + TUBE_WALL_THICKNESS) + (el.La +
                                                                        (extra_fact * ap_left + TUBE_WALL_THICKNESS)
                                                                        * np.sin(el.ang)) * np.sin(el.ang)])
    point8 = np.array([el.Lb, -extra_fact * ap_left - TUBE_WALL_THICKNESS])  # bottom middle when theta=0
    point9 = np.array([el.Lb, -half_width])  # bottom middle when theta=0
    point10 = np.array([el.space, -half_width])  # bottom middle when theta=0
    point11 = np.array([el.space, -ap_left - TUBE_WALL_THICKNESS])  # bottom middle when theta=0
    point12 = np.array([0, -ap_left - TUBE_WALL_THICKNESS])  # bottom left when theta=0
    points_outer = [point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11,
                    point12]
    for i, point in enumerate(points_outer):
        points_outer[i] = el.R_Out @ point + el.r2[:2]
    return points_outer


def make_halbach_combiner_inner_points(el: Element) -> list[np.ndarray]:
    assert type(el) is CombinerHalbachLensSim
    Lb_vac = el.Lb if type(el) is CombinerIdeal else el.Lb + FLAT_WALL_VACUUM_THICKNESS
    ap, acceptance_width = el.ap, el.acceptance_width

    m = np.tan(el.ang)
    assert type(el) is CombinerHalbachLensSim
    y_top = lambda x: m * x + (el.acceptance_width - m * el.Lb)  # upper limit
    y_right = lambda x: (-1 / m) * x + el.La * np.sin(el.ang) + (el.Lb + el.La * np.cos(el.ang)) / m
    y_bot = lambda x: m * x + (-el.acceptance_width - m * el.Lb)

    q1_inner = np.asarray([0, ap])  # top left ( in standard xy plane) when theta=0
    q2_inner = np.asarray([Lb_vac, ap])  # top middle when theta=0

    q3_inner = np.asarray([Lb_vac, y_top(Lb_vac)])  # top middle when theta=0

    q4_inner = np.asarray([el.Lb + (el.La - acceptance_width * np.sin(el.ang)) * np.cos(el.ang),
                           acceptance_width + (el.La - acceptance_width * np.sin(el.ang)) * np.sin(
                               el.ang)])  # top right when theta=0

    q5_inner = np.asarray([el.Lb + (el.La + acceptance_width * np.sin(el.ang)) * np.cos(el.ang),
                           -acceptance_width + (el.La + acceptance_width * np.sin(el.ang)) * np.sin(
                               el.ang)])  # bottom right when theta=0

    q6_inner = np.asarray([Lb_vac, y_bot(Lb_vac)])  # bottom middle when theta=0

    q7_inner = np.asarray([Lb_vac, -ap])  # bottom middle when theta=0
    q8_inner = np.asarray([0, -ap])  # bottom left when theta=0
    points_inner = [q1_inner, q2_inner, q3_inner, q4_inner, q5_inner, q6_inner, q7_inner, q8_inner]
    for point in points_inner:
        point[:] = el.R_Out @ point + el.r2[:2]
    return points_inner


def make_combiner_inner_points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the inner (vacuum tube) geometry of a halbach combiner.
    Basically a rectangle with a wider tilted rectangle coming off one end (the input)"""

    # todo: this doesn't work with combiner ideal or sim
    assert type(el) in (CombinerIdeal, CombinerSim)

    ap_right, ap_left = el.ap_right, el.ap_left
    q1_inner = np.asarray([0, ap_right])  # top left ( in standard xy plane) when theta=0
    q2_inner = np.asarray([el.Lb, ap_right])  # top middle when theta=0
    q3_inner = np.asarray([el.Lb + (el.La - ap_right * np.sin(el.ang)) * np.cos(el.ang),
                           ap_right + (el.La - ap_right * np.sin(el.ang)) * np.sin(el.ang)])  # top right when theta=0

    q4_inner = np.asarray([el.Lb + (el.La + ap_left * np.sin(el.ang)) * np.cos(el.ang),
                           - ap_left + (el.La + ap_left * np.sin(el.ang)) * np.sin(
                               el.ang)])  # bottom right when theta=0
    q5_inner = np.asarray([el.Lb, -ap_left])  # bottom middle when theta=0
    q6_inner = np.asarray([0, -ap_left])  # bottom left when theta=0
    points_inner = [q1_inner, q2_inner, q3_inner, q4_inner, q5_inner, q6_inner]
    for point in points_inner:
        point[:] = el.R_Out @ point + el.r2[:2]
    return points_inner


def make_rectangle(L: float, half_width: float) -> list[np.ndarray]:
    """Make a simple rectangle. Used to model drift regions or interior(vacuum tube) of lenses"""
    q1_inner = np.asarray([0.0, half_width])  # top left when theta=0
    q2_inner = np.asarray([L, half_width])  # top right when theta=0
    q3_inner = np.asarray([L, -half_width])  # bottom right when theta=0
    q4_inner = np.asarray([0, -half_width])  # bottom left when theta=0
    return [q1_inner, q2_inner, q3_inner, q4_inner]


def make_lens_shape(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of lens elements and drift
    region. Drift regions are just vacuum tubes between elements"""
    assert type(el) in (LensIdeal, HalbachLensSim)
    points_inner = make_rectangle(el.L, el.ap)
    if type(el) is HalbachLensSim:
        points_outer = make_halbach_lens_outer_points(el)
    else:
        points_outer = copy.deepcopy(points_inner)
    for point in [*points_inner, *points_outer]:
        point[:] = el.R_Out @ point + el.r1[:2]  # must modify the original array!
    return Polygon(points_outer), Polygon(points_inner)


def is_drift_input_output_tilt_valid(L: RealNum, half_width: RealNum, theta1: RealNum,
                                     theta2: RealNum) -> bool:
    """Check that te drift region, a trapezoid shape, is concave. theta1 and theta2 can violate this condition
    depending on their value"""
    m1, m2 = tan(pi / 2 + theta1), tan(pi / 2 + theta2)
    y_cross = np.inf if theta1 == theta2 == 0.0 else m1 * m2 * L / (m2 - m1 + SMALL_NUMBER)
    return not (abs(y_cross) < half_width or abs(theta1) > pi / 2 or abs(theta2) > pi / 2)


def make_trapezoid_points(L, theta1, theta2, half_width):
    """Make the coordinates of the vertices of the trapezoid."""
    assert is_drift_input_output_tilt_valid(L, half_width, theta1, theta2)
    points = [np.array([-half_width * tan(theta1), half_width]),
              np.array([L - half_width * tan(theta2), half_width]),
              np.array([L + half_width * tan(theta2), -half_width]),
              np.array([half_width * tan(theta1), -half_width])]
    return points


def make_drift_shape(el: Drift):
    """Make shapely objects for drift element. Drift element is trapezoid shaped to allow for tilted input/output"""
    L, ap, outer_half_width, theta1, theta2 = el.L, el.ap, el.outer_half_width, el.input_tilt_angle, el.output_tilt_angle
    points_inner = make_trapezoid_points(L, theta1, theta2, ap)
    points_outer = make_trapezoid_points(L, theta1, theta2, outer_half_width)
    for point in [*points_inner, *points_outer]:
        point[:] = el.R_Out @ point + el.r1[:2]  # must modify the original array!
    return Polygon(points_outer), Polygon(points_inner)


def make_bender_shape(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of bender elements"""

    assert type(el) in (BenderIdeal, HalbachBenderSimSegmented)
    half_width = el.ap
    theta = el.theta
    phiArr = np.linspace(0, -el.ang, BENDER_POINTS) + theta + pi / 2  # angles swept out
    r0 = el.r0.copy()
    x_inner = (el.rb - half_width) * np.cos(phiArr) + r0[0]  # x values for inner bend
    y_inner = (el.rb - half_width) * np.sin(phiArr) + r0[1]  # y values for inner bend
    x_outer = np.flip((el.rb + half_width) * np.cos(phiArr) + r0[0])  # x values for outer bend
    y_outer = np.flip((el.rb + half_width) * np.sin(phiArr) + r0[1])  # y values for outer bend

    if isinstance(el, HalbachBenderSimSegmented):
        x_inner = np.append(x_inner[0] + el.nb[0] * el.L_cap, x_inner)
        y_inner = np.append(y_inner[0] + el.nb[1] * el.L_cap, y_inner)
        x_inner = np.append(x_inner, x_inner[-1] + el.ne[0] * el.L_cap)
        y_inner = np.append(y_inner, y_inner[-1] + el.ne[1] * el.L_cap)
        x_outer = np.append(x_outer, x_outer[-1] + el.nb[0] * el.L_cap)
        y_outer = np.append(y_outer, y_outer[-1] + el.nb[1] * el.L_cap)
        x_outer = np.append(x_outer[0] + el.ne[0] * el.L_cap, x_outer)
        y_outer = np.append(y_outer[0] + el.ne[1] * el.L_cap, y_outer)

    x = np.append(x_inner, x_outer)  # list of x values in order
    y = np.append(y_inner, y_outer)  # list of y values in order
    points_inner = np.column_stack((x, y))  # shape the coordinates and make the object
    if type(el) is BenderIdeal:
        points_outer = copy.deepcopy(points_inner)
    elif type(el) is HalbachBenderSimSegmented:
        points_outer = make_hexapole_bender_outer_points(el)
    else:
        raise NotImplementedError
    return Polygon(points_outer), Polygon(points_inner)


def make_combiner_shape(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of combiner elements"""
    assert type(el) in (CombinerIdeal, CombinerSim, CombinerHalbachLensSim)

    if type(el) in (CombinerIdeal, CombinerSim):
        points_inner = make_combiner_inner_points(el)
        points_outer = copy.deepcopy(points_inner)
    elif type(el) is CombinerHalbachLensSim:
        points_inner = make_halbach_combiner_inner_points(el)
        points_outer = make_hexapole_combiner_outer_points(el)
    else:
        raise NotImplementedError
    return Polygon(points_outer), Polygon(points_inner)


def make_element_shape(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of elementPT objects such
    as lenses, drifts, benders and combiners"""
    if type(el) in (HalbachLensSim, LensIdeal):
        shapely_outer, shapely_inner = make_lens_shape(el)
    elif type(el) in (BenderIdeal, HalbachBenderSimSegmented):
        shapely_outer, shapely_inner = make_bender_shape(el)
    elif type(el) in (CombinerIdeal, CombinerSim, CombinerHalbachLensSim):
        shapely_outer, shapely_inner = make_combiner_shape(el)
    elif type(el) is Drift:
        shapely_outer, shapely_inner = make_drift_shape(el)
    else:
        raise NotImplementedError
    assert shapely_outer.buffer(1e-9).contains(shapely_inner)  # inner object must be smaller or same size as outer.
    # add a small buffer for numerical issues
    return shapely_outer, shapely_inner


def build_shapely_objects(elements: Iterable) -> None:
    """Build the inner and outer shapely obejcts that represent the 2D geometries of the each element. This is the
    projection of an element onto the horizontal plane that bisects it. _inner geometry repsents vacuum, and outer
     the eternal profile. Ideal elements have to out profile"""

    # todo: there is a mild disagreement here between shapely and is inside method with the combiner

    for el in elements:
        shapely_object_outer, shapely_object_inner = make_element_shape(el)
        el.SO = shapely_object_inner
        el.SO_outer = shapely_object_outer
