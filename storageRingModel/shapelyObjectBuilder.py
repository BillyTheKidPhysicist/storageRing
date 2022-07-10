"""
Functions to construct shapely geometry objects for elements in ElementPT.py. These shapely objects are primarily
used for:
- enforcing spatial constraints such as elements not overlapping and the ring fitting in the room
- visualizing the floorplan layout
- a secondary method of testing if particles are in a specific element
"""
import copy
from math import pi, tan

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


def make_Halbach_Lens_Outer_Points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a halbach lens. Overall shape is
    a rectangle overlayd on a shorter but wider rectangle. This represents the width of the magnets"""
    assert type(el) is HalbachLensSim
    half_width = el.outer_half_width
    vacuum_tube_outer_width = el.ap + TUBE_WALL_THICKNESS
    fringe_length = el.fringeFieldLength
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


def make_Hexapole_Bender_Caps_Outer_Points(el: Element) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Make points that describe the shape of the input and outputs of the hexapole bender. They have
    a stepped shape from the width of the magnets. Output cap points along -y and is easy to specify the
    coordinates and is then mirrored to produce the input cap as well."""

    vacuum_tube_outer_width = el.ap + TUBE_WALL_THICKNESS
    points_cap_start = [np.array([el.rb - el.outer_half_width, -el.Lm / 2.0]),
                      np.array([el.rb - vacuum_tube_outer_width, -el.Lm / 2.0]),
                      np.array([el.rb - vacuum_tube_outer_width, -el.Lcap]),
                      np.array([el.rb + vacuum_tube_outer_width, -el.Lcap]),
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


def make_Hexapole_Bender_Outer_Points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a hexapole bending section.
    Shape is a toroid with short straight section at input/ouput, with another wider but shorter toroid ontop """

    assert type(el) is HalbachBenderSimSegmented
    phi_arr = np.linspace(el.ang, 0.0, BENDER_POINTS)  # + el.theta + np.pi / 2  # angles swept out

    x_inner = (el.rb - el.outer_half_width) * np.cos(phi_arr)  # x values for inner bend
    y_inner = (el.rb - el.outer_half_width) * np.sin(phi_arr)  # y values for inner bend
    x_outer = np.flip((el.rb + el.outer_half_width) * np.cos(phi_arr))  # x values for outer bend
    y_outer = np.flip((el.rb + el.outer_half_width) * np.sin(phi_arr))  # y values for outer bend

    points_cap_start, points_cap_end = make_Hexapole_Bender_Caps_Outer_Points(el)
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


def make_Hexapole_Combiner_Outer_Points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a halbach combiner. Very similiar
    to halbach lens geometry, but with tiled input and enlarged input"""

    #todo: the tube wall stuff here is wrong, need to use combienr tube wall stuff

    #todo: these equations should be condensed into a single function because they are used so much. It has caused
    #alot of trouble. I'm raelly violating DRY here and abstraction here

    # pylint: disable=too-many-locals
    assert type(el) is CombinerHalbachLensSim
    apR, apL = el.ap, el.ap
    extra_fact = el.acceptance_width/el.ap
    half_width = el.outer_half_width
    point1 = np.array([0, apR + TUBE_WALL_THICKNESS])  # top left ( in standard xy plane) when theta=0
    point2 = np.array([el.space, apR + TUBE_WALL_THICKNESS])  # top left ( in standard xy plane) when theta=0
    point3 = np.array([el.space, half_width])  # top left ( in standard xy plane) when theta=0
    point4 = np.array([el.Lb, half_width])  # top middle when theta=0
    point5 = np.array([el.Lb, extra_fact*apR + TUBE_WALL_THICKNESS])  # top middle when theta=0
    point6 = np.array([el.Lb + (el.La - (extra_fact*apR + TUBE_WALL_THICKNESS) * np.sin(el.ang)) * np.cos(el.ang),
                       (extra_fact*apR + TUBE_WALL_THICKNESS) + (
                               el.La - (extra_fact*apR + TUBE_WALL_THICKNESS) * np.sin(el.ang)) * np.sin(el.ang)])
    point7 = np.array([el.Lb + (el.La +( extra_fact * apL + TUBE_WALL_THICKNESS) * np.sin(el.ang)) * np.cos(el.ang),
                       -( extra_fact * apL + TUBE_WALL_THICKNESS) + (el.La  +
                                                                     ( extra_fact * apL + TUBE_WALL_THICKNESS)
                                                                     * np.sin(el.ang)) * np.sin(el.ang)])
    point8 = np.array([el.Lb, -extra_fact * apL - TUBE_WALL_THICKNESS])  # bottom middle when theta=0
    point9 = np.array([el.Lb, -half_width])  # bottom middle when theta=0
    point10 = np.array([el.space, -half_width])  # bottom middle when theta=0
    point11 = np.array([el.space, -apL - TUBE_WALL_THICKNESS])  # bottom middle when theta=0
    point12 = np.array([0, -apL - TUBE_WALL_THICKNESS])  # bottom left when theta=0
    pointsOuter = [point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11,
                   point12]
    for i, point in enumerate(pointsOuter):
        pointsOuter[i] = el.R_Out @ point + el.r2[:2]
    return pointsOuter

def make_Halbach_Combiner_Inner_Points(el: Element) -> list[np.ndarray]:
    assert type(el) is CombinerHalbachLensSim
    LbVac = el.Lb if type(el) is CombinerIdeal else el.Lb + FLAT_WALL_VACUUM_THICKNESS
    ap,acceptance_width=el.ap,el.acceptance_width
    
    m = np.tan(el.ang)
    assert type(el) is CombinerHalbachLensSim
    yT = lambda x: m * x + (el.acceptance_width - m * el.Lb)  # upper limit
    YR = lambda x: (-1 / m) * x + el.La * np.sin(el.ang) + (el.Lb + el.La * np.cos(el.ang)) / m
    YB = lambda x: m * x + (-el.acceptance_width - m * el.Lb)

    q1Inner = np.asarray([0, ap])  # top left ( in standard xy plane) when theta=0
    q2Inner = np.asarray([LbVac, ap])  # top middle when theta=0

    q3Inner = np.asarray([LbVac, yT(LbVac)])  # top middle when theta=0

    q4Inner = np.asarray([el.Lb + (el.La - acceptance_width * np.sin(el.ang)) * np.cos(el.ang),
                          acceptance_width + (el.La - acceptance_width * np.sin(el.ang)) * np.sin(
                              el.ang)])  # top right when theta=0

    q5Inner = np.asarray([el.Lb + (el.La + acceptance_width * np.sin(el.ang)) * np.cos(el.ang),
                          -acceptance_width + (el.La + acceptance_width * np.sin(el.ang)) * np.sin(
                              el.ang)])  # bottom right when theta=0

    q6Inner = np.asarray([LbVac, YB(LbVac)])  # bottom middle when theta=0

    q7Inner = np.asarray([LbVac, -ap])  # bottom middle when theta=0
    q8Inner = np.asarray([0, -ap])  # bottom left when theta=0
    pointsInner = [q1Inner, q2Inner, q3Inner, q4Inner, q5Inner, q6Inner, q7Inner, q8Inner]
    for point in pointsInner:
        point[:] = el.R_Out @ point + el.r2[:2]
    return pointsInner

def make_Combiner_Inner_Points(el: Element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the inner (vacuum tube) geometry of a halbach combiner.
    Basically a rectangle with a wider tilted rectangle coming off one end (the input)"""

    #todo: this doesn't work with combiner ideal or sim
    

    assert type(el) in (CombinerIdeal, CombinerSim)
    
    apR, apL = el.apR, el.apL
    
    

    q1Inner = np.asarray([0, apR])  # top left ( in standard xy plane) when theta=0
    q2Inner = np.asarray([el.Lb, apR])  # top middle when theta=0
    q3Inner = np.asarray([el.Lb + (el.La - apR * np.sin(el.ang)) * np.cos(el.ang),
                          apR + (el.La - apR * np.sin(el.ang)) * np.sin(el.ang)])  # top right when theta=0

    q4Inner = np.asarray([el.Lb + (el.La +  apL * np.sin(el.ang)) * np.cos(el.ang),
                          - apL + (el.La +  apL * np.sin(el.ang)) * np.sin(
                              el.ang)])  # bottom right when theta=0
    q5Inner = np.asarray([el.Lb, -apL])  # bottom middle when theta=0
    q6Inner = np.asarray([0, -apL])  # bottom left when theta=0
    pointsInner = [q1Inner, q2Inner, q3Inner, q4Inner, q5Inner, q6Inner]
    for point in pointsInner:
        point[:] = el.R_Out @ point + el.r2[:2]
    return pointsInner


def make_Rectangle(L: float, halfWidth: float) -> list[np.ndarray]:
    """Make a simple rectangle. Used to model drift regions or interior(vacuum tube) of lenses"""
    q1Inner = np.asarray([0.0, halfWidth])  # top left when theta=0
    q2Inner = np.asarray([L, halfWidth])  # top right when theta=0
    q3Inner = np.asarray([L, -halfWidth])  # bottom right when theta=0
    q4Inner = np.asarray([0, -halfWidth])  # bottom left when theta=0
    return [q1Inner, q2Inner, q3Inner, q4Inner]


def make_Lens_Shapely_Objects(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of lens elements and drift
    region. Drift regions are just vacuum tubes between elements"""
    assert type(el) in (LensIdeal, HalbachLensSim)
    pointsInner = make_Rectangle(el.L, el.ap)
    if type(el) is HalbachLensSim:
        pointsOuter = make_Halbach_Lens_Outer_Points(el)
    else:
        pointsOuter = copy.deepcopy(pointsInner)
    for point in [*pointsInner, *pointsOuter]:
        point[:] = el.R_Out @ point + el.r1[:2]  # must modify the original array!
    return Polygon(pointsOuter), Polygon(pointsInner)


def is_Drift_Input_Output_Tilt_Valid(L: RealNum, halfWidth: RealNum, theta1: RealNum,
                                     theta2: RealNum) -> bool:
    """Check that te drift region, a trapezoid shape, is concave. theta1 and theta2 can violate this condition
    depending on their value"""
    m1, m2 = tan(pi / 2 + theta1), tan(pi / 2 + theta2)
    yCross = np.inf if theta1 == theta2 == 0.0 else m1 * m2 * L / (m2 - m1 + SMALL_NUMBER)
    return not (abs(yCross) < halfWidth or abs(theta1) > pi / 2 or abs(theta2) > pi / 2)


def make_Trapezoid_Points(L, theta1, theta2, halfWidth):
    """Make the coordinates of the vertices of the trapezoid."""
    assert is_Drift_Input_Output_Tilt_Valid(L, halfWidth, theta1, theta2)
    points = [np.array([-halfWidth * tan(theta1), halfWidth]),
              np.array([L - halfWidth * tan(theta2), halfWidth]),
              np.array([L + halfWidth * tan(theta2), -halfWidth]),
              np.array([halfWidth * tan(theta1), -halfWidth])]
    return points


def make_Drift_Shapely_Objects(el: Drift):
    """Make shapely objects for drift element. Drift element is trapezoid shaped to allow for tilted input/output"""
    L, ap, outer_half_width, theta1, theta2 = el.L, el.ap, el.outer_half_width, el.inputTiltAngle, el.outputTiltAngle
    pointsInner = make_Trapezoid_Points(L, theta1, theta2, ap)
    pointsOuter = make_Trapezoid_Points(L, theta1, theta2, outer_half_width)
    for point in [*pointsInner, *pointsOuter]:
        point[:] = el.R_Out @ point + el.r1[:2]  # must modify the original array!
    return Polygon(pointsOuter), Polygon(pointsInner)


def make_Bender_Shapely_Object(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of bender elements"""

    assert type(el) in (BenderIdeal, HalbachBenderSimSegmented)
    halfWidth = el.ap
    theta = el.theta
    phiArr = np.linspace(0, -el.ang, BENDER_POINTS) + theta + pi / 2  # angles swept out
    r0 = el.r0.copy()
    xInner = (el.rb - halfWidth) * np.cos(phiArr) + r0[0]  # x values for inner bend
    yInner = (el.rb - halfWidth) * np.sin(phiArr) + r0[1]  # y values for inner bend
    xOuter = np.flip((el.rb + halfWidth) * np.cos(phiArr) + r0[0])  # x values for outer bend
    yOuter = np.flip((el.rb + halfWidth) * np.sin(phiArr) + r0[1])  # y values for outer bend

    if isinstance(el, HalbachBenderSimSegmented):
        xInner = np.append(xInner[0] + el.nb[0] * el.Lcap, xInner)
        yInner = np.append(yInner[0] + el.nb[1] * el.Lcap, yInner)
        xInner = np.append(xInner, xInner[-1] + el.ne[0] * el.Lcap)
        yInner = np.append(yInner, yInner[-1] + el.ne[1] * el.Lcap)
        xOuter = np.append(xOuter, xOuter[-1] + el.nb[0] * el.Lcap)
        yOuter = np.append(yOuter, yOuter[-1] + el.nb[1] * el.Lcap)
        xOuter = np.append(xOuter[0] + el.ne[0] * el.Lcap, xOuter)
        yOuter = np.append(yOuter[0] + el.ne[1] * el.Lcap, yOuter)

    x = np.append(xInner, xOuter)  # list of x values in order
    y = np.append(yInner, yOuter)  # list of y values in order
    pointsInner = np.column_stack((x, y))  # shape the coordinates and make the object
    if type(el) is BenderIdeal:
        pointsOuter = copy.deepcopy(pointsInner)
    elif type(el) is HalbachBenderSimSegmented:
        pointsOuter = make_Hexapole_Bender_Outer_Points(el)
    else:
        raise NotImplementedError
    return Polygon(pointsOuter), Polygon(pointsInner)


def make_Combiner_Shapely_Object(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of combiner elements"""
    assert type(el) in (CombinerIdeal, CombinerSim, CombinerHalbachLensSim)

    if type(el) in (CombinerIdeal, CombinerSim):
        pointsInner=make_Combiner_Inner_Points(el)
        pointsOuter = copy.deepcopy(pointsInner)
    elif type(el) is CombinerHalbachLensSim:
        pointsInner=make_Halbach_Combiner_Inner_Points(el)
        pointsOuter = make_Hexapole_Combiner_Outer_Points(el)
    else:
        raise NotImplementedError
    return Polygon(pointsOuter), Polygon(pointsInner)


def make_Element_Shapely_Object(el: Element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of elementPT objects such
    as lenses, drifts, benders and combiners"""
    if type(el) in (HalbachLensSim, LensIdeal):
        shapelyOuter, shapelyInner = make_Lens_Shapely_Objects(el)
    elif type(el) in (BenderIdeal, HalbachBenderSimSegmented):
        shapelyOuter, shapelyInner = make_Bender_Shapely_Object(el)
    elif type(el) in (CombinerIdeal, CombinerSim, CombinerHalbachLensSim):
        shapelyOuter, shapelyInner = make_Combiner_Shapely_Object(el)
    elif type(el) is Drift:
        shapelyOuter, shapelyInner = make_Drift_Shapely_Objects(el)
    else:
        raise NotImplementedError
    assert shapelyOuter.buffer(1e-9).contains(shapelyInner)  # inner object must be smaller or same size as outer.
    # add a small buffer for numerical issues
    return shapelyOuter, shapelyInner


def build_Shapely_Objects(elementList: list[Element]) -> None:
    """Build the inner and outer shapely obejcts that represent the 2D geometries of the each element. This is the
    projection of an element onto the horizontal plane that bisects it. Inner geometry repsents vacuum, and outer
     the eternal profile. Ideal elements have to out profile"""

    #todo: there is a mild disagreement here between shapely and is inside method with the combiner

    for el in elementList:
        shapelyObject_Outer, shapelyObject_Inner = make_Element_Shapely_Object(el)
        el.SO = shapelyObject_Inner
        el.SO_outer = shapelyObject_Outer
