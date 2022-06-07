"""
Functions to construct shapely geometry objects for elements in ElementPT.py. These shapely objects are primarily
used for:
- enforcing spatial constraints such as elements not overlapping and the ring fitting in the room
- visualizing the floorplan layout
- a secondary method of testing if particles are in a specific element
"""
import copy
from math import pi, tan
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from shapely.geometry import Polygon

import elementPT
from constants import FLAT_WALL_VACUUM_THICKNESS, VACUUM_TUBE_THICKNESS
from typeHints import element

BENDER_POINTS = 250  # how many points to represent the bender with along each curve
realNum = Union[float, int]
SMALL_NUMBER = 1e-16


def make_Halbach_Lens_Outer_Points(el: element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a halbach lens. Overall shape is
    a rectangle overlayd on a shorter but wider rectangle. This represents the width of the magnets"""
    assert type(el) is elementPT.HalbachLensSim
    halfWidth = el.outerHalfWidth
    vacuumTubeOuterWidth = el.ap + VACUUM_TUBE_THICKNESS
    fringeLength = el.fringeFieldLength
    point1 = np.asarray([0.0, vacuumTubeOuterWidth])
    point2 = np.asarray([fringeLength, vacuumTubeOuterWidth])
    point3 = np.asarray([fringeLength, halfWidth])
    point4 = np.asarray([el.L - fringeLength, halfWidth])
    point5 = np.asarray([el.L - fringeLength, vacuumTubeOuterWidth])
    point6 = np.asarray([el.L, vacuumTubeOuterWidth])
    topPoints = [point1, point2, point3, point4, point5, point6]

    bottomPoints = np.flip(np.row_stack(topPoints), axis=0)  # points need to go clockwise
    bottomPoints[:, -1] *= -1
    pointsOuter = [*topPoints, *bottomPoints]
    return pointsOuter


def make_Hexapole_Bender_Caps_Outer_Points(el: element) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Make points that describe the shape of the input and outputs of the hexapole bender. They have
    a stepped shape from the width of the magnets. Output cap points along -y and is easy to specify the
    coordinates and is then mirrored to produce the input cap as well."""

    vacuumTubeOuterWidth = el.ap + VACUUM_TUBE_THICKNESS
    pointsCapStart = [np.array([el.rb - el.outerHalfWidth, -el.Lm / 2.0]),
                      np.array([el.rb - vacuumTubeOuterWidth, -el.Lm / 2.0]),
                      np.array([el.rb - vacuumTubeOuterWidth, -el.Lcap]),
                      np.array([el.rb + vacuumTubeOuterWidth, -el.Lcap]),
                      np.array([el.rb + vacuumTubeOuterWidth, -el.Lm / 2.0]),
                      np.array([el.rb + el.outerHalfWidth, -el.Lm / 2.0])]

    pointsCapEnd = []
    m = np.tan(el.ang / 2.0)
    for point in pointsCapStart:
        xStart, yStart = point
        d = (xStart + yStart * m) / (1 + m ** 2)
        pointEnd = np.array([2 * d - xStart, 2 * d * m - yStart])
        pointsCapEnd.append(pointEnd)
    return pointsCapStart, pointsCapEnd


def make_Hexapole_Bender_Outer_Points(el: element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a hexapole bending section.
    Shape is a toroid with short straight section at input/ouput, with another wider but shorter toroid ontop """

    assert type(el) is elementPT.HalbachBenderSimSegmented
    phiArr = np.linspace(el.ang, 0.0, BENDER_POINTS)  # + el.theta + np.pi / 2  # angles swept out

    xInner = (el.rb - el.outerHalfWidth) * np.cos(phiArr)  # x values for inner bend
    yInner = (el.rb - el.outerHalfWidth) * np.sin(phiArr)  # y values for inner bend
    xOuter = np.flip((el.rb + el.outerHalfWidth) * np.cos(phiArr))  # x values for outer bend
    yOuter = np.flip((el.rb + el.outerHalfWidth) * np.sin(phiArr))  # y values for outer bend

    pointsCapStart, pointsCapEnd = make_Hexapole_Bender_Caps_Outer_Points(el)
    pointsCapStart, pointsCapEnd = np.array(pointsCapStart), np.array(pointsCapEnd)
    xInner = np.append(np.flip(pointsCapEnd[:, 0]), xInner)
    yInner = np.append(np.flip(pointsCapEnd[:, 1]), yInner)
    xInner = np.append(xInner, pointsCapStart[:, 0])
    yInner = np.append(yInner, pointsCapStart[:, 1])
    x = np.append(xInner, xOuter)  # list of x values in order
    y = np.append(yInner, yOuter)  # list of y values in order
    pointsOuter = np.column_stack((x, y))  # shape the coordinates and make the object
    rotMatrix2D = Rot.from_rotvec([0, 0, el.theta - el.ang + pi / 2]).as_matrix()[:2, :2]
    for i, point in enumerate(pointsOuter):
        pointsOuter[i] = rotMatrix2D @ point
    pointsOuter += el.r0[:2]
    return pointsOuter


def make_Hexapole_Combiner_Outer_Points(el: element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the outer geometry of a halbach combiner. Very similiar
    to halbach lens geometry, but with tiled input and enlarged input"""
    # pylint: disable=too-many-locals
    assert type(el) is elementPT.CombinerHalbachLensSim
    apR, apL = el.ap, el.ap
    extraFact = 1.5
    halfWidth = el.outerHalfWidth
    point1 = np.array([0, apR + VACUUM_TUBE_THICKNESS])  # top left ( in standard xy plane) when theta=0
    point2 = np.array([el.space, apR + VACUUM_TUBE_THICKNESS])  # top left ( in standard xy plane) when theta=0
    point3 = np.array([el.space, halfWidth])  # top left ( in standard xy plane) when theta=0
    point4 = np.array([el.Lb, halfWidth])  # top middle when theta=0
    point5 = np.array([el.Lb, apR + VACUUM_TUBE_THICKNESS])  # top middle when theta=0
    point6 = np.array([el.Lb + (el.La - (apR + VACUUM_TUBE_THICKNESS) * np.sin(el.ang)) * np.cos(el.ang),
                       (apR + VACUUM_TUBE_THICKNESS) + (
                               el.La - (apR + VACUUM_TUBE_THICKNESS) * np.sin(el.ang)) * np.sin(el.ang)])
    point7 = np.array([el.Lb + (el.La + extraFact * (apL + VACUUM_TUBE_THICKNESS) * np.sin(el.ang)) * np.cos(el.ang),
                       -extraFact * (apL + VACUUM_TUBE_THICKNESS) + ((el.La + VACUUM_TUBE_THICKNESS) + extraFact *
                                                                     (apL + VACUUM_TUBE_THICKNESS)
                                                                     * np.sin(el.ang)) * np.sin(el.ang)])
    point8 = np.array([el.Lb, -extraFact * (apL + VACUUM_TUBE_THICKNESS)])  # bottom middle when theta=0
    point9 = np.array([el.Lb, -halfWidth])  # bottom middle when theta=0
    point10 = np.array([el.space, -halfWidth])  # bottom middle when theta=0
    point11 = np.array([el.space, -apL - VACUUM_TUBE_THICKNESS])  # bottom middle when theta=0
    point12 = np.array([0, -apL - VACUUM_TUBE_THICKNESS])  # bottom left when theta=0
    pointsOuter = [point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11,
                   point12]
    for i, point in enumerate(pointsOuter):
        pointsOuter[i] = el.ROut @ point + el.r2[:2]
    return pointsOuter


def make_Any_Combiner_Inner_Points(el: element) -> list[np.ndarray]:
    """Construct a list of points of coordinates of corners of the inner (vacuum tube) geometry of a halbach combiner.
    Basically a rectangle with a wider tilted rectangle coming off one end (the input)"""
    assert type(el) in (elementPT.CombinerHalbachLensSim, elementPT.CombinerIdeal, elementPT.CombinerSim)
    LbVac = el.Lb if type(el) is elementPT.CombinerIdeal else el.Lb + FLAT_WALL_VACUUM_THICKNESS
    apR, apL = (el.apR, el.apL) if el.shape == 'COMBINER_SQUARE' else (el.ap, el.ap)
    extraFact = 1.5 if type(el) is elementPT.CombinerHalbachLensSim else 1.0
    q1Inner = np.asarray([0, apR])  # top left ( in standard xy plane) when theta=0
    q2Inner = np.asarray([LbVac, apR])  # top middle when theta=0
    q3Inner = np.asarray([el.Lb + (el.La - apR * np.sin(el.ang)) * np.cos(el.ang),
                          apR + (el.La - apR * np.sin(el.ang)) * np.sin(el.ang)])  # top right when theta=0
    q4Inner = np.asarray([el.Lb + (el.La + extraFact * apL * np.sin(el.ang)) * np.cos(el.ang),
                          -extraFact * apL + (el.La + extraFact * apL * np.sin(el.ang)) * np.sin(
                              el.ang)])  # bottom right when theta=0
    q5Inner = np.asarray([LbVac, -extraFact * apL])  # bottom middle when theta=0
    q6Inner = np.asarray([LbVac, -apL])  # bottom middle when theta=0
    q7Inner = np.asarray([0, -apL])  # bottom left when theta=0
    pointsInner = [q1Inner, q2Inner, q3Inner, q4Inner, q5Inner, q6Inner, q7Inner]
    for point in pointsInner:
        point[:] = el.ROut @ point + el.r2[:2]
    return pointsInner


def make_Rectangle(L: float, halfWidth: float) -> list[np.ndarray]:
    """Make a simple rectangle. Used to model drift regions or interior(vacuum tube) of lenses"""
    q1Inner = np.asarray([0.0, halfWidth])  # top left when theta=0
    q2Inner = np.asarray([L, halfWidth])  # top right when theta=0
    q3Inner = np.asarray([L, -halfWidth])  # bottom right when theta=0
    q4Inner = np.asarray([0, -halfWidth])  # bottom left when theta=0
    return [q1Inner, q2Inner, q3Inner, q4Inner]


def make_Lens_Shapely_Objects(el: element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of lens elements and drift
    region. Drift regions are just vacuum tubes between elements"""
    assert type(el) in (elementPT.LensIdeal, elementPT.HalbachLensSim)
    pointsInner = make_Rectangle(el.L, el.ap)
    if type(el) is elementPT.HalbachLensSim:
        pointsOuter = make_Halbach_Lens_Outer_Points(el)
    else:
        pointsOuter = copy.deepcopy(pointsInner)
    for point in [*pointsInner, *pointsOuter]:
        point[:] = el.ROut @ point + el.r1[:2]  # must modify the original array!
    return Polygon(pointsOuter), Polygon(pointsInner)


def is_Drift_Input_Output_Tilt_Valid(L: realNum, halfWidth: realNum, theta1: realNum, theta2: realNum) -> bool:
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


def make_Drift_Shapely_Objects(el: elementPT.Drift):
    """Make shapely objects for drift element. Drift element is trapezoid shaped to allow for tilted input/output"""
    L, ap, outerHalfWidth, theta1, theta2 = el.L, el.ap, el.outerHalfWidth, el.inputTiltAngle, el.outputTiltAngle
    pointsInner = make_Trapezoid_Points(L, theta1, theta2, ap)
    pointsOuter = make_Trapezoid_Points(L, theta1, theta2, outerHalfWidth)
    for point in [*pointsInner, *pointsOuter]:
        point[:] = el.ROut @ point + el.r1[:2]  # must modify the original array!
    return Polygon(pointsOuter), Polygon(pointsInner)


def make_Bender_Shapely_Object(el: element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of bender elements"""

    assert type(el) in (elementPT.BenderIdeal, elementPT.HalbachBenderSimSegmented)
    halfWidth = el.ap
    theta = el.theta
    phiArr = np.linspace(0, -el.ang, BENDER_POINTS) + theta + pi / 2  # angles swept out
    r0 = el.r0.copy()
    xInner = (el.rb - halfWidth) * np.cos(phiArr) + r0[0]  # x values for inner bend
    yInner = (el.rb - halfWidth) * np.sin(phiArr) + r0[1]  # y values for inner bend
    xOuter = np.flip((el.rb + halfWidth) * np.cos(phiArr) + r0[0])  # x values for outer bend
    yOuter = np.flip((el.rb + halfWidth) * np.sin(phiArr) + r0[1])  # y values for outer bend

    if isinstance(el, elementPT.HalbachBenderSimSegmented):
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
    if type(el) is elementPT.BenderIdeal:
        pointsOuter = copy.deepcopy(pointsInner)
    elif type(el) is elementPT.HalbachBenderSimSegmented:
        pointsOuter = make_Hexapole_Bender_Outer_Points(el)
    else:
        raise NotImplementedError
    return Polygon(pointsOuter), Polygon(pointsInner)


def make_Combiner_Shapely_Object(el: element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of combiner elements"""
    assert type(el) in (elementPT.CombinerIdeal, elementPT.CombinerSim, elementPT.CombinerHalbachLensSim)
    pointsInner = make_Any_Combiner_Inner_Points(el)
    if type(el) in (elementPT.CombinerIdeal, elementPT.CombinerSim):
        pointsOuter = copy.deepcopy(pointsInner)
    elif type(el) is elementPT.CombinerHalbachLensSim:
        pointsOuter = make_Hexapole_Combiner_Outer_Points(el)
    else:
        raise NotImplementedError
    return Polygon(pointsOuter), Polygon(pointsInner)


def make_Element_Shapely_Object(el: element) -> tuple[Polygon, Polygon]:
    """Make shapely object that represent the inner (vacuum) and outer (exterior profile) of elementPT objects such
    as lenses, drifts, benders and combiners"""
    if type(el) in (elementPT.HalbachLensSim, elementPT.LensIdeal):
        shapelyOuter, shapelyInner = make_Lens_Shapely_Objects(el)
    elif type(el) in (elementPT.BenderIdeal, elementPT.HalbachBenderSimSegmented):
        shapelyOuter, shapelyInner = make_Bender_Shapely_Object(el)
    elif type(el) in (elementPT.CombinerIdeal, elementPT.CombinerSim, elementPT.CombinerHalbachLensSim):
        shapelyOuter, shapelyInner = make_Combiner_Shapely_Object(el)
    elif type(el) is elementPT.Drift:
        shapelyOuter, shapelyInner = make_Drift_Shapely_Objects(el)
    else:
        raise NotImplementedError
    assert shapelyOuter.buffer(1e-12).contains(shapelyInner)  # inner object must be smaller or same size as outer.
    # add a small buffer for numerical issues
    return shapelyOuter, shapelyInner


def build_Shapely_Objects(elementList: list[element]) -> None:
    """Build the inner and outer shapely obejcts that represent the 2D geometries of the each element. This is the
    projection of an element onto the horizontal plane that bisects it. Inner geometry repsents vacuum, and outer
     the eternal profile. Ideal elements have to out profile"""
    for el in elementList:
        shapelyObject_Outer, shapelyObject_Inner = make_Element_Shapely_Object(el)
        el.SO = shapelyObject_Inner
        el.SO_Outer = shapelyObject_Outer
