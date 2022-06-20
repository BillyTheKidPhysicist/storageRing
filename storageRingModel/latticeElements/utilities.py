from math import atan2

import numpy as np

from constants import SPACE_BETWEEN_MAGNETS_IN_MOUNT
from typeHints import FloatTuple, RealNumber

TINY_STEP = 1e-9
TINY_OFFSET = 1e-12  # tiny offset to avoid out of bounds right at edges of element
SMALL_OFFSET = 1e-9  # small offset to avoid out of bounds right at edges of element
MAGNET_ASPECT_RATIO = 4  # length of individual neodymium magnet relative to width of magnet


def halbach_Magnet_Width(rp: RealNumber, magnetSeparation: RealNumber = SPACE_BETWEEN_MAGNETS_IN_MOUNT) -> RealNumber:
    assert rp > 0.0
    halfAngle = 2 * np.pi / 24
    maxMagnetWidth = rp * np.tan(halfAngle) * 2
    widthReductin = magnetSeparation / np.cos(halfAngle)
    magnetWidth = maxMagnetWidth - widthReductin
    return magnetWidth


def get_Halbach_Layers_Radii_And_Magnet_Widths(rp_first: RealNumber, numConcentricLayers: int,
                                               magnetSeparation: RealNumber = SPACE_BETWEEN_MAGNETS_IN_MOUNT) -> tuple[
    FloatTuple, FloatTuple]:
    """Given a starting bore radius, construct the maximum magnet widths to build the specified number of concentric
    layers"""
    assert rp_first > 0.0 and isinstance(numConcentricLayers, int)
    rpLayers = []
    magnetWidths = []
    for _ in range(numConcentricLayers):
        next_rpLayer = rp_first + sum(magnetWidths)
        rpLayers.append(next_rpLayer)
        nextMagnetWidth = halbach_Magnet_Width(next_rpLayer, magnetSeparation=magnetSeparation)
        magnetWidths.append(nextMagnetWidth)
    return tuple(rpLayers), tuple(magnetWidths)


def max_Tube_Radius_In_Segmented_Bend(rb: float, rp: float, Lm: float, tubeWallThickness: float) -> float:
    """What is the maximum size that will fit in a segmented bender and respect the geometry"""
    assert rb > 0.0 and 0.0 < rp < rb and Lm > 0.0 and 0 <= tubeWallThickness < rp
    radiusCorner = np.sqrt((rb - rp) ** 2 + (Lm / 2) ** 2)
    maximumTubeRadius = rb - radiusCorner - tubeWallThickness
    assert maximumTubeRadius > 0.0
    return maximumTubeRadius


def full_Arctan(q: np.ndarray):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    assert len(q) == 3 and q.ndim == 1
    phi = atan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


def get_Unit_Cell_Angle(lengthSegment: float, radius: float, segmentWidth: float) -> float:
    """Get the arc angle associate with a single unit cell. Each lens contains two unit cells."""
    assert lengthSegment > 0.0 and radius > 0.0 and radius > segmentWidth >= 0.0
    return np.arctan(.5 * lengthSegment / (radius - segmentWidth))  # radians


def is_Even(x: int) -> bool:
    """Test if a number is even"""
    assert type(x) is int and x > 0
    return True if x % 2 == 0 else False


def mirror_Across_Angle(x: RealNumber, y: RealNumber, ang: RealNumber) -> tuple[float, float]:
    """mirror_Across_Angle x and y across a line at angle "ang" that passes through the origin"""
    m = np.tan(ang)
    d = (x + y * m) / (1 + m ** 2)
    xMirror = 2 * d - x
    yMirror = 2 * d * m - y
    return xMirror, yMirror


class ElementDimensionError(Exception):
    """Some dimension of an element is causing an unphysical configuration. Rather general error"""


class ElementTooShortError(Exception):
    """An element is too short. Because space is required for fringe fields this can result in negative material
    lengths, or nullify my approximation that fields drop to 1% when the element ends."""


class CombinerIterExceededError(Exception):
    """When solving for the geometry of the combiner, Newton's method is used to set the offset. Throw this if
    iterations are exceeded"""


class CombinerDimensionError(Exception):
    """Not all configurations of combiner parameters are valid. For one thing, the beam needs to fit into the
    combiner."""
