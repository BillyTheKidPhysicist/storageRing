from typing import Union

import numpy as np

realNumber = (int, float)
lst_tup_arr = Union[list, tuple, np.ndarray]

TINY_STEP = 1e-9
TINY_OFFSET = 1e-12  # tiny offset to avoid out of bounds right at edges of element
SMALL_OFFSET = 1e-9  # small offset to avoid out of bounds right at edges of element
MAGNET_ASPECT_RATIO = 4  # length of individual neodymium magnet relative to width of magnet

ELEMENT_PLOT_COLORS: dict[str] = {'drift': 'grey', 'lens': 'magenta', 'combiner': 'blue', 'bender': 'black'}


def full_Arctan(q):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


def is_Even(x: int) -> bool:
    """Test if a number is even"""

    assert type(x) is int and x > 0
    return True if x % 2 == 0 else False


def mirror_Across_Angle(x: float, y: float, ang: float) -> tuple[float, float]:
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
