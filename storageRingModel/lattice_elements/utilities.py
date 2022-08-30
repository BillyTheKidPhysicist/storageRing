from math import atan2, floor

import numpy as np

from constants import SPACE_BETWEEN_MAGNETS_IN_MOUNT, SIMULATION_MAGNETON
from helper_tools import inch_to_meter
from type_hints import FloatTuple, RealNum, ndarray

TINY_STEP = 1e-9
TINY_OFFSET = 1e-12  # tiny offset to avoid out of bounds right at edges of element
SMALL_OFFSET = 1e-9  # small offset to avoid out of bounds right at edges of element
MAGNET_ASPECT_RATIO = 2  # length of individual neodymium magnet relative to width of magnet

B_GRAD_STEP_SIZE = 1e-7
INTERP_MAGNET_MATERIAL_OFFSET = 1.5 * B_GRAD_STEP_SIZE
TINY_INTERP_STEP = 1e-12

INTERP_ROUNDING_GUARD: float = 1e-12


def round_down_to_imperial(value: RealNum) -> float:
    """Round 'value' down to its nearest imperial multiple of 1/16 inch"""
    min_mult = inch_to_meter(1 / 16)
    assert min_mult <= value
    multiples = value / min_mult
    multiples += 1e-12  # when given the exact value, the function can round down because of precision issues, so add
    # a small number
    value_rounded = min_mult * floor(multiples)
    return value_rounded


def round_down_to_nearest_valid_mag_width(width_proposed: RealNum) -> float:
    """Given a proposed magnet width, round down the nearest available width (in L x width x width)"""
    width_rounded = round_down_to_imperial(width_proposed)
    assert width_rounded <= inch_to_meter(1.5)
    return width_rounded


def round_down_to_nearest_tube_OD(OD_proposed: RealNum) -> float:
    """Given a proposed tube OD, round down the nearest available OD """
    return round_down_to_imperial(OD_proposed)


def halbach_magnet_width(rp: RealNum, magnetSeparation: RealNum = SPACE_BETWEEN_MAGNETS_IN_MOUNT,
                         use_standard_sizes: bool = False) -> RealNum:
    assert rp > 0.0
    half_angle = 2 * np.pi / 24
    max_magnet_width = rp * np.tan(half_angle) * 2
    width_reduction = magnetSeparation / np.cos(half_angle)
    magnet_width = max_magnet_width - width_reduction
    magnet_width = round_down_to_nearest_valid_mag_width(magnet_width) if use_standard_sizes else magnet_width
    return magnet_width


def get_halbach_layers_radii_and_magnet_widths(rp_first: RealNum, numConcentricLayers: int,
                                               magnetSeparation: RealNum = SPACE_BETWEEN_MAGNETS_IN_MOUNT,
                                               use_standard_sizes: bool = False) -> tuple[FloatTuple, FloatTuple]:
    """Given a starting bore radius, construct the maximum magnet widths to build the specified number of concentric
    layers"""
    assert rp_first > 0.0 and isinstance(numConcentricLayers, int)
    rp_layers = []
    magnet_widths = []
    for _ in range(numConcentricLayers):
        next_rp_layer = rp_first + sum(magnet_widths)
        rp_layers.append(next_rp_layer)
        next_magnet_width = halbach_magnet_width(next_rp_layer, magnetSeparation=magnetSeparation,
                                                 use_standard_sizes=use_standard_sizes)
        magnet_widths.append(next_magnet_width)
    return tuple(rp_layers), tuple(magnet_widths)


def max_tube_OR_in_segmented_bend(rb: float, rp: float, Lm: float, use_standard_sizes: bool = False) -> float:
    """What is the maximum size that will fit in a segmented bender and respect the geometry"""
    assert rb > 0.0 and 0.0 < rp < rb and Lm > 0.0
    radius_corner = np.sqrt((rb - rp) ** 2 + (Lm / 2) ** 2)
    max_tube_OR = rb - radius_corner  # outside radius
    max_standard_tube_OR = round_down_to_nearest_tube_OD(2 * max_tube_OR) / 2.0
    max_tube_OR = max_standard_tube_OR if use_standard_sizes else max_tube_OR
    return max_tube_OR


def max_tube_IR_in_segmented_bend(rb: float, rp: float, Lm: float, tube_wall_thickness: float,
                                  use_standard_sizes: bool = False) -> float:
    "Maximum geometry limited inside radius of a vacuum tube going through the segmented bender"
    assert 0 <= tube_wall_thickness < rp
    max_tube_OR = max_tube_OR_in_segmented_bend(rb, rp, Lm, use_standard_sizes=use_standard_sizes)
    max_tube_IR = max_tube_OR - tube_wall_thickness
    return max_tube_IR


def min_Bore_Radius_From_Tube_OD(tube_OD: float, rb: float, Lm: float) -> float:
    rp = rb - np.sqrt((rb - tube_OD / 2.0) ** 2 - (Lm / 2) ** 2)  # geometry of beinding radius minus
    # radius of corner of magnet
    return rp


def full_arctan2(q: np.ndarray):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    assert len(q) == 3 and q.ndim == 1
    phi = atan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


def calc_unit_cell_angle(length_seg: float, radius: float, segmentWidth: float) -> float:
    """Get the arc angle associate with a single unit cell. Each lens contains two unit cells."""
    assert length_seg > 0.0 and radius > 0.0 and radius > segmentWidth >= 0.0
    return np.arctan(.5 * length_seg / (radius - segmentWidth))  # radians


def is_even(x: int) -> bool:
    """Test if a number is even"""
    assert type(x) is int and x > 0
    return True if x % 2 == 0 else False


def mirror_across_angle(x: RealNum, y: RealNum, ang: RealNum) -> tuple[float, float]:
    """mirror_across_angle x and y across a line at angle "ang" that passes through the origin"""
    m = np.tan(ang)
    d = (x + y * m) / (1 + m ** 2)
    x_mirror = 2 * d - x
    y_mirror = 2 * d * m - y
    return x_mirror, y_mirror


def shape_field_data_3D(data: ndarray) -> tuple[ndarray, ...]:
    """
    Shape 3D field data for fast linear interpolation method

    Take an array with the shape (n,7) where n is the number of points in space. Each row
    must have the format [x,y,z,gradxB,gradyB,gradzB,B] where B is the magnetic field norm at x,y,z and grad is the
    partial derivative. The data must be from a 3D grid of points with no missing points or any other funny business
    and the order of points doesn't matter. Return arrays are raveled for use by fast interpolater

    :param data: (n,7) numpy array of points originating from a 3d grid
    :return: tuple of 7 arrays, first 3 are grid edge coords (x,y,z) and last 4 are flattened field values
    (Fx,Fy,Fz,V)
    """
    assert data.shape[1] == 7 and len(data) > 2 ** 3
    x_arr = np.unique(data[:, 0])
    y_arr = np.unique(data[:, 1])
    z_arr = np.unique(data[:, 2])
    assert all(not np.any(np.isnan(arr)) for arr in (x_arr, y_arr, z_arr))

    num_x = x_arr.shape[0]
    num_y = y_arr.shape[0]
    num_z = z_arr.shape[0]
    Fx_matrix = np.empty((num_x, num_y, num_z))
    Fy_matrix = np.empty((num_x, num_y, num_z))
    Fz_matrix = np.empty((num_x, num_y, num_z))
    V_matrix = np.zeros((num_x, num_y, num_z))
    x_indices = np.argwhere(data[:, 0][:, None] == x_arr)[:, 1]
    y_indices = np.argwhere(data[:, 1][:, None] == y_arr)[:, 1]
    z_indices = np.argwhere(data[:, 2][:, None] == z_arr)[:, 1]
    Fx_matrix[x_indices, y_indices, z_indices] = -SIMULATION_MAGNETON * data[:, 3]
    Fy_matrix[x_indices, y_indices, z_indices] = -SIMULATION_MAGNETON * data[:, 4]
    Fz_matrix[x_indices, y_indices, z_indices] = -SIMULATION_MAGNETON * data[:, 5]
    V_matrix[x_indices, y_indices, z_indices] = SIMULATION_MAGNETON * data[:, 6]
    V_Flat, Fx_Flat, Fy_Flat, Fz_Flat = V_matrix.ravel(), Fx_matrix.ravel(), Fy_matrix.ravel(), Fz_matrix.ravel()
    return x_arr, y_arr, z_arr, Fx_Flat, Fy_Flat, Fz_Flat, V_Flat


def shape_field_data_2D(data: ndarray) -> tuple[ndarray, ...]:
    """2D version of shape_field_data_3D. Data must be shape (n,5), with each row [x,y,Fx,Fy,V]"""
    assert data.shape[1] == 5 and len(data) > 2 ** 3
    x_arr = np.unique(data[:, 0])
    y_arr = np.unique(data[:, 1])
    num_x = x_arr.shape[0]
    num_y = y_arr.shape[0]
    B_grad_x_matrix = np.zeros((num_x, num_y))
    B_grad_y_matrix = np.zeros((num_x, num_y))
    B0_matrix = np.zeros((num_x, num_y))
    x_indices = np.argwhere(data[:, 0][:, None] == x_arr)[:, 1]
    y_indices = np.argwhere(data[:, 1][:, None] == y_arr)[:, 1]

    B_grad_x_matrix[x_indices, y_indices] = data[:, 2]
    B_grad_y_matrix[x_indices, y_indices] = data[:, 3]
    B0_matrix[x_indices, y_indices] = data[:, 4]
    Fx_matrix = -SIMULATION_MAGNETON * B_grad_x_matrix
    FyMatrix = -SIMULATION_MAGNETON * B_grad_y_matrix
    V_matrix = SIMULATION_MAGNETON * B0_matrix
    V_Flat, Fx_Flat, Fy_Flat = np.ravel(V_matrix), np.ravel(Fx_matrix), np.ravel(FyMatrix)
    return x_arr, y_arr, Fx_Flat, Fy_Flat, V_Flat


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
