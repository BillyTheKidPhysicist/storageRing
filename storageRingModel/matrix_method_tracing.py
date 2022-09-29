import functools
from collections.abc import Sequence
from math import sqrt, cos, sin, cosh, sinh, tan, acos, atan2, isclose
from typing import Iterable, Union, Callable, Any

import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy.linalg import inv, det

from constants import SIMULATION_MAGNETON, SIMULATION_MASS, DEFAULT_ATOM_SPEED
from field_generators import Collection
from helper_tools import multiply_matrices
from lattice_elements.bender_sim import mirror_across_angle, speed_with_energy_correction
from lattice_elements.elements import Drift as Drift_Sim
from lattice_elements.elements import Element as SimElement
from lattice_elements.elements import HalbachLensSim, BenderSim, CombinerLensSim, BenderIdeal, LensIdeal
from lattice_elements.orbit_trajectories import combiner_orbit_coords_el_frame
from particle import Swarm, Particle
from particle_tracer_lattice import ParticleTracerLattice
from swarm_tracer import histogram_particle_survival
from type_hints import ndarray, RealNum, sequence

SMALL_ROUNDING_NUM: float = 1e-14
NUM_FRINGE_MAGNETS_MIN: int = 5  # number of fringe magnets to accurately model internal behaviour of bender
ORBIT_DIM_INDEX = {'x': 0, 'y': 1}

ThreeNumpyMatrices = tuple[ndarray, ndarray, ndarray]


# IMPROVEMENT: unify profile naming


def make_arrays_read_only(*arrays) -> None:
    """When using functools.lru_cache it is important to no edit mutable objects that are returned, otherwise the
    mutated object will be returned which can screw with results"""
    for arr in arrays:
        arr.flags.writeable = False


@numba.njit
def Mu_exit(Mu_total, Mu_entrance) -> ndarray:
    """Return transfer matrix for remaining portion of element given the first portion of the transfer matrix"""
    return Mu_total @ inv(Mu_entrance)


def sim_element_fingerprint(el: SimElement) -> tuple:
    """hashable fingerprint of an element. Used for memoizing"""
    if type(el) in (BenderSim, CombinerLensSim, HalbachLensSim):
        finger_print = (el.ap, el.rp, el.L, el.Lm, el.ang, el.magnet.magnet_grade)
    else:
        raise NotImplementedError
    return finger_print


class Memoize_Elements:
    """To save time, memorize the results of elements that are identical from the matrix elements perspective.
    Known as memoization"""
    max_cache_size = 16

    def __init__(self, el_func):
        self.el_func = el_func
        self.cache = {}

    def __call__(self, el):
        key = sim_element_fingerprint(el)
        if key in self.cache:
            return self.cache[key]
        else:
            result = self.el_func(el)
            self.cache[key] = result
            if len(self.cache) > self.max_cache_size:
                first_key = list(self.cache.keys())[0]
                self.cache.pop(first_key)
            return result


def length_hard_edge(el: SimElement) -> float:
    """Return the hard edge length of a simulation element"""
    if type(el) is BenderSim:
        L = el.ang * el.ro
    elif type(el) is HalbachLensSim:
        L = el.Lm
    else:
        raise NotImplementedError
    return L


def cumulative_path_length(coords: ndarray) -> ndarray:
    """Return the cumulative length of a path specified as an array of coordinates"""
    _, dim = coords.shape
    dr = coords[1:] - coords[:-1]
    dr = np.row_stack((np.zeros(dim), dr))
    dr = np.linalg.norm(dr, axis=1)
    path_length = np.cumsum(dr)
    return path_length


def make_s_vals(rp: float, s_max: float) -> ndarray:
    """Return array of position values along path"""
    num_s_points_per_bore_radius = 10
    num_s_points = round((s_max / rp) * num_s_points_per_bore_radius)
    s_min = 0.0
    s_vals = np.linspace(s_min, s_max, num_s_points)
    return s_vals


def total_momentum_kick(F_vals: ndarray, s_vals: ndarray) -> float:
    """Return total momentum kick from force along path defined by s_vals"""
    return np.trapz(F_vals, s_vals)


def effective_Bp(F_vals: ndarray, s_vals: ndarray, r_offset: float, L_hard_edge: float, rp: float) -> float:
    """Return an effective magnetic field at the pole face (Bp) that accounts for the fringin field for some degree
    by using the total transfered momentum"""
    delta_p = total_momentum_kick(F_vals, s_vals)
    K = -delta_p / (r_offset * L_hard_edge)
    Bp_effective = K * rp ** 2 / (2 * SIMULATION_MAGNETON)
    return Bp_effective


def Bp_effective_lens(el: SimElement) -> float:
    """Return the effective magnetic field at the pole face for a lens"""
    assert type(el) is HalbachLensSim
    magnets = el.magnet.magpylib_magnets_model(False, False)
    s_vals = make_s_vals(el.rp, el.L)
    r_offset = el.rp / 4.0
    x_vals = np.ones(len(s_vals)) * r_offset
    y_vals = np.zeros(len(s_vals))
    coords = np.column_stack((s_vals, x_vals, y_vals))
    Fx_vals = magnet_force(magnets, coords)[:, 1]
    L_hard_edge = length_hard_edge(el)
    Bp_effective = effective_Bp(Fx_vals, s_vals, r_offset, L_hard_edge, el.rp)
    return Bp_effective


def Bp_effective_bender(el: SimElement) -> float:
    """Return the effective magnetic field at the pole face for a lens"""
    assert type(el) is BenderSim
    magnets = el.magnet.magpylib_magnets_model(False)
    sc_max = 2 * el.L_cap + el.ang * el.rb
    sc_vals = make_s_vals(el.rp, sc_max)
    r_offset = el.rp / 4.0
    coords = np.array([el.convert_center_to_cartesian_coords(s, r_offset, 0.0) for s in sc_vals])
    unit_vecs_perp = np.array([xo_unit_vector_bender_el_frame(el, coord) for coord in coords])
    F_vals = magnet_force(magnets, coords)
    Fr_vals = np.array([np.dot(unit_vec, F) for unit_vec, F in zip(unit_vecs_perp, F_vals)])
    s_vals = cumulative_path_length(coords)
    L_hard_edge = length_hard_edge(el)
    Bp_effective = effective_Bp(Fr_vals, s_vals, r_offset, L_hard_edge, el.rp)
    return Bp_effective


@numba.njit
def index_in_increasing_arr(x: RealNum, arr: ndarray) -> int:
    """Return index of first value in array that is greater than 'x'"""
    return int(np.argmax(x < arr))


@numba.njit
def update_Mu_and_Mu_cum(Mu_previous, Mu_cum, Ku, L, R=None):
    """Update cumulative transfer matrix IN PLACE, and append a copy to a list"""
    Mu_slice = transfer_matrix(Ku, L, R=R)
    Mu_previous[:] = Mu_slice @ Mu_previous
    Mu_current = Mu_previous.copy()
    # now append a copy of the new array.
    Mu_cum.append(Mu_current)  # must come last


@numba.njit
def arr_float64_list():
    """Because numba is type sensitive, the type of empty lists must be known. It can often be inferred, but if not a
    contrived solution must be used"""
    temp = [np.ones((1, 1))]
    temp = temp[1:]
    return temp


def M_cum_index(s0: RealNum, s_vals: ndarray, s_max: RealNum, idx_max: int) -> int:
    """Return cumulative transfer matrix corresponding to position value 's0'"""
    if isclose(s0, s_max, abs_tol=1e-9):
        idx = idx_max
    elif isclose(s0, 0, abs_tol=1e-9):
        idx = 0
    else:
        idx = index_in_increasing_arr(s0, s_vals)
    return idx


def delta(atom_speed: RealNum) -> float:
    """Return value of delta"""
    return (atom_speed - DEFAULT_ATOM_SPEED) / DEFAULT_ATOM_SPEED


def cumulative_M_func(s_vals: ndarray, s_max: float, cumulatice_transfer_matrices: Callable) -> Callable:
    """Return a function that return the transfer matrix at a specified position in the element. Uses a saved list
    of cumulative transfer matrices to dramatically reduce computation time"""

    idx_max = len(s_vals) - 1

    def M_func(s0: RealNum, atom_speed: RealNum) -> ThreeNumpyMatrices:
        """Return x and y transfer matrix at a specified position. Because the element is modeled as a series of
        slices, there are only discrete values for the transfer matrix"""
        assert 0 <= s0 <= s_max
        M_cum_x, M_cum_y, M_cum_d = cumulatice_transfer_matrices(atom_speed)
        if isclose(s0, s_max, abs_tol=1e-9):
            idx = idx_max
        elif isclose(s0, 0, abs_tol=1e-9):
            idx = 0
        else:
            idx = index_in_increasing_arr(s0, s_vals)
        return M_cum_x[idx], M_cum_y[idx], M_cum_d[idx]

    return M_func


def make_atom_speed_kwarg_iterable(func) -> Callable:
    """Take a function that has atom_speed as a keyword argument, and make the function able to accept an iterable
    and a float for atom_speed. If atom_speed is a float, return the results as a tuple of floats. If atom_speed is
    an iterable, return the results as a tuple of tuples"""

    def wrapper(*args, **kwargs):
        if 'atom_speed' in kwargs and isinstance(kwargs['atom_speed'], Iterable):
            atom_speed_iterable = kwargs['atom_speed']
            vals = []
            for atom_speed in atom_speed_iterable:
                kwargs['atom_speed'] = atom_speed
                vals.append(func(*args, **kwargs))
            vals = tuple([np.array(val) for val in zip(*vals)])  # convert into arrays
            return vals
        else:
            return func(*args, **kwargs)

    return wrapper


@numba.njit()
def transfer_matrix(K: RealNum, L: RealNum, R=None) -> ndarray:
    """Build the 2x2 transfer matrix (ABCD matrix) for an element that represents a harmonic potential.
    Transfer matrix is for one dimension only

    unfortunately I use reverse K convention than I do in my thesis..
    """
    assert L >= 0.0
    if K == 0.0:
        A = 1.0
        B = L
        C = 0.0
        D = 1
    elif K > 0.0:
        phi = sqrt(K) * L
        A = cos(phi)
        B = sin(phi) / sqrt(K)
        C = -sqrt(K) * sin(phi)
        D = cos(phi)
    else:
        K = abs(K)
        psi = sqrt(K) * L
        A = cosh(psi)
        B = sinh(psi) / sqrt(K)
        C = sqrt(K) * sinh(psi)
        D = cosh(psi)
    if R is not None:
        if K == 0.0:
            assert R == np.inf
        phi = sqrt(K) * L
        if K == 0.0:
            E = F = 0.0
        else:
            E = 2 * (1 - cos(phi)) / (K * R)
            F = 2 * sin(phi) / (sqrt(K) * R)
        M = np.array([[A, B, E], [C, D, F], [0.0, 0.0, 1.0]])
    else:
        M = np.array([[A, B], [C, D]])
    return M


def magnet_force(magnets, coords):
    """Force in magnet at given point"""
    B_norm_grad = magnets.B_norm_grad(coords, use_approx=True, dx=1e-6, diff_method='forward')
    forces = -SIMULATION_MAGNETON * B_norm_grad
    return forces


def bender_orbit_radius_energy_correction(Bp: RealNum, rb: RealNum, rp: RealNum, atom_speed: RealNum) -> float:
    """Calculate the orbit radius for particle in a bender. This is balancing the centrifugal "force" with the magnet
    force, and also accounts for the small difference from energy conservation of particle slowing down when entering
    bender"""
    term1 = 3 * rb / 4.0
    term2 = Bp * SIMULATION_MAGNETON * rb ** 2
    term3 = 4 * SIMULATION_MASS * (rp * atom_speed) ** 2
    term4 = 4 * sqrt(Bp * SIMULATION_MAGNETON)
    radius_orbit = term1 + sqrt(term2 + term3) / term4  # quadratic formula
    return radius_orbit


def orbit_offset_bender(el: BenderSim, speed_ratio):
    """Atom orbits are offset from center of bender because of centrifugal force"""
    delta_r = (el.ro - el.rb) * speed_ratio ** 2
    return delta_r


def bender_orbit_radius_no_energy_correction(Bp: RealNum, rb: RealNum, rp: RealNum, atom_speed: RealNum) -> float:
    term1 = .5 * rb
    term2 = rb ** 2
    term3 = 2 * SIMULATION_MASS * (atom_speed * rp) ** 2 / (SIMULATION_MAGNETON * Bp)
    term4 = .5 * np.sqrt(term2 + term3)
    return term1 + term4


def is_sim_lattice_viable(simulated_lattice: ParticleTracerLattice, atom_speed: RealNum) -> bool:
    """Check that the matrix lattice generated from the simulated lattice is viable. It will not be if the
    atom speed is such that the orbit in the bender is shifted into the wall"""
    # IMPROVEMENT: reimplement this
    speed_ratio = atom_speed / simulated_lattice.design_speed
    for el in simulated_lattice:
        if type(el) is BenderSim:
            if orbit_offset_bender(el, speed_ratio) > el.ap:
                return False
    return True


def spring_constant_lens(Bp: RealNum, rp: RealNum, atom_speed: RealNum) -> float:
    """Spring constant (F=-Kx) for lens's harmonic potential"""
    assert rp > 0.0
    K = 2 * SIMULATION_MAGNETON * Bp / (SIMULATION_MASS * (atom_speed * rp) ** 2)
    return K


def bender_spring_constant(Bp: RealNum, rp: RealNum, ro: RealNum, atom_speed: RealNum) -> float:
    """Spring constant (F=-Kx) for bender's harmonic potential"""
    assert Bp >= 0.0 and 0.0 < rp
    K_cent = 3 * SIMULATION_MASS / ro ** 2  # centrifugal term
    K_lens = spring_constant_lens(Bp, rp, atom_speed)
    K = K_cent + K_lens
    return K


def matrix_components(M: ndarray) -> tuple[RealNum, ...]:
    """Unpack a 2x2 or 3x3 matrix into its components"""
    if M.shape == (2, 2):
        m11, m12 = M[0]  # first row
        m21, m22 = M[1]  # second row
        return m11, m12, m21, m22
    elif M.shape == (3, 3):
        m11, m12, m13 = M[0]  # first row
        m21, m22, m23 = M[1]  # second row
        m31, m32, m33 = M[2]  # thirt row
        return m11, m12, m13, m21, m22, m23, m31, m32, m33
    else:
        raise NotImplementedError


def Ki_mag_lens(magnets: Collection, s: RealNum, num_samples: int, which_dim: str, range_max: RealNum) -> float:
    """Magnetic spring constants from lens at position s along xo or yo axes"""
    x_vals = np.ones(num_samples) * s
    dim_vals = np.linspace(-range_max, range_max, num_samples)
    zeros = np.zeros(num_samples)
    if which_dim == 'xo':
        y_vals, z_vals, index_dim = dim_vals, zeros, 1
    elif which_dim == 'yo':
        y_vals, z_vals, index_dim = zeros, dim_vals, 2
    else:
        raise ValueError
    coords = np.column_stack((x_vals, y_vals, z_vals))
    force_xo = magnet_force(magnets, coords)[:, index_dim]
    return fit_k_to_force(dim_vals, force_xo)


def lens_K_mags_for_slice(s: RealNum, magnets: Collection, el: HalbachLensSim) -> tuple[float, float]:
    """Magnetic spring constans for lens element at position x. Return spring constant for xo and yo axes"""
    num_samples = 7
    fit_range_max = el.rp / 4.0
    Kx = Ki_mag_lens(magnets, s, num_samples, 'xo', fit_range_max)
    Ky = Ki_mag_lens(magnets, s, num_samples, 'yo', fit_range_max)
    return Kx, Ky


def s_slices_and_lengths_lens(el: HalbachLensSim) -> tuple[ndarray, ndarray, float, ndarray]:
    """Get the slice positions and lengths along lens. Symmetry is exploited if the lens is long enough such that the
    lens is split into two region, and fringing region and inner region. The inner region is only used if the lens is
    long enough that the fringe fields don't impact the inner region meaningfully"""
    s_fringe_depth = (el.fringe_frac_outer + 3) * el.rp
    if s_fringe_depth >= el.L / 2.0:
        s_fringe_depth = el.L / 2.0
    num_slices_per_bore_rad = 10
    num_slices = round(num_slices_per_bore_rad * s_fringe_depth / el.rp)
    num_slices = int(2 * (num_slices // 2))
    s_slices_fringing, slice_length_fringe = split_range_into_slices(0.0, s_fringe_depth, num_slices)
    slice_lengths_fringe = np.ones(len(s_slices_fringing)) * slice_length_fringe
    if s_fringe_depth >= el.L / 2.0:
        s_inner, lengths_inner = None, None
    else:
        s_inner = s_fringe_depth + (el.L / 2.0 - s_fringe_depth) / 2.0
        length_inner = el.L - s_fringe_depth * 2.0
        num_inner = round(length_inner / slice_lengths_fringe[0])
        lengths_inner = np.ones(num_inner) * length_inner / num_inner
    return s_slices_fringing, slice_lengths_fringe, s_inner, lengths_inner


def K_mag_vals_and_slice_lengths_from_lens_el(el: HalbachLensSim) -> tuple[ndarray, ndarray]:
    """Magnetic spring constants and lengths of slices along the length of the lens. used to construct short transfer
    matrice at each slice"""
    magnets = el.magnet.magpylib_magnets_model(False, False)
    s_slices_fringing, slice_lengths_fringe, s_inner, lengths_inner = s_slices_and_lengths_lens(el)
    K_vals_mag_fringe = [lens_K_mags_for_slice(s, magnets, el) for s in s_slices_fringing]
    if s_inner is not None:
        K_mag_inner = lens_K_mags_for_slice(s_inner, magnets, el)
        K_vals_mag_inner = [K_mag_inner] * len(lengths_inner)
        K_vals_mag = [*K_vals_mag_fringe, *K_vals_mag_inner, *reversed(K_vals_mag_fringe)]
        slice_lengths = [*slice_lengths_fringe, *lengths_inner, *slice_lengths_fringe]
    else:
        K_vals_mag = [*K_vals_mag_fringe, *reversed(K_vals_mag_fringe)]
        slice_lengths = [*slice_lengths_fringe] * 2
    return np.array(slice_lengths), np.array(K_vals_mag)


def transfer_matrix_func_from_lens(el: HalbachLensSim) -> Callable:
    """Create equivalent transfer matric from lens. Built by slicing lens into short transfer matrices"""
    slice_lengths, K_vals_mag = K_mag_vals_and_slice_lengths_from_lens_el(el)
    s_max = el.Lo
    s_vals = np.cumsum(slice_lengths)

    @functools.lru_cache
    @numba.njit
    def cumulatice_transfer_matrices(atom_speed: RealNum):
        """Return cumulative transfer matrices for x,y and dispersion"""
        Mx, My, Md = np.eye(2), np.eye(2), np.eye(3)
        Mx_cum, My_cum, Md_cum = arr_float64_list(), arr_float64_list(), arr_float64_list()
        for [Kx_mag, Ky_mag], L in zip(K_vals_mag, slice_lengths):
            Kx = Kx_mag / atom_speed ** 2
            Ky = Ky_mag / atom_speed ** 2
            update_Mu_and_Mu_cum(Mx, Mx_cum, Kx, L)
            update_Mu_and_Mu_cum(My, My_cum, Ky, L)
            update_Mu_and_Mu_cum(Md, Md_cum, Kx, L, R=np.inf)
        return Mx_cum, My_cum, Md_cum

    M_func = cumulative_M_func(s_vals, s_max, cumulatice_transfer_matrices)
    return M_func


def split_range_into_slices(x_min: RealNum, x_max: RealNum, num_slices: int) -> tuple[ndarray, float]:
    """Split a range into equally spaced points, that are half a spacing away from start and end"""
    slice_length = (x_max - x_min) / num_slices
    x_slices = np.linspace(x_min, x_max, num_slices + 1)[:-1]
    x_slices += slice_length / 2.0
    return x_slices, slice_length


def unit_vec_perp_to_path(path_coords: ndarray) -> ndarray:
    """Along the path 'path_coords', find the perpindicualr normal vector for each point. Assume the path is
    counter-clockwise, and the normal points radially outwards"""
    # IMPROVEMENT: This could be more accurate by using the speed at each point I think
    norm_perps = []
    vec_vertical = np.array([0, 0, 1.0])
    for i in range(len(path_coords)):
        if i != len(path_coords) - 1:
            dr = path_coords[i + 1] - path_coords[i]
        else:
            dr = path_coords[i] - path_coords[i - 1]
        perp = np.cross(dr, vec_vertical)
        norm_perp = perp / np.linalg.norm(perp)
        norm_perps.append(norm_perp)
    return np.array(norm_perps)


def combiner_slice_length_at_traj_index(index: int, coords_path: ndarray) -> float:
    """Length of short slice along length of trajectory through combiner"""
    assert index < len(coords_path)
    if index == 0:
        dr = np.linalg.norm(coords_path[1] - coords_path[0])
        slice_length = dr / 2.0
    elif index == len(coords_path) - 1:
        dr = np.linalg.norm(coords_path[-1] - coords_path[-2])
        slice_length = dr / 2.0
    else:
        dr2 = np.linalg.norm(coords_path[index + 1] - coords_path[index])
        dr1 = np.linalg.norm(coords_path[index] - coords_path[index - 1])
        slice_length = dr1 / 2 + dr2 / 2
    return slice_length


def combiner_Ki_mag_for_slice(coord: ndarray, normi: ndarray, magnets: Collection, qo_vals: sequence) -> float:
    """Magnetic spring constant for a slice in combiner along the vector normi perpindicular to coord"""
    coords = np.array([coord + qo * normi for qo in qo_vals])
    forces = magnet_force(magnets, coords)
    forces_yo = [np.dot(force, normi) for force in forces]
    Ki = fit_k_to_force(qo_vals, forces_yo)
    return Ki


def combiner_K_mags_for_slice(coord: ndarray, norm_xo: ndarray, magnets: Collection,
                              xo_vals: sequence, yo_vals: sequence) -> tuple[float, float]:
    """Magnetic spring constants for combiner element at position coord. Return spring constant for xo and yo axes"""
    norm_yo = np.array([0, 0, 1.0])
    Kx = combiner_Ki_mag_for_slice(coord, norm_xo, magnets, xo_vals)
    Ky = combiner_Ki_mag_for_slice(coord, norm_yo, magnets, yo_vals)
    return Kx, Ky


def bending_radius(index: int, p_path: ndarray, coords_path: ndarray):
    """Bending radius at a goven index along the trajectory of the combiner. Uses simple geometry of angle subtended
    by and arc of known path length"""
    assert index >= 0
    if index != len(p_path) - 1:
        v1, v2 = p_path[index], p_path[index + 1]
        dL = np.linalg.norm(coords_path[index + 1] - coords_path[index])
    else:
        v1, v2 = p_path[index], p_path[index - 1]
        dL = np.linalg.norm(coords_path[index - 1] - coords_path[index])
    theta1 = atan2(v1[1], v1[0])
    theta2 = atan2(v2[1], v2[0])
    R = abs(dL / tan(theta2 - theta1))
    return R


def combiner_K_mag_and_R_vals(coords_path: ndarray, p_path: ndarray, el: CombinerLensSim) \
        -> tuple[ndarray, ndarray]:
    """Mganetic spring constants and bending radius along path of trajectory through combiner."""
    magnets = el.magnet.magpylib_magnets_model(False, False)
    norms_xo_path = unit_vec_perp_to_path(coords_path)
    xo_max = min([(el.rp - el.output_offset), el.output_offset])
    num_points = 7
    xo_vals = np.linspace(-xo_max / 5.0, xo_max / 5.0, num_points)
    yo_max = sqrt(el.rp ** 2 - el.output_offset ** 2)
    yo_vals = np.linspace(-yo_max / 5.0, yo_max / 5.0, num_points)
    K_vals_mag = [combiner_K_mags_for_slice(coord, norm, magnets, xo_vals, yo_vals) for coord, norm in
                  zip(coords_path, norms_xo_path)]
    R_vals = [bending_radius(idx, p_path, coords_path) for idx in range(len(K_vals_mag))]
    return np.array(K_vals_mag), np.array(R_vals)


def transfer_matrix_func_from_combiner(el: CombinerLensSim) -> Callable:
    """Compute the transfer matric for the combiner. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    coords_path, p_path = combiner_orbit_coords_el_frame(el)
    coords_path, p_path = np.flip(coords_path, axis=0), np.flip(p_path, axis=0)
    speeds_path = np.linalg.norm(p_path, axis=1)
    lengths = np.array([combiner_slice_length_at_traj_index(idx, coords_path) for idx in range(len(speeds_path))])
    s_vals = np.cumsum(lengths)
    K_vals_mag, R_vals = combiner_K_mag_and_R_vals(coords_path, p_path, el)
    s_max = el.Lo

    @functools.lru_cache
    @numba.njit
    def cumulatice_transfer_matrices(atom_speed: RealNum):
        """Return cumulative transfer matrices for x,y and dispersion"""
        Mx, My, Md = np.eye(2), np.eye(2), np.eye(3)
        Mx_cum, My_cum, Md_cum = arr_float64_list(), arr_float64_list(), arr_float64_list()
        speed_factor = atom_speed / DEFAULT_ATOM_SPEED
        for idx, [[Kx_mag, Ky_mag], R, speed, L] in enumerate(zip(K_vals_mag, R_vals, speeds_path, lengths)):
            speed *= speed_factor
            Kx = Kx_mag / speed ** 2
            Ky = Ky_mag / speed ** 2
            R *= speed_factor ** 2
            K_cent = 3 / R ** 2
            Kx += K_cent
            # IMPROVEMENT: Should the length change depending on the speed factor and the bending radius?
            update_Mu_and_Mu_cum(Mx, Mx_cum, Kx, L)
            update_Mu_and_Mu_cum(My, My_cum, Ky, L)
            update_Mu_and_Mu_cum(Md, Md_cum, Kx, L, R=R)
        return Mx_cum, My_cum, Md_cum

    M_func = cumulative_M_func(s_vals, s_max, cumulatice_transfer_matrices)
    return M_func


def xo_unit_vector_bender_el_frame(el: BenderSim, coord: ndarray) -> ndarray:
    """get unit vector pointing along xo in the bender orbit frame"""
    which_section = el.in_which_section_of_bender(coord)
    if which_section == 'ARC':
        norm = coord / np.linalg.norm(coord)
    elif which_section == 'OUT':
        norm = np.array([1.0, 0.0, 0.0])
    elif which_section == 'IN':
        nx, ny = mirror_across_angle(1.0, 0, el.ang / 2.0)
        norm = np.array([nx, ny, 0.0])
    else:
        raise ValueError
    return norm


speed_with_energy_correction = numba.njit(speed_with_energy_correction)


def fit_k_to_force(pos_vals: sequence, force: sequence) -> float:
    """given position and force values, find the spring constant"""
    m = np.polyfit(pos_vals, force, 1)[0]
    K = -m
    return K


def bender_K_mag_xo(el: BenderSim, s: RealNum, magnets: Collection, xo_max: RealNum, num_samples: int) -> float:
    """Spring constant in bender along xo axis"""
    xo_vals = np.linspace(-xo_max / 4.0, xo_max / 4.0, num_samples)
    coords = np.array([el.convert_orbit_to_cartesian_coords(s, xo, 0.0) for xo in xo_vals])
    forces = magnet_force(magnets, coords)
    norm_perps = xo_unit_vector_bender_el_frame(el, coords[0])
    forces_r = [np.dot(norm_perps, force) for force in forces]
    K = fit_k_to_force(xo_vals, forces_r)
    return K


def bender_K_mag_yo(el: BenderSim, s: RealNum, magnets: Collection, zo_max: RealNum, num_samples: int) -> float:
    """Spring constant in bender along yo axis"""
    zo_vals = np.linspace(-zo_max / 4.0, zo_max / 4.0, num_samples)
    coords = np.array([el.convert_orbit_to_cartesian_coords(s, 0.0, zo) for zo in zo_vals])
    forces = magnet_force(magnets, coords)
    forces_z = [force[2] for force in forces]
    K = fit_k_to_force(zo_vals, forces_z)
    return K


def s_slices_and_lengths_bender(el) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """slices and lengths through bender. Slice positions for the fringnig region and symmetry region are returned,
     as well as slice positions and length for the entire bender
    """
    s_unit_cell = el.ro * el.ucAng
    s_length_magnet = 2 * s_unit_cell
    s_fringe_depth = el.L_cap + NUM_FRINGE_MAGNETS_MIN * s_length_magnet
    num_slices_per_bore_radius = 10
    num_slices_fringe = round(num_slices_per_bore_radius * s_fringe_depth / el.rp)
    num_slices_uc = round(num_slices_per_bore_radius * s_unit_cell / el.rp)
    s_slices_fringe, _ = split_range_into_slices(0.0, s_fringe_depth, num_slices_fringe)
    s_slices_uc, _ = split_range_into_slices(s_fringe_depth, s_fringe_depth + s_unit_cell, num_slices_uc)

    s_internal_start = s_fringe_depth
    s_total_length = el.Lo
    s_internal_stop = s_total_length - s_fringe_depth
    num_internal_mags = (el.num_lenses - 2 * NUM_FRINGE_MAGNETS_MIN)
    num_internal_slices = num_internal_mags * 2 * len(s_slices_uc)
    s_slices_internal, length_internal = split_range_into_slices(s_internal_start, s_internal_stop, num_internal_slices)
    s_slices_fringe_end, length_fringe = split_range_into_slices(s_internal_stop, s_total_length, len(s_slices_fringe))
    s_slices_total = np.array([*s_slices_fringe, *s_slices_internal, *s_slices_fringe_end])
    lengths_fringe = np.ones(len(s_slices_fringe_end)) * length_fringe
    lengths_internal = np.ones(len(s_slices_internal)) * length_internal
    lengths_total = np.array([*lengths_fringe, *lengths_internal, *lengths_fringe])
    return s_slices_fringe, s_slices_uc, s_slices_total, lengths_total


def bender_K_mag_for_slice(s: RealNum, magnets, el) -> tuple[float, float]:
    """This slice transfer matrix for a point along the bender"""
    deltar_orbit = el.ro - el.rb
    xo_max = el.ap - deltar_orbit
    zo_max = sqrt(el.rp ** 2 - deltar_orbit ** 2)
    num_samples = 7
    Kx = bender_K_mag_xo(el, s, magnets, xo_max, num_samples)
    Ky = bender_K_mag_yo(el, s, magnets, zo_max, num_samples)
    return Kx, Ky


def stitch_bender_values(values_start: sequence, values_uc: sequence, num_internal_mags: int) -> ndarray:
    """Given spring constant, or physically similiar values, from the fringing and symmetry region, construct a list
    that represents the entire bender"""
    values_internal = [*values_uc, *reversed(values_uc)] * num_internal_mags
    values_full = [*values_start, *values_internal, *reversed(values_start)]
    return np.array(values_full)


def K_mag_vals_and_lengths_from_bender_el(el: BenderSim) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Magnetic spring constants and lengths at each spring constant value along the orbit trajectory through the
    bender"""
    magnets = el.magnet.magpylib_magnets_model(False)
    s_slices_fringe, s_slices_uc, s_slices_total, lengths_total = s_slices_and_lengths_bender(el)
    num_internal_mags = (el.num_lenses - 2 * NUM_FRINGE_MAGNETS_MIN)
    K_vals_mag_uc = np.array([bender_K_mag_for_slice(s, magnets, el) for s in s_slices_uc])
    Kx_vals_mag_uc, Ky_vals_mag_uc = K_vals_mag_uc.T
    K_vals_mag_start = np.array([bender_K_mag_for_slice(s, magnets, el) for s in s_slices_fringe])
    Kx_vals_mag_start, Ky_vals_mag_start = K_vals_mag_start.T
    Kx_vals_mag = stitch_bender_values(Kx_vals_mag_start, Kx_vals_mag_uc, num_internal_mags)
    Ky_vals_mag = stitch_bender_values(Ky_vals_mag_start, Ky_vals_mag_uc, num_internal_mags)
    return s_slices_total, lengths_total, Kx_vals_mag, Ky_vals_mag


def transfer_matrix_func_from_bender(el: BenderSim) -> Callable:
    """Return the transfer matric for the bender. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    if el.num_lenses < 2 * NUM_FRINGE_MAGNETS_MIN:
        raise NotImplementedError
    s_vals, lengths_total, Kx_vals_mag, Ky_vals_mag = K_mag_vals_and_lengths_from_bender_el(el)
    s_max, ro, rb, L_cap, ang = el.Lo, el.ro, el.rb, el.L_cap, el.ang

    @functools.lru_cache
    @numba.njit
    def cumulatice_transfer_matrices(atom_speed: RealNum):
        """Return cumulative transfer matrices for x,y and dispersion"""
        Mx, My, Md = np.eye(2), np.eye(2), np.eye(3)
        Mx_cum, My_cum, Md_cum = arr_float64_list(), arr_float64_list(), arr_float64_list()
        for s, L, Kx_mag, Ky_mag in zip(s_vals, lengths_total, Kx_vals_mag, Ky_vals_mag):
            orbit_offset = (ro - rb) * (atom_speed / DEFAULT_ATOM_SPEED) ** 2
            V = .5 * Kx_mag * orbit_offset ** 2
            atom_speed_corrected = speed_with_energy_correction(V, atom_speed)
            Kx = Kx_mag / atom_speed_corrected ** 2
            Ky = Ky_mag / atom_speed_corrected ** 2

            if L_cap < s < ang * ro + L_cap:
                R = rb + orbit_offset
            else:
                R = np.inf
            K_cent = 3.0 / R ** 2
            Kx += K_cent
            update_Mu_and_Mu_cum(Mx, Mx_cum, Kx, L)
            update_Mu_and_Mu_cum(My, My_cum, Ky, L)
            update_Mu_and_Mu_cum(Md, Md_cum, Kx, L, R=R)
        return Mx_cum, My_cum, Md_cum

    M_func = cumulative_M_func(s_vals, s_max, cumulatice_transfer_matrices)
    return M_func


class Element:
    """ Base element representing a transfer matrix (ABCD matrix)"""

    def __init__(self, L: RealNum, ap: Union[RealNum, sequence] = np.inf):
        assert L > 0.0
        ap = (ap, ap) if isinstance(ap, (int, float)) else tuple(ap)
        assert len(ap) == 2
        self.L = L
        self.ap = ap

    def M_func(self, s: RealNum, atom_speed: RealNum) -> ThreeNumpyMatrices:
        """Return the x and y transfer matrices at a given location 's' inside the element for an atom with longitudinal
        velocity 'atom_speed'"""
        raise NotImplementedError

    def M(self, atom_speed=DEFAULT_ATOM_SPEED, s=None) -> ThreeNumpyMatrices:
        """Return the x and y transfer matrices. If no position is provided, total length is used."""
        if s is not None and not 0.0 <= s <= self.L:
            raise ValueError("Specified position must be inside element")
        s = self.L if s is None else s
        return self.M_func(s, atom_speed)


class NumericElement(Element):
    """An element representing a component whos transfer matrix has been computed numerically. Typically from the
    procedure of splitting an element up into slices to more accuretly account for fringe fields"""

    def __init__(self, L, M_func, ap=np.inf):
        super().__init__(L, ap=ap)
        assert isinstance(M_func, Callable)
        self._M_func = M_func

    def M_func(self, s: RealNum, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        return self._M_func(s, atom_speed)


class Lens(Element):
    """Element representing a lens"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum, ap=None):
        self.Bp = Bp
        self.rp = rp
        ap = rp if ap is None else ap
        super().__init__(L, ap=ap)

    def M_func(self, s: RealNum, atom_speed: RealNum) -> ThreeNumpyMatrices:
        assert 0 <= s <= self.L
        K = spring_constant_lens(self.Bp, self.rp, atom_speed)
        Mx = My = transfer_matrix(K, s)
        Md = transfer_matrix(K, s, R=np.inf)
        return Mx, My, Md


class Combiner(Element):
    """Element representing a combiner"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum):
        self.Bp = Bp
        self.rp = rp
        super().__init__(L)

    def M_func(self, s: RealNum, atom_speed: RealNum) -> ThreeNumpyMatrices:
        assert 0 <= s <= self.L
        K = spring_constant_lens(self.Bp, self.rp, atom_speed)
        Mx = My = transfer_matrix(K, s)
        Md = transfer_matrix(K, s, R=np.inf)
        return Mx, My, Md


class Drift(Element):
    """Element representing free space"""

    def __init__(self, L: RealNum, ap):
        super().__init__(L, ap=ap)

    def M_func(self, s: RealNum, atom_speed: RealNum) -> ThreeNumpyMatrices:
        assert 0 <= s <= self.L
        K = 0.0
        Mx = transfer_matrix(K, s)
        My = Mx.copy()
        Md = transfer_matrix(K, s, R=np.inf)
        return Mx, My, Md


class Bender(Element):
    """Element representing a bending component"""

    def __init__(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum, ro=None, ap=None):
        if ro is None:
            self.ro = bender_orbit_radius_no_energy_correction(Bp, rb, rp, DEFAULT_ATOM_SPEED)
        else:
            self.ro = ro
        ap = rp if ap is None else ap
        centrifugal_offset = self.ro - rb
        if centrifugal_offset >= rp:
            raise ValueError("particle cannot survive in bender because centrifugal offset is too large")
        self.rb = rb
        self.rp = rp
        self.Bp = Bp
        L = self.ro * bending_angle  # length of particle orbit
        ap_x = ap - centrifugal_offset
        ap_y = sqrt(ap ** 2 - centrifugal_offset ** 2)  # IMPROVEMENT: implement two different values
        super().__init__(L, ap=(ap_x, ap_y))

    def M_func(self, s: RealNum, atom_speed: RealNum) -> ThreeNumpyMatrices:
        assert 0 <= s <= self.L
        Kx = bender_spring_constant(self.Bp, self.rp, self.ro, atom_speed)
        Ky = spring_constant_lens(self.Bp, self.rp, atom_speed)
        Mx = transfer_matrix(Kx, s)
        My = transfer_matrix(Ky, s)
        Md = transfer_matrix(Kx, s, R=self.ro)
        return Mx, My, Md


HashableElements = Union[tuple[Element], dict[Any, Element]]


def make_effective_lens_elements(el: HalbachLensSim) -> tuple[Drift, Lens, Drift]:
    """Return matrix elements (Drift, lens,Drift) that most effectively capture the behaviour of a fully simulated
    lens"""
    assert type(el) is HalbachLensSim
    Bp = Bp_effective_lens(el)
    ap = el.ap
    L_lens = length_hard_edge(el)
    L_drift = el.fringe_frac_outer * el.rp
    drift1 = Drift(L_drift, ap)
    drift2 = Drift(L_drift, ap)
    lens = Lens(L_lens, Bp, el.rp)
    return drift1, lens, drift2


def make_effective_bender_elements(el: BenderSim) -> tuple[Drift, Bender, Drift]:
    """Return matrix elements (Drift, Bender,Drift) that most effectively capture the behaviour of a fully simulated
    bender"""
    assert type(el) is BenderSim
    Bp = Bp_effective_bender(el)
    L_drift = el.fringe_frac_outer * el.rp

    ap = el.ap
    drift1 = Drift(L_drift, ap)
    drift2 = Drift(L_drift, ap)
    bender = Bender(Bp, el.rb, el.rp, el.ang, ro=el.ro)
    return drift1, bender, drift2


@Memoize_Elements
def make_effective_elements(el) -> tuple[Element, Element, Element]:
    """Return matrix elements that most effectively model the behaviour of a fully simulated element"""
    if type(el) is BenderSim:
        return make_effective_bender_elements(el)
    elif type(el) is HalbachLensSim:
        return make_effective_lens_elements(el)
    else:
        raise NotImplementedError


def mult_el_matrices(Mx: ndarray, My: ndarray, Md: ndarray, el: Element,
                     atom_speed, s: RealNum = None) -> ThreeNumpyMatrices:
    """Given (x,y) transfer matrices chain new (x,y) transfer matrices from an element and return the result"""
    Mx_el, My_el, Md_el = el.M(s=s, atom_speed=atom_speed)
    Mx = Mx_el @ Mx
    My = My_el @ My
    Md = Md_el @ Md
    return Mx, My, Md


def M_exit(el, s_start, atom_speed):
    """Return the transfer matrix representing exit(leaving) an element starting from position s_start.
    The alogirhtm requires applying the inverse of the entrance portion to the entire matrix"""
    Mx_total, My_total, Md_total = el.M(atom_speed=atom_speed)
    s_entrance = el.L - s_start
    Mx_entrance, My_entrance, Md_entrance = el.M(s=s_entrance, atom_speed=atom_speed)
    Mx_exit = Mu_exit(Mx_total, Mx_entrance)
    My_exit = Mu_exit(My_total, My_entrance)
    Md_exit = Mu_exit(Md_total, Md_entrance)
    return Mx_exit, My_exit, Md_exit


@functools.lru_cache
def lattice_cumulatice_length(elements: HashableElements) -> ndarray:
    """Return array of cumulative length of lattice"""
    return np.cumsum([el.L for el in elements])


def lattice_transfer_matrix_at_s(s, elements: HashableElements, atom_speed) -> ThreeNumpyMatrices:
    """Find total transfer matrix from a point s in the lattice. Lattice is assumed to be periodic"""
    s = s % total_length(elements)
    length_cumulative = lattice_cumulatice_length(elements)
    el_index = index_in_increasing_arr(s, length_cumulative)
    el_inside = elements[el_index]
    delta_s = s - length_cumulative[el_index - 1] if el_index > 0 else s

    Mx, My, Md = M_exit(el_inside, el_inside.L - delta_s, atom_speed)
    for el in elements[el_index + 1:]:
        Mx, My, Md = mult_el_matrices(Mx, My, Md, el, atom_speed)
    for el in elements[:el_index]:
        Mx, My, Md = mult_el_matrices(Mx, My, Md, el, atom_speed)
    Mx, My, Md = mult_el_matrices(Mx, My, Md, el_inside, atom_speed, s=delta_s)
    return Mx, My, Md


def phi_twiss(m11: RealNum, m12: RealNum, unused_m21: RealNum, m22: RealNum) -> float:
    """return phi twiss parameter assuming periodicity"""
    arg = (m11 + m22) / 2.0
    if m12 < 0:
        return -np.arccos(arg) + 2 * np.pi
    else:
        return np.arccos(arg)


def alpha_from_components(m11: RealNum, m12: RealNum, unused_m21: RealNum, m22: RealNum) -> float:
    """Return alpha twiss parameter assuming periodicity"""
    value = (m22 - m11) / np.sqrt(4 - (m11 + m22) ** 2)
    sign = 1 if m12 < 0 else -1
    return value * sign


def beta_from_components(m11: RealNum, m12: RealNum, unused_m21: RealNum, m22: RealNum) -> float:
    """Return beta twiss parameter assuming periodicity"""
    value = 2 * m12 / np.sqrt(4 - (m11 + m22) ** 2)
    sign = -1 if m12 < 0 else 1
    return value * sign


def gamma_from_components(m11: RealNum, m12: RealNum, m21: RealNum, m22: RealNum) -> float:
    """Value of gamma twiss parameter assuming periodicity"""
    value = 2 * m21 / np.sqrt(4 - (m11 + m22) ** 2)
    sign = 1 if m12 < 0 else -1
    return value * sign


def dispersion_slope_from_components(m11, unused_m12, m13, m21, m22, m23, unused_m31, unused_m32, unused_m33):
    D_prime = (m21 * m13 + m23 * (1 - m11)) / (2 - m11 - m22)
    return D_prime


def dispersion_from_components(m11, m12, m13, m21, m22, m23, m31, m32, m33):
    """Return value of dispersion function given dispersion transfer matrix values"""
    D_prime = dispersion_slope_from_components(m11, m12, m13, m21, m22, m23, m31, m32, m33)
    D = (m12 * D_prime + m13) / (1 - m11)
    return D


def is_wronskian_valid(M: ndarray) -> bool:
    """Return True if the Wronskian of the matrix 'M' is valid"""
    return isclose(det(M), 1.0, abs_tol=1e-9)


def twiss_parameters(M: ndarray) -> tuple[float, float, float]:
    """Return twiss parameters (phi, alpha, beta, gamma)"""
    matrix_vals = matrix_components(M)
    assert is_wronskian_valid(M)
    # phi = phi_twiss(*matrix_vals)
    alpha = alpha_from_components(*matrix_vals)
    beta = beta_from_components(*matrix_vals)
    gamma = gamma_from_components(*matrix_vals)
    return alpha, beta, gamma


def beta(M: ndarray) -> float:
    """Return beta twiss parameter assuming periodicity"""
    return beta_from_components(*matrix_components(M))


def dispersion_slope(M: ndarray) -> float:
    return dispersion_slope_from_components(*matrix_components(M))


def dispersion(M: ndarray) -> float:
    """Return value of dispersion function from matrix 'M', assuming periodicity"""
    return dispersion_from_components(*matrix_components(M))


def betas_at_s(s: RealNum, elements: HashableElements, atom_speed: RealNum) -> tuple[float, float]:
    """Return the (x,y) beta function value at position s. Assumes periodicity"""
    Mx, My, _ = lattice_transfer_matrix_at_s(s, elements, atom_speed)
    beta_x = beta(Mx)
    beta_y = beta(My)
    return beta_x, beta_y


def dispersion_at_s(s: RealNum, elements: HashableElements, atom_speed: RealNum) -> float:
    """Return value of dispersion function at location s in lattice, assuming periodicity"""
    _, _, Md = lattice_transfer_matrix_at_s(s, elements, atom_speed)
    D = dispersion(Md)
    return D


def dispersion_slope_at_s(s: RealNum, elements: HashableElements, atom_speed: RealNum) -> float:
    _, _, Md = lattice_transfer_matrix_at_s(s, elements, atom_speed)
    D_prime = dispersion_slope(Md)
    return D_prime


@functools.lru_cache
def beta_profile(elements, atom_speed, num_points):
    """Return (x,y) beta functions value along lattice. Also return the locations of the beta functions"""
    # IMPROVEMENT: return the stable beta profile
    L = total_length(elements)
    s_vals = np.linspace(0, L, num_points)
    if not is_stable_both_xy(elements, atom_speed):
        nan_arr = np.nan * np.ones(len(s_vals))
        make_arrays_read_only(s_vals, nan_arr)
        return s_vals, (nan_arr, nan_arr)
    else:
        betas = np.array([betas_at_s(s, elements, atom_speed) for s in s_vals])
        betas_x, betas_y = np.abs(betas.T)
        make_arrays_read_only(s_vals, betas_x, betas_y)
        return s_vals, (betas_x, betas_y)


def alpha_profile(elements: HashableElements, atom_speed: float,
                  num_points: int) -> tuple[ndarray, tuple[ndarray, ndarray]]:
    """Return gamma profile through periodic lattice of elements"""
    s_vals, (beta_x, beta_y) = beta_profile(elements, atom_speed, num_points)
    alphas_x = np.gradient(beta_x, s_vals)
    alphas_y = np.gradient(beta_y, s_vals)
    return s_vals, (alphas_x, alphas_y)


def gamma_profiles(elements: HashableElements, atom_speed: float,
                   num_points: int) -> tuple[ndarray, tuple[ndarray, ndarray]]:
    """Return gamma profile through periodic lattice of elements"""
    s_vals, (betas_x, betas_y) = beta_profile(elements, atom_speed, num_points)
    _, (alphas_x, alphas_y) = alpha_profile(elements, atom_speed, num_points)
    gammas_x = (1 + alphas_x ** 2) / betas_x
    gammas_y = (1 + alphas_y ** 2) / betas_y
    return s_vals, (gammas_x, gammas_y)


@functools.lru_cache
def dispersion_profile(elements: HashableElements, atom_speed: RealNum,
                       num_points: int) -> tuple[ndarray, ndarray]:
    """Return profile of dispersion function along lattice, assuming periodicity"""
    L = total_length(elements)
    s_vals = np.linspace(0, L, num_points)
    if not is_stable_both_xy(elements, atom_speed):
        nan_arr = np.nan * np.ones(len(s_vals))
        make_arrays_read_only(s_vals, nan_arr)
        return s_vals, nan_arr
    else:
        dispersions = np.array([dispersion_at_s(s, elements, atom_speed) for s in s_vals])
        make_arrays_read_only(s_vals, dispersions)
        return s_vals, dispersions


def tunes_absolute(elements: HashableElements, atom_speed: RealNum, num_points) -> tuple[float, float]:
    """Return absolute value of tune. Calculated by integrating beta function profile along lattice"""
    s_vals, (betas_x, betas_y) = beta_profile(elements, atom_speed, num_points)
    tune_x = np.trapz(1 / betas_x, x=s_vals) / (2 * np.pi)
    tune_y = np.trapz(1 / betas_y, x=s_vals) / (2 * np.pi)
    return tune_x, tune_y


def tunes_incremental(elements: HashableElements, atom_speed: RealNum) -> tuple[float, float]:
    """Return tune value tune value, ie the total tune minus nearest half integer int. This is between 0 and .5 .
    This method cannot distinguish between a tune of 1.25 and 1.75, both would result in .25"""
    Mx, My, Md = total_lattice_transfer_matrix(elements, atom_speed)
    tunes = []
    for M in [Mx, My]:
        if not is_stable_matrix(M):
            tunes.append(np.nan)
        else:
            m11, m12, m21, m22 = matrix_components(M)
            stability_factor(m11, m12, m21, m22)
            tune = acos((m11 + m22) / 2.0) / (2 * np.pi)
            tunes.append(tune)
    tune_x, tune_y = tunes
    return tune_x, tune_y


@functools.lru_cache
def total_lattice_transfer_matrix(elements: HashableElements, atom_speed) -> ThreeNumpyMatrices:
    """Transfer matrix for a sequence of elements start to end"""
    matrices_xyd = [el.M(atom_speed=atom_speed) for el in elements]
    matrices_x = [entry[0] for entry in matrices_xyd]
    matrices_y = [entry[1] for entry in matrices_xyd]
    matrices_d = [entry[2] for entry in matrices_xyd]
    Mx, My, Md = multiply_matrices(matrices_x), multiply_matrices(matrices_y), multiply_matrices(matrices_d)
    return Mx, My, Md


def stability_factor(m11: RealNum, m12: RealNum, m21: RealNum, m22: RealNum) -> float:
    """Factor describing stability of periodic lattice of elements. If value is greater than 1 it is stable, though
    higher values are "more" stable in some sense"""
    return 2.0 - (m11 ** 2 + 2 * m12 * m21 + m22 ** 2)


def stability_factors_lattice(elements: HashableElements, atom_speed: RealNum) -> tuple[float, float]:
    """Factor describing stability of periodic lattice of elements. If value is greater than 0 it is stable, though
    higher values are "more" stable in some sense"""

    Mx, My, _ = total_lattice_transfer_matrix(elements, atom_speed)
    stability_factor_x = stability_factor(*matrix_components(Mx))
    stability_factor_y = stability_factor(*matrix_components(My))
    return stability_factor_x, stability_factor_y


def is_stable_xy(elements: HashableElements, atom_speed: RealNum) -> tuple[bool, bool]:
    """Determine if the lattice is stable. This can be done by computing eigenvalues, or the method below works. If
    unstable, then raising he transfer matrix to N results in large matrix elements for large N"""
    stability_factor_x, stability_factor_y = stability_factors_lattice(elements, atom_speed)
    return stability_factor_x > 0.0, stability_factor_y > 0.0


def is_stable_both_xy(elements: HashableElements, atom_speed: RealNum) -> bool:
    """Return True if the lattice is stable in both dimensions"""
    return all(is_stable_xy(elements, atom_speed))


def is_stable_matrix(M: ndarray) -> bool:
    """Return True if periodic matrix is stable, False if unstable"""
    return stability_factor(*matrix_components(M)) >= 0


@functools.lru_cache
def total_length(elements: HashableElements) -> float:
    """Sum of the lengths of elements"""
    length = sum([el.L for el in elements])
    return length


def lattice_apertures(s, elements: HashableElements):
    """Return aperture of lattice of position s. Assumed to be same for both dimensions"""
    s = s % total_length(elements)
    length_cumulative = lattice_cumulatice_length(elements)
    el_index = index_in_increasing_arr(s, length_cumulative)
    return elements[el_index].ap


@functools.lru_cache
def lattice_apertures_arr(elements: HashableElements, num_points: int) -> tuple[ndarray, ndarray]:
    """Return two arrays of lattice aperture along lattice at locations 's_vals' """
    s_vals = np.linspace(0.0, total_length(elements), num_points)
    aps_x, aps_y = np.array([lattice_apertures(s, elements) for s in s_vals]).T
    make_arrays_read_only(aps_x, aps_y)
    return aps_x, aps_y


def acceptance_profile(elements: HashableElements,
                       atom_speed: RealNum, num_points: int) -> tuple[ndarray, tuple[ndarray, ndarray]]:
    """Return (x,y) profile for acceptance through the lattice, and the corresponding position values. Assumes periodic
    lattice. This is the maximum emittance that would survive at each point in the lattice."""
    s_vals, (beta_x, beta_y) = beta_profile(elements, atom_speed, num_points)
    aps_x, aps_y = lattice_apertures_arr(elements, num_points)
    _, dispersions = dispersion_profile(elements, atom_speed, num_points)
    dispersion_shift = dispersions * delta(atom_speed)
    aps_x = aps_x - dispersion_shift
    acceptances_x = aps_x ** 2 / beta_x
    acceptances_y = aps_y ** 2 / beta_y
    return s_vals, (acceptances_x, acceptances_y)


def revolutions_from_matrix_method(particle: Particle, elements: HashableElements, T_max: RealNum) -> float:
    """Return number of expected revolutions for a particle"""
    revs_max = T_max * abs(particle.pi[0]) / total_length(elements)
    will_clip = will_particle_clip_on_aperture(particle, elements)
    return 0.0 if will_clip else revs_max


def minimum_acceptance(elements: HashableElements, atom_speed: RealNum, num_points: int) -> tuple[float, float]:
    """Return the minimum acceptance value in lattice. """
    if not is_stable_both_xy(elements, atom_speed):
        return np.nan, np.nan
    else:
        _, (acceptances_x, acceptances_y) = acceptance_profile(elements, atom_speed, num_points)
        return np.min(acceptances_x), np.min(acceptances_y)


def emittance_from_ellipse_parameters(u: float, u_slope: float, alpha: float, beta: float, gamma: float) -> float:
    """return emittance of particle given the phase space coordinates and twiss parameters"""
    assert beta > 0.0 and gamma >= 0.0
    return gamma * u ** 2 + 2 * alpha * u * u_slope + beta * u_slope ** 2


def twiss_paremeters_from_lattice(elements: HashableElements, atom_speed) -> tuple:
    """Return twiss parameters for x and y for given lattice. Parameters are (alpha, beta, gamma)"""
    Mx, My, Md = total_lattice_transfer_matrix(elements, atom_speed)
    twiss_params = []
    for M in (Mx, My):
        twiss_params.append(twiss_parameters(M))
    return tuple(twiss_params)


def orbit_phase_space_coords_final(particle: Particle) -> tuple[float, tuple, tuple, float]:
    """Return orbit phase space coordinates from particle. Requires that orbit coordinates were logged"""
    z, x, y = particle.qo_vals[-1]
    orbit_velocity, px, py = particle.po_vals[-1]
    x_slope = px / orbit_velocity
    y_slope = py / orbit_velocity
    return z, (x, x_slope), (y, y_slope), orbit_velocity


def emittance_from_particle(particle, elements: HashableElements, which_coords='initial') -> tuple:
    """Return x and y emittances of a particle in a periodic lattice of elements"""
    if which_coords == 'initial':
        _, X, Y, atom_speed = orbit_phase_space_coords_initial(particle)
        z = 0
        Mx, My, _ = total_lattice_transfer_matrix(elements, atom_speed)
    elif which_coords == 'final':
        z, X, Y, atom_speed = orbit_phase_space_coords_final(particle)
        Mx, My, _ = lattice_transfer_matrix_at_s(z, elements, atom_speed)
    else:
        raise NotImplementedError
    D = dispersion_at_s(z, elements, atom_speed)
    D_prime = dispersion_slope_at_s(z, elements, atom_speed)
    X = list(X)
    X[0] -= D * delta(atom_speed)
    X[1] -= D_prime * delta(atom_speed)
    emittances = []
    for i, ((ui, ui_slope), Mu) in enumerate(zip((X, Y), (Mx, My))):
        twiss_params = twiss_parameters(Mu)
        emittances.append(emittance_from_ellipse_parameters(ui, ui_slope, *twiss_params))
    return tuple(emittances)


def does_envelope_clip_on_aperture(xo: ndarray, yo: ndarray, apx: ndarray,
                                   apy: ndarray, dispersion_shift: ndarray) -> bool:
    """Return True if the particle's envelope increases beyond the value of an aperture in a periodic lattice. Assumes
    that the aperture are all circular, though the orbit may be displaced horizontally in the bore"""
    aps_r = (apx ** 2 + apy ** 2) / (2 * apx)
    delta_x_bore = (apy - apx) * (apy + apx) / (2 * apx)
    x_bore = xo + delta_x_bore + dispersion_shift
    y_bore = yo
    r = np.sqrt(x_bore ** 2 + y_bore ** 2)
    return np.any(r > aps_r)


def orbit_phase_space_coords_initial(particle: Particle) -> tuple[float, tuple, tuple, float]:
    """Return phase space coords in orbit frame from initial particle coords. This assumes that the particle is located
    at the origin and aimed along the negative x direction. Further, it is assumed that the positive x dimension of the
    orbit points radially outward for clockwise trajectories and thus points in the negative cartesian y direction
    for a linear lattice"""
    x_lab, y_lab, z_lab = particle.qi
    px_lab, py_lab, pz_lab = particle.pi
    assert px_lab < 0.0
    assert isclose(x_lab, -1e-10, abs_tol=1e-12)
    orbit_velocity = abs(px_lab)
    xi, yi = -y_lab, z_lab
    xi_slope, yi_slope = -py_lab / orbit_velocity, pz_lab / orbit_velocity
    z = 0.0
    return z, (xi, xi_slope), (yi, yi_slope), orbit_velocity


def will_particle_clip_on_aperture(particle: Particle, elements: HashableElements,
                                   num_envelope_points: int = 500) -> bool:
    """Return whether particle will be lost clipping on an element aperture, assuming the lattice is periodic.
    This is determined by checking if the particle's emittance is larger than the minimum viable emittance in either
    dimension"""

    # IMPROVEMENT: this can falsely predict a particle clipping if one of the profile is much smaller than the other

    _, _, _, atom_speed = orbit_phase_space_coords_initial(particle)
    if not is_stable_both_xy(elements, atom_speed):
        return True
    else:
        emittance_x, emittance_y = emittance_from_particle(particle, elements)
        _, d = dispersion_profile(elements, atom_speed, num_envelope_points)
        s_vals, (betas_x, betas_y) = beta_profile(elements, atom_speed, num_envelope_points)
        x_envelope, y_envelope = np.sqrt(betas_x * emittance_x), np.sqrt(betas_y * emittance_y)
        dispersion_shift = d * delta(atom_speed)
        aps_x, aps_y = lattice_apertures_arr(elements, len(s_vals))
        return does_envelope_clip_on_aperture(x_envelope, y_envelope, aps_x, aps_y, dispersion_shift)


def swarm_flux_mult_from_matrix_method(swarm_initial: Swarm, elements: HashableElements,
                                       T_max: RealNum, num_points_speed_range: int) -> float:
    """Return expected flux multiplication for a swarm in periodic lattice. To reduce computation, particle speeds are
    rounded to nearest value in an array of speed so memoization of underlying functions can be used. Swarm MUST be
    oriented such that it is nominally traveling along -x initially"""
    _atom_speeds = np.abs(swarm_initial[:, 'pi', 0])
    speed_range = np.linspace(min(_atom_speeds), max(_atom_speeds), num_points_speed_range)
    flux_mat_method = 0.0
    for particle in swarm_initial:
        atom_speed = abs(particle.pi[0])
        speed_rounded = speed_range[np.argmin(np.abs(atom_speed - speed_range))]
        particle_new = particle.copy()
        particle_new.pi[0] = -speed_rounded
        flux = revolutions_from_matrix_method(particle_new, elements, T_max)
        flux_mat_method += flux
    return flux_mat_method / len(swarm_initial)


def plot_particle_and_acceptance(particle: Particle, elements: HashableElements) -> None:
    """Plot particle's x and y initial values in side phase space ellipse from minimal surviving emittance given by
    minimum acceptance. Accounts for disperion"""
    _, (xi, xi_slope), (yi, yi_slope), orbit_velocity = orbit_phase_space_coords_initial(particle)
    ellipsex, ellipsey = acceptance_ellipses(elements, orbit_velocity, 100)
    plt.plot(*ellipsex)
    plt.scatter(xi, xi_slope)
    plt.plot(*ellipsey)
    plt.scatter(yi, yi_slope)
    plt.show()


def plot_swarm_survival_against_emittance(elements: HashableElements, swarm_traced: Swarm,
                                          atom_speed: RealNum, which_orbit_dim: str, save_title) -> None:
    """Generate plot of particle's simulated survival in phase space against acceptance ellipse in one dimension,
    x or y"""
    idx = ORBIT_DIM_INDEX[which_orbit_dim]
    assert atom_speed > 0
    image, binx, biny = histogram_particle_survival(swarm_traced, 'revolutions', which_orbit_dim)
    if which_orbit_dim == 'x':
        # need to flip for sign convention
        image = np.rot90(image, k=2)
        binx = np.flip(-binx)
        biny = np.flip(-biny)

    stabilities = is_stable_xy(elements, atom_speed)
    if stabilities[idx]:
        ellipse_x, ellipse_y = acceptance_ellipses(elements, atom_speed, 500)
        pos_ellipse, vel_ellipse = (ellipse_x, ellipse_y)[idx]
        pos_ellipse, vel_ellipse = np.array(pos_ellipse), np.array(vel_ellipse)
        vel_ellipse *= atom_speed
        pos_ellipse /= 1e-3
        plt.plot(pos_ellipse, vel_ellipse, linewidth=5, c='r', linestyle=':', label='Acceptance ellipse')
    extent = [binx.min(), binx.max(), biny.min(), biny.max()]
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    plt.imshow(image, extent=extent, aspect=aspect, alpha=.75)
    plt.xlabel("Position, mm")
    plt.ylabel("Velocity, m/s")
    plt.colorbar(label='Relative survival')
    plt.legend()
    if save_title is not None:
        plt.savefig(save_title, dpi=300)
    plt.show()


def ellipse_profile(x: float, eps: float, twiss_params: tuple[float, ...]) -> tuple[float, float]:
    """Return the y values (positive and negative) in the ellipse equation gamma*x^2+2*alpha*x*y+beta*y^2=eps"""
    alpha, beta, gamma = twiss_params
    term = np.sqrt((alpha * x) ** 2 + beta * eps - beta * gamma * x ** 2)
    val1 = (-alpha * x + term) / beta
    val2 = (-alpha * x - term) / beta
    return val1, val2


def phase_space_ellipse(emittance: RealNum, twiss_params: tuple[float, ...]) -> tuple[list[float], list[float]]:
    """Return x and y values of path on curve of phase space ellipse. Easily used for plotting"""
    _, beta, _ = twiss_params
    offset_to_prevent_nan = 1e-12
    x_max = np.sqrt(emittance * beta) - offset_to_prevent_nan
    x_min = -x_max
    x_vals = np.linspace(x_min, x_max, 10000)
    y_vals = np.array([ellipse_profile(x, emittance, twiss_params) for x in x_vals])
    x_vals_path = [*x_vals, *np.flip(x_vals), x_vals[0]]
    y_vals_path = [*y_vals[:, 0], *np.flip(y_vals[:, 1]), y_vals[0, 0]]
    return x_vals_path, y_vals_path


def acceptance_ellipses(elements: HashableElements, atom_speed: RealNum, num_points: int) -> list[tuple, tuple]:
    """Return phase ellipses for x and y dimensions of minimum acceptance at the input of the first element, assuming
    periodicity. Each ellipse is a list of values for plotting  as (x_plot_vals, y_plot_vals)"""
    Mx, My, Md = total_lattice_transfer_matrix(elements, atom_speed)
    max_emittances = minimum_acceptance(elements, atom_speed, num_points)
    ellipses = []
    for eps, M in zip(max_emittances, [Mx, My]):
        twiss_params = twiss_parameters(M)
        ellipses.append(phase_space_ellipse(eps, twiss_params))
    return ellipses


@Memoize_Elements  # reuse identical output rather than recreating them
def build_numeric_element(el: SimElement) -> NumericElement:
    """Return a numeric matrix element given a ParticleTracerLattice element"""
    if type(el) is HalbachLensSim:
        M_func = transfer_matrix_func_from_lens(el)
        matrix_element = NumericElement(el.Lo, M_func=M_func, ap=el.ap)
    elif type(el) is BenderSim:
        M_func = transfer_matrix_func_from_bender(el)
        r_offset = (el.ro - el.rb)
        apx = el.ap - r_offset
        apy = sqrt(el.ap ** 2 - r_offset ** 2)
        matrix_element = NumericElement(el.Lo, M_func=M_func, ap=(apx, apy))
    elif type(el) is CombinerLensSim:
        M_func = transfer_matrix_func_from_combiner(el)
        apx = el.ap - el.output_offset
        apy = sqrt(el.ap ** 2 - el.output_offset ** 2)
        matrix_element = NumericElement(el.Lo, M_func=M_func, ap=(apx, apy))
    else:
        raise NotImplementedError
    return matrix_element


def dipole_resonance_strength_factor_amplitude(elements: HashableElements, atom_speed):
    """Return factor that relates the total dipole errors to orbit offset. Larger values result in more impact of
     dipole errors on particle orbit"""
    tunex, tuney = tunes_incremental(elements, atom_speed)
    resx_rel_amp = 1 / np.sin(np.pi * tunex)
    resy_rel_amp = 1 / np.sin(np.pi * tuney)
    return resx_rel_amp, resy_rel_amp


class Lattice(Sequence):
    """Model of a series (assumed to be periodic) of elements"""

    def __init__(self):
        self.elements: tuple[Element, ...] = ()

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def add_element(self, element):
        self.elements = (*self.elements, element)

    def add_drift(self, L: RealNum, ap=np.inf) -> None:
        self.add_element(Drift(L, ap))

    def add_lens(self, L: RealNum, Bp: RealNum, rp: RealNum, ap=None) -> None:
        self.add_element(Lens(L, Bp, rp, ap=ap))

    def add_bender(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum, ap=None, ro=None):
        self.add_element(Bender(Bp, rb, rp, bending_angle, ap=ap, ro=ro))

    def add_combiner(self, L: RealNum, Bp: RealNum, rp: RealNum) -> None:
        self.add_element(Combiner(L, Bp, rp))

    def twiss_parameters(self, atom_speed=DEFAULT_ATOM_SPEED):
        return twiss_paremeters_from_lattice(self.elements, atom_speed)

    def is_stable_xy(self, atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[bool, bool]:
        return is_stable_xy(self.elements, atom_speed)

    def stability_factors(self, atom_speed: Union[RealNum, sequence] = DEFAULT_ATOM_SPEED,
                          clip_to_positive: bool = False):
        fact_x, fact_y = stability_factors_lattice(self.elements, atom_speed)
        if clip_to_positive:
            return np.clip(fact_x, 0.0, np.inf), np.clip(fact_y, 0.0, np.inf)
        else:
            return fact_x, fact_y

    def M(self, atom_speed: float = DEFAULT_ATOM_SPEED, s: RealNum = None) -> ThreeNumpyMatrices:
        if s is None:
            Mx, My, Md = self.M_total(atom_speed=atom_speed)
        else:
            Mx, My, Md = lattice_transfer_matrix_at_s(s, self.elements, atom_speed)
        return Mx, My, Md

    def M_total(self, atom_speed: float = DEFAULT_ATOM_SPEED):
        Mx, My, Md = total_lattice_transfer_matrix(self.elements, atom_speed)
        return Mx, My, Md

    def M_total_components(self, atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[tuple[float, ...], ...]:
        Mx, My, Md = self.M(atom_speed=atom_speed)
        x_components = matrix_components(Mx)
        y_components = matrix_components(My)
        d_components = matrix_components(Md)
        return x_components, y_components, d_components

    def acceptance_ellipses(self, atom_speed=DEFAULT_ATOM_SPEED, num_points=300):
        return acceptance_ellipses(self.elements, atom_speed, num_points)

    def beta_profiles(self, atom_speed: RealNum = DEFAULT_ATOM_SPEED, num_points: int = 300):
        s_vals, (beta_x, beta_y) = beta_profile(self.elements, atom_speed, num_points)
        return s_vals, (beta_x, beta_y)

    def alpha_profiles(self, atom_speed: RealNum = DEFAULT_ATOM_SPEED, num_points: int = 300):
        s_vals, (alphas_x, alphas_y) = alpha_profile(self.elements, atom_speed, num_points)
        return s_vals, (alphas_x, alphas_y)

    def gamma_profiles(self, atom_speed: RealNum = DEFAULT_ATOM_SPEED, num_points: int = 300):
        s_vals, (gammas_x, gammas_y) = gamma_profiles(self.elements, atom_speed, num_points)
        return s_vals, (gammas_x, gammas_y)

    def acceptance_profile(self, atom_speed=DEFAULT_ATOM_SPEED, num_points: int = 300):
        return acceptance_profile(self.elements, atom_speed, num_points)

    def dispersion_profile(self, atom_speed=DEFAULT_ATOM_SPEED, num_points: int = 300):
        return dispersion_profile(self.elements, atom_speed, num_points)

    def minimum_acceptance(self, atom_speed: RealNum = DEFAULT_ATOM_SPEED, num_points: int = 300):
        return minimum_acceptance(self.elements, atom_speed, num_points)

    def tunes_incremental(self, atom_speed: RealNum = DEFAULT_ATOM_SPEED):
        return tunes_incremental(self.elements, atom_speed)

    def tunes_absolute(self, atom_speed: RealNum = DEFAULT_ATOM_SPEED, num_points: int = 300):
        return tunes_absolute(self.elements, atom_speed, num_points)

    def trace(self, Xi, atom_speed: RealNum, which='y') -> ndarray:
        assert which in ('y', 'z')
        Mx, My, Md = self.M(atom_speed=atom_speed)
        M = Mx if which == 'y' else My
        Xf = M @ Xi
        if which == 'y':  # because orientation is clockwise in particle tracer lattices, and orientation generally
            # starts along -x, needs to change sign
            Xf *= -1
        return Xf

    def total_length(self) -> float:
        return total_length(self.elements)

    def add_matrix_elements_from_sim_lattice(self, simulated_lattice: ParticleTracerLattice,
                                             use_effective_elements=False) -> None:
        """Build the lattice from an existing ParticleTracerLattice object"""

        assert isclose(abs(simulated_lattice.initial_ang), np.pi, abs_tol=1e-9)  # must be pointing along -x
        # in polar coordinates
        assert not simulated_lattice.include_mag_errors
        for el in simulated_lattice:
            if type(el) is Drift_Sim:
                self.add_drift(el.L, ap=el.ap)
            elif type(el) is CombinerLensSim or (
                    type(el) in (HalbachLensSim, BenderSim) and not use_effective_elements):
                numeric_element = build_numeric_element(el)
                self.add_element(numeric_element)
            elif type(el) in (HalbachLensSim, BenderSim) and use_effective_elements:
                effective_elements = make_effective_elements(el)
                for el in effective_elements:
                    self.add_element(el)
            else:
                raise NotImplementedError

    def add_matrix_elements_from_ideal_lattice(self, ideal_lattice: ParticleTracerLattice):
        for el in ideal_lattice:
            if type(el) is LensIdeal:
                self.add_lens(el.L, el.Bp, el.rp, ap=el.ap)
            elif type(el) is Drift_Sim:
                self.add_drift(el.L, ap=el.ap)
            elif type(el) is BenderIdeal:
                self.add_bender(el.Bp, el.rb, el.rp, el.ang, ap=el.ap, ro=el.ro)
            else:
                raise NotImplementedError

    def predicted_swarm_flux(self, swarm: Swarm, T_max: RealNum, num_points_speed_range: int = 50):
        return swarm_flux_mult_from_matrix_method(swarm, self.elements, T_max, num_points_speed_range)

    def plot_swarm_survival_against_emittance(self, swarm_traced: Swarm, which_orbit_dim: str, save_title=None) -> None:
        atom_speeds = np.array(swarm_traced[:, 'pi', 0])
        assert isclose(np.std(atom_speeds), 0.0)  # must all be the same
        atom_speed = abs(atom_speeds[0])
        plot_swarm_survival_against_emittance(self.elements, swarm_traced, atom_speed, which_orbit_dim, save_title)

    def trace_swarm(self, swarm, copy_swarm=True, atom_speed=DEFAULT_ATOM_SPEED):
        assert len(self.elements) > 0
        if copy_swarm:
            swarm = swarm.copy()
        Mx, My, Md = self.M(atom_speed=atom_speed)
        L = self.total_length()
        directionality_signs = {1: -1.0, 2: 1.0}  # to square results with simulated lattice direction,
        # because particles are assumed to circulate clockwise, forces the transverse horizontal unit vector to have
        # opposite direction as the convential value (y vs -y)
        for particle in swarm:
            particle.qf, particle.pf = np.zeros(3), np.zeros(3)
            particle.qf[0] = L
            particle.pf[0] = atom_speed
            for idx, M in zip([1, 2], [Mx, My]):
                sign = directionality_signs[idx]
                pos_i, v_i = particle.qi[idx] * sign, particle.pi[idx] * sign
                slope_i = v_i / atom_speed
                Qi = [pos_i, slope_i]
                pos_f, slope_f = M @ Qi
                particle.qf[idx] = pos_f
                particle.pf[idx] = slope_f * atom_speed
        return swarm
