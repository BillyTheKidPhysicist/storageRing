import copy
from collections.abc import Sequence
from math import sqrt, cos, sin, cosh, sinh, tan, acos, atan2, isclose
from typing import Iterable, Union, Optional, Callable

import numba
import numpy as np

from constants import SIMULATION_MAGNETON, SIMULATION_MASS, DEFAULT_ATOM_SPEED
from field_generators import BenderSim as HalbachBender_FieldGenerator
from field_generators import Collection
from helper_tools import multiply_matrices
from lattice_elements.bender_sim import mirror_across_angle, speed_with_energy_correction
from lattice_elements.elements import Drift as Drift_Sim
from lattice_elements.elements import HalbachLensSim, BenderSim, CombinerLensSim
from lattice_elements.orbit_trajectories import combiner_orbit_coords_el_frame
from particle_tracer_lattice import ParticleTracerLattice
from type_hints import ndarray, RealNum
from type_hints import sequence

SMALL_ROUNDING_NUM: float = 1e-14
NUM_FRINGE_MAGNETS_MIN: int = 5  # number of fringe magnets to accurately model internal behaviour of bender


class NonViableLattice(Exception):
    pass


class ElementRecycler:
    """Class to assist with recycling transfer matrices generated from identical lattice elements. This saves 
    significant time"""

    def __init__(self):
        self.elements_sim = []
        self.elements_matrix = []

    def add_sim_and_matrix(self, el_sim, el_matrix):
        self.elements_sim.append(el_sim)
        self.elements_matrix.append(el_matrix)

    def reusable_matrix_el(self, el_sim):
        for i, el in enumerate(self.elements_sim):
            if type(el_sim) is BenderSim and type(el) is type(el_sim):
                if el.rp == el_sim.rp and el.num_magnets == el_sim.num_magnets and el.ro == el_sim.ro and el.ucAng == el_sim.ucAng:
                    return self.elements_matrix[i]
            elif type(el_sim) is HalbachLensSim and type(el) is type(el_sim):
                if el.rp == el_sim.rp and el.L == el_sim.L and el.Lm == el_sim.Lm:
                    return self.elements_matrix[i]
        return None


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
def transfer_matrix(K: RealNum, L: RealNum, K_dispersion=None) -> ndarray:
    """Build the 2x2 transfer matrix (ABCD matrix) for an element that represents a harmonic potential.
    Transfer matrix is for one dimension only"""
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
    if K_dispersion is not None:
        phi = sqrt(K) * L
        E = (1 - cos(phi)) / sqrt(K_dispersion)
        F = sin(phi)
        M = np.array([[A, B, E], [C, D, F], [0, 0, 1]])
    else:
        M = np.array([[A, B], [C, D]])
    return M


def magnet_force(magnets, coords):
    """Force in magnet at given point"""
    use_approx = True if type(magnets) is HalbachBender_FieldGenerator else False
    B_norm_grad = magnets.B_norm_grad(coords, use_approx=use_approx, dx=1e-6, diff_method='central')
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
    speed_ratio = atom_speed / simulated_lattice.speed_nominal
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


def bender_dispersion(ro):
    return 2 / ro


def matrix_components(M: ndarray) -> tuple[RealNum, RealNum, RealNum, RealNum]:
    m11 = M[0, 0]
    m12 = M[0, 1]
    m21 = M[1, 0]
    m22 = M[1, 1]
    return m11, m12, m21, m22


def lens_transfer_matrix_for_slice(rp: RealNum, magnets: Collection, x_slice: RealNum,
                                   slice_length: RealNum, atom_speed: RealNum) -> ndarray:
    num_samples = 30
    x_vals = np.ones(num_samples) * x_slice
    y_vals = np.linspace(-rp / 4.0, rp / 4.0, num_samples)
    z_vals = np.zeros(num_samples)
    coords = np.column_stack((x_vals, y_vals, z_vals))
    F_y = magnet_force(magnets, coords)[:, 1]
    m = np.polyfit(y_vals, F_y, 1)[0]
    K = -m / atom_speed ** 2
    M_slice = transfer_matrix(K, slice_length)
    return M_slice


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
    num_samples = 30
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
    num_slices_per_bore_rad = 100
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

    @numba.njit()
    def M_func(s0, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        # IMPROVEMENT: this method is really slow for some things I'm doing. I should return a list of matrices
        assert 0 <= s0 <= s_max
        s = 0
        Mx = np.eye(2)
        My = np.eye(2)
        for [Kx_mag, Ky_mag], L in zip(K_vals_mag, slice_lengths):
            s += L
            if s > s0 + SMALL_ROUNDING_NUM:
                break
            Kx = Kx_mag / atom_speed ** 2
            Ky = Ky_mag / atom_speed ** 2
            Mx = transfer_matrix(Kx, L) @ Mx
            My = transfer_matrix(Ky, L) @ My
        return Mx, My

    return M_func


def split_range_into_slices(x_min: RealNum, x_max: RealNum, num_slices: int) -> tuple[ndarray, float]:
    """Split a range into equally spaced points, that are half a spacing away from start and end"""
    slice_length = (x_max - x_min) / num_slices
    x_slices = np.linspace(x_min, x_max, num_slices + 1)[:-1]
    x_slices += slice_length / 2.0
    return x_slices, slice_length


def unit_vec_perp_to_path(path_coords: ndarray) -> ndarray:
    """Along the path path_coords, find the perpindicualr normal vector for each point. Assume the path is
    counter-clockwise, and the normal point radially outwards
    """
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
    num_points = 11
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

    K_vals_mag, R_vals = combiner_K_mag_and_R_vals(coords_path, p_path, el)
    s_max = el.Lo

    @numba.njit()
    def M_func(s0, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        assert 0 <= s0 <= s_max
        s = 0
        Mx = np.eye(2)
        My = np.eye(2)
        speed_factor = atom_speed / DEFAULT_ATOM_SPEED
        for idx, [[Kx_mag, Ky_mag], R, speed, L] in enumerate(zip(K_vals_mag, R_vals, speeds_path, lengths)):
            s += L
            if s > s0 + SMALL_ROUNDING_NUM:
                break

            speed *= speed_factor
            Kx = Kx_mag / speed ** 2
            Ky = Ky_mag / speed ** 2
            R *= speed_factor ** 2
            K_cent = 3 / R ** 2
            Kx += K_cent
            # IMPROVEMENT: Should the length change depending on the speed factor and the bending radius?
            Mx = transfer_matrix(Kx, L) @ Mx
            My = transfer_matrix(Ky, L) @ My
        return Mx, My

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
    num_slices_per_bore_radius = 8
    num_slices_fringe = round(num_slices_per_bore_radius * s_fringe_depth / el.rp)
    num_slices_uc = round(num_slices_per_bore_radius * s_unit_cell / el.rp)
    s_slices_fringe, _ = split_range_into_slices(0.0, s_fringe_depth, num_slices_fringe)
    s_slices_uc, _ = split_range_into_slices(s_fringe_depth, s_fringe_depth + s_unit_cell, num_slices_uc)

    s_internal_start = s_fringe_depth
    s_total_length = el.Lo
    s_internal_stop = s_total_length - s_fringe_depth
    num_internal_mags = (el.num_magnets - 2 * NUM_FRINGE_MAGNETS_MIN)
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
    num_samples = 11
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
    magnets = el.magnet.magpylib_magnets_model()
    s_slices_fringe, s_slices_uc, s_slices_total, lengths_total = s_slices_and_lengths_bender(el)
    num_internal_mags = (el.num_magnets - 2 * NUM_FRINGE_MAGNETS_MIN)
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
    if el.num_magnets < 2 * NUM_FRINGE_MAGNETS_MIN:
        raise NotImplementedError
    s_slices_total, lengths_total, Kx_vals_mag, Ky_vals_mag = K_mag_vals_and_lengths_from_bender_el(el)
    s_max, ro, rb, L_cap, ang = el.Lo, el.ro, el.rb, el.L_cap, el.ang

    @numba.njit()
    def M_func(s0, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        assert 0 <= s0 <= s_max
        Mx = np.eye(2)
        My = np.eye(2)
        for s, L, Kx_mag, Ky_mag in zip(s_slices_total, lengths_total, Kx_vals_mag, Ky_vals_mag):
            if s > s0 + SMALL_ROUNDING_NUM:
                break
            orbit_offset = (ro - rb) * (atom_speed / DEFAULT_ATOM_SPEED) ** 2
            V = .5 * Kx_mag * orbit_offset ** 2
            atom_speed_corrected = speed_with_energy_correction(V, atom_speed)
            Kx = Kx_mag / atom_speed_corrected ** 2
            Ky = Ky_mag / atom_speed_corrected ** 2

            if L_cap < s < ang * ro + L_cap:
                r_orbit = rb + orbit_offset
                K_cent = 3.0 / r_orbit ** 2
                Kx += K_cent
            Mx = transfer_matrix(Kx, L) @ Mx
            My = transfer_matrix(Ky, L) @ My
        return Mx, My

    return M_func


class Element:
    """ Base element representing a transfer matrix (ABCD matrix)"""

    def __init__(self, L: RealNum, ap: RealNum = np.inf):
        assert L > 0.0
        self.L = L
        self.ap = ap

    def M_func(self, s: RealNum, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        """Return the x and y transfer matrices at a given location 's' inside the element for an atom with longitudinal
        velocity 'atom_speed'"""
        raise NotImplementedError

    def M(self, atom_speed=DEFAULT_ATOM_SPEED, s=None) -> tuple[ndarray, ndarray]:
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

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum, ):
        self.Bp = Bp
        self.rp = rp
        super().__init__(L, ap=rp)

    def M_func(self, s: RealNum, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        assert 0 <= s <= self.L
        K = spring_constant_lens(self.Bp, self.rp, atom_speed)
        Mx = My = transfer_matrix(K, s)
        return Mx, My


class Combiner(Element):
    """Element representing a combiner"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum):
        self.Bp = Bp
        self.rp = rp
        super().__init__(L)

    def M_func(self, s: RealNum, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        assert 0 <= s <= self.L
        K = spring_constant_lens(self.Bp, self.rp, atom_speed)
        Mx = My = transfer_matrix(K, s)
        return Mx, My


class Drift(Element):
    """Element representing free space"""

    def __init__(self, L: RealNum, ap):
        super().__init__(L, ap=ap)

    def M_func(self, s: RealNum, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        assert 0 <= s <= self.L
        Mx = transfer_matrix(0.0, s)
        My = Mx.copy()
        return Mx, My


class Bender(Element):
    """Element representing a bending component"""

    def __init__(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum):
        self.ro = bender_orbit_radius_no_energy_correction(Bp, rb, rp, DEFAULT_ATOM_SPEED)
        self.rb = rb
        self.rp = rp
        self.Bp = Bp
        L = self.ro * bending_angle  # length of particle orbit
        super().__init__(L, ap=self.rp)

    def M_func(self, s: RealNum, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        assert 0 <= s <= self.L
        Kx = bender_spring_constant(self.Bp, self.rp, self.ro, atom_speed)
        K_dispersion = bender_dispersion(self.ro)
        Ky = spring_constant_lens(self.Bp, self.rp, atom_speed)
        Mx = transfer_matrix(Kx, self.L)
        My = transfer_matrix(Ky, self.L)
        return Mx, My


def mult_el_matrices(Mx: ndarray, My: ndarray, el: Element,
                     atom_speed, s: RealNum = None) -> tuple[ndarray, ndarray]:
    """Given (x,y) transfer matrices chain new (x,y) transfer matrices from an element and return the result"""
    Mx_el, My_el = el.M(s=s, atom_speed=atom_speed)
    Mx = Mx_el @ Mx
    My = My_el @ My
    return Mx, My


def M_exit(el, s_start, atom_speed):
    """Return the transfer matrix representing exit(leaving) an element starting from position s_start. 
    The alogirhtm requires applying the inverse of the entrance portion to the entire matrix"""
    Mx_total, My_total = el.M(atom_speed=atom_speed)
    s_entrance = el.L - s_start
    Mx_entrance, My_entrance = el.M(s=s_entrance, atom_speed=atom_speed)
    Mx_exit = Mx_total @ np.linalg.inv(Mx_entrance)
    My_exit = My_total @ np.linalg.inv(My_entrance)
    return Mx_exit, My_exit


def lattice_transfer_matrix_at_s(s, elements: Sequence[Element], atom_speed) -> tuple[ndarray, ndarray]:
    """Find total transfer matrix from a point s in the lattice. Lattice is assumed to be periodic"""
    s = s % total_length(elements)
    length_cumulative = np.cumsum([el.L for el in elements])
    el_index = int(np.argmax(s < length_cumulative))
    el_inside = elements[el_index]
    delta_s = s - length_cumulative[el_index - 1] if el_index > 0 else s

    Mx, My = M_exit(el_inside, el_inside.L - delta_s, atom_speed)
    for el in elements[el_index + 1:]:
        Mx, My = mult_el_matrices(Mx, My, el, atom_speed)
    for el in elements[:el_index]:
        Mx, My = mult_el_matrices(Mx, My, el, atom_speed)
    Mx, My = mult_el_matrices(Mx, My, el_inside, atom_speed, s=delta_s)
    return Mx, My


def beta(M: ndarray) -> float:
    """Return the value of the beta function for a transfer matrix. Assumes periodicity"""
    m11, m12, m21, m22 = matrix_components(M)
    beta = 2 * m12 / np.sqrt(2 - m11 ** 2 - 2 * m12 * m21 - m22 ** 2)
    return beta


def betas_at_s(s: RealNum, elements: Sequence[Element], atom_speed: RealNum) -> tuple[float, float]:
    """Return the (x,y) beta function value at position s. Assumes periodicity"""
    Mx, My = lattice_transfer_matrix_at_s(s, elements, atom_speed)
    beta_x = beta(Mx)
    beta_y = beta(My)
    return beta_x, beta_y


def beta_profile(elements: Sequence[Element], atom_speed: RealNum,
                 num_points: int) -> tuple[ndarray, tuple[ndarray, ndarray]]:
    """Return (x,y) beta functions value along lattice. Also return the locations of the beta functions"""
    L = total_length(elements)
    s_vals = np.linspace(0, L, num_points)
    betas = np.array([betas_at_s(s, elements, atom_speed) for s in s_vals])
    betas_x, betas_y = np.abs(betas.T)
    return s_vals, (betas_x, betas_y)


def tunes_absolute(elements: Sequence[Element], atom_speed: RealNum = DEFAULT_ATOM_SPEED,
                   num_points: int = 1000) -> tuple[float, float]:
    """Return absolute value of tune. Calculated by integrating beta function profile along lattice"""
    s_vals, (betas_x, betas_y) = beta_profile(elements, atom_speed, num_points)
    tune_x = np.trapz(1 / betas_x, x=s_vals) / (2 * np.pi)
    tune_y = np.trapz(1 / betas_y, x=s_vals) / (2 * np.pi)
    return tune_x, tune_y


@make_atom_speed_kwarg_iterable
def tunes_incremental(elements: Iterable[Element],
                      atom_speed: RealNum = DEFAULT_ATOM_SPEED) -> tuple[float, float]:
    """Return tune value tune value, ie the total tune minus nearest half integer int. This is between 0 and .5 .
    This method cannot distinguish between a tune of 1.25 and 1.75, both would result in .25"""
    Mx, My = total_lattice_transfer_matrix(elements, atom_speed)
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


def total_lattice_transfer_matrix(elements: Iterable[Element], atom_speed) -> tuple[ndarray, ndarray]:
    """Transfer matrix for a sequence of elements start to end"""
    matrices_x_and_y = [el.M(atom_speed=atom_speed) for el in elements]
    matrices_x = [entry[0] for entry in matrices_x_and_y]
    matrices_y = [entry[1] for entry in matrices_x_and_y]
    Mx, My = multiply_matrices(matrices_x), multiply_matrices(matrices_y)
    return Mx, My


def stability_factor(m11, m12, m21, m22):
    """Factor describing stability of periodic lattice of elements. If value is greater than 1 it is stable, though
    higher values are "more" stable in some sense"""
    return 2.0 - (m11 ** 2 + 2 * m12 * m21 + m22 ** 2)


@make_atom_speed_kwarg_iterable
def stability_factors_lattice(elements: Iterable[Element],
                              atom_speed: Union[RealNum, Iterable] = DEFAULT_ATOM_SPEED):
    """Factor describing stability of periodic lattice of elements. If value is greater than 0 it is stable, though
    higher values are "more" stable in some sense"""

    Mx, My = total_lattice_transfer_matrix(elements, atom_speed=atom_speed)
    stability_factor_x = stability_factor(*matrix_components(Mx))
    stability_factor_y = stability_factor(*matrix_components(My))
    return stability_factor_x, stability_factor_y


def is_stable_lattice(elements: Iterable[Element],
                      atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[bool, bool]:
    """Determine if the lattice is stable. This can be done by computing eigenvalues, or the method below works. If
    unstable, then raising he transfer matrix to N results in large matrix elements for large N"""
    stability_factor_x, stability_factor_y = stability_factors_lattice(elements, atom_speed=atom_speed)
    return stability_factor_x > 0.0, stability_factor_y > 0.0


def is_stable_matrix(M: ndarray) -> bool:
    return stability_factor(*matrix_components(M)) >= 0


def total_length(elements: Iterable[Element]) -> float:
    """Sum of the lengths of elements"""
    length = sum([el.L for el in elements])
    return length


def lattice_aperture(s, elements: Sequence[Element]):
    """Return aperture of lattice of position s. Assumed to be same for both dimensions"""
    s = s % total_length(elements)
    length_cumulative = np.cumsum([el.L for el in elements])
    el_index = int(np.argmax(s < length_cumulative))
    return elements[el_index].ap


def acceptance_profile(elements: Sequence[Element], atom_speed=DEFAULT_ATOM_SPEED,
                       num_points=300) -> tuple[ndarray, tuple[ndarray, ndarray]]:
    """Return (x,y) profile for acceptance through the lattice, and the corresponding position values. Assumes periodic
    lattice. This is the maximum emittance that would survive at each point in the lattice."""
    s_vals, (beta_x, beta_y) = beta_profile(elements, atom_speed, num_points)
    aps = np.array([lattice_aperture(s, elements) for s in s_vals])
    acceptances_x = aps ** 2 / beta_x
    acceptances_y = aps ** 2 / beta_y
    return s_vals, (acceptances_x, acceptances_y)


@make_atom_speed_kwarg_iterable
def minimum_acceptance(elements: Sequence[Element], atom_speed: RealNum = DEFAULT_ATOM_SPEED,
                       num_points: int = 300):
    """Return the minimum acceptance value in lattice. """
    Mx, My = total_lattice_transfer_matrix(elements, atom_speed)
    if not is_stable_matrix(Mx) and not is_stable_matrix(My):
        return np.nan, np.nan
    else:
        _, (acceptances_x, acceptances_y) = acceptance_profile(elements, atom_speed=atom_speed, num_points=num_points)
        return np.min(acceptances_x), np.min(acceptances_y)


class Lattice(Sequence):
    """Model of a series (assumed to be periodic) of elements"""

    def __init__(self):
        self.elements: Optional[list[Element]] = []

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def add_drift(self, L: RealNum, ap=np.inf) -> None:
        self.elements.append(Drift(L, ap))

    def add_lens(self, L: RealNum, Bp: RealNum, rp: RealNum) -> None:
        self.elements.append(Lens(L, Bp, rp))

    def add_bender(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum):
        self.elements.append(Bender(Bp, rb, rp, bending_angle))

    def add_combiner(self, L: RealNum, Bp: RealNum, rp: RealNum) -> None:
        self.elements.append(Combiner(L, Bp, rp))

    def is_stable(self, atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[bool, bool]:
        return is_stable_lattice(self, atom_speed=atom_speed)

    def stability_factors(self, atom_speed: Union[RealNum, sequence] = DEFAULT_ATOM_SPEED,
                          clip_to_positive: bool = False):
        fact_x, fact_y = stability_factors_lattice(self, atom_speed=atom_speed)
        if clip_to_positive:
            return np.clip(fact_x, 0.0, np.inf), np.clip(fact_y, 0.0, np.inf)
        else:
            return fact_x, fact_y

    def M(self, atom_speed: float = DEFAULT_ATOM_SPEED, s=None) -> tuple[ndarray, ndarray]:
        if s is None:
            Mx, My = total_lattice_transfer_matrix(self.elements, atom_speed)
        else:
            Mx, My = lattice_transfer_matrix_at_s(s, self, atom_speed)
        return Mx, My

    def M_total_components(self, atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[tuple[float, ...], tuple[float, ...]]:
        Mx, My = self.M(atom_speed=atom_speed)
        x_components = matrix_components(Mx)
        y_components = matrix_components(My)
        return x_components, y_components

    def beta_profiles(self, atom_speed=DEFAULT_ATOM_SPEED, num_points=500):
        s_vals, (beta_x, beta_y) = beta_profile(self.elements, atom_speed=atom_speed, num_points=num_points)
        return s_vals, (beta_x, beta_y)

    def acceptance_profile(self, atom_speed=DEFAULT_ATOM_SPEED, num_points=300):
        return acceptance_profile(self, atom_speed=atom_speed, num_points=num_points)

    def minimum_acceptance(self, atom_speed=DEFAULT_ATOM_SPEED, num_points=300):
        return minimum_acceptance(self, atom_speed=atom_speed, num_points=num_points)

    def tunes_incremental(self, atom_speed=DEFAULT_ATOM_SPEED):
        return tunes_incremental(self.elements, atom_speed=atom_speed)

    def trace(self, Xi, atom_speed=DEFAULT_ATOM_SPEED, which='y') -> ndarray:
        assert which in ('y', 'z')
        Mx, My = self.M(atom_speed=atom_speed)
        M = Mx if which == 'y' else My
        Xf = M @ Xi
        if which == 'y':  # because orientation is clockwise in particle tracer lattices, and orientation generally
            # starts along -x, needs to change sign
            Xf *= -1
        return Xf

    def total_length(self) -> float:
        return total_length(self.elements)

    def add_element_from_sim_lens(self, el_lens: HalbachLensSim) -> None:
        M_func = transfer_matrix_func_from_lens(el_lens)
        self.elements.append(NumericElement(el_lens.Lo, M_func=M_func, ap=el_lens.ap))

    def add_element_from_sim_seg_bender(self, el_bend: BenderSim) -> None:
        M_func = transfer_matrix_func_from_bender(el_bend)
        ap = el_bend.ap - (el_bend.ro - el_bend.rb)
        self.elements.append(NumericElement(el_bend.Lo, M_func=M_func, ap=ap))

    def add_element_from_sim_combiner(self, el_combiner: CombinerLensSim) -> None:
        M_func = transfer_matrix_func_from_combiner(el_combiner)
        ap = el_combiner.ap - el_combiner.output_offset
        self.elements.append(NumericElement(el_combiner.Lo, M_func=M_func, ap=ap))

    def build_matrix_lattice_from_sim_lattice(self, simulated_lattice: ParticleTracerLattice) -> None:
        """Build the lattice from an existing ParticleTracerLattice object"""

        assert isclose(abs(simulated_lattice.initial_ang), np.pi)  # must be pointing along -x in polar coordinates

        reuser = ElementRecycler()  # saves time to recycle the matrices for the same elements
        for el in simulated_lattice:
            el_matrix_reusable = reuser.reusable_matrix_el(el)
            if el_matrix_reusable is not None:
                el_matrix_reusable = copy.deepcopy(el_matrix_reusable)
                self.elements.append(el_matrix_reusable)
            else:
                if type(el) is Drift_Sim:
                    self.add_drift(el.L, ap=el.ap)
                elif type(el) is HalbachLensSim:
                    self.add_element_from_sim_lens(el)
                elif type(el) is BenderSim:
                    self.add_element_from_sim_seg_bender(el)
                elif type(el) is CombinerLensSim:
                    self.add_element_from_sim_combiner(el)
                else:
                    raise NotImplementedError
                reuser.add_sim_and_matrix(el, self.elements[-1])

    def trace_swarm(self, swarm, copy_swarm=True, atom_speed=DEFAULT_ATOM_SPEED):
        assert len(self.elements) > 0
        if copy_swarm:
            swarm = swarm.copy()
        Mx, My = self.M(atom_speed=atom_speed)
        L = self.total_length()
        directionality_signs = {1: -1.0,
                                2: 1.0}  # to square results with simulated lattice direction, which because particles are
        # assumed to circulate clockwise, forces the transverse horizontal unit vector to have opposite direction
        # as the convential value (y vs -y)
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
