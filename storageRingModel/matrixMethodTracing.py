import copy
from math import sqrt, cos, sin, cosh, sinh
from typing import Iterable, Union, Optional, Callable

import numpy as np

from HalbachLensClass import Collection, SegmentedBenderHalbach
from ParticleTracerLatticeClass import ParticleTracerLattice
from constants import SIMULATION_MAGNETON, SIMULATION_MASS, DEFAULT_ATOM_SPEED
from helperTools import multiply_matrices
from latticeElements.class_HalbachBenderSegmented import mirror_across_angle
from latticeElements.elements import Drift as Drift_Sim
from latticeElements.elements import HalbachLensSim, HalbachBender, CombinerHalbachLensSim
from latticeElements.orbitTrajectories import combiner_halbach_orbit_coords_el_frame
from typeHints import ndarray, RealNum
from typeHints import sequence

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
            if type(el_sim) is HalbachBender and type(el) is type(el_sim):
                if el.rp == el_sim.rp and el.num_magnets == el_sim.num_magnets and el.ro == el_sim.ro and el.ucAng == el_sim.ucAng:
                    return self.elements_matrix[i]
            elif type(el_sim) is HalbachLensSim and type(el) is type(el_sim):
                if el.rp == el_sim.rp and el.L == el_sim.L and el.Lm == el_sim.Lm:
                    return self.elements_matrix[i]
        return None


def transfer_matrix(K: RealNum, L: RealNum) -> ndarray:
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

    return np.array([[A, B], [C, D]])


def magnet_force(magnets, coords):
    """Force in magnet at given point"""
    use_approx = True if type(magnets) is SegmentedBenderHalbach else False
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


def orbit_offset_bender(el: HalbachBender, speed_ratio):
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
        if type(el) is HalbachBender:
            if orbit_offset_bender(el, speed_ratio) > el.ap:
                return False
    return True


def bender_spring_constant(Bp: RealNum, rb: RealNum, rp: RealNum, ro: RealNum, atom_speed: RealNum) -> float:
    """Spring constant (F=-Kx) for bender's harmonic potential"""
    assert Bp >= 0.0 and 0.0 < rp < rb and rb > 0.0
    term1 = SIMULATION_MASS / ro ** 2  # centrifugal term
    term2 = 2 * SIMULATION_MAGNETON * Bp / (SIMULATION_MASS * (atom_speed * rp) ** 2)  # magnetic term
    K = term1 + term2
    return K


def spring_constant_lens(Bp: RealNum, rp: RealNum, atom_speed: RealNum) -> float:
    """Spring constant (F=-Kx) for lens's harmonic potential"""
    assert rp > 0.0
    K = 2 * SIMULATION_MAGNETON * Bp / (SIMULATION_MASS * (atom_speed * rp) ** 2)
    return K


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


def K_mag_vals_and_slice_lengths_from_lens_el(el: HalbachLensSim) -> tuple[sequence, sequence]:
    """Magnetic spring constants and lengths of slices along the length of the lens. used to construct short transfer
    matrice at each slice"""
    magnets = el.magnet.make_magpylib_magnets(False, False)
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
    return slice_lengths, K_vals_mag


def transfer_matrix_func_from_lens(el: HalbachLensSim) -> Callable:
    """Create equivalent transfer matric from lens. Built by slicing lens into short transfer matrices"""
    slice_lengths, K_vals_mag = K_mag_vals_and_slice_lengths_from_lens_el(el)

    def M_func(atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        Mx = np.eye(2)
        My = np.eye(2)
        for [Kx_mag, Ky_mag], L in zip(K_vals_mag, slice_lengths):
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
    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])
    R = abs(dL / np.tan(theta2 - theta1))
    return R


def combiner_K_mag_and_R_vals(coords_path: ndarray, p_path: ndarray, el: CombinerHalbachLensSim) -> tuple[list, list]:
    """Mganetic spring constants and bending radius along path of trajectory through combiner."""
    magnets = el.magnet.make_magpylib_magnets(False, False)
    norms_xo_path = unit_vec_perp_to_path(coords_path)
    xo_max = min([(el.rp - el.output_offset), el.output_offset])
    num_points = 11
    xo_vals = np.linspace(-xo_max / 5.0, xo_max / 5.0, num_points)
    yo_max = sqrt(el.rp ** 2 - el.output_offset ** 2)
    yo_vals = np.linspace(-yo_max / 5.0, yo_max / 5.0, num_points)
    K_vals_mag = [combiner_K_mags_for_slice(coord, norm, magnets, xo_vals, yo_vals) for coord, norm in
                  zip(coords_path, norms_xo_path)]
    R_vals = [bending_radius(idx, p_path, coords_path) for idx in range(len(K_vals_mag))]
    return K_vals_mag, R_vals


def transfer_matrix_func_from_combiner(el: CombinerHalbachLensSim) -> Callable:
    """Compute the transfer matric for the combiner. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    coords_path, p_path = combiner_halbach_orbit_coords_el_frame(el)
    coords_path, p_path = np.flip(coords_path, axis=0), np.flip(p_path, axis=0)
    speeds_path = np.linalg.norm(p_path, axis=1)

    K_vals_mag, R_vals = combiner_K_mag_and_R_vals(coords_path, p_path, el)

    def M_func(atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        Mx = np.eye(2)
        My = np.eye(2)
        speed_factor = atom_speed / DEFAULT_ATOM_SPEED
        for idx, [[Kx_mag, Ky_mag], R, speed] in enumerate(zip(K_vals_mag, R_vals, speeds_path)):
            L = combiner_slice_length_at_traj_index(idx, coords_path)
            speed *= speed_factor
            Kx = Kx_mag / speed ** 2
            Ky = Ky_mag / speed ** 2
            R *= speed_factor ** 2
            K_cent = 1 / R ** 2
            Kx += K_cent
            # IMPROVEMENT: Should the length change depending on the speed factor and the bending radius?
            Mx = transfer_matrix(Kx, L) @ Mx
            My = transfer_matrix(Ky, L) @ My
        return Mx, My

    return M_func


def xo_unit_vector_bender_el_frame(el: HalbachBender, coord: ndarray) -> ndarray:
    """get unit vector pointing along xo in the bender orbit frame"""
    which_section = el.in_which_section_of_bender(coord)
    if which_section == 'ARC':
        theta = np.arctan2(coord[1], coord[0])
        norm = np.array([np.cos(theta), np.sin(theta), 0.0])
    elif which_section == 'OUT':
        norm = np.array([1.0, 0.0, 0.0])
    elif which_section == 'IN':
        nx, ny = mirror_across_angle(1.0, 0, el.ang / 2.0)
        norm = np.array([nx, ny, 0.0])
    else:
        raise ValueError
    return norm


def speed_with_energy_correction(V: RealNum, atom_speed: RealNum) -> float:
    """Energy conservation for atom speed accounting for magnetic fields"""
    E0 = .5 * atom_speed ** 2
    KE = E0 - V
    speed_corrected = np.sqrt(2 * KE)
    return speed_corrected


def fit_k_to_force(pos_vals: sequence, force: sequence) -> float:
    """given position and force values, find the spring constant"""
    m = np.polyfit(pos_vals, force, 1)[0]
    K = -m
    return K


def bender_K_mag_xo(el: HalbachBender, s: RealNum, magnets: Collection, xo_max: RealNum, num_samples: int) -> float:
    """Spring constant in bender along xo axis"""
    xo_vals = np.linspace(-xo_max / 4.0, xo_max / 4.0, num_samples)
    coords = np.array([el.convert_orbit_to_cartesian_coords(s, xo, 0.0) for xo in xo_vals])
    forces = magnet_force(magnets, coords)
    norm_perps = xo_unit_vector_bender_el_frame(el, coords[0])
    forces_r = [np.dot(norm_perps, force) for force in forces]
    K = fit_k_to_force(xo_vals, forces_r)
    return K


def bender_K_mag_yo(el: HalbachBender, s: RealNum, magnets: Collection, zo_max: RealNum, num_samples: int) -> float:
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


def stitch_bender_values(values_start: sequence, values_uc: sequence, num_internal_mags: int) -> list:
    """Given spring constant, or physically similiar values, from the fringing and symmetry region, construct a list
    that represents the entire bender"""
    values_internal = [*values_uc, *reversed(values_uc)] * num_internal_mags
    values_full = [*values_start, *values_internal, *reversed(values_start)]
    return values_full


def K_mag_vals_and_lengths_from_bender_el(el: HalbachBender) -> tuple[ndarray, ndarray, list, list]:
    """Magnetic spring constants and lengths at each spring constant value along the orbit trajectory through the
    bender"""
    magnets = el.build_bender(True, (True, False), num_lenses=NUM_FRINGE_MAGNETS_MIN * 3)
    s_slices_fringe, s_slices_uc, s_slices_total, lengths_total = s_slices_and_lengths_bender(el)
    num_internal_mags = (el.num_magnets - 2 * NUM_FRINGE_MAGNETS_MIN)
    K_vals_mag_uc = np.array([bender_K_mag_for_slice(s, magnets, el) for s in s_slices_uc])
    Kx_vals_mag_uc, Ky_vals_mag_uc = K_vals_mag_uc.T
    K_vals_mag_start = np.array([bender_K_mag_for_slice(s, magnets, el) for s in s_slices_fringe])
    Kx_vals_mag_start, Ky_vals_mag_start = K_vals_mag_start.T
    Kx_vals_mag = stitch_bender_values(Kx_vals_mag_start, Kx_vals_mag_uc, num_internal_mags)
    Ky_vals_mag = stitch_bender_values(Ky_vals_mag_start, Ky_vals_mag_uc, num_internal_mags)
    return s_slices_total, lengths_total, Kx_vals_mag, Ky_vals_mag


def transfer_matrix_func_from_bender(el: HalbachBender) -> Callable:
    """Return the transfer matric for the bender. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    if el.num_magnets < 2 * NUM_FRINGE_MAGNETS_MIN:
        raise NotImplementedError
    s_slices_total, lengths_total, Kx_vals_mag, Ky_vals_mag = K_mag_vals_and_lengths_from_bender_el(el)

    def M_func(atom_speed: RealNum) -> tuple[ndarray, ndarray]:
        Mx = np.eye(2)
        My = np.eye(2)
        for s, L, Kx_mag, Ky_mag in zip(s_slices_total, lengths_total, Kx_vals_mag, Ky_vals_mag):
            orbit_offset = (el.ro - el.rb) * (atom_speed / DEFAULT_ATOM_SPEED) ** 2
            V = .5 * Kx_mag * orbit_offset ** 2
            atom_speed_corrected = speed_with_energy_correction(V, atom_speed)
            Kx = Kx_mag / atom_speed_corrected ** 2
            Ky = Kx_mag / atom_speed_corrected ** 2
            if el.L_cap < s < el.ang * el.ro + el.L_cap:
                r_orbit = el.rb + orbit_offset
                Kx += 1 / r_orbit ** 2
                L *= r_orbit / el.ro
            Mx = transfer_matrix(Kx, L) @ Mx
            My = transfer_matrix(Ky, L) @ My
        return Mx, My

    return M_func


class Element:
    """ Base element representing a transfer matrix (ABCD matrix)"""

    def __init__(self, L: RealNum, M_func: Callable = None):
        assert L > 0.0
        self.M_func = M_func
        self.L = L

    def M(self, atom_speed=DEFAULT_ATOM_SPEED) -> tuple[ndarray, ndarray]:
        if self.M_func is None:
            raise NotImplementedError
        else:
            Mx, My = self.M_func(atom_speed)
            return Mx, My


class Lens(Element):
    """Element representing a lens"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum, ):
        # K = spring_constant_lens(Bp, rp, DEFAULT_ATOM_SPEED)
        super().__init__(L)


class Combiner(Element):
    """Element representing a combiner"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum):
        # K = spring_constant_lens(Bp, rp, DEFAULT_ATOM_SPEED)
        super().__init__(L)


class Drift(Element):
    """Element representing free space"""

    def __init__(self, L: RealNum):
        super().__init__(L)

    def M(self, atom_speed=DEFAULT_ATOM_SPEED):
        Mx = transfer_matrix(0.0, self.L)
        My = Mx.copy()
        return Mx, My


class Bender(Element):
    """Element representing a bending component"""

    def __init__(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum):
        self.ro = bender_orbit_radius_no_energy_correction(Bp, rb, rp, DEFAULT_ATOM_SPEED)
        # K = bender_spring_constant(Bp, rb, rp, self.ro, DEFAULT_ATOM_SPEED)
        L = self.ro * bending_angle  # length of particle orbit
        super().__init__(L)


MatrixLatticeElement = Union[Drift, Lens, Bender, Combiner, Element]


def full_transfer_matrices(elements: Iterable[MatrixLatticeElement],
                           atom_speed=DEFAULT_ATOM_SPEED) -> tuple[ndarray, ndarray]:
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


def stability_factors_lattice(elements: Iterable[MatrixLatticeElement],
                              atom_speed: Union[RealNum, sequence] = DEFAULT_ATOM_SPEED):
    """Factor describing stability of periodic lattice of elements. If value is greater than 0 it is stable, though
    higher values are "more" stable in some sense"""
    speeds = [atom_speed] if isinstance(atom_speed, (int, float)) else atom_speed

    matrices_full_xy = [full_transfer_matrices(elements, atom_speed=speed) for speed in speeds]
    stability_factors_x = [stability_factor(*matrix_components(Mx)) for Mx, My in matrices_full_xy]
    stability_factors_y = [stability_factor(*matrix_components(My)) for Mx, My in matrices_full_xy]
    if isinstance(atom_speed, (int, float)):
        return stability_factors_x[0], stability_factors_y[0]
    else:
        return stability_factors_x, stability_factors_y


def is_stable_lattice(elements: Iterable[MatrixLatticeElement],
                      atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[bool, bool]:
    """Determine if the lattice is stable. This can be done by computing eigenvalues, or the method below works. If
    unstable, then raising he transfer matrix to N results in large matrix elements for large N"""
    stability_factor_x, stability_factor_y = stability_factors_lattice(elements, atom_speed=atom_speed)
    return stability_factor_x > 0.0, stability_factor_y > 0.0


def total_length(elements: Iterable[MatrixLatticeElement]) -> float:
    """Sum of the lengths of elements"""
    length = sum([el.L for el in elements])
    return length


class Lattice:
    """Model of a sequence (possibly periodic) of elements"""

    def __init__(self):
        self.elements: Optional[list[MatrixLatticeElement]] = []

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def add_drift(self, L: RealNum) -> None:
        self.elements.append(Drift(L))

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

    def M_total(self, revolutions: int = 1, atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[ndarray, ndarray]:
        Mx_single_rev, My_single_rev = full_transfer_matrices(self.elements, atom_speed=atom_speed)
        Mx = np.linalg.matrix_power(Mx_single_rev, revolutions)
        My = np.linalg.matrix_power(My_single_rev, revolutions)
        return Mx, My

    def M_total_components(self, atom_speed: float = DEFAULT_ATOM_SPEED) -> tuple[tuple[float, ...], tuple[float, ...]]:
        Mx, My = self.M_total(atom_speed=atom_speed)
        x_components = matrix_components(Mx)
        y_components = matrix_components(My)
        return x_components, y_components

    def trace(self, Xi) -> ndarray:
        raise NotImplementedError
        return self.M_total() @ Xi

    def total_length(self) -> float:
        return total_length(self.elements)

    def add_element_from_sim_lens(self, el_lens: HalbachLensSim) -> None:
        M_func = transfer_matrix_func_from_lens(el_lens)
        self.elements.append(Element(el_lens.L, M_func=M_func))

    def add_element_from_sim_seg_bender(self, el_bend: HalbachBender) -> None:
        M_func = transfer_matrix_func_from_bender(el_bend)
        self.elements.append(Element(el_bend.Lo, M_func=M_func))

    def add_element_from_sim_combiner(self, el_combiner: CombinerHalbachLensSim) -> None:
        M_func = transfer_matrix_func_from_combiner(el_combiner)
        self.elements.append(Element(el_combiner.L, M_func=M_func))

    def build_matrix_lattice_from_sim_lattice(self, simulated_lattice: ParticleTracerLattice) -> None:
        """Build the lattice from an existing ParticleTracerLattice object"""
        if simulated_lattice.lattice_type != 'storage_ring':
            raise NotImplementedError

        reuser = ElementRecycler()  # saves time to recycle the matrices for the same elements
        for el in simulated_lattice:
            el_matrix_reusable = reuser.reusable_matrix_el(el)
            if el_matrix_reusable is not None:
                el_matrix_reusable = copy.deepcopy(el_matrix_reusable)
                self.elements.append(el_matrix_reusable)
            else:
                if type(el) is Drift_Sim:
                    self.add_drift(el.L)
                elif type(el) is HalbachLensSim:
                    self.add_element_from_sim_lens(el)
                elif type(el) is HalbachBender:
                    self.add_element_from_sim_seg_bender(el)
                elif type(el) is CombinerHalbachLensSim:
                    self.add_element_from_sim_combiner(el)
                else:
                    raise NotImplementedError
                reuser.add_sim_and_matrix(el, self.elements[-1])

    def trace_swarm(self, swarm, revolutions=1, copy_swarm=True):
        raise NotImplementedError
        assert revolutions > 0 and isinstance(revolutions, int)
        assert len(self.elements) > 0
        if copy_swarm:
            swarm = swarm.copy()
        M_tot = self.M_total(revolutions=revolutions)
        directionality_sign = -1.0  # particle are assumed to being launched leftwards
        for particle in swarm:
            xo_i, pxo = particle.qi[1] * directionality_sign, particle.pi[1] * directionality_sign
            slope_xo_i = pxo / DEFAULT_ATOM_SPEED
            Xi = [xo_i, slope_xo_i]
            xo_f, slope_xo_f = M_tot @ Xi
            particle.qf, particle.pf = np.zeros(3), np.zeros(3)
            particle.qf[1] = xo_f
            particle.pf[1] = slope_xo_f * DEFAULT_ATOM_SPEED
        return swarm
