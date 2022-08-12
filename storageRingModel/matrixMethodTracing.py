import copy
import itertools
from math import sqrt, cos, sin, cosh, sinh
from typing import Iterable, Union
from typing import Optional

import numpy as np

from HalbachLensClass import Collection
from ParticleTracerLatticeClass import ParticleTracerLattice
from constants import SIMULATION_MAGNETON, SIMULATION_MASS, DEFAULT_ATOM_SPEED
from helperTools import multiply_matrices
from latticeElements.class_HalbachBenderSegmented import mirror_across_angle
from latticeElements.elements import Drift as Drift_Sim
from latticeElements.elements import HalbachLensSim, HalbachBenderSimSegmented, CombinerHalbachLensSim
from latticeElements.orbitTrajectories import combiner_halbach_orbit_coords_el_frame
from typeHints import ndarray, RealNum

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

    def add_sim_and_matrx(self, el_sim, el_matrix):
        self.elements_sim.append(el_sim)
        self.elements_matrix.append(el_matrix)

    def reusable_matrix_el(self, el_sim):
        for i, el in enumerate(self.elements_sim):
            if type(el_sim) is HalbachBenderSimSegmented and type(el) is type(el_sim):
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


def orbit_offset_bender(el: HalbachBenderSimSegmented, speed_ratio):
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
    speed_ratio = atom_speed / simulated_lattice.speed_nominal
    for el in simulated_lattice:
        if type(el) is HalbachBenderSimSegmented:
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


def full_transfer_matrix(elements: Iterable) -> ndarray:
    """Transfer matrix for a sequence of elements start to end"""
    matrices = [el.M for el in elements]
    return multiply_matrices(matrices)


def matrix_components(M):
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
    F_y = -SIMULATION_MAGNETON * magnets.B_norm_grad(coords)[:, 1]
    m = np.polyfit(y_vals, F_y, 1)[0]
    K = -m / atom_speed ** 2
    M_slice = transfer_matrix(K, slice_length)
    return M_slice


def s_slices_and_lengths_lens(el: HalbachLensSim) -> tuple[ndarray, ndarray]:
    s_fringe_depth = (el.fringe_frac_outer + 3) * el.rp
    if s_fringe_depth >= el.L / 2.0:
        s_fringe_depth = el.L / 2.0
    num_slices_per_bore_rad = 100
    num_slices = round(num_slices_per_bore_rad * s_fringe_depth / el.rp)
    num_slices = int(2 * (num_slices // 2))
    s_slices_fringing, slice_length_fringe = split_range_into_slices(0.0, s_fringe_depth, num_slices)
    slice_lengths_fringe = np.ones(len(s_slices_fringing)) * slice_length_fringe
    if s_fringe_depth >= el.L / 2.0:
        s_slices, slice_lengths = s_slices_fringing, slice_lengths_fringe
    else:
        s_inner = s_fringe_depth + (el.L / 2.0 - s_fringe_depth) / 2.0
        slice_length_inner = (el.L / 2.0 - s_fringe_depth)  # /4.0
        s_slices = [*s_slices_fringing, s_inner]
        slice_lengths = [*slice_lengths_fringe, slice_length_inner]
    return s_slices, slice_lengths


def transfer_matrix_from_lens(el: HalbachLensSim, atom_speed: RealNum) -> ndarray:
    magnet = el.magnet.make_magpylib_magnets(False, False)
    s_slices, slice_lengths = s_slices_and_lengths_lens(el)
    M = np.eye(2)
    M_slices = [lens_transfer_matrix_for_slice(el.rp, magnet, x, slice_length, atom_speed) for x, slice_length in
                zip(s_slices, slice_lengths)]

    for M_slice in itertools.chain(M_slices, reversed(M_slices)):
        M = M_slice @ M
    return M


def stability_factor_lattice(elements: Iterable):
    """Factor describing stability of periodic lattice of elements. If value is greater than 1 it is stable, though
    higher values are "more" stable in some sense"""
    M_total = full_transfer_matrix(elements)
    m11, m12, m21, m22 = matrix_components(M_total)
    return 2.0 - (m11 ** 2 + 2 * m12 * m21 + m22 ** 2)


def is_stable_lattice(elements: Iterable) -> bool:
    """Determine if the lattice is stable. This can be done by computing eigenvalues, or the method below works. If
    unstable, then raising he transfer matrix to N results in large matrix elements for large N"""
    return stability_factor_lattice(elements) > 0.0


def total_length(elements: Iterable) -> float:
    """Sum of the lengths of elements"""
    length = sum([el.L for el in elements])
    return length


def split_range_into_slices(x_min: RealNum, x_max: RealNum, num_slices: int) -> tuple[ndarray, float]:
    """Split a range into equally spaced points, that are half a spacing away from start and end"""
    slice_length = (x_max - x_min) / num_slices
    x_slices = np.linspace(x_min, x_max, num_slices + 1)[:-1]
    x_slices += slice_length / 2.0
    return x_slices, slice_length


def unit_vec_perp_to_path(path_coords: ndarray) -> ndarray:
    """Along the path path_coords, find the perpindicualr normal vector for each point. Assume the path is
    counter-clockwise, and the normal point radially outwards"""
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
    """"""
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


def K_centrifugal_combiner_at_path_index(coord: ndarray, norm: ndarray, magnets: Collection,
                                         atom_speed: RealNum) -> float:
    """Compute the centrifugal spring constant"""
    force = SIMULATION_MAGNETON * magnets.B_norm_grad(coord)
    force_xo = np.dot(force, norm)
    K_cent = force_xo ** 2 / (SIMULATION_MASS * atom_speed ** 4)
    return K_cent


def combiner_transfer_matrix_at_path_index(index: int, magnets: Collection, coords_path: ndarray,
                                           speeds_path: np.ndarray, norms_path: ndarray,
                                           xo_vals: ndarray, atom_speed: RealNum, design_speed: RealNum) -> ndarray:
    """Compute the thin transfer matrix that corresponds to the location at coords_path[index]"""
    coord = coords_path[index]
    norm = norms_path[index]
    coords = np.array([coord + xo * norm for xo in xo_vals])
    B_norm_grad = magnets.B_norm_grad(coords)
    forces = -SIMULATION_MAGNETON * B_norm_grad
    forces_xo = [np.dot(force, norm) for force in forces]
    speed_path = speeds_path[index] * atom_speed / design_speed
    m = np.polyfit(xo_vals, forces_xo, 1)[0]
    K = -m / speed_path ** 2
    K_cent = K_centrifugal_combiner_at_path_index(coord, norm, magnets, speed_path)
    K += K_cent
    slice_length = combiner_slice_length_at_traj_index(index, coords_path)
    M = transfer_matrix(K, slice_length)
    return M


def transfer_matrix_from_combiner(el_combiner: CombinerHalbachLensSim, atom_speed: RealNum) -> ndarray:
    """Compute the transfer matric for the combiner. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    coords_path, p_path = combiner_halbach_orbit_coords_el_frame(el_combiner)
    speeds_path = np.linalg.norm(p_path, axis=1)
    norms_path = unit_vec_perp_to_path(coords_path)
    xo_max = min([(el_combiner.rp - el_combiner.output_offset), el_combiner.output_offset]) / 5.0
    xo_vals = np.linspace(-xo_max, xo_max, 11)
    magnets = el_combiner.magnet.make_magpylib_magnets(False, False)
    M = np.eye(2)
    for idx, _ in enumerate(coords_path):
        M_thin_slice = combiner_transfer_matrix_at_path_index(idx, magnets, coords_path, speeds_path, norms_path,
                                                              xo_vals, atom_speed, el_combiner.PTL.speed_nominal)
        M = M_thin_slice @ M
    return M


def xo_unit_vector_bender_el_frame(el: HalbachBenderSimSegmented, coord: ndarray) -> ndarray:
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


def bender_transfer_matrix_for_slice(s: RealNum, magnets: Collection, el: HalbachBenderSimSegmented,
                                     slice_c_length: RealNum, atom_speed: RealNum) -> ndarray:
    """This slice transfer matrix for a point along the bender"""
    speed_ratio = atom_speed / el.PTL.speed_nominal
    deltar_orbit = orbit_offset_bender(el, speed_ratio)
    ro = el.rb + deltar_orbit
    xo_max = el.ap - deltar_orbit
    num_samples = 11
    xo_vals = np.linspace(-xo_max / 4.0, xo_max / 4.0, num_samples)
    coords = np.array([el.convert_center_to_cartesian_coords(s, deltar_orbit + xo, 0.0) for xo in xo_vals])
    B_norm_grad = magnets.B_norm_grad(coords, use_approx=True, dx=1e-6, diff_method='central')
    forces = -SIMULATION_MAGNETON * B_norm_grad
    coord_center_orbit = np.array(el.convert_center_to_cartesian_coords(s, deltar_orbit, 0.0))
    V = SIMULATION_MAGNETON * magnets.B_norm(coord_center_orbit)
    norm_perps = xo_unit_vector_bender_el_frame(el, coords[0])

    force_r = [np.dot(norm_perps, force) for force in forces]
    atom_speed_corrected = speed_with_energy_correction(V, atom_speed)
    m = np.polyfit(xo_vals, force_r, 1)[0]
    K = -m / atom_speed_corrected ** 2
    if el.L_cap < s < el.L_cap + el.ang * el.rb:
        K_cent = SIMULATION_MASS / ro ** 2  # centrifugal term
        K += K_cent
        slice_length = slice_c_length * ro / el.rb
    else:
        slice_length = slice_c_length
    M = transfer_matrix(K, slice_length)
    return M


def s_slices_and_lengths_bender(el: HalbachBenderSimSegmented) -> tuple[ndarray, ndarray]:
    """Return arrays of orbit position for slices used to compute transfer matrix of fringing and periodic internal
    regions of bender"""
    s_unit_cell = el.rb * el.ucAng
    s_length_magnet = 2 * s_unit_cell
    s_fringe_depth = el.L_cap + NUM_FRINGE_MAGNETS_MIN * s_length_magnet
    num_slices_per_bore_radius = 5
    num_slices_min_uc = 10
    num_slices_fringe = round(num_slices_per_bore_radius * s_fringe_depth / el.rp)
    num_slices_uc = round(num_slices_per_bore_radius * s_unit_cell / el.rp)
    num_slices_uc += num_slices_min_uc
    s_slices_fringe, slice_length_fringe = split_range_into_slices(0.0, s_fringe_depth, num_slices_fringe)
    s_slices_uc, slice_length_uc = split_range_into_slices(s_fringe_depth, s_fringe_depth + s_unit_cell, num_slices_uc)
    return s_slices_fringe, s_slices_uc


def bender_transfer_matrix_start_end(s_slices_fringe: ndarray, magnets: Collection,
                                     el: HalbachBenderSimSegmented, atom_speed: RealNum) -> tuple[ndarray, ndarray]:
    """Return transfer matrices representing beginning and ending segments of bender. These are the region where the
    impact of fringe fields cannot be ignored."""
    slice_length_fringe = s_slices_fringe[1] - s_slices_fringe[0]
    M_slices_fringe = [bender_transfer_matrix_for_slice(s, magnets, el, slice_length_fringe, atom_speed) for s in
                       s_slices_fringe]
    M_start = multiply_matrices(M_slices_fringe)
    M_end = multiply_matrices(M_slices_fringe, reverse=True)
    return M_start, M_end


def bender_transfer_matrix_internal(s_slices_uc: ndarray, magnets: Collection,
                                    el: HalbachBenderSimSegmented, atom_speed: RealNum) -> ndarray:
    """Return transfer matrix that represents interior of bender where the impact of fringe fields can be ignored. This is
    rapidly computed by exploiting the symmetry of the unit cell model"""
    num_internal_mags = (el.num_magnets - 2 * NUM_FRINGE_MAGNETS_MIN)
    slice_length_uc = s_slices_uc[1] - s_slices_uc[0]
    M_slices_uc = [bender_transfer_matrix_for_slice(s, magnets, el, slice_length_uc, atom_speed) for s in s_slices_uc]
    M_uc_exit = multiply_matrices(M_slices_uc)  # first unit cell in bender is a half, so an 'exit', so 2N later is
    # also exit
    M_uc_entrance = multiply_matrices(M_slices_uc, reverse=True)  # next unit cell is 'entrance', or first half o unit
    # cell
    M_exit_to_entrace = M_uc_exit @ M_uc_entrance
    M_internal = np.linalg.matrix_power(M_exit_to_entrace, num_internal_mags)
    return M_internal


def transfer_matrix_from_bender(el: HalbachBenderSimSegmented, atom_speed: RealNum) -> ndarray:
    """Return the transfer matric for the bender. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    if el.num_magnets < 2 * NUM_FRINGE_MAGNETS_MIN:
        raise NotImplementedError
    magnets = el.build_bender(True, (True, False), num_lenses=NUM_FRINGE_MAGNETS_MIN * 3)
    s_slices_fringe, s_slices_uc = s_slices_and_lengths_bender(el)
    M_start, M_end = bender_transfer_matrix_start_end(s_slices_fringe, magnets, el, atom_speed)
    M_internal = bender_transfer_matrix_internal(s_slices_uc, magnets, el, atom_speed)
    M_full = M_end @ M_internal @ M_start
    return M_full


class CompositeElement:
    """Element representing a series of elements, which in this case is a single transfer matrix"""

    def __init__(self, M: ndarray, L: RealNum):
        self.M = M
        self.L = L

    def M_func(self, x) -> ndarray:
        raise NotImplementedError


class Element:
    """ Base element representing a transfer matrix (ABCD matrix)"""

    def __init__(self, K, L):
        assert L > 0.0
        self.K = K
        self.L = L
        self.M = self.M_func(L)

    def M_func(self, x) -> ndarray:
        assert 0.0 <= x <= self.L
        return transfer_matrix(self.K, x)


class Lens(Element):
    """Element representing a lens"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum, atom_speed: RealNum):
        K = spring_constant_lens(Bp, rp, atom_speed)
        super().__init__(K, L)


class Combiner(Element):
    """Element representing a combiner"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum, atom_speed: RealNum):
        K = spring_constant_lens(Bp, rp, atom_speed)
        super().__init__(K, L)


class Drift(Element):
    """Element representing free space"""

    def __init__(self, L: RealNum):
        K = 0.0
        super().__init__(K, L)


class Bender(Element):
    """Element representing a bending component"""

    def __init__(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum, atom_speed: RealNum):
        self.ro = bender_orbit_radius_no_energy_correction(Bp, rb, rp, atom_speed)
        K = bender_spring_constant(Bp, rb, rp, self.ro, atom_speed)
        L = self.ro * bending_angle  # length of particle orbit
        super().__init__(K, L)


MatrixLatticeElement = Union[Drift, Lens, Bender, Combiner, CompositeElement]


class Lattice:
    """Model of a sequence (possibly periodic) of elements"""

    def __init__(self, atom_speed=DEFAULT_ATOM_SPEED):
        self.elements: Optional[list[MatrixLatticeElement]] = []
        self.atom_speed = atom_speed

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def add_drift(self, L: RealNum) -> None:
        self.elements.append(Drift(L))

    def add_lens(self, L: RealNum, Bp: RealNum, rp: RealNum) -> None:
        self.elements.append(Lens(L, Bp, rp, self.atom_speed))

    def add_bender(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum):
        self.elements.append(Bender(Bp, rb, rp, bending_angle, self.atom_speed))

    def add_combiner(self, L: RealNum, Bp: RealNum, rp: RealNum) -> None:
        self.elements.append(Combiner(L, Bp, rp, self.atom_speed))

    def is_stable(self) -> bool:
        return is_stable_lattice(self)

    def stability_factor(self) -> bool:
        return stability_factor_lattice(self)

    def M_total(self, revolutions: int = 1) -> Optional[ndarray]:
        M_single_rev = full_transfer_matrix(self.elements)
        return np.linalg.matrix_power(M_single_rev, revolutions)

    def M_total_components(self) -> tuple[float, float, float, float]:
        return matrix_components(self.M_total())

    def trace(self, Xi) -> ndarray:
        return self.M_total() @ Xi

    def total_length(self) -> float:
        return total_length(self.elements)

    def add_elements_from_sim_lens(self, el_lens: HalbachLensSim) -> None:
        M = transfer_matrix_from_lens(el_lens, self.atom_speed)
        self.elements.append(CompositeElement(M, el_lens.L))

    def add_elements_from_sim_seg_bender(self, el_bend: HalbachBenderSimSegmented) -> None:
        M = transfer_matrix_from_bender(el_bend, self.atom_speed)
        self.elements.append(CompositeElement(M, el_bend.Lo))

    def add_elements_from_sim_combiner(self, el_combiner: CombinerHalbachLensSim) -> None:
        M = transfer_matrix_from_combiner(el_combiner, self.atom_speed)
        self.elements.append(CompositeElement(M, el_combiner.L))

    def build_matrix_lattice_from_sim_lattice(self, simulated_lattice: ParticleTracerLattice) -> None:
        """Build the lattice from an existing ParticleTracerLattice object"""
        if simulated_lattice.lattice_type != 'storage_ring':
            raise NotImplementedError
        if not is_sim_lattice_viable(simulated_lattice, self.atom_speed):
            raise NonViableLattice

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
                    self.add_elements_from_sim_lens(el)
                elif type(el) is HalbachBenderSimSegmented:
                    self.add_elements_from_sim_seg_bender(el)
                elif type(el) is CombinerHalbachLensSim:
                    self.add_elements_from_sim_combiner(el)
                else:
                    raise NotImplementedError
                reuser.add_sim_and_matrx(el, self.elements[-1])

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
