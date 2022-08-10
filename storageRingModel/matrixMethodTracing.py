from math import sqrt, cos, sin, cosh, sinh
from typing import Iterable, Union

import numpy as np

from HalbachLensClass import Collection
from constants import SIMULATION_MAGNETON, SIMULATION_MASS, DEFAULT_ATOM_SPEED
from latticeElements.class_HalbachBenderSegmented import mirror_across_angle
from latticeElements.elements import Drift as Drift_Sim
from latticeElements.elements import HalbachLensSim, HalbachBenderSimSegmented, CombinerHalbachLensSim
from latticeElements.orbitTrajectories import combiner_halbach_orbit_coords_el_frame
from typeHints import ndarray, RealNum

SMALL_ROUNDING_NUM = 1e-14


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


def bender_orbit_radius_energy_correction(Bp: RealNum, rb: RealNum, rp: RealNum) -> float:
    """Calculate the orbit radius for particle in a bender. This is balancing the centrifugal "force" with the magnet
    force, and also accounts for the small difference from energy conservation of particle slowing down when entering
    bender"""
    term1 = 3 * rb / 4.0
    term2 = Bp * SIMULATION_MAGNETON * rb ** 2
    term3 = 4 * SIMULATION_MASS * (rp * DEFAULT_ATOM_SPEED) ** 2
    term4 = 4 * sqrt(Bp * SIMULATION_MAGNETON)
    radius_orbit = term1 + sqrt(term2 + term3) / term4  # quadratic formula
    return radius_orbit


def bender_orbit_radius_no_energy_correction(Bp: RealNum, rb: RealNum, rp: RealNum) -> float:
    term1 = .5 * rb
    term2 = rb ** 2
    term3 = 2 * SIMULATION_MASS * (DEFAULT_ATOM_SPEED * rp) ** 2 / (SIMULATION_MAGNETON * Bp)
    term4 = .5 * np.sqrt(term2 + term3)
    return term1 + term4


def bender_spring_constant(Bp: RealNum, rb: RealNum, rp: RealNum, ro: RealNum) -> float:
    """Spring constant (F=-Kx) for bender's harmonic potential"""
    assert Bp >= 0.0 and 0.0 < rp < rb and rb > 0.0
    term1 = SIMULATION_MASS / ro ** 2  # centrifugal term
    term2 = 2 * SIMULATION_MAGNETON * Bp / (SIMULATION_MASS * (DEFAULT_ATOM_SPEED * rp) ** 2)  # magnetic term
    K = term1 + term2
    return K


def spring_constant_lens(Bp: RealNum, rp: RealNum) -> float:
    """Spring constant (F=-Kx) for lens's harmonic potential"""
    assert rp > 0.0
    K = 2 * SIMULATION_MAGNETON * Bp / (SIMULATION_MASS * (DEFAULT_ATOM_SPEED * rp) ** 2)
    return K


def full_transfer_matrix(elements: Iterable) -> ndarray:
    """Transfer matrix for a sequence of elements start to end"""
    M = np.eye(2)
    for el in elements:
        M = el.M @ M
    return M


def matrix_components(M):
    m11 = M[0, 0]
    m12 = M[0, 1]
    m21 = M[1, 0]
    m22 = M[1, 1]
    return m11, m12, m21, m22


def lens_transfer_matrix_for_slice(rp: RealNum, magnets: Collection, x_slice: RealNum,
                                   slice_length: RealNum) -> ndarray:
    num_samples = 30
    x_vals = np.ones(num_samples) * x_slice
    y_vals = np.linspace(-rp / 4.0, rp / 4.0, num_samples)
    z_vals = np.zeros(num_samples)
    coords = np.column_stack((x_vals, y_vals, z_vals))
    F_y = -SIMULATION_MAGNETON * magnets.B_norm_grad(coords)[:, 1]
    m = np.polyfit(y_vals, F_y, 1)[0]
    K = -m / DEFAULT_ATOM_SPEED ** 2
    M_slice = transfer_matrix(K, slice_length)
    return M_slice


def transfer_matrix_from_lens(el: HalbachLensSim) -> ndarray:
    magnet = el.magnet.make_magpylib_magnets(False, False)
    num_slices_per_bore_rad = 300  # Don't use less than 100
    num_slices = round(num_slices_per_bore_rad * el.L / el.rp)
    x_slices, slice_length = split_range_into_slices(0.0, el.L, num_slices)
    M = np.eye(2)
    for x in x_slices:
        M = lens_transfer_matrix_for_slice(el.rp, magnet, x, slice_length) @ M
    return M


def is_stable_lattice(elements: Iterable) -> bool:
    """Determine if the lattice is stable. This can be done by computing eigenvalues, or the method below works. If
    unstable, then raising he transfer matrix to N results in large matrix elements for large N"""
    M_total = full_transfer_matrix(elements)
    m11, m12, m21, m22 = matrix_components(M_total)
    return 2.0 - (m11 ** 2 + 2 * m12 * m21 + m22 ** 2) > 0.0


def total_length(elements: Iterable) -> float:
    """Sum of the lengths of elements"""
    length = sum([el.L for el in elements])
    return length


def split_range_into_slices(x_min: float, x_max: float, num_slices: int) -> tuple[ndarray, float]:
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


def K_centrifugal_combiner_at_path_index(coord: ndarray, norm: ndarray, magnets: Collection) -> float:
    """Compute the centrifugal spring constant"""
    force = SIMULATION_MAGNETON * magnets.B_norm_grad(coord)
    force_xo = np.dot(force, norm)
    K_cent = force_xo ** 2 / (SIMULATION_MASS * DEFAULT_ATOM_SPEED ** 4)
    return K_cent


def combiner_transfer_matrix_at_path_index(index: int, magnets: Collection, coords_path: ndarray,
                                           norms_path: ndarray, xo_vals: ndarray) -> ndarray:
    """Compute the thin transfer matrix that corresponds to the location at coords_path[index]"""
    coord = coords_path[index]
    norm = norms_path[index]
    coords = np.array([coord + xo * norm for xo in xo_vals])
    B_norm_grad = magnets.B_norm_grad(coords)
    forces = -SIMULATION_MAGNETON * B_norm_grad
    V = SIMULATION_MAGNETON * magnets.B_norm(coord)
    forces_xo = [np.dot(force, norm) for force in forces]
    atom_speed = speed_with_energy_correction(V)
    m = np.polyfit(xo_vals, forces_xo, 1)[0]
    K = -m / atom_speed ** 2
    K_cent = K_centrifugal_combiner_at_path_index(coord, norm, magnets)
    K += K_cent
    slice_length = combiner_slice_length_at_traj_index(index, coords_path)
    M = transfer_matrix(K, slice_length)
    return M


def transfer_matrix_from_combiner(el_combiner: CombinerHalbachLensSim) -> ndarray:
    """Compute the transfer matric for the combiner. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    coords_path = combiner_halbach_orbit_coords_el_frame(el_combiner)
    norms_path = unit_vec_perp_to_path(coords_path)
    xo_max = min([(el_combiner.rp - el_combiner.output_offset), el_combiner.output_offset]) / 5.0
    xo_vals = np.linspace(-xo_max, xo_max, 11)
    magnets = el_combiner.magnet.make_magpylib_magnets(False, False)
    M = np.eye(2)
    for idx, _ in enumerate(coords_path):
        M_thin_slice = combiner_transfer_matrix_at_path_index(idx, magnets, coords_path, norms_path, xo_vals)
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


def speed_with_energy_correction(V: float) -> float:
    """Energy conservation for atom speed accounting for magnetic fields"""
    E0 = .5 * DEFAULT_ATOM_SPEED ** 2
    KE = E0 - V
    speed_corrected = np.sqrt(2 * KE)
    return speed_corrected


def bender_transfer_matrix_for_slice(s: float, magnets: Collection, el: HalbachBenderSimSegmented,
                                     slice_c_length: float) -> ndarray:
    """This slice transfer matrix for a point along the bender"""
    deltar_orbit = el.ro - el.rb
    xo_max = el.rp - deltar_orbit
    num_samples = 11
    xo_vals = np.linspace(-xo_max / 4.0, xo_max / 4.0, num_samples)
    coords = np.array([el.convert_center_to_cartesian_coords(s, deltar_orbit + xo, 0.0) for xo in xo_vals])
    B_norm_grad = magnets.B_norm_grad(coords, use_approx=True)
    forces = -SIMULATION_MAGNETON * B_norm_grad
    B_norm_at_center = el.convert_center_to_cartesian_coords(s, deltar_orbit, 0.0)
    V = SIMULATION_MAGNETON * np.mean(B_norm_at_center)
    norm_perps = xo_unit_vector_bender_el_frame(el, coords[0])
    force_r = [np.dot(norm_perps, force) for force in forces]
    atom_speed = speed_with_energy_correction(V)
    m = np.polyfit(xo_vals, force_r, 1)[0]
    K = -m / atom_speed ** 2
    if el.L_cap < s < el.L_cap + el.ang * el.rb:
        K_cent = SIMULATION_MASS / el.ro ** 2  # centrifugal term
        K += K_cent
        slice_length = slice_c_length * el.ro / el.rb
    else:
        slice_length = slice_c_length
    M = transfer_matrix(K, slice_length)
    return M


def transfer_matrix_from_bender(el: HalbachBenderSimSegmented) -> ndarray:
    """Compute the transfer matric for the bender. This is done by splitting the element into many thin matrices, and
    multiplying them together"""
    magnets = el.build_full_bender_model()
    num_slices_per_mag = 5
    num_slices = 10 + el.num_magnets * num_slices_per_mag
    sc_max = el.ang * el.rb + 2 * el.L_cap
    s_slices, slice_c_length = split_range_into_slices(0.0, sc_max, num_slices)
    M = np.eye(2)
    for s in s_slices:
        M_thin_slice = bender_transfer_matrix_for_slice(s, magnets, el, slice_c_length)
        M = M_thin_slice @ M
    return M


class CompositeElement:
    """Element representing a series of elements, which in this case is a single transfer matrix"""

    def __init__(self, M: ndarray, L: float):
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

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum):
        K = spring_constant_lens(Bp, rp)
        super().__init__(K, L)


class Combiner(Element):
    """Element representing a combiner"""

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum):
        K = spring_constant_lens(Bp, rp)
        super().__init__(K, L)


class Drift(Element):
    """Element representing free space"""

    def __init__(self, L: RealNum):
        K = 0.0
        super().__init__(K, L)


class Bender(Element):
    """Element representing a bending component"""

    def __init__(self, Bp: RealNum, rb: RealNum, rp: RealNum, bending_angle: RealNum):
        self.ro = bender_orbit_radius_no_energy_correction(Bp, rb, rp)
        K = bender_spring_constant(Bp, rb, rp, self.ro)
        L = self.ro * bending_angle  # length of particle orbit
        super().__init__(K, L)


MatrixLatticeElement = Union[Drift, Lens, Bender, Combiner, CompositeElement]


class Lattice:
    """Model of a sequence (possibly periodic) of elements"""

    def __init__(self):
        self.elements: list[MatrixLatticeElement] = []

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

    def is_stable(self) -> bool:
        return is_stable_lattice(self.elements)

    def M_total(self, revolutions: int = 1) -> ndarray:
        M_single_rev = full_transfer_matrix(self.elements)
        return np.linalg.matrix_power(M_single_rev, revolutions)

    def trace(self, Xi) -> ndarray:
        return self.M_total() @ Xi

    def total_length(self) -> float:
        return total_length(self.elements)

    def tune(self) -> float:
        raise NotImplementedError
        phase = 0.0
        for el in self:
            w = np.sqrt(el.K)
            phase += w * el.L
        tune = phase / (2 * np.pi)
        return tune

    def add_elements_from_sim_lens(self, el_lens: HalbachLensSim) -> None:
        M = transfer_matrix_from_lens(el_lens)
        self.elements.append(CompositeElement(M, el_lens.L))

    def add_elements_from_sim_seg_bender(self, el_bend: HalbachBenderSimSegmented) -> None:
        M = transfer_matrix_from_bender(el_bend)
        self.elements.append(CompositeElement(M, el_bend.Lo))

    def add_elements_from_sim_combiner(self, el_combiner: CombinerHalbachLensSim) -> None:
        M = transfer_matrix_from_combiner(el_combiner)
        self.elements.append(CompositeElement(M, el_combiner.L))

    def build_matrix_lattice_from_sim_lattice(self, simulated_lattice) -> None:
        """Build the lattice from an existing ParticleTracerLattice object"""
        if simulated_lattice.lattice_type != 'storage_ring':
            raise NotImplementedError
        assert len(self.elements) == 0
        for el in simulated_lattice:
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

    def trace_swarm(self, swarm, revolutions=1, copy_swarm=True):
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