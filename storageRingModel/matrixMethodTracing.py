from math import sqrt, cos, sin
from typing import Iterable, Union

import numpy as np

from constants import SIMULATION_MAGNETON, SIMULATION_MASS, DEFAULT_ATOM_SPEED
from helperTools import arr_product
from helperTools import make_dense_curve_1D_linear
from latticeElements.combiner_characterizer import calculate_trajectory_length
from latticeElements.elements import Drift as Drift_Sim
from latticeElements.elements import HalbachLensSim, HalbachBenderSimSegmented, CombinerHalbachLensSim
from latticeElements.orbitTrajectories import combiner_halbach_orbit_coords_el_frame
from typeHints import ndarray, RealNum

SMALL_ROUNDING_NUM = 1e-14


def transfer_matrix(K: RealNum, L: RealNum) -> ndarray:
    """Build the 2x2 transfer matrix (ABCD matrix) for an element that represents a harmonic potential.
    Transfer matrix is for one dimension only"""
    assert K >= 0.0 and L >= 0.0
    if K == 0.0:
        A = 1.0
        B = L
        C = 0.0
        D = 1
    else:
        phi = sqrt(K) * L
        A = cos(phi)
        B = sin(phi) / sqrt(K)
        C = -sqrt(K) * sin(phi)
        D = cos(phi)
    return np.array([[A, B], [C, D]])


def bender_orbit_radius(Bp: RealNum, rb: RealNum, rp: RealNum) -> float:
    """Calculate the orbit radius for particle in a bender. This is balancing the centrifugal "force" with the magnet
    force, and also accounts for the small difference from energy conservation of particle slowing down when entering
    bender"""
    term1 = 3 * rb / 4.0
    term2 = Bp * SIMULATION_MAGNETON * rb ** 2
    term3 = 4 * SIMULATION_MASS * (rp * DEFAULT_ATOM_SPEED) ** 2
    term4 = 4 * sqrt(Bp * SIMULATION_MAGNETON)
    radius_orbit = term1 + sqrt(term2 + term3) / term4  # quadratic formula
    return radius_orbit


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


def is_stable_lattice(elements: Iterable) -> bool:
    """Determine if the lattice is stable. This can be done by computing eigenvalues, or the method below works. If
    unstable, then raising he transfer matrix to N results in large matrix elements for large N"""
    M_total = full_transfer_matrix(elements)
    m11, m12, m21, m22 = matrix_components(M_total)
    return 2.0 - (m11 ** 2 + 2 * m12 * m21 + m22 ** 2) > 0.0


def total_length(elements: Iterable) -> float:
    length = sum([el.L for el in elements])
    return length


def best_fit_magnetic_field_lens(el: HalbachLensSim) -> float:
    """Use the simulated magnetic fields from a simulated element to determine optimal magnetic field for use in the
    ideal force formula. Basically integrate through the simulated element, and choose a B value that gives the same
    integral for the ideal element"""
    magnets = el.magnet.make_magpylib_magnets(False, False)
    x_vals = np.linspace(0.0, el.L, 1000)
    y = el.rp * .5
    coords = arr_product(x_vals, [y], [0])
    F_xo = SIMULATION_MAGNETON * magnets.B_norm_grad(coords)[:, 1]
    Kappa = abs(np.trapz(F_xo, x=x_vals))
    gamma = el.Lm * SIMULATION_MAGNETON * 2 * y / el.rp ** 2
    Bp_optimal = Kappa / gamma
    return Bp_optimal


def cumulative_trajectory_length(coords: ndarray) -> ndarray:
    """Given a 3D path, return the cumulative length along the path"""
    ds_vec = coords[1:] - coords[:-1]
    ds = np.linalg.norm(ds_vec, axis=1)
    s = np.cumsum(ds)
    s = np.append(0, s)  # need to add back first step
    return s


def dense_combiner_trajectory_curve(el: CombinerHalbachLensSim) -> ndarray:
    """Because the combiner trajectory is used to calculate magnet and drift lengths, it's important that it is fine
    enough that when splitting the trajectory up the loss of a point won't meaningfully change the values. """
    coords = combiner_halbach_orbit_coords_el_frame(el)
    x_vals, y_vals, _ = coords.T
    x_vals, y_vals = make_dense_curve_1D_linear(x_vals, y_vals, 100_000)
    z_vals = np.zeros(len(y_vals))
    coords_dense = np.column_stack((x_vals, y_vals, z_vals))
    return coords_dense


def drift_lengths_before_after_combiner(el: CombinerHalbachLensSim) -> tuple[float, float]:
    """Find the drift lengths before and after the combiner. Before refers to the direction stream of particles enter,
    and after refers to the direction they leave by"""
    coords = dense_combiner_trajectory_curve(el)
    indices_before_combiner = coords[:, 0] > el.Lm + el.space
    indices_after_combiner = coords[:, 0] < el.space
    L_before = calculate_trajectory_length(coords[indices_before_combiner])
    L_after = calculate_trajectory_length(coords[indices_after_combiner])
    return L_before, L_after


def best_fit_magnetic_field_combiner(el: CombinerHalbachLensSim, L_eff_combiner: float) -> float:
    """find the best fit for the magnetic field for the combiner. This is done along the trajectory of the nominal
    particle. """
    magnets = el.magnet.make_magpylib_magnets(False, False)

    x = np.linspace(0.0, el.L, 1000)
    y0 = (abs(el.output_offset) + abs(el.input_offset)) / 2.0
    coords = arr_product(x, [y0], [0])
    forces = -SIMULATION_MAGNETON * magnets.B_norm_grad(coords)

    Kappa = np.trapz(forces[:, 1], x=x)
    gamma = -L_eff_combiner * SIMULATION_MAGNETON * 2 * y0 / el.rp ** 2
    Bp_optimal = Kappa / gamma

    return Bp_optimal


def combiner_effective_length(el: CombinerHalbachLensSim) -> float:
    """Effective material length of combiner. This equals the length of the trajectory through the combiner material"""
    coords = dense_combiner_trajectory_curve(el)
    indices_in_combiner = (coords[:, 0] > el.space) & (coords[:, 0] < el.Lm + el.space)
    coords_in_combiner = coords[indices_in_combiner]
    length = calculate_trajectory_length(coords_in_combiner)
    return length


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


def best_fit_magnetic_field_seg_bender(el: HalbachBenderSimSegmented) -> float:
    """Use the simulated magnetic fields from a simulated element to determine optimal magnetic field for use in the
    ideal force formula. Basically integrate through the simulated element, and choose a B value that gives the same
    integral for the ideal element"""
    magnets = el.build_bender(True, (True, True), use_method_of_moments=False, num_lenses=el.num_magnets)
    x0 = el.ro - el.rb  # x offset in orbit frame
    s_max = el.ang * el.rb + 2 * el.L_cap
    s_vals = np.linspace(0.0, s_max, 1000)
    coords = np.array([el.convert_center_to_cartesian_coords(s, x0, 0.0) for s in s_vals])
    forces = -SIMULATION_MAGNETON * magnets.B_norm_grad(coords)
    norm_perps = unit_vec_perp_to_path(coords)
    force_r = [np.dot(norm, force) for norm, force in zip(norm_perps, forces)]
    Kappa = abs(np.trapz(force_r, x=s_vals))
    L_orbit = el.ro * el.ang
    delta_r = el.ro - el.rb
    gamma = L_orbit * SIMULATION_MAGNETON * 2 * delta_r / el.rp ** 2
    Bp_optimal = Kappa / gamma
    return Bp_optimal


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

    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum):
        K = spring_constant_lens(Bp, rp)
        print(K)
        super().__init__(K, L)


class Drift(Element):
    """Element representing free space"""

    def __init__(self, L: RealNum):
        K = 0.0
        super().__init__(K, L)


class Bender(Element):
    """Element representing a bending component"""

    def __init__(self, Bp: RealNum, rb: RealNum, rp: RealNum, ro, bending_angle: RealNum):
        K = bender_spring_constant(Bp, rb, rp, ro)
        L = ro * bending_angle  # length of particle orbit
        super().__init__(K, L)


MatrixLatticeElement = Union[Drift, Lens, Bender, Combiner]


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

    def add_bender(self, Bp: RealNum, rb: RealNum, rp: RealNum, ro: RealNum, bending_angle: RealNum):
        self.elements.append(Bender(Bp, rb, rp, ro, bending_angle))

    def add_combiner(self, L: RealNum, Bp: RealNum, rp: RealNum) -> None:
        self.elements.append(Combiner(L, Bp, rp))

    def is_stable(self) -> bool:
        return is_stable_lattice(self.elements)

    def M_total(self) -> ndarray:
        return full_transfer_matrix(self.elements)

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
        Bp = best_fit_magnetic_field_lens(el_lens)
        L_lens, rp = el_lens.Lm, el_lens.rp
        L_drift = HalbachLensSim.fringe_frac_outer * rp
        self.add_drift(L_drift)
        self.add_lens(L_lens, Bp, rp)
        self.add_drift(L_drift)

    def add_elements_from_sim_seg_bender(self, el_bend: HalbachBenderSimSegmented) -> None:
        L_drift = el_bend.L_cap
        Bp = best_fit_magnetic_field_seg_bender(el_bend)
        self.add_drift(L_drift)
        rb, rp, ro, bending_ang = el_bend.rb, el_bend.rp, el_bend.ro, el_bend.ang
        self.add_bender(Bp, rb, rp, ro, bending_ang)
        self.add_drift(L_drift)

    def add_elements_from_sim_combiner(self, el_combiner: CombinerHalbachLensSim) -> None:
        L_eff_drift_before, L_eff_drift_after = drift_lengths_before_after_combiner(el_combiner)
        L_eff_combiner = combiner_effective_length(el_combiner)
        rp = el_combiner.rp
        Bp = best_fit_magnetic_field_combiner(el_combiner, L_eff_combiner)
        self.add_drift(L_eff_drift_before)
        self.add_combiner(L_eff_combiner, Bp, rp)
        self.add_drift(L_eff_drift_after)

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
        M_tot = np.linalg.matrix_power(self.M_total(), revolutions)
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
