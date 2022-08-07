from math import sqrt, cos, sin
from typing import Iterable, Optional

import numpy as np

from constants import SIMULATION_MAGNETON, SIMULATION_MASS, DEFAULT_ATOM_SPEED
from helperTools import arr_product
from latticeElements.elements import Drift as Drift_Sim
from latticeElements.elements import HalbachLensSim, Element
from typeHints import ndarray, RealNum

SMALL_ROUNDING_NUM = 1e-14


def lens_transfer_matrix(K: RealNum, L: RealNum) -> ndarray:
    phi = sqrt(K) * L
    A = cos(phi)
    B = sin(phi) / sqrt(K)
    C = -sqrt(K) * sin(phi)
    D = cos(phi)
    return np.array([[A, B], [C, D]])


def drift_transfer_matrix(L: RealNum) -> ndarray:
    return np.array([[1, L], [0, 1]])


def full_transfer_matrix(elements: Iterable) -> ndarray:
    M = np.eye(2)
    for el in elements:
        M = el.M @ M
    return M


def is_stable_lattice(elements: Iterable) -> bool:
    M_total = full_transfer_matrix(elements)
    eigenvalues_norm = np.abs(np.linalg.eig(M_total)[0])
    return np.all(eigenvalues_norm <= 1.0 + SMALL_ROUNDING_NUM)


def best_fit_magnetic_field(el: HalbachLensSim) -> float:
    magnets = el.magnet.make_magpylib_magnets(False, False)
    x_vals = np.linspace(0.0, el.L, 1000)
    y = el.rp * .333
    coords = arr_product(x_vals, [y], [0])
    F_y = SIMULATION_MAGNETON * magnets.B_norm_grad(coords)[:, 1]
    Kappa = abs(np.trapz(F_y, x=x_vals))
    gamma = el.Lm * SIMULATION_MAGNETON * 2 * y / el.rp ** 2
    Bp_optimal = Kappa / gamma
    return Bp_optimal


class Lens:
    def __init__(self, L: RealNum, Bp: RealNum, rp: RealNum):
        assert L > 0.0 and Bp >= 0.0 and rp > 0.0
        self.L = L
        self.Bp = Bp
        self.K = 2 * SIMULATION_MAGNETON * Bp / (SIMULATION_MASS * DEFAULT_ATOM_SPEED ** 2 * rp ** 2)
        self.M = lens_transfer_matrix(self.K, L)

    def M_func(self, x) -> ndarray:
        assert 0.0 <= x <= self.L
        return lens_transfer_matrix(self.K, x)


class Drift:
    def __init__(self, L: RealNum):
        assert L > 0.0
        self.L = L
        self.M = drift_transfer_matrix(L)

    def M_func(self, x: RealNum) -> ndarray:
        assert 0.0 <= x <= self.L
        return drift_transfer_matrix(x)


MatrixLatticeElement = Optional[Drift, Lens]


class Lattice:
    def __init__(self):
        self.elements: list[MatrixLatticeElement] = []

    def add_drift(self, L: RealNum) -> None:
        self.elements.append(Drift(L))

    def add_lens(self, L: RealNum, Bp: RealNum, rp: RealNum) -> None:
        self.elements.append(Lens(L, Bp, rp))

    def is_stable(self) -> bool:
        return is_stable_lattice(self.elements)

    def M_total(self) -> ndarray:
        return full_transfer_matrix(self.elements)

    def trace(self, Xi) -> ndarray:
        return self.M_total() @ Xi

    def add_elements_from_sim_lens(self, el_lens: Element) -> None:
        Bp = best_fit_magnetic_field(el_lens)
        L_lens, rp = el_lens.Lm, el_lens.rp
        L_drift = HalbachLensSim.fringe_frac_outer * rp
        self.add_drift(L_drift)
        self.add_lens(L_lens, Bp, rp)
        self.add_drift(L_drift)

    def build_matrix_lattice_from_sim_lattice(self, simulated_lattice) -> None:
        for el in simulated_lattice:
            if type(el) is Drift_Sim:
                self.add_drift(el.L)
            elif type(el) is HalbachLensSim:
                self.add_elements_from_sim_lens(el)
            else:
                raise NotImplementedError
