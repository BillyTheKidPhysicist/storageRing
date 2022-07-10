import numpy as np

from constants import SIMULATION_MAGNETON
from latticeElements.class_BaseElement import BaseElement
from numbaFunctionsAndObjects import idealLensFastFunctions


class LensIdeal(BaseElement):
    """
    Ideal model of lens with hard edge. Force inside is calculated from field at pole face and bore radius as
    F=2*ub*r/rp**2 where rp is bore radius, and ub the simulation bohr magneton where the mass of lithium7=1kg.
    This will prevent energy conservation because of the absence of fringe fields between elements to reduce
    forward velocity. Interior vacuum tube is a cylinder
    """

    def __init__(self, PTL, L: float, Bp: float, rp: float, ap: float):
        """
        :param PTL: Instance of ParticleTracerLatticeClass
        :param L: Total length of element and lens, m. Not always the same because of need to contain fringe fields
        :param Bp: Magnetic field at the pole face, T.
        :param rp: Bore radius, m. Distance from center of magnet to the magnetic material
        :param ap: Aperture of bore, m. Typically is the radius of the vacuum tube
        """
        # fillParams is used to avoid filling the parameters in inherited classes
        super().__init__(PTL, L=L)
        self.Bp = Bp
        self.rp = rp
        self.ap = rp if ap is None else ap  # size of apeture radially
        self.K = None

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.K = self.field_fact * (2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
        if self.L is not None:
            self.Lo = self.L

    def build_fast_field_helper(self) -> None:

        numba_func_constants=self.K,self.L,self.ap,self.field_fact
        force_args = (numba_func_constants,)
        potential_args = (numba_func_constants,)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(idealLensFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

    def transform_lab_coords_into_element_frame(self, q_lab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = q_lab.copy()
        qNew -= self.r1
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_element_coords_into_lab_frame(self, q_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = q_el.copy()
        qNew = self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew += self.r1
        return qNew

    def transform_orbit_frame_into_lab_frame(self, q_orbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = q_orbit.copy()
        qNew[:2] = self.R_Out @ qNew[:2]
        qNew += self.r1
        return qNew

    def transform_element_coords_into_local_orbit_frame(self, q_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Element and orbit frame is identical in simple
        straight elements"""

        return q_el.copy()

    def set_length(self, L: float) -> None:
        """this is used typically for setting the length after satisfying constraints"""

        assert L > 0.0
        self.L = L
        self.Lo = self.L

    def transform_element_momentum_into_local_orbit_frame(self, q_el: np.ndarray, p_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Element and orbit frame is identical in simple
        straight elements"""

        return p_el.copy()
