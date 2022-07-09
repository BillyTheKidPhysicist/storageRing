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

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.K = self.fieldFact * (2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
        if self.L is not None:
            self.Lo = self.L

    def build_Fast_Field_Helper(self) -> None:

        numba_func_constants=self.K,self.L,self.ap,self.fieldFact
        force_args = (numba_func_constants,)
        potential_args = (numba_func_constants,)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(idealLensFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

    def transform_Lab_Coords_Into_Element_Frame(self, qLab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = qLab.copy()
        qNew -= self.r1
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_Element_Coords_Into_Lab_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = qEl.copy()
        qNew = self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew += self.r1
        return qNew

    def transform_Orbit_Frame_Into_Lab_Frame(self, qOrbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = qOrbit.copy()
        qNew[:2] = self.ROut @ qNew[:2]
        qNew += self.r1
        return qNew

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Element and orbit frame is identical in simple
        straight elements"""

        return qEl.copy()

    def set_length(self, L: float) -> None:
        """this is used typically for setting the length after satisfying constraints"""

        assert L > 0.0
        self.L = L
        self.Lo = self.L

    def transform_Element_Momentum_Into_Local_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Element and orbit frame is identical in simple
        straight elements"""

        return pEl.copy()
