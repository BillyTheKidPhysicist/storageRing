import numpy as np

from latticeElements.class_BaseElement import BaseElement
from numbaFunctionsAndObjects import combinerIdealFastFunction


# from latticeElements.class_CombinerHalbachLensSim import CombinerHalbachLensSim

class CombinerIdeal(BaseElement):
    # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
    # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
    # angle is decided by tracing particles through the combiner and finding the bending angle.

    def __init__(self, PTL, Lm: float, c1: float, c2: float, apL: float, apR: float, apZ: float, sizeScale: float):
        super().__init__(PTL)
        self.field_fact = -1.0 if self.PTL.lattice_type == 'injector' else 1.0
        self.sizeScale = sizeScale  # the fraction that the combiner is scaled up or down to. A combiner twice the size would
        # use sizeScale=2.0
        self.apR = apR
        self.apL = apL
        self.apz = apZ
        self.ap = None
        self.Lm = Lm
        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet
        self.c1 = c1
        self.c2 = c2
        self.space = 0  # space at the end of the combiner to account for fringe fields

        self.shape = 'COMBINER_SQUARE'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        from latticeElements.combiner_characterizer import characterize_CombinerIdeal
        self.apR, self.apL, self.apz, self.Lm = [val * self.sizeScale for val in
                                                 (self.apR, self.apL, self.apz, self.Lm)]
        self.c1, self.c2 = self.c1 / self.sizeScale, self.c2 / self.sizeScale
        self.Lb = self.Lm  # length of segment after kink after the inlet
        # self.fast_field_helper = get_Combiner_Ideal([self.c1, self.c2, np.nan, self.Lb,
        #                                                   self.apL, self.apR, np.nan, np.nan])
        inputAngle, inputOffset, trajectoryLength = characterize_CombinerIdeal(self)

        self.Lo = trajectoryLength  # np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = inputAngle
        self.inputOffset = inputOffset
        self.La = .5 * (self.apR + self.apL) * np.sin(self.ang)
        self.L = self.La * np.cos(
            self.ang) + self.Lb  # TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING. Is it used?

    def build_fast_field_felper(self) -> None:

        numba_func_constants=self.c1,self.c2, self.ang, self.La, self.Lb, self.apz, self.apL, self.apR, self.field_fact


        force_args = (numba_func_constants, )
        potential_args = (numba_func_constants, )
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(combinerIdealFastFunction, force_args, potential_args, is_coord_in_vacuum_args)
        q=np.array([5e-3,5e-3,5e-3])

    def compute_Trajectory_Length(self, qTracedArr: np.ndarray) -> float:
        # to find the trajectory length model the trajectory as a bunch of little deltas for each step and add up their
        # length
        x = qTracedArr[:, 0]
        y = qTracedArr[:, 1]
        xDelta = np.append(x[0], x[1:] - x[:-1])  # have to add the first value to the length of difference because
        # it starts at zero
        yDelta = np.append(y[0], y[1:] - y[:-1])
        dLArr = np.sqrt(xDelta ** 2 + yDelta ** 2)
        Lo = float(np.sum(dLArr))
        return Lo

    def transform_lab_coords_into_element_frame(self, q_lab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        q_el = self.transform_Lab_Frame_Vector_Into_Element_Frame(q_lab - self.r2)  # a simple vector trick
        return q_el

    def transform_element_coords_into_local_orbit_frame(self, q_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        # NOTE: THIS NOT GOING TO BE CORRECT IN GENERALY BECAUSE THE TRAJECTORY IS NOT SMOOTH AND I HAVE NOT WORKED IT OUT
        # YET
        qo = q_el.copy()
        qo[0] = self.Lo - qo[0]
        qo[1] = 0  # qo[1]
        return qo

    def transform_element_momentum_into_local_orbit_frame(self, q_el: np.ndarray, p_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Not supported at the moment, so returns np.nan array instead"""

        return np.array([np.nan, np.nan, np.nan])

    def transform_element_coords_into_lab_frame(self, q_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qNew = q_el.copy()
        qNew[:2] = self.ROut @ qNew[:2] + self.r2[:2]
        return qNew

    def transform_orbit_frame_into_lab_frame(self, q_orbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qNew = q_orbit.copy()
        qNew[0] = -qNew[0]
        qNew[:2] = self.ROut @ qNew[:2]
        qNew += self.r1
        return qNew
