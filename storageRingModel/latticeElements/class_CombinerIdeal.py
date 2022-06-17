import numpy as np

from latticeElements.class_BaseElement import BaseElement
from numbaFunctionsAndObjects.fieldHelpers import get_Combiner_Ideal


# from latticeElements.class_CombinerHalbachLensSim import CombinerHalbachLensSim

class CombinerIdeal(BaseElement):
    # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
    # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
    # angle is decided by tracing particles through the combiner and finding the bending angle.

    def __init__(self, PTL, Lm: float, c1: float, c2: float, apL: float, apR: float, apZ: float, sizeScale: float):
        super().__init__(PTL)
        self.fieldFact = -1.0 if self.PTL.latticeType == 'injector' else 1.0
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
        # self.fastFieldHelper = get_Combiner_Ideal([self.c1, self.c2, np.nan, self.Lb,
        #                                                   self.apL, self.apR, np.nan, np.nan])
        inputAngle, inputOffset, trajectoryLength = characterize_CombinerIdeal(self)

        self.Lo = trajectoryLength  # np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = inputAngle
        self.inputOffset = inputOffset
        self.La = .5 * (self.apR + self.apL) * np.sin(self.ang)
        self.L = self.La * np.cos(
            self.ang) + self.Lb  # TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING. Is it used?

    def build_Fast_Field_Helper(self, extraFieldSources) -> None:
        self.fastFieldHelper = get_Combiner_Ideal([self.c1, self.c2, self.La, self.Lb,
                                                   self.apL, self.apR, self.apz, self.ang])

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

    def transform_Lab_Coords_Into_Element_Frame(self, qLab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qEl = self.transform_Lab_Frame_Vector_Into_Element_Frame(qLab - self.r2)  # a simple vector trick
        return qEl

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        # NOTE: THIS NOT GOING TO BE CORRECT IN GENERALY BECAUSE THE TRAJECTORY IS NOT SMOOTH AND I HAVE NOT WORKED IT OUT
        # YET
        qo = qEl.copy()
        qo[0] = self.Lo - qo[0]
        qo[1] = 0  # qo[1]
        return qo

    def transform_Element_Momentum_Into_Local_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Not supported at the moment, so returns np.nan array instead"""

        return np.array([np.nan, np.nan, np.nan])

    def transform_Element_Coords_Into_Lab_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qNew = qEl.copy()
        qNew[:2] = self.ROut @ qNew[:2] + self.r2[:2]
        return qNew

    def transform_Orbit_Frame_Into_Lab_Frame(self, qOrbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qNew = qOrbit.copy()
        qNew[0] = -qNew[0]
        qNew[:2] = self.ROut @ qNew[:2]
        qNew += self.r1
        return qNew
