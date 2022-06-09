from latticeElements.class_BaseElement import BaseElement
from math import sqrt
from latticeElements.utilities import ELEMENT_PLOT_COLORS,full_Arctan

import numpy as np
# from latticeElements.class_CombinerHalbachLensSim import CombinerHalbachLensSim
from scipy.spatial.transform import Rotation as Rot

class CombinerIdeal(BaseElement):
    # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
    # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
    # angle is decided by tracing particles through the combiner and finding the bending angle.

    def __init__(self, PTL, Lm: float, c1: float, c2: float, apL: float, apR: float, apZ: float, mode: str,
                 sizeScale: float):
        super().__init__( PTL, ELEMENT_PLOT_COLORS['combiner'])
        assert mode in ('injector', 'storageRing')
        self.fieldFact = -1.0 if mode == 'injector' else 1.0
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
        self.apR, self.apL, self.apz, self.Lm = [val * self.sizeScale for val in
                                                 (self.apR, self.apL, self.apz, self.Lm)]
        self.c1, self.c2 = self.c1 / self.sizeScale, self.c2 / self.sizeScale
        self.Lb = self.Lm  # length of segment after kink after the inlet
        self.fastFieldHelper = self.init_fastFieldHelper([self.c1, self.c2, np.nan, self.Lb,
                                                          self.apL, self.apR, np.nan, np.nan])
        inputAngle, inputOffset, qTracedArr, _ = self.compute_Input_Angle_And_Offset()
        self.Lo = np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = inputAngle
        self.inputOffset = inputOffset
        self.La = .5 * (self.apR + self.apL) * np.sin(self.ang)
        self.L = self.La * np.cos(
            self.ang) + self.Lb  # TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING. Is it used?
        self.fastFieldHelper = self.init_fastFieldHelper([self.c1, self.c2, self.La, self.Lb,
                                                          self.apL, self.apR, self.apz, self.ang])

    def compute_Input_Angle_And_Offset(self, outputOffset: float = 0.0, h: float = 1e-6,
                                       ap: float = None) -> tuple:
        # this computes the output angle and offset for a combiner magnet.
        # NOTE: for the ideal combiner this gives slightly inaccurate results because of lack of conservation of energy!
        # NOTE: for the simulated bender, this also give slightly unrealisitc results because the potential is not allowed
        # to go to zero (finite field space) so the the particle will violate conservation of energy
        # limit: how far to carry the calculation for along the x axis. For the hard edge magnet it's just the hard edge
        # length, but for the simulated magnets, it's that plus twice the length at the ends.
        # h: timestep
        # lowField: wether to model low or high field seekers
        # if type(self) == CombinerHalbachLensSim:
        try:
            assert 0.0 <= outputOffset < self.ap
        except:
            pass

        def force(x):
            if ap is not None and (x[0] < self.Lm + self.space and sqrt(x[1] ** 2 + x[2] ** 2) > ap):
                return np.empty(3) * np.nan
            Force = np.array(self.fastFieldHelper.force_Without_isInside_Check(x[0], x[1], x[2]))
            Force[2] = 0.0  ##only interested in xy plane bending
            return Force

        q = np.asarray([0.0, -outputOffset, 0.0])
        p = np.asarray([self.PTL.v0Nominal, 0.0, 0.0])
        qList = []

        xPosStopTracing = self.Lm + 2 * self.space
        forcePrev = force(q)  # recycling the previous force value cut simulation time in half
        while True:
            F = forcePrev
            q_n = q + p * h + .5 * F * h ** 2
            if not 0 <= q_n[0] <= xPosStopTracing:  # if overshot, go back and walk up to the edge assuming no force
                dr = xPosStopTracing - q[0]
                dt = dr / p[0]
                qFinal = q + p * dt
                pFinal = p
                qList.append(qFinal)
                break
            F_n = force(q_n)
            assert not np.any(np.isnan(F_n))
            p_n = p + .5 * (F + F_n) * h
            q, p = q_n, p_n
            forcePrev = F_n
            qList.append(q)
        assert qFinal[2] == 0.0  # only interested in xy plane bending, expected to be zero
        qArr = np.asarray(qList)
        outputAngle = np.arctan2(pFinal[1], pFinal[0])
        inputOffset = qFinal[1]
        if ap is not None:
            lensCorner = np.asarray([self.space + self.Lm, -ap, 0.0])
            minSepBottomRightMagEdge = np.min(np.linalg.norm(qArr - lensCorner, axis=1))
        else:
            minSepBottomRightMagEdge = None
        return outputAngle, inputOffset, qArr, minSepBottomRightMagEdge

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
