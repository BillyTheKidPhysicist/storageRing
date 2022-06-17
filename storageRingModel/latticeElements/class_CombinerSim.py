import numpy as np
import pandas as pd

from latticeElements.class_CombinerIdeal import CombinerIdeal
from numbaFunctionsAndObjects.fieldHelpers import get_Combiner_Sim


class CombinerSim(CombinerIdeal):

    def __init__(self, PTL, combinerFileName: str, mode: str, sizeScale: float = 1.0):
        # PTL: particle tracing lattice object
        # combinerFile: File with data with dimensions (n,6) where n is the number of points and each row is
        # (x,y,z,gradxB,gradyB,gradzB,B). Data must have come from a grid. Data must only be from the upper quarter
        # quadrant, ie the portion with z>0 and x< length/2
        # mode: wether the combiner is functioning as a loader, or a circulator.
        # sizescale: factor to scale up or down all dimensions. This modifies the field strength accordingly, ie
        # doubling dimensions halves the gradient
        assert mode in ('injector', 'storageRing')
        assert sizeScale > 0 and isinstance(combinerFileName, str)
        Lm = .187
        apL = .015
        apR = .025
        apZ = 6e-3
        super().__init__(PTL, Lm, np.nan, np.nan, apL, apR, apZ, sizeScale)
        self.fringeSpace = 5 * 1.1e-2
        self.combinerFileName = combinerFileName

    def open_And_Shape_Field_Data(self):
        data = np.asarray(pd.read_csv(self.combinerFileName, delim_whitespace=True, header=None))

        # use the new size scaling to adjust the provided data
        data[:, :3] = data[:, :3] * self.sizeScale  # scale the dimensions
        data[:, 3:6] = data[:, 3:6] / self.sizeScale  # scale the field gradient
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        fieldData = self.shape_Field_Data_3D(data)
        return fieldData

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        from latticeElements.combiner_characterizer import characterize_CombinerSim
        self.space = self.fringeSpace * self.sizeScale  # extra space past the hard edge on either end to account for fringe fields
        self.apL = self.apL * self.sizeScale
        self.apR = self.apR * self.sizeScale
        self.apz = self.apz * self.sizeScale

        self.fastFieldHelper = get_Combiner_Sim([self.open_And_Shape_Field_Data(), np.nan, self.Lb, self.Lm,
                                                 self.space, self.apL, self.apR, self.apz, np.nan,
                                                 self.fieldFact])
        inputAngle, inputOffset, trajectoryLength = characterize_CombinerSim(self)
        self.L = self.Lo = trajectoryLength
        self.ang = inputAngle
        y0 = inputOffset
        x0 = self.space
        theta = inputAngle
        self.La = (y0 + x0 / np.tan(theta)) / (np.sin(theta) + np.cos(theta) ** 2 / np.sin(theta))

        self.inputOffset = inputOffset - np.tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge

        self.update_Field_Fact(self.fieldFact)

    def build_Fast_Field_Helper(self, extraFieldSources) -> None:
        self.fastFieldHelper = get_Combiner_Sim([self.open_And_Shape_Field_Data(), self.La, self.Lb,
                                                 self.Lm, self.space, self.apL, self.apR, self.apz, self.ang,
                                                 self.fieldFact])

    def update_Field_Fact(self, fieldStrengthFact) -> None:
        self.fastFieldHelper.numbaJitClass.fieldFact = fieldStrengthFact
        self.fieldFact = fieldStrengthFact
