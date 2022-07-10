import numpy as np
import pandas as pd

from latticeElements.class_CombinerIdeal import CombinerIdeal
from numbaFunctionsAndObjects import combinerSimFastFunction


class CombinerSim(CombinerIdeal):

    def __init__(self, PTL, combinerFileName: str, mode: str, size_scale: float = 1.0):
        # PTL: particle tracing lattice object
        # combinerFile: File with data with dimensions (n,6) where n is the number of points and each row is
        # (x,y,z,gradxB,gradyB,gradzB,B). Data must have come from a grid. Data must only be from the upper quarter
        # quadrant, ie the portion with z>0 and x< length/2
        # mode: wether the combiner is functioning as a loader, or a circulator.
        # sizescale: factor to scale up or down all dimensions. This modifies the field strength accordingly, ie
        # doubling dimensions halves the gradient
        assert mode in ('injector', 'storageRing')
        assert size_scale > 0 and isinstance(combinerFileName, str)
        Lm = .187
        apL = .015
        apR = .025
        apZ = 6e-3
        super().__init__(PTL, Lm, np.nan, np.nan, apL, apR, apZ, size_scale)
        self.fringeSpace = 5 * 1.1e-2
        self.combinerFileName = combinerFileName

    def open_And_Shape_Field_Data(self):
        data = np.asarray(pd.read_csv(self.combinerFileName, delim_whitespace=True, header=None))

        # use the new size scaling to adjust the provided data
        data[:, :3] = data[:, :3] * self.size_scale  # scale the dimensions
        data[:, 3:6] = data[:, 3:6] / self.size_scale  # scale the field gradient
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        field_data = self.shape_field_data_3D(data)
        return field_data

    def fill_pre_constrained_parameters(self) -> None:
        """Overrides abstract method from Element"""
        from latticeElements.combiner_characterizer import characterize_CombinerSim
        self.space = self.fringeSpace * self.size_scale  # extra space past the hard edge on either end to account for fringe fields
        self.apL = self.apL * self.size_scale
        self.apR = self.apR * self.size_scale
        self.apz = self.apz * self.size_scale

        inputAngle, inputOffset, trajectoryLength = characterize_CombinerSim(self)
        self.L = self.Lo = trajectoryLength
        self.ang = inputAngle
        y0 = inputOffset
        x0 = self.space
        theta = inputAngle
        self.La = (y0 + x0 / np.tan(theta)) / (np.sin(theta) + np.cos(theta) ** 2 / np.sin(theta))

        self.inputOffset = inputOffset - np.tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge

    def build_fast_field_helper(self) -> None:
        numba_func_constants = (self.ang, self.La, self.Lb, self.Lm,self.apz,self.apL,self.apR, self.space, self.field_fact)

        field_data=self.open_And_Shape_Field_Data()

        force_args = (numba_func_constants, field_data)
        potential_args = (numba_func_constants, field_data)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(combinerSimFastFunction, force_args, potential_args, is_coord_in_vacuum_args)

    def update_Field_Fact(self, fieldStrengthFact) -> None:
        raise NotImplementedError
        self.fast_field_helper.numbaJitClass.field_fact = fieldStrengthFact
        self.field_fact = fieldStrengthFact
