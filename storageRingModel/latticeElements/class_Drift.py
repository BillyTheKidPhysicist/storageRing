import warnings
from math import sqrt, isclose
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.optimize as spo
from scipy.spatial.transform import Rotation as Rot
from shapely.geometry import Polygon

import fastNumbaMethodsAndClass
from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from HalbachLensClass import SegmentedBenderHalbach as _HalbachBenderFieldGenerator
from HalbachLensClass import billyHalbachCollectionWrapper
from constants import SIMULATION_MAGNETON, VACUUM_TUBE_THICKNESS, MIN_MAGNET_MOUNT_THICKNESS
from latticeElements.utilities import ELEMENT_PLOT_COLORS
from latticeElements.class_LensIdeal import LensIdeal
from helperTools import arr_Product, iscloseAll, make_Odd, max_Tube_Radius_In_Segmented_Bend

# todo: this needs a good scrubbing and refactoring


realNumber = (int, float)
lst_tup_arr = Union[list, tuple, np.ndarray]

TINY_STEP = 1e-9
TINY_OFFSET = 1e-12  # tiny offset to avoid out of bounds right at edges of element
SMALL_OFFSET = 1e-9  # small offset to avoid out of bounds right at edges of element
MAGNET_ASPECT_RATIO = 4  # length of individual neodymium magnet relative to width of magnet


class Drift(LensIdeal):
    """
    Simple model of free space. Effectively a cylinderical vacuum tube
    """

    def __init__(self, PTL, L: float, ap: float, outerHalfWidth: Optional[float],
                 inputTiltAngle: float, outputTiltAngle: float):
        super().__init__(PTL, L, 0, np.inf, ap)  # set Bp to zero and bore radius to infinite
        self.plotColor = ELEMENT_PLOT_COLORS['drift']
        self.inputTiltAngle, self.outputTiltAngle = inputTiltAngle, outputTiltAngle
        self.fastFieldHelper = self.init_fastFieldHelper([L, ap, inputTiltAngle, outputTiltAngle])
        self.outerHalfWidth = ap + VACUUM_TUBE_THICKNESS if outerHalfWidth is None else outerHalfWidth
        assert self.outerHalfWidth > ap

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.Lo = self.L