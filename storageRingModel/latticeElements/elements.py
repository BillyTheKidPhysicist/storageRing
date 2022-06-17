from typing import Union

from latticeElements.class_BenderIdeal import BenderIdeal
from latticeElements.class_CombinerHalbachLensSim import CombinerHalbachLensSim
from latticeElements.class_CombinerIdeal import CombinerIdeal
from latticeElements.class_CombinerSim import CombinerSim
from latticeElements.class_Drift import Drift
from latticeElements.class_HalbachBenderSegmented import HalbachBenderSimSegmented
from latticeElements.class_HalbachLensSim import HalbachLensSim
from latticeElements.class_LensIdeal import LensIdeal

Element = Union[CombinerHalbachLensSim, BenderIdeal, CombinerIdeal,
                CombinerSim, Drift, LensIdeal, HalbachBenderSimSegmented, HalbachLensSim]

ELEMENT_PLOT_COLORS = {Drift: 'grey', LensIdeal: 'magenta', HalbachLensSim: 'magenta', CombinerIdeal: 'blue',
                       CombinerSim: 'blue', CombinerHalbachLensSim: 'blue', HalbachBenderSimSegmented: 'black',
                       BenderIdeal: 'black'}
