from typing import Union

from latticeElements.elements import Drift, HalbachBenderSimSegmented, CombinerHalbachLensSim, HalbachLensSim, \
    LensIdeal, BenderIdeal, CombinerIdeal, CombinerSim

element = Union[CombinerHalbachLensSim, HalbachLensSim, CombinerSim, CombinerIdeal,
                BenderIdeal, HalbachBenderSimSegmented, Drift, LensIdeal]
