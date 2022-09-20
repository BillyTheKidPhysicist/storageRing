"""
Module for collecting elements in one place and ease importing.
"""
from typing import Union

from lattice_elements.bender_ideal import BenderIdeal
from lattice_elements.bender_sim import BenderSim
from lattice_elements.combiner_ideal import CombinerIdeal
from lattice_elements.combiner_lens_sim import CombinerLensSim
from lattice_elements.combiner_quad_sim import CombinerSim
from lattice_elements.drift import Drift
from lattice_elements.lens_ideal import LensIdeal
from lattice_elements.lens_sim import HalbachLensSim

Element = Union[CombinerLensSim, BenderIdeal, CombinerIdeal, CombinerSim, Drift, LensIdeal, BenderSim, HalbachLensSim]

ELEMENT_PLOT_COLORS = {Drift: 'grey', LensIdeal: 'magenta', HalbachLensSim: 'magenta', CombinerIdeal: 'blue',
                       CombinerSim: 'blue', CombinerLensSim: 'blue', BenderSim: 'black',
                       BenderIdeal: 'black'}
