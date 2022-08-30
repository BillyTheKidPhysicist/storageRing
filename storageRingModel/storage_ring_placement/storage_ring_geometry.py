# pylint: disable=missing-module-docstring
from typing import Iterable
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np

from storage_ring_placement.shapes import Kink, Bend, CappedSlicedBend, Line, \
    SlicedBend  # pylint: disable=import-error

Shape = Union[Kink, Bend, CappedSlicedBend, Line]


class StorageRingGeometry:
    """
    Largely a container for shapes that represent the storage ring. More exactly, these shapes represent the ideal
    orbit of a particle through the storage ring.
    """

    def __init__(self, elements: list[Shape]):
        self.elements: list[Shape] = elements
        self.combiner: Optional[Kink] = self.get_combiner(elements)
        self.benders: list[CappedSlicedBend] = self.get_benders(elements)
        self.numBenders: int = len(self.benders)

    def __iter__(self):
        return iter(self.elements)

    def get_combiner(self, elements: Iterable[Shape]) -> Optional[Kink]:
        """Get storage ring combiner. Must be one and only one"""

        combiner = None
        for element in elements:
            if isinstance(element, Kink):
                assert combiner is None  # should only be one combiner
                combiner = element
        return combiner

    def get_benders(self, elements: Iterable[Shape]) -> list[CappedSlicedBend]:
        """Get bender elements in storage ring. Currently, only sliced bender is supported"""

        benders = []
        for element in elements:
            assert not type(element) is Bend or type(element) is not SlicedBend  # not supported
            if isinstance(element, CappedSlicedBend):
                benders.append(element)
        return benders

    def build(self) -> None:
        """Build the storage ring by daisy chaining elements together"""

        assert self.elements[0].is_placed()  # first element must have been already built for daisy chaining to work
        for i, element in enumerate(self.elements):
            if i != 0:
                element.daisy_chain(self.elements[i - 1])

    def show_geometry(self) -> None:
        """Simple plot of the geometry of the storage ring"""

        for element in self.elements:
            c = 'r' if type(element) is Line else 'black'
            plt.plot(*element.get_plot_coords(), c=c)
            plt.scatter(*element.pos_out, c='black')
        plt.gca().set_aspect('equal')
        plt.show()

    def get_end_separation_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Get difference vectors for the location and colinearity of the beginning and ending of the storage ring.
        Beginning and ending should be colinear and at the same location for a valid ring"""

        first_el, last_el = self.elements[0], self.elements[-1]
        pos_initial, n_initial = first_el.get_pos_and_normal('in')
        pos_final, n_final = last_el.get_pos_and_normal('out')
        delta_pos = pos_final - pos_initial
        delta_normal = n_final + n_initial  # normals point AWAY from element input/output
        return delta_pos, delta_normal
