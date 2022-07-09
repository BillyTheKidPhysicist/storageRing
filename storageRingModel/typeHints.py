from typing import Union
import numpy as np

from typing import Protocol,Sequence

class sequence(Protocol):
    """This type captures the features that are common between a list, tuple, array and dict. Surprisingly this
    feature doesn't exist yet. Different than collections.abc.Sequence! """
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __contains__(self, item):
        raise NotImplementedError
    def __iter__(self):
        raise NotImplementedError

FloatTuple=tuple[float,...]
RealNum = Union[int, float]