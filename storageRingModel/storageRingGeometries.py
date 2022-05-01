#pylint: disable= missing-module-docstring
from math import isclose,sqrt
from typing import Union
import numpy as np
from scipy.spatial.transform import Rotation

lst_arr_tple = Union[list, np.ndarray, tuple]

def norm_2D(vec: lst_arr_tple)->float:
    """Quickly get norm of a 2D vector. Faster than numpy.linalg.norm. """
    assert len(vec)==2
    return sqrt(vec[0]**2+vec[1]**2)

class Shape:

    def __init__(self):
        self.pos_in: np.ndarray = None
        self.pos_out: np.ndarray = None
        self.n_in: np.ndarray    = None
        self.n_out: np.ndarray   = None

    def get_Pos_And_Normal(self, which: str)-> tuple[np.ndarray,np.ndarray]:
        """Get position coordinates and normal vector of input or output of an element"""
        
        assert which in ('out', 'in')
        if which == 'out':
            return self.pos_out, self.n_out
        else:
            return self.pos_in, self.n_in

    def is_Placed(self) -> bool:
        """Check if the element has been placed/built. If not, some parameters are unfilled and cannot be used"""

        return all(val is not None for val in [self.pos_in, self.pos_out, self.n_in, self.n_out])

    def daisy_Chain(self, geometry) -> None:
        """Using a previous element which has already been placed, place this the current element"""

        raise NotImplementedError
    
    def place(self,*args,**kwargs):
        """With arguments required to constrain the position of a shape, set the location parameters of the shape"""
        
        raise NotImplementedError

    def get_Plot_Coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Get coordinates for plotting. plot with plt.plot(*coords)"""

        raise NotImplementedError


class Line(Shape):
    """A simple line geometry"""

    def __init__(self, length, constrained = False):
        assert length > 0.0
        super().__init__()
        self.length: float = length
        self.constrained: bool = constrained

    def set_Length(self, length: float)-> None:
        """Set the length of the line"""
        
        assert length > 0.0
        self.length = length

    def get_Plot_Coords(self) -> tuple[np.ndarray, np.ndarray]:

        assert self.is_Placed()
        xVals = np.array([self.pos_in[0], self.pos_out[0]])
        yVals = np.array([self.pos_in[1], self.pos_out[1]])
        return xVals, yVals

    def place(self, pos_in: np.ndarray, tiltAngle: float) -> None: #pylint: disable=arguments-differ
        self.pos_in = pos_in
        self.n_out = np.array([np.cos(tiltAngle), np.sin(tiltAngle)])
        self.n_in = -1*self.n_out
        self.pos_out = self.pos_in + self.length * self.n_out

    def daisy_Chain(self, geometry: Shape) -> None:
        pos_out, n_out = geometry.pos_out, geometry.n_out
        tiltAngle = np.arctan2(n_out[1], n_out[0])
        self.place(pos_out, tiltAngle)


class Bend(Shape):
    """A simple bend geometry, ie an arc of a circle. Angle convention is opposite of usual, ie clockwise is positive
    angles"""

    def __init__(self, radius: float, bendingAngle: float):
        assert radius > 0.0
        super().__init__()
        self.radius = radius
        self.bendingAngle = bendingAngle
        self.benderCenter: np.ndarray = None

    def set_Arc_Angle(self, angle):
        """Set the arc angle (bending angle) of the bend bend"""
        
        assert 0 < angle <= 2 * np.pi
        self.bendingAngle = angle

    def set_Radius(self, radius):
        """Set the radius of the bender"""

        assert radius > 0.0
        self.radius = radius

    def get_Plot_Coords(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.is_Placed()
        angleIn = np.arctan2(self.n_in[1], self.n_in[0]) - np.pi / 2
        angleArr = np.linspace(0, -self.bendingAngle, 10_000) + angleIn
        xVals = self.radius * np.cos(angleArr) + self.benderCenter[0]
        yVals = self.radius * np.sin(angleArr) + self.benderCenter[1]
        return xVals, yVals

    def place(self, pos_in: np.ndarray, n_in: np.ndarray) -> None: #pylint: disable=arguments-differ
        assert n_in[0] != 0.0 and n_in[1] != 0.0 and isclose(norm_2D(n_in), 1.0, abs_tol=1e-12)
        self.pos_in, self.n_in = pos_in, n_in
        _radiusVector = [-self.n_in[1]* self.radius, self.n_in[0]* self.radius]
        self.benderCenter=[self.pos_in[0]+_radiusVector[0],self.pos_in[1]+_radiusVector[1]]
        R = Rotation.from_rotvec([0, 0, -self.bendingAngle]).as_matrix()[:2, :2]
        self.pos_out = R @ (self.pos_in - self.benderCenter) + self.benderCenter
        self.n_out = R @ (-1*self.n_in)

    def daisy_Chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.n_out)


class SlicedBend(Bend):
    """Bending geometry, but with the geometry composed of integer numbers of segments"""

    def __init__(self, lengthSegment: float, numMagnets: int, radius: float):
        assert lengthSegment > 0.0 and numMagnets > 0.0 and isinstance(numMagnets, int)
        self.lengthSegment = lengthSegment
        self.numMagnets = numMagnets
        self.radius = radius
        self.bendingAngle = self.get_Arc_Angle()
        super().__init__(radius, self.bendingAngle)

    def get_Unit_Cell_Angle(self) -> float:
        """Get the arc angle associate with a single unit cell. Each magnet contains two unit cells."""

        return np.arctan(self.lengthSegment / (2 * self.radius)) #radians

    def get_Arc_Angle(self) -> float:
        """Get arc angle (bending angle) of bender"""

        unitCellAngle = self.get_Unit_Cell_Angle() #radians
        bendingAngle = 2 * unitCellAngle * self.numMagnets
        assert 0 < bendingAngle < 2 * np.pi
        return bendingAngle

    def set_Number_Magnets(self, numMagnets: int) -> None:
        """Set number of magnets (half number of unit cells) in the bender"""

        assert numMagnets > 0 and isinstance(numMagnets, int)
        self.numMagnets = numMagnets
        self.bendingAngle = self.get_Arc_Angle() #radians

    def set_Radius(self, radius: float) -> None:
        super().set_Radius(radius)
        self.bendingAngle = self.get_Arc_Angle() #radians


class Kink(Shape):
    """Two line meeting at an angle. Represents the combiner element"""

    def __init__(self, kinkAngle: float, La: float, Lb: float):

        super().__init__()
        self.kinkAngle = kinkAngle
        self.La = La
        self.Lb = Lb

    def get_Plot_Coords(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.is_Placed()
        xVals, yVals = [self.pos_in[0]], [self.pos_in[1]]
        posCenter = self.pos_in + (-1*self.n_in) * self.La
        xVals.extend([posCenter[0], self.pos_out[0]])
        yVals.extend([posCenter[1], self.pos_out[1]])
        return np.array(xVals), np.array(yVals)

    def place(self, pos_in: np.ndarray, n_in: np.ndarray) -> None: #pylint: disable=arguments-differ
        assert n_in[0] != 0.0 and n_in[1] != 0.0
        assert isclose(norm_2D(n_in), 1.0, abs_tol=1e-12)
        self.pos_in, self.n_in = pos_in, n_in
        rotMatrix = Rotation.from_rotvec([0, 0, self.kinkAngle]).as_matrix()[:2, :2]
        self.n_out = rotMatrix @ (-1*self.n_in)
        posCenter = self.pos_in + (-1*self.n_in) * self.La
        self.pos_out = posCenter + self.n_out * self.Lb

    def daisy_Chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.n_out)
