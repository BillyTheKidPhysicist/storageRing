from math import isclose, sqrt
from typing import Union, Optional

import numpy as np
from scipy.spatial.transform import Rotation

lst_arr_tple = Union[list, np.ndarray, tuple]


def norm_2D(vec: lst_arr_tple) -> float:
    """Quickly get norm of a 2D vector. Faster than numpy.linalg.norm. """
    assert len(vec) == 2
    return sqrt(vec[0] ** 2 + vec[1] ** 2)


class Shape:

    def __init__(self):
        self.pos_in: np.ndarray = None
        self.pos_out: np.ndarray = None
        self.n_in: np.ndarray = None
        self.n_out: np.ndarray = None

    def get_Pos_And_Normal(self, which: str) -> tuple[np.ndarray, np.ndarray]:
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

    def place(self, *args, **kwargs):
        """With arguments required to constrain the position of a shape, set the location parameters of the shape"""

        raise NotImplementedError

    def get_Plot_Coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Get coordinates for plotting. plot with plt.plot(*coords)"""

        raise NotImplementedError


class Line(Shape):
    """A simple line geometry"""

    def __init__(self, length: Optional[float], constrained: bool = False):
        assert length > 0.0 if length is not None else True
        super().__init__()
        self.length = length
        self.constrained = constrained

    def set_Length(self, length: float) -> None:
        """Set the length of the line"""

        assert length > 0.0
        self.length = length

    def get_Plot_Coords(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.is_Placed()
        xVals = np.array([self.pos_in[0], self.pos_out[0]])
        yVals = np.array([self.pos_in[1], self.pos_out[1]])
        return xVals, yVals

    def place(self, pos_in: np.ndarray, n_in: np.ndarray) -> None:
        self.pos_in = pos_in
        self.n_in = n_in
        self.n_out = -1 * self.n_in
        self.pos_out = self.pos_in + self.length * self.n_out

    def daisy_Chain(self, geometry: Shape) -> None:
        pos_in, n_in = geometry.pos_out, -geometry.n_out
        self.place(pos_in, n_in)


class LineWithAngledEnds(Line):
    def __init__(self, length, inputTilt, outputTilt):
        assert abs(inputTilt) < np.pi / 2 and abs(outputTilt) < np.pi / 2
        super().__init__(length)
        self.inputTilt = inputTilt
        self.outputTilt = outputTilt

    def place(self, pos_in: np.ndarray, n_in: np.ndarray) -> None:  # pylint: disable=arguments-differ
        self.pos_in = pos_in
        self.n_in = n_in
        from scipy.spatial.transform import Rotation as Rot
        self.pos_out = self.pos_in + self.length * self.n_From_Input_To_Output_Pos()
        self.n_out = Rot.from_rotvec([0, 0, self.outputTilt - self.inputTilt]).as_matrix()[:2, :2] @ (-n_in)

    def n_From_Input_To_Output_Pos(self):
        from scipy.spatial.transform import Rotation as Rot
        return -Rot.from_rotvec([0, 0, -self.inputTilt]).as_matrix()[:2, :2] @ self.n_in


class Bend(Shape):
    """A simple bend geometry, ie an arc of a circle. Angle convention is opposite of usual, ie clockwise is positive
    angles"""

    def __init__(self, radius: float, bendingAngle: Optional[float]):
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

    def place(self, pos_in: np.ndarray, n_in: np.ndarray) -> None:  # pylint: disable=arguments-differ
        assert isclose(norm_2D(n_in), 1.0, abs_tol=1e-12)
        self.pos_in, self.n_in = pos_in, n_in
        _radiusVector = [-self.n_in[1] * self.radius, self.n_in[0] * self.radius]
        self.benderCenter = [self.pos_in[0] + _radiusVector[0], self.pos_in[1] + _radiusVector[1]]
        R = Rotation.from_rotvec([0, 0, -self.bendingAngle]).as_matrix()[:2, :2]
        self.pos_out = R @ (self.pos_in - self.benderCenter) + self.benderCenter
        self.n_out = R @ (-1 * self.n_in)

    def daisy_Chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.n_out)


class SlicedBend(Bend):
    """Bending geometry, but with the geometry composed of integer numbers of segments"""

    def __init__(self, lengthSegment: float, numMagnets: Optional[int], magnetDepth: float, radius: float):
        assert lengthSegment > 0.0
        assert (numMagnets > 0 and isinstance(numMagnets, int)) if numMagnets is not None else True
        assert 0.0 <= magnetDepth < .1 and radius > 10 * magnetDepth  # this wouldn't make sense
        self.lengthSegment = lengthSegment
        self.numMagnets = numMagnets
        self.magnetDepth = magnetDepth
        self.radius = radius
        self.bendingAngle = self.get_Arc_Angle() if numMagnets is not None else None
        super().__init__(radius, self.bendingAngle)

    def get_Unit_Cell_Angle(self) -> float:
        """Get the arc angle associate with a single unit cell. Each magnet contains two unit cells."""

        return np.arctan(.5 * self.lengthSegment / (self.radius - self.magnetDepth))  # radians

    def get_Arc_Angle(self) -> float:
        """Get arc angle (bending angle) of bender"""

        unitCellAngle = self.get_Unit_Cell_Angle()  # radians
        bendingAngle = 2 * unitCellAngle * self.numMagnets
        assert 0 < bendingAngle < 2 * np.pi
        return bendingAngle

    def set_Number_Magnets(self, numMagnets: int) -> None:
        """Set number of magnets (half number of unit cells) in the bender"""

        assert numMagnets > 0 and isinstance(numMagnets, int)
        self.numMagnets = numMagnets
        self.bendingAngle = self.get_Arc_Angle()  # radians

    def set_Radius(self, radius: float) -> None:
        super().set_Radius(radius)
        self.bendingAngle = self.get_Arc_Angle()  # radians


class CappedSlicedBend(SlicedBend):

    def __init__(self, lengthSegment: float, numMagnets: Optional[int],  # pylint: disable=too-many-arguments
                 magnetDepth: float, lengthCap: float, radius: float):
        super().__init__(lengthSegment, numMagnets, magnetDepth, radius)
        self.lengthCap = lengthCap
        self.caps: list[Line] = [Line(self.lengthCap), Line(self.lengthCap)]

    def place(self, pos_in: np.ndarray, n_in: np.ndarray) -> None:  # pylint: disable=arguments-differ
        self.caps[0].place(pos_in, n_in)
        super().place(self.caps[0].pos_out, -1 * self.caps[0].n_out)
        self.caps[1].daisy_Chain(self)
        self.pos_in, self.n_in = self.caps[0].pos_in, self.caps[0].n_in
        self.pos_out, self.n_out = self.caps[1].pos_out, self.caps[1].n_out

    def daisy_Chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.n_out)

    def get_Plot_Coords(self) -> tuple[np.ndarray, np.ndarray]:
        x1Vals, y1Vals = self.caps[0].get_Plot_Coords()
        x2Vals, y2Vals = super().get_Plot_Coords()
        x3Vals, y3Vals = self.caps[1].get_Plot_Coords()
        xVals = np.concatenate((x1Vals, x2Vals, x3Vals))
        yVals = np.concatenate((y1Vals, y2Vals, y3Vals))
        return xVals, yVals


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
        posCenter = self.pos_in + (-1 * self.n_in) * self.La
        xVals.extend([posCenter[0], self.pos_out[0]])
        yVals.extend([posCenter[1], self.pos_out[1]])
        return np.array(xVals), np.array(yVals)

    def place(self, pos_in: np.ndarray, n_in: np.ndarray) -> None:
        assert isclose(norm_2D(n_in), 1.0, abs_tol=1e-12)
        self.pos_in, self.n_in = pos_in, n_in
        rotMatrix = Rotation.from_rotvec([0, 0, self.kinkAngle]).as_matrix()[:2, :2]
        self.n_out = rotMatrix @ (-1 * self.n_in)
        posCenter = self.pos_in + (-1 * self.n_in) * self.La
        self.pos_out = posCenter + self.n_out * self.Lb

    def daisy_Chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.n_out)
