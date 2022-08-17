from math import isclose, sqrt
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from lattice_elements.utilities import calc_unit_cell_angle
from type_hints import sequence


def norm_2D(vec: sequence) -> float:
    """Quickly get norm of a 2D vector. Faster than numpy.linalg.norm. """
    assert len(vec) == 2
    return sqrt(vec[0] ** 2 + vec[1] ** 2)


class Shape:

    def __init__(self):
        self.pos_in = None
        self.pos_out = None
        self.norm_in = None
        self.norm_out = None

    def get_pos_and_normal(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        """Get position coordinates and normal vector of input or output of an element"""

        assert which in ('out', 'in')
        if which == 'out':
            return self.pos_out, self.norm_out
        else:
            return self.pos_in, self.norm_in

    def is_placed(self) -> bool:
        """Check if the element has been placed/built. If not, some parameters are unfilled and cannot be used"""

        return all(val is not None for val in [self.pos_in, self.pos_out, self.norm_in, self.norm_out])

    def daisy_chain(self, geometry) -> None:
        """Using a previous element which has already been placed, place this the current element"""

        raise NotImplementedError

    def place(self, *args, **kwargs):
        """With arguments required to constrain the position of a shape, set the location parameters of the shape"""

        raise NotImplementedError

    def get_plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Get coordinates for plotting. plot with plt.plot(*coords)"""

        raise NotImplementedError


class Line(Shape):
    """A simple line geometry"""

    def __init__(self, length: Optional[float], constrained: bool = False):
        assert length > 0.0 if length is not None else True
        super().__init__()
        self.length = length
        self.constrained = constrained

    def set_length(self, length: float) -> None:
        """Set the length of the line"""

        assert length > 0.0
        self.length = length

    def get_plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.is_placed()
        x_vals = np.array([self.pos_in[0], self.pos_out[0]])
        y_vals = np.array([self.pos_in[1], self.pos_out[1]])
        return x_vals, y_vals

    def place(self, pos_in: np.ndarray, norm_in: np.ndarray) -> None:
        self.pos_in = pos_in
        self.norm_in = norm_in
        self.norm_out = -1 * self.norm_in
        self.pos_out = self.pos_in + self.length * self.norm_out

    def daisy_chain(self, geometry: Shape) -> None:
        pos_in, norm_in = geometry.pos_out, -geometry.norm_out
        self.place(pos_in, norm_in)


class LineWithAngledEnds(Line):
    def __init__(self, length, inputTilt, outputTilt):
        assert abs(inputTilt) < np.pi / 2 and abs(outputTilt) < np.pi / 2
        super().__init__(length)
        self.inputTilt = inputTilt
        self.outputTilt = outputTilt

    def place(self, pos_in: np.ndarray, norm_in: np.ndarray) -> None:  # pylint: disable=arguments-differ
        self.pos_in = pos_in
        self.norm_in = norm_in
        from scipy.spatial.transform import Rotation as Rot
        self.pos_out = self.pos_in + self.length * self.input_to_output_unit_vec()
        self.norm_out = Rot.from_rotvec([0, 0, self.outputTilt - self.inputTilt]).as_matrix()[:2, :2] @ (-norm_in)

    def input_to_output_unit_vec(self):
        from scipy.spatial.transform import Rotation as Rot
        return -Rot.from_rotvec([0, 0, -self.inputTilt]).as_matrix()[:2, :2] @ self.norm_in


class Bend(Shape):
    """A simple bend geometry, ie an arc of a circle. Angle convention is opposite of usual, ie clockwise is positive
    angles"""

    def __init__(self, radius: float, bending_ang: Optional[float]):
        assert radius > 0.0
        super().__init__()
        self.radius = radius
        self.bending_ang = bending_ang
        self.benderCenter = None

    def set_arc_angle(self, angle):
        """Set the arc angle (bending angle) of the bend bend"""

        assert 0 < angle <= 2 * np.pi
        self.bending_ang = angle

    def set_radius(self, radius):
        """Set the radius of the bender"""

        assert radius > 0.0
        self.radius = radius

    def get_plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.is_placed()
        angle_in = np.arctan2(self.norm_in[1], self.norm_in[0]) - np.pi / 2
        angles = np.linspace(0, -self.bending_ang, 10_000) + angle_in
        x_vals = self.radius * np.cos(angles) + self.benderCenter[0]
        y_vals = self.radius * np.sin(angles) + self.benderCenter[1]
        return x_vals, y_vals

    def place(self, pos_in: np.ndarray, norm_in: np.ndarray) -> None:  # pylint: disable=arguments-differ
        assert isclose(norm_2D(norm_in), 1.0, abs_tol=1e-12)
        self.pos_in, self.norm_in = pos_in, norm_in
        _radiusVector = [-self.norm_in[1] * self.radius, self.norm_in[0] * self.radius]
        self.benderCenter = [self.pos_in[0] + _radiusVector[0], self.pos_in[1] + _radiusVector[1]]
        R = Rotation.from_rotvec([0, 0, -self.bending_ang]).as_matrix()[:2, :2]
        self.pos_out = R @ (self.pos_in - self.benderCenter) + self.benderCenter
        self.norm_out = R @ (-1 * self.norm_in)

    def daisy_chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.norm_out)


class SlicedBend(Bend):
    """Bending geometry, but with the geometry composed of integer numbers of segments"""

    def __init__(self, length_seg: float, num_magnets: Optional[int], magnet_depth: float, radius: float):
        assert length_seg > 0.0
        assert (num_magnets > 0 and isinstance(num_magnets, int)) if num_magnets is not None else True
        assert 0.0 <= magnet_depth < .1 and radius > 10 * magnet_depth  # this wouldn't make sense
        self.length_seg = length_seg
        self.num_magnets = num_magnets
        self.magnet_depth = magnet_depth
        self.radius = radius
        self.bending_ang = self.get_arc_angle() if num_magnets is not None else None
        super().__init__(radius, self.bending_ang)

    def get_arc_angle(self) -> float:
        """Get arc angle (bending angle) of bender"""

        unit_cell_angle = calc_unit_cell_angle(self.length_seg, self.radius, self.magnet_depth)
        bending_ang = 2 * unit_cell_angle * self.num_magnets
        assert 0 < bending_ang < 2 * np.pi
        return bending_ang

    def set_number_magnets(self, num_magnets: int) -> None:
        """Set number of magnets (half number of unit cells) in the bender"""

        assert num_magnets > 0 and isinstance(num_magnets, int)
        self.num_magnets = num_magnets
        self.bending_ang = self.get_arc_angle()  # radians

    def set_radius(self, radius: float) -> None:
        super().set_radius(radius)
        self.bending_ang = self.get_arc_angle()  # radians


class CappedSlicedBend(SlicedBend):

    def __init__(self, length_seg: float, num_magnets: Optional[int],  # pylint: disable=too-many-arguments
                 magnet_depth: float, lengthCap: float, radius: float):
        super().__init__(length_seg, num_magnets, magnet_depth, radius)
        self.lengthCap = lengthCap
        self.caps: list[Line] = [Line(self.lengthCap), Line(self.lengthCap)]

    def place(self, pos_in: np.ndarray, norm_in: np.ndarray) -> None:  # pylint: disable=arguments-differ
        self.caps[0].place(pos_in, norm_in)
        super().place(self.caps[0].pos_out, -1 * self.caps[0].norm_out)
        self.caps[1].daisy_chain(self)
        self.pos_in, self.norm_in = self.caps[0].pos_in, self.caps[0].norm_in
        self.pos_out, self.norm_out = self.caps[1].pos_out, self.caps[1].norm_out

    def daisy_chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.norm_out)

    def get_plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        x1_vals, y1_vals = self.caps[0].get_plot_coords()
        x2_vals, y2_vals = super().get_plot_coords()
        x3_vals, y3_vals = self.caps[1].get_plot_coords()
        x_vals = np.concatenate((x1_vals, x2_vals, x3_vals))
        y_vals = np.concatenate((y1_vals, y2_vals, y3_vals))
        return x_vals, y_vals


class Kink(Shape):
    """Two line meeting at an angle. Represents the combiner element"""

    def __init__(self, kinkAngle: float, La: float, Lb: float):
        super().__init__()
        self.kinkAngle = kinkAngle
        self.La = La
        self.Lb = Lb

    def get_plot_coords(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.is_placed()
        x_vals, y_vals = [self.pos_in[0]], [self.pos_in[1]]
        pos_center = self.pos_in + (-1 * self.norm_in) * self.La
        x_vals.extend([pos_center[0], self.pos_out[0]])
        y_vals.extend([pos_center[1], self.pos_out[1]])
        return np.array(x_vals), np.array(y_vals)

    def place(self, pos_in: np.ndarray, norm_in: np.ndarray) -> None:
        assert isclose(norm_2D(norm_in), 1.0, abs_tol=1e-12)
        self.pos_in, self.norm_in = pos_in, norm_in
        rot_matrix = Rotation.from_rotvec([0, 0, self.kinkAngle]).as_matrix()[:2, :2]
        self.norm_out = rot_matrix @ (-1 * self.norm_in)
        pos_center = self.pos_in + (-1 * self.norm_in) * self.La
        self.pos_out = pos_center + self.norm_out * self.Lb

    def daisy_chain(self, geometry: Shape) -> None:
        self.place(geometry.pos_out, -geometry.norm_out)
