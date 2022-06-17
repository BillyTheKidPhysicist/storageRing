from typing import Optional

import numpy as np
from shapely.geometry import Polygon

from constants import SIMULATION_MAGNETON


# todo: a base geometry inheritance is most logical
class BaseElement:
    """
    Base class for other elements. Contains universal attributes and methods.

    An element is the fundamental component of a neutral atom storage ring/injector. An arrangment of elements is called
    a lattice, as is done in accelerator physics. Elements are intended to be combined together such that particles can
    smoothly move from one to another, and many class variables serves this purpose. An element also contains methods
    for force vectors and magnetic potential at a point in space. It will also contain methods to generate fields values
    and construct itself, which is not always trivial.
    """

    def __init__(self, PTL, ang: float = 0.0, L=None):
        self.theta = None  # angle that describes an element's rotation in the xy plane.
        # SEE EACH ELEMENT FOR MORE DETAILS
        # -Straight elements like lenses and drifts: theta=0 is the element's input at the origin and the output pointing
        # east. for theta=90 the output is pointing up.
        # -Bending elements without caps: at theta=0 the outlet is at (bending radius,0) pointing south with the input
        # at some angle counterclockwise. a 180 degree bender would have the inlet at (-bending radius,0) pointing south.
        # force is a continuous function of r and theta, ie a revolved cross section of a hexapole
        # -Bending  elements with caps: same as without caps, but keep in mind that the cap on the output would be BELOW
        # y=0
        # combiner: theta=0 has the outlet at the origin and pointing to the west, with the inlet some distance to the right
        # and pointing in the NE direction
        # todo: r1,r2,ne,nb are not consistent. They describe either orbit coordinates, or physical element coordinates
        self.PTL = PTL  # particle tracer lattice object. Used for various constants
        self.nb: Optional[np.ndarray] = None  # normal vector to beginning (clockwise sense) of element.
        self.ne: Optional[np.ndarray] = None  # normal vector to end (clockwise sense) of element

        self.ROut: Optional[np.ndarray] = None  # 2d matrix to rotate a vector out of the element's reference frame
        self.RIn: Optional[np.ndarray] = None  # 2d matrix to rotate a vector into the element's reference frame
        self.r1: Optional[np.ndarray] = None  # 3D coordinates of beginning (clockwise sense) of element in lab frame
        self.r2: Optional[np.ndarray] = None  # 3D coordinates of ending (clockwise sense) of element in lab frame
        self.SO: Optional[Polygon] = None  # the shapely object for the element. These are used for plotting, and for
        # finding if the coordinates
        # # are inside an element that can't be found with simple geometry
        self.SO_Outer: Optional[Polygon] = None  # shapely object that represents the outer edge of the element
        self.outerHalfWidth: Optional[
            float] = None  # outer diameter/width of the element, where applicable. For example,
        # outer diam of lens is the bore radius plus magnets and mount material radial thickness
        self.ang = ang  # bending angle of the element. 0 for lenses and drifts
        self.L: Optional[float] = L
        self.index: Optional[int] = None
        self.Lo: Optional[float] = None  # length of orbit for particle. For lenses and drifts this is the same as the
        # length. This is a nominal value because for segmented benders the path length is not simple to compute
        self.outputOffset: float = 0.0  # some elements have an output offset, like from bender's centrifugal force or
        # #lens combiner
        self.fieldFact: float = 1.0  # factor to modify field values everywhere in space by, including force
        self.fastFieldHelper = None

    def build_Fast_Field_Helper(self, extraFieldSources) -> None:
        raise NotImplementedError

    def set_fieldFact(self, fieldFact: bool):
        assert fieldFact > 0.0
        self.fieldFact = fieldFact
        self.fastFieldHelper.fieldFact = fieldFact

    def perturb_Element(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float):
        """
        perturb the alignment of the element relative to the vacuum tube. The vacuum tube will remain unchanged, but
        the element will be shifted, and therefore the force it applies will be as well. This is modeled as shifting
        and rotating the supplied coordinates to force and magnetic field function, then rotating the force

        :param shiftY: Shift in the y direction in element frame
        :param shiftZ: Shift in the z direction in the element frame
        :param rotY: Rotation about y axis of the element
        :param rotZ: Rotation about z axis of the element
        :return:
        """
        self.fastFieldHelper.numbaJitClass.numbaJitClass.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)

    def magnetic_Potential(self, qEl: np.ndarray) -> float:
        """
        Return magnetic potential energy at position qEl.

        Return magnetic potential energy of a lithium atom in simulation units, where the mass of a lithium-7 atom is
        1kg, at cartesian 3D coordinate qEl in the local element frame. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.

        :param qEl: 3D cartesian position vector in local element frame, numpy.array([x,y,z])
        :return: magnetic potential energy of a lithium atom in simulation units, float
        """
        return self.fastFieldHelper.numbaJitClass.magnetic_Potential(*qEl)  # will raise NotImplementedError if called

    def force(self, qEl: np.ndarray) -> np.ndarray:
        """
        Return force at position qEl.

        Return 3D cartesian force of a lithium at cartesian 3D coordinate qEl in the local element frame. Force vector
        has simulation units where lithium-7 mass is 1kg. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.


        :param qEl: 3D cartesian position vector in local element frame,numpy.array([x,y,z])
        :return: New 3D cartesian force vector, numpy.array([Fx,Fy,Fz])
        """
        return np.asarray(self.fastFieldHelper.numbaJitClass.force(*qEl))  # will raise NotImplementedError if called

    def transform_Element_Coords_Into_Global_Orbit_Frame(self, qEl: np.ndarray, cumulativeLength: float) -> np.ndarray:
        """
        Generate coordinates in the non-cartesian global orbit frame that grows cumulatively with revolutions, from
        observer/lab cartesian coordinates.

        :param qLab: 3D cartesian position vector in observer/lab frame,numpy.array([x,y,z])
        :param cumulativeLength: total length in orbit frame traveled so far. For a series of linear elements this
        would simply be the sum of their length, float
        :return: New 3D global orbit frame position, numpy.ndarray([x,y,z])
        """

        qOrbit = self.transform_Element_Coords_Into_Local_Orbit_Frame(qEl)
        qOrbit[0] = qOrbit[0] + cumulativeLength  # longitudinal component grows
        return qOrbit

    def transform_Element_Momentum_Into_Global_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """wraps self.transform_Element_Momentum_Into_Local_Orbit_Frame"""

        return self.transform_Element_Momentum_Into_Local_Orbit_Frame(qEl, pEl)

    def transform_Lab_Coords_Into_Element_Frame(self, qLab: np.ndarray) -> np.ndarray:
        """
        Generate local cartesian element frame coordinates from cartesian observer/lab frame coordinates

        :param qLab: 3D cartesian position vector in observer/lab frame,numpy.array([x,y,z])
        :return: New 3D cartesian element frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_Element_Coords_Into_Lab_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """
        Generate cartesian observer/lab frame coordinates from local cartesian element frame coordinates

        :param qEl: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_Orbit_Frame_Into_Lab_Frame(self, qOrbit: np.ndarray) -> np.ndarray:
        """
        Generate global cartesian observer/lab frame coords from non-cartesian local orbit frame coords. Orbit coords
        are similiar to the Frenet-Serret Frame.

        :param qOrbit: 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        :return: New 3D cartesian observer/lab frame position, numpy.ndarray([x,y,z])
        """
        raise NotImplementedError

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """
        Generate non-cartesian local orbit frame coords from local cartesian element frame coords. Orbit coords are
        similiar to the Frenet-Serret Frame.

        :param qEl: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: New 3D non-cartesian orbit frame position, numpy.ndarray([so,xo,yo]). so is the distance along
            the orbit trajectory. xo is in the xy lab plane, yo is perpdindicular Not necessarily the same as the
            distance along the center of the element.
        """
        raise NotImplementedError

    def transform_Element_Momentum_Into_Local_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """
        Transform momentum vector in element frame in frame moving along with nominal orbit. In this frame px is the
        momentum tangent to the orbit, py is perpindicular and horizontal, pz is vertical.

        :param qEl: 3D Position vector in element frame
        :param pEl: 3D Momentum vector in element frame
        :return: New 3D momentum vector in orbit frame
        """
        raise NotImplementedError

    def transform_Lab_Frame_Vector_Into_Element_Frame(self, vecLab: np.ndarray) -> np.ndarray:
        """
        Generate element frame vector from observer/lab frame vector.

        :param vecLab: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz])
        :return: 3D cartesian vector in element frame,numpy.array([vx,vy,vz])
        """""
        vecNew = vecLab.copy()  # prevent editing
        vecNew[:2] = self.RIn @ vecNew[:2]
        return vecNew

    def transform_Element_Frame_Vector_Into_Lab_Frame(self, vecEl: np.ndarray) -> np.ndarray:
        """
        Generate observer/lab frame vector from element frame vector.

        :param vecEl: 3D cartesian vector in element frame,numpy.array([vx,vy,vz])
        :return: 3D cartesian vector in observer/lab frame,numpy.array([vx,vy,vz])
        """""
        vecNew = vecEl.copy()  # prevent editing
        vecNew[:2] = self.ROut @ vecNew[:2]
        return vecNew

    def is_Coord_Inside(self, qEl: np.ndarray) -> bool:
        """
        Check if a 3D cartesian element frame coordinate is contained within an element's vacuum tube

        :param qEl: 3D cartesian position vector in element frame,numpy.array([x,y,z])
        :return: True if the coordinate is inside, False if outside
        """
        return self.fastFieldHelper.numbaJitClass.is_Coord_Inside_Vacuum(
            *qEl)  # will raise NotImplementedError if called

    def fill_Pre_Constrained_Parameters(self):
        """Fill available geometric parameters before constrained lattice layout is solved. Fast field helper, shapely
        objects, and positions still need to be solved for/computed after this point. Most elements call compute all
        their internal parameters before the floorplan is solved, but lenses may have length unspecified and bending
        elements may have bending angle or number of magnets unspecified for example"""
        raise NotImplementedError

    def fill_Post_Constrained_Parameters(self):
        """Fill internal parameters after constrained lattice layout is solved. See fill_Pre_Constrainted_Parameters.
        At this point everything about the geometry of the element is specified"""

    def shape_Field_Data_3D(self, data: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Shape 3D field data for fast linear interpolation method

        Take an array with the shape (n,7) where n is the number of points in space. Each row
        must have the format [x,y,z,gradxB,gradyB,gradzB,B] where B is the magnetic field norm at x,y,z and grad is the
        partial derivative. The data must be from a 3D grid of points with no missing points or any other funny business
        and the order of points doesn't matter. Return arrays are raveled for use by fast interpolater

        :param data: (n,7) numpy array of points originating from a 3d grid
        :return: tuple of 7 arrays, first 3 are grid edge coords (x,y,z) and last 4 are flattened field values
        (Fx,Fy,Fz,V)
        """
        assert data.shape[1] == 7 and len(data) > 2 ** 3
        xArr = np.unique(data[:, 0])
        yArr = np.unique(data[:, 1])
        zArr = np.unique(data[:, 2])
        assert all(not np.any(np.isnan(arr)) for arr in (xArr, yArr, zArr))

        numx = xArr.shape[0]
        numy = yArr.shape[0]
        numz = zArr.shape[0]
        FxMatrix = np.empty((numx, numy, numz))
        FyMatrix = np.empty((numx, numy, numz))
        FzMatrix = np.empty((numx, numy, numz))
        VMatrix = np.zeros((numx, numy, numz))
        xIndices = np.argwhere(data[:, 0][:, None] == xArr)[:, 1]
        yIndices = np.argwhere(data[:, 1][:, None] == yArr)[:, 1]
        zIndices = np.argwhere(data[:, 2][:, None] == zArr)[:, 1]
        FxMatrix[xIndices, yIndices, zIndices] = -SIMULATION_MAGNETON * data[:, 3]
        FyMatrix[xIndices, yIndices, zIndices] = -SIMULATION_MAGNETON * data[:, 4]
        FzMatrix[xIndices, yIndices, zIndices] = -SIMULATION_MAGNETON * data[:, 5]
        VMatrix[xIndices, yIndices, zIndices] = SIMULATION_MAGNETON * data[:, 6]
        VFlat, FxFlat, FyFlat, FzFlat = VMatrix.ravel(), FxMatrix.ravel(), FyMatrix.ravel(), FzMatrix.ravel()
        return xArr, yArr, zArr, FxFlat, FyFlat, FzFlat, VFlat

    def shape_Field_Data_2D(self, data: np.ndarray) -> tuple[np.ndarray, ...]:
        """2D version of shape_Field_Data_3D. Data must be shape (n,5), with each row [x,y,Fx,Fy,V]"""
        assert data.shape[1] == 5 and len(data) > 2 ** 3
        xArr = np.unique(data[:, 0])
        yArr = np.unique(data[:, 1])
        numx = xArr.shape[0]
        numy = yArr.shape[0]
        BGradxMatrix = np.zeros((numx, numy))
        BGradyMatrix = np.zeros((numx, numy))
        B0Matrix = np.zeros((numx, numy))
        xIndices = np.argwhere(data[:, 0][:, None] == xArr)[:, 1]
        yIndices = np.argwhere(data[:, 1][:, None] == yArr)[:, 1]

        BGradxMatrix[xIndices, yIndices] = data[:, 2]
        BGradyMatrix[xIndices, yIndices] = data[:, 3]
        B0Matrix[xIndices, yIndices] = data[:, 4]
        FxMatrix = -SIMULATION_MAGNETON * BGradxMatrix
        FyMatrix = -SIMULATION_MAGNETON * BGradyMatrix
        VMatrix = SIMULATION_MAGNETON * B0Matrix
        VFlat, FxFlat, FyFlat = np.ravel(VMatrix), np.ravel(FxMatrix), np.ravel(FyMatrix)
        return xArr, yArr, FxFlat, FyFlat, VFlat

    def get_Valid_Jitter_Amplitude(self):
        """If jitter (radial misalignment) amplitude is too large, it is clipped."""
        return self.PTL.jitterAmp
