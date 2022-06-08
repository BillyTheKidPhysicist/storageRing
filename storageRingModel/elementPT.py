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
from constants import SIMULATION_MAGNETON, VACUUM_TUBE_THICKNESS, ELEMENT_PLOT_COLORS, MIN_MAGNET_MOUNT_THICKNESS
from helperTools import arr_Product, iscloseAll, make_Odd

# todo: this needs a good scrubbing and refactoring


realNumber = (int, float)
lst_tup_arr = Union[list, tuple, np.ndarray]

TINY_STEP = 1e-9
TINY_OFFSET = 1e-12  # tiny offset to avoid out of bounds right at edges of element
SMALL_OFFSET = 1e-9  # small offset to avoid out of bounds right at edges of element
MAGNET_ASPECT_RATIO = 4  # length of individual neodymium magnet relative to width of magnet


def full_Arctan(q):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(q[1], q[0])
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


def is_Even(x: int) -> bool:
    """Test if a number is even"""

    assert type(x) is int and x > 0
    return True if x % 2 == 0 else False


def mirror_Across_Angle(x: float, y: float, ang: float) -> tuple[float, float]:
    """mirror_Across_Angle x and y across a line at angle "ang" that passes through the origin"""
    m = np.tan(ang)
    d = (x + y * m) / (1 + m ** 2)
    xMirror = 2 * d - x
    yMirror = 2 * d * m - y
    return xMirror, yMirror


class ElementDimensionError(Exception):
    """Some dimension of an element is causing an unphysical configuration. Rather general error"""


class ElementTooShortError(Exception):
    """An element is too short. Because space is required for fringe fields this can result in negative material
    lengths, or nullify my approximation that fields drop to 1% when the element ends."""


class CombinerIterExceededError(Exception):
    """When solving for the geometry of the combiner, Newton's method is used to set the offset. Throw this if
    iterations are exceeded"""


class CombinerDimensionError(Exception):
    """Not all configurations of combiner parameters are valid. For one thing, the beam needs to fit into the
    combiner."""


class Element:
    """
    Base class for other elements. Contains universal attributes and methods.

    An element is the fundamental component of a neutral atom storage ring/injector. An arrangment of elements is called
    a lattice, as is done in accelerator physics. Elements are intended to be combined together such that particles can
    smoothly move from one to another, and many class variables serves this purpose. An element also contains methods
    for force vectors and magnetic potential at a point in space. It will also contain methods to generate fields values
    and construct itself, which is not always trivial.
    """

    def __init__(self, PTL, plotColor: str, ang: float = 0.0, L=None):
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
        self.r0: Optional[np.ndarray] = None  # coordinates of center of bender, minus any caps
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
        self.plotColor = plotColor
        self.L: Optional[float] = L
        self.Lm: Optional[float] = None  # hard edge length of magnet
        self.index: Optional[int] = None
        self.Lo: Optional[float] = None  # length of orbit for particle. For lenses and drifts this is the same as the
        # length. This is a nominal value because for segmented benders the path length is not simple to compute
        self.bumpVector = np.zeros(3)  # positoin vector of the bump displacement. zero vector for no bump amount
        self.outputOffset: float = 0.0  # some elements have an output offset, like from bender's centrifugal force or
        # #lens combiner
        self.fieldFact: float = 1.0  # factor to modify field values everywhere in space by, including force
        self.fastFieldHelper = None
        self.maxCombinerAng: float = .2  # because the field value is a right rectangular prism, it must extend to past the
        # #end of the tilted combiner. This maximum value is used to set that extra extent, and can't be exceede by ang
        self._fastFieldHelperInitParams = None  # variable to assist in workaround for jitclass pickling issue
        self._fastFieldHelperInternalParam = None
        self.shape: Optional[str] = None

    def init_fastFieldHelper(self, initParams):
        """Because jitclass cannot be pickled, I need to do some gymnastics. An Element object must be able to fully
        release references in __dict__ to the fastFieldHelper. Thus, each Element cannot carry around an uninitiliazed
        class of fastFieldHelper. Instead, it must be initialized from the global namespace."""
        if type(self) is Element:
            return fastNumbaMethodsAndClass.BaseClassFieldHelper_Numba(*initParams)
        elif type(self) is LensIdeal:
            return fastNumbaMethodsAndClass.IdealLensFieldHelper_Numba(*initParams)
        elif type(self) is Drift:
            return fastNumbaMethodsAndClass.DriftFieldHelper_Numba(*initParams)
        elif type(self) is CombinerIdeal:
            return fastNumbaMethodsAndClass.CombinerIdealFieldHelper_Numba(*initParams)
        elif type(self) is CombinerSim:
            return fastNumbaMethodsAndClass.CombinerSimFieldHelper_Numba(*initParams)
        elif type(self) is CombinerHalbachLensSim:
            return fastNumbaMethodsAndClass.CombinerHalbachLensSimFieldHelper_Numba(*initParams)
        elif type(self) is HalbachBenderSimSegmented:
            return fastNumbaMethodsAndClass.SegmentedBenderSimFieldHelper_Numba(*initParams)
        elif type(self) is BenderIdeal:
            return fastNumbaMethodsAndClass.BenderIdealFieldHelper_Numba(*initParams)
        elif type(self) is HalbachLensSim:
            return fastNumbaMethodsAndClass.LensHalbachFieldHelper_Numba(*initParams)
        else:
            raise NotImplementedError

    def __getstate__(self):
        self._fastFieldHelperInitParams = self.fastFieldHelper.get_Init_Params()
        self._fastFieldHelperInternalParam = self.fastFieldHelper.get_Internal_Params()
        paramDict = {}
        for key, val in self.__dict__.items():
            if key != 'fastFieldHelper':
                paramDict[key] = val
        return paramDict

    def __setstate__(self, d):
        self.__dict__ = d
        self.__dict__['fastFieldHelper'] = self.init_fastFieldHelper(self._fastFieldHelperInitParams)
        self.fastFieldHelper.set_Internal_Params(self._fastFieldHelperInternalParam)

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
        self.fastFieldHelper.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)

    def magnetic_Potential(self, qEl: np.ndarray) -> float:
        """
        Return magnetic potential energy at position qEl.

        Return magnetic potential energy of a lithium atom in simulation units, where the mass of a lithium-7 atom is
        1kg, at cartesian 3D coordinate qEl in the local element frame. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.

        :param qEl: 3D cartesian position vector in local element frame, numpy.array([x,y,z])
        :return: magnetic potential energy of a lithium atom in simulation units, float
        """
        return self.fastFieldHelper.magnetic_Potential(*qEl)  # will raise NotImplementedError if called

    def force(self, qEl: np.ndarray) -> np.ndarray:
        """
        Return force at position qEl.

        Return 3D cartesian force of a lithium at cartesian 3D coordinate qEl in the local element frame. Force vector
        has simulation units where lithium-7 mass is 1kg. This is done by calling up fastFieldHelper, a
        jitclass, which does the actual math/interpolation.


        :param qEl: 3D cartesian position vector in local element frame,numpy.array([x,y,z])
        :return: New 3D cartesian force vector, numpy.array([Fx,Fy,Fz])
        """
        return np.asarray(self.fastFieldHelper.force(*qEl))  # will raise NotImplementedError if called

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
        return self.fastFieldHelper.is_Coord_Inside_Vacuum(*qEl)  # will raise NotImplementedError if called

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


class LensIdeal(Element):
    """
    Ideal model of lens with hard edge. Force inside is calculated from field at pole face and bore radius as
    F=2*ub*r/rp**2 where rp is bore radius, and ub the simulation bohr magneton where the mass of lithium7=1kg.
    This will prevent energy conservation because of the absence of fringe fields between elements to reduce
    forward velocity. Interior vacuum tube is a cylinder
    """

    def __init__(self, PTL, L: float, Bp: float, rp: float, ap: float, build=True):
        """
        :param PTL: Instance of ParticleTracerLatticeClass
        :param L: Total length of element and lens, m. Not always the same because of need to contain fringe fields
        :param Bp: Magnetic field at the pole face, T.
        :param rp: Bore radius, m. Distance from center of magnet to the magnetic material
        :param ap: Aperture of bore, m. Typically is the radius of the vacuum tube
        """
        # fillParams is used to avoid filling the parameters in inherited classes
        super().__init__(PTL, ELEMENT_PLOT_COLORS['lens'], build=False, L=L)
        self.Bp = Bp
        self.rp = rp
        self.ap = rp if ap is None else ap  # size of apeture radially
        self.shape = 'STRAIGHT'  # The element's geometry
        self.K = None

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.K = self.fieldFact * (2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
        if self.L is not None:
            self.Lo = self.L
        self.fastFieldHelper = fastNumbaMethodsAndClass.IdealLensFieldHelper_Numba(self.L, self.K, self.ap)

    def transform_Lab_Coords_Into_Element_Frame(self, qLab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = qLab.copy()
        qNew -= self.r1
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_Element_Coords_Into_Lab_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = qEl.copy()
        qNew = self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew += self.r1
        return qNew

    def transform_Orbit_Frame_Into_Lab_Frame(self, qOrbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element. A simple translation and rotation completes the transformation"""
        qNew = qOrbit.copy()
        qNew[:2] = self.ROut @ qNew[:2]
        qNew += self.r1
        return qNew

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Element and orbit frame is identical in simple
        straight elements"""

        return qEl.copy()

    def set_Length(self, L: float) -> None:
        """this is used typically for setting the length after satisfying constraints"""

        assert L > 0.0
        self.L = L
        self.Lo = self.L

    def transform_Element_Momentum_Into_Local_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Element and orbit frame is identical in simple
        straight elements"""

        return pEl.copy()


class Drift(LensIdeal):
    """
    Simple model of free space. Effectively a cylinderical vacuum tube
    """

    def __init__(self, PTL, L: float, ap: float, outerHalfWidth: Optional[float],
                 inputTiltAngle: float, outputTiltAngle: float, build=True):
        super().__init__(PTL, L, 0, np.inf, ap, build=False)  # set Bp to zero and bore radius to infinite
        self.plotColor = ELEMENT_PLOT_COLORS['drift']
        self.inputTiltAngle, self.outputTiltAngle = inputTiltAngle, outputTiltAngle
        self.fastFieldHelper = self.init_fastFieldHelper([L, ap, inputTiltAngle, outputTiltAngle])
        self.outerHalfWidth = ap + VACUUM_TUBE_THICKNESS if outerHalfWidth is None else outerHalfWidth
        assert self.outerHalfWidth > ap

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.Lo = self.L


class BenderIdeal(Element):
    """
        Element representing a bender/waveguide. Base class for other benders

        Simple ideal model of bending/waveguide element modeled as a toroid. The force is linearly proportional to the
        particle's minor radius in the toroid. In this way, it works as a bent lens. Particles will travel through the
        bender displaced from the center of the  potential (minor radius=0.0) because of the centrifugal effect. Thus,
        to minimize oscillations, the nominal particle trajectory is not straight through the bender, but offset a small
        distance. For the ideal bender, this offset can be calculated analytically. Because there are no fringe fields,
        energy conservation is not expected

        Attributes
        ----------
        Bp: Magnetic field at poleface of bender bore, Teslas.

        rp: Radius (minor) to poleface of bender bore, meters.

        ap: Radius (minor) of aperture bender bore, meters. Effectively the vacuum tube inner radius

        rb: Nominal ending radius of bender/waveguide, meters. This is major radius of the toroid. Note that atoms will
            revolve at a slightly larger radius because of centrifugal effect

        shape: Gemeotric shape of element used for placement. ParticleTracerLatticeClass uses this to assemble lattice

        ro: Orbit bending radius, meter. Larger than self.rb because of centrifugal effect

        segmented: Wether the element is made up of discrete segments, or is continuous. Used in
            ParticleTracerLatticeClass
        """

    def __init__(self, PTL, ang: float, Bp: float, rp: float, rb: float, ap: float, build=True):
        super().__init__(PTL, ELEMENT_PLOT_COLORS['bender'], ang=ang, build=False)
        self.Bp: float = Bp
        self.rp: float = rp
        self.ap: float = self.rp if ap is None else ap
        self.rb: float = rb
        self.K: Optional[float] = None
        self.shape: str = 'BEND'
        self.ro: Optional[float] = None  # bending radius of orbit, ie rb + rOffset.
        self.segmented: bool = False  # wether the element is made up of discrete segments, or is continuous

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.K = (2 * self.Bp * SIMULATION_MAGNETON / self.rp ** 2)  # 'spring' constant
        self.outputOffset = sqrt(
            self.rb ** 2 / 4 + self.PTL.v0Nominal ** 2 / self.K) - self.rb / 2  # self.output_Offset(self.rb)
        self.ro = self.rb + self.outputOffset
        if self.ang is not None:  # calculation is being delayed until constraints are solved
            self.L = self.rb * self.ang
            self.Lo = self.ro * self.ang
        self.fastFieldHelper = self.init_fastFieldHelper([self.ang, self.K, self.rp, self.rb, self.ap])

    def fill_Post_Constrained_Parameters(self):
        self.fill_In_And_Out_Rotation_Matrices()

    def fill_In_And_Out_Rotation_Matrices(self):
        rot = self.theta - self.ang + np.pi / 2
        self.ROut = Rot.from_rotvec([0.0, 0.0, rot]).as_matrix()[:2, :2]
        self.RIn = Rot.from_rotvec([0.0, 0.0, -rot]).as_matrix()[:2, :2]

    def transform_Lab_Coords_Into_Element_Frame(self, qLab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element."""
        qNew = qLab - self.r0
        qNew = self.transform_Lab_Frame_Vector_Into_Element_Frame(qNew)
        return qNew

    def transform_Element_Coords_Into_Lab_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element."""
        qNew = qEl.copy()
        qNew = self.transform_Element_Frame_Vector_Into_Lab_Frame(qNew)
        qNew = qNew + self.r0
        return qNew

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element."""
        qo = qEl.copy()
        phi = self.ang - full_Arctan(qo)  # angle swept out by particle in trajectory. This is zero
        # when the particle first enters
        ds = self.ro * phi
        qos = ds
        qox = sqrt(qEl[0] ** 2 + qEl[1] ** 2) - self.ro
        qo[0] = qos
        qo[1] = qox
        return qo

    def transform_Orbit_Frame_Into_Lab_Frame(self, qOrbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element."""
        raise NotImplementedError  # there is an error here with yo
        xo, yo, zo = qOrbit
        phi = self.ang - xo / self.ro
        xLab = self.ro * np.cos(phi)
        yLab = self.ro * np.sin(phi)
        zLab = zo
        qLab = np.asarray([xLab, yLab, zLab])
        qLab[:2] = self.ROut @ qLab[:2]
        qLab += self.r0
        return qLab

    def transform_Element_Momentum_Into_Local_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Simple cartesian to cylinderical coordinates"""

        x, y = qEl[:2]
        xDot, yDot, zDot = pEl
        r = np.sqrt(x ** 2 + y ** 2)
        rDot = (x * xDot + y * yDot) / r
        thetaDot = (x * yDot - xDot * y) / r ** 2
        velocityTangent = -r * thetaDot
        return np.array([velocityTangent, rDot, zDot])  # tanget, perpindicular horizontal, perpindicular vertical


class CombinerIdeal(Element):
    # combiner: This is is the element that bends the two beams together. The logic is a bit tricky. It's geometry is
    # modeled as a straight section, a simple square, with a segment coming of at the particle in put at an angle. The
    # angle is decided by tracing particles through the combiner and finding the bending angle.

    def __init__(self, PTL, Lm: float, c1: float, c2: float, apL: float, apR: float, apZ: float, mode: str,
                 sizeScale: float):
        Element.__init__(self, PTL, ELEMENT_PLOT_COLORS['combiner'])
        assert mode in ('injector', 'storageRing')
        self.fieldFact = -1.0 if mode == 'injector' else 1.0
        self.sizeScale = sizeScale  # the fraction that the combiner is scaled up or down to. A combiner twice the size would
        # use sizeScale=2.0
        self.apR = apR
        self.apL = apL
        self.apz = apZ
        self.ap = None
        self.Lm = Lm
        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet
        self.c1 = c1
        self.c2 = c2
        self.space = 0  # space at the end of the combiner to account for fringe fields

        self.shape = 'COMBINER_SQUARE'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.apR, self.apL, self.apz, self.Lm = [val * self.sizeScale for val in
                                                 (self.apR, self.apL, self.apz, self.Lm)]
        self.c1, self.c2 = self.c1 / self.sizeScale, self.c2 / self.sizeScale
        self.Lb = self.Lm  # length of segment after kink after the inlet
        self.fastFieldHelper = self.init_fastFieldHelper([self.c1, self.c2, np.nan, self.Lb,
                                                          self.apL, self.apR, np.nan, np.nan])
        inputAngle, inputOffset, qTracedArr, _ = self.compute_Input_Angle_And_Offset()
        self.Lo = np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.ang = inputAngle
        self.inputOffset = inputOffset
        self.La = .5 * (self.apR + self.apL) * np.sin(self.ang)
        self.L = self.La * np.cos(
            self.ang) + self.Lb  # TODO: WHAT IS WITH THIS? TRY TO FIND WITH DEBUGGING. Is it used?
        self.fastFieldHelper = self.init_fastFieldHelper([self.c1, self.c2, self.La, self.Lb,
                                                          self.apL, self.apR, self.apz, self.ang])

    def compute_Input_Angle_And_Offset(self, outputOffset: float = 0.0, h: float = 1e-6,
                                       ap: Optional[float] = None) -> tuple:
        # this computes the output angle and offset for a combiner magnet.
        # NOTE: for the ideal combiner this gives slightly inaccurate results because of lack of conservation of energy!
        # NOTE: for the simulated bender, this also give slightly unrealisitc results because the potential is not allowed
        # to go to zero (finite field space) so the the particle will violate conservation of energy
        # limit: how far to carry the calculation for along the x axis. For the hard edge magnet it's just the hard edge
        # length, but for the simulated magnets, it's that plus twice the length at the ends.
        # h: timestep
        # lowField: wether to model low or high field seekers
        if type(self) == CombinerHalbachLensSim:
            assert 0.0 <= outputOffset < self.ap

        def force(x):
            if ap is not None and (x[0] < self.Lm + self.space and sqrt(x[1] ** 2 + x[2] ** 2) > ap):
                return np.empty(3) * np.nan
            Force = np.array(self.fastFieldHelper.force_Without_isInside_Check(x[0], x[1], x[2]))
            Force[2] = 0.0  ##only interested in xy plane bending
            return Force

        q = np.asarray([0.0, -outputOffset, 0.0])
        p = np.asarray([self.PTL.v0Nominal, 0.0, 0.0])
        qList = []

        xPosStopTracing = self.Lm + 2 * self.space
        forcePrev = force(q)  # recycling the previous force value cut simulation time in half
        while True:
            F = forcePrev
            q_n = q + p * h + .5 * F * h ** 2
            if not 0 <= q_n[0] <= xPosStopTracing:  # if overshot, go back and walk up to the edge assuming no force
                dr = xPosStopTracing - q[0]
                dt = dr / p[0]
                qFinal = q + p * dt
                pFinal = p
                qList.append(qFinal)
                break
            F_n = force(q_n)
            assert not np.any(np.isnan(F_n))
            p_n = p + .5 * (F + F_n) * h
            q, p = q_n, p_n
            forcePrev = F_n
            qList.append(q)
        assert qFinal[2] == 0.0  # only interested in xy plane bending, expected to be zero
        qArr = np.asarray(qList)
        outputAngle = np.arctan2(pFinal[1], pFinal[0])
        inputOffset = qFinal[1]
        if ap is not None:
            lensCorner = np.asarray([self.space + self.Lm, -ap, 0.0])
            minSepBottomRightMagEdge = np.min(np.linalg.norm(qArr - lensCorner, axis=1))
        else:
            minSepBottomRightMagEdge = None
        return outputAngle, inputOffset, qArr, minSepBottomRightMagEdge

    def compute_Trajectory_Length(self, qTracedArr: np.ndarray) -> float:
        # to find the trajectory length model the trajectory as a bunch of little deltas for each step and add up their
        # length
        x = qTracedArr[:, 0]
        y = qTracedArr[:, 1]
        xDelta = np.append(x[0], x[1:] - x[:-1])  # have to add the first value to the length of difference because
        # it starts at zero
        yDelta = np.append(y[0], y[1:] - y[:-1])
        dLArr = np.sqrt(xDelta ** 2 + yDelta ** 2)
        Lo = float(np.sum(dLArr))
        return Lo

    def transform_Lab_Coords_Into_Element_Frame(self, qLab: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qEl = self.transform_Lab_Frame_Vector_Into_Element_Frame(qLab - self.r2)  # a simple vector trick
        return qEl

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        # NOTE: THIS NOT GOING TO BE CORRECT IN GENERALY BECAUSE THE TRAJECTORY IS NOT SMOOTH AND I HAVE NOT WORKED IT OUT
        # YET
        qo = qEl.copy()
        qo[0] = self.Lo - qo[0]
        qo[1] = 0  # qo[1]
        return qo

    def transform_Element_Momentum_Into_Local_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Not supported at the moment, so returns np.nan array instead"""

        return np.array([np.nan, np.nan, np.nan])

    def transform_Element_Coords_Into_Lab_Frame(self, qEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qNew = qEl.copy()
        qNew[:2] = self.ROut @ qNew[:2] + self.r2[:2]
        return qNew

    def transform_Orbit_Frame_Into_Lab_Frame(self, qOrbit: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element"""
        qNew = qOrbit.copy()
        qNew[0] = -qNew[0]
        qNew[:2] = self.ROut @ qNew[:2]
        qNew += self.r1
        return qNew


class CombinerSim(CombinerIdeal):

    def __init__(self, PTL, combinerFileName: str, mode: str, sizeScale: float = 1.0, build: bool = True):
        # PTL: particle tracing lattice object
        # combinerFile: File with data with dimensions (n,6) where n is the number of points and each row is
        # (x,y,z,gradxB,gradyB,gradzB,B). Data must have come from a grid. Data must only be from the upper quarter
        # quadrant, ie the portion with z>0 and x< length/2
        # mode: wether the combiner is functioning as a loader, or a circulator.
        # sizescale: factor to scale up or down all dimensions. This modifies the field strength accordingly, ie
        # doubling dimensions halves the gradient
        assert mode in ('injector', 'storageRing')
        assert sizeScale > 0 and isinstance(combinerFileName, str)
        Lm = .187
        apL = .015
        apR = .025
        apZ = 6e-3
        super().__init__(PTL, Lm, np.nan, np.nan, apL, apR, apZ, mode, sizeScale, build=False)
        self.fringeSpace = 5 * 1.1e-2
        self.combinerFileName = combinerFileName

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        self.space = self.fringeSpace * self.sizeScale  # extra space past the hard edge on either end to account for fringe fields
        self.apL = self.apL * self.sizeScale
        self.apR = self.apR * self.sizeScale
        self.apz = self.apz * self.sizeScale
        data = np.asarray(pd.read_csv(self.combinerFileName, delim_whitespace=True, header=None))

        # use the new size scaling to adjust the provided data
        data[:, :3] = data[:, :3] * self.sizeScale  # scale the dimensions
        data[:, 3:6] = data[:, 3:6] / self.sizeScale  # scale the field gradient
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input
        fieldData = self.shape_Field_Data_3D(data)
        self.fastFieldHelper = self.init_fastFieldHelper([fieldData, np.nan, self.Lb, self.Lm,
                                                          self.space, self.apL, self.apR, self.apz, np.nan,
                                                          self.fieldFact])
        inputAngle, inputOffset, qTracedArr, _ = self.compute_Input_Angle_And_Offset()
        self.Lo = self.compute_Trajectory_Length(
            qTracedArr)
        self.L = self.Lo
        self.ang = inputAngle
        y0 = inputOffset
        x0 = self.space
        theta = inputAngle
        self.La = (y0 + x0 / np.tan(theta)) / (np.sin(theta) + np.cos(theta) ** 2 / np.sin(theta))

        self.inputOffset = inputOffset - np.tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge
        self.fastFieldHelper = self.init_fastFieldHelper([fieldData, self.La, self.Lb,
                                                          self.Lm, self.space, self.apL, self.apR, self.apz, self.ang,
                                                          self.fieldFact])
        self.update_Field_Fact(self.fieldFact)

    def update_Field_Fact(self, fieldStrengthFact) -> None:
        self.fastFieldHelper.fieldFact = fieldStrengthFact
        self.fieldFact = fieldStrengthFact


class HalbachBenderSimSegmented(BenderIdeal):
    # magnet
    # this element is a model of a bending magnet constructed of segments. There are three models from which data is
    # extracted required to construct the element. All exported data must be in a grid, though it the spacing along
    # each dimension may be different.
    # 1:  A model of the repeating segments of magnets that compose the bulk of the bender. A magnet, centered at the
    # bending radius, sandwiched by other magnets (at the appropriate angle) to generate the symmetry. The central magnet
    # is position with z=0, and field values are extracted from z=0-TINY_STEP to some value that extends slightly past
    # the tilted edge. See docs/images/HalbachBenderSimSegmentedImage1.png
    # 2: A model of the magnet between the last magnet, and the inner repeating section. This is required becasuse I found
    # that the assumption that I could jump straight from the outwards magnet to the unit cell portion was incorrect,
    # the force was very discontinuous. To model this I the last few segments of a bender, then extrac the field from
    # z=0 up to a little past halfway the second magnet. Make sure to have the x bounds extend a bit to capture
    # #everything. See docs/images/HalbachBenderSimSegmentedImage2.png
    # 3: a model of the input portion of the bender. This portions extends half a magnet length past z=0. Must include
    # enough extra space to account for fringe fields. See docs/images/HalbachBenderSimSegmentedImage3.png

    fringeFracOuter: float = 1.5  # multiple of bore radius to accomodate fringe field

    def __init__(self, PTL, Lm: float, rp: float, numMagnets: Optional[int], rb: float, ap: Optional[float],
                 extraSpace: float,
                 rOffsetFact: float, useStandardMagErrors: bool):
        assert all(val > 0 for val in (Lm, rp, rb, rOffsetFact))
        assert extraSpace >= 0
        assert rb > rp / 10  # this would be very dubious
        super().__init__(PTL, None, None, rp, rb, None, build=False)
        self.rb = rb
        self.space = extraSpace
        self.Lm = Lm
        self.rp = rp
        self.ap = ap
        self.Lseg: float = self.Lm + self.space * 2
        self.magnetWidth = rp * np.tan(2 * np.pi / 24) * 2
        self.yokeWidth = self.magnetWidth
        self.ucAng: Optional[float] = None
        self.rOffsetFact = rOffsetFact  # factor to times the theoretic optimal bending radius by
        self.Lcap = self.fringeFracOuter * self.rp
        self.numMagnets = numMagnets
        self.segmented: bool = True
        self.RIn_Ang: Optional[np.ndarray] = None
        self.M_uc: Optional[np.ndarray] = None
        self.M_ang: Optional[np.ndarray] = None
        self.numPointsBoreAp: int = make_Odd(
            round(25 * self.PTL.fieldDensityMultiplier))  # This many points should span the
        # bore ap for good field sampling
        self.longitudinalCoordSpacing: float = (
                                                       .8 * self.rp / 10.0) / self.PTL.fieldDensityMultiplier  # Spacing through unit
        # cell. .8 was carefully chosen
        self.numModelLenses: int = 7  # number of lenses in halbach model to represent repeating system. Testing has shown
        # this to be optimal
        self.cap: bool = True
        self.K: Optional[float] = None  # spring constant of field strength to set the offset of the lattice
        self.K_Func: Optional[
            callable] = None  # function that returns the spring constant as a function of bending radii. This is used in the
        # constraint solver
        self.useStandardMagErrors = useStandardMagErrors

    def compute_Maximum_Aperture(self) -> float:
        # beacuse the bender is segmented, the maximum vacuum tube allowed is not the bore of a single magnet
        # use simple geoemtry of the bending radius that touches the top inside corner of a segment
        radiusCorner = np.sqrt((self.rb - self.rp) ** 2 + (self.Lm / 2) ** 2)
        apMaxGeom = self.rb - radiusCorner - VACUUM_TUBE_THICKNESS  # max aperture without clipping magnet
        safetyFactor = .95
        apMaxGoodField = safetyFactor * self.numPointsBoreAp * self.rp / (
                self.numPointsBoreAp + np.sqrt(2))  # max aperture
        # without particles seeing field interpolation reaching into magnetic materal. Will not be exactly true for
        # several reasons (using int, and non equal grid in xy), so I include a small safety factor
        apMax = min([apMaxGeom, apMaxGoodField])
        return apMax

    def set_BpFact(self, BpFact: float):
        assert 0.0 <= BpFact
        self.fieldFact = BpFact

    def fill_Pre_Constrained_Parameters(self) -> None:
        self.outputOffset = self.find_Optimal_Radial_Offset() * self.rOffsetFact
        self.ro = self.outputOffset + self.rb

    def find_Optimal_Radial_Offset(self) -> float:
        """Find the radial offset that accounts for the centrifugal force moving the particles deeper into the
        potential well"""

        m = 1  # in simulation units mass is 1kg
        ucAngApprox = self.get_Unit_Cell_Angle()  # this will be different if the bore radius changes
        lens = _HalbachBenderFieldGenerator(self.rp, self.rb, ucAngApprox, self.Lm, numLenses=5,
                                            applyMethodOfMoments=True)
        thetaArr = np.linspace(-ucAngApprox, ucAngApprox, 100)
        yArr = np.zeros(len(thetaArr))

        def offset_Error(rOffset):
            assert abs(rOffset) < self.rp
            xArr = (self.rb + rOffset) * np.cos(thetaArr)
            zArr = (self.rb + rOffset) * np.sin(thetaArr)
            coords = np.column_stack((xArr, yArr, zArr))
            F = lens.BNorm_Gradient(coords) * SIMULATION_MAGNETON
            Fr = np.linalg.norm(F[:, [0, 2]], axis=1)
            FrMean = np.mean(Fr)
            FCen = m * self.PTL.v0Nominal ** 2 / (self.rb + rOffset)
            return (FCen - FrMean) ** 2

        rOffsetMax = .9 * self.rp
        bounds = [(0.0, rOffsetMax)]
        sol = spo.minimize(offset_Error, np.array([self.rp / 2.0]), bounds=bounds, method='Nelder-Mead',
                           options={'xatol': 1e-6})
        rOffsetOptimal = sol.x[0]
        if isclose(rOffsetOptimal, rOffsetMax, abs_tol=1e-6):
            raise Exception("The bending bore radius is too large to accomodate a reasonable solution")
        return rOffsetOptimal

    def get_Unit_Cell_Angle(self) -> float:
        """Get the angle that a single unit cell spans. Each magnet is composed of two unit cells because of symmetry.
        The unit cell includes half of the magnet and half the gap between the two"""

        ucAng = np.arctan(.5 * self.Lseg / (self.rb - self.rp - self.yokeWidth))
        return ucAng

    def fill_Post_Constrained_Parameters(self) -> None:

        self.ap = self.ap if self.ap is not None else self.compute_Maximum_Aperture()
        assert self.ap <= self.compute_Maximum_Aperture()
        assert self.rb - self.rp - self.yokeWidth > 0.0
        self.ucAng = self.get_Unit_Cell_Angle()
        # 500um works very well, but 1mm may be acceptable
        self.ang = 2 * self.numMagnets * self.ucAng
        self.fill_In_And_Out_Rotation_Matrices()
        assert self.ang < 2 * np.pi * 3 / 4
        self.RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])
        m = np.tan(self.ucAng)
        self.M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        m = np.tan(self.ang / 2)
        self.M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        self.ro = self.rb + self.outputOffset
        self.L = self.ang * self.rb
        self.Lo = self.ang * self.ro + 2 * self.Lcap
        self.outerHalfWidth = self.rp + self.magnetWidth + MIN_MAGNET_MOUNT_THICKNESS

    def build_Fast_Field_Helper(self, extraFieldSources) -> None:
        """compute field values and build fast numba helper"""
        fieldDataSeg = self.generate_Segment_Field_Data()
        fieldDataInternal = self.generate_Internal_Fringe_Field_Data()
        fieldDataCap = self.generate_Cap_Field_Data()
        fieldDataPerturbation = self.generate_Perturbation_Data() if self.useStandardMagErrors else None
        assert np.all(fieldDataCap[0] == fieldDataInternal[0]) and np.all(fieldDataCap[1] == fieldDataInternal[1])
        self.fastFieldHelper = self.init_fastFieldHelper(
            [fieldDataSeg, fieldDataInternal, fieldDataCap, fieldDataPerturbation
                , self.ap, self.ang,
             self.ucAng, self.rb, self.numMagnets, self.Lcap, self.M_uc, self.M_ang, self.RIn_Ang])
        self.fastFieldHelper.force(self.rb + 1e-3, 1e-3, 1e-3)  # force numba to compile
        self.fastFieldHelper.magnetic_Potential(self.rb + 1e-3, 1e-3, 1e-3)  # force numba to compile

    def make_Grid_Coords(self, xMin: float, xMax: float, zMin: float, zMax: float) -> np.ndarray:
        """Make Array of points that the field will be evaluted at for fast interpolation. only x and s values change.
        """
        assert not is_Even(self.numPointsBoreAp)  # points should be odd to there is a point at zero field, if possible
        numPointsX = make_Odd(round(self.numPointsBoreAp * (xMax - xMin) / self.ap))
        yMin, yMax = -(self.ap + TINY_STEP), TINY_STEP  # same for every part of bender
        numPointsY = self.numPointsBoreAp
        numPointsZ = make_Odd(round((zMax - zMin) / self.longitudinalCoordSpacing))
        assert (numPointsX + 1) / numPointsY >= (xMax - xMin) / (yMax - yMin)  # should be at least this ratio
        xArrArgs, yArrArgs, zArrArgs = (xMin, xMax, numPointsX), (yMin, yMax, numPointsY), (zMin, zMax, numPointsZ)
        coordArrList = [np.linspace(arrArgs[0], arrArgs[1], arrArgs[2]) for arrArgs in (xArrArgs, yArrArgs, zArrArgs)]
        gridCoords = np.asarray(np.meshgrid(*coordArrList)).T.reshape(-1, 3)
        return gridCoords

    def convert_Center_To_Cartesian_Coords(self, s: float, xc: float, yc: float) -> tuple[float, float, float]:
        """Convert center coordinates [s,xc,yc] to cartesian coordinates[x,y,z]"""

        if -TINY_OFFSET <= s < self.Lcap:
            x, y, z = self.rb + xc, yc, s - self.Lcap
        elif self.Lcap <= s < self.Lcap + self.ang * self.rb:
            theta = (s - self.Lcap) / self.rb
            r = self.rb + xc
            x, y, z = np.cos(theta) * r, yc, np.sin(theta) * r
        elif self.Lcap + self.ang * self.rb <= s <= self.ang * self.rb + 2 * self.Lcap + TINY_OFFSET:
            theta = self.ang
            r = self.rb + xc
            x0, z0 = np.cos(theta) * r, np.sin(theta) * r
            deltaS = s - (self.ang * self.rb + self.Lcap)
            thetaPerp = np.pi + np.arctan(-1 / np.tan(theta))
            x, y, z = x0 + np.cos(thetaPerp) * deltaS, yc, z0 + np.sin(thetaPerp) * deltaS
        else:
            raise ValueError
        return x, y, z

    def make_Perturbation_Data_Coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Make coordinates for computing and interpolation perturbation data. The perturbation field exists in an
        evenly spaced grid in "center" coordinates [s,xc,yc] where s is distance along bender through center, xc is
        radial distance from center with positive meaning along larger radius and 0 meaning right  at the center,pu
        and yc is distance along z axis. HalbachLensClass.SegmentedBenderHalbach is in (x,z) plane with z=0 at start
        and going clockwise in +y. This needs to be converted to cartesian coordinates to actually evaluate the field
        value"""

        Ls = 2 * self.Lcap + self.ang * self.rb
        numS = make_Odd(round(5 * (self.numMagnets + 2)))  # carefully measured
        numYc = make_Odd(round(35 * self.PTL.fieldDensityMultiplier))
        numXc = numYc

        sArr = np.linspace(-TINY_OFFSET, Ls + TINY_OFFSET, numS)  # distance through bender along center
        xcArr = np.linspace(-self.ap - TINY_OFFSET, self.ap + TINY_OFFSET, numXc)  # radial deviation along major radius
        ycArr = np.linspace(-self.ap - TINY_OFFSET, self.ap + TINY_OFFSET,
                            numYc)  # deviation in vertical from center of
        # bender, along y in cartesian
        assert not is_Even(len(sArr)) and not is_Even(len(xcArr)) and not is_Even(len(ycArr))
        coordsCenter = arr_Product(sArr, xcArr, ycArr)
        coords = np.asarray([self.convert_Center_To_Cartesian_Coords(*coordCenter) for coordCenter in coordsCenter])
        return coordsCenter, coords

    def generate_Perturbation_Data(self) -> tuple[np.ndarray, ...]:
        coordsCenter, coordsCartesian = self.make_Perturbation_Data_Coords()
        lensMisaligned = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                      numLenses=self.numMagnets, positiveAngleMagnetsOnly=True,
                                                      useMagnetError=True, useHalfCapEnd=(True, True),
                                                      applyMethodOfMoments=False)
        lensAligned = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                   numLenses=self.numMagnets, positiveAngleMagnetsOnly=True,
                                                   useMagnetError=False, useHalfCapEnd=(True, True),
                                                   applyMethodOfMoments=False)
        rCenterArr = np.linalg.norm(coordsCenter[:, 1:], axis=1)
        validIndices = rCenterArr < self.rp
        valsMisaligned = np.column_stack(self.compute_Valid_Field_Vals(lensMisaligned, coordsCartesian, validIndices))
        valsAligned = np.column_stack(self.compute_Valid_Field_Vals(lensAligned, coordsCartesian, validIndices))
        valsPerturbation = valsMisaligned - valsAligned
        valsPerturbation[np.isnan(valsPerturbation)] = 0.0
        interpData = np.column_stack((coordsCenter, valsPerturbation))
        interpData = self.shape_Field_Data_3D(interpData)

        return interpData

    def generate_Cap_Field_Data(self) -> tuple[np.ndarray, ...]:
        # x and y bounds should match with internal fringe bounds
        xMin = (self.rb - self.ap) * np.cos(2 * self.ucAng) - TINY_STEP
        xMax = self.rb + self.ap + TINY_STEP
        zMin = -self.Lcap - TINY_STEP
        zMax = TINY_STEP
        fieldCoords = self.make_Grid_Coords(xMin, xMax, zMin, zMax)
        validIndices = np.sqrt((fieldCoords[:, 0] - self.rb) ** 2 + fieldCoords[:, 1] ** 2) < self.rp
        lens = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                            numLenses=self.numModelLenses, positiveAngleMagnetsOnly=True,
                                            applyMethodOfMoments=True,
                                            useHalfCapEnd=(True, False))
        return self.compute_Valid_Field_Data(lens, fieldCoords, validIndices)

    def is_Valid_Internal_Fringe(self, coord0: np.ndarray) -> bool:
        """Return True if coord does NOT enter magnetic material, else False"""
        xzAngle = np.arctan2(coord0[2], coord0[0])
        coord = coord0.copy()
        assert -2 * TINY_STEP / self.rb <= xzAngle < 3 * self.ucAng
        if self.ucAng < xzAngle <= 3 * self.ucAng:
            rotAngle = 2 * self.ucAng if xzAngle <= 2 * self.ucAng else 3 * self.ucAng
            coord = Rot.from_rotvec(np.asarray([0.0, rotAngle, 0.0])).as_matrix() @ coord
        return np.sqrt((coord[0] - self.rb) ** 2 + coord[1] ** 2) < self.rp

    def generate_Internal_Fringe_Field_Data(self) -> tuple[np.ndarray, ...]:
        """An magnet slices are required to model the region going from the cap to the repeating unit cell,otherwise
        there is too large of an energy discontinuity"""
        # x and y bounds should match with cap bounds
        xMin = (self.rb - self.ap) * np.cos(2 * self.ucAng) - TINY_STEP  # inward enough to account for the tilt
        xMax = self.rb + self.ap + TINY_STEP
        zMin = -TINY_STEP
        zMax = np.tan(2 * self.ucAng) * (self.rb + self.ap) + TINY_STEP
        fieldCoords = self.make_Grid_Coords(xMin, xMax, zMin, zMax)
        validIndices = [self.is_Valid_Internal_Fringe(coord) for coord in fieldCoords]
        lens = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                            numLenses=self.numModelLenses, positiveAngleMagnetsOnly=True,
                                            applyMethodOfMoments=True,
                                            useHalfCapEnd=(True, False))
        return self.compute_Valid_Field_Data(lens, fieldCoords, validIndices)

    def generate_Segment_Field_Data(self) -> tuple[np.ndarray, ...]:
        """Internal repeating unit cell segment. This is modeled as a tilted portion with angle self.ucAng to the
        z axis, with its bottom face at z=0 alinged with the xy plane"""
        xMin = (self.rb - self.ap) * np.cos(self.ucAng) - TINY_STEP
        xMax = self.rb + self.ap + TINY_STEP
        zMin = -TINY_STEP
        zMax = np.tan(self.ucAng) * (self.rb + self.ap) + TINY_STEP
        fieldCoords = self.make_Grid_Coords(xMin, xMax, zMin, zMax)
        validIndices = np.sqrt((fieldCoords[:, 0] - self.rb) ** 2 + fieldCoords[:, 1] ** 2) < self.rp
        assert not is_Even(self.numModelLenses)  # must be odd so magnet is centered at z=0
        lens = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                            numLenses=self.numModelLenses, applyMethodOfMoments=True,
                                            positiveAngleMagnetsOnly=False)
        return self.compute_Valid_Field_Data(lens, fieldCoords, validIndices)

    def compute_Valid_Field_Vals(self, lens: _HalbachBenderFieldGenerator, fieldCoords: np.ndarray,
                                 validIndices: lst_tup_arr) -> tuple[np.ndarray, np.ndarray]:
        BNormGradArr, BNormArr = np.zeros((len(fieldCoords), 3)) * np.nan, np.zeros(len(fieldCoords)) * np.nan
        BNormGradArr[validIndices], BNormArr[validIndices] = lens.BNorm_Gradient(fieldCoords[validIndices],
                                                                                 returnNorm=True, useApprox=True)
        return BNormGradArr, BNormArr

    def compute_Valid_Field_Data(self, lens: _HalbachBenderFieldGenerator, fieldCoords: np.ndarray,
                                 validIndices: lst_tup_arr) -> tuple[np.ndarray, ...]:
        BNormGradArr, BNormArr = self.compute_Valid_Field_Vals(lens, fieldCoords, validIndices)
        fieldDataUnshaped = np.column_stack((fieldCoords, BNormGradArr, BNormArr))
        return self.shape_Field_Data_3D(fieldDataUnshaped)

    def in_Which_Section_Of_Bender(self, qEl: np.ndarray) -> str:
        """Find which section of the bender qEl is in. options are:
            - 'IN' refers to the westward cap. at some angle
            - 'OUT' refers to the eastern. input is aligned with y=0
            - 'ARC' in the bending arc between input and output caps
        Return 'NONE' if not inside the bender"""

        angle = full_Arctan(qEl)
        if 0 <= angle <= self.ang:
            return 'ARC'
        capNames = ['IN', 'OUT']
        for name in capNames:
            xCap, yCap = mirror_Across_Angle(qEl[0], qEl[1], self.ang / 2.0) if name == 'IN' else qEl[:2]
            if (self.rb - self.ap < xCap < self.rb + self.ap) and (0 > yCap > -self.Lcap):
                return name
        return 'NONE'

    def transform_Element_Coords_Into_Local_Orbit_Frame(self, qEl: np.ndarray) -> np.ndarray:

        whichSection = self.in_Which_Section_Of_Bender(qEl)
        if whichSection == 'ARC':
            phi = self.ang - full_Arctan(qEl)
            xo = sqrt(qEl[0] ** 2 + qEl[1] ** 2) - self.ro
            so = self.ro * phi + self.Lcap  # include the distance traveled throught the end cap
        elif whichSection == 'OUT':
            so = self.Lcap + self.ang * self.ro + (-qEl[1])
            xo = qEl[0] - self.ro
        elif whichSection == 'IN':
            xMirror, yMirror = mirror_Across_Angle(qEl[0], qEl[1], self.ang / 2.0)
            so = self.Lcap + yMirror
            xo = xMirror - self.ro
        else:
            raise ValueError
        qo = np.array([so, xo, qEl[2]])
        return qo

    def transform_Element_Momentum_Into_Local_Orbit_Frame(self, qEl: np.ndarray, pEl: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Mildly tricky. Need to determine if the position is in
        one of the caps or the bending segment, then handle accordingly"""

        whichSection = self.in_Which_Section_Of_Bender(qEl)
        if whichSection == 'ARC':
            return super().transform_Element_Momentum_Into_Local_Orbit_Frame(qEl, pEl)
        elif whichSection == 'OUT':
            pso, pxo = -pEl[1], pEl[0]
        elif whichSection == 'IN':
            pxo, pso = mirror_Across_Angle(pEl[0], pEl[1], self.ang / 2.0)
        else:
            raise ValueError
        pOrbit = np.array([pso, pxo, qEl[-2]])
        return pOrbit

    def _get_Shapely_Object_Of_Bore(self):
        """Shapely object of bore in x,z plane with y=0. Not of vacuum tube, but of largest possible bore. For two
        unit cells."""
        bore = Polygon([(self.rb + self.rp, 0.0), (self.rb + self.rp, (self.rb + self.rp) * np.tan(self.ucAng)),
                        ((self.rb + self.rp) * np.cos(self.ucAng * 2),
                         (self.rb + self.rp) * np.sin(self.ucAng * 2)),
                        ((self.rb - self.rp) * np.cos(self.ucAng * 2), (self.rb - self.rp) * np.sin(self.ucAng * 2))
                           , (self.rb - self.rp, (self.rb - self.rp) * np.tan(self.ucAng)), (self.rb - self.rp, 0.0)])
        return bore


class HalbachLensSim(LensIdeal):
    fringeFracOuter: float = 1.5

    def __init__(self, PTL, rpLayers: tuple, L: Optional[float], ap: Optional[float],
                 magnetWidths: Optional[tuple], useStandardMagErrors: bool):
        assert all(rp > 0 for rp in rpLayers)
        # if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        # to accomdate the new rp such as force values and positions
        self.magnetWidths = self.set_Magnet_Widths(rpLayers, magnetWidths)
        self.fringeFracInnerMin = 4.0  # if the total hard edge magnet length is longer than this value * rp, then it can
        # can safely be modeled as a magnet "cap" with a 2D model of the interior
        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        assert self.fringeFracOuter == 1.5 and self.fringeFracInnerMin == 4.0, "May need to change numgrid points if " \
                                                                               "this changes"
        rp = min(rpLayers)
        self.numGridPointsZ = make_Odd(round(21 * PTL.fieldDensityMultiplier))
        self.numGridPointsXY = make_Odd(round(25 * PTL.fieldDensityMultiplier))
        self.apMaxGoodField = self.calculate_Maximum_Good_Field_Aperture(rp)
        ap = self.apMaxGoodField - TINY_OFFSET if ap is None else ap
        self.fringeFieldLength = max(rpLayers) * self.fringeFracOuter
        assert ap <= self.apMaxGoodField
        assert ap > 5 * rp / self.numGridPointsXY  # ap shouldn't be too small. Value below may be dubiuos from interpolation
        super().__init__(PTL, L, None, rp, ap, build=False)
        self.L = L
        self.methodOfMomentsHighPrecision = False
        self.useStandardMagErrors = useStandardMagErrors
        self.Lo = None
        self.rpLayers = rpLayers  # can be multiple bore radius for different layers

        self.effectiveLength: Optional[float] = None  # if the magnet is very long, to save simulation
        # time use a smaller length that still captures the physics, and then model the inner portion as 2D
        self.Lcap: Optional[float] = None
        self.extraFieldLength: Optional[float] = None  # extra field added to end of lens to account misalignment
        self.minLengthLongSymmetry = self.fringeFracInnerMin * max(self.rpLayers)
        self.fieldFact = 1.0  # factor to multiply field values by for tunability
        self.individualMagnetLength: float = None
        # or down

    def calculate_Maximum_Good_Field_Aperture(self, rp: float) -> float:
        """ from geometric arguments of grid inside circle.
        imagine two concentric rings on a grid, such that no grid box which has a portion outside the outer ring
        has any portion inside the inner ring. This is to prevent interpolation reaching into magnetic material"""
        apMax = (rp - SMALL_OFFSET) * (1 - np.sqrt(2) / (self.numGridPointsXY - 1))
        return apMax

    def fill_Pre_Constrained_Parameters(self):
        pass

    def fill_Post_Constrained_Parameters(self):
        self.set_extraFieldLength()
        self.fill_Geometric_Params()

    def set_Length(self, L: float) -> None:
        assert L > 0.0
        self.L = L

    def set_extraFieldLength(self) -> None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped"""

        jitterAmp = self.get_Valid_Jitter_Amplitude(Print=True)
        tiltMax = np.arctan(jitterAmp / self.L)
        assert 0.0 <= tiltMax < .1  # small angle. Not sure if valid outside that range
        self.extraFieldLength = self.rp * tiltMax * 1.5  # safety factor for approximations

    def set_Effective_Length(self):
        """If a lens is very long, then longitudinal symmetry can possibly be exploited because the interior region
        is effectively isotropic a sufficient depth inside. This is then modeled as a 2d slice, and the outer edges
        as 3D slice"""

        self.effectiveLength = self.minLengthLongSymmetry if self.minLengthLongSymmetry < self.Lm else self.Lm

    def set_Magnet_Widths(self, rpLayers: tuple[float, ...], magnetWidthsProposed: Optional[tuple[float, ...]]) \
            -> tuple[float, ...]:
        """
        Return transverse width(w in L x w x w) of individual neodymium permanent magnets used in each layer to
        build lens. Check that sizes are valid

        :param rpLayers: tuple of bore radius of each concentric layer
        :param magnetWidthsProposed: tuple of magnet widths in each concentric layer, or None, in which case the maximum value
            will be calculated based on geometry
        :return: tuple of transverse widths of magnets
        """

        maximumMagnetWidth = tuple(rp * np.tan(2 * np.pi / 24) * 2 for rp in rpLayers)
        magnetWidths = maximumMagnetWidth if magnetWidthsProposed is None else magnetWidthsProposed
        assert len(magnetWidths) == len(rpLayers)
        assert all(width <= maxWidth for width, maxWidth in zip(magnetWidths, maximumMagnetWidth))
        if len(rpLayers) > 1:
            for indexPrev, rp in enumerate(rpLayers[1:]):
                assert rp >= rpLayers[indexPrev] + magnetWidths[indexPrev] - 1e-12
        return magnetWidths

    def fill_Geometric_Params(self) -> None:
        """Compute dependent geometric values"""

        self.Lm = self.L - 2 * self.fringeFracOuter * max(self.rpLayers)  # hard edge length of magnet
        if self.Lm < .5 * self.rp:  # If less than zero, unphysical. If less than .5rp, this can screw up my assumption
            # about fringe fields
            raise ElementTooShortError
        self.individualMagnetLength = min(
            [(MAGNET_ASPECT_RATIO * min(self.magnetWidths)), self.Lm])  # this may get rounded
        # up later to satisfy that the total length is Lm
        self.Lo = self.L
        self.set_Effective_Length()
        self.Lcap = self.effectiveLength / 2 + self.fringeFracOuter * max(self.rpLayers)
        mountThickness = 1e-3  # outer thickness of mount, likely from space required by epoxy and maybe clamp
        self.outerHalfWidth = max(self.rpLayers) + self.magnetWidths[np.argmax(self.rpLayers)] + mountThickness

    def make_Grid_Coord_Arrays(self, useSymmetry: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        """

        yMin, yMax = -TINY_OFFSET, self.rp - TINY_OFFSET
        zMin, zMax = -TINY_OFFSET, self.Lcap + TINY_OFFSET + self.extraFieldLength
        numPointsXY, numPointsZ = self.numGridPointsXY, self.numGridPointsZ
        if not useSymmetry:  # range will have to fully capture lens.
            numSlices = self.get_Num_Lens_Slices()
            yMin = -yMax
            zMax = self.L / 2 + self.extraFieldLength + TINY_OFFSET
            zMin = -zMax
            assert self.fringeFracOuter == 1.5  # pointsperslice mildly depends on this value
            pointsPerSlice = 5
            numPointsZ = make_Odd(round(max([pointsPerSlice * (numSlices + 2), 2 * numPointsZ - 1])))
            assert numPointsZ < 150  # things might start taking unreasonably long if not careful
            numPointsXY = 45
        assert not is_Even(numPointsXY) and not is_Even(numPointsZ)
        yArr_Quadrant = np.linspace(yMin, yMax, numPointsXY)
        xArr_Quadrant = -yArr_Quadrant.copy()
        zArr = np.linspace(zMin, zMax, numPointsZ)
        return xArr_Quadrant, yArr_Quadrant, zArr

    def make_2D_Field_Data(self, fieldGenerator: billyHalbachCollectionWrapper, xArr: np.ndarray,
                           yArr: np.ndarray) -> Optional[np.ndarray]:
        """
        Make 2d field data for interpolation.

        This comes from the center of the lens, and models a continuous segment homogenous in x (element frame)
        that allows for constructing very long lenses. If lens is too short, return None

        :param fieldGenerator: magnet object to compute fields values
        :param xArr: Grid edge x values of quarter of plane
        :param yArr: Grid edge y values of quarter of plane
        :return: Either 2d array of field data, or None
        """
        if self.Lm < self.minLengthLongSymmetry:
            data2D = None
        else:
            # ignore fringe fields for interior  portion inside then use a 2D plane to represent the inner portion to
            # save resources
            planeCoords = np.asarray(np.meshgrid(xArr, yArr, 0)).T.reshape(-1, 3)
            validIndices = np.linalg.norm(planeCoords, axis=1) <= self.rp
            BNormGrad, BNorm = np.zeros((len(validIndices), 3)) * np.nan, np.ones(len(validIndices)) * np.nan
            BNormGrad[validIndices], BNorm[validIndices] = fieldGenerator.BNorm_Gradient(planeCoords[validIndices],
                                                                                         returnNorm=True)
            data2D = np.column_stack((planeCoords[:, :2], BNormGrad[:, :2], BNorm))  # 2D is formated as
            # [[x,y,z,B0Gx,B0Gy,B0],..]
        return data2D

    def make_3D_Field_Data(self, fieldGenerator: billyHalbachCollectionWrapper, xArr: np.ndarray, yArr: np.ndarray,
                           zArr: np.ndarray) -> np.ndarray:
        """
        Make 3d field data for interpolation from end of lens region

        If the lens is sufficiently long compared to bore radius then this is only field data from the end region
        (fringe frields and interior near end) because the interior region is modeled as a single plane to exploit
        longitudinal symmetry. Otherwise, it is exactly half of the lens and fringe fields

        :param fieldGenerator: magnet objects to compute fields values
        :param xArr: Grid edge x values of quarter of plane if using symmetry, else full
        :param yArr: Grid edge y values of quarter of plane if using symmetry, else full
        :param zArr: Grid edge z values of half of lens, or region near end if long enough, if using symmetry. Else
            full length
        :return: 2D array of field data
        """
        volumeCoords = np.asarray(np.meshgrid(xArr, yArr, zArr)).T.reshape(-1,
                                                                           3)  # note that these coordinates can have
        # the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and
        # input coordinates will be shifted in a wrapper function
        validXY = np.linalg.norm(volumeCoords[:, :2], axis=1) <= self.rp
        validZ = volumeCoords[:, 2] >= self.Lm / 2
        validIndices = np.logical_or(validXY, validZ)
        BNormGrad, BNorm = np.zeros((len(validIndices), 3)) * np.nan, np.ones(len(validIndices)) * np.nan
        BNormGrad[validIndices], BNorm[validIndices] = fieldGenerator.BNorm_Gradient(volumeCoords[validIndices],
                                                                                     returnNorm=True)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        return data3D

    def get_Num_Lens_Slices(self) -> int:
        numSlices = round(self.Lm / self.individualMagnetLength)
        assert numSlices > 0
        return numSlices

    def make_Field_Data(self, useSymmetry: bool, useStandardMagnetErrors: bool, extraFieldSources,
                        enforceGoodField: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Make 2D and 3D field data. 2D may be None if lens is to short for symmetry."""
        lensLength = self.effectiveLength if useSymmetry else self.Lm
        numSlices = None if not useStandardMagnetErrors else self.get_Num_Lens_Slices()
        lens = _HalbachLensFieldGenerator(self.rpLayers, self.magnetWidths, lensLength,
                                          applyMethodOfMoments=True, useStandardMagErrors=useStandardMagnetErrors,
                                          numSlices=numSlices, useSolenoidField=self.PTL.useSolenoidField)
        sources = [src.copy() for src in [*lens.sources_all, *extraFieldSources]]
        fieldGenerator = billyHalbachCollectionWrapper(sources)
        xArr_Quadrant, yArr_Quadrant, zArr = self.make_Grid_Coord_Arrays(useSymmetry)
        maxGridSep = np.sqrt((xArr_Quadrant[1] - xArr_Quadrant[0]) ** 2 + (xArr_Quadrant[1] - xArr_Quadrant[0]) ** 2)
        if enforceGoodField == True:
            assert self.rp - maxGridSep >= self.apMaxGoodField
        data2D = self.make_2D_Field_Data(fieldGenerator, xArr_Quadrant, yArr_Quadrant) if useSymmetry else None
        data3D = self.make_3D_Field_Data(fieldGenerator, xArr_Quadrant, yArr_Quadrant, zArr)
        return data2D, data3D

    def build_Fast_Field_Helper(self, extraFieldSources) -> None:
        """Generate magnetic field gradients and norms for numba jitclass field helper. Low density sampled imperfect
        data may added on top of high density symmetry exploiting perfect data. """

        data2D, data3D = self.make_Field_Data(True, False, extraFieldSources)
        xArrEnd, yArrEnd, zArrEnd, FxArrEnd, FyArrEnd, FzArrEnd, VArrEnd = self.shape_Field_Data_3D(data3D)
        if data2D is not None:  # if no inner plane being used
            xArrIn, yArrIn, FxArrIn, FyArrIn, VArrIn = self.shape_Field_Data_2D(data2D)
        else:
            xArrIn, yArrIn, FxArrIn, FyArrIn, VArrIn = [np.ones(1) * np.nan] * 5
        fieldData = (
            xArrEnd, yArrEnd, zArrEnd, FxArrEnd, FyArrEnd, FzArrEnd, VArrEnd, xArrIn, yArrIn, FxArrIn, FyArrIn, VArrIn)
        fieldDataPerturbations = self.make_Field_Perturbation_Data(extraFieldSources)
        self.fastFieldHelper = self.init_fastFieldHelper([fieldData, fieldDataPerturbations, self.L, self.Lcap, self.ap,
                                                          self.extraFieldLength])
        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lcap, self.ap / 2, .0])))
        assert F_edge / F_center < .01

    def make_Field_Perturbation_Data(self, extraFieldSources) -> Optional[tuple]:
        """Make data for fields coming from magnet imperfections and misalingnmet. Imperfect field values are calculated
        and perfect fiel values are subtracted. The difference is then added later on top of perfect field values. This
        force is small, and so I can get away with interpolating with low density, while keeping my high density
        symmetry region. interpolation points inside magnet material are set to zero, so the interpolation may be poor
        near bore of magnet. This is done to avoid dealing with mistmatch  between good field region of ideal and
        perturbation interpolation"""

        if self.useStandardMagErrors:
            data2D_1, data3D_NoPerturbations = self.make_Field_Data(False, False, extraFieldSources,
                                                                    enforceGoodField=False)
            data2D_2, data3D_Perturbations = self.make_Field_Data(False, True, extraFieldSources,
                                                                  enforceGoodField=False)
            assert len(data3D_Perturbations) == len(data3D_NoPerturbations)
            assert iscloseAll(data3D_Perturbations[:, :3], data3D_NoPerturbations[:, :3], 1e-12)
            assert data2D_1 is None and data2D_2 is None
            data3D_Perturbations[np.isnan(data3D_Perturbations)] = 0.0
            data3D_NoPerturbations[np.isnan(data3D_NoPerturbations)] = 0.0
            coords = data3D_NoPerturbations[:, :3]
            fieldValsDifference = data3D_Perturbations[:, 3:] - data3D_NoPerturbations[:, 3:]
            data3D_Difference = np.column_stack((coords, fieldValsDifference))
            data3D_Difference[np.isnan(data3D_Difference)] = 0.0
            data3D_Difference = tuple(self.shape_Field_Data_3D(data3D_Difference))
        else:
            data3D_Difference = None
        return data3D_Difference

    def update_Field_Fact(self, fieldStrengthFact: float) -> None:
        """Update value used to model magnet strength tunability. fieldFact multiplies force and magnetic potential to
        model increasing or reducing magnet strength """
        self.fastFieldHelper.fieldFact = fieldStrengthFact
        self.fieldFact = fieldStrengthFact

    def get_Valid_Jitter_Amplitude(self, Print=False):
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        jitterAmpProposed = self.PTL.jitterAmp
        assert jitterAmpProposed >= 0.0
        maxJitterAmp = self.apMaxGoodField - self.ap
        if maxJitterAmp == 0.0 and jitterAmpProposed != 0.0:
            print('Aperture is set to maximum, no room to misalign element')
        jitterAmp = maxJitterAmp if jitterAmpProposed > maxJitterAmp else jitterAmpProposed
        if Print:
            if jitterAmpProposed == maxJitterAmp and jitterAmpProposed != 0.0:
                print(
                    'jitter amplitude of:' + str(jitterAmpProposed) + ' clipped to maximum value:' + str(maxJitterAmp))
        return jitterAmp

    def perturb_Element(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float) -> None:
        """Overrides abstract method from Element. Add catches for ensuring particle stays in good field region of
        interpolation"""

        if self.PTL.jitterAmp == 0.0 and self.PTL.jitterAmp != 0.0:
            warnings.warn("No jittering was accomodated for, so their will be no effect")
        assert abs(rotZ) < .05 and abs(rotZ) < .05  # small angle
        totalShiftY = shiftY + np.tan(rotZ) * self.L
        totalShiftZ = shiftZ + np.tan(rotY) * self.L
        totalShift = np.sqrt(totalShiftY ** 2 + totalShiftZ ** 2)
        maxShift = self.get_Valid_Jitter_Amplitude()
        if totalShift > maxShift:
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact = .95 * maxShift / totalShift  # safety factor
            print('proposed', totalShift, 'new', reductionFact * maxShift)
            shiftY, shiftZ, rotY, rotZ = [val * reductionFact for val in [shiftY, shiftZ, rotY, rotZ]]
        self.fastFieldHelper.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)


class CombinerHalbachLensSim(CombinerIdeal):
    outerFringeFrac: float = 1.5

    def __init__(self, PTL, Lm: float, rp: float, loadBeamDiam: float, layers: int, ap: Optional[float], mode: str,
                 useStandardMagErrors: bool):
        # PTL: object of ParticleTracerLatticeClass
        # Lm: hardedge length of magnet.
        # loadBeamDiam: Expected diameter of loading beam. Used to set the maximum combiner bending
        # layers: Number of concentric layers
        # mode: wether storage ring or injector. Injector uses high field seeking, storage ring used low field seeking
        assert mode in ('storageRing', 'injector')
        assert all(val > 0 for val in (Lm, rp, loadBeamDiam, layers))
        CombinerIdeal.__init__(self, PTL, Lm, None, None, None, None, None, mode, 1.0)

        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        assert self.maxCombinerAng == .2 and self.outerFringeFrac == 1.5, "May need to change " \
                                                                          "numgrid points if this changes"
        pointPerBoreRadZ = 2
        self.numGridPointsZ: int = make_Odd(
            max([round(pointPerBoreRadZ * (Lm + 2 * self.outerFringeFrac * rp) / rp), 10]))
        # less than 10 and maybe my model to find optimal value doesn't work so well
        self.numGridPointsZ = make_Odd(round(self.numGridPointsZ * self.PTL.fieldDensityMultiplier))
        self.numGridPointsXY: int = make_Odd(round(25 * self.PTL.fieldDensityMultiplier))

        self.Lm = Lm
        self.rp = rp
        self.layers = layers
        self.ap = ap
        self.loadBeamDiam = loadBeamDiam
        self.PTL = PTL
        self.magnetWidths = None
        self.fieldFact: float = -1.0 if mode == 'injector' else 1.0
        self.space: Optional[float] = None
        self.extraFieldLength: float = 0.0
        self.apMaxGoodField: Optional[float] = None
        self.useStandardMagErrors = useStandardMagErrors
        self.extraLoadApFrac = 1.5

        self.La: Optional[
            float] = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb: Optional[
            float] = None  # length of straight section after the kink after the inlet actuall inside the magnet

        self.shape: str = 'COMBINER_CIRCULAR'
        self.inputOffset: Optional[
            float] = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0
        self.lens: Optional[_HalbachLensFieldGenerator] = None

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        rpList = []
        magnetWidthList = []
        for _ in range(self.layers):
            rpList.append(self.rp + sum(magnetWidthList))
            nextMagnetWidth = (self.rp + sum(magnetWidthList)) * np.tan(2 * np.pi / 24) * 2
            magnetWidthList.append(nextMagnetWidth)
        self.magnetWidths = tuple(magnetWidthList)
        self.space = max(rpList) * self.outerFringeFrac
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input in a straight line. This is that section
        individualMagnetLength = min(
            [(MAGNET_ASPECT_RATIO * min(magnetWidthList)), self.Lm])  # this will get rounded up
        # or down
        numSlicesApprox = 1 if not self.useStandardMagErrors else round(self.Lm / individualMagnetLength)
        # print('combiner:',numSlicesApprox)
        self.lens = _HalbachLensFieldGenerator(tuple(rpList), tuple(magnetWidthList), self.Lm,
                                               applyMethodOfMoments=True,
                                               useStandardMagErrors=self.useStandardMagErrors,
                                               numSlices=numSlicesApprox,
                                               useSolenoidField=self.PTL.useSolenoidField)  # must reuse lens
        # because field values are computed twice from same lens. Otherwise, magnet errors would change
        inputAngle, inputOffset, trajectoryLength = self.compute_Input_Orbit_Characteristics()
        assert trajectoryLength > self.Lm + 2 * self.space
        self.Lo = trajectoryLength  # np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1)))
        self.L = self.Lo
        self.ang = inputAngle
        y0 = inputOffset
        x0 = self.space
        theta = inputAngle
        self.La = (y0 + x0 / np.tan(theta)) / (np.sin(theta) + np.cos(theta) ** 2 / np.sin(theta))
        self.inputOffset = inputOffset - np.tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge
        self.outerHalfWidth = max(rpList) + max(magnetWidthList) + MIN_MAGNET_MOUNT_THICKNESS

    def build_Fast_Field_Helper(self, extraSources):
        fieldData = self.make_Field_Data(self.La, self.ang, True)
        self.set_extraFieldLength()
        self.fastFieldHelper = self.init_fastFieldHelper([fieldData, self.La,
                                                          self.Lb, self.Lm, self.space, self.ap, self.ang,
                                                          self.fieldFact,
                                                          self.extraFieldLength, not self.useStandardMagErrors])

        self.fastFieldHelper.force(1e-3, 1e-3, 1e-3)  # force compile
        self.fastFieldHelper.magnetic_Potential(1e-3, 1e-3, 1e-3)  # force compile

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lm / 2 + self.space, self.ap / 2, .0])))
        assert F_edge / F_center < .01

    def make_Grid_Coords_Arrays(self, La: float, ang: float, accomodateJitter: bool) -> tuple:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant

        yMax = self.rp + (La + self.rp * np.sin(abs(ang))) * np.sin(abs(ang))
        yMax_Minimum = self.rp * 1.5 * 1.1
        yMax = yMax if yMax > yMax_Minimum else yMax_Minimum
        yMax = np.clip(yMax, self.rp, np.inf)
        # yMax=yMax if not accomodateJitter else yMax+self.PTL.jitterAmp
        yMin = -TINY_OFFSET if not self.useStandardMagErrors else -yMax
        xMin = -(self.rp - TINY_OFFSET)
        xMax = TINY_OFFSET if not self.useStandardMagErrors else -xMin
        numY = self.numGridPointsXY if not self.useStandardMagErrors else make_Odd(
            round(.9 * (self.numGridPointsXY * 2 - 1)))
        # minus 1 ensures same grid spacing!!
        numX = make_Odd(round(self.numGridPointsXY * self.rp / yMax))
        numX = numX if not self.useStandardMagErrors else make_Odd(round(.9 * (2 * numX - 1)))
        numZ = self.numGridPointsZ if not self.useStandardMagErrors else make_Odd(
            round(1 * (self.numGridPointsZ * 2 - 1)))
        zMax = self.compute_Valid_zMax(La, ang, accomodateJitter)
        zMin = -TINY_OFFSET if not self.useStandardMagErrors else -zMax

        yArr_Quadrant = np.linspace(yMin, yMax, numY)  # this remains y in element frame
        xArr_Quadrant = np.linspace(xMin, xMax, numX)  # this becomes z in element frame, with sign change
        zArr = np.linspace(zMin, zMax, numZ)  # this becomes x in element frame
        assert not is_Even(len(xArr_Quadrant)) and not is_Even(len(yArr_Quadrant)) and not is_Even(len(zArr))
        assert not np.any(np.isnan(xArr_Quadrant))
        assert not np.any(np.isnan(yArr_Quadrant))
        assert not np.any(np.isnan(zArr))
        return xArr_Quadrant, yArr_Quadrant, zArr

    def compute_Valid_zMax(self, La: float, ang: float, accomodateJitter: bool) -> float:
        """Interpolation points inside magnetic material are set to nan. This can cause a problem near externel face of
        combiner because particles may see np.nan when they are actually in a valid region. To circumvent, zMax is
        chosen such that the first z point above the lens is just barely above it, and vacuum tube is configured to
        respect that. See fastNumbaMethodsAndClasses.CombinerHalbachLensSimFieldHelper_Numba.is_Coord_Inside_Vacuum"""

        firstValidPointSpacing = 1e-6
        maxLength = (self.Lb + (La + self.rp * np.sin(abs(ang))) * np.cos(abs(ang)))
        symmetryPlaneX = self.Lm / 2 + self.space  # field symmetry plane location. See how force is computed
        zMax = maxLength - symmetryPlaneX  # subtle. The interpolation must extend to long enough to account for the
        # combiner not being symmetric, but the interpolation field being symmetric. See how force symmetry is handled
        zMax = zMax + self.extraFieldLength if not accomodateJitter else zMax
        pointSpacing = zMax / (self.numGridPointsZ - 1)
        if pointSpacing > self.Lm / 2:
            raise CombinerDimensionError
        lastPointInLensIndex = int((self.Lm / 2) / pointSpacing)  # last point in magnetic material
        distToJustOutsideLens = firstValidPointSpacing + self.Lm / 2 - lastPointInLensIndex * pointSpacing  # just outside material
        extraSpacePerPoint = distToJustOutsideLens / lastPointInLensIndex
        zMax += extraSpacePerPoint * (self.numGridPointsZ - 1)
        assert abs((lastPointInLensIndex * zMax / (self.numGridPointsZ - 1) - self.Lm / 2) - 1e-6), 1e-12
        return zMax

    def make_Field_Data(self, La: float, ang: float, accomodateJitter: bool) -> tuple[np.ndarray, ...]:
        """Make field data as [[x,y,z,Fx,Fy,Fz,V]..] to be used in fast grid interpolator"""
        xArr, yArr, zArr = self.make_Grid_Coords_Arrays(La, ang, accomodateJitter)
        self.apMaxGoodField = self.rp - np.sqrt((xArr[1] - xArr[0]) ** 2 + (yArr[1] - yArr[0]) ** 2)
        self.ap = self.apMaxGoodField - TINY_OFFSET if self.ap is None else self.ap
        assert self.ap < self.apMaxGoodField
        volumeCoords = np.asarray(np.meshgrid(xArr, yArr, zArr)).T.reshape(-1, 3)
        BNormGrad, BNorm = np.zeros((len(volumeCoords), 3)) * np.nan, np.zeros(len(volumeCoords)) * np.nan
        validIndices = np.logical_or(np.linalg.norm(volumeCoords[:, :2], axis=1) <= self.rp,
                                     volumeCoords[:, 2] >= self.Lm / 2)  # tricky
        BNormGrad[validIndices], BNorm[validIndices] = self.lens.BNorm_Gradient(volumeCoords[validIndices],
                                                                                returnNorm=True)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        fieldData = self.shape_Field_Data_3D(data3D)
        return fieldData

    def compute_Input_Orbit_Characteristics(self) -> tuple:
        """compute characteristics of the input orbit. This applies for injected beam, or recirculating beam"""

        LaMax = (self.rp + self.space / np.tan(self.maxCombinerAng)) / (np.sin(self.maxCombinerAng) +
                                                                        np.cos(self.maxCombinerAng) ** 2 / np.sin(
                    self.maxCombinerAng))
        fieldData = self.make_Field_Data(LaMax, self.maxCombinerAng, False)
        self.fastFieldHelper = self.init_fastFieldHelper([fieldData, np.nan, self.Lb,
                                                          self.Lm, self.space, self.ap, np.nan, self.fieldFact,
                                                          self.extraFieldLength, not self.useStandardMagErrors])
        self.outputOffset = self.find_Ideal_Offset()
        inputAngle, inputOffset, qTracedArr, _ = self.compute_Input_Angle_And_Offset(self.outputOffset, ap=self.ap)
        trajectoryLength = self.compute_Trajectory_Length(qTracedArr)
        assert np.abs(inputAngle) < self.maxCombinerAng  # tilt can't be too large or it exceeds field region.
        assert inputAngle * self.fieldFact > 0  # satisfied if low field is positive angle and high is negative.
        # Sometimes this can happen because the lens is to long so an oscilattory behaviour is required by injector
        return inputAngle, inputOffset, trajectoryLength

    def update_Field_Fact(self, fieldStrengthFact) -> None:
        self.fastFieldHelper.fieldFact = fieldStrengthFact
        self.fieldFact = fieldStrengthFact

    def get_Valid_Jitter_Amplitude(self, Print=False):
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        assert self.PTL.jitterAmp >= 0.0
        jitterAmpProposed = self.PTL.jitterAmp
        maxJitterAmp = self.apMaxGoodField - self.ap
        jitterAmp = maxJitterAmp if jitterAmpProposed > maxJitterAmp else jitterAmpProposed
        if Print:
            if jitterAmpProposed == maxJitterAmp and jitterAmpProposed != 0.0:
                print(
                    'jitter amplitude of:' + str(jitterAmpProposed) + ' clipped to maximum value:' + str(maxJitterAmp))
        return jitterAmp

    def set_extraFieldLength(self) -> None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped. Misalignment is a translational and/or rotational,
        so extra length needs to be accounted for in the case of rotational."""
        jitterAmp = self.get_Valid_Jitter_Amplitude(Print=True)
        tiltMax1D = np.arctan(jitterAmp / self.L)  # Tilt in x,y can be higher but I only need to consider 1D
        # because interpolation grid is square
        assert tiltMax1D < .05  # insist small angle approx
        self.extraFieldLength = self.rp * np.tan(tiltMax1D) * 1.5  # safety factor

    def perturb_Element(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float) -> None:
        """Overrides abstract method from Element. Add catches for ensuring particle stays in good field region of
        interpolation"""
        raise NotImplementedError

        assert abs(rotZ) < .05 and abs(rotZ) < .05  # small angle
        totalShiftY = shiftY + np.tan(rotZ) * self.L
        totalShiftZ = shiftZ + np.tan(rotY) * self.L
        totalShift = np.sqrt(totalShiftY ** 2 + totalShiftZ ** 2)
        maxShift = self.get_Valid_Jitter_Amplitude()
        if maxShift == 0.0 and self.PTL.jitterAmp != 0.0:
            warnings.warn("No jittering was accomodated for, so their will be no effect")
        if totalShift > maxShift:
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact = .95 * maxShift / totalShift  # safety factor
            print('proposed', totalShift, 'new', reductionFact * totalShift)
            shiftY, shiftZ, rotY, rotZ = [val * reductionFact for val in [shiftY, shiftZ, rotY, rotZ]]
        self.fastFieldHelper.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)

    def find_Ideal_Offset(self) -> float:
        """use newton's method to find where the vertical translation of the combiner wher the minimum seperation
        between atomic beam path and lens is equal to the specified beam diameter for INJECTED beam. This requires
        modeling high field seekers. Particle is traced backwards from the output of the combiner to the input.
        Can possibly error out from modeling magnet or assembly error"""

        if self.loadBeamDiam / 2 > self.rp * .9:  # beam doens't fit in combiner
            raise CombinerDimensionError
        fieldFact0 = self.fieldFact
        self.update_Field_Fact(-1.0)
        yInitial = self.ap / 10.0
        try:
            inputAngle, _, _, seperationInitial = self.compute_Input_Angle_And_Offset(yInitial, ap=self.ap)
        except:
            raise CombinerDimensionError
        assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
        gradientInitial = (seperationInitial - self.ap) / (yInitial - 0.0)
        y = yInitial
        seperation = seperationInitial  # initial value of lens/atom seperation. This should be equal to input deam diamter/2 eventuall
        gradient = gradientInitial
        i, iterMax = 0, 10  # to prevent possibility of ifnitne loop
        tolAbsolute = 1e-6  # m
        targetSep = self.loadBeamDiam / 2
        while not isclose(seperation, targetSep, abs_tol=tolAbsolute):
            deltaX = -.8 * (seperation - targetSep) / gradient  # I like to use a little damping
            deltaX = -y / 2 if y + deltaX < 0 else deltaX  # restrict deltax to allow value
            y = y + deltaX
            inputAngle, _, _, seperationNew = self.compute_Input_Angle_And_Offset(y, ap=self.ap)
            assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
            gradient = (seperationNew - seperation) / deltaX
            seperation = seperationNew
            i += 1
            if i > iterMax:
                raise CombinerIterExceededError
        assert 0.0 < y < self.ap
        self.update_Field_Fact(fieldFact0)
        return y
#
# class geneticLens(LensIdeal):
#     def __init__(self, PTL, geneticLens, ap):
#         # if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
#         # to accomdate the new rp such as force values and positions
#         # super().__init__(PTL, geneticLens.length, geneticLens.maximum_Radius(), np.nan,np.nan,'injector',fillParams=False)
#         raise NotImplementedError #under construction still
#         super().__init__(PTL, geneticLens.length, None, geneticLens.maximum_Radius(), ap, 0.0, fillParams=False)
#         self.fringeFracOuter = 4.0
#         self.L = geneticLens.length + 2 * self.fringeFracOuter * self.rp
#         self.Lo = None
#         self.shape = 'STRAIGHT'
#         self.lens = geneticLens
#         assert self.lens.minimum_Radius() >= ap
#         self.fringeFracInnerMin = np.inf  # if the total hard edge magnet length is longer than this value * rp, then it can
#         # can safely be modeled as a magnet "cap" with a 2D model of the interior
#         self.self.effectiveLength = None  # if the magnet is very long, to save simulation
#         # time use a smaller length that still captures the physics, and then model the inner portion as 2D
#
#         self.magnetic_Potential_Func_Fringe = None
#         self.magnetic_Potential_Func_Inner = None
#         self.fieldFact = 1.0  # factor to multiply field values by for tunability
#         if self.L is not None:
#             self.build()
#
#     def set_Length(self, L):
#         assert L > 0.0
#         self.L = L
#         self.build()
#
#     def build(self):
#         more robust way to pick number of points in element. It should be done by using the typical lengthscale
#         # of the bore radius
#
#         numPointsLongitudinal = 31
#         numPointsTransverse = 31
#
#         self.Lm = self.L - 2 * self.fringeFracOuter * self.rp  # hard edge length of magnet
#         assert np.abs(self.Lm - self.lens.length) < 1e-6
#         assert self.Lm > 0.0
#         self.Lo = self.L
#
#         numXY = numPointsTransverse
#         # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
#         # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
#         # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
#         yArr_Quadrant = np.linspace(-TINY_STEP, self.ap + TINY_STEP, numXY)
#         xArr_Quadrant = np.linspace(-(self.ap + TINY_STEP), TINY_STEP, numXY)
#
#         zMin = -TINY_STEP
#         zMax = self.L / 2 + TINY_STEP
#         zArr = np.linspace(zMin, zMax, num=numPointsLongitudinal)  # add a little extra so interp works as expected
#
#         # assert (zArr[-1]-zArr[-2])/self.rp<.2, "spatial step size must be small compared to radius"
#         assert len(xArr_Quadrant) % 2 == 1 and len(yArr_Quadrant) % 2 == 1
#         assert all((arr[-1] - arr[-2]) / self.rp < .1 for arr in [xArr_Quadrant, yArr_Quadrant]), "" \
#                                                                "spatial step size must be small compared to radius"
#
#         volumeCoords = np.asarray(np.meshgrid(xArr_Quadrant, yArr_Quadrant, zArr)).T.reshape(-1,
#                                                                             3)  # note that these coordinates can have
#         # the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and
#         # input coordinates will be shifted in a wrapper function
#         BNormGrad, BNorm = self.lens.BNorm_Gradient(volumeCoords, returnNorm=True)
#         data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
#         self.fill_Field_Func(data3D)
#         # self.compile_fast_Force_Function()
#
#         # F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
#         # F_center = np.linalg.norm(self.force(np.asarray([self.L / 2, self.ap / 2, .0])))
#         # assert F_edge / F_center < .01
#
#     def force(self, q, searchIsCoordInside=True):
#         raise Exception("under construction")
#
#         # if np.isnan(F[0])==False:
#         #     if q[0]<2*self.rp*self.fringeFracOuter or q[0]>self.L-2*self.rp*self.fringeFracOuter:
#         #         return np.zeros(3)
#         # F = self.fieldFact * np.asarray(F)
#         # return F
#
#     def fill_Field_Func(self, data):
#         interpF, interpV = self.make_Interp_Functions(data)
#
#         # wrap the function in a more convenietly accesed function
#         @numba.njit(numba.types.UniTuple(numba.float64, 3)(numba.float64, numba.float64, numba.float64))
#         def force_Func(x, y, z):
#             Fx0, Fy0, Fz0 = interpF(-z, y, x)
#             Fx = Fz0
#             Fy = Fy0
#             Fz = -Fx0
#             return Fx, Fy, Fz
#
#         self.force_Func = force_Func
#         self.magnetic_Potential_Func = lambda x, y, z: interpV(-z, y, x)
#
#     def magnetic_Potential(self, q):
#         # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
#         x, y, z = q
#         y = abs(y)  # confine to upper right quadrant
#         z = abs(z)
#         if self.is_Coord_Inside(q) == False:
#             raise Exception(ValueError)
#
#         if 0 <= x <= self.L / 2:
#             x = self.L / 2 - x
#             V = self.magnetic_Potential_Func(x, y, z)
#         elif self.L / 2 < x:
#             x = x - self.L / 2
#             V = self.magnetic_Potential_Func(x, y, z)
#         else:
#             raise Exception(ValueError)
#         return V
