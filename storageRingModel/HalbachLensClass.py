import copy
from numbers import Number

import magpylib.current
import numba
import numpy as np
from magpylib import Collection as _Collection
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.obj_classes.class_BaseTransform import apply_move
from magpylib.magnet import Cuboid as _Cuboid
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from constants import MAGNETIC_PERMEABILITY, MAGNET_WIRE_DIAM, SPIN_FLIP_AVOIDANCE_FIELD, GRADE_MAGNETIZATION
from demag_functions import apply_demag
from helperTools import Union, Optional, math, inch_To_Meter, radians, within_Tol, time
from latticeElements.utilities import max_tube_radius_in_segmented_bend, halbach_magnet_width

list_tuple_arr = Union[list, tuple, np.ndarray]
tuple3Float = tuple[float, float, float]

magpyMagnetization_ToSI: float = 1 / (1e3 * MAGNETIC_PERMEABILITY)
SI_MagnetizationToMagpy: float = 1 / magpyMagnetization_ToSI
METER_TO_mm = 1e3  # magpy takes distance in mm

COILS_PER_RADIUS = 4  # number of longitudinal coils per length is this number divided by radius of element


@numba.njit()
def B_NUMBA(r: np.ndarray, r0: np.ndarray, m: np.ndarray) -> np.ndarray:
    r = r - r0  # convert to difference vector
    r_norm_temp = np.sqrt(np.sum(r ** 2, axis=1))
    r_norm = np.empty((r_norm_temp.shape[0], 1))
    r_norm[:, 0] = r_norm_temp
    mr_dot_temp = np.sum(m * r, axis=1)
    mr_dot = np.empty((r_norm_temp.shape[0], 1))
    mr_dot[:, 0] = mr_dot_temp
    B_vec = (MAGNETIC_PERMEABILITY / (4 * np.pi)) * (3 * r * mr_dot / r_norm ** 5 - m / r_norm ** 3)
    return B_vec


class Sphere:

    def __init__(self, radius: float, magnet_grade: str = 'legacy'):
        # angle: symmetry plane angle. There is a negative and positive one
        # radius: radius in inches
        # M: magnetization
        assert radius > 0
        self.angle: Optional[float] = None  # angular location of the magnet
        self.radius: float = radius
        self.volume: float = (4 * np.pi / 3) * self.radius ** 3  # m^3
        self.m0: float = MAGNETIC_PERMEABILITY[magnet_grade] * self.volume  # dipole moment
        self.r0: Optional[np.ndarray] = None  # location of sphere
        self.n: Optional[np.ndarray] = None  # orientation
        self.m: Optional[np.ndarray] = None  # vector sphere moment
        self.theta: Optional[float] = None  # phi position
        self.phi: Optional[float] = None  # orientation of dipole. From lab z axis
        self.psi: Optional[float] = None  # orientation of dipole. in lab xy plane
        self.z: Optional[float] = None
        self.r: Optional[float] = None

    def position_sphere(self, r: float, theta: float, z: float) -> None:
        self.r, self.theta, self.z = r, theta, z
        assert None not in (theta, z, r)
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)
        self.r0 = np.asarray([x, y, self.z])

    def update_size(self, radius: float) -> None:
        self.radius = radius
        self.volume = (4 * np.pi / 3) * self.radius ** 3
        M = 1.15e6  # magnetization density
        self.m0 = M * (4 / 3) * np.pi * self.radius ** 3  # dipole moment
        self.m = self.m0 * self.n  # vector sphere moment

    def orient(self, phi: float, psi: float) -> None:
        # tilt the sphere in spherical coordinates. These are in the lab frame, there is no sphere frame
        # phi,psi =(pi/2,0) is along +x
        # phi,psi=(pi/2,pi/2) is along +y
        # phi,psi=(0.0,anything) is along +z
        self.phi = phi
        self.psi = psi
        self.n = np.asarray([np.sin(phi) * np.cos(psi), np.sin(phi) * np.sin(psi), np.cos(phi)])  # x,y,z
        self.m = self.m0 * self.n

    def B(self, r: np.ndarray) -> np.ndarray:
        assert len(r.shape) == 2 and r.shape[1] == 3
        return B_NUMBA(r, self.r0, self.m)

    def B_shim(self, r: np.ndarray, plane_symmetry: bool = True, negative_symmetry: bool = True,
               rotation_angle: float = np.pi / 3) -> np.ndarray:
        # a single magnet actually represents 12 magnet
        # r: array of N position vectors to get field at. Shape (N,3)
        # plane_symmetry: Wether to exploit z symmetry or not
        # plt.quiver(self.r0[0],self.r0[1],self.m[0],self.m[1],color='r')
        arr = np.zeros(r.shape)
        arr += self.B(r)
        arr += self.B_symmetry(r, 1, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 2, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 3, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 4, negative_symmetry, rotation_angle, not plane_symmetry)
        arr += self.B_symmetry(r, 5, negative_symmetry, rotation_angle, not plane_symmetry)

        if plane_symmetry:
            arr += self.B_symmetry(r, 0, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 1, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 2, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 3, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 4, negative_symmetry, rotation_angle, plane_symmetry)
            arr += self.B_symmetry(r, 5, negative_symmetry, rotation_angle, plane_symmetry)

        return arr

    def B_symmetry(self, r: np.ndarray, rotations: float, negative_symmetry: float, rotation_angle: float,
                   planeReflection: float) -> np.ndarray:
        rot_angle = rotation_angle * rotations
        M_rot = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
        r0_sym = self.r0.copy()
        r0_sym[:2] = M_rot @ r0_sym[:2]
        m_sym = self.m.copy()
        m_sym[:2] = M_rot @ m_sym[:2]
        if negative_symmetry:
            m_sym[:2] *= (-1) ** rotations
        if planeReflection:  # another dipole on the other side of the z=0 line
            r0_sym[2] = -r0_sym[2]
            m_sym[-1] *= -1
        # plt.quiver(r0_sym[0], r0_sym[1], m_sym[0], m_sym[1])
        B_vec_arr = B_NUMBA(r, r0_sym, m_sym)

        return B_vec_arr


class Collection(_Collection):

    def __init__(self, *sources, **kwargs):
        super().__init__(*sources, **kwargs)

    def rotate(self, rotation, anchor=None, start=-1):
        super().rotate(rotation, anchor=0.0, start=start)

    def move(self, displacement, start="auto"):
        displacement = [entry * METER_TO_mm for entry in displacement]
        for child in self.children_all:
            apply_move(child, displacement)

    def _getB_wrapper(self, evalCoords_mm: np.ndarray, sizeMax: float = 500_000) -> np.ndarray:
        """To reduce ram usage, split the sources up into smaller chunks. A bit slower, but works realy well. Only
        applied to sources when ram usage would be too hight"""
        sources_all = self.sources_all
        size = len(evalCoords_mm) * len(self.sources_all)
        splits = min([int(size / sizeMax), len(sources_all)])
        splits = 1 if splits < 1 else splits
        split_size = math.ceil(len(sources_all) / splits)
        split_sources = [sources_all[split_size * i:split_size * (i + 1)] for i in range(splits)] if splits > 1 else [
            sources_all]
        B_vec = np.zeros(evalCoords_mm.shape)
        counter = 0
        for sources in split_sources:
            if len(sources) == 0:
                break
            counter += len(sources)
            B_vec += getBH_level2(sources, evalCoords_mm, sumup=True, squeeze=True, pixel_agg=None, field="B")
        B_vec = B_vec[0] if len(B_vec) == 1 else B_vec
        assert counter == len(sources_all)
        return B_vec

    def B_vec(self, eval_coords: np.ndarray, use_approx: int = False) -> np.ndarray:
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        assert len(self) > 0
        if use_approx:
            raise NotImplementedError  # this is only implement on the bender
        mTesla_To_Tesla = 1e-3
        evalCoords_mm = METER_TO_mm * eval_coords
        B_vec = mTesla_To_Tesla * self._getB_wrapper(evalCoords_mm)
        return B_vec

    def B_norm(self, eval_coords: np.ndarray, use_approx: bool = False) -> np.ndarray:
        # r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array

        B_vec = self.B_vec(eval_coords, use_approx=use_approx)
        if len(eval_coords.shape) == 1:
            return norm(B_vec)
        elif len(eval_coords) == 1:
            return np.asarray([norm(B_vec)])
        else:
            return norm(B_vec, axis=1)

    def central_diff(self, eval_coords: np.ndarray, return_norm: bool, use_approx: bool, dx: float = 1e-7) -> \
            Union[tuple[np.ndarray, ...], np.ndarray]:
        assert dx>0.0
        
        def grad(index: int) -> np.ndarray:
            coord_b = eval_coords.copy()  # upper step
            coord_b[:, index] += dx
            B_norm_b = self.B_norm(coord_b, use_approx=use_approx)
            coord_a = eval_coords.copy()  # upper step
            coord_a[:, index] += -dx
            B_norm_a = self.B_norm(coord_a, use_approx=use_approx)
            return (B_norm_b - B_norm_a) / (2 * dx)

        B_norm_grad = np.column_stack((grad(0), grad(1), grad(2)))
        if return_norm:
            B_norm = self.B_norm(eval_coords, use_approx=use_approx)
            return B_norm_grad, B_norm
        else:
            return B_norm_grad

    def forward_Diff(self, eval_coords: np.ndarray, return_norm: bool, use_approx: bool, dx: float = 1e-7) \
            -> Union[tuple[np.ndarray, ...], np.ndarray]:
        assert dx > 0.0
        B_norm = self.B_norm(eval_coords, use_approx=use_approx)

        def grad(index):
            coord_b = eval_coords.copy()  # upper step
            coord_b[:, index] += dx
            B_norm_b = self.B_norm(coord_b, use_approx=use_approx)
            return (B_norm_b - B_norm) / dx

        B_norm_grad = np.column_stack((grad(0), grad(1), grad(2)))
        if return_norm:
            return B_norm_grad, B_norm
        else:
            return B_norm_grad

    def shape_eval_coords(self, eval_coords: np.ndarray) -> np.ndarray:
        """Shape the coordinates that the field values are evaluated at. valid input shapes are (3) and (N,3) where N
        is the number of points to evaluate. (3) is converted to (1,3)"""

        assert eval_coords.ndim in (1, 2)
        eval_coords_shaped = np.array([eval_coords]) if eval_coords.ndim != 2 else eval_coords
        return eval_coords_shaped

    def B_norm_grad(self, eval_coords: np.ndarray, return_norm: bool = False, diff_method='forward',
                       use_approx: bool = False, dx: float = 1e-7) -> Union[np.ndarray, tuple]:
        # Return the gradient of the norm of the B field. use forward difference theorom
        # r: (N,3) vector of coordinates or (3) vector of coordinates.
        # return_norm: Wether to return the norm as well as the gradient.
        # dr: step size
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array

        eval_coords_shaped = self.shape_eval_coords(eval_coords)

        assert diff_method in ('central', 'forward')
        results = self.central_diff(eval_coords_shaped, return_norm, use_approx,dx=dx) if\
        diff_method == 'central' else self.forward_Diff(eval_coords_shaped, return_norm, use_approx,dx=dx)
        if len(eval_coords.shape) == 1:
            if return_norm:
                [[B_grad_x, B_grad_y, B_grad_z]], [B0] = results
                results = (np.array([B_grad_x, B_grad_y, B_grad_z]), B0)
            else:
                [[B_grad_x, B_grad_y, B_grad_z]] = results
                results = np.array([B_grad_x, B_grad_y, B_grad_z])
        return results

    def apply_method_of_moments(self):
        apply_demag(self)


class Cuboid(_Cuboid):
    def __init__(self, mur: float = 1.0, *args, **kwargs):  # todo: change default to 1.05
        super().__init__(*args, **kwargs)
        self.mur = mur
        self.magnetization0 = self.magnetization.copy()


class Layer(Collection):
    # class object for a layer of the magnet. Uses the RectangularPrism object

    numMagnetsInLayer = 12

    def __init__(self, rp: float, magnetWidth: float, length: float, magnet_grade: str, position: tuple3Float = None,
                 orientation: Rotation = None, mur: float = 1.05,
                 rMagnetShift=None, thetaShift=None, phiShift=None, M_NormShiftRelative=None, dimShift=None,
                 M_AngleShift=None, applyMethodOfMoments=False):
        super().__init__()
        assert magnetWidth > 0.0 and length > 0.0
        assert isinstance(orientation, (type(None), Rotation))
        position = (0.0, 0.0, 0.0) if position is None else position
        self.rMagnetShift: np.ndarray = self.make_Arr_If_None_Else_Copy(rMagnetShift)
        self.thetaShift: np.ndarray = self.make_Arr_If_None_Else_Copy(thetaShift)
        self.phiShift: np.ndarray = self.make_Arr_If_None_Else_Copy(phiShift)
        self.M_NormShiftRelative: np.ndarray = self.make_Arr_If_None_Else_Copy(M_NormShiftRelative)
        self.dimShift: np.ndarray = self.make_Arr_If_None_Else_Copy(dimShift, numParams=3)
        self.M_AngleShift: np.ndarray = self.make_Arr_If_None_Else_Copy(M_AngleShift, numParams=2)
        self.rp: tuple = (rp,) * self.numMagnetsInLayer
        self.mur = mur  # relative permeability
        self.positionToSet = position
        self.orientationToSet = orientation  # orientation about body frame #todo: this "to set" stuff is pretty wonky
        self.magnetWidth: float = magnetWidth
        self.length: float = length
        self.applyMethodOfMoments = applyMethodOfMoments
        self.M: float = GRADE_MAGNETIZATION[magnet_grade]
        self.build()

    def make_Arr_If_None_Else_Copy(self, variable: Optional[list_tuple_arr], numParams=1) -> np.ndarray:
        """If no misalignment is supplied, making correct shaped array of zeros, also check shape of provided array
        is correct"""
        assert variable.shape[1] == numParams if variable is not None else True
        variableArr = np.zeros((self.numMagnetsInLayer, numParams)) if variable is None else copy.copy(variable)
        assert len(variableArr) == self.numMagnetsInLayer and len(variableArr.shape) == 2
        return variableArr

    def build(self) -> None:
        # build the elements that form the layer. The 'home' magnet's center is located at x=r0+width/2,y=0, and its
        # magnetization points along positive x
        # how I do this is confusing
        magnetizationArr = self.make_Mag_Vec_Arr_Magpy()
        dimensionArr = self.make_Cuboid_Dimensions_Magpy()
        positionArr = self.make_Cuboid_Positions_Magpy()
        orientationList = self.make_Cuboid_Orientation_Magpy()
        for M, dim, pos, orientation in zip(magnetizationArr, dimensionArr, positionArr, orientationList):
            box = Cuboid(magnetization=M, dimension=dim,
                         position=pos, orientation=orientation, mur=self.mur)
            self.add(box)

        if self.orientationToSet is not None:
            self.rotate(self.orientationToSet, anchor=0.0)
        self.move(self.positionToSet)
        if self.applyMethodOfMoments:
            self.apply_method_of_moments()

    def make_Cuboid_Orientation_Magpy(self):
        """Make orientations of each magpylib cuboid. A list of scipy Rotation objects. add error effects
        (may be zero though)"""
        phiArr = np.pi + np.arange(0, 12) * 2 * np.pi / 3  # direction of magnetization
        phiArr += np.ravel(self.phiShift)  # add specified rotation, typically errors
        rotationAll = [Rotation.from_rotvec([0.0, 0.0, phi]) for phi in phiArr]
        assert len(rotationAll) == self.numMagnetsInLayer
        return rotationAll

    def make_Cuboid_Positions_Magpy(self):
        """Array of position of each magpylib cuboid, in units of mm. add error effects (may be zero though)"""
        thetaArr = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # location of 12 magnets.
        thetaArr += np.ravel(self.thetaShift)  # add specified rotation, typically errors
        rArr = self.rp + np.ravel(self.rMagnetShift)

        # rArr+=1e-3*(2*(np.random.random_sample(12)-.5))

        rMagnetCenter = rArr + self.magnetWidth / 2  # add specified rotation, typically errors
        xCenter, yCenter = rMagnetCenter * np.cos(thetaArr), rMagnetCenter * np.sin(thetaArr)
        positionAll = np.column_stack((xCenter, yCenter, np.zeros(self.numMagnetsInLayer)))
        positionAll *= METER_TO_mm
        assert positionAll.shape == (self.numMagnetsInLayer, 3)
        return positionAll

    def make_Cuboid_Dimensions_Magpy(self):
        """Make array of dimension of each magpylib cuboid in units of mm. add error effects (may be zero though)"""
        dimensionSingle = np.asarray([self.magnetWidth, self.magnetWidth, self.length])
        dimensionAll = dimensionSingle * np.ones((self.numMagnetsInLayer, 3))
        dimensionAll += self.dimShift
        assert np.all(10 * np.abs(self.dimShift) < dimensionAll)  # model probably doesn't work
        dimensionAll *= METER_TO_mm
        assert dimensionAll.shape == (self.numMagnetsInLayer, 3)
        return dimensionAll

    def make_Mag_Vec_Arr_Magpy(self):
        """Make array of magnetization vector of each magpylib cuboid in units of mT. add error effects (may be zero
        though)"""
        M_MagpyUnit = self.M * SI_MagnetizationToMagpy  # uses units of mT for magnetization.
        magnetizationSingle = np.asarray([M_MagpyUnit, 0.0, 0.0])
        magnetizationAll = magnetizationSingle * np.ones((self.numMagnetsInLayer, 3))
        magnetizationAll *= (1 + self.M_NormShiftRelative)  # add specified fraction shifts, typically errors
        for M, M_Angle in zip(magnetizationAll, self.M_AngleShift):
            R = Rotation.from_rotvec([0.0, M_Angle[0], M_Angle[1]])
            M[:] = R.as_matrix() @ M[:]  # edit in place
        assert magnetizationAll.shape == (self.numMagnetsInLayer, 3)
        return magnetizationAll


class HalbachLens(Collection):
    numMagnetsInLayer = 12

    def __init__(self, rp: Union[float, tuple], magnetWidth: Union[float, tuple], length: float, magnet_grade: str,
                 position: list_tuple_arr = None, orientation: Rotation = None,
                 numDisks: int = 1, applyMethodOfMoments=False, useStandardMagErrors=False,
                 use_solenoid_field: bool = False, sameSeed=False):
        #todo: why does initializing with non zero position indicate zero position still
        # todo: Better seeding system
        super().__init__()
        assert length > 0.0
        assert (isinstance(numDisks, int) and numDisks >= 1)
        assert isinstance(orientation, (type(None), Rotation))
        assert isinstance(rp, (float, tuple)) and isinstance(magnetWidth, (float, tuple))
        position = (0.0, 0.0, 0.0) if position is None else position
        self.rp: tuple = rp if isinstance(rp, tuple) else (rp,)
        assert length / min(self.rp) >= .5 if use_solenoid_field else True  # shorter than this and the solenoid model
        # is dubious
        self.length: float = length

        self.positionToSet = position
        self.orientationToSet = orientation  # orientation about body frame
        self.magnetWidth: tuple = magnetWidth if isinstance(magnetWidth, tuple) else (magnetWidth,)
        self.applyMethodOfMoments: bool = applyMethodOfMoments
        self.useStandardMagErrors: bool = useStandardMagErrors
        if sameSeed is True:
            raise Exception
        self.sameSeed: bool = sameSeed
        self.magnet_grade = magnet_grade
        self.numDisks = numDisks
        self.numLayers = len(self.rp)
        self.use_solenoid_field = use_solenoid_field
        self.mur = 1.05

        self.layerList: list[Layer] = []
        self.build()

    def build(self):
        zArr, lengthArr = self.subdivide_Lens()
        for zLayer, length in zip(zArr, lengthArr):
            for radiusLayer, widthLayer in zip(self.rp, self.magnetWidth):
                if self.useStandardMagErrors:
                    dimVariation, magVecAngleVariation, magNormVariation = self.standard_Magnet_Errors()
                else:
                    dimVariation, magVecAngleVariation, magNormVariation = np.zeros((12, 3)), np.zeros(
                        (12, 2)), np.zeros((12, 1))
                layer = Layer(radiusLayer, widthLayer, length, magnet_grade=self.magnet_grade, position=(0, 0, zLayer),
                              M_AngleShift=magVecAngleVariation, dimShift=dimVariation,
                              M_NormShiftRelative=magNormVariation, mur=self.mur)
                self.add(layer)
                self.layerList.append(layer)
        if self.applyMethodOfMoments:  # this must come before adding solenoids because the demag does not play nice with
            # coils
            self.apply_method_of_moments()
        if self.use_solenoid_field:
            self.add_Solenoid_Coils()
        if self.orientationToSet is not None:
            self.rotate(self.orientationToSet, anchor=0.0)
        self.move(self.positionToSet)

    def standard_Magnet_Errors(self):
        """Make standard tolerances for permanent magnets. From various sources, particularly K&J magnetics"""
        if self.sameSeed:
            np.random.seed(42)
        dimTol = inch_To_Meter(.004)  # dimension variation,inch to meter, +/- meters
        MagVecAngleTol = radians(2)  # magnetization vector angle tolerane,degree to radian,, +/- radians
        MagNormTol = .0125  # magnetization value tolerance, +/- fraction
        dimVariation = self.make_Base_Error_Arr_Cartesian(numParams=3) * dimTol
        MagVecAngleVariation = self.make_Base_Error_Arr_Circular() * MagVecAngleTol
        magNormVariation = self.make_Base_Error_Arr_Cartesian() * MagNormTol
        if self.sameSeed:
            # todo: Does this random seed thing work, or is it a pitfall?
            np.random.seed(int(time.time()))
        return dimVariation, MagVecAngleVariation, magNormVariation

    def make_Base_Error_Arr_Cartesian(self, numParams: int = 1) -> np.ndarray:
        """values range between -1 and 1 with shape (12,numParams)"""
        return 2 * (np.random.random_sample((self.numMagnetsInLayer, numParams)) - .5)

    def make_Base_Error_Arr_Circular(self) -> np.ndarray:
        """Make error array confined inside unit circle. Return results in cartesian with shape (12,2)"""
        theta = 2 * np.pi * np.random.random_sample(self.numMagnetsInLayer)
        radius = np.random.random_sample(self.numMagnetsInLayer)
        x, y = np.cos(theta) * radius, np.sin(theta) * radius
        return np.column_stack((x, y))

    def subdivide_Lens(self) -> tuple[np.ndarray, np.ndarray]:
        """To improve accuracu of magnetostatic method of moments, divide the layers into smaller layers. Also used
         if the lens is composed of slices"""
        LArr = np.ones(self.numDisks) * self.length / self.numDisks
        zArr = np.cumsum(LArr) - .5 * self.length - .5 * self.length / self.numDisks
        assert within_Tol(np.sum(LArr), self.length) and within_Tol(np.mean(zArr), 0.0)
        return zArr, LArr

    def add_Solenoid_Coils(self) -> None:
        """Add simple coils through length of lens. This is to remove the region of non zero magnetic field to prevent
        spin flips"""
        coilDiam = 1.95 * min(self.rp) * METER_TO_mm  # slightly smaller than magnet bore
        zArr, lengthArr = self.subdivide_Lens()
        zLensMin, zLensMax = zArr[0] - lengthArr[0] / 2, zArr[-1] + lengthArr[-1] / 2
        numCoils = max([round(COILS_PER_RADIUS * (zLensMax - zLensMin) / min(self.rp)), 1])
        B_dot_dl = SPIN_FLIP_AVOIDANCE_FIELD * (zLensMax - zLensMin)  # amperes law
        currentInfiniteSolenoid = B_dot_dl / (MAGNETIC_PERMEABILITY * numCoils)  # amperes law
        current = currentInfiniteSolenoid * np.sqrt(1 + (2 * min(self.rp) / self.length) ** 2)
        coilLocationsZArr = METER_TO_mm * np.linspace(zLensMin, zLensMax, numCoils)
        for coilPosZ in coilLocationsZArr:
            loop = magpylib.current.Loop(current=current, diameter=coilDiam, position=(0, 0, coilPosZ))
            self.add(loop)


class SegmentedBenderHalbach(Collection):
    # a model of odd number lenses to represent the symmetry of the segmented bender. The inner lens represents the fully
    # symmetric field

    def __init__(self, rp: float, rb: float, UCAngle: float, Lm: float, magnet_grade: str, numLenses,
                 useHalfCapEnd: tuple[bool, bool],
                 positiveAngleMagnetsOnly: bool = False, applyMethodOfMoments=False, useMagnetError: bool = False,
                 use_solenoid_field: bool = False, magnetWidth: float = None):
        # todo: by default I think it should be positive angles only
        super().__init__()
        assert all(isinstance(value, Number) for value in (rp, rb, UCAngle, Lm)) and isinstance(numLenses, int)
        self.useHalfCapEnd = (False, False) if useHalfCapEnd is None else useHalfCapEnd
        self.rp: float = rp  # radius of bore of magnet, ie to the pole
        self.rb: float = rb  # bending radius
        self.UCAngle: float = UCAngle  # unit cell angle of a HALF single magnet, ie HALF the bending angle of a single magnet. It
        # is called the unit cell because obviously one only needs to use half the magnet and can use symmetry to
        # solve the rest
        self.Lm: float = Lm  # length of single magnet
        self.magnet_grade = magnet_grade
        self.use_solenoid_field = use_solenoid_field
        self.positiveAngleMagnetsOnly: bool = positiveAngleMagnetsOnly  # This is used to model the cap amgnet, and the first full
        # segment. No magnets can be below z=0, but a magnet can be right at z=0. Very different behavious wether negative
        # or positive
        self.magnetWidth: float = halbach_magnet_width(rp, magnetSeparation=0.0) if magnetWidth is None else magnetWidth
        assert np.tan(.5 * Lm / (rb - self.magnetWidth)) <= UCAngle  # magnets should not overlap!
        self.numLenses: int = numLenses  # number of lenses in the model
        self.lensList: list[HalbachLens] = []  # list to hold lenses
        self.lensAnglesArr: np.ndarray = self.make_Lens_Angle_Array()
        self.applyMethodsOfMoments = applyMethodOfMoments
        self.useStandardMagnetErrors = useMagnetError
        self._build()

    def make_Lens_Angle_Array(self) -> np.ndarray:
        if self.numLenses == 1:
            if self.positiveAngleMagnetsOnly:
                raise Exception('Not applicable with only 1 magnet')
            angleArr = np.asarray([0.0])
        else:
            angleArr = np.linspace(-2 * self.UCAngle * (self.numLenses - 1) / 2,
                                   2 * self.UCAngle * (self.numLenses - 1) / 2, num=self.numLenses)
        angleArr = angleArr - angleArr.min() if self.positiveAngleMagnetsOnly else angleArr
        return angleArr

    def lens_Length_And_Angle_Iter(self) -> iter:
        """Create an iterable for length of lenses and their angles. This handles the case of using only a half length
        lens at the beginning and/or end of the bender"""

        angleArr = self.lensAnglesArr.copy()
        assert len(angleArr) > 1
        lengthMagnetList = [self.Lm] * self.numLenses
        angleSep = (angleArr[1] - angleArr[0])
        # the first lens (clockwise sense in xz plane) is a half length lens
        lengthMagnetList[0] = lengthMagnetList[0] / 2 if self.useHalfCapEnd[0] else lengthMagnetList[0]
        angleArr[0] = angleArr[0] + angleSep * .25 if self.useHalfCapEnd[0] else angleArr[0]
        # the last lens (clockwise sense in xz plane) is a half length lens
        lengthMagnetList[-1] = lengthMagnetList[-1] / 2 if self.useHalfCapEnd[1] else lengthMagnetList[-1]
        angleArr[-1] = angleArr[-1] - angleSep * .25 if self.useHalfCapEnd[1] else angleArr[-1]

        return zip(lengthMagnetList, angleArr)

    def _build(self) -> None:
        for Lm, angle in self.lens_Length_And_Angle_Iter():
            lens = HalbachLens(self.rp, self.magnetWidth, Lm, magnet_grade=self.magnet_grade,
                               position=(self.rb, 0.0, 0.0),
                               useStandardMagErrors=self.useStandardMagnetErrors,
                               applyMethodOfMoments=False)
            R = Rotation.from_rotvec([0.0, -angle, 0.0])
            lens.rotate(R, anchor=0)
            # my angle convention is unfortunately opposite what it should be here. positive theta
            # is clockwise about y axis in the xz plane looking from the negative side of y
            # lens.position(r0)
            self.lensList.append(lens)
            self.add(lens)
        if self.applyMethodsOfMoments:  # must be done before adding coils because coils dont' play nice
            self.apply_method_of_moments()
        if self.use_solenoid_field:
            self.add_Solenoid_Coils()

    def get_Seperated_Split_Indices(self, thetaArr: np.ndarray, deltaTheta: float, thetaMin: float, thetaMax: float) \
            -> tuple[int, int]:
        """Return indices that split thetaArr such that the beginning and ending stretch past by -deltaTheta and
        deltaTheta respectively. If thetaMin (thetaMax) is <=thetaArr.min() (>=thetaArr.max()) then don't check that
        the seperation is satisfied at the beginning (ending). The ending index is one index past the desired index
         per slicing rules"""

        assert thetaMax > thetaMin and len(thetaArr.shape) == 1
        assert np.all(thetaArr == thetaArr[np.argsort(thetaArr)])  # must be ascending order
        indexStart = 0 if thetaMin - deltaTheta < thetaArr.min() else np.argmax(thetaArr > thetaMin - deltaTheta) - 1
        # remember, indexEnd is one past the last index!
        indexEnd = len(thetaArr) if thetaMax + deltaTheta > thetaArr.max() else np.argmax(
            thetaArr > thetaMax + deltaTheta)
        if indexStart != 0:
            assert thetaArr[indexStart] <= thetaMin - deltaTheta
        if not indexEnd == len(thetaArr):
            assert thetaArr[indexEnd] >= thetaMax + deltaTheta
        return indexStart, indexEnd

    def get_Valid_SubCoord_Indices(self, thetaArr: np.ndarray, lensSplitIndex1: int, lensSplitIndex2: int,
                                   thetaLower: float, thetaUpper: float) -> tuple[int, int]:
        """Get indices of field coordinates that lie within thetaLower and thetaUpper. If the lens being used is the
        first (last), then all coords before (after) that lens in theta are valid"""

        if lensSplitIndex2 == len(self.lensAnglesArr) and lensSplitIndex1 == 0:  # use all coords indices because all
            # lenses are used
            validCoordIndices = np.ones(len(thetaArr)).astype(bool)
        elif lensSplitIndex1 == 0:
            validCoordIndices = thetaArr <= thetaUpper
        elif lensSplitIndex2 == len(self.lensAnglesArr):
            validCoordIndices = thetaArr > thetaLower
        else:
            validCoordIndices = (thetaArr <= thetaUpper) & (thetaArr > thetaLower)
        return validCoordIndices

    def B_Vec_Approx(self, eval_coords: np.ndarray) -> np.ndarray:
        """Compute the magnetic field vector without using all the individual lenses, only the lenses that are close.
        This should be accurate within 1% based on testing, but 5 times faster. There are some very annoying issues with
        degeneracy of angles from -pi to pi and 0 to 2pi. I avoid this by insisting the bender starts at theta=0 and is
        no longer than 1.5pi when the total length is longer than 1pi. If the total length is less than 1pi
        (from thetaArr.max()-thetaArr.min()), the bender has be confined between -pi and pi continuously. It of course
        doesn't actually have to be, but then it will look longer with the max-min approach. Basically this function
        will not work for benders that are either greater in length than 1pi and don't start at 0, or benders that are
        greater in length than some cutoff and start at 0.0. This can be tricked by a very coarse bender, which I try
        to catch"""
        assert self.Lm / self.rp >= 1  # this isn't validated for smaller aspect ratios
        assert self.lensAnglesArr[1] - self.lensAnglesArr[
            0] < np.pi / 10.0  # i don't expect this to work with small angular
        # differences
        # todo: assert that the spacing is the same
        # todo: make a test that this accepts benders as exptected, and behaves as epxcted. Look at temp4 for a good way to do it
        # todo: rename stuff to be more intelligeable
        thetaArrCoords = np.arctan2(eval_coords[:, 2], eval_coords[:, 0])
        angularLength = self.lensAnglesArr.max() - self.lensAnglesArr.min()
        if angularLength < np.pi:  # bender exists between -pi and pi. Don't need to change anything
            pass
        else:
            angleSymmetryCutoff = 1.5 * np.pi
            assert angularLength < angleSymmetryCutoff
            assert not np.any(
                (self.lensAnglesArr < 0) & (self.lensAnglesArr > angleSymmetryCutoff - 2 * np.pi))  # see docstring
            thetaArrCoords[
                thetaArrCoords < angleSymmetryCutoff - 2 * np.pi] += 2 * np.pi  # if an angle is larger than 3.14,
            # arctan2 doesn't know this, and confines angles between -pi to pi. so I assume the bender starts at 0, then
            # change the values
        numLensBorder = 5  # number of lenses boarding region for field computationg. Must be enough for valid approx
        lensBorderAngleSep = 2 * self.UCAngle * numLensBorder + 1e-6
        splitFactor = 3  # roughly number of lenses (minus number of lenses bordering) per split

        numSplits = round(len(self.lensList) / (2 * numLensBorder + splitFactor))
        numSplits = 1 if numSplits == 0 else numSplits
        splitAngles = np.linspace(self.lensAnglesArr.min(), self.lensAnglesArr.max(), numSplits + 1)
        B_vec = np.zeros(eval_coords.shape)
        indicesEvaluated = np.zeros(
            len(thetaArrCoords))  # to track which indices fields are computed for. Only for an assert check
        for i in range(len(splitAngles) - 1):
            thetaLower, thetaUpper = splitAngles[i], splitAngles[i + 1]
            lensSplitIndex1, lensSplitIndex2 = self.get_Seperated_Split_Indices(self.lensAnglesArr, lensBorderAngleSep,
                                                                                thetaLower, thetaUpper)
            benderLensesSubSection = self.lensList[lensSplitIndex1:lensSplitIndex2]
            validCoordIndices = self.get_Valid_SubCoord_Indices(thetaArrCoords, lensSplitIndex1, lensSplitIndex2,
                                                                thetaLower, thetaUpper)
            if sum(validCoordIndices) > 0:
                for lens in benderLensesSubSection:
                    B_vec[validCoordIndices] += lens.B_vec(eval_coords[validCoordIndices])
                indicesEvaluated += validCoordIndices
        assert np.all(indicesEvaluated == 1)  # check that every coord index was used once and only once
        return B_vec

    def B_vec(self, eval_coords: np.ndarray, use_approx=False) -> np.ndarray:
        """
        overrides Collection

        :param eval_coords: Coordinate to evaluate magnetic field vector at, m. shape (n,3)
        :param use_approx: Wether to use the approximately true, within 1%, method of neglecting lenses that are
        far from a given coordinate in evalCorods
        :return: The magnetic field vector, T. shape (n,3)
        """

        if use_approx:
            return self.B_Vec_Approx(eval_coords)
        else:
            return super().B_vec(eval_coords)

    def add_Solenoid_Coils(self) -> None:
        """Add simple coils through length of lens. This is to remove the region of non zero magnetic field to prevent
        spin flips. Solenoid wraps around an imaginary vacuum tube such that the wires but up against the inside edge
        of the magnets where they approach the bending radius the closest"""

        coilDiam = METER_TO_mm * 2 * max_tube_radius_in_segmented_bend(self.rb, self.rp, self.Lm,
                                                                       tubeWallThickness=MAGNET_WIRE_DIAM)
        angleStart= self.lensAnglesArr[0] if self.useHalfCapEnd[0] else self.lensAnglesArr[0]-self.UCAngle
        angleEnd=self.lensAnglesArr[-1] if self.useHalfCapEnd[1] else self.lensAnglesArr[-1]+self.UCAngle
        circumference = self.rb * (angleEnd - angleStart)
        numCoils = max([round(COILS_PER_RADIUS * circumference / self.rp), 1])
        B_dot_dl = SPIN_FLIP_AVOIDANCE_FIELD * circumference  # amperes law
        current = B_dot_dl / (MAGNETIC_PERMEABILITY * numCoils)  # amperes law
        for theta in np.linspace(angleStart, angleEnd, numCoils):
            loop = magpylib.current.Loop(current=current, diameter=coilDiam, position=(self.rb * METER_TO_mm, 0, 0))
            loop.rotate(Rotation.from_rotvec([0, -theta, 0]), anchor=0)
            self.add(loop)
