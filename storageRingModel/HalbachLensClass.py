import copy
from numbers import Number

import magpylib.current
from magpylib import Collection
from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.obj_classes.class_BaseTransform import apply_move
from magpylib.magnet import Cuboid as _Cuboid
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from constants import MAGNETIC_PERMEABILITY
from demag_functions import apply_demag
from helperTools import numba, np, Union, Optional, math, inch_To_Meter, radians, within_Tol, time, gauss_To_Tesla

M_Default = 1.018E6  # default magnetization value, SI. Magnetization for N48 grade

list_tuple_arr = Union[list, tuple, np.ndarray]
tuple3Float = tuple[float, float, float]

magpyMagnetization_ToSI: float = 1 / (1e3 * MAGNETIC_PERMEABILITY)
SI_MagnetizationToMagpy: float = 1 / magpyMagnetization_ToSI
METER_TO_mm = 1e3  # magpy takes distance in mm
SPIN_FLIP_AVOIDANCE_FIELD = gauss_To_Tesla(3)
COILS_PER_RADIUS = 4  # number of longitudinal coils per length is this number divided by radius of element


@numba.njit()
def B_NUMBA(r: np.ndarray, r0: np.ndarray, m: np.ndarray) -> np.ndarray:
    r = r - r0  # convert to difference vector
    rNormTemp = np.sqrt(np.sum(r ** 2, axis=1))
    rNorm = np.empty((rNormTemp.shape[0], 1))
    rNorm[:, 0] = rNormTemp
    mrDotTemp = np.sum(m * r, axis=1)
    mrDot = np.empty((rNormTemp.shape[0], 1))
    mrDot[:, 0] = mrDotTemp
    Bvec = (MAGNETIC_PERMEABILITY / (4 * np.pi)) * (3 * r * mrDot / rNorm ** 5 - m / rNorm ** 3)
    return Bvec


class Sphere:

    def __init__(self, radius: float, M: float = M_Default):
        # angle: symmetry plane angle. There is a negative and positive one
        # radius: radius in inches
        # M: magnetization
        assert radius > 0 and M > 0
        self.angle: Optional[float] = None  # angular location of the magnet
        self.radius: float = radius
        self.volume: float = (4 * np.pi / 3) * self.radius ** 3  # m^3
        self.m0: float = M * self.volume  # dipole moment
        self.r0: Optional[np.ndarray] = None  # location of sphere
        self.n: Optional[np.ndarray] = None  # orientation
        self.m: Optional[np.ndarray] = None  # vector sphere moment
        self.theta: Optional[float] = None  # phi position
        self.phi: Optional[float] = None  # orientation of dipole. From lab z axis
        self.psi: Optional[float] = None  # orientation of dipole. in lab xy plane
        self.z: Optional[float] = None
        self.r: Optional[float] = None

    def position_Sphere(self, r: float, theta: float, z: float) -> None:
        self.r, self.theta, self.z = r, theta, z
        assert not None in (theta, z, r)
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)
        self.r0 = np.asarray([x, y, self.z])

    def update_Size(self, radius: float) -> None:
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

    def B_Shim(self, r: np.ndarray, planeSymmetry: bool = True, negativeSymmetry: bool = True,
               rotationAngle: float = np.pi / 3) -> np.ndarray:
        # a single magnet actually represents 12 magnet
        # r: array of N position vectors to get field at. Shape (N,3)
        # planeSymmetry: Wether to exploit z symmetry or not
        # plt.quiver(self.r0[0],self.r0[1],self.m[0],self.m[1],color='r')
        arr = np.zeros(r.shape)
        arr += self.B(r)
        arr += self.B_Symmetry(r, 1, negativeSymmetry, rotationAngle, not planeSymmetry)
        arr += self.B_Symmetry(r, 2, negativeSymmetry, rotationAngle, not planeSymmetry)
        arr += self.B_Symmetry(r, 3, negativeSymmetry, rotationAngle, not planeSymmetry)
        arr += self.B_Symmetry(r, 4, negativeSymmetry, rotationAngle, not planeSymmetry)
        arr += self.B_Symmetry(r, 5, negativeSymmetry, rotationAngle, not planeSymmetry)

        if planeSymmetry:
            arr += self.B_Symmetry(r, 0, negativeSymmetry, rotationAngle, planeSymmetry)
            arr += self.B_Symmetry(r, 1, negativeSymmetry, rotationAngle, planeSymmetry)
            arr += self.B_Symmetry(r, 2, negativeSymmetry, rotationAngle, planeSymmetry)
            arr += self.B_Symmetry(r, 3, negativeSymmetry, rotationAngle, planeSymmetry)
            arr += self.B_Symmetry(r, 4, negativeSymmetry, rotationAngle, planeSymmetry)
            arr += self.B_Symmetry(r, 5, negativeSymmetry, rotationAngle, planeSymmetry)

        return arr

    def B_Symmetry(self, r: np.ndarray, rotations: float, negativeSymmetry: float, rotationAngle: float,
                   planeReflection: float) -> np.ndarray:
        rotAngle = rotationAngle * rotations
        M_Rot = np.array([[np.cos(rotAngle), -np.sin(rotAngle)], [np.sin(rotAngle), np.cos(rotAngle)]])
        r0Sym = self.r0.copy()
        r0Sym[:2] = M_Rot @ r0Sym[:2]
        mSym = self.m.copy()
        mSym[:2] = M_Rot @ mSym[:2]
        if negativeSymmetry:
            mSym[:2] *= (-1) ** rotations
        if planeReflection:  # another dipole on the other side of the z=0 line
            r0Sym[2] = -r0Sym[2]
            mSym[-1] *= -1
        # plt.quiver(r0Sym[0], r0Sym[1], mSym[0], mSym[1])
        BVecArr = B_NUMBA(r, r0Sym, mSym)
        return BVecArr


class billyHalbachCollectionWrapper(Collection):

    def __init__(self, *sources, **kwargs):
        super().__init__(*sources, **kwargs)

    def rotate(self, rotation, anchor=None, start=-1):
        if anchor is None:
            raise NotImplementedError  # not sure how to best deal with rotating collection about itself
        super().rotate(rotation, anchor=0.0, start=start)

    def move(self, displacement, start="auto"):
        displacement = [entry * METER_TO_mm for entry in displacement]
        for child in self.children_all:
            apply_move(child, displacement)

    def _getB_Wrapper(self, evalCoords_mm: np.ndarray, sizeMax: float = 500_000) -> np.ndarray:
        """To reduce ram usage, split the sources up into smaller chunks. A bit slower, but works realy well. Only
        applied to sources when ram usage would be too hight"""
        sourcesAll = self.sources_all
        size = len(evalCoords_mm) * len(self.sources_all)
        splits = min([int(size / sizeMax), len(sourcesAll)])
        splits = 1 if splits < 1 else splits
        splitSize = math.ceil(len(sourcesAll) / splits)
        splitSources = [sourcesAll[splitSize * i:splitSize * (i + 1)] for i in range(splits)] if splits > 1 else [
            sourcesAll]
        BVec = np.zeros(evalCoords_mm.shape)
        counter = 0
        for sources in splitSources:
            if len(sources) == 0:
                break
            counter += len(sources)
            BVec += getBH_level2(sources, evalCoords_mm, sumup=True, squeeze=True, pixel_agg=None, field="B")
        BVec = BVec[0] if len(BVec) == 1 else BVec
        assert counter == len(sourcesAll)
        return BVec

    def B_Vec(self, evalCoords: np.ndarray, useApprox: int = False) -> np.ndarray:
        # r: Coordinates to evaluate at with dimension (N,3) where N is the number of evaluate points
        assert len(self) > 0
        if useApprox:
            raise NotImplementedError  # this is only implement on the bender
        mTesla_To_Tesla = 1e-3
        evalCoords_mm = METER_TO_mm * evalCoords
        BVec = mTesla_To_Tesla * self._getB_Wrapper(evalCoords_mm)
        return BVec

    def BNorm(self, evalCoords: np.ndarray, useApprox: bool = False) -> np.ndarray:
        # r: coordinates to evaluate the field at. Either a (N,3) array, where N is the number of points, or a (3) array.
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array

        BVec = self.B_Vec(evalCoords, useApprox=useApprox)
        if len(evalCoords.shape) == 1:
            return norm(BVec)
        elif len(evalCoords) == 1:
            return np.asarray([norm(BVec)])
        else:
            return norm(BVec, axis=1)

    def central_Difference(self, evalCoords: np.ndarray, returnNorm: bool, useApprox: bool, dx: float = 1e-7) -> \
            Union[tuple[np.ndarray, ...], np.ndarray]:

        def grad(index: int) -> np.ndarray:
            coordb = evalCoords.copy()  # upper step
            coordb[:, index] += dx
            BNormB = self.BNorm(coordb, useApprox=useApprox)
            coorda = evalCoords.copy()  # upper step
            coorda[:, index] += -dx
            BNormA = self.BNorm(coorda, useApprox=useApprox)
            return (BNormB - BNormA) / (2 * dx)

        BNormGrad = np.column_stack((grad(0), grad(1), grad(2)))
        if returnNorm:
            BNorm = self.BNorm(evalCoords, useApprox=useApprox)
            return BNormGrad, BNorm
        else:
            return BNormGrad

    def forward_Difference(self, evalCoords: np.ndarray, returnNorm: bool, useApprox: bool, dx: float = 1e-7) \
            -> Union[tuple[np.ndarray, ...], np.ndarray]:
        BNorm = self.BNorm(evalCoords, useApprox=useApprox)

        def grad(index):
            coordb = evalCoords.copy()  # upper step
            coordb[:, index] += dx
            BNormB = self.BNorm(coordb, useApprox=useApprox)
            return (BNormB - BNorm) / dx

        BNormGrad = np.column_stack((grad(0), grad(1), grad(2)))
        if returnNorm:
            return BNormGrad, BNorm
        else:
            return BNormGrad

    def shape_Eval_Coords(self, evalCoords: np.ndarray) -> np.ndarray:
        """Shape the coordinates that the field values are evaluated at. valid input shapes are (3) and (N,3) where N
        is the number of points to evaluate. (3) is converted to (1,3)"""

        assert evalCoords.ndim in (1, 2)
        evalCoordsShaped = np.array([evalCoords]) if evalCoords.ndim != 2 else evalCoords
        return evalCoordsShaped

    def BNorm_Gradient(self, evalCoords: np.ndarray, returnNorm: bool = False, differenceMethod='forward',
                       useApprox: bool = False) -> Union[np.ndarray, tuple]:
        # Return the gradient of the norm of the B field. use forward difference theorom
        # r: (N,3) vector of coordinates or (3) vector of coordinates.
        # returnNorm: Wether to return the norm as well as the gradient.
        # dr: step size
        # Returns a either a (N,3) or (3) array, whichever matches the shape of the r array

        evalCoordsShaped = self.shape_Eval_Coords(evalCoords)

        assert differenceMethod in ('central', 'forward')
        results = self.central_Difference(evalCoordsShaped, returnNorm, useApprox) if differenceMethod == 'central' else \
            self.forward_Difference(evalCoordsShaped, returnNorm, useApprox)
        if len(evalCoords.shape) == 1:
            if returnNorm:
                [[Bgradx, Bgrady, Bgradz]], [B0] = results
                results = (np.array([Bgradx, Bgrady, Bgradz]), B0)
            else:
                [[Bgradx, Bgrady, Bgradz]] = results
                results = np.array([Bgradx, Bgrady, Bgradz])
        return results

    def method_Of_Moments(self):
        apply_demag(self)


class Cuboid(_Cuboid):
    def __init__(self, mur: float = 1.0, *args, **kwargs):  # todo: change default to 1.05
        super().__init__(*args, **kwargs)
        self.mur = mur
        self.magnetization0 = self.magnetization.copy()


class Layer(billyHalbachCollectionWrapper):
    # class object for a layer of the magnet. Uses the RectangularPrism object

    numMagnetsInLayer = 12

    def __init__(self, rp: float, magnetWidth: float, length: float, position: tuple3Float = None,
                 orientation: Rotation = None, M: float = M_Default, mur: float = 1.05,
                 rMagnetShift=None, thetaShift=None, phiShift=None, M_NormShiftRelative=None, dimShift=None,
                 M_AngleShift=None, applyMethodOfMoments=False):
        super().__init__()
        assert magnetWidth > 0.0 and length > 0.0 and M > 0.0
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
        self.M: float = M
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
            self.method_Of_Moments()

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


class HalbachLens(billyHalbachCollectionWrapper):
    numMagnetsInLayer = 12

    def __init__(self, rp: Union[float, tuple], magnetWidth: Union[float, tuple], length: float,
                 position: list_tuple_arr = None, orientation: Rotation = None,
                 M: float = M_Default, numSlices: Optional[int] = 1, applyMethodOfMoments=False,
                 useStandardMagErrors=False, useSolenoidField: bool = False,
                 sameSeed=False):
        # todo: Better seeding system
        super().__init__()
        assert length > 0.0 and M > 0.0
        assert (isinstance(numSlices, int) and numSlices >= 1) if numSlices is not None else numSlices is None
        assert isinstance(orientation, (type(None), Rotation))
        assert isinstance(rp, (float, tuple)) and isinstance(magnetWidth, (float, tuple))
        position = (0.0, 0.0, 0.0) if position is None else position
        self.rp: tuple = rp if isinstance(rp, tuple) else (rp,)
        assert length / min(self.rp) >= .5 if useSolenoidField else True #shorter than this and the solenoid model
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
        self.M: float = M
        self.numSlices = numSlices
        self.numLayers = len(self.rp)
        self.useSolenoidField = useSolenoidField
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
                layer = Layer(radiusLayer, widthLayer, length, M=self.M, position=(0, 0, zLayer),
                              M_AngleShift=magVecAngleVariation, dimShift=dimVariation,
                              M_NormShiftRelative=magNormVariation, mur=self.mur)
                self.add(layer)
                self.layerList.append(layer)
        if self.applyMethodOfMoments:  # this must come before adding solenoids because the demag does not play nice with
            # coils
            self.method_Of_Moments()
        if self.useSolenoidField:
            self.add_Solenoid_Coils()
        if self.orientationToSet is not None:
            self.rotate(self.orientationToSet, anchor=0.0)
        self.move(self.positionToSet)

    def standard_Magnet_Errors(self):
        """Make standard tolerances for permanent magnets. From various sources, particularly K&J magnetics"""
        if self.sameSeed:
            np.random.seed(42)
        dimTol = inch_To_Meter(.004)  # dimension variation,inch to meter, +/- meters
        MagVecAngleTol = radians(1.5)  # magnetization vector angle tolerane,degree to radian,, +/- radians
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
        if self.numSlices is None:
            LArr, zArr = np.array([self.length]), np.zeros(1)
        else:
            LArr = np.ones(self.numSlices) * self.length / self.numSlices
            zArr = np.cumsum(LArr) - self.length / 2 - .5 * self.length / self.numSlices
        assert within_Tol(np.sum(LArr), self.length) and within_Tol(np.mean(zArr),
                                                                    0.0)  # length adds up and centered on 0
        return zArr, LArr

    def add_Solenoid_Coils(self) -> None:
        """Add simple coils through length of lens. This is to remove the region of non zero magnetic field to prevent
        spin flips"""
        coilDiam = 1.95 * min(self.rp) * METER_TO_mm  # slightly smaller than magnet bore
        zArr, lengthArr = self.subdivide_Lens()
        zLensMin, zLensMax = zArr[0] - lengthArr[0] / 2, zArr[-1] + lengthArr[-1] / 2
        numCoils = round((zLensMax - zLensMin) / (min(self.rp) / COILS_PER_RADIUS)) + 1  # prevent zero by adding 1
        B_dot_dl = SPIN_FLIP_AVOIDANCE_FIELD * (zLensMax - zLensMin)
        currentInfiniteSolenoid = B_dot_dl / (MAGNETIC_PERMEABILITY * numCoils)
        scale = np.sqrt(1 + (2 * min(self.rp) / self.length) ** 2)
        currentFiniteSolenoid = currentInfiniteSolenoid * scale
        coilLocationsZArr = METER_TO_mm * np.linspace(zLensMin, zLensMax, numCoils)
        for z in coilLocationsZArr:
            loop = magpylib.current.Loop(current=currentFiniteSolenoid, diameter=coilDiam, position=(0, 0, z))
            self.add(loop)


class SegmentedBenderHalbach(billyHalbachCollectionWrapper):
    # a model of odd number lenses to represent the symmetry of the segmented bender. The inner lens represents the fully
    # symmetric field

    def __init__(self, rp: float, rb: float, UCAngle: float, Lm: float, numLenses: int = 3,
                 M: float = M_Default, positiveAngleMagnetsOnly: bool = False, applyMethodOfMoments=False,
                 useMagnetError: bool = False, useHalfCapEnd: tuple = None, useSolenoidField: bool = False):
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
        self.M: float = M  # magnetization, SI
        self.useSolenoidField = useSolenoidField
        self.positiveAngleMagnetsOnly: bool = positiveAngleMagnetsOnly  # This is used to model the cap amgnet, and the first full
        # segment. No magnets can be below z=0, but a magnet can be right at z=0. Very different behavious wether negative
        # or positive
        self.magnetWidth: float = rp * np.tan(2 * np.pi / 24) * 2  # set to size that exactly fits
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
            lens = HalbachLens(self.rp, self.magnetWidth, Lm, M=self.M, position=(self.rb, 0.0, 0.0),
                               useStandardMagErrors=self.useStandardMagnetErrors,
                               applyMethodOfMoments=False)
            R = Rotation.from_rotvec([0.0, -angle, 0.0])
            lens.rotate(R, anchor=0)
            # my angle convention is unfortunately opposite what it should be here. positive theta
            # is clockwise about y axis in the xz plane looking from the negative side of y
            # lens.position(r0)
            self.lensList.append(lens)
            self.add(lens)
        if self.applyMethodsOfMoments:
            self.method_Of_Moments()

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

    def B_Vec_Approx(self, evalCoords: np.ndarray) -> np.ndarray:
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
        thetaArrCoords = np.arctan2(evalCoords[:, 2], evalCoords[:, 0])
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
        BVec = np.zeros(evalCoords.shape)
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
                    BVec[validCoordIndices] += lens.B_Vec(evalCoords[validCoordIndices])
                indicesEvaluated += validCoordIndices
        assert np.all(indicesEvaluated == 1)  # check that every coord index was used once and only once
        return BVec

    def B_Vec(self, evalCoords: np.ndarray, useApprox=False) -> np.ndarray:
        """
        overrides billyHalbachCollectionWrapper

        :param evalCoords: Coordinate to evaluate magnetic field vector at, m. shape (n,3)
        :param useApprox: Wether to use the approximately true, within 1%, method of neglecting lenses that are
        far from a given coordinate in evalCorods
        :return: The magnetic field vector, T. shape (n,3)
        """

        if useApprox:
            return self.B_Vec_Approx(evalCoords)
        else:
            return super().B_Vec(evalCoords)
