from math import isclose
from typing import Optional

import numpy as np

from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from constants import MIN_MAGNET_MOUNT_THICKNESS
from helperTools import make_Odd
from latticeElements.class_CombinerIdeal import CombinerIdeal
from latticeElements.utilities import MAGNET_ASPECT_RATIO, TINY_OFFSET, CombinerDimensionError, \
    CombinerIterExceededError, is_Even


class CombinerHalbachLensSim(CombinerIdeal):
    outerFringeFrac: float = 1.5

    def __init__(self, PTL, Lm: float, rp: float, loadBeamOffset: float, layers: int, ap: Optional[float], mode: str,
                 useStandardMagErrors: bool):
        # PTL: object of ParticleTracerLatticeClass
        # Lm: hardedge length of magnet.
        # loadBeamOffset: Expected diameter of loading beam. Used to set the maximum combiner bending
        # layers: Number of concentric layers
        # mode: wether storage ring or injector. Injector uses high field seeking, storage ring used low field seeking
        assert mode in ('storageRing', 'injector')
        assert all(val > 0 for val in (Lm, rp, loadBeamOffset, layers))
        assert ap < rp if ap is not None else True
        CombinerIdeal.__init__(self, PTL, Lm, None, None, None, None, None, mode, 1.0)

        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        assert self.maxCombinerAng == .2 and self.outerFringeFrac == 1.5, "May need to change " \
                                                                          "numgrid points if this changes"
        pointPerBoreRadZ = 2
        self.numGridPointsZ: int = make_Odd(
            max([round(pointPerBoreRadZ * (Lm + 2 * self.outerFringeFrac * rp) / rp), 10]))
        # less than 10 and maybe my model to find optimal value doesn't work so well
        self.numGridPointsZ = make_Odd(round(self.numGridPointsZ * PTL.fieldDensityMultiplier))
        self.numGridPointsXY: int = make_Odd(round(30 * PTL.fieldDensityMultiplier))

        self.Lm = Lm
        self.rp = rp
        self.layers = layers
        self.ap = .9 * rp if ap is None else ap
        self.loadBeamOffset = loadBeamOffset
        self.PTL = PTL
        self.magnetWidths = None
        self.fieldFact: float = -1.0 if mode == 'injector' else 1.0
        self.space = None
        self.extraFieldLength = 0.0
        self.apMaxGoodField = None
        self.useStandardMagErrors = useStandardMagErrors
        self.extraLoadApFrac = 1.5

        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet

        self.shape: str = 'COMBINER_CIRCULAR'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0
        self.lens = None

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
        assert self.ap <= self.get_Ap_Max_Good_Field()

    def build_Fast_Field_Helper(self, extraSources):
        fieldData = self.make_Field_Data()
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

    def make_Grid_Coords_Arrays(self) -> tuple:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant

        yMax = self.rp + (self.La + self.rp * np.sin(abs(self.ang))) * np.sin(abs(self.ang))
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
        zMax = self.compute_Valid_zMax()
        zMin = -TINY_OFFSET if not self.useStandardMagErrors else -zMax

        yArr_Quadrant = np.linspace(yMin, yMax, numY)  # this remains y in element frame
        xArr_Quadrant = np.linspace(xMin, xMax, numX)  # this becomes z in element frame, with sign change
        zArr = np.linspace(zMin, zMax, numZ)  # this becomes x in element frame
        assert not is_Even(len(xArr_Quadrant)) and not is_Even(len(yArr_Quadrant)) and not is_Even(len(zArr))
        assert not np.any(np.isnan(xArr_Quadrant))
        assert not np.any(np.isnan(yArr_Quadrant))
        assert not np.any(np.isnan(zArr))
        return xArr_Quadrant, yArr_Quadrant, zArr

    def compute_Valid_zMax(self) -> float:
        """Interpolation points inside magnetic material are set to nan. This can cause a problem near externel face of
        combiner because particles may see np.nan when they are actually in a valid region. To circumvent, zMax is
        chosen such that the first z point above the lens is just barely above it, and vacuum tube is configured to
        respect that. See fastNumbaMethodsAndClasses.CombinerHalbachLensSimFieldHelper_Numba.is_Coord_Inside_Vacuum"""

        firstValidPointSpacing = 1e-6
        maxLength = (self.Lb + (self.La + self.rp * np.sin(abs(self.ang))) * np.cos(abs(self.ang)))
        symmetryPlaneX = self.Lm / 2 + self.space  # field symmetry plane location. See how force is computed
        zMax = maxLength - symmetryPlaneX  # subtle. The interpolation must extend to long enough to account for the
        # combiner not being symmetric, but the interpolation field being symmetric. See how force symmetry is handled
        # zMax = zMax + self.extraFieldLength if not accomodateJitter else zMax
        pointSpacing = zMax / (self.numGridPointsZ - 1)
        if pointSpacing > self.Lm / 2:
            raise CombinerDimensionError
        lastPointInLensIndex = int((self.Lm / 2) / pointSpacing)  # last point in magnetic material
        distToJustOutsideLens = firstValidPointSpacing + self.Lm / 2 - lastPointInLensIndex * pointSpacing  # just outside material
        extraSpacePerPoint = distToJustOutsideLens / lastPointInLensIndex
        zMax += extraSpacePerPoint * (self.numGridPointsZ - 1)
        assert abs((lastPointInLensIndex * zMax / (self.numGridPointsZ - 1) - self.Lm / 2) - 1e-6), 1e-12
        return zMax

    def get_Ap_Max_Good_Field(self):
        xArr, yArr, _ = self.make_Grid_Coords_Arrays()
        apMaxGoodField = self.rp - np.sqrt((xArr[1] - xArr[0]) ** 2 + (yArr[1] - yArr[0]) ** 2)
        return apMaxGoodField

    def make_Field_Data(self) -> tuple[np.ndarray, ...]:
        """Make field data as [[x,y,z,Fx,Fy,Fz,V]..] to be used in fast grid interpolator"""
        xArr, yArr, zArr = self.make_Grid_Coords_Arrays()
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
        from latticeElements.combiner_characterizer import characterize_CombinerHalbach

        self.outputOffset = self.find_Ideal_Offset()
        atomState = 'HIGH_FIELD_SEEKING' if self.fieldFact == -1 else 'LOW_FIELD_SEEKING'

        inputAngle, inputOffset, trajectoryLength, minBeamLensSep = characterize_CombinerHalbach(self, atomState,
                                                                                                 particleOffset=self.outputOffset)
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
        maxJitterAmp = self.get_Ap_Max_Good_Field() - self.ap
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
        raise NotImplementedError #need to reimplement the accomodate jitter stuff

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
        from latticeElements.combiner_characterizer import characterize_CombinerHalbach

        if self.loadBeamOffset / 2 > self.rp * .9:  # beam doens't fit in combiner
            raise CombinerDimensionError
        yInitial = self.ap / 10.0
        try:
            inputAngle, _, _, seperationInitial = characterize_CombinerHalbach(self, 'HIGH_FIELD_SEEKING',
                                                                               particleOffset=yInitial)
        except:
            raise CombinerDimensionError
        assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
        gradientInitial = (seperationInitial - self.ap) / (yInitial - 0.0)
        y = yInitial
        seperation = seperationInitial  # initial value of lens/atom seperation. This should be equal to input deam diamter/2 eventuall
        gradient = gradientInitial
        i, iterMax = 0, 10  # to prevent possibility of ifnitne loop
        tolAbsolute = 1e-6  # m
        targetSep = self.loadBeamOffset / 2
        while not isclose(seperation, targetSep, abs_tol=tolAbsolute):
            deltaX = -.8 * (seperation - targetSep) / gradient  # I like to use a little damping
            deltaX = -y / 2 if y + deltaX < 0 else deltaX  # restrict deltax to allow value
            y = y + deltaX
            inputAngle, _, _, seperationNew = characterize_CombinerHalbach(self, 'HIGH_FIELD_SEEKING', particleOffset=y)
            assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
            gradient = (seperationNew - seperation) / deltaX
            seperation = seperationNew
            i += 1
            if i > iterMax:
                raise CombinerIterExceededError
        assert 0.0 < y < self.ap
        return y
