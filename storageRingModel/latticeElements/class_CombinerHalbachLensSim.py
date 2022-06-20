from math import isclose
from math import sin, sqrt, cos, atan,tan
from typing import Optional

import numpy as np

from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from constants import MIN_MAGNET_MOUNT_THICKNESS, COMBINER_VACUUM_TUBE_THICKNESS
from helperTools import round_And_Make_Odd,make_Odd
from latticeElements.class_CombinerIdeal import CombinerIdeal
from latticeElements.utilities import MAGNET_ASPECT_RATIO, TINY_OFFSET, CombinerDimensionError, \
    CombinerIterExceededError, is_Even, get_Halbach_Layers_Radii_And_Magnet_Widths
from numbaFunctionsAndObjects.fieldHelpers import get_Combiner_Halbach_Field_Helper

DEFAULT_SEED = 42


class CombinerHalbachLensSim(CombinerIdeal):
    outerFringeFrac: float = 1.5
    numGridPointsX: int =30
    numGridPointsY: int =100

    def __init__(self, PTL, Lm: float, rp: float, loadBeamOffset: float, numLayers: int, ap: Optional[float], seed):
        # PTL: object of ParticleTracerLatticeClass
        # Lm: hardedge length of magnet.
        # loadBeamOffset: Expected diameter of loading beam. Used to set the maximum combiner bending
        # layers: Number of concentric layers
        # mode: wether storage ring or injector. Injector uses high field seeking, storage ring used low field seeking
        assert all(val > 0 for val in (Lm, rp, loadBeamOffset, numLayers))
        assert ap < rp if ap is not None else True
        CombinerIdeal.__init__(self, PTL, Lm, None, None, None, None, None, 1.0)

        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        assert self.outerFringeFrac == 1.5, "May need to change numgrid points if this changes"
        pointPerBoreRadZ = 2
        self.numGridPointsZ: int = make_Odd(
            max([round(pointPerBoreRadZ * (Lm + 2 * self.outerFringeFrac * rp) / rp), 15]))
        self.numGridPointsZ = round_And_Make_Odd(self.numGridPointsZ * PTL.fieldDensityMultiplier)

        self.Lm = Lm
        self.rp = rp
        self.numLayers = numLayers
        self.ap = rp - COMBINER_VACUUM_TUBE_THICKNESS if ap is None else ap
        self.loadBeamOffset = loadBeamOffset
        self.PTL = PTL
        self.magnetWidths = None
        self.fieldFact: float = -1.0 if PTL.latticeType == 'injector' else 1.0
        self.space = None
        self.extraFieldLength = 0.0
        self.extraLoadApFrac = 1.5

        self.La = None  # length of segment between inlet and straight section inside the combiner. This length goes from
        # the center of the inlet to the center of the kink
        self.Lb = None  # length of straight section after the kink after the inlet actuall inside the magnet

        self.shape: str = 'COMBINER_CIRCULAR'
        self.inputOffset = None  # offset along y axis of incoming circulating atoms. a particle entering at this offset in
        # the y, with angle self.ang, will exit at x,y=0,0

        self.seed = seed

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        rpLayers, magnetWidths = get_Halbach_Layers_Radii_And_Magnet_Widths(self.rp, self.numLayers)
        self.magnetWidths = magnetWidths
        self.space = max(rpLayers) * self.outerFringeFrac
        self.Lb = self.space + self.Lm  # the combiner vacuum tube will go from a short distance from the ouput right up
        # to the hard edge of the input in a straight line. This is that section
        # or down
        # because field values are computed twice from same lens. Otherwise, magnet errors would change
        inputAngle, inputOffset, trajectoryLength = self.compute_Input_Orbit_Characteristics()
        self.Lo = trajectoryLength
        self.L = self.Lo
        self.ang = inputAngle
        y0 = inputOffset
        x0 = self.space
        self.La = (y0 + x0 / tan(inputAngle)) / (sin(inputAngle) + cos(inputAngle) ** 2 / sin(inputAngle))
        self.inputOffset = inputOffset - tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge
        self.outerHalfWidth = max(rpLayers) + max(magnetWidths) + MIN_MAGNET_MOUNT_THICKNESS
        assert self.ap <= self.max_Ap_Good_Field()

    def make_Lens(self) -> _HalbachLensFieldGenerator:
        """Make field generating lens. A seed is required to reproduce the same magnet if magnet errors are being
        used because this is called multiple times."""
        rpLayers, magnetWidths = get_Halbach_Layers_Radii_And_Magnet_Widths(self.rp, self.numLayers)
        individualMagnetLengthApprox = min([(MAGNET_ASPECT_RATIO * min(magnetWidths)), self.Lm])
        numDisks = 1 if not self.PTL.standardMagnetErrors else round(self.Lm / individualMagnetLengthApprox)

        seed = DEFAULT_SEED if self.seed is None else self.seed
        state = np.random.get_state()
        np.random.seed(seed)
        lens = _HalbachLensFieldGenerator(rpLayers, magnetWidths, self.Lm, self.PTL.magnetGrade,
                                          applyMethodOfMoments=True,
                                          useStandardMagErrors=self.PTL.standardMagnetErrors,
                                          numDisks=numDisks,
                                          useSolenoidField=self.PTL.useSolenoidField)  # must reuse lens
        np.random.set_state(state)
        return lens

    def build_Fast_Field_Helper(self, extraSources):
        self.set_extraFieldLength()
        fieldData = self.make_Field_Data()
        self.fastFieldHelper = get_Combiner_Halbach_Field_Helper([fieldData, self.La,
                                                                  self.Lb, self.Lm, self.space, self.ap, self.ang,
                                                                  self.fieldFact,
                                                                  self.extraFieldLength,
                                                                  not self.PTL.standardMagnetErrors])

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lm / 2 + self.space, self.ap / 2, .0])))
        assert F_edge / F_center < .01

    def make_Grid_Coords_Arrays(self) -> tuple:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        numGridPointsX: int = round_And_Make_Odd(self.numGridPointsX * self.PTL.fieldDensityMultiplier)
        numGridPointsY: int = round_And_Make_Odd(self.numGridPointsY * self.PTL.fieldDensityMultiplier)
        yMax = self.rp + (self.La + self.rp * sin(abs(self.ang))) * sin(abs(self.ang))
        yMax_Minimum = self.rp * 1.5 * 1.1
        yMax = yMax if yMax > yMax_Minimum else yMax_Minimum
        # yMax=yMax if not accomodateJitter else yMax+self.PTL.jitterAmp
        yMin = -TINY_OFFSET if not self.PTL.standardMagnetErrors else -yMax
        xMin = -(self.rp - TINY_OFFSET)
        xMax = TINY_OFFSET if not self.PTL.standardMagnetErrors else -xMin
        numY = numGridPointsY if not self.PTL.standardMagnetErrors else round_And_Make_Odd(.9 * (numGridPointsY * 2 - 1))
        # minus 1 ensures same grid spacing!!
        numX = round_And_Make_Odd(numGridPointsX * self.rp / yMax)
        numX = numX if not self.PTL.standardMagnetErrors else round_And_Make_Odd(.9 * (2 * numX - 1))
        numZ = self.numGridPointsZ if not self.PTL.standardMagnetErrors else round_And_Make_Odd(1 * (self.numGridPointsZ * 2 - 1))
        zMax = self.compute_Valid_zMax()
        zMin = -TINY_OFFSET if not self.PTL.standardMagnetErrors else -zMax

        yArr_Quadrant = np.linspace(yMin, yMax, numY)  # this remains y in element frame
        xArr_Quadrant = np.linspace(xMin, xMax, numX)  # this becomes z in element frame, with sign change
        zArr = np.linspace(zMin, zMax, numZ)  # this becomes x in element frame
        assert not is_Even(len(xArr_Quadrant)) and not is_Even(len(yArr_Quadrant)) and not is_Even(len(zArr))
        return xArr_Quadrant, yArr_Quadrant, zArr

    def compute_Valid_zMax(self) -> float:
        """Interpolation points inside magnetic material are set to nan. This can cause a problem near externel face of
        combiner because particles may see np.nan when they are actually in a valid region. To circumvent, zMax is
        chosen such that the first z point above the lens is just barely above it, and vacuum tube is configured to
        respect that. See fastNumbaMethodsAndClasses.CombinerHalbachLensSimFieldHelper_Numba.is_Coord_Inside_Vacuum"""

        firstValidPointSpacing = 1e-6
        maxLength = (self.Lb + (self.La + self.rp * sin(abs(self.ang))) * cos(abs(self.ang)))
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

    def max_Ap_Good_Field(self):
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
        BNormGrad[validIndices], BNorm[validIndices] = self.make_Lens().BNorm_Gradient(volumeCoords[validIndices],
                                                                                       returnNorm=True)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        fieldData = self.shape_Field_Data_3D(data3D)
        return fieldData

    def compute_Input_Orbit_Characteristics(self) -> tuple:
        """compute characteristics of the input orbit. This applies for injected beam, or recirculating beam"""
        from latticeElements.combiner_characterizer import characterize_CombinerHalbach

        self.outputOffset = self.find_Ideal_Offset()
        atomState = 'HIGH_FIELD_SEEKING' if self.fieldFact == -1 else 'LOW_FIELD_SEEKING'

        inputAngle, inputOffset, trajectoryLength, _ = characterize_CombinerHalbach(self, atomState,
                                                                                    particleOffset=self.outputOffset)
        assert inputAngle * self.fieldFact > 0  # satisfied if low field is positive angle and high is negative.
        # Sometimes this can be triggered because the lens is to long so an oscilattory behaviour is required by
        # injector
        return inputAngle, inputOffset, trajectoryLength

    def update_Field_Fact(self, fieldStrengthFact) -> None:
        self.fastFieldHelper.numbaJitClass.fieldFact = fieldStrengthFact
        self.fieldFact = fieldStrengthFact

    def get_Valid_Jitter_Amplitude(self, Print=False):
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        assert self.PTL.jitterAmp >= 0.0
        jitterAmpProposed = self.PTL.jitterAmp
        maxJitterAmp = self.max_Ap_Good_Field() - self.ap
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
        tiltMax1D = atan(jitterAmp / self.L)  # Tilt in x,y can be higher but I only need to consider 1D
        # because interpolation grid is square
        assert tiltMax1D < .05  # insist small angle approx
        self.extraFieldLength = self.rp * np.tan(tiltMax1D) * 1.5  # safety factor

    def perturb_Element(self, shiftY: float, shiftZ: float, rotY: float, rotZ: float) -> None:
        """Overrides abstract method from Element. Add catches for ensuring particle stays in good field region of
        interpolation"""
        raise NotImplementedError  # need to reimplement the accomodate jitter stuff

        assert abs(rotZ) < .05 and abs(rotZ) < .05  # small angle
        totalShiftY = shiftY + np.tan(rotZ) * self.L
        totalShiftZ = shiftZ + np.tan(rotY) * self.L
        totalShift = sqrt(totalShiftY ** 2 + totalShiftZ ** 2)
        maxShift = self.get_Valid_Jitter_Amplitude()
        if maxShift == 0.0 and self.PTL.jitterAmp != 0.0:
            warnings.warn("No jittering was accomodated for, so their will be no effect")
        if totalShift > maxShift:
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact = .95 * maxShift / totalShift  # safety factor
            print('proposed', totalShift, 'new', reductionFact * totalShift)
            shiftY, shiftZ, rotY, rotZ = [val * reductionFact for val in [shiftY, shiftZ, rotY, rotZ]]
        self.fastFieldHelper.numbaJitClass.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)

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
            deltaX = -(seperation - targetSep) / gradient  # I like to use a little damping
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
