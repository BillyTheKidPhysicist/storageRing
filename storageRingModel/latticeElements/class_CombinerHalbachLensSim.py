from math import sin, sqrt, cos, atan, tan, isclose
from typing import Optional

import numpy as np

from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from constants import MIN_MAGNET_MOUNT_THICKNESS, COMBINER_TUBE_WALL_THICKNESS
from helperTools import round_And_Make_Odd
from latticeElements.class_CombinerIdeal import CombinerIdeal
from latticeElements.utilities import MAGNET_ASPECT_RATIO, CombinerDimensionError, \
    CombinerIterExceededError, is_Even, get_Halbach_Layers_Radii_And_Magnet_Widths, round_down_to_nearest_valid_tube_OD
from numbaFunctionsAndObjects.fieldHelpers import get_Combiner_Halbach_Field_Helper

DEFAULT_SEED = 42
SPACE_STEP_SIZE = 1e-7
INTERP_OFFSET = 1.5 * SPACE_STEP_SIZE
TINY_STEP = 1e-12
ndarray = np.ndarray


def make_and_check_arrays_are_odd(xMin, xMax, yMin, yMax, zMin, zMax, numX, numY, numZ) -> tuple[
    ndarray, ndarray, ndarray]:
    assert xMax > xMin and yMax > yMin and zMax > zMin
    xArr = np.linspace(xMin, xMax, numX)  # this becomes z in element frame, with sign change
    yArr = np.linspace(yMin, yMax, numY)  # this remains y in element frame
    zArr = np.linspace(zMin, zMax, numZ)  # this becomes x in element frame
    assert not is_Even(len(xArr)) and not is_Even(len(yArr)) and not is_Even(len(zArr))
    return xArr, yArr, zArr


class CombinerHalbachLensSim(CombinerIdeal):
    outerFringeFrac: float = 1.5
    numGridPointsXY: int = 30
    pointsPerRadiusZ: int = 5

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

        self.Lm = Lm
        self.rp = rp
        self.numLayers = numLayers
        self.ap = self.valid_aperture() if ap is None else ap
        assert self.is_apeture_valid(self.ap)
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

        self.acceptance_width = None
        self.seed = seed

    def fill_Pre_Constrained_Parameters(self) -> None:
        """Overrides abstract method from Element"""
        rpLayers, magnetWidths = get_Halbach_Layers_Radii_And_Magnet_Widths(self.rp, self.numLayers, use_standard_sizes=
        self.PTL.standard_mag_sizes)
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
        self.La = (inputOffset + self.space / tan(inputAngle)) / (
                    sin(inputAngle) + cos(inputAngle) ** 2 / sin(inputAngle))
        self.inputOffset = inputOffset - tan(
            inputAngle) * self.space  # the input offset is measured at the end of the hard edge
        self.outerHalfWidth = max(rpLayers) + max(magnetWidths) + MIN_MAGNET_MOUNT_THICKNESS
        self.acceptance_width = self.get_Acceptance_Width()

    def get_Acceptance_Width(self) -> float:
        extra_large_ange = 2 * self.ang
        width_overshoot = self.rp + (self.La + self.rp * sin(abs(extra_large_ange))) * sin(abs(extra_large_ange))
        return width_overshoot

    def is_apeture_valid(self, ap) -> bool:
        return ap < self.rp - INTERP_OFFSET and ap < self.max_ap_internal_interp_region()

    def valid_aperture(self):
        boreOD = 2 * self.rp
        apLargestTube = round_down_to_nearest_valid_tube_OD(boreOD) / 2.0 - COMBINER_TUBE_WALL_THICKNESS
        apMaxInterp = self.max_ap_internal_interp_region()
        assert apLargestTube < apMaxInterp
        return apLargestTube

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

    def num_points_z_interp(self, zMin: float, zMax: float) -> int:
        assert zMax > zMin
        return round_And_Make_Odd(self.PTL.fieldDensityMultiplier * self.pointsPerRadiusZ * (zMax - zMin) / self.rp)

    def make_Internal_Field_data_symmetry(self) -> tuple[ndarray, ...]:
        xArr, yArr, zArr = self.make_Grid_Coords_Arrays_Internal_Symmetry()
        fieldData = self.make_Field_Data(xArr, yArr, zArr)
        return fieldData

    def make_Field_data_full(self) -> tuple[ndarray, ...]:
        xArr, yArr, zArr = self.make_Full_Grid_Coord_Arrays()
        fieldData = self.make_Field_Data(xArr, yArr, zArr)
        return fieldData

    def make_External_Field_data_symmetry(self) -> tuple[ndarray, ...]:
        xArr, yArr, zArr = self.make_Grid_Coords_Arrays_External_Symmetry()
        zArrInterp = zArr.copy()
        zArrInterp[0] += 2 * TINY_STEP  # to avoid interpolating inside the magnetic material I cheat here
        fieldData = self.make_Field_Data(xArr, yArr, zArrInterp)
        fieldData[2][:] = zArr
        return fieldData

    def build_Fast_Field_Helper(self, extraSources) -> None:
        self.set_extraFieldLength()
        if self.PTL.standardMagnetErrors:
            fieldDataInternal = self.make_Field_data_full()
        else:
            fieldDataInternal = self.make_Internal_Field_data_symmetry()
        fieldDataExternal = self.make_External_Field_data_symmetry()
        self.fastFieldHelper = get_Combiner_Halbach_Field_Helper([fieldDataInternal, fieldDataExternal, self.La,
                                                                  self.Lb, self.Lm, self.space, self.ap, self.ang,
                                                                  self.fieldFact, self.extraFieldLength,
                                                                  not self.PTL.standardMagnetErrors,
                                                                  self.acceptance_width])

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lm / 2 + self.space, self.ap / 2, .0])))
        assert F_edge / F_center < .01

    def make_Full_Grid_Coord_Arrays(self) -> tuple[ndarray, ndarray, ndarray]:

        xMin = -(self.rp - INTERP_OFFSET)
        xMax = -xMin
        zMin = - (self.Lb + (self.La + self.rp * sin(abs(self.ang))) * cos(abs(self.ang))) / 2.0 + self.extraFieldLength
        zMax = -zMin
        m = abs(np.tan(self.ang))
        yMax = m * zMax + (self.acceptance_width + m * self.Lb) + TINY_STEP
        yMin = -yMax
        numX = numY0 = round_And_Make_Odd(self.numGridPointsXY * self.PTL.fieldDensityMultiplier)
        numY = round_And_Make_Odd(numY0 * yMax / self.rp)
        numZ = self.num_points_z_interp(zMin, zMax)
        numX = round_And_Make_Odd(numX * 2 - 1)
        numY = round_And_Make_Odd(numY * 2 - 1)
        numZ = round_And_Make_Odd(numZ * 2 - 1)
        return make_and_check_arrays_are_odd(xMin, xMax, yMin, yMax, zMin, zMax, numX, numY, numZ)

    def make_Grid_Coords_Arrays_External_Symmetry(self) -> tuple[ndarray, ndarray, ndarray]:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        numX = numY0 = round_And_Make_Odd(self.numGridPointsXY * self.PTL.fieldDensityMultiplier)
        yMax = self.acceptance_width + TINY_STEP
        numY = round_And_Make_Odd(numY0 * yMax / self.rp)
        xMin = -(self.rp - INTERP_OFFSET)
        xMax = TINY_STEP
        zMin = self.Lm / 2 - TINY_STEP
        zMax = (self.Lb + (self.La + self.rp * sin(abs(self.ang))) * cos(abs(self.ang))) / 2.0 + TINY_STEP
        numZ = self.num_points_z_interp(zMin, zMax)
        m = abs(np.tan(self.ang))
        yMax = m * zMax + (self.acceptance_width + m * self.Lb) + INTERP_OFFSET
        yMin = -INTERP_OFFSET
        return make_and_check_arrays_are_odd(xMin, xMax, yMin, yMax, zMin, zMax, numX, numY, numZ)

    def make_Grid_Coords_Arrays_Internal_Symmetry(self) -> tuple[ndarray,ndarray,ndarray]:
        # because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        # tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        # quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant

        numX = numY = round_And_Make_Odd(self.numGridPointsXY * self.PTL.fieldDensityMultiplier)
        xMin = -(self.rp - INTERP_OFFSET)
        xMax = INTERP_OFFSET
        yMin = -INTERP_OFFSET
        yMax = (self.rp - INTERP_OFFSET)
        zMin = -INTERP_OFFSET
        zMax = self.Lm / 2 + INTERP_OFFSET
        numZ = self.num_points_z_interp(zMin, zMax)
        return make_and_check_arrays_are_odd(xMin, xMax, yMin, yMax, zMin, zMax, numX, numY, numZ)

    def max_ap_internal_interp_region(self) -> float:
        xArr, yArr, _ = self.make_Grid_Coords_Arrays_Internal_Symmetry()
        assert max(np.abs(xArr)) == max(np.abs(yArr))
        radius_interp_region = max(np.abs(xArr))
        apMaxGoodField = radius_interp_region - np.sqrt((xArr[1] - xArr[0]) ** 2 + (yArr[1] - yArr[0]) ** 2)
        return apMaxGoodField

    def make_Field_Data(self, xArr, yArr, zArr) -> tuple[ndarray, ...]:
        """Make field data as [[x,y,z,Fx,Fy,Fz,V]..] to be used in fast grid interpolator"""
        volumeCoords = np.asarray(np.meshgrid(xArr, yArr, zArr)).T.reshape(-1, 3)
        BNormGrad, BNorm = np.zeros((len(volumeCoords), 3)), np.zeros(len(volumeCoords))
        validIndices = np.logical_or(np.linalg.norm(volumeCoords[:, :2], axis=1) <= self.rp,
                                     volumeCoords[:, 2] >= self.Lm / 2)  # tricky
        BNormGrad[validIndices], BNorm[validIndices] = self.make_Lens().BNorm_Gradient(volumeCoords[validIndices],
                                                                                       returnNorm=True,
                                                                                       dx=SPACE_STEP_SIZE)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        fieldData = self.shape_Field_Data_3D(data3D)
        return fieldData

    def compute_Input_Orbit_Characteristics(self) -> tuple:
        """compute characteristics of the input orbit. This applies for injected beam, or recirculating beam"""
        from latticeElements.combiner_characterizer import characterize_CombinerHalbach

        self.outputOffset = self.find_Ideal_Offset()

        inputAngle, inputOffset, trajectoryLength, _ = characterize_CombinerHalbach(self)
        assert inputAngle * self.fieldFact > 0  # satisfied if low field is positive angle and high is negative.
        # Sometimes this can be triggered because the lens is to long so an oscilattory behaviour is required by
        # injector
        return inputAngle, inputOffset, trajectoryLength

    def update_Field_Fact(self, fieldStrengthFact) -> None:
        self.fastFieldHelper.numbaJitClass.fieldFact = fieldStrengthFact
        self.fieldFact = fieldStrengthFact

    def get_Valid_Jitter_Amplitude(self, Print=False) -> float:
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        assert self.PTL.jitterAmp >= 0.0
        jitterAmpProposed = self.PTL.jitterAmp
        maxJitterAmp = self.max_ap_internal_interp_region() - self.ap
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

        if self.loadBeamOffset >= self.ap:  # beam doens't fit in combiner
            raise CombinerDimensionError
        yInitial = self.ap / 10.0
        try:
            inputAngle, _, _, seperationInitial = characterize_CombinerHalbach(self, atomState='HIGH_FIELD_SEEKING',
                                                                               particleOffset=yInitial)
        except:
            raise CombinerDimensionError
        assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
        gradientInitial = (seperationInitial - self.ap) / (yInitial - 0.0)
        y = yInitial
        seperation = seperationInitial  # initial value of lens/atom seperation.
        gradient = gradientInitial
        i, iterMax = 0, 20  # to prevent possibility of ifnitne loop
        tolAbsolute = 1e-6  # m
        targetSep = self.loadBeamOffset
        while not isclose(seperation, targetSep, abs_tol=tolAbsolute):
            deltaX = -(seperation - targetSep) / gradient  # I like to use a little damping
            deltaX = -y / 2 if y + deltaX < 0 else deltaX  # restrict deltax to allow value
            y = y + deltaX
            inputAngle, _, _, seperationNew = characterize_CombinerHalbach(self, atomState='HIGH_FIELD_SEEKING',
                                                                           particleOffset=y)
            assert inputAngle < 0  # loading beam enters from y<0, if positive then this is circulating beam
            gradient = (seperationNew - seperation) / deltaX
            seperation = seperationNew
            i += 1
            if i > iterMax:
                raise CombinerIterExceededError
        assert 0.0 < y < self.ap
        return y
