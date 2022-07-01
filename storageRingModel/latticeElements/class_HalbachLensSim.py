import warnings
from math import tan, sqrt, inf
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from HalbachLensClass import billyHalbachCollectionWrapper
from constants import TUBE_WALL_THICKNESS
from helperTools import iscloseAll
from helperTools import make_Odd
from latticeElements.class_LensIdeal import LensIdeal
from latticeElements.utilities import MAGNET_ASPECT_RATIO, is_Even, \
    ElementTooShortError, halbach_Magnet_Width, round_down_to_nearest_valid_tube_OD, B_GRAD_STEP_SIZE, TINY_INTERP_STEP, \
    INTERP_MAGNET_OFFSET
from numbaFunctionsAndObjects.fieldHelpers import get_Halbach_Lens_Helper


# todo: the structure here is confusing and brittle because of the extra field length logic


class HalbachLensSim(LensIdeal):
    fringeFracOuter: float = 1.5
    fringeFracInnerMin = 4.0  # if the total hard edge magnet length is longer than this value * rp, then it can

    # can safely be modeled as a magnet "cap" with a 2D model of the interior

    def __init__(self, PTL, rpLayers: tuple, L: Optional[float], ap: Optional[float],
                 magnetWidths: Optional[tuple]):
        assert all(rp > 0 for rp in rpLayers)
        # if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        # to accomdate the new rp such as force values and positions
        # ----num points depends on a few paremters to be the same as when I determined the optimal values
        assert self.fringeFracOuter == 1.5 and self.fringeFracInnerMin == 4.0, "May need to change numgrid points if " \
                                                                               "this changes"
        self.rp = min(rpLayers)
        self.numGridPointsX = make_Odd(round(21 * PTL.fieldDensityMultiplier))
        self.numGridPointsR = make_Odd(round(25 * PTL.fieldDensityMultiplier))
        self.fringeFieldLength = max(rpLayers) * self.fringeFracOuter
        super().__init__(PTL, L, None, self.rp,
                         None)  # todo: there should be multiple inheritance here for geometries
        self.magnetWidths = self.set_Magnet_Widths(rpLayers, magnetWidths)
        self.ap = self.valid_Aperture() if ap is None else ap
        assert self.ap <= self.maximum_Good_Field_Aperture()
        assert self.ap > 5 * self.rp / self.numGridPointsR  # ap shouldn't be too small. Value below may be dubiuos from interpolation
        self.rpLayers = rpLayers  # can be multiple bore radius for different layers
        self.Lm = None
        self.Lcap: Optional[float] = None  # todo: ridiculous and confusing name
        self.extraFieldLength: Optional[float] = None  # extra field added to end of lens to account misalignment
        self.individualMagnetLength = None
        # or down

    def maximum_Good_Field_Aperture(self) -> float:
        """ from geometric arguments of grid inside circle.
        imagine two concentric rings on a grid, such that no grid box which has a portion outside the outer ring
        has any portion inside the inner ring. This is to prevent interpolation reaching into magnetic material"""
        # todo: why is this so different from the combiner version? It should be like that version instead
        apMax = (self.rp - INTERP_MAGNET_OFFSET) * (1 - sqrt(2) / (self.numGridPointsR - 1))
        return apMax

    def fill_Pre_Constrained_Parameters(self):
        pass

    def fill_Post_Constrained_Parameters(self):
        self.set_extraFieldLength()
        self.fill_Geometric_Params()

    def set_Length(self, L: float) -> None:
        assert L > 0.0
        self.L = L

    def valid_Aperture(self):
        """Get a valid magnet aperture. This is either limited by the good field region, or the available dimensions
        of standard vacuum tubes if specified"""

        # todo: this may give results which are not limited by aperture, but by interpolation region validity
        boreOD = 2 * self.rp
        apLargestTube = round_down_to_nearest_valid_tube_OD(
            boreOD) / 2.0 - TUBE_WALL_THICKNESS if self.PTL.standard_tube_ODs else inf
        apLargestInterp = self.maximum_Good_Field_Aperture()
        ap_valid = apLargestTube if apLargestTube < apLargestInterp else apLargestInterp
        return ap_valid

    def set_extraFieldLength(self) -> None:
        """Set factor that extends field interpolation along length of lens to allow for misalignment. If misalignment
        is too large for good field region, extra length is clipped"""

        jitterAmp = self.get_Valid_Jitter_Amplitude(Print=True)
        tiltMax = np.arctan(jitterAmp / self.L)
        assert 0.0 <= tiltMax < .1  # small angle. Not sure if valid outside that range
        self.extraFieldLength = self.rp * tiltMax * 1.5  # safety factor for approximations

    def effective_Material_Length(self) -> float:
        """If a lens is very long, then longitudinal symmetry can possibly be exploited because the interior region
        is effectively isotropic a sufficient depth inside. This is then modeled as a 2d slice, and the outer edges
        as 3D slice"""
        minimumEffectiveMaterialLength = self.fringeFracInnerMin * max(self.rpLayers)
        return minimumEffectiveMaterialLength if minimumEffectiveMaterialLength < self.Lm else self.Lm

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
        if magnetWidthsProposed is None:
            magnetWidths = tuple(halbach_Magnet_Width(rp, use_standard_sizes=self.PTL.standard_mag_sizes) for
                                 rp in rpLayers)
        else:
            assert len(magnetWidthsProposed) == len(rpLayers)
            maxMagnetWidths = tuple(halbach_Magnet_Width(rp, magnetSeparation=0.0) for rp in rpLayers)
            assert all(width <= maxWidth for width, maxWidth in zip(magnetWidthsProposed, maxMagnetWidths))
            if len(rpLayers) > 1:
                for indexPrev, rp in enumerate(rpLayers[1:]):
                    assert rp >= rpLayers[indexPrev] + magnetWidthsProposed[indexPrev] - 1e-12
            magnetWidths = magnetWidthsProposed
        return magnetWidths

    def fill_Geometric_Params(self) -> None:
        """Compute dependent geometric values"""
        assert self.L is not None  # must be initialized at this point
        self.Lm = self.L - 2 * self.fringeFracOuter * max(self.rpLayers)  # hard edge length of magnet
        if self.Lm < .5 * self.rp:  # If less than zero, unphysical. If less than .5rp, this can screw up my assumption
            # about fringe fields
            raise ElementTooShortError
        self.individualMagnetLength = min(
            [(MAGNET_ASPECT_RATIO * min(self.magnetWidths)), self.Lm])  # this may get rounded
        # up later to satisfy that the total length is Lm
        self.Lo = self.L
        self.Lcap = self.effective_Material_Length() / 2 + self.fringeFracOuter * max(self.rpLayers)
        mountThickness = 1e-3  # outer thickness of mount, likely from space required by epoxy and maybe clamp
        self.outerHalfWidth = max(self.rpLayers) + self.magnetWidths[np.argmax(self.rpLayers)] + mountThickness

    def make_Grid_Coord_Arrays(self, useSymmetry: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        """

        yMin, yMax = -TINY_INTERP_STEP, self.rp - INTERP_MAGNET_OFFSET
        xMin, xMax = -(self.extraFieldLength + TINY_INTERP_STEP), self.Lcap + TINY_INTERP_STEP
        numPointsXY, numPointsZ = self.numGridPointsR, self.numGridPointsX
        if not useSymmetry:  # range will have to fully capture lens.
            numSlices = self.get_Num_Lens_Slices()
            yMin = -yMax
            xMax = self.L + self.extraFieldLength + TINY_INTERP_STEP
            xMin = -(self.extraFieldLength + TINY_INTERP_STEP)
            assert self.fringeFracOuter == 1.5  # pointsperslice mildly depends on this value
            pointsPerSlice = 5
            numPointsZ = make_Odd(round(max([pointsPerSlice * (numSlices + 2), 2 * numPointsZ - 1])))
            assert numPointsZ < 150  # things might start taking unreasonably long if not careful
            numPointsXY = 45
        assert not is_Even(numPointsXY) and not is_Even(numPointsZ)
        xArr = np.linspace(xMin, xMax, numPointsZ)
        yArr_Quadrant = np.linspace(yMin, yMax, numPointsXY)
        zArr_Quadrant = yArr_Quadrant.copy()
        return xArr, yArr_Quadrant, zArr_Quadrant

    def make_2D_Field_Data(self, fieldGenerator: billyHalbachCollectionWrapper, yArr: np.ndarray,
                           zArr: np.ndarray, lensCenter: float) -> Optional[np.ndarray]:

        exploitVeryLongLens = True if self.effective_Material_Length() < self.Lm else False
        if exploitVeryLongLens:
            # ignore fringe fields for interior  portion inside then use a 2D plane to represent the inner portion to
            # save resources
            planeCoords = np.asarray(np.meshgrid(lensCenter, yArr, zArr)).T.reshape(-1, 3)
            validIndices = np.linalg.norm(planeCoords[:, 1:], axis=1) < self.rp - B_GRAD_STEP_SIZE
            BNormGrad, BNorm = np.zeros((len(validIndices), 3)) * np.nan, np.ones(len(validIndices)) * np.nan
            BNormGrad[validIndices], BNorm[validIndices] = fieldGenerator.BNorm_Gradient(planeCoords[validIndices],
                                                                                         returnNorm=True)
            data2D = np.column_stack((planeCoords[:, 1:], BNormGrad[:, 1:], BNorm))  # 2D is formated as
            # [[x,y,z,B0Gx,B0Gy,B0],..]
        else:
            data2D = None
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
        validX = volumeCoords[:, 0] < self.rp * self.fringeFracOuter - B_GRAD_STEP_SIZE
        validYZ = np.linalg.norm(volumeCoords[:, 1:], axis=1) < self.rp - B_GRAD_STEP_SIZE
        validIndices = np.logical_or(validX, validYZ)
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
        lensLength = self.effective_Material_Length() if useSymmetry else self.Lm
        numDisks = 1 if not useStandardMagnetErrors else self.get_Num_Lens_Slices()
        lensCenter = lensLength / 2.0 + self.fringeFracOuter * self.rp
        lens = _HalbachLensFieldGenerator(self.rpLayers, self.magnetWidths, lensLength, self.PTL.magnetGrade,
                                          applyMethodOfMoments=True, useStandardMagErrors=useStandardMagnetErrors,
                                          numDisks=numDisks, useSolenoidField=self.PTL.useSolenoidField,
                                          position=(lensCenter, 0, 0),
                                          orientation=Rot.from_rotvec([0, np.pi / 2.0, 0.0]))
        sources = [src.copy() for src in [*lens.sources_all, *extraFieldSources]]
        fieldGenerator = billyHalbachCollectionWrapper(sources)
        xArr, yArr_Quadrant, zArr_Quadrant = self.make_Grid_Coord_Arrays(useSymmetry)
        maxGridSep = np.sqrt(2) * (yArr_Quadrant[1] - yArr_Quadrant[0])
        if enforceGoodField:  # Don't want to enforce when including perturbation effects
            assert self.rp - B_GRAD_STEP_SIZE - maxGridSep > self.maximum_Good_Field_Aperture()
        data2D = self.make_2D_Field_Data(fieldGenerator, yArr_Quadrant, zArr_Quadrant, lensCenter) if useSymmetry \
            else None
        data3D = self.make_3D_Field_Data(fieldGenerator, xArr, yArr_Quadrant, zArr_Quadrant)
        return data2D, data3D

    def build_Fast_Field_Helper(self, extraFieldSources) -> None:
        """Generate magnetic field gradients and norms for numba jitclass field helper. Low density sampled imperfect
        data may added on top of high density symmetry exploiting perfect data. """

        data2D, data3D = self.make_Field_Data(True, False, extraFieldSources)
        xArrEnd, yArrEnd, zArrEnd, FxArrEnd, FyArrEnd, FzArrEnd, VArrEnd = self.shape_Field_Data_3D(data3D)
        if data2D is not None:  # if no inner plane being used
            yArrIn, zArrIn, FyArrIn, FzArrIn, VArrIn = self.shape_Field_Data_2D(data2D)
        else:
            yArrIn, zArrIn, FyArrIn, FzArrIn, VArrIn = [np.ones(1) * np.nan] * 5
        fieldData = (
            xArrEnd, yArrEnd, zArrEnd, FxArrEnd, FyArrEnd, FzArrEnd, VArrEnd, yArrIn, zArrIn, FyArrIn, FzArrIn, VArrIn)
        fieldDataPerturbations = self.make_Field_Perturbation_Data(extraFieldSources)
        self.fastFieldHelper = get_Halbach_Lens_Helper([fieldData, fieldDataPerturbations, self.L, self.Lcap, self.ap,
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

        if self.PTL.standardMagnetErrors:
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
        self.fastFieldHelper.numbaJitClass.fieldFact = fieldStrengthFact
        self.fieldFact = fieldStrengthFact

    def get_Valid_Jitter_Amplitude(self, Print=False):
        """If jitter (radial misalignment) amplitude is too large, it is clipped"""
        jitterAmpProposed = self.PTL.jitterAmp
        assert jitterAmpProposed >= 0.0
        maxJitterAmp = self.maximum_Good_Field_Aperture() - self.ap
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
        totalShiftY = shiftY + tan(rotZ) * self.L
        totalShiftZ = shiftZ + tan(rotY) * self.L
        totalShift = sqrt(totalShiftY ** 2 + totalShiftZ ** 2)
        maxShift = self.get_Valid_Jitter_Amplitude()
        if totalShift > maxShift:
            print('Misalignment is moving particles to bad field region, misalingment will be clipped')
            reductionFact = .95 * maxShift / totalShift  # safety factor
            print('proposed', totalShift, 'new', reductionFact * maxShift)
            shiftY, shiftZ, rotY, rotZ = [val * reductionFact for val in [shiftY, shiftZ, rotY, rotZ]]
        self.fastFieldHelper.update_Element_Perturb_Params(shiftY, shiftZ, rotY, rotZ)
