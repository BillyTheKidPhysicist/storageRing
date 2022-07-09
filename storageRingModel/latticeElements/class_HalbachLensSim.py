import warnings
from math import tan, sqrt, inf
from typing import Optional

from latticeElements.Magnets import MagneticLens

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from numbaFunctionsAndObjects import halbachLensFastFunctions

from HalbachLensClass import HalbachLens as _HalbachLensFieldGenerator
from HalbachLensClass import Collection
from constants import TUBE_WALL_THICKNESS
from helperTools import iscloseAll
from helperTools import round_And_Make_Odd
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
        self.numGridPointsX = round_And_Make_Odd(21 * PTL.fieldDensityMultiplier)
        self.numGridPointsR = round_And_Make_Odd(25 * PTL.fieldDensityMultiplier)
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
        self.magnet=None
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
        lensLength = self.effective_Material_Length()
        self.magnet = MagneticLens(lensLength, self.rpLayers, self.magnetWidths, self.PTL.magnet_grade,
                                   self.PTL.use_solenoid_field, self.fringeFracOuter * self.rp)

    def set_length(self, L: float) -> None:
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
            yMin = -yMax
            xMax = self.L + self.extraFieldLength + TINY_INTERP_STEP
            xMin = -(self.extraFieldLength + TINY_INTERP_STEP)
            assert self.fringeFracOuter == 1.5  # pointsperslice mildly depends on this value
            pointsPerBoreRadius = 5
            numPointsZ = round_And_Make_Odd(max([pointsPerBoreRadius*self.Lm/self.rp, 2 * numPointsZ - 1]))
            assert numPointsZ < 150  # things might start taking unreasonably long if not careful
            numPointsXY = 45
        assert not is_Even(numPointsXY) and not is_Even(numPointsZ)
        xArr = np.linspace(xMin, xMax, numPointsZ)
        yArr_Quadrant = np.linspace(yMin, yMax, numPointsXY)
        zArr_Quadrant = yArr_Quadrant.copy()
        return xArr, yArr_Quadrant, zArr_Quadrant

    def make_unshaped_interp_data_2D(self) -> np.ndarray:

        # ignore fringe fields for interior  portion inside then use a 2D plane to represent the inner portion to
        # save resources
        xArr, yArr, zArr = self.make_Grid_Coord_Arrays(True)
        lensCenter=self.magnet.Lm/2.0+self.magnet.x_in_offset
        planeCoords = np.asarray(np.meshgrid(lensCenter, yArr, zArr)).T.reshape(-1, 3)
        B_norm_grad, B_norm = self.magnet.get_valid_field_values(planeCoords,B_GRAD_STEP_SIZE,False)
        data2D = np.column_stack((planeCoords[:, 1:], B_norm_grad[:, 1:], B_norm))  # 2D is formated as
        # [[x,y,z,B0Gx,B0Gy,B0],..]
        return data2D

    def make_unshaped_interp_data_3D(self,useSymmetry=True,use_magnet_errors=False) -> np.ndarray:
        """
        Make 3d field data for interpolation from end of lens region

        If the lens is sufficiently long compared to bore radius then this is only field data from the end region
        (fringe frields and interior near end) because the interior region is modeled as a single plane to exploit
        longitudinal symmetry. Otherwise, it is exactly half of the lens and fringe fields

        """
        assert not (use_magnet_errors and useSymmetry)
        xArr, yArr, zArr = self.make_Grid_Coord_Arrays(useSymmetry)

        volumeCoords = np.asarray(np.meshgrid(xArr, yArr, zArr)).T.reshape(-1,3)  # note that these coordinates can have
        # the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and

        B_norm_grad, B_norm=self.magnet.get_valid_field_values(volumeCoords,B_GRAD_STEP_SIZE,use_magnet_errors)
        data3D = np.column_stack((volumeCoords, B_norm_grad, B_norm))

        return data3D

    def make_interp_data_ideal(self)->tuple[tuple,tuple]:
        exploitVeryLongLens = True if self.effective_Material_Length() < self.Lm else False
        interp_data_3D = self.shape_Field_Data_3D(self.make_unshaped_interp_data_3D())

        if exploitVeryLongLens:
            interp_data_2D = self.shape_Field_Data_2D(self.make_unshaped_interp_data_2D())
        else:
            interp_data_2D = (np.ones(1) * np.nan,) * 5 #dummy data to make Numba happy

        yArr,zArr=interp_data_3D[1],interp_data_3D[2]
        maxGridSep = np.sqrt(2) * (yArr[1] - yArr[0])
        assert self.rp - B_GRAD_STEP_SIZE - maxGridSep > self.maximum_Good_Field_Aperture()
        return interp_data_2D,interp_data_3D

    def make_interp_data(self,apply_perturbation)->tuple[tuple,tuple,tuple]:
        data3D_Difference_no_perturb = (np.ones(1) * np.nan,) * 7

        fieldData2D, fieldData3D = self.make_interp_data_ideal()
        fieldDataPerturbations = self.make_Field_Perturbation_Data() if apply_perturbation else \
            data3D_Difference_no_perturb
        return  fieldData3D, fieldData2D, fieldDataPerturbations

    def build_Fast_Field_Helper(self) -> None:
        """Generate magnetic field gradients and norms for numba jitclass field helper. Low density sampled imperfect
        data may added on top of high density symmetry exploiting perfect data. """
        apply_perturbation = True if self.PTL.standardMagnetErrors or len(self.magnet.neighbors) > 0 else False
        fieldData=self.make_interp_data(apply_perturbation)

        numba_func_constants=(self.L, self.ap, self.Lcap, self.extraFieldLength, self.fieldFact,apply_perturbation)

        force_args = (numba_func_constants, fieldData)
        potential_args = (numba_func_constants, fieldData)
        is_coord_in_vacuum_args = (numba_func_constants,)

        self.assign_numba_functions(halbachLensFastFunctions, force_args, potential_args, is_coord_in_vacuum_args)

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.Lcap, self.ap / 2, .0])))
        assert F_edge / F_center < .015

    def make_Field_Perturbation_Data(self) -> tuple:
        """Make data for fields coming from magnet imperfections and misalingnmet. Imperfect field values are calculated
        and perfect fiel values are subtracted. The difference is then added later on top of perfect field values. This
        force is small, and so I can get away with interpolating with low density, while keeping my high density
        symmetry region. interpolation points inside magnet material are set to zero, so the interpolation may be poor
        near bore of magnet. This is done to avoid dealing with mistmatch  between good field region of ideal and
        perturbation interpolation"""
        data3D_NoPerturbations=self.make_unshaped_interp_data_3D(useSymmetry=False,use_magnet_errors=False)
        data3D_Perturbations=self.make_unshaped_interp_data_3D(useSymmetry=False,use_magnet_errors=True)

        assert len(data3D_Perturbations) == len(data3D_NoPerturbations)
        assert iscloseAll(data3D_Perturbations[:, :3], data3D_NoPerturbations[:, :3], 1e-12)

        data3D_Perturbations[np.isnan(data3D_Perturbations)] = 0.0
        data3D_NoPerturbations[np.isnan(data3D_NoPerturbations)] = 0.0
        coords = data3D_NoPerturbations[:, :3]
        fieldValsDifference = data3D_Perturbations[:, 3:] - data3D_NoPerturbations[:, 3:]
        data3D_Difference = np.column_stack((coords, fieldValsDifference))
        data3D_Difference[np.isnan(data3D_Difference)] = 0.0
        data3D_Difference = tuple(self.shape_Field_Data_3D(data3D_Difference))
        return data3D_Difference

    def update_Field_Fact(self, fieldStrengthFact: float) -> None:
        """Update value used to model magnet strength tunability. fieldFact multiplies force and magnetic potential to
        model increasing or reducing magnet strength """
        warnings.warn("extra field sources are being ignore here. Funcitnality is currently broken")
        self.fieldFact = fieldStrengthFact
        self.build_Fast_Field_Helper()

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
