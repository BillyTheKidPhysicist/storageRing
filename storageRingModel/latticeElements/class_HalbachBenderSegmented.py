from math import isclose
from math import sqrt
from typing import Optional

import numpy as np
import scipy.optimize as spo
from scipy.spatial.transform import Rotation as Rot
from shapely.geometry import Polygon

from HalbachLensClass import SegmentedBenderHalbach as _HalbachBenderFieldGenerator
from constants import MIN_MAGNET_MOUNT_THICKNESS, SIMULATION_MAGNETON, VACUUM_TUBE_THICKNESS
from helperTools import max_Tube_Radius_In_Segmented_Bend, arr_Product
from latticeElements.class_BenderIdeal import BenderIdeal
from helperTools import make_Odd
from latticeElements.utilities import TINY_OFFSET, is_Even, TINY_STEP, mirror_Across_Angle, full_Arctan, \
    lst_tup_arr


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
        super().__init__(PTL, None, None, rp, rb, None)
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
        apMaxGeom = max_Tube_Radius_In_Segmented_Bend(self.rb, self.rp, self.Lm, VACUUM_TUBE_THICKNESS)
        safetyFactor = .95
        apMaxGoodField = safetyFactor * self.numPointsBoreAp * self.rp / (
                self.numPointsBoreAp + np.sqrt(2))  # max aperture
        # without particles seeing field interpolation reaching into magnetic materal. Will not be exactly true for
        # several reasons (using int, and non equal grid in xy), so I include a small safety factor
        apMax = min([apMaxGeom, apMaxGoodField])
        assert apMax < self.rp
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
                                            applyMethodOfMoments=True, useSolenoidField=self.PTL.useSolenoidField)
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
                                                      applyMethodOfMoments=False,
                                                      useSolenoidField=self.PTL.useSolenoidField)
        lensAligned = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm,
                                                   numLenses=self.numMagnets, positiveAngleMagnetsOnly=True,
                                                   useMagnetError=False, useHalfCapEnd=(True, True),
                                                   applyMethodOfMoments=False,
                                                   useSolenoidField=self.PTL.useSolenoidField)
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
                                            applyMethodOfMoments=True, useHalfCapEnd=(True, False),
                                            useSolenoidField=self.PTL.useSolenoidField)
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
                                            applyMethodOfMoments=True, useHalfCapEnd=(True, False),
                                            useSolenoidField=self.PTL.useSolenoidField)
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
                                            positiveAngleMagnetsOnly=False, useSolenoidField=self.PTL.useSolenoidField)
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
