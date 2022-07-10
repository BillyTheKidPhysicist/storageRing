import warnings
from math import isclose, tan, cos, sin, sqrt, atan, pi
from typing import Optional

import numpy as np
import scipy.optimize as spo
from scipy.spatial.transform import Rotation as Rot

from numbaFunctionsAndObjects import benderHalbachFastFunctions

from HalbachLensClass import SegmentedBenderHalbach as _HalbachBenderFieldGenerator
from constants import MIN_MAGNET_MOUNT_THICKNESS, SIMULATION_MAGNETON, TUBE_WALL_THICKNESS
from helperTools import arr_Product, round_And_Make_Odd
from latticeElements.class_BenderIdeal import BenderIdeal
from latticeElements.utilities import TINY_OFFSET, is_even, mirror_across_angle, full_arctan2, \
    max_tube_radius_in_segmented_bend, halbach_magnet_width, get_Unit_Cell_Angle, B_GRAD_STEP_SIZE, \
    INTERP_MAGNET_OFFSET, TINY_INTERP_STEP
from numbaFunctionsAndObjects.fieldHelpers import get_Halbach_Bender
from typeHints import sequence

dummy_field_data_empty=(np.ones(1)*np.nan,)*7

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

    fringe_frac_outer: float = 1.5  # multiple of bore radius to accomodate fringe field

    numModelLenses: int = 7  # number of lenses in halbach model to represent repeating system. Testing has shown
    # this to be optimal

    numPointsBoreApDefault = 25

    def __init__(self, PTL, Lm: float, rp: float, num_magnets: Optional[int], rb: float, ap: Optional[float],
                 r_offset_fact: float):
        assert all(val > 0 for val in (Lm, rp, rb, r_offset_fact))
        assert rb > rp * 10  # this would be very dubious
        super().__init__(PTL, None, None, rp, rb, None)
        self.rb = rb
        self.Lm = Lm
        self.rp = rp
        self.ap = ap
        self.magnet_width = halbach_magnet_width(rp, use_standard_sizes=PTL.use_standard_mag_size)
        self.ucAng: Optional[float] = None
        self.r_offset_fact = r_offset_fact  # factor to times the theoretic optimal bending radius by
        self.Lcap = self.fringe_frac_outer * self.rp
        self.num_magnets = num_magnets
        self.numPointsBoreAp: int = round_And_Make_Odd(self.numPointsBoreApDefault * self.PTL.field_dens_mult)
        # This many points should span the bore ap for good field sampling

    def compute_Maximum_Aperture(self) -> float:
        # beacuse the bender is segmented, the maximum vacuum tube allowed is not the bore of a single magnet
        # use simple geoemtry of the bending radius that touches the top inside corner of a segment
        ap_maxGeom = max_tube_radius_in_segmented_bend(self.rb, self.rp, self.Lm, TUBE_WALL_THICKNESS,
                                                      use_standard_sizes=self.PTL.use_standard_mag_size)
        # todo: revisit this, I am doubtful of how correct this is
        safetyFactor = .95
        ap_maxGoodField = safetyFactor * self.numPointsBoreAp * (self.rp - INTERP_MAGNET_OFFSET) / \
                         (self.numPointsBoreAp + sqrt(2))
        # without particles seeing field interpolation reaching into magnetic materal. Will not be exactly true for
        # several reasons (using int, and non equal grid in xy), so I include a small safety factor
        if ap_maxGoodField < ap_maxGeom:  # for now, I want this to be the case
            warnings.warn("bender aperture being limited by the good field region")
        ap_max = min([ap_maxGeom, ap_maxGoodField])
        assert ap_max < self.rp
        return ap_max

    def set_BpFact(self, BpFact: float):
        assert 0.0 <= BpFact
        self.field_fact = BpFact

    def fill_pre_constrained_parameters(self) -> None:
        self.output_offset = self.find_Optimal_Radial_Offset() * self.r_offset_fact
        self.ro = self.output_offset + self.rb

    def find_Optimal_Radial_Offset(self) -> float:
        """Find the radial offset that accounts for the centrifugal force moving the particles deeper into the
        potential well"""

        m = 1  # in simulation units mass is 1kg
        ucAngApprox = self.get_Unit_Cell_Angle()  # this will be different if the bore radius changes
        lens = _HalbachBenderFieldGenerator(self.rp, self.rb, ucAngApprox, self.Lm, self.PTL.magnet_grade, 10,
                                            (False, False), use_pos_ang_mags_only=False,
                                            magnet_width=self.magnet_width,
                                            use_method_of_moments=True, use_solenoid_field=self.PTL.use_solenoid_field)
        lens.rotate(Rot.from_rotvec([np.pi / 2, 0, 0]))
        theta_arr = np.linspace(0.0, 2 * ucAngApprox, 100)
        z_arr = np.zeros(len(theta_arr))

        def offset_Error(rOffset):
            assert abs(rOffset) < self.rp
            r = self.rb + rOffset
            x_arr = r * np.cos(theta_arr)
            y_arr = r * np.sin(theta_arr)
            coords = np.column_stack((x_arr, y_arr, z_arr))
            F = lens.B_norm_grad(coords) * SIMULATION_MAGNETON
            Fr = np.linalg.norm(F[:, :2], axis=1)
            FCen = np.ones(len(coords)) * m * self.PTL.speed_nominal ** 2 / r
            FCen[coords[:, 2] < 0.0] = 0.0
            error = np.sum((Fr - FCen) ** 2)
            return error

        rOffsetMax = .9 * self.rp
        bounds = [(0.0, rOffsetMax)]
        sol = spo.minimize(offset_Error, np.array([self.rp / 3.0]), bounds=bounds, method='Nelder-Mead',
                           options={'xatol': 1e-5})
        rOffsetOptimal = sol.x[0]
        if isclose(rOffsetOptimal, rOffsetMax, abs_tol=1e-6):
            raise Exception("The bending bore radius is too large to accomodate a reasonable solution")
        return rOffsetOptimal

    def get_Unit_Cell_Angle(self) -> float:
        """Get the angle that a single unit cell spans. Each magnet is composed of two unit cells because of symmetry.
        The unit cell includes half of the magnet and half the gap between the two"""
        return get_Unit_Cell_Angle(self.Lm, self.rb, self.rp + self.magnet_width)

    def fill_post_constrained_parameters(self) -> None:
        self.ap = self.ap if self.ap is not None else self.compute_Maximum_Aperture()
        assert self.ap <= self.compute_Maximum_Aperture()
        self.ucAng = self.get_Unit_Cell_Angle()
        self.ang = 2 * self.num_magnets * self.ucAng
        self.fill_In_And_Out_Rotation_Matrices()
        assert self.ang < 2 * pi * 3 / 4  # not sure why i put this here
        self.ro = self.rb + self.output_offset
        self.L = self.ang * self.rb
        self.Lo = self.ang * self.ro + 2 * self.Lcap
        self.outer_half_width = self.rp + self.magnet_width + MIN_MAGNET_MOUNT_THICKNESS

    def build_fast_field_helper(self) -> None:
        """compute field values and build fast numba helper"""
        fieldDataSeg = self.generate_Segment_Field_Data()
        fieldDataInternal = self.generate_Internal_Fringe_Field_Data()
        fieldDataCap = self.generate_Cap_Field_Data()
        fieldDataPerturbation = self.generate_Perturbation_Data() if self.PTL.use_mag_errors else dummy_field_data_empty
        assert np.all(fieldDataCap[0] == fieldDataInternal[0]) and np.all(fieldDataCap[2] == fieldDataInternal[2])
        field_data=(fieldDataSeg, fieldDataInternal, fieldDataCap, fieldDataPerturbation)
        useFieldPerturbations = self.PTL.use_mag_errors

        m = np.tan(self.ucAng)
        M_uc = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        m = np.tan(self.ang / 2)
        M_ang = np.asarray([[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]) * 1 / (1 + m ** 2)  # reflection matrix
        RIn_Ang = np.asarray([[np.cos(self.ang), np.sin(self.ang)], [-np.sin(self.ang), np.cos(self.ang)]])

        numba_func_constants=(self.rb,self.ap,self.Lcap,self.ang,self.num_magnets,
                              self.ucAng,M_ang,RIn_Ang,M_uc,self.field_fact,useFieldPerturbations)

        force_args=(numba_func_constants,field_data)
        potential_args=(numba_func_constants,field_data)
        is_coord_in_vacuum_args=(numba_func_constants,)
        

        self.assign_numba_functions(benderHalbachFastFunctions,force_args,potential_args,is_coord_in_vacuum_args)


    def make_Grid_Coords(self, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
        """Make Array of points that the field will be evaluted at for fast interpolation. only x and s values change.
        """
        assert not is_even(self.numPointsBoreAp)  # points should be odd to there is a point at zero field, if possible
        longitudinalCoordSpacing: float = (.8 * self.rp / 10.0) / self.PTL.field_dens_mult  # Spacing
        # through unit cell. .8 was carefully chosen
        numPointsX = round_And_Make_Odd(self.numPointsBoreAp * (x_max - x_min) / self.ap)
        z_min, z_max = -TINY_INTERP_STEP, (self.ap + TINY_INTERP_STEP)  # same for every part of bender
        num_points_z = self.numPointsBoreAp
        numPointsY = round_And_Make_Odd((y_max - y_min) / longitudinalCoordSpacing)
        assert (numPointsX + 1) / num_points_z >= (x_max - x_min) / (z_max - z_min)  # should be at least this ratio
        xArrArgs, yArrArgs, z_arrArgs = (x_min, x_max, numPointsX), (y_min, y_max, numPointsY), (z_min, z_max, num_points_z)
        coordArrList = [np.linspace(arrArgs[0], arrArgs[1], arrArgs[2]) for arrArgs in (xArrArgs, yArrArgs, z_arrArgs)]
        gridCoords = np.asarray(np.meshgrid(*coordArrList)).T.reshape(-1, 3)
        return gridCoords

    def convert_Center_To_Cartesian_Coords(self, s: float, xc: float, yc: float) -> tuple[float, float, float]:
        """Convert center coordinates [s,xc,yc] to cartesian coordinates[x,y,z]"""

        if -TINY_OFFSET <= s < self.Lcap:
            x, y, z = self.rb + xc, s - self.Lcap, yc
        elif self.Lcap <= s < self.Lcap + self.ang * self.rb:
            theta = (s - self.Lcap) / self.rb
            r = self.rb + xc
            x, y, z = cos(theta) * r, sin(theta) * r, yc
        elif self.Lcap + self.ang * self.rb <= s <= self.ang * self.rb + 2 * self.Lcap + TINY_OFFSET:
            theta = self.ang
            r = self.rb + xc
            x0, y0 = cos(theta) * r, sin(theta) * r
            deltaS = s - (self.ang * self.rb + self.Lcap)
            thetaPerp = pi + atan(-1 / tan(theta))
            x, y, z = x0 + cos(thetaPerp) * deltaS, y0 + sin(thetaPerp) * deltaS, yc
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
        numS = round_And_Make_Odd(5 * (self.num_magnets + 2))  # carefully measured
        numYc = round_And_Make_Odd(35 * self.PTL.field_dens_mult)
        numXc = numYc

        sArr = np.linspace(-TINY_OFFSET, Ls + TINY_OFFSET, numS)  # distance through bender along center
        xcArr = np.linspace(-self.ap - TINY_OFFSET, self.ap + TINY_OFFSET, numXc)  # radial deviation along major radius
        ycArr = np.linspace(-self.ap - TINY_OFFSET, self.ap + TINY_OFFSET,
                            numYc)  # deviation in vertical from center of
        # bender, along y in cartesian
        assert not is_even(len(sArr)) and not is_even(len(xcArr)) and not is_even(len(ycArr))
        coordsCenter = arr_Product(sArr, xcArr, ycArr)
        coords = np.asarray([self.convert_Center_To_Cartesian_Coords(*coordCenter) for coordCenter in coordsCenter])
        return coordsCenter, coords

    def generate_Perturbation_Data(self) -> tuple[np.ndarray, ...]:
        coordsCenter, coordsCartesian = self.make_Perturbation_Data_Coords()
        lensImperfect = self.build_Bender(True, (True, True), methodOfMoments=False, numLenses=self.num_magnets,
                                          use_mag_errors=True)
        lensPerfect = self.build_Bender(True, (True, True), methodOfMoments=False, numLenses=self.num_magnets)
        rCenterArr = np.linalg.norm(coordsCenter[:, 1:], axis=1)
        validIndices = rCenterArr < self.rp
        valsImperfect = np.column_stack(self.compute_Valid_Field_Vals(lensImperfect, coordsCartesian, validIndices))
        valsPerfect = np.column_stack(self.compute_Valid_Field_Vals(lensPerfect, coordsCartesian, validIndices))
        valsPerturbation = valsImperfect - valsPerfect
        valsPerturbation[np.isnan(valsPerturbation)] = 0.0
        interpData = np.column_stack((coordsCenter, valsPerturbation))
        interpData = self.shape_field_data_3D(interpData)

        return interpData

    def generate_Cap_Field_Data(self) -> tuple[np.ndarray, ...]:
        # x and y bounds should match with internal fringe bounds
        x_min = (self.rb - self.ap) * cos(2 * self.ucAng) - TINY_INTERP_STEP
        x_max = self.rb + self.ap + TINY_INTERP_STEP
        y_min = -(self.Lcap + TINY_INTERP_STEP)
        y_max = TINY_INTERP_STEP
        fieldCoords = self.make_Grid_Coords(x_min, x_max, y_min, y_max)
        validIndices = np.sqrt((fieldCoords[:, 0] - self.rb) ** 2 + fieldCoords[:, 2] ** 2) < self.rp - B_GRAD_STEP_SIZE
        lens = self.build_Bender(True, (True, False))
        return self.compute_Valid_Field_Data(lens, fieldCoords, validIndices)

    def generate_Internal_Fringe_Field_Data(self) -> tuple[np.ndarray, ...]:
        """An magnet slices are required to model the region going from the cap to the repeating unit cell,otherwise
        there is too large of an energy discontinuity"""
        # x and y bounds should match with cap bounds
        x_min = (self.rb - self.ap) * cos(2 * self.ucAng) - TINY_INTERP_STEP  # inward enough to account for the tilt
        x_max = self.rb + self.ap + TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max = tan(2 * self.ucAng) * (self.rb + self.ap) + TINY_INTERP_STEP
        fieldCoords = self.make_Grid_Coords(x_min, x_max, y_min, y_max)
        lens = self.build_Bender(True, (True, False))
        validIndices = self.get_Valid_Indices_Internal(fieldCoords, 3)
        return self.compute_Valid_Field_Data(lens, fieldCoords, validIndices)

    def is_Valid_In_Lens_Of_Bender(self, x: bool, y: bool, z: bool) -> bool:
        """Check that the coordinates x,y,z are valid for a lens in the bender. The lens is centered on (self.rb,0,0)
        aligned with the z axis. If the coordinates are outside the double unit cell containing the lens, or inside
        the toirodal cylinder enveloping the magnet material, the coordinate is invalid"""
        yUC_Line = tan(self.ucAng) * x
        minor_radius = sqrt((x - self.rb) ** 2 + z ** 2)
        lens_radius_valid_inner = self.rp - B_GRAD_STEP_SIZE
        lens_radius_valid_outer = self.rp + self.magnet_width + INTERP_MAGNET_OFFSET
        if abs(y) < (self.Lm + B_GRAD_STEP_SIZE) / 2.0 and \
                lens_radius_valid_inner <= minor_radius < lens_radius_valid_outer:
            return False
        elif abs(y) <= yUC_Line:
            return True
        else:
            return False

    def get_Valid_Indices_Internal(self, coords: np.ndarray, maxRotations: int) -> list[bool]:
        """Check if coords are not in the magnetic material region of the bender. Check up to maxRotations of the
        coords going counterclockwise about y axis by rotating coords"""
        R = Rot.from_rotvec([0, 0, -self.ucAng]).as_matrix()
        validIndices = []
        for [x, y, z] in coords:
            if self.is_Valid_In_Lens_Of_Bender(x, y, z):
                validIndices.append(True)
            else:
                loopStart, loopStop = 1, maxRotations + 1
                for i in range(loopStart, loopStop):
                    numRotations = (i + 1) // 2
                    x, y, z = (R ** numRotations) @ [x, y, z]
                    if self.is_Valid_In_Lens_Of_Bender(x, y, z):
                        validIndices.append(True)
                        break
                    elif i == loopStop - 1:
                        validIndices.append(False)
        return validIndices

    def generate_Segment_Field_Data(self) -> tuple[np.ndarray, ...]:
        """Internal repeating unit cell segment. This is modeled as a tilted portion with angle self.ucAng to the
        z axis, with its bottom face at z=0 alinged with the xy plane. In magnet frame coordinates"""
        x_min = (self.rb - self.ap) * cos(self.ucAng) - TINY_INTERP_STEP
        x_max = self.rb + self.ap + TINY_INTERP_STEP
        y_min = -TINY_INTERP_STEP
        y_max = tan(self.ucAng) * (self.rb + self.ap) + TINY_INTERP_STEP
        fieldCoords = self.make_Grid_Coords(x_min, x_max, y_min, y_max)

        validIndices = self.get_Valid_Indices_Internal(fieldCoords, 1)
        lens = self.build_Bender(False, (False, False))
        return self.compute_Valid_Field_Data(lens, fieldCoords, validIndices)

    def compute_Valid_Field_Vals(self, lens: _HalbachBenderFieldGenerator, fieldCoords: np.ndarray,
                                 validIndices: sequence) -> tuple[np.ndarray, np.ndarray]:
        BNormGradArr, BNormArr = np.zeros((len(fieldCoords), 3)) * np.nan, np.zeros(len(fieldCoords)) * np.nan
        BNormGradArr[validIndices], BNormArr[validIndices] = lens.B_norm_grad(fieldCoords[validIndices],
                                                                                 return_norm=True, use_approx=True)
        return BNormGradArr, BNormArr

    def compute_Valid_Field_Data(self, lens: _HalbachBenderFieldGenerator, fieldCoords: np.ndarray,
                                 validIndices: sequence) -> tuple[np.ndarray, ...]:
        BNormGradArr, BNormArr = self.compute_Valid_Field_Vals(lens, fieldCoords, validIndices)
        fieldDataUnshaped = np.column_stack((fieldCoords, BNormGradArr, BNormArr))
        return self.shape_field_data_3D(fieldDataUnshaped)

    def build_Bender(self, positiveAngleOnly: bool, use_half_cap_end: tuple[bool, bool], methodOfMoments: bool = True,
                     numLenses: int = None, use_mag_errors: bool = False):
        numLenses = self.numModelLenses if numLenses is None else numLenses
        benderFieldGenerator = _HalbachBenderFieldGenerator(self.rp, self.rb, self.ucAng, self.Lm, self.PTL.magnet_grade,
                                                            numLenses, use_half_cap_end,
                                                            use_method_of_moments=methodOfMoments,
                                                            use_pos_ang_mags_only=positiveAngleOnly,
                                                            use_solenoid_field=self.PTL.use_solenoid_field,
                                                            use_mag_errors=use_mag_errors,
                                                            magnet_width=self.magnet_width)
        benderFieldGenerator.rotate(Rot.from_rotvec([-np.pi / 2, 0, 0]))
        return benderFieldGenerator

    def in_Which_Section_Of_Bender(self, q_el: np.ndarray) -> str:
        """Find which section of the bender q_el is in. options are:
            - 'IN' refers to the westward cap. at some angle
            - 'OUT' refers to the eastern. input is aligned with y=0
            - 'ARC' in the bending arc between input and output caps
        Return 'NONE' if not inside the bender"""

        angle = full_arctan2(q_el)
        if 0 <= angle <= self.ang:
            return 'ARC'
        capNames = ['IN', 'OUT']
        for name in capNames:
            xCap, yCap = mirror_across_angle(q_el[0], q_el[1], self.ang / 2.0) if name == 'IN' else q_el[:2]
            if (self.rb - self.ap < xCap < self.rb + self.ap) and (0 > yCap > -self.Lcap):
                return name
        return 'NONE'

    def transform_element_coords_into_local_orbit_frame(self, q_el: np.ndarray) -> np.ndarray:

        whichSection = self.in_Which_Section_Of_Bender(q_el)
        if whichSection == 'ARC':
            phi = self.ang - full_arctan2(q_el)
            xo = sqrt(q_el[0] ** 2 + q_el[1] ** 2) - self.ro
            so = self.ro * phi + self.Lcap  # include the distance traveled throught the end cap
        elif whichSection == 'OUT':
            so = self.Lcap + self.ang * self.ro + (-q_el[1])
            xo = q_el[0] - self.ro
        elif whichSection == 'IN':
            xMirror, yMirror = mirror_across_angle(q_el[0], q_el[1], self.ang / 2.0)
            so = self.Lcap + yMirror
            xo = xMirror - self.ro
        else:
            raise ValueError
        qo = np.array([so, xo, q_el[2]])
        return qo

    def transform_element_momentum_into_local_orbit_frame(self, q_el: np.ndarray, p_el: np.ndarray) -> np.ndarray:
        """Overrides abstract method from Element class. Mildly tricky. Need to determine if the position is in
        one of the caps or the bending segment, then handle accordingly"""

        whichSection = self.in_Which_Section_Of_Bender(q_el)
        if whichSection == 'ARC':
            return super().transform_element_momentum_into_local_orbit_frame(q_el, p_el)
        elif whichSection == 'OUT':
            pso, pxo = -p_el[1], p_el[0]
        elif whichSection == 'IN':
            pxo, pso = mirror_across_angle(p_el[0], p_el[1], self.ang / 2.0)
        else:
            raise ValueError
        pOrbit = np.array([pso, pxo, q_el[-2]])
        return pOrbit
