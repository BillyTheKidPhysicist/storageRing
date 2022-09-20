import itertools
import os
import sys
import time
from math import isclose, isnan
from typing import Optional

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.spatial.transform import Rotation as Rot
from shapely.geometry import Point

from constants import SIMULATION_MAGNETON, DEFAULT_ATOM_SPEED, GRAVITATIONAL_ACCELERATION
from field_generators import HalbachBender as HalbachBender_FieldGenerator
from field_generators import HalbachLens
from helper_tools import is_close_all
from lattice_elements.elements import BenderIdeal, Drift, LensIdeal, CombinerIdeal, CombinerLensSim, \
    BenderSim, HalbachLensSim, Element
from lattice_elements.utilities import halbach_magnet_width
from particle import Particle
from particle_tracer import ParticleTracer
from particle_tracer_lattice import ParticleTracerLattice
from helper_tools import *


def absDif(x1, x2):
    return abs(abs(x1) - abs(x2))


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


class PTL_Dummy:
    """fake class to just pass some args along"""

    def __init__(self, field_dens_mult=1.0, include_mag_errors=False):
        self.field_dens_mult = field_dens_mult
        self.include_mag_errors = include_mag_errors
        self.jitter_amp = 0.0
        self.design_speed = DEFAULT_ATOM_SPEED
        self.use_solenoid_field = False
        self.magnet_grade = 'N52'
        self.use_standard_tube_OD = False
        self.include_misalignments = False


class ElementTestHelper:
    ElementBaseClass = type(Element)

    def __init__(self, elType: ElementBaseClass, particle0: Optional[Particle], qf0: Optional[np.ndarray],
                 pf0: Optional[np.ndarray], useShapelyTest: bool, testMisalignment: bool, testMagnetErrors: bool):
        self.elType = elType
        self.PTL = self.make_Latice()
        self.el = self.get_Element(self.PTL)
        self.particle0 = particle0
        self.qf0 = qf0
        self.pf0 = pf0
        self.coordTestRules = self.make_coordTestRules()
        self.useShapelyTest = useShapelyTest
        self.testMisalignment = testMisalignment
        self.testMagnetErrors = testMagnetErrors

    def make_coordTestRules(self) -> tuple[st.floats, st.floats, st.floats]:
        raise NotImplementedError

    def get_Element(self, PTL) -> Element:
        if any(self.elType == el for el in (Drift, LensIdeal, HalbachLensSim)):
            el = PTL.el_list[0]
        elif any(self.elType == el for el in (BenderIdeal, BenderSim, CombinerIdeal,
                                              CombinerLensSim)):
            el = PTL.el_list[1]
        else:
            raise ValueError
        assert type(el) == self.elType
        return el

    def run_Tests(self):
        tester = ElementTestRunner(self)
        tester.run_Tests()

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0) -> ParticleTracerLattice:
        raise NotImplementedError

    def convert_Test_Coord_To_El_Frame(self, x1, x2, x3) -> np.ndarray:
        """Simple cartesian coordinates work for most elements"""
        if any(self.elType == el for el in (Drift, LensIdeal, HalbachLensSim, CombinerIdeal, CombinerLensSim)):
            x, y, z = x1, x2, x3
        elif any(self.elType == el for el in (BenderIdeal, BenderSim)):
            r, theta, z = x1, x2, x3
            x, y = r * np.cos(theta), r * np.sin(theta)
        else:
            raise ValueError
        return np.asarray([x, y, z])


class DriftTestHelper(ElementTestHelper):

    def __init__(self):
        self.L, self.ap = .15432, .0392
        particle0 = Particle(qi=np.asarray([-1e-14, 1e-3, -2e-3]), pi=np.asarray([-201.0, 5.2, -3.8]))
        qf0 = np.array([-0.15376500000000048, 0.004978000000000016, -0.004909867602500013])
        pf0 = np.array([-201., 5.2, -3.8074970000000286])
        super().__init__(Drift, particle0, qf0, pf0, True, False, False)

    def make_coordTestRules(self):
        floatyz = st.floats(min_value=-1.5 * self.ap, max_value=1.5 * self.ap)
        floatx = st.floats(min_value=-self.L * .25, max_value=self.L * 1.25)
        return (floatx, floatyz, floatyz)

    def run_Tests(self):
        tester = ElementTestRunner(self)
        tester.run_Tests()
        self.test_Drift()

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0, include_mag_errors=magnetErrors)
        PTL.add_drift(self.L, self.ap)
        PTL.end_lattice(constrain=False)
        return PTL

    def test_Drift(self):
        # test that particle travels through drift region and ends where expected
        particleTracer = ParticleTracer(self.PTL)
        tol, h = 1e-5, 1e-7
        vx = -200.0
        delta_x = -self.L  # negative because particles start by moving along -x
        slopez_initial = .5 * self.ap / delta_x
        zi = -self.ap / 4.0
        zf = delta_x * slopez_initial + zi - .5 * GRAVITATIONAL_ACCELERATION * (delta_x / vx) ** 2
        slopey = .25 * self.ap / delta_x
        yi = self.ap / 4.0
        yf = delta_x * slopey + yi
        particle = Particle(qi=np.asarray([-1e-14, yi, zi]), pi=np.asarray([vx, slopey * vx, slopez_initial * vx]))
        particleTraced = particleTracer.trace(particle, h, 1.0, fast_mode=True)
        qfTrace, pfTrace = particleTraced.qf, particleTraced.pf
        slopeyTrace, slopezTrace = pfTrace[1] / pfTrace[0], pfTrace[2] / pfTrace[0]
        yfTrace, zfTrace = qfTrace[1], qfTrace[2]
        slopez_final = slopez_initial - GRAVITATIONAL_ACCELERATION * abs(delta_x / vx) / vx
        assert isclose(slopey, slopeyTrace, abs_tol=tol) and isclose(slopez_final, slopezTrace, abs_tol=tol)
        assert abs(yf - yfTrace) < tol and abs(zf - zfTrace) < tol


class _TiltedDriftTester(ElementTestHelper):

    def __init__(self, input_ang, outputAngle):
        self.L, self.ap = .15432, .0392
        self.input_ang = input_ang
        self.outputAngle = outputAngle
        super().__init__(Drift, None, None, None, True, False, False)

    def make_coordTestRules(self):
        floatyz = st.floats(min_value=-1.5 * self.ap, max_value=1.5 * self.ap)
        floatx = st.floats(min_value=-self.L * .25, max_value=self.L * 1.25)
        return (floatx, floatyz, floatyz)

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0, include_mag_errors=magnetErrors)
        PTL.add_drift(self.L, self.ap, input_tilt_angle=self.input_ang, output_tilt_angle=self.outputAngle)
        PTL.end_lattice(constrain=False)
        return PTL


class TiltedDriftTestHelper:
    def __init__(self):
        pass

    def run_Tests(self):
        anglesToTest = [-.1, 0.0, .1]
        for input_ang, outputAngle in itertools.product(anglesToTest, anglesToTest):
            _TiltedDriftTester(input_ang, outputAngle).run_Tests()


class HexapoleLensSimTestHelper(ElementTestHelper):

    def __init__(self):
        self.L = .1321432
        self.rp = .01874832
        self.magnet_width = .0254 * self.rp / .05
        particle0 = Particle(qi=np.asarray([-.01, 5e-3, -7.43e-3]), pi=np.asarray([-201.0, 5.0, -8.2343]))
        qf0 = np.array([-0.1312807574312542  ,  0.005178788282905027,-0.00816648689675629 ])
        pf0 = np.array([-201.14794370397757  ,   -3.7926521311036514,4.931167809763893 ])
        super().__init__(HalbachLensSim, particle0, qf0, pf0, True, True, True)

    def run_Tests(self):
        tester = ElementTestRunner(self)
        tester.run_Tests()
        self.test_Field_Deviations_And_Interpolation()

    def test_Field_Deviations_And_Interpolation(self):
        """Test that the interpolation of magnetic fields match with calculating the magnetic field at each point. Test
        this with magnet imperfections as well. Keep in mind that interpolation points inside magnet material are set to
        zero, so the interpolation may be poor near bore of magnet. This is done to avoid dealing with mistmatch
        between good field region of ideal and perturbation interpolation"""
        L = self.L * .5
        seed = int(time.time())
        tol = .025  # tolerance on the maximum value
        np.random.seed(seed)
        lensElement = HalbachLensSim(PTL_Dummy(field_dens_mult=2.0, include_mag_errors=True), (self.rp,), L,
                                     None, (self.magnet_width,))
        lensElement.fill_pre_constrained_parameters()
        lensElement.r1 = lensElement.r2 = lensElement.nb = lensElement.ne = np.zeros(3)
        lensElement.fill_post_constrained_parameters()
        lensElement.build_fast_field_helper()
        gridSpacing = lensElement.max_interp_radius() / lensElement.num_points_r
        np.random.seed(seed)
        numSlices = int(round(lensElement.Lm / lensElement.individualMagnetLength))
        lensFieldGenerator = HalbachLens(self.rp, self.magnet_width, lensElement.Lm, 'N52',
                                         use_method_of_moments=True, use_standard_mag_errors=True,
                                         num_disks=numSlices)
        rMax = .95 * lensElement.max_interp_radius()
        q_maxField = np.asarray([lensElement.L / 2, rMax / np.sqrt(2), rMax / np.sqrt(2)])
        FMax = np.linalg.norm(lensElement.force(q_maxField))
        VMax = lensElement.magnetic_potential(q_maxField)
        assert np.isnan(FMax) == False

        @given(*self.coordTestRules)
        @settings(max_examples=500, deadline=None)
        def check_Force_Agrees(x, y, z):
            qEl1 = np.asarray([x, y, z])
            minInterpPointsFromOrigin = 2  # if I test very near the origin, the differences can be much larger.
            maxAperture = .8 * lensElement.rp  # larger aperture require finer grid for comparison
            if lensElement.is_coord_inside(qEl1) == True and np.sqrt(y ** 2 + z ** 2) < maxAperture and \
                    y / gridSpacing > minInterpPointsFromOrigin and z / gridSpacing > minInterpPointsFromOrigin:
                Fx1, Fy1, Fz1 = lensElement.force(qEl1)
                V1 = lensElement.magnetic_potential(qEl1)
                x, y, z = -z, y, x - lensElement.L / 2  # shift coords for lens field generator. Points along z instead of x
                qEl2 = np.asarray([x, y, z])
                (BGradx, BGrady, BGradz), B_norm = lensFieldGenerator.B_norm_grad(qEl2, return_norm=True)
                Fx2, Fy2, Fz2 = -np.array([BGradx, BGrady, BGradz]) * SIMULATION_MAGNETON
                Fx2, Fz2 = Fz2, -Fx2
                V2 = B_norm * SIMULATION_MAGNETON
                for Fi1, Fi2 in zip([Fx1, Fy1, Fz1], [Fx2, Fy2, Fz2]):
                    assert isclose(Fi1, Fi2, abs_tol=tol * FMax)
                assert isclose(V1, V2, abs_tol=tol * VMax)

        check_Force_Agrees()

    def make_coordTestRules(self):
        floatyz = st.floats(min_value=-1.5 * self.rp, max_value=1.5 * self.rp)
        floatx = st.floats(min_value=-self.L / 10.0, max_value=self.L * 1.25)
        return floatx, floatyz, floatyz

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0, include_mag_errors=magnetErrors)
        PTL.add_halbach_lens_sim(self.rp, self.L, ap=.9 * self.rp)
        PTL.end_lattice(constrain=False)
        return PTL


class LensIdealTestHelper(ElementTestHelper):

    def __init__(self):
        self.L, self.rp, self.Bp = .51824792317429, .024382758923, 1.832484234
        particle0 = Particle(qi=np.asarray([-.01, 1e-3, -2e-3]), pi=np.asarray([-201.0, 8.2, -6.8]))
        qf0 = np.array([-5.1752499999999624e-01, -1.5840926495604049e-03, 4.1005739568729166e-04])
        pf0 = np.array([-201., 7.735215296352372, -8.061999426907333])
        super().__init__(LensIdeal, particle0, qf0, pf0, True, True, False)

    def make_coordTestRules(self):
        # test generic conditions of the element
        floatyz = st.floats(min_value=-1.5 * self.rp, max_value=1.25 * self.rp)
        floatx = st.floats(min_value=-self.L * .25, max_value=self.L * 1.25)
        return floatx, floatyz, floatyz

    def run_Tests(self):
        tester = ElementTestRunner(self)
        tester.run_Tests()
        self.theory_Test()

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0, include_mag_errors=magnetErrors)
        PTL.add_lens_ideal(self.L, self.Bp, self.rp)
        PTL.end_lattice(constrain=False)
        return PTL

    def theory_Test(self):
        tol = 1e-2  # test to 1% accuracy
        # does the particle behave as theory predicts? It should be very close because it's ideal
        particleTracer = ParticleTracer(self.PTL)
        particle = Particle(qi=np.asarray([-1e-14, self.rp / 2.0, 0.]))
        particle = particleTracer.trace(particle, 1e-6, 1.0)
        yi, yf, pyf = particle.qi[1], particle.qf[1], particle.pf[1]
        yRMS, pyRMS = np.std(particle.q_vals[:, 1]), np.std(particle.p_vals[:, 1])
        K = 2 * SIMULATION_MAGNETON * self.Bp / (particle.pi[0] ** 2 * self.rp ** 2)
        phi = np.sqrt(K) * self.L
        yfTheory, pyfTheory = yi * np.cos(phi), -abs(particle.pi[0]) * yi * np.sin(phi) * np.sqrt(K)
        assert abs(yf - yfTheory) < tol * yi and abs(pyf - pyfTheory) < tol * pyRMS


class BenderIdealTestHelper(ElementTestHelper):

    def __init__(self):
        self.ang = np.pi / 2.0
        self.Bp = .8934752374
        self.rb = .94830284532
        self.rp = .01853423
        particle0 = Particle(qi=np.asarray([-.01, 1e-3, -2e-3]), pi=np.asarray([-201.0, 5.2, -6.8]))
        qf0 = np.array([-9.639381030969734e-01, 9.576855890781706e-01, 8.143718656072168e-05])
        pf0 = np.array([-4.741346774755123, 200.6975241302656, 7.920531222158079])
        super().__init__(BenderIdeal, particle0, qf0, pf0, False, False, False)

    def make_coordTestRules(self):
        # test generic conditions of the element
        floatr = st.floats(min_value=self.rb - self.rp * 2, max_value=self.rb + self.rp * 2)
        floatphi = st.floats(min_value=-self.ang * .1, max_value=1.1 * self.ang)
        floatz = st.floats(min_value=-self.rp / 2, max_value=self.rp / 2)
        return (floatr, floatphi, floatz)

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0, include_mag_errors=magnetErrors)
        PTL.add_drift(5e-3)
        PTL.add_bender_ideal(self.ang, self.Bp, self.rb, self.rp)
        PTL.end_lattice(constrain=False)
        return PTL


class HexapoleSegmentedBenderTestHelper(ElementTestHelper):

    def __init__(self):
        self.Lm = .0254
        self.rp = .014832
        self.num_lenses = 150
        self.rb = 1.02324
        self.ang = self.num_lenses * self.Lm / self.rb
        particle0 = Particle(qi=np.asarray([-.01, 1e-3, -2e-3]), pi=np.asarray([-201.0, 1.0, -.5]))
        qf0 = np.array([6.2559654843256451e-01, 1.8268225133423863e+00,8.3597785201163064e-04])
        pf0 = np.array([ 156.4668861635784 , -126.11406318501261,   -4.06292085256291])
        super().__init__(BenderSim, particle0, qf0, pf0, False, False, False)

    def make_coordTestRules(self):
        # test generic conditions of the element
        floatr = st.floats(min_value=self.rb - self.rp * 2, max_value=self.rb + self.rp * 2)
        floatphi = st.floats(min_value=-self.ang * .1, max_value=1.1 * self.ang)
        floatz = st.floats(min_value=-self.rp / 2, max_value=self.rp / 2)
        return floatr, floatphi, floatz

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0)
        PTL.add_drift(5e-3)
        PTL.add_segmented_halbach_bender(self.Lm, self.rp, self.num_lenses, self.rb)
        PTL.end_lattice(constrain=False)
        return PTL

    def run_Tests(self):
        tester = ElementTestRunner(self)
        tester.run_Tests()


class CombinerIdealTestHelper(ElementTestHelper):

    def __init__(self):
        self.Lm = .218734921
        self.ap = .014832794
        particle0 = Particle(qi=np.asarray([-.01, 1e-3, -2e-3]), pi=np.asarray([-201.0, 0.0, 0.0]))
        qf0 = np.array([-0.22328507641728806, 0.009988699191880843, -0.002341300261489336])
        pf0 = np.array([-199.53548090705544, 16.878732582305176, -0.6778345463513105])
        super().__init__(CombinerIdeal, particle0, qf0, pf0, True, False, False)

    def make_coordTestRules(self):
        # test generic conditions of the element
        floatx = st.floats(min_value=-self.Lm * 0.1, max_value=1.25 * self.Lm)
        floaty = st.floats(min_value=-2 * self.ap, max_value=2 * self.ap)
        floatz = st.floats(min_value=-2 * self.ap / 2, max_value=2 * self.ap / 2)  # z ap is half of y
        return floatx, floaty, floatz

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0, include_mag_errors=magnetErrors)
        PTL.add_drift(5e-3)
        PTL.add_combiner_ideal(Lm=self.Lm, ap=self.ap)
        PTL.end_lattice(constrain=False)
        return PTL


class CombinerHalbachTestHelper(ElementTestHelper):

    def __init__(self):
        self.Lm = .1453423
        self.rp = .0223749
        particle0 = Particle(qi=np.asarray([-.01, 5e-3, -3.43e-3]), pi=np.asarray([-201.0, 5.0, -3.2343]))
        qf0 = np.array([-2.5204629069268808e-01, 7.0864219592001437e-03, -2.1422344432320994e-04])
        pf0 = np.array([-200.93616612463057, 1.3732481410737705, 7.7133453110329455])
        super().__init__(CombinerLensSim, particle0, qf0, pf0, True, False, False)

    def make_coordTestRules(self):
        # test generic conditions of the element
        floatyz = st.floats(min_value=-3 * self.rp, max_value=3 * self.rp)
        floatx = st.floats(min_value=-self.el.L / 10.0, max_value=self.el.L * 1.1)
        return floatx, floatyz, floatyz

    def make_Latice(self, magnetErrors=False, jitter_amp=0.0):
        PTL = ParticleTracerLattice(design_speed=200.0, include_mag_errors=magnetErrors)
        PTL.add_drift(5e-3)
        PTL.add_combiner_sim_lens(self.Lm, self.rp, ap=None, layers=2)
        PTL.end_lattice(constrain=False)
        return PTL


class ElementTestRunner:

    def __init__(self, ElementTestHelper: ElementTestHelper):
        self.elTestHelper = ElementTestHelper
        self.timeStepTracing = 5e-6
        self.numericTol = 1e-12

    def run_Tests(self):
        # IMPROVEMENT: reenable these tests
        self.test_Tracing()
        self.test_Coord_Consistency()
        self.test_Coord_Conversions()
        self.test_Magnet_Imperfections()
        # self.test_Imperfections_Tracing()
        # self.test_Misalignment1()

    def test_Tracing(self):
        """Test that particle tracing yields the same results"""
        particleTracedList = self.trace_Different_Conditions(self.elTestHelper.PTL)
        self.assert_Particle_List_Is_Expected(particleTracedList, self.elTestHelper.qf0, self.elTestHelper.pf0)

    def test_Coord_Consistency(self):
        """Test that force returns nan when the coord is outside the element. test this agrees with shapely geometry
        when applicable"""
        isInsideList = []
        el = self.elTestHelper.el

        @given(*self.elTestHelper.coordTestRules)
        @settings(max_examples=2_000, deadline=None)
        def is_Inside_Consistency(x1: float, x2: float, x3: float):
            q_el = self.elTestHelper.convert_Test_Coord_To_El_Frame(x1, x2, x3)
            F = el.force(q_el)
            V = el.magnetic_potential(q_el)
            qEl_2D = q_el[:2].copy()
            if self.elTestHelper.useShapelyTest:
                self.test_Coord2D_Against_Shapely(qEl_2D)
            isInsideFull3D = el.is_coord_inside(q_el)
            isInsideList.append(isInsideFull3D)
            # todo: clean this up
            if isInsideFull3D == False:
                assert isnan(F[0]) == True and isnan(V) == True, str(F) + ',' + str(V)
            else:
                assert np.any(np.isnan(F)) == False and np.isnan(V) == False, str(F) + ',' + str(V) + ',' + str(
                    isInsideFull3D)

        is_Inside_Consistency()
        numTrue = sum(isInsideList)
        assert numTrue > 50  # at least 50 true seems reasonable

    def test_Coord_Conversions(self):
        # check that coordinate conversions work
        tol = 1e-12
        el = self.elTestHelper.el

        @given(*self.elTestHelper.coordTestRules)
        @settings(max_examples=1_000, deadline=None)
        def convert_Invert_Consistency(x1, x2, x3):
            coordEl0 = self.elTestHelper.convert_Test_Coord_To_El_Frame(x1, x2, x3)
            coordLab = el.transform_element_coords_into_lab_frame(coordEl0)
            coordsEl = el.transform_lab_coords_into_element_frame(coordLab)
            assert np.all(np.abs(coordEl0 - coordsEl) < tol)
            vecEl0 = coordEl0
            vecLab = el.transform_lab_frame_vector_into_element_frame(vecEl0)
            vecEl = el.transform_element_frame_vector_into_lab_frame(vecLab)
            assert np.all(np.abs(vecEl0 - vecEl) < tol)

        convert_Invert_Consistency()

    def test_Magnet_Imperfections(self):
        """Test that there is no symmetry. Misalinging should break the symmetry. This is sort of as silly test"""
        if self.elTestHelper.testMagnetErrors == True:
            PTL = self.elTestHelper.make_Latice(magnetErrors=True)
            el = self.elTestHelper.get_Element(PTL)

            @given(*self.elTestHelper.coordTestRules)
            @settings(max_examples=300, deadline=None)
            def test_Magnetic_Imperfection_Field_Symmetry(x1: float, x2: float, x3: float):
                coord = self.elTestHelper.convert_Test_Coord_To_El_Frame(x1, x2, x3)
                if any(isclose(x, 0.0, abs_tol=1e-3) for x in [x1, x2, x3]):  # or el.is_coord_inside(coord):
                    return
                else:
                    F0 = np.abs(el.force(coord))
                    for y in [coord[1], -coord[1]]:
                        z = -coord[2]
                        if np.isnan(F0[0]) == False and y != 0 and z != 0:
                            FSym = np.abs(el.force(np.array([coord[0], y, z])))
                            np.set_printoptions(precision=100)
                            assert is_close_all(F0, FSym, 1e-10) == False  # assert there is no symmetry

            test_Magnetic_Imperfection_Field_Symmetry()

    def test_Imperfections_Tracing(self):
        """test that misalignment and errors change results"""
        if self.elTestHelper.testMisalignment == True or self.elTestHelper.testMagnetErrors == True:
            particleList = self.trace_Different_Conditions(self.elTestHelper.PTL)
            qf0Temp, pf0Temp = particleList[0].qf, particleList[0].pf

            def wrapper(magnetError, jitter_amp):
                PTL = self.elTestHelper.make_Latice(magnetErrors=magnetError)
                el = self.elTestHelper.get_Element(PTL)
                if jitter_amp != 0.0:
                    el.perturb_element(.1 * jitter_amp, .1 * jitter_amp, .1 * jitter_amp / el.L, .1 * jitter_amp / el.L)
                particleList = self.trace_Different_Conditions(PTL)
                with pytest.raises(ValueError) as excInfo:
                    # it's possible this won't trigger if the magnet or misalignment errors are really small
                    self.assert_Particle_List_Is_Expected(particleList, qf0Temp, pf0Temp, absTol=1e-6)
                assert str(excInfo.value) == "particle test mismatch"

            if self.elTestHelper.testMisalignment == True: wrapper(False, 1e-3)
            if self.elTestHelper.testMagnetErrors: wrapper(True, 0.0)

    def test_Misalignment1(self):
        """Test that misaligning element does not results in any errors or conflicts with other methods"""

        def _test_Miaslignment(jitter_amp):
            PTL = self.elTestHelper.make_Latice()
            el = self.elTestHelper.get_Element(PTL)
            jitterArr = 2 * (np.random.random_sample(4) - .5) * jitter_amp
            jitterArr[2:] *= 1 / el.L  # rotation component
            blockPrint()
            el.perturb_element(*jitterArr)
            enablePrint()

            @given(*self.elTestHelper.coordTestRules)
            @settings(max_examples=500, deadline=None)
            def test_Geometry_And_Fields(x1: float, x2: float, x3: float):
                """Test that the force and magnetic fields functions return nan when the particle is outside the vacuum
                 and that it agrees with is_Coords_Inside"""
                coord = self.elTestHelper.convert_Test_Coord_To_El_Frame(x1, x2, x3)
                F = el.force(coord)
                V = el.magnetic_potential(coord)
                isInside = el.is_coord_inside(coord)
                assert (not isnan(F[0])) == isInside and (not isnan(V)) == isInside

            test_Geometry_And_Fields()

        if self.elTestHelper.testMisalignment == True:
            jitterAmpArr = np.linspace(0.0, self.elTestHelper.el.rp / 10.0, 5)
            np.random.seed(42)
            [_test_Miaslignment(jitter) for jitter in jitterAmpArr]
            np.random.seed(int(time.time()))

    def trace_Different_Conditions(self, PTL):
        PT = ParticleTracer(PTL)
        particleList = []
        for use_fast_mode in (True, False):
            if self.elTestHelper.particle0 is not None:
                particleList.append(
                    PT.trace(self.elTestHelper.particle0.copy(), self.timeStepTracing, 1.0, fast_mode=use_fast_mode))
        return particleList

    def assert_Particle_List_Is_Expected(self, particleList: list[Particle], qf0: np.ndarray, pf0: np.ndarray,
                                         absTol: float = 1e-14) -> None:
        for particle in particleList:
            qf, pf = particle.qf, particle.pf
            np.set_printoptions(precision=100)
            if is_close_all(qf, qf0, absTol) == False or is_close_all(pf, pf0, absTol) == False:
                print(repr(qf), repr(pf))
                print(repr(qf0), repr(pf0))
                raise ValueError("particle test mismatch")

    def is_Inside_Shapely(self, qEl_2D):
        """Check with Shapely library that point resides in 2D footprint of element. It's possible that the point may
        fall just on the edge of the object, so return result with and without small padding"""
        el = self.elTestHelper.el
        qLab_2D = el.transform_element_coords_into_lab_frame(np.append(qEl_2D, 0))
        isInsideUnpadded = el.SO.contains(Point(qLab_2D))
        SO_Padded = el.SO.buffer(1e-9)  # add padding to avoid issues of point right on edge
        isInsidePadded = SO_Padded.contains(Point(qLab_2D))
        SO_NegPadded = el.SO.buffer(-1e-9)  # add padding to avoid issues of point right on edge
        isInsideNegPadded = SO_NegPadded.contains(Point(qLab_2D))
        return isInsideUnpadded, isInsidePadded, isInsideNegPadded

    def test_Coord2D_Against_Shapely(self, qEl_2D):
        isInside2DShapely, isInside2DShapelyPadded, isInside2DShapelyNegPadded = self.is_Inside_Shapely(qEl_2D)
        if isInside2DShapely == isInside2DShapelyPadded and isInside2DShapely == isInside2DShapelyNegPadded:
            # if consistent for both its not a weird situation of being
            # right on the edge, so go ahead and check
            isInside_Fast = self.elTestHelper.el.is_coord_inside(np.append(qEl_2D, 0))
            if not isInside2DShapely == isInside_Fast:
                print(qEl_2D)
                print(isInside2DShapely, isInside2DShapelyPadded, isInside2DShapelyNegPadded, isInside_Fast)
                el = self.elTestHelper.el
                qLab_2D = el.transform_element_coords_into_lab_frame(np.append(qEl_2D, 0))
                isInsideUnpadded = el.SO.contains(Point(qLab_2D))
                SO_Padded = el.SO.buffer(2e-9)  # add padding to avoid issues of point right on edge
                isInsidePadded = SO_Padded.contains(Point(qLab_2D))
                SO_NegPadded = el.SO.buffer(-2e-9)  # add padding to avoid issues of point right on edge
                isInsideNegPadded = SO_NegPadded.contains(Point(qLab_2D))
                import matplotlib.pyplot as plt
                plt.scatter(qLab_2D[0], qLab_2D[1])
                plt.plot(*SO_Padded.exterior.xy)
                plt.plot(*SO_NegPadded.exterior.xy)
                plt.show()
            assert isInside2DShapely == isInside_Fast  # this can be falsely triggered by circles in shapely!!


def test_Elements(parallel=True):
    testersToRun = [DriftTestHelper,
                    TiltedDriftTestHelper,
                    LensIdealTestHelper,
                    BenderIdealTestHelper,
                    CombinerIdealTestHelper,
                    CombinerHalbachTestHelper,
                    HexapoleLensSimTestHelper,
                    HexapoleSegmentedBenderTestHelper]

    def run_Tester(tester):
        tester().run_Tests()

    processes = -1 if parallel == True else 1
    parallel_evaluate(run_Tester, testersToRun, processes=processes)
