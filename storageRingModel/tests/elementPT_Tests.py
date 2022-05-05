import time

import numpy as np

from helperTools import *
import sys
import pytest
from elementPT import Element,BenderIdeal,Drift,LensIdeal,CombinerIdeal,CombinerHalbachLensSim,HalbachBenderSimSegmented,HalbachLensSim
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleTracerClass import ParticleTracer
from ParticleClass import Particle
from hypothesis import given,settings,strategies as st
from constants import SIMULATION_MAGNETON, DEFAULT_ATOM_SPEED,GRAVITATIONAL_ACCELERATION
from shapely.geometry import Point
from math import isclose
from HalbachLensClass import HalbachLens,SegmentedBenderHalbach

def absDif(x1,x2):
    return abs(abs(x1)-abs(x2))

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

class PTL_Dummy:
    """fake class to just pass some args along"""
    def __init__(self,fieldDensityMultiplier=1.0):
        self.fieldDensityMultiplier = fieldDensityMultiplier
        self.jitterAmp = 0.0
        self.v0Nominal=DEFAULT_ATOM_SPEED


class ElementTestHelper:

    ElementBaseClass=type(Element)
    def __init__(self,elType: ElementBaseClass,particle0: Particle,qf0: np.ndarray,pf0: np.ndarray,useShapelyTest: bool,
                 testMisalignment: bool,testMagnetErrors: bool):
        self.elType=elType
        self.PTL=self.make_Latice()
        self.el=self.get_Element(self.PTL)
        self.particle0=particle0
        self.qf0 = qf0
        self.pf0 = pf0
        self.coordTestRules=self.make_coordTestRules()
        self.useShapelyTest=useShapelyTest
        self.testMisalignment=testMisalignment
        self.testMagnetErrors=testMagnetErrors

    def make_coordTestRules(self)-> tuple[st.floats,st.floats,st.floats]:
        raise NotImplementedError

    def get_Element(self,PTL)-> Element:
        if any(self.elType == el for el in (Drift, LensIdeal, HalbachLensSim)):
            el = PTL.elList[0]
        elif any(self.elType == el for el in (BenderIdeal, HalbachBenderSimSegmented,CombinerIdeal,
                                              CombinerHalbachLensSim)):
            el = PTL.elList[1]
        else: raise ValueError
        assert type(el) == self.elType
        return el

    def run_Tests(self):
        tester=ElementTestRunner(self)
        tester.run_Tests()

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0)->ParticleTracerLattice:
        raise NotImplementedError

    def convert_Test_Coord_To_El_Frame(self,x1,x2,x3)-> np.ndarray:
        """Simple cartesian coordinates work for most elements"""
        if any(self.elType==el for el in (Drift,LensIdeal,HalbachLensSim,CombinerIdeal,CombinerHalbachLensSim)):
            x,y,z=x1,x2,x3
        elif any(self.elType==el for el in (BenderIdeal,HalbachBenderSimSegmented)):
            r, theta, z = x1, x2, x3
            x, y = r * np.cos(theta), r * np.sin(theta)
        else: raise ValueError
        return np.asarray([x, y, z])


class DriftTestHelper(ElementTestHelper):

    def __init__(self):
        self.L, self.ap = .15432, .0392
        particle0=Particle(qi=np.asarray([0.0,1e-3,-2e-3]),pi=np.asarray([-201.0,5.2,-3.8]))
        qf0 = np.array([-0.15376500000000048 ,  0.004978000000000016,-0.004909867602500013])
        pf0 = np.array([-201.                ,    5.2               ,-3.8074970000000286])
        super().__init__(Drift,particle0,qf0,pf0,True,False,False)

    def make_coordTestRules(self):
        floatyz = st.floats(min_value=-1.5 * self.ap, max_value=1.5 * self.ap)
        floatx = st.floats(min_value=-self.L * .25, max_value=self.L * 1.25)
        return (floatx, floatyz, floatyz)

    def run_Tests(self):
        tester=ElementTestRunner(self)
        tester.run_Tests()
        self.test_Drift()

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0):
        PTL=ParticleTracerLattice(v0Nominal=200.0,standardMagnetErrors=magnetErrors,jitterAmp=jitterAmp)
        PTL.add_Drift(self.L,self.ap)
        PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        return PTL

    def test_Drift(self):
        # test that particle travels through drift region and ends where expected
        particleTracer = ParticleTracer(self.PTL)
        tol, h = 1e-5, 1e-7
        vx = -200.0
        deltaX = -self.L  # negative because particles start by moving along -x
        slopez_initial = .5 * self.ap / deltaX
        zi = -self.ap / 4.0
        zf = deltaX * slopez_initial + zi-.5*GRAVITATIONAL_ACCELERATION*(deltaX/vx)**2
        slopey = .25 * self.ap / deltaX
        yi = self.ap / 4.0
        yf = deltaX * slopey + yi
        particle = Particle(qi=np.asarray([0.0, yi, zi]), pi=np.asarray([vx, slopey * vx, slopez_initial * vx]))
        particleTraced = particleTracer.trace(particle, h, 1.0, fastMode=True)
        qfTrace, pfTrace = particleTraced.qf, particleTraced.pf
        slopeyTrace, slopezTrace = pfTrace[1] / pfTrace[0], pfTrace[2] / pfTrace[0]
        yfTrace, zfTrace = qfTrace[1], qfTrace[2]
        slopez_final=slopez_initial-GRAVITATIONAL_ACCELERATION*abs(deltaX/vx)/vx
        assert isclose(slopey, slopeyTrace,abs_tol=tol) and isclose(slopez_final, slopezTrace,abs_tol=tol)
        assert abs(yf - yfTrace) < tol and abs(zf - zfTrace) < tol


class HexapoleLensSimTestHelper(ElementTestHelper):

    def __init__(self):
        self.L=.1321432
        self.rp=.01874832
        self.magnetWidth=.0254*self.rp/.05
        particle0=Particle(qi=np.asarray([-.01,5e-3,-7.43e-3]),pi=np.asarray([-201.0,5.0,-8.2343]))
        qf0=np.array([-0.13132158088021298 ,  0.005449978510930973,-0.008573858031929796])
        pf0=np.array([-201.18865808223083  ,   -2.9780591461212036,3.695121970829412 ])
        super().__init__(HalbachLensSim,particle0,qf0,pf0,True,True,True)

    def run_Tests(self):
        tester=ElementTestRunner(self)
        tester.run_Tests()
        self.test_Field_Deviations_And_Interpolation()

    def test_Field_Deviations_And_Interpolation(self):
        """Test that the interpolation of magnetic fields match with calculating the magnetic field at each point. Test
        this with magnet imperfections as well. Keep in mind that interpolation points inside magnet material are set to
        zero, so the interpolation may be poor near bore of magnet. This is done to avoid dealing with mistmatch
        between good field region of ideal and perturbation interpolation"""
        L=self.L*.5
        seed = int(time.time())
        tol=.025 #tolerance on the maximum value
        magnetErrors = True
        np.random.seed(seed)
        lensElement = HalbachLensSim(PTL_Dummy(fieldDensityMultiplier=2.0), (self.rp,), L, None,
                                     (self.magnetWidth,), useStandardMagErrors=magnetErrors)
        gridSpacing = lensElement.apMaxGoodField / lensElement.numGridPointsXY
        np.random.seed(seed)
        numSlices = None if not magnetErrors else int(round(lensElement.Lm / lensElement.individualMagnetLength))
        lensFieldGenerator = HalbachLens(self.rp, self.magnetWidth, lensElement.Lm,
                                    applyMethodOfMoments=True, useStandardMagErrors=magnetErrors,numSlices=numSlices)
        rMax=.95 * lensElement.apMaxGoodField
        qMaxField = np.asarray([lensElement.L / 2, rMax / np.sqrt(2), rMax / np.sqrt(2)])
        FMax = np.linalg.norm(lensElement.force(qMaxField))
        VMax = lensElement.magnetic_Potential(qMaxField)
        assert np.isnan(FMax) == False
        @given(*self.coordTestRules)
        @settings(max_examples=500, deadline=None)
        def check_Force_Agrees(x, y, z):
            qEl1 = np.asarray([x, y, z])
            minInterpPointsFromOrigin = 2  # if I test very near the origin, the differences can be much larger.
            maxAperture = .8 * lensElement.rp  # larger aperture require finer grid for comparison
            if lensElement.is_Coord_Inside(qEl1) == True and np.sqrt(y ** 2 + z ** 2) < maxAperture and \
                    y / gridSpacing > minInterpPointsFromOrigin and z / gridSpacing > minInterpPointsFromOrigin:
                Fx1, Fy1, Fz1 = lensElement.force(qEl1)
                V1 = lensElement.magnetic_Potential(qEl1)
                x, y, z = -z, y, x - lensElement.L / 2 #shift coords for lens field generator. Points along z instead of x
                qEl2 = np.asarray([x, y, z])
                (BGradx, BGrady, BGradz), BNorm = lensFieldGenerator.BNorm_Gradient(qEl2, returnNorm=True)
                Fx2, Fy2, Fz2 = -np.array([BGradx, BGrady, BGradz]) * SIMULATION_MAGNETON
                Fx2, Fz2 = Fz2, -Fx2
                V2 = BNorm * SIMULATION_MAGNETON
                for Fi1, Fi2 in zip([Fx1, Fy1, Fz1], [Fx2, Fy2, Fz2]):
                    assert isclose(Fi1, Fi2, abs_tol=tol * FMax)
                assert isclose(V1, V2, abs_tol=tol * VMax)
        check_Force_Agrees()

    def make_coordTestRules(self):
        floatyz = st.floats(min_value=-1.5 * self.rp, max_value=1.5 * self.rp)
        floatx = st.floats(min_value=-self.L / 10.0, max_value=self.L * 1.25)
        return floatx,floatyz,floatyz

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0):
        PTL = ParticleTracerLattice(v0Nominal=200.0,standardMagnetErrors=magnetErrors,jitterAmp=jitterAmp)
        PTL.add_Halbach_Lens_Sim(self.rp, self.L)
        PTL.end_Lattice(constrain=False, surpressWarning=True, enforceClosedLattice=False)
        return PTL

class LensIdealTestHelper(ElementTestHelper):

    def __init__(self):
        self.L, self.rp, self.Bp = .51824792317429, .024382758923, 1.832484234
        particle0=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,8.2,-6.8]))
        qf0 = np.array([-5.1752499999999624e-01, -1.5840926495604049e-03, 4.1005739568729166e-04])
        pf0 = np.array([-201.               ,    7.735215296352372,   -8.061999426907333])
        super().__init__(LensIdeal,particle0,qf0,pf0,True,True,False)

    def make_coordTestRules(self):
        #test generic conditions of the element
        floatyz = st.floats(min_value=-1.5 * self.rp, max_value=1.25 * self.rp)
        floatx = st.floats(min_value=-self.L*.25, max_value=self.L * 1.25)
        return floatx,floatyz,floatyz

    def run_Tests(self):
        tester=ElementTestRunner(self)
        tester.run_Tests()
        self.theory_Test()

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0):
        PTL=ParticleTracerLattice(v0Nominal=200.0,standardMagnetErrors=magnetErrors,jitterAmp=jitterAmp)
        PTL.add_Lens_Ideal(self.L,self.Bp,self.rp)
        PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        return PTL

    def theory_Test(self):
        tol=1e-2 #test to 1% accuracy
        #does the particle behave as theory predicts? It should be very close because it's ideal
        particleTracer=ParticleTracer(self.PTL)
        particle=Particle(qi=np.asarray([0.0,self.rp/2.0,0.]))
        particle=particleTracer.trace(particle,1e-6,1.0)
        yi,yf,pyf=particle.qi[1],particle.qf[1],particle.pf[1]
        yRMS,pyRMS=np.std(particle.qArr[:,1]),np.std(particle.pArr[:,1])
        K = 2 * SIMULATION_MAGNETON * self.Bp / (particle.pi[0] ** 2 * self.rp ** 2)
        phi = np.sqrt(K) * self.L
        yfTheory,pyfTheory=yi*np.cos(phi),-abs(particle.pi[0])*yi*np.sin(phi)*np.sqrt(K)
        assert abs(yf-yfTheory)<tol*yi and abs(pyf-pyfTheory)<tol*pyRMS


class BenderIdealTestHelper(ElementTestHelper):

    def __init__(self):
        self.ang = np.pi / 2.0
        self.Bp = .8934752374
        self.rb = .94830284532
        self.rp = .01853423
        particle0=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,5.2,-6.8]))
        qf0 = np.array([-9.639381030969734e-01,  9.576855890781706e-01,8.143718656072168e-05])
        pf0 = np.array([ -4.741346774755123, 200.6975241302656  ,   7.920531222158079])
        super().__init__(BenderIdeal,particle0,qf0,pf0,False,False,False)

    def make_coordTestRules(self):
        #test generic conditions of the element
        floatr = st.floats(min_value=self.rb-self.rp*2, max_value=self.rb+self.rp*2)
        floatphi=st.floats(min_value=-self.ang*.1, max_value=1.1*self.ang)
        floatz=st.floats(min_value=-self.rp/2,max_value=self.rp/2)
        return (floatr,floatphi,floatz)

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0):
        PTL=ParticleTracerLattice(v0Nominal=200.0,standardMagnetErrors=magnetErrors,jitterAmp=jitterAmp)
        PTL.add_Drift(5e-3)
        PTL.add_Bender_Ideal(self.ang,self.Bp,self.rb,self.rp)
        PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        return PTL


class HexapoleSegmentedBenderTestHelper(ElementTestHelper):

    def __init__(self):
        self.Lm=.0254
        self.rp=.014832
        self.numMagnets=150
        self.rb=1.02324
        self.ang=self.numMagnets*self.Lm/self.rb
        particle0=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,1.0,-.5]))
        qf0 = np.array([6.2725608062667604e-01, 1.8270221569898277e+00,1.1769975335643556e-03])
        pf0 = np.array([ 159.33655611175072 , -122.28076734138395 ,    4.535124132136746])
        super().__init__(HalbachBenderSimSegmented,particle0,qf0,pf0,False,False,False)

    def make_coordTestRules(self):
        #test generic conditions of the element
        floatr = st.floats(min_value=self.rb-self.rp*2, max_value=self.rb+self.rp*2)
        floatphi=st.floats(min_value=-self.ang*.1, max_value=1.1*self.ang)
        floatz=st.floats(min_value=-self.rp/2,max_value=self.rp/2)
        return floatr,floatphi,floatz

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0):
        PTL=ParticleTracerLattice(v0Nominal=200.0)
        PTL.add_Drift(5e-3)
        PTL.add_Halbach_Bender_Sim_Segmented(self.Lm,self.rp,self.numMagnets,self.rb)
        PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        return PTL

    def run_Tests(self):
        tester=ElementTestRunner(self)
        tester.run_Tests()
        self.test_Perturbation_Field()

    def test_Perturbation_Field(self):
        """Test that perturbation field from bender agrees with directly calculating the perturbation field values.
        This also implictely tests the accuracy of the unit cell model becuase the force is calculated assuming the
        bender"""

        numMagnets=20
        PTL=PTL_Dummy()
        np.random.seed(42)
        elDeviation = HalbachBenderSimSegmented(PTL, self.Lm, self.rp, numMagnets, self.rb, 1e-3, 1.0, True)
        elPerfect = HalbachBenderSimSegmented(PTL, self.Lm, self.rp, numMagnets, self.rb, 1e-3, 1.0, False)
        Ls = 2 * elPerfect.Lcap + elPerfect.ang * elPerfect.rb
        coordsCenter, coordsCartesian = elPerfect.make_Perturbation_Data_Coords()
        np.random.seed(42)
        lensIdeal = SegmentedBenderHalbach(elPerfect.rp, elPerfect.rb, elPerfect.ucAng, elPerfect.Lm,
                                           numLenses=numMagnets, applyMethodOfMoments=False,
                                           positiveAngleMagnetsOnly=True, useMagnetError=False)
        np.random.seed(42)
        lensDeviated = SegmentedBenderHalbach(elDeviation.rp, elDeviation.rb, elDeviation.ucAng, elDeviation.Lm,
                                              numLenses=numMagnets, applyMethodOfMoments=False,
                                              positiveAngleMagnetsOnly=True, useMagnetError=True)
        testStrategy = st.integers(min_value=0, max_value=len(coordsCartesian)-1)
        @given(testStrategy)
        @settings(max_examples=100,deadline=None)
        def check_Field_Perturbation(index):
            x, y, z = coordsCartesian[index]
            s,xc,yc=coordsCenter[index]
            if np.sqrt(xc**2+yc**2)>elPerfect.ap or not 0.0<s<Ls:
                return
            qEl = np.asarray([x, y, z])
            [Bgradx, Bgrady, Bgradz], B0 = lensIdeal.BNorm_Gradient(qEl, returnNorm=True,useApprox=True)
            valsIdeal = np.array([Bgradx, Bgrady, Bgradz, B0])
            [Bgradx, Bgrady, Bgradz], B0 = lensDeviated.BNorm_Gradient(qEl, returnNorm=True,useApprox=True)
            valsDeviated = np.array([Bgradx, Bgrady, Bgradz, B0])
            vals = valsDeviated - valsIdeal
            Fx, Fy, Fz = -vals[:3] * SIMULATION_MAGNETON
            Fy, Fz = Fz, -Fy
            deltaV_Direct=vals[-1]*SIMULATION_MAGNETON
            deltaF_Direct=np.asarray([Fx,Fy,Fz])
            x, y, z = coordsCartesian[index]
            x, y, z = x, z, -y
            qEl = np.asarray([x, y, z])
            deltaF_el=elDeviation.force(qEl) - elPerfect.force(qEl)
            assert iscloseAll(deltaF_el,deltaF_Direct,abstol=1e-6)
            deltaV_El=elDeviation.magnetic_Potential(qEl)-elPerfect.magnetic_Potential(qEl)
            assert isclose(deltaV_El,deltaV_Direct,abs_tol=1e-6)


class CombinerIdealTestHelper(ElementTestHelper):

    def __init__(self):
        self.Lm=.218734921
        self.ap=.014832794
        particle0=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,0.0,0.0]))
        qf0 = np.array([-0.22328330064926563  ,  0.00998854916179846  ,-0.0023411835183533964])
        pf0 = np.array([-199.53214662301482  ,   16.878448509595106 ,-0.6776825317367712])
        super().__init__(CombinerIdeal,particle0,qf0,pf0,True,False,False)

    def make_coordTestRules(self):
        #test generic conditions of the element
        floatx = st.floats(min_value=-self.Lm*0.1, max_value=1.25*self.Lm)
        floaty = st.floats(min_value=-2*self.ap, max_value=2*self.ap)
        floatz = st.floats(min_value=-2*self.ap/2, max_value=2*self.ap/2) #z ap is half of y
        return floatx,floaty,floatz

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0):
        PTL=ParticleTracerLattice(v0Nominal=200.0,standardMagnetErrors=magnetErrors,jitterAmp=jitterAmp)
        PTL.add_Drift(5e-3)
        PTL.add_Combiner_Ideal(Lm=self.Lm,ap=self.ap)
        PTL.end_Lattice(constrain=False, surpressWarning=True, enforceClosedLattice=False)
        return PTL


class CombinerHalbachTestHelper(ElementTestHelper):

    def __init__(self):
        self.Lm=.1453423
        self.rp=.0123749
        particle0=Particle(qi=np.asarray([-.01,5e-3,-3.43e-3]),pi=np.asarray([-201.0,5.0,-3.2343]))
        qf0=np.array([-0.2069017180107142   , -0.005645918307301063 ,0.0038662287776495517])
        pf0=np.array([-200.37613625317655 ,  -13.477494327817205,   10.273175593186906])
        super().__init__(CombinerHalbachLensSim,particle0,qf0,pf0,True,True,True)

    def make_coordTestRules(self):
        #test generic conditions of the element
        floatyz = st.floats(min_value=-3 * self.rp, max_value=3 * self.rp)
        floatx = st.floats(min_value=-self.el.L/10.0, max_value=self.el.L*1.1)
        return floatx,floatyz,floatyz

    def make_Latice(self,magnetErrors=False,jitterAmp=0.0):
        PTL = ParticleTracerLattice(v0Nominal=200.0,standardMagnetErrors=magnetErrors,jitterAmp=jitterAmp)
        PTL.add_Drift(5e-3)
        PTL.add_Combiner_Sim_Lens(self.Lm,self.rp,apFrac=.8)
        PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        return PTL


class ElementTestRunner:

    def __init__(self, ElementTestHelper: ElementTestHelper):
        self.elTestHelper = ElementTestHelper
        self.timeStepTracing = 5e-6
        self.numericTol=1e-12

    def run_Tests(self):
        self.test_Tracing()
        self.test_Coord_Consistency()
        self.test_Coord_Conversions()
        self.test_Magnet_Imperfections()
        self.test_Imperfections_Tracing()
        self.test_Misalignment1()

    def test_Tracing(self):
        """Test that particle tracing yields the same results"""
        particleTracedList = self.trace_Different_Conditions(self.elTestHelper.PTL)
        self.assert_Particle_List_Is_Expected(particleTracedList,self.elTestHelper.qf0,self.elTestHelper.pf0)

    def test_Coord_Consistency(self):
        """Test that force returns nan when the coord is outside the element. test this agrees with shapely geometry
        when applicable"""
        isInsideList = []
        el=self.elTestHelper.el
        @given(*self.elTestHelper.coordTestRules)
        @settings(max_examples=1_000, deadline=None)
        def is_Inside_Consistency(x1: float, x2: float, x3: float):
            coord = self.elTestHelper.convert_Test_Coord_To_El_Frame(x1, x2, x3)
            F = el.force(coord)
            V = el.magnetic_Potential(coord)
            coord2D = coord.copy()
            coord2D[2] = 0.0
            if self.elTestHelper.useShapelyTest == True:
                self.test_Coord2D_Against_Shapely(coord2D)
            isInsideFull3D = el.is_Coord_Inside(coord)
            isInsideList.append(isInsideFull3D)
            #todo: clean this up
            if isInsideFull3D == False:
                assert math.isnan(F[0]) == True and math.isnan(V) == True, str(F) + ',' + str(V)
            else:
                assert np.any(np.isnan(F)) == False and np.isnan(V) == False, str(F) + ',' + str(V) + ',' + str(
                    isInsideFull3D)
        is_Inside_Consistency()
        numTrue = sum(isInsideList)
        assert numTrue > 50  # at least 50 true seems reasonable

    def test_Coord_Conversions(self):
        #check that coordinate conversions work
        tol=1e-12
        el = self.elTestHelper.el
        @given(*self.elTestHelper.coordTestRules)
        @settings(max_examples=1_000,deadline=None)
        def convert_Invert_Consistency(x1,x2,x3):
            coordEl0=self.elTestHelper.convert_Test_Coord_To_El_Frame(x1,x2,x3)
            coordLab=el.transform_Element_Coords_Into_Lab_Frame(coordEl0)
            coordsEl=el.transform_Lab_Coords_Into_Element_Frame(coordLab)
            assert np.all(np.abs(coordEl0-coordsEl)<tol)
            vecEl0=coordEl0
            vecLab=el.transform_Lab_Frame_Vector_Into_Element_Frame(vecEl0)
            vecEl=el.transform_Element_Frame_Vector_Into_Lab_Frame(vecLab)
            assert np.all(np.abs(vecEl0 - vecEl) < tol)
        convert_Invert_Consistency()

    def test_Magnet_Imperfections(self):
        """Test that there is no symmetry. Misalinging should break the symmetry. This is sort of as silly test"""
        if self.elTestHelper.testMagnetErrors==True:
            PTL = self.elTestHelper.make_Latice(magnetErrors=True)
            el = self.elTestHelper.get_Element(PTL)
            @given(*self.elTestHelper.coordTestRules)
            @settings(max_examples=100, deadline=None)
            def test_Magnetic_Imperfection_Field_Symmetry(x1: float, x2: float, x3: float):
                if any(isclose(x,0.0,abs_tol=1e-6) for x in [x1,x2,x3]):
                    return
                else:
                    coord = self.elTestHelper.convert_Test_Coord_To_El_Frame(x1, x2, x3)
                    F0 = np.abs(el.force(coord))
                    for y in [coord[1],-coord[1]]:
                        z=-coord[2]
                        if np.isnan(F0[0])==False and y!=0 and z!=0:
                            FSym=np.abs(el.force(np.array([coord[0],y,z])))
                            assert iscloseAll(F0,FSym,1e-10)==False #assert there is no symmetry
            test_Magnetic_Imperfection_Field_Symmetry()

    def test_Imperfections_Tracing(self):
        """test that misalignment and errors change results"""
        if self.elTestHelper.testMisalignment==True or self.elTestHelper.testMagnetErrors==True:
            particleList = self.trace_Different_Conditions(self.elTestHelper.PTL)
            qf0Temp,pf0Temp=particleList[0].qf,particleList[0].pf
            def wrapper(magnetError,jitterAmp):
                PTL=self.elTestHelper.make_Latice(magnetErrors=magnetError,jitterAmp=jitterAmp)
                el=self.elTestHelper.get_Element(PTL)
                el.perturb_Element(.1*jitterAmp,.1*jitterAmp,.1*jitterAmp/el.L,.1*jitterAmp/el.L)
                particleList = self.trace_Different_Conditions(PTL)
                with pytest.raises(ValueError) as excInfo:
                    #it's possible this won't trigger if the magnet or misalignment errors are really small
                    self.assert_Particle_List_Is_Expected(particleList, qf0Temp, pf0Temp,absTol=1e-6)
                assert str(excInfo.value)=="particle test mismatch"
            if self.elTestHelper.testMisalignment==True: wrapper(False,1e-3)
            if self.elTestHelper.testMagnetErrors: wrapper(True,0.0)

    def test_Misalignment1(self):
        """Test that misaligning element does not results in any errors or conflicts with other methods"""
        def _test_Miaslignment(jitterAmp):
            PTL = self.elTestHelper.make_Latice(jitterAmp=jitterAmp)
            el = self.elTestHelper.get_Element(PTL)
            jitterArr = 2 * (np.random.random_sample(4) - .5) * jitterAmp
            jitterArr[2:] *= 1 / el.L  # rotation component
            blockPrint()
            el.perturb_Element(*jitterArr)
            assert el.get_Valid_Jitter_Amplitude()<=jitterAmp*np.sqrt(2)+1e-12
            enablePrint()
            @given(*self.elTestHelper.coordTestRules)
            @settings(max_examples=500, deadline=None)
            def test_Geometry_And_Fields(x1: float, x2: float, x3: float):
                """Test that the force and magnetic fields functions return nan when the particle is outside the vacuum
                 and that it agrees with is_Coords_Inside"""
                coord = self.elTestHelper.convert_Test_Coord_To_El_Frame(x1, x2, x3)
                F = el.force(coord)
                V = el.magnetic_Potential(coord)
                isInside = el.is_Coord_Inside(coord)
                assert (not math.isnan(F[0]))==isInside and (not math.isnan(V))==isInside
            test_Geometry_And_Fields()
        if self.elTestHelper.testMisalignment==True:
            jitterAmpArr=np.linspace(0.0,self.elTestHelper.el.rp/10.0,5)
            np.random.seed(42)
            [_test_Miaslignment(jitter) for jitter in jitterAmpArr]
            np.random.seed(int(time.time()))

    def trace_Different_Conditions(self,PTL):
        PT = ParticleTracer(PTL)
        particleList = []
        for fastMode in (True, False):
            for accelerated in (True, False):
                particleList.append(
                    PT.trace(self.elTestHelper.particle0.copy(), self.timeStepTracing, 1.0, fastMode=fastMode,
                             accelerated=accelerated))
        return particleList

    def assert_Particle_List_Is_Expected(self, particleList: list[Particle],qf0: np.ndarray,pf0: np.ndarray,
                                         absTol:float=1e-14)-> None:
        for particle in particleList:
            qf, pf = particle.qf, particle.pf
            np.set_printoptions(precision=100)
            if iscloseAll(qf,qf0,absTol)==False or iscloseAll(pf,pf0,absTol)==False:
                # print(repr(qf), repr(pf))
                # print(repr(qf0), repr(pf0))
                raise ValueError("particle test mismatch")

    def is_Inside_Shapely(self,qEl):
        """Check with Shapely library that point resides in 2D footprint of element. It's possible that the point may
        fall just on the edge of the object, so return result with and without small padding"""
        el=self.elTestHelper.el
        qLab = el.transform_Element_Coords_Into_Lab_Frame(qEl)
        isInsideUnpadded=el.SO.contains(Point(qLab[:2]))
        SO_Padded = el.SO.buffer(1e-9)  # add padding to avoid issues of point right on edge
        isInsidePadded = SO_Padded.contains(Point(qLab[:2]))
        return isInsidePadded,isInsideUnpadded

    def test_Coord2D_Against_Shapely(self,coord2D):
        isInside2DShapely, isInside2DShapelyPadded = self.is_Inside_Shapely(coord2D)
        if isInside2DShapely == isInside2DShapelyPadded:  # if consistent for both its not a weird situation of being
            # right on the edge, so go ahead and check
            isInside2D = self.elTestHelper.el.is_Coord_Inside(coord2D)
            assert isInside2DShapely==isInside2D #this can be falsely triggered by circles in shapely!!


def run_Tests(parallel=False):
    testersToRun=[DriftTestHelper,
    LensIdealTestHelper,
    BenderIdealTestHelper,
    CombinerIdealTestHelper,
    CombinerHalbachTestHelper,
    HexapoleLensSimTestHelper,
    HexapoleSegmentedBenderTestHelper]
    def run_Tester(tester):
        tester().run_Tests()
    processes=-1 if parallel==True else 1
    tool_Parallel_Process(run_Tester,testersToRun,processes=processes)