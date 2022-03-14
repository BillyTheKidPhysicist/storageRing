import time
import numpy as np
import math
from elementPT import BenderIdeal,Drift,LensIdeal,CombinerIdeal,CombinerHexapoleSim,HalbachBenderSimSegmented,HalbachLensSim
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleTracerClass import ParticleTracer
from ParticleClass import Particle
from hypothesis import given,settings,strategies as st
import matplotlib.pyplot as plt
from constants import SIMULATION_MAGNETON
from shapely.geometry import Point
def absDif(x1,x2):
    return abs(abs(x1)-abs(x2))

def trace_Different_Conditions(PTL,particleInitial,h):
    PT=ParticleTracer(PTL)
    particleList=[]
    for fastMode in (True,False):
        for accelerated in (True,False):
            particleList.append(PT.trace(particleInitial.copy(), h, 1.0, fastMode=fastMode,accelerated=accelerated))
    return particleList

def assert_Particle_List_Is_Expected(particleList,qf0,pf0):
    tol=1e-14
    for particle in particleList:
        qf,pf=particle.qf,particle.pf
        # np.set_printoptions(precision=100)
        # print(repr(qf),repr(pf))
        assert np.all(np.abs((qf-qf0))<tol), str(repr(qf))+','+str(repr(qf0))
        assert np.all(np.abs((pf-pf0))<tol), str(repr(pf))+','+str(repr(pf0))

class genericElementTestHelper:
    #elements have generic functions and features common to all kinds. This tests those

    def __init__(self,el,coordsTestRules,coordFrame):
        assert len(coordsTestRules)==3
        assert coordFrame in ('cylinderical','cartesian')
        if coordFrame=='cylinderical': assert any(type(el)==elType for elType in (BenderIdeal,HalbachBenderSimSegmented))
        else: assert any(type(el)==elType for elType in (Drift,LensIdeal,CombinerIdeal,CombinerHexapoleSim,HalbachLensSim))
        self.el=el
        self.coordsTestRules=coordsTestRules
        self.coordFrame=coordFrame
        self.testShapely=False if coordFrame=='cylinderical' else True #circles are not well modeled in shapely

    def convert_Coord(self,x1,x2,x3):
        if self.coordFrame=='cartesian':
            x,y,z=x1,x2,x3
        else:
            r,theta,z=x1,x2,x3
            x,y=r*np.cos(theta),r*np.sin(theta)
        return np.asarray([x, y, z])

    def is_Inside_Shapely(self,qEl):
        """Check with Shapely library that point resides in 2D footprint of element. It's possible that the point may
        fall just on the edge of the object, so return result with and without small padding"""
        qLab = self.el.transform_Element_Coords_Into_Lab_Frame(qEl)
        isInsideUnpadded=self.el.SO.contains(Point(qLab[:2]))
        SO_Padded = self.el.SO.buffer(1e-9)  # add padding to avoid issues of point right on edge
        isInsidePadded = SO_Padded.contains(Point(qLab[:2]))
        return isInsidePadded,isInsideUnpadded

    def test_Against_Shapely(self,coord2D):
        isInside2DShapely, isInside2DShapelyPadded = self.is_Inside_Shapely(coord2D)
        if isInside2DShapely == isInside2DShapelyPadded:  # if consistent for both its not a weird situation of being
            # right on the edge, so go ahead and check
            isInside2D = self.el.is_Coord_Inside(coord2D)
            assert isInside2DShapely==isInside2D #this can be falsely triggered by circles in shapely!!

    def run_Tests(self):
        self.test1()
        self.test2()

    def test1(self):
        isInsideList=[]
        @given(*self.coordsTestRules)
        @settings(max_examples=5_000,deadline=None)
        def is_Inside_Consistency(x1:float,x2:float,x3:float):
            coord=self.convert_Coord(x1,x2,x3)
            F=self.el.force(coord)
            V=self.el.magnetic_Potential(coord)
            coord2D=coord.copy()
            coord2D[2]=0.0
            if self.testShapely==True:
                self.test_Against_Shapely(coord2D)
            isInsideFull3D=self.el.is_Coord_Inside(coord)
            isInsideList.append(isInsideFull3D)
            if isInsideFull3D==False: assert math.isnan(F[0])==True and math.isnan(V)==True,str(F)+','+str(V)
            else: assert np.any(np.isnan(F))==False and np.isnan(V)==False,str(F)+','+str(V)+','+str(isInsideFull3D)
        is_Inside_Consistency()
        numTrue=sum(isInsideList)
        assert numTrue>50 #at least 50 true seems reasonable

    def test2(self):
        #check that coordinate conversions work
        tol=1e-12
        @given(*self.coordsTestRules)
        @settings(max_examples=5_000,deadline=None)
        def convert_Invert_Consistency(x1,x2,x3):
            coordEl0=self.convert_Coord(x1,x2,x3)
            coordLab=self.el.transform_Element_Coords_Into_Lab_Frame(coordEl0)
            coordsEl=self.el.transform_Lab_Coords_Into_Element_Frame(coordLab)
            assert np.all(np.abs(coordEl0-coordsEl)<tol)
            vecEl0=coordEl0
            vecLab=self.el.transform_Lab_Frame_Vector_Into_Element_Frame(vecEl0)
            vecEl=self.el.transform_Element_Frame_Vector_Into_Lab_Frame(vecLab)
            assert np.all(np.abs(vecEl0 - vecEl) < tol)
        convert_Invert_Consistency()


class driftTestHelper:

    def __init__(self):
        self.L,self.ap=.15432,.0392
        self.PTL=ParticleTracerLattice(200.0)
        self.PTL.add_Drift(self.L,self.ap)
        self.PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        self.particleTracer=ParticleTracer(self.PTL)
        self.el=self.PTL.elList[0]

    def run_Tests(self):
        self.test1()
        self.test2()
        self.test3()

    def test1(self):
        #test generic conditions of the element
        floatyz = st.floats(min_value=-1.5 * self.ap, max_value=1.5 * self.ap)
        floatx = st.floats(min_value=-self.L*.25, max_value=self.L * 1.25)
        coordTestRules=(floatx,floatyz,floatyz)
        genericElementTestHelper(self.el,coordTestRules,'cartesian').run_Tests()

    def test2(self):
        # test that particle travels through drift region and ends where expected
        tol,h=1e-12,1e-7
        deltaX=-self.L #negative because particles start by moving along -x
        slopez=.5*self.ap/deltaX
        zi=-self.ap/4.0
        zf=deltaX*slopez+zi
        slopey = .25 * self.ap /deltaX
        yi = self.ap / 4.0
        yf = deltaX* slopey + yi
        vx=-200.0
        particle=Particle(qi=np.asarray([0.0,yi,zi]),pi=np.asarray([vx,slopey*vx,slopez*vx]))
        particleTraced=self.particleTracer.trace(particle,1e-7,1.0,fastMode=True)
        qfTrace,pfTrace=particleTraced.qf,particle.pf
        slopeyTrace,slopezTrace=pfTrace[1]/pfTrace[0],pfTrace[2]/pfTrace[0]
        yfTrace,zfTrace=qfTrace[1],qfTrace[2]
        assert absDif(slopey,slopeyTrace)<tol and absDif(slopez,slopezTrace)<tol
        assert abs(yf-yfTrace)<tol and abs(zf-zfTrace)<tol

    def test3(self):
        """Compare previous to current tracing. ParticleTracerClass can affect this"""
        particle=Particle(qi=np.asarray([0.0,1e-3,-2e-3]),pi=np.asarray([-201.0,5.2,-3.8]))
        qf0 = np.array([-0.15432, 0.004992358208955224,-0.004917492537313433])
        pf0 = np.array([-201., 5.2, -3.8])
        particleList=trace_Different_Conditions(self.PTL,particle,1e-5)
        assert_Particle_List_Is_Expected(particleList,qf0,pf0)


class lensIdealTestHelper:

    def __init__(self):
        self.L, self.rp,self.Bp = .51824792317429, .024382758923,1.832484234
        self.PTL=ParticleTracerLattice(200.0)
        self.PTL.add_Lens_Ideal(self.L,self.Bp,self.rp)
        self.PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        self.el,self.particleTracer=self.PTL.elList[0],ParticleTracer(self.PTL)

    def run_Tests(self):
        self.test1()
        self.test2()
        self.test3()

    def test1(self):
        #test generic conditions of the element
        floatyz = st.floats(min_value=-1.5 * self.rp, max_value=1.25 * self.rp)
        floatx = st.floats(min_value=-self.L*.25, max_value=self.L * 1.25)
        coordTestRules=(floatx,floatyz,floatyz)
        genericElementTestHelper(self.el,coordTestRules,'cartesian').run_Tests()

    def test2(self):
        tol=1e-2 #test to 1% accuracy
        #does the particle behave as theory predicts? It should be very close because it's ideal
        particle=Particle(qi=np.asarray([0.0,self.rp/2.0,0.]))
        particle=self.particleTracer.trace(particle,1e-6,1.0)
        yi,yf,pyf=particle.qi[1],particle.qf[1],particle.pf[1]
        yRMS,pyRMS=np.std(particle.qArr[:,1]),np.std(particle.pArr[:,1])
        K = 2 * SIMULATION_MAGNETON * self.Bp / (particle.pi[0] ** 2 * self.rp ** 2)
        phi = np.sqrt(K) * self.L
        yfTheory,pyfTheory=yi*np.cos(phi),-abs(particle.pi[0])*yi*np.sin(phi)*np.sqrt(K)
        assert abs(yf-yfTheory)<tol*yi and abs(pyf-pyfTheory)<tol*pyRMS

    def test3(self):
        """Compare previous to current tracing. ParticleTracerClass can affect this"""
        particle=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,8.2,-6.8]))
        qf0 = np.array([-5.1652000000000176e-01, -1.6224839396623185e-03,4.5059256741117402e-04])
        pf0 = np.array([-201.               ,    7.696116749320538,   -8.054201761708454])
        particleList=trace_Different_Conditions(self.PTL,particle,1e-5)
        assert_Particle_List_Is_Expected(particleList,qf0,pf0)


class benderIdealTestHelper:

    def __init__(self):
        self.ang=np.pi/2.0
        self.Bp=.8934752374
        self.rb=.94830284532
        self.rp=.01853423
        self.PTL=ParticleTracerLattice(200.0)
        self.PTL.add_Drift(5e-3)
        self.PTL.add_Bender_Ideal(self.ang,self.Bp,self.rb,self.rp)
        self.PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        self.el,self.particleTracer=self.PTL.elList[1],ParticleTracer(self.PTL)

    def run_Tests(self):
        self.test1()
        self.test2()

    def test1(self):
        #test generic conditions of the element
        floatr = st.floats(min_value=self.rb-self.rp*2, max_value=self.rb+self.rp*2)
        floatphi=st.floats(min_value=-self.ang*.1, max_value=1.1*self.ang)
        floatz=st.floats(min_value=-self.rp/2,max_value=self.rp/2)
        coordTestRules=(floatr,floatphi,floatz)
        genericElementTestHelper(self.el,coordTestRules,'cylinderical').run_Tests()

    def test2(self):
        """Compare previous to current tracing. ParticleTracerClass can affect this"""
        particle=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,5.2,-6.8]))
        qf0 = np.array([-9.639381030969734e-01,  9.576855890781706e-01,8.586086892465979e-05])
        pf0 = np.array([ -4.741346774755123, 200.6975241302656  ,   7.922912689270208])
        particleList=trace_Different_Conditions(self.PTL,particle,5e-6)
        assert_Particle_List_Is_Expected(particleList,qf0,pf0)


class combinerIdealTestHelper:

    def __init__(self):
        self.Lm=.218734921
        self.ap=.014832794
        self.PTL=ParticleTracerLattice(200.0)
        self.PTL.add_Drift(5e-3)
        self.PTL.add_Combiner_Ideal(Lm=self.Lm,ap=self.ap)
        self.PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        self.el,self.particleTracer=self.PTL.elList[1],ParticleTracer(self.PTL)

    def run_Tests(self):
        self.test1()
        self.test2()

    def test1(self):
        #test generic conditions of the element
        floatx = st.floats(min_value=-self.Lm*0.1, max_value=1.25*self.Lm)
        floaty = st.floats(min_value=-2*self.ap, max_value=2*self.ap)
        floatz = st.floats(min_value=-2*self.ap/2, max_value=2*self.ap/2) #z ap is half of y
        coordTestRules=(floatx,floaty,floatz)
        genericElementTestHelper(self.el,coordTestRules,'cartesian').run_Tests()

    def test2(self):
        """Compare previous to current tracing. ParticleTracerClass can affect this"""
        particle=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,0.0,0.0]))
        qf0 = np.array([-0.22328330008309497 ,  0.009988555672041705,-0.002335465242450282])
        pf0 = np.array([-199.532144335288  , 16.87847481554656 ,-0.6666278524886048])
        particleList=trace_Different_Conditions(self.PTL,particle,5e-6)
        assert_Particle_List_Is_Expected(particleList,qf0,pf0)


class hexapoleSegmentedBenderSimTestHelper:

    def __init__(self):
        self.Lm=.0254
        self.rp=.014832
        self.numMagnets=150
        self.rb=1.02324
        self.ang=self.numMagnets*self.Lm/self.rb
        self.PTL=ParticleTracerLattice(200.0)
        self.PTL.add_Drift(5e-3)
        self.PTL.add_Halbach_Bender_Sim_Segmented(self.Lm,self.rp,self.numMagnets,self.rb)
        self.PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        self.el,self.particleTracer=self.PTL.elList[1],ParticleTracer(self.PTL)

    def run_Tests(self):
        self.test1()
        self.test2()

    def test1(self):
        #test generic conditions of the element
        floatr = st.floats(min_value=self.rb-self.rp*2, max_value=self.rb+self.rp*2)
        floatphi=st.floats(min_value=-self.ang*.1, max_value=1.1*self.ang)
        floatz=st.floats(min_value=-self.rp/2,max_value=self.rp/2)
        coordTestRules=(floatr,floatphi,floatz)
        genericElementTestHelper(self.el,coordTestRules,'cylinderical').run_Tests()

    def test2(self):
        """Compare previous to current tracing. ParticleTracerClass can affect this"""
        particle=Particle(qi=np.asarray([-.01,1e-3,-2e-3]),pi=np.asarray([-201.0,1.0,-.5]))
        qf0 = np.array([0.6246876805773034   , 1.8195263869875777   ,0.0021494963567847674])
        pf0 = np.array([ 158.1515485477083   , -124.06038699242039  ,0.3732855519313561])
        particleList=trace_Different_Conditions(self.PTL,particle,5e-6)
        assert_Particle_List_Is_Expected(particleList,qf0,pf0)


class hexapoleLensSimTestHelper:

    def __init__(self):
        self.L=.1321432
        self.rp=.01874832
        self.PTL=ParticleTracerLattice(200.0)
        self.PTL.add_Halbach_Lens_Sim(self.rp,self.L)
        self.PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        self.el,self.particleTracer=self.PTL.elList[0],ParticleTracer(self.PTL)

    def run_Tests(self):
        self.test1()
        self.test2()

    def test1(self):
        #test generic conditions of the element
        floatyz = st.floats(min_value=-2 * self.rp, max_value=2 * self.rp)
        floatx = st.floats(min_value=-self.L / 10.0, max_value=self.L * 1.25)
        coordTestRules=(floatx,floatyz,floatyz)
        genericElementTestHelper(self.el,coordTestRules,'cartesian').run_Tests()

    def test2(self):
        """Compare previous to current tracing. ParticleTracerClass can affect this"""
        particle=Particle(qi=np.asarray([-.01,5e-3,-7.43e-3]),pi=np.asarray([-201.0,5.0,-8.2343]))
        qf0=np.array([-0.13131255560602847 ,  0.005389042193361335,-0.008480447325995597])
        pf0=np.array([-201.18045750427612  ,   -3.1617793105855436,3.9793250478945374])
        particleList=trace_Different_Conditions(self.PTL,particle,5e-6)
        assert_Particle_List_Is_Expected(particleList,qf0,pf0)


class combinerHexapoleSimTestHelper:

    def __init__(self):
        self.Lm=.1453423
        self.rp=.0123749
        self.PTL=ParticleTracerLattice(200.0)

        self.PTL.add_Drift(5e-3)
        self.PTL.add_Combiner_Sim_Lens(self.Lm,self.rp)
        self.PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
        self.el,self.particleTracer=self.PTL.elList[1],ParticleTracer(self.PTL)

    def run_Tests(self):
        self.test1()
        self.test2()

    def test1(self):
        #test generic conditions of the element
        floatyz = st.floats(min_value=-1.5 * self.rp, max_value=1.5 * self.rp)
        floatx = st.floats(min_value=-self.el.L/10.0, max_value=self.el.L*1.1)
        coordTestRules=(floatx,floatyz,floatyz)
        genericElementTestHelper(self.el,coordTestRules,'cartesian').run_Tests()
    def test2(self):
        """Compare previous to current tracing. ParticleTracerClass can affect this"""
        particle=Particle(qi=np.asarray([-.01,5e-3,-3.43e-3]),pi=np.asarray([-201.0,5.0,-3.2343]))
        qf0=np.array([-0.20686255547198076  , -0.005818682643949487 ,0.0039419965265814014])
        pf0=np.array([-200.42385219087774 ,  -12.914640613791557,   10.07638500140596 ])
        particleList=trace_Different_Conditions(self.PTL,particle,5e-6)
        assert_Particle_List_Is_Expected(particleList,qf0,pf0)

def run_Tests():
    driftTestHelper().run_Tests()
    lensIdealTestHelper().run_Tests()
    benderIdealTestHelper().run_Tests()
    combinerIdealTestHelper().run_Tests()
    hexapoleSegmentedBenderSimTestHelper().run_Tests()
    hexapoleLensSimTestHelper().run_Tests()
    combinerHexapoleSimTestHelper().run_Tests()