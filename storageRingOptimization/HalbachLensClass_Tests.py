import matplotlib.pyplot as plt
import numpy as np
from HalbachLensClass import Sphere,RectangularPrism,Layer,HalbachLens
import scipy.optimize as spo
class SphereTestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines
    def run_Tests(self):
        self.test1()
        self.test2()
        self.test3()
    def test1(self):
        #Test that the field points in the right direction
        sphere=Sphere(.0254)
        sphere.position_Sphere(r=.05,phi=0.0,z=0.0)
        sphere.orient(np.pi/2,0.0)
        rCenter=np.zeros((1,3))
        BVec_0=np.asarray([0.11180404595756577 ,-0.0 ,-3.4230116753414134e-18]) #point along x only
        BVec=sphere.B(rCenter)[0]
        assert np.all(np.abs(BVec-BVec_0)<self.numericTol) and np.all(np.abs(BVec[1:])<self.numericTol )
    def test2(self):
        #test that rotation symmetry works as expected
        sphere1=Sphere(.0254)
        sphere1.position_Sphere(r=.05,phi=0.0,z=0.0)
        sphere1.orient(np.pi/2,0.0)
        sphere2 = Sphere(.0254)
        sphere2.position_Sphere(r=.05, phi=np.pi/3, z=0.0)
        sphere2.orient(np.pi / 2, 4*np.pi/3)
        rTest = np.ones((1,3))*.01
        BVec1=sphere1.B_Shim(rTest,planeSymmetry=False)[0]
        BVec2=sphere2.B_Shim(rTest,planeSymmetry=False)[0]
        BVec1_0=np.asarray([-0.0011941881467123633 ,-0.16959218399899806 ,0.025757119925902405])
        BVec2_0=np.asarray([-0.001194188146712627 ,-0.16959218399899786 ,0.025757119925902378])
        assert np.all(np.abs(BVec1 - BVec1_0) < self.numericTol)
        assert np.all(np.abs(BVec2 - BVec2_0) < self.numericTol)
        assert np.all(np.abs(BVec2 - BVec1) < self.numericTol)
    def test3(self):
        # test that reflection symmetry works as expected
        sphere1 = Sphere(.0254)
        sphere1.position_Sphere(r=.05, phi=0.0, z=.1)
        sphere1.orient(np.pi / 4, 0.0)
        sphere2 = Sphere(.0254)
        sphere2.position_Sphere(r=.05, phi=0.0, z=-.1)
        sphere2.orient(3*np.pi / 4, 0.0)
        rTest = np.ones((1, 3)) * .01
        BVec_Symm1=sphere1.B_Shim(rTest,planeSymmetry=True)[0]
        BVec_Symm2=sphere1.B_Shim(rTest,planeSymmetry=False)[0]+sphere2.B_Shim(rTest,planeSymmetry=False)[0]
        BVec_Symm1_0=np.asarray([-0.005073988068039584, -0.004641226428626564 ,0.010301384134222108])
        BVec_Symm2_0=np.asarray([-0.005073988068039583, -0.004641226428626562 ,0.010301384134222113])
        assert np.all(np.abs(BVec_Symm1 - BVec_Symm1_0) < self.numericTol)
        assert np.all(np.abs(BVec_Symm2 - BVec_Symm2_0) < self.numericTol)
        assert np.all(np.abs(BVec_Symm2 - BVec_Symm1) < self.numericTol)
class RectangularPrismTestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines
    def run_Tests(self):
        self.test1()
    def test1(self):
        #Test that fields point as expected
        prism=RectangularPrism(.0254,.1)
        prism.place(.07,0.0,0.0,np.pi)
        rCenter=np.zeros((1,3))
        BVec=prism.B(rCenter)[0]
        BVec_0=np.asarray([-0.026118798585274296 ,3.1986303085030583e-18 ,0.0])
        assert np.all(np.abs(BVec[1:])<self.numericTol) and BVec[0]<0.0
        assert np.all(np.abs(BVec-BVec_0)<self.numericTol)
class LayerTestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines
    def run_Tests(self):
        self.test1()
    def test1(self):
        #test that fields point as expected
        layer1=Layer(1.0,.02,.5,.05)
        rTest=np.asarray([[.02,.02,1.0]])
        BVec=layer1.B(rTest)[0]
        BVec_0=np.asarray([7.04735665657541e-09 ,0.1796475065591648 ,0.0])
        assert abs(BVec[2])<self.numericTol
        assert np.all(np.abs(BVec_0-BVec)<self.numericTol)
class HalbachLensTestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines
        self.rp=5e-2
        self.length=.15
        self.magnetWidth=.0254
    def run_Tests(self):
        self.test1()
        self.test2()
    def hexapole_Fit(self,r,B0):
        return B0*(r/self.rp)**2
    def test1(self):
        #test that concentric layers work as expected
        lensAB = HalbachLens(self.length, (self.magnetWidth, self.magnetWidth*1.5), (self.rp,self.rp+self.magnetWidth))
        lensA = HalbachLens(self.length, (self.magnetWidth,), (self.rp,))
        lensB = HalbachLens(self.length, (self.magnetWidth*1.5,), (self.rp + self.magnetWidth,))
        xArr = np.linspace(-self.rp * .3, self.rp * .3,10)
        coords = np.asarray(np.meshgrid(xArr, xArr, xArr)).T.reshape(-1, 3)
        coords[:,2]+=self.length/3 #break symmetry of test points
        coords[:,1]+=self.rp/10.0 #break symmetry of test points
        coords[:,0]-=self.rp/8.0 #break symmetry of test points
        BNormVals1=np.linalg.norm(lensAB.B_Vec(coords),axis=1)
        BNormVals2=np.linalg.norm(lensA.B_Vec(coords)+lensB.B_Vec(coords),axis=1)
        RMS1,RMS2=np.std(BNormVals1),np.std(BNormVals2)
        mean1,mean2=np.mean(BNormVals1),np.mean(BNormVals2)
        RMS1_0,RMS2_0=0.07039945056080353, 0.07039945056080353
        mean1_0,mean2_0=0.089077017634823, 0.089077017634823
        assert np.abs(RMS1_0-RMS1)<self.numericTol and np.abs(RMS2_0-RMS2)<self.numericTol
        assert np.abs(mean1-mean1_0)<self.numericTol and np.abs(mean2-mean2_0)<self.numericTol
    def test2(self):
        #test that the lens is well fit to a parabolic potential
        magnetWidth1,magnetWidth2=.0254,.0254*1.5
        lens=HalbachLens(self.length,(magnetWidth1,magnetWidth2),(self.rp,self.rp+.0254))
        xArr=np.linspace(-self.rp*.9,self.rp*.9)
        coords=np.asarray(np.meshgrid(xArr,xArr,0.0)).T.reshape(-1,3)
        rArr=np.linalg.norm(coords[:,:2],axis=1)
        coords=coords[rArr<self.rp]
        rArr=rArr[rArr<self.rp]
        BNormVals=lens.BNorm(coords)
        params=spo.curve_fit(self.hexapole_Fit,rArr,BNormVals)[0]
        residuals=1e2*np.abs(np.sum(BNormVals-self.hexapole_Fit(rArr,*params)))/np.sum(BNormVals)
        residuals0=0.03770965561603838
        assert abs(residuals-residuals0)<self.numericTol
        # plt.scatter(rArr,BNormVals)
        # plt.show()
def test_Sphere():
    SphereTestHelper().run_Tests()
def test_Rectangular_Prism():
    RectangularPrismTestHelper().run_Tests()
def test_Layer():
    LayerTestHelper().run_Tests()
def test_Halbach_Lens():
    HalbachLensTestHelper().run_Tests()