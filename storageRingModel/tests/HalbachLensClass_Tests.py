import matplotlib.pyplot as plt
import numpy as np
import time
from HalbachLensClass import Layer,HalbachLens,Sphere
import scipy.optimize as spo
import multiprocess as mp
class SpheretestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines
    def run_tests(self):
        self.test1()
        self.test2()
        self.test3()
    def test1(self):
        #test that the field points in the right direction
        sphere=Sphere(.0254)
        sphere.position_Sphere(r=.05,theta=0.0,z=0.0)
        sphere.orient(np.pi/2,0.0)
        rCenter=np.zeros((1,3))
        BVec_0=np.asarray([0.11180404595756577 ,-0.0 ,-3.4230116753414134e-18]) #point along x only
        BVec=sphere.B(rCenter)[0]
        assert np.all(np.abs(BVec-BVec_0)<self.numericTol) and np.all(np.abs(BVec[1:])<self.numericTol )
    def test2(self):
        #test that rotation symmetry works as expected
        sphere1=Sphere(.0254)
        sphere1.position_Sphere(.05,0.0,0.0)
        sphere1.orient(np.pi/2,0.0)
        sphere2 = Sphere(.0254)
        sphere2.position_Sphere(.05,np.pi/3,0.0)
        sphere2.orient(np.pi / 2, 4*np.pi/3)
        rtest = np.ones((1,3))*.01
        BVec1=sphere1.B_Shim(rtest,planeSymmetry=False)[0]
        BVec2=sphere2.B_Shim(rtest,planeSymmetry=False)[0]
        BVec1_0=np.asarray([-0.0011941881467123633 ,-0.16959218399899806 ,0.025757119925902405])
        BVec2_0=np.asarray([-0.001194188146712627 ,-0.16959218399899786 ,0.025757119925902378])
        assert np.all(np.abs(BVec1 - BVec1_0) < self.numericTol)
        assert np.all(np.abs(BVec2 - BVec2_0) < self.numericTol)
        assert np.all(np.abs(BVec2 - BVec1) < self.numericTol)
    def test3(self):
        # test that reflection symmetry works as expected
        sphere1 = Sphere(.0254)
        sphere1.position_Sphere(.05, 0.0, .1)
        sphere1.orient(np.pi / 4, np.pi/3)
        sphere2 = Sphere(.0254)
        sphere2.position_Sphere(.05, 0.0, -.1)
        sphere2.orient(3*np.pi / 4,np.pi/3)
        rtest = np.ones((1, 3)) * .01
        BVec_Symm1=sphere1.B_Shim(rtest,planeSymmetry=True)[0]
        BVec_Symm2=sphere1.B_Shim(rtest,planeSymmetry=False)[0]+sphere2.B_Shim(rtest,planeSymmetry=False)[0]
        BVec_Symm1_0=np.asarray([-0.0058071761934043635, -0.004844616334816022 ,0.010212674466403442])
        BVec_Symm2_0=np.asarray([-0.005807176193404366, -0.004844616334816021 ,0.010212674466403436])
        assert np.all(np.abs(BVec_Symm1 - BVec_Symm1_0) < self.numericTol)
        assert np.all(np.abs(BVec_Symm2 - BVec_Symm2_0) < self.numericTol)
        assert np.all(np.abs(BVec_Symm2 - BVec_Symm1) < self.numericTol)


class LayertestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines
    def run_tests(self):
        self.test1()
        self.test2()
    def test1(self):
        #test that fields point as expected
        z,width,length,rp=1.0,.02,.5,.05
        layer1=Layer(rp,length,width,position=(0.0,0.0,z))
        rtest=np.asarray([[.02,.02,1.0]])
        BVec=layer1.B_Vec(rtest)
        BVec_0=np.asarray([7.04735665657541e-09 ,0.1796475065591648 ,0.0])
        assert abs(BVec[2])<self.numericTol
        assert np.all(np.abs(BVec_0-BVec)<self.numericTol)
    def test2(self):
        #test that misalingments actually change field
        np.random.seed(42)
        BVecNoError=np.asarray([7.04735665657541e-09 ,0.1796475065591648 ,0.0])
        BVecErrorEachLens_0=np.array([[ 2.8064507038888563e-04,  1.7071255670757210e-01,0.0],
       [-8.9278583100593778e-04,  1.8543806485401598e-01,0.0],
       [-4.3864928880437407e-04,  1.7965469190407585e-01,0.0],
       [ 8.1041300350840118e-05,  1.7963843761049603e-01,0.0]])
        rtest=np.asarray([[.02,.02,1.0]])
        z,width,length,rp=1.0,.02,.5,.05
        layer1=Layer(rp,length,width,position=(0.0,0.0,z),rMagnetShift=1e-3*np.random.random_sample((12,1)))
        layer2=Layer(rp,length,width,position=(0.0,0.0,z),dimShift=1e-3*np.random.random_sample((12,3)))
        layer3=Layer(rp,length,width,position=(0.0,0.0,z),thetaShift=1e-3*np.random.random_sample((12,1)))
        layer4=Layer(rp,length,width,position=(0.0,0.0,z),phiShift=1e-3*np.random.random_sample((12,1)))
        BVecEachLens=[]
        for layer,BVecError0 in zip([layer1,layer2,layer3,layer4],BVecErrorEachLens_0):
            BVec=layer.B_Vec(rtest)
            BVecEachLens.append(BVec)
            assert np.all(np.abs(BVecNoError - BVec)[:2] > self.numericTol) #difference must exist
            assert np.all(np.abs(BVecError0 - BVec)[:2] < self.numericTol) #difference must be reliable with
            #same seed
        np.random.seed(int(time.time()))


class HalbachLenstestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines
        self.rp=5e-2
        self.length=.15
        self.magnetWidth=.0254
    def run_tests(self):
        # self.test1()
        # self.test2()
        self.test3()
    def hexapole_Fit(self,r,B0):
        return B0*(r/self.rp)**2
    def test1(self):
        #test that concentric layers work as expected
        lensAB = HalbachLens((self.rp,self.rp+self.magnetWidth), (self.magnetWidth, self.magnetWidth*1.5),self.length )
        lensA = HalbachLens( self.rp,self.magnetWidth,self.length)
        lensB = HalbachLens(self.rp + self.magnetWidth, self.magnetWidth*1.5,self.length )
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
        lens=HalbachLens((self.rp,self.rp+.0254),(magnetWidth1,magnetWidth2),self.length)
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
    def test3(self):
        #test that standard magnet tolerances changes values
        np.random.seed(42) #for repeatable results
        BNormsVals_STD_NoError=0.11674902392610367
        BNormsVals_STD_Error=0.11648535357552772
        magnetWidth1,magnetWidth2=.0254,.0254*1.5
        lens=HalbachLens((self.rp,self.rp+.0254),(magnetWidth1,magnetWidth2),self.length,useStandardMagErrors=True)
        xArr=np.linspace(-self.rp*.5,self.rp*.5)
        coords=np.asarray(np.meshgrid(xArr,xArr,0.0)).T.reshape(-1,3)
        BNormsVals_STD = np.std(lens.BNorm(coords))
        assert BNormsVals_STD!=BNormsVals_STD_NoError #assert magnet errors causes field changes
        assert abs(BNormsVals_STD-BNormsVals_STD_Error)<self.numericTol #assert magnet errors cause same field value
        # changes with same seed
        np.random.seed(int(time.time()))


def run_Tests(parallel=False):
    def run(func):
        func()
    funcList=[SpheretestHelper().run_tests,LayertestHelper().run_tests,HalbachLenstestHelper().run_tests]
    if parallel==True:
        with mp.Pool() as pool:
            pool.map(run,funcList)
    else:
        list(map(run,funcList))