import time
from HalbachLensClass import Layer, HalbachLens, Sphere, SegmentedBenderHalbach
import scipy.optimize as spo
import multiprocess as mp
import itertools
from helperTools import iscloseAll,tool_Parallel_Process,arr_Product
from typeHints import RealNum
import math
import numpy as np

numericTol = 1e-14  # my working numeric tolerance


def within_Tol(a: RealNum, b: RealNum):
    return math.isclose(a, b, abs_tol=numericTol, rel_tol=0.0)


class SpheretestHelper:
    def __init__(self):
        self.numericTol = 1e-14  # same approach should be this accurate on different machines

    def run_tests(self):
        self.test1()
        self.test2()
        self.test3()

    def test1(self):
        # test that the field points in the right direction
        sphere = Sphere(.0254)
        sphere.position_sphere(r=.05, theta=0.0, z=0.0)
        sphere.orient(np.pi / 2, 0.0)
        rCenter = np.zeros((1, 3))
        BVec_0 = np.asarray([0.11180404595756577, -0.0, -3.4230116753414134e-18])  # point along x only
        B_vec = sphere.B(rCenter)[0]
        assert np.all(np.abs(B_vec - BVec_0) < self.numericTol) and np.all(np.abs(B_vec[1:]) < self.numericTol)

    def test2(self):
        # test that rotation symmetry works as expected
        sphere1 = Sphere(.0254)
        sphere1.position_sphere(.05, 0.0, 0.0)
        sphere1.orient(np.pi / 2, 0.0)
        sphere2 = Sphere(.0254)
        sphere2.position_sphere(.05, np.pi / 3, 0.0)
        sphere2.orient(np.pi / 2, 4 * np.pi / 3)
        rtest = np.ones((1, 3)) * .01
        BVec1 = sphere1.B_shim(rtest, plane_symmetry=False)[0]
        BVec2 = sphere2.B_shim(rtest, plane_symmetry=False)[0]
        BVec1_0 = np.asarray([-0.0011941881467123633, -0.16959218399899806, 0.025757119925902405])
        BVec2_0 = np.asarray([-0.001194188146712627, -0.16959218399899786, 0.025757119925902378])
        assert np.all(np.abs(BVec1 - BVec1_0) < self.numericTol)
        assert np.all(np.abs(BVec2 - BVec2_0) < self.numericTol)
        assert np.all(np.abs(BVec2 - BVec1) < self.numericTol)

    def test3(self):
        # test that reflection symmetry works as expected
        sphere1 = Sphere(.0254)
        sphere1.position_sphere(.05, 0.0, .1)
        sphere1.orient(np.pi / 4, np.pi / 3)
        sphere2 = Sphere(.0254)
        sphere2.position_sphere(.05, 0.0, -.1)
        sphere2.orient(3 * np.pi / 4, np.pi / 3)
        rtest = np.ones((1, 3)) * .01
        BVec_Symm1 = sphere1.B_shim(rtest, plane_symmetry=True)[0]
        BVec_Symm2 = sphere1.B_shim(rtest, plane_symmetry=False)[0] + sphere2.B_shim(rtest, plane_symmetry=False)[0]
        BVec_Symm1_0 = np.asarray([-0.0058071761934043635, -0.004844616334816022, 0.010212674466403442])
        BVec_Symm2_0 = np.asarray([-0.005807176193404366, -0.004844616334816021, 0.010212674466403436])
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
        # test that fields point as expected
        z, width, length, rp = 1.0, .02, .5, .05
        layer1 = Layer(rp, width, length, position=(0.0, 0.0, z))
        rtest = np.asarray([[.02, .02, 1.0]])
        B_vec = layer1.B_vec(rtest)
        BVec_0 = np.asarray([7.04735665657541e-09, 0.1796475065591648, 0.0])
        assert abs(B_vec[2]) < self.numericTol
        assert np.all(np.abs(BVec_0 - B_vec) < self.numericTol)

    def test2(self):
        # test that misalingments actually change field
        np.random.seed(42)
        BVecNoError = np.asarray([7.04735665657541e-09, 0.1796475065591648, 0.0])
        BVecErrorEachLens_0 = np.array([[2.8064507038888563e-04, 1.7071255670757210e-01, 0.0],
                                        [-8.9278583100593778e-04, 1.8543806485401598e-01, 0.0],
                                        [-4.3864928880437407e-04, 1.7965469190407585e-01, 0.0],
                                        [8.1041300350840118e-05, 1.7963843761049603e-01, 0.0]])
        rtest = np.asarray([[.02, .02, 1.0]])
        z, width, length, rp = 1.0, .02, .5, .05
        layer1 = Layer(rp, width, length, position=(0.0, 0.0, z), rMagnetShift=1e-3 * np.random.random_sample((12, 1)))
        layer2 = Layer(rp, width, length, position=(0.0, 0.0, z), dimShift=1e-3 * np.random.random_sample((12, 3)))
        layer3 = Layer(rp, width, length, position=(0.0, 0.0, z), thetaShift=1e-3 * np.random.random_sample((12, 1)))
        layer4 = Layer(rp, width, length, position=(0.0, 0.0, z), phiShift=1e-3 * np.random.random_sample((12, 1)))
        BVecEachLens = []
        for layer, BVecError0 in zip([layer1, layer2, layer3, layer4], BVecErrorEachLens_0):
            B_vec = layer.B_vec(rtest)
            BVecEachLens.append(B_vec)
            assert np.all(np.abs(BVecNoError - B_vec)[:2] > self.numericTol)  # difference must exist
            assert np.all(np.abs(BVecError0 - B_vec)[:2] < self.numericTol)  # difference must be reliable with
            # same seed
        np.random.seed(int(time.time()))


class HalbachLenstestHelper:
    def __init__(self):
        self.rp = 5e-2
        self.length = .15
        self.magnetWidth = .0254

    def run_tests(self):
        self.test1()
        self.test2()
        self.test3()
        self.test4()
        self.test5()
        self.test6()
        self.test7()
        self.test8()

    def hexapole_Fit(self, r, B0):
        return B0 * (r / self.rp) ** 2

    def test1(self):
        # test that concentric layers work as expected
        lensAB = HalbachLens((self.rp, self.rp + self.magnetWidth), (self.magnetWidth, self.magnetWidth * 1.5),
                             self.length)
        lensA = HalbachLens(self.rp, self.magnetWidth, self.length)
        lensB = HalbachLens(self.rp + self.magnetWidth, self.magnetWidth * 1.5, self.length)
        xArr = np.linspace(-self.rp * .3, self.rp * .3, 10)
        coords = np.asarray(np.meshgrid(xArr, xArr, xArr)).T.reshape(-1, 3)
        coords[:, 2] += self.length / 3  # break symmetry of test points
        coords[:, 1] += self.rp / 10.0  # break symmetry of test points
        coords[:, 0] -= self.rp / 8.0  # break symmetry of test points
        BNormVals1 = np.linalg.norm(lensAB.B_vec(coords), axis=1)
        BNormVals2 = np.linalg.norm(lensA.B_vec(coords) + lensB.B_vec(coords), axis=1)
        RMS1, RMS2 = np.std(BNormVals1), np.std(BNormVals2)
        mean1, mean2 = np.mean(BNormVals1), np.mean(BNormVals2)
        RMS1_0, RMS2_0 = 0.07039945056080353, 0.07039945056080353
        mean1_0, mean2_0 = 0.089077017634823, 0.089077017634823
        assert within_Tol(RMS1_0, RMS1) and within_Tol(RMS2_0, RMS2)
        assert within_Tol(mean1, mean1_0) and within_Tol(mean2, mean2_0)

    def test2(self):
        # test that the lens is well fit to a parabolic potential
        magnetWidth1, magnetWidth2 = .0254, .0254 * 1.5
        lens = HalbachLens((self.rp, self.rp + .0254), (magnetWidth1, magnetWidth2), self.length)
        xArr = np.linspace(-self.rp * .9, self.rp * .9)
        coords = np.asarray(np.meshgrid(xArr, xArr, 0.0)).T.reshape(-1, 3)
        rArr = np.linalg.norm(coords[:, :2], axis=1)
        coords = coords[rArr < self.rp]
        rArr = rArr[rArr < self.rp]
        BNormVals = lens.B_norm(coords)
        params = spo.curve_fit(self.hexapole_Fit, rArr, BNormVals)[0]
        residuals = 1e2 * np.abs(np.sum(BNormVals - self.hexapole_Fit(rArr, *params))) / np.sum(BNormVals)
        residuals0 = 0.03770965561603838
        assert within_Tol(residuals, residuals0)

    def test3(self):
        # test that standard magnet tolerances changes values
        np.random.seed(42)  # for repeatable results
        BNormsVals_STD_NoError = 0.11674902392610367
        BNormsVals_STD_Error = 0.11653023297363536
        magnetWidth1, magnetWidth2 = .0254, .0254 * 1.5
        lens = HalbachLens((self.rp, self.rp + .0254), (magnetWidth1, magnetWidth2), self.length,
                           useStandardMagErrors=True
                           , sameSeed=True)
        xArr = np.linspace(-self.rp * .5, self.rp * .5)
        coords = np.asarray(np.meshgrid(xArr, xArr, 0.0)).T.reshape(-1, 3)
        BNormsVals_STD = np.std(lens.B_norm(coords))
        assert BNormsVals_STD != BNormsVals_STD_NoError  # assert magnet errors causes field changes
        assert within_Tol(BNormsVals_STD, BNormsVals_STD_Error)  # assert magnet errors cause same field value
        # changes with same seed
        np.random.seed(int(time.time()))

    def test4(self):
        # test that the a single layer lens has the same field as a single layer
        lens = HalbachLens(self.rp, self.magnetWidth, self.length)
        layer = Layer(self.rp, self.magnetWidth, self.length)
        xArr = np.linspace(-self.rp * .5, self.rp * .5, 20)
        testCoords = np.asarray(list(itertools.product(xArr, xArr, xArr)))
        BNormsValsLayer_STD = np.std(layer.B_norm(testCoords))
        BNormsValsLens_STD = np.std(lens.B_norm(testCoords))
        assert within_Tol(BNormsValsLens_STD, BNormsValsLayer_STD)

    def test5(self):
        # test that slices of the lens results in same field values without magnetostatic method of moments, and the
        # length and coordinates work
        lensSingle = HalbachLens(self.rp, self.magnetWidth, self.length)
        lensSliced = HalbachLens(self.rp, self.magnetWidth, self.length, numDisks=10)
        lengthCounted = sum(layer.length for layer in lensSliced.layerList)
        assert within_Tol(lensSliced.length, lengthCounted) and within_Tol(lensSliced.length, lensSliced.length)
        zCenter = sum(layer.position[2] for layer in lensSliced.layerList) / lensSliced.numSlices
        assert within_Tol(zCenter, 0.0)
        xArr = np.linspace(-self.rp * .5, self.rp * .5, 20)
        testCoords = np.asarray(list(itertools.product(xArr, xArr, xArr)))
        BValsSingle, BValsSliced = lensSingle.B_norm(testCoords), lensSliced.B_norm(testCoords)
        assert iscloseAll(BValsSingle, BValsSliced, numericTol)

    def test6(self):
        # test that magnet errors change field values
        lensPerfect = HalbachLens(self.rp, self.magnetWidth, self.length)
        np.random.seed(42)
        lensError = HalbachLens(self.rp, self.magnetWidth, self.length, useStandardMagErrors=True)
        np.random.seed(42)
        lensErrorSliced = HalbachLens(self.rp, self.magnetWidth, self.length, useStandardMagErrors=True, numDisks=10)
        self.test_All_Three_Are_Different(lensPerfect, lensError, lensErrorSliced)

    def test7(self):
        # test that magnetostatic method of moments (MOM) changes field values
        lensNaive = HalbachLens(self.rp, self.magnetWidth, self.length)
        lensMOM = HalbachLens(self.rp, self.magnetWidth, self.length, applyMethodOfMoments=True)
        lensMOMSliced = HalbachLens(self.rp, self.magnetWidth, self.length, applyMethodOfMoments=True, numDisks=10)
        self.test_All_Three_Are_Different(lensNaive, lensMOM, lensMOMSliced)

    def test8(self):
        """For systems with many magnets and/or many coordinates I split up the coords into smaller chunks to prevent
        memory overflow"""

        lens = HalbachLens(.05, .025, .1, numDisks=10)
        xArr = np.linspace(.01, .01, 10)
        coords = arr_Product(xArr, xArr, xArr)
        BVals_Unsplit = lens.B_norm_grad(coords)
        lens._getB_wrapper = lambda x: HalbachLens._getB_wrapper(lens, x, sizeMax=5)
        BVals_Split = lens.B_norm_grad(coords)
        assert iscloseAll(BVals_Split, BVals_Unsplit, 0.0)  # should be exactly the same

    def test_All_Three_Are_Different(self, lens1, lens2, lens3):
        xArr = np.linspace(-self.rp * .5, self.rp * .5)
        coords = np.asarray(list(itertools.product(xArr, xArr, [0])))
        vals1, vals2, vals3 = [lens.B_norm(coords) for lens in [lens1, lens2, lens3]]
        differenceTol = 1e-6
        for valsA, valsB in [[vals1, vals2], [vals1, vals3], [vals2, vals3]]:
            iscloseAll(valsA, valsB, differenceTol)


class SegmentedBenderHalbachHelper:

    def __init__(self):
        pass

    def run_test(self):
        self.test1()

    def make_Bender_Test_Coords(self, bender: SegmentedBenderHalbach) -> np.ndarray:
        """Coords for testing bender that span the angular length and a little more."""
        ucAngle, rb, rp = bender.UCAngle, bender.rb, bender.rp
        thetaArr = np.linspace(bender.lensAnglesArr.min() - 2 * ucAngle, bender.lensAnglesArr.max() + 2 * ucAngle,
                               2 * bender.numLenses)
        rArr = np.linspace(rb - rp / 2, rb + rp / 2, 5)
        yArr = np.linspace(-rp / 2, rp / 2, 6)
        coordsPolar = np.array(list(itertools.product(thetaArr, rArr, yArr)))
        xArr, zArr = np.cos(coordsPolar[:, 0]) * coordsPolar[:, 1], np.sin(coordsPolar[:, 0]) * coordsPolar[:, 1]
        yArr = coordsPolar[:, 2]
        coords = np.column_stack((xArr, yArr, zArr))
        return coords

    def test_Bender_Approx(self, bender: SegmentedBenderHalbach) -> None:
        """Test that the bender satisifes speed and accuracy limits over it's length for exact and approximate
        values."""
        coords = self.make_Bender_Test_Coords(bender)
        t = time.time()
        BVec_Exact = bender.B_vec(coords)
        t1 = time.time() - t
        t = time.time()
        BVec_Approx = bender.B_vec(coords, use_approx=True)
        t2 = time.time() - t
        assert t2 < .5 * t1  # should be faster. For some cases it is much faster
        precisionCutoff = 1e-9  # absolute values below this are neglected for tolerance reasons
        nanIndices = np.abs(BVec_Exact) < precisionCutoff
        BVec_Exact[nanIndices] = np.nan
        BVec_Approx[nanIndices] = np.nan
        error = 1e2 * np.abs((BVec_Exact - BVec_Approx) / BVec_Exact)
        percentErrorMax = .015
        assert np.nanmax(error) < percentErrorMax

    def test1(self):
        """Test that the approximate method of finding field values by splitting the bender up is faster and
        still very accurate"""
        Lm, rb, rp = .0254, .9, .011
        ucAngle = .6 * Lm / 1.0
        # -------bender starting at theta=0
        bender = SegmentedBenderHalbach(rp, rb, ucAngle, Lm, numLenses=130, positiveAngleMagnetsOnly=True)
        self.test_Bender_Approx(bender)
        # ------bender symmetric about theta=0
        bender = SegmentedBenderHalbach(rp, rb, ucAngle, Lm, numLenses=80, positiveAngleMagnetsOnly=False)
        self.test_Bender_Approx(bender)


def test(parallel=True):
    def run(func):
        func()

    funcList = [SpheretestHelper().run_tests,
                LayertestHelper().run_tests,
                HalbachLenstestHelper().run_tests,
                SegmentedBenderHalbachHelper().run_test]
    processes = -1 if parallel == True else 1
    tool_Parallel_Process(run, funcList, processes=processes)
