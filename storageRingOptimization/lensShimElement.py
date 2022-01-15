import time
import numba
from HalbachLensClass import Sphere
import numpy as np
from elementPT import LensIdeal
from HalbachLensClass import HalbachLens
import fastElementNUMBAFunctions
import numpy.linalg as npl
import joblib
import multiprocessing

TINY_STEP = 1e-9


class ShimmedInjectorLens(LensIdeal):
    def __init__(self, PTL, rp, L, apFrac,parallel=False):
        super().__init__(PTL, L, None, rp, rp * apFrac, fillParams=False)
        self.numXY_SpatialSteps = 31
        self.numZ_SpatialSteps = 61
        self.fringeFrac = 3.0
        self.Lm = L - 2 * self.fringeFrac * rp
        assert self.Lm > 0.0
        self.Lo = self.L
        magnetWidth = self.rp * np.tan(2 * np.pi / 24) * 2
        self.yokeWidth=magnetWidth +5e-3 #add half a centimeter
        self.parallel=parallel
        self.BVecCoordsArr = None  # array to hold coordinates of Bvec evaluations
        self.BVecArr_Lens = None
        self.gradCoordArr = None  # the gradient is calculated using central difference, therefore the coordinates of the
        # gradient is difference than the coordinate of the b field vector points
        self.BNormArr = None
        self.BNormGradArr = None
        self.magnetic_Potential_Func = None
        self.force_Func = None
        self.gradStepSize = 1e-6
        self.fill_Coordinate_Arrays()
        self.fill_B_Vector_Values_Eighth_Lens()

        dummyValues = np.zeros(self.BVecCoordsArr.shape)
        self.update_Force_And_Potential_Function(dummyValues)
    def fill_Coordinate_Arrays(self):
        zMin = -TINY_STEP
        zMax = self.L / 2 + TINY_STEP
        xMin = -(self.ap + TINY_STEP)
        xMax = TINY_STEP
        yMin = -TINY_STEP
        yMax = self.ap + TINY_STEP

        xArr = np.linspace(xMin, xMax, self.numXY_SpatialSteps)
        yArr = np.linspace(yMin, yMax, self.numXY_SpatialSteps)
        zArr = np.linspace(zMin, zMax, self.numZ_SpatialSteps)
        coordList = []
        for x in xArr:
            for y in yArr:
                for z in zArr:
                    coordList.append([x - self.gradStepSize, y, z])
                    coordList.append([x + self.gradStepSize, y, z])
                    coordList.append([x, y - self.gradStepSize, z])
                    coordList.append([x, y + self.gradStepSize, z])
                    coordList.append([x, y, z - self.gradStepSize])
                    coordList.append([x, y, z + self.gradStepSize])
        self.BVecCoordsArr = np.asarray(coordList)
        self.gradCoordArr = (self.BVecCoordsArr[4::6] + self.BVecCoordsArr[
                                                        5::6]) / 2  # gradient coordinate is at midpoint
    def update_Lens_Radius(self,rList):
        #rList: list of radi for each layer of lens
        raise Exception("not implemented yet")
    def fill_B_Vector_Values_Eighth_Lens(self):
        # fill field values assuming a vertical lens with the upper half and only one quadrant
        magnetWidth = self.rp * np.tan(2 * np.pi / 24) * 2
        lens = HalbachLens(self.Lm, (magnetWidth,), (self.rp,),numSpherePerDim=3)
        if self.parallel==True: njobs=-1
        else: njobs=1
        numChunks=2*multiprocessing.cpu_count()
        assert self.BVecCoordsArr.shape[0]>numChunks
        parallelCoords=np.array_split(self.BVecCoordsArr,numChunks)
        results=joblib.Parallel(n_jobs=njobs)(joblib.delayed(lens.B_Vec)(coord) for coord in parallelCoords)
        self.BVecArr_Lens=np.row_stack(results)
    def fill_Gradient_And_Norm_And_Array(self, newBVecValues):
        assert newBVecValues.shape == self.BVecArr_Lens.shape
        BVecArrNew = self.BVecArr_Lens + newBVecValues

        # now get the derivative
        B0Gradx = (npl.norm(BVecArrNew[1::6], axis=1) - npl.norm(BVecArrNew[0::6], axis=1)) / (2 * self.gradStepSize)
        B0Grady = (npl.norm(BVecArrNew[3::6], axis=1) - npl.norm(BVecArrNew[2::6], axis=1)) / (2 * self.gradStepSize)
        B0Gradz = (npl.norm(BVecArrNew[5::6], axis=1) - npl.norm(BVecArrNew[4::6], axis=1)) / (2 * self.gradStepSize)
        self.BNormGradArr = np.column_stack((B0Gradx, B0Grady, B0Gradz))
        BNormArrIntermediate = np.zeros(BVecArrNew[0::6].shape)
        for i in range(6):
            BNormArrIntermediate += BVecArrNew[i::6]
        self.BNormArr = npl.norm(BNormArrIntermediate / 6.0, axis=1)

        # BGradx=BVecArrNew[1::6][:,0]-BVecArrNew[0::6][:,0]
        # BGrady=BVecArrNew[3::6][:,1]-BVecArrNew[2::6][:,1]
        # BGradz=BVecArrNew[5::6][:,2]-BVecArrNew[4::6][:,2]
        # maxwellArr=BGradx+BGrady+BGradz
        # print(np.sum(maxwellArr))

    def update_Force_And_Potential_Function(self, newBVecValues):
        self.fill_Gradient_And_Norm_And_Array(newBVecValues)
        data = np.column_stack((self.gradCoordArr, self.BNormGradArr, self.BNormArr))

        interpF, interpV = self.make_Interp_Functions(data)

        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_Func_Outer(x, y, z):
            Fx0, Fy0, Fz0 = interpF(-z, y, x)
            Fx = Fz0
            Fy = Fy0
            Fz = -Fx0
            return Fx, Fy, Fz
        self.force_Func = force_Func_Outer
        self.magnetic_Potential_Func = lambda x, y, z: interpV(-z, y, x)
        self.compile_Fast_Numba_Force_Function()

    def magnetic_Potential(self, q):
        x, y, z = q
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if 0 <= x <= self.L / 2:
            x = self.L / 2 - x
            V0 = self.magnetic_Potential_Func(x, y, z)
        elif self.L / 2 < x <= self.L:  # this one is tricky with the scaling
            x = x - self.L / 2
            V0 = self.magnetic_Potential_Func(x, y, z)
        else:
            V0 = 0
        return V0

    def force(self, q):
        F = fastElementNUMBAFunctions.lens_Shim_Halbach_Force_NUMBA(q, self.L, self.ap, self.force_Func)
        F = np.asarray(F)
        return self.fieldFact * F

    def compile_Fast_Numba_Force_Function(self):
        forceNumba = fastElementNUMBAFunctions.lens_Shim_Halbach_Force_NUMBA
        L = self.L
        ap = self.ap
        force_Func = self.force_Func

        @numba.njit()
        def force_NUMBA_Wrapper(q):
            return forceNumba(q, L, ap, force_Func)

        self.fast_Numba_Force_Function = force_NUMBA_Wrapper
        self.fast_Numba_Force_Function(np.zeros(3))  #