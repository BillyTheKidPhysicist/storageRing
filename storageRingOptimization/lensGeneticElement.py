import time
import matplotlib.pyplot as plt
import numpy as np
import numba
from elementPT import LensIdeal
from geneticLensClass import GeneticLens



@numba.njit()
def genetic_Lens_Force_NUMBA(x,y,z, L,ap,  force_Func):
    FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
    FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
    y = abs(y)  # confine to upper right quadrant
    z = abs(z)
    if np.sqrt(y**2+z**2)>ap:
        return np.nan,np.nan,np.nan
    if 0<=x <=L/2:
        x = L/2 - x
        Fx,Fy,Fz= force_Func(x, y, z)
        Fx=-Fx
    elif L/2<x<L:
        x=x-L/2
        Fx,Fy,Fz = force_Func(x, y, z)
    else:
        return np.nan,np.nan,np.nan
    Fy = Fy * FySymmetryFact
    Fz = Fz * FzSymmetryFact
    return Fx,Fy,Fz



TINY_STEP=1e-9
class geneticLensElement(LensIdeal):
    def __init__(self,PTL, geneticLens:GeneticLens,ap):
        #if rp is set to None, then the class sets rp to whatever the comsol data is. Otherwise, it scales values
        #to accomdate the new rp such as force values and positions
        # super().__init__(PTL, geneticLens.length, geneticLens.maximum_Radius(), np.nan,np.nan,'injector',fillParams=False)
        super().__init__(PTL, geneticLens.length, None,geneticLens.maximum_Radius(), ap,fillParams=False)
        self.fringeFracOuter=4.0
        self.L=geneticLens.length+2*self.fringeFracOuter*self.rp
        self.Lo=None
        self.type = 'STRAIGHT'
        self.lens=geneticLens
        assert self.lens.minimum_Radius()>=ap
        self.fringeFracInnerMin=np.inf #if the total hard edge magnet length is longer than this value * rp, then it can
        #can safely be modeled as a magnet "cap" with a 2D model of the interior
        self.lengthEffective=None #if the magnet is very long, to save simulation
        #time use a smaller length that still captures the physics, and then model the inner portion as 2D

        self.magnetic_Potential_Func_Fringe = None
        self.magnetic_Potential_Func_Inner = None
        self.fieldFact = 1.0 #factor to multiply field values by for tunability
        if self.L is not None:
            self.fill_Params()

    def set_Length(self,L):
        assert L>0.0
        self.L=L
        self.fill_Params()
    def fill_Params(self,externalDataProvided=False):
        #todo: more robust way to pick number of points in element. It should be done by using the typical lengthscale
        #of the bore radius

        numPointsLongitudinal=35
        numPointsTransverse=31

        self.Lm=self.L-2*self.fringeFracOuter*self.rp  #hard edge length of magnet
        assert np.abs(self.Lm-self.lens.length)<1e-6
        assert self.Lm>0.0
        self.Lo=self.L


        numXY=numPointsTransverse
        #because the magnet here is orienated along z, and the field will have to be titled to be used in the particle
        #tracer module, and I want to exploit symmetry by computing only one quadrant, I need to compute the upper left
        #quadrant here so when it is rotated -90 degrees about y, that becomes the upper right in the y,z quadrant
        yArr_Quadrant=np.linspace(-TINY_STEP,self.ap+TINY_STEP,numXY)
        xArr_Quadrant=np.linspace(-(self.ap+TINY_STEP),TINY_STEP,numXY)


        zMin = -TINY_STEP
        zMaxHalf = self.L / 2+TINY_STEP
        zArr = np.linspace(zMin, zMaxHalf, num=numPointsLongitudinal)  # add a little extra so interp works as expected

        # assert (zArr[-1]-zArr[-2])/self.rp<.2, "spatial step size must be small compared to radius"
        assert len(xArr_Quadrant)%2==1 and len(yArr_Quadrant)%2==1
        assert all((arr[-1]-arr[-2])/self.rp<.1 for arr in [xArr_Quadrant,yArr_Quadrant]),"" \
                                                                    "spatial step size must be small compared to radius"


        volumeCoords=np.asarray(np.meshgrid(xArr_Quadrant,yArr_Quadrant,zArr)).T.reshape(-1,3) #note that these coordinates can have
        #the wrong value for z if the magnet length is longer than the fringe field effects. This is intentional and
        #input coordinates will be shifted in a wrapper function
        BNormGrad,BNorm = self.lens.BNorm_Gradient(volumeCoords,returnNorm=True)
        data3D = np.column_stack((volumeCoords, BNormGrad, BNorm))
        self.fill_Field_Func(data3D)
        # self.compile_Fast_Numba_Force_Function()

        F_edge = np.linalg.norm(self.force(np.asarray([0.0, self.ap / 2, .0])))
        F_center = np.linalg.norm(self.force(np.asarray([self.L/2, self.ap / 2, .0])))
        # assert F_edge / F_center < .01


    def compile_Fast_Numba_Force_Function(self):
        forceNumba = genetic_Lens_Force_NUMBA
        L=self.L
        ap=self.ap
        force_Func=self.force_Func
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_NUMBA_Wrapper(x,y,z):
            return forceNumba(x,y,z,L,ap,force_Func)
        self.fast_Numba_Force_Function=force_NUMBA_Wrapper
    def force(self, q,searchIsCoordInside=True):
        F=genetic_Lens_Force_NUMBA(q[0],q[1],q[2],self.L,self.ap,self.force_Func)
        # if np.isnan(F[0])==False:
        #     if q[0]<2*self.rp*self.fringeFracOuter or q[0]>self.L-2*self.rp*self.fringeFracOuter:
        #         return np.zeros(3)
        F=self.fieldFact*np.asarray(F)
        return F
    def fill_Field_Func(self,data):
        interpF, interpV = self.make_Interp_Functions(data)
        # wrap the function in a more convenietly accesed function
        @numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64))
        def force_Func(x,y,z):
            Fx0,Fy0,Fz0=interpF(-z, y,x)
            Fx=Fz0
            Fy=Fy0
            Fz=-Fx0
            return Fx,Fy,Fz
        self.force_Func=force_Func
        self.magnetic_Potential_Func = lambda x, y, z: interpV(-z, y, x)

    def magnetic_Potential(self, q):
        # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
        x, y, z = q
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if self.is_Coord_Inside(q) == False:
            raise Exception(ValueError)
        
        if 0 <= x <= self.L / 2:
            x = self.L / 2 - x
            V = self.magnetic_Potential_Func(x, y, z)
        elif self.L / 2 < x:
            x = x - self.L / 2
            V = self.magnetic_Potential_Func(x, y, z)
        else:
            raise Exception(ValueError)
        return V


