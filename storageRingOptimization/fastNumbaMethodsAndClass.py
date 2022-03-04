import numpy.linalg as npl
import numpy as np
import numba
from math import floor
from constants import SIMULATION_MAGNETON
@numba.njit()
def scalar_interp3D(x,y,z,xCoords,yCoords,zCoords,vec):
    X,Y,Z=len(xCoords),len(yCoords),len(zCoords)
    assert 2<X and 2<Y and 2<Z, "need at least 2 points to interpolate"
    min_x,max_x=xCoords[0],xCoords[-1]
    min_y,max_y=yCoords[0],yCoords[-1]
    min_z,max_z=zCoords[0],zCoords[-1]
    delta_x=(max_x-min_x)/(xCoords.shape[0]-1)
    delta_y=(max_y-min_y)/(yCoords.shape[0]-1)
    delta_z=(max_z-min_z)/(zCoords.shape[0]-1)

    x=(x-min_x)/delta_x
    y=(y-min_y)/delta_y
    z=(z-min_z)/delta_z
    x0=int(x)
    x1=x0+1
    y0=int(y)
    y1=y0+1
    z0=int(z)
    z1=z0+1
    xd=(x-x0)/(x1-x0)
    yd=(y-y0)/(y1-y0)
    zd=(z-z0)/(z1-z0)
    indexA=Y*Z*x0+Z*y0+z0
    indexB=Y*Z*x1+Z*y0+z0
    indexC=Y*Z*x0+Z*y0+z1
    indexD=Y*Z*x1+Z*y0+z1
    indexE=Y*Z*x0+Z*y1+z0
    indexF=Y*Z*x1+Z*y1+z0
    indexG=Y*Z*x0+Z*y1+z1
    indexH=Y*Z*x1+Z*y1+z1

    if x>=0.0 and y>=0.0 and z>=0.0 and x1<X and y1<Y and z1<Z:
        c00=vec[indexA]*(1-xd)+vec[indexB]*xd
        c01=vec[indexC]*(1-xd)+vec[indexD]*xd
        c10=vec[indexE]*(1-xd)+vec[indexF]*xd
        c11=vec[indexG]*(1-xd)+vec[indexH]*xd
        c0=c00*(1-yd)+c10*yd
        c1=c01*(1-yd)+c11*yd
        c=c0*(1-zd)+c1*zd
    else:
        raise Exception('out of bounds')

    return c

@numba.njit()
def vec_interp3D(x,y,z,xCoords,yCoords,zCoords,vecX,vecY,vecZ):
    X,Y,Z=len(xCoords),len(yCoords),len(zCoords)
    assert 2<X and 2<Y and 2<Z, "need at least 2 points to interpolate"
    min_x,max_x=xCoords[0],xCoords[-1]
    min_y,max_y=yCoords[0],yCoords[-1]
    min_z,max_z=zCoords[0],zCoords[-1]
    delta_x=(max_x-min_x)/(xCoords.shape[0]-1)
    delta_y=(max_y-min_y)/(yCoords.shape[0]-1)
    delta_z=(max_z-min_z)/(zCoords.shape[0]-1)

    x=(x-min_x)/delta_x
    y=(y-min_y)/delta_y
    z=(z-min_z)/delta_z
    x0=int(x)
    x1=x0+1
    y0=int(y)
    y1=y0+1
    z0=int(z)
    z1=z0+1
    xd=(x-x0)/(x1-x0)
    yd=(y-y0)/(y1-y0)
    zd=(z-z0)/(z1-z0)
    indexA=Y*Z*x0+Z*y0+z0
    indexB=Y*Z*x1+Z*y0+z0
    indexC=Y*Z*x0+Z*y0+z1
    indexD=Y*Z*x1+Z*y0+z1
    indexE=Y*Z*x0+Z*y1+z0
    indexF=Y*Z*x1+Z*y1+z0
    indexG=Y*Z*x0+Z*y1+z1
    indexH=Y*Z*x1+Z*y1+z1

    if x>=0.0 and y>=0.0 and z>=0.0 and x1<X and y1<Y and z1<Z:
        c00_x=vecX[indexA]*(1-xd)+vecX[indexB]*xd
        c01_x=vecX[indexC]*(1-xd)+vecX[indexD]*xd
        c10_x=vecX[indexE]*(1-xd)+vecX[indexF]*xd
        c11_x=vecX[indexG]*(1-xd)+vecX[indexH]*xd
        c0_x=c00_x*(1-yd)+c10_x*yd
        c1_x=c01_x*(1-yd)+c11_x*yd
        c_x=c0_x*(1-zd)+c1_x*zd
        c00_y=vecY[indexA]*(1-xd)+vecY[indexB]*xd
        c01_y=vecY[indexC]*(1-xd)+vecY[indexD]*xd
        c10_y=vecY[indexE]*(1-xd)+vecY[indexF]*xd
        c11_y=vecY[indexG]*(1-xd)+vecY[indexH]*xd
        c0_y=c00_y*(1-yd)+c10_y*yd
        c1_y=c01_y*(1-yd)+c11_y*yd
        c_y=c0_y*(1-zd)+c1_y*zd
        c00_z=vecZ[indexA]*(1-xd)+vecZ[indexB]*xd
        c01_z=vecZ[indexC]*(1-xd)+vecZ[indexD]*xd
        c10_z=vecZ[indexE]*(1-xd)+vecZ[indexF]*xd
        c11_z=vecZ[indexG]*(1-xd)+vecZ[indexH]*xd
        c0_z=c00_z*(1-yd)+c10_z*yd
        c1_z=c01_z*(1-yd)+c11_z*yd
        c_z=c0_z*(1-zd)+c1_z*zd
    else:
        raise Exception('out of bounds')

    return c_x,c_y,c_z





@numba.njit()
def interp2D(x,y,xCoords,yCoords,v_c):
    X, Y = len(xCoords),len(yCoords)
    min_x, max_x = xCoords[0], xCoords[-1]
    min_y, max_y = yCoords[0], yCoords[-1]
    delta_x = (max_x - min_x) / (xCoords.shape[0] - 1)
    delta_y = (max_y - min_y) / (yCoords.shape[0] - 1)
    x=(x-min_x)/delta_x
    y=(y-min_y)/delta_y
    x0=int(x)
    x1=x0+1
    y0=int(y)
    y1=y0+1
    xd=(x-x0)/(x1-x0)
    yd=(y-y0)/(y1-y0)
    if x0>=0 and y0>=0 and x1<X and y1<Y:
        c00=v_c[Y*x0+y0]*(1-xd)+v_c[Y*x1+y0]*xd
        c10=v_c[Y*x0+y1]*(1-xd)+v_c[Y*x1+y1]*xd
        c=c00*(1-yd)+c10*yd
    else:
        raise Exception('out of bounds')
    return c
@numba.njit()
def full_Arctan2(y,x):
    phi = np.arctan2(y,x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi

from numba.experimental import jitclass
from numba.typed import List

#xArrEnd,yArrEnd,zArrEnd,FxArrEnd,FyArrEnd,FzArrEnd,VArrEnd,xArrIn,yArrIn,FxArrIn,FyArrIn,VArrIn
spec = [
    ('V_ArrFringe',numba.float64[::1]),
    ('xArrEnd',numba.float64[::1]),
    ('yArrEnd',numba.float64[::1]),
    ('zArrEnd',numba.float64[::1]),
    ('FxArrEnd',numba.float64[::1]),
    ('FyArrEnd',numba.float64[::1]),
    ('FzArrEnd',numba.float64[::1]),
    ('VArrEnd',numba.float64[::1]),
    ('xArrIn',numba.float64[::1]),
    ('yArrIn',numba.float64[::1]),
    ('FxArrEnd',numba.float64[::1]),
    ('FyArrEnd',numba.float64[::1]),
    ('VArrEnd',numba.float64[::1]),
    ('L', numba.float64),
    ('Lcap', numba.float64),
    ('ap', numba.float64),
]

# @jitclass(spec)
class LensHalbachFieldHelper_Numba:
    def __init__(self,fieldData,L,Lcap,ap):
        self.xArrEnd,self.yArrEnd,self.zArrEnd,self.FxArrEnd,self.FyArrEnd,self.FzArrEnd,self.VArrEnd,self.xArrIn,\
        self.yArrIn,self.FxArrIn,self.FyArrIn,self.VArrIn=fieldData
        self.L=L
        self.Lcap=Lcap
        self.ap=ap
    def _magnetic_Potential_Func_Fringe(self,x,y,z):
        V=scalar_interp3D(-z, y, x,self.xArrEnd,self.yArrEnd,self.zArrEnd,self.VArrEnd)
        return V

    def _magnetic_Potential_Func_Inner(self,x,y,z):
        V=interp2D(-z, y,self.xArrIn,self.yArrIn,self.VArrIn)
        return V
    def _force_Func_Outer(self,x,y,z):
        Fx0,Fy0,Fz0=vec_interp3D(-z, y,x,self.xArrEnd,self.yArrEnd,self.zArrEnd,
                                 self.FxArrEnd, self.FyArrEnd,self.FzArrEnd)
        Fx = Fz0
        Fy = Fy0
        Fz = -Fx0
        return Fx,Fy,Fz
    def _force_Func_Inner(self,x,y,z):
        Fx = 0.0
        Fy = interp2D(-z,y, self.xArrIn,self.yArrIn,self.FyArrIn)
        Fz = -interp2D(-z,y,self.xArrIn,self.yArrIn,self.FxArrIn)
        return Fx, Fy, Fz
    def force(self,x, y, z):
        if np.sqrt(y ** 2 + z ** 2) > self.ap:
            return np.nan, np.nan, np.nan
        FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if 0 <= x and x<= self.Lcap:
            x = self.Lcap - x
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
            Fx = -Fx
        elif self.Lcap < x and x <= self.L - self.Lcap:
            Fx=0.0
            Fy = interp2D(-z, y,  self.xArrIn, self.yArrIn, self.FyArrIn)
            Fz = -interp2D(-z, y, self.xArrIn, self.yArrIn, self.FxArrIn)
        elif 0 <= x and x <= self.L:
            x = self.Lcap - (self.L - x)
            Fx, Fy, Fz = self._force_Func_Outer(x, y, z)
        else:
            return np.nan, np.nan, np.nan
        Fy = Fy * FySymmetryFact
        Fz = Fz * FzSymmetryFact
        return Fx, Fy, Fz
    def magnetic_Potential(self, x,y,z):
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if x <= self.Lcap:
            x = self.Lcap - x
            V0 = self._magnetic_Potential_Func_Fringe(x, y, z)
        elif self.Lcap < x <= self.L - self.Lcap:
            V0 = self._magnetic_Potential_Func_Inner(x,y,z)
        elif x <= self.L: #this one is tricky with the scaling
            x=self.Lcap-(self.L-x)
            V0 = self._magnetic_Potential_Func_Fringe(x, y, z)
        else:
            V0=0
        return V0


spec = [
    ('L', numba.float64),
    ('K', numba.float64),
    ('ap', numba.float64),
]
@jitclass(spec)
class IdealLensFieldHelper_Numba:
    def __init__(self,L,K,ap):
        self.L=L
        self.K=K
        self.ap=ap
    def magnetic_Potential(self, x,y,z):
        # potential energy at provided coordinates
        # q coords in element frame
        r = np.sqrt(y ** 2 + +z** 2)
        if r < self.ap:
            return .5*self.K * r ** 2
        else:
            return 0.0
    def force(self, x,y,z):
        # note: for the perfect lens, in it's frame, there is never force in the x direction. Force in x is always zero
        if 0 <= x <= self.L and y ** 2 + z ** 2 < self.ap**2:
            Fx=0.0
            Fy = -self.K * y
            Fz = -self.K * z
            return Fx,Fy,Fz
        else:
            return np.nan,np.nan,np.nan
spec = [
    ('L', numba.float64),
    ('ap', numba.float64),
]
@jitclass(spec)
class DriftFieldHelper_Numba:
    def __init__(self,L,ap):
        self.L=L
        self.ap=ap
    def magnetic_Potential(self, x,y,z):
        # potential energy at provided coordinates
        # q coords in element frame
        return 0.0
    def force(self, x,y,z):
        # note: for the perfect lens, in it's frame, there is never force in the x direction. Force in x is always zero
        if 0 <= x <= self.L and np.sqrt(y ** 2 + z ** 2) < self.ap:
            return 0.0,0.0,0.0
        else:
            return np.nan, np.nan, np.nan

spec = [
    ('ang', numba.float64),
    ('K', numba.float64),
    ('rp', numba.float64),
    ('rb', numba.float64),
    ('ap', numba.float64),
]
@jitclass(spec)
class BenderFieldHelper_Numba:
    def __init__(self,ang, K, rp, rb, ap):
        self.ang=ang
        self.K=K
        self.rp=rp
        self.rb=rb
        self.ap=ap

    def magnetic_Potential(self, x,y,z):
        # potential energy at provided coordinates
        # q coords in element frame
        r = np.sqrt(x ** 2 + +y ** 2)
        if self.rb + self.rp > r > self.rb - self.rp and np.abs(z) < self.ap:
            rNew = np.sqrt((r - self.rb) ** 2 + z ** 2)
            return .5*self.K * SIMULATION_MAGNETON * rNew ** 2
        else:
            return 0.0
    def force(self, x,y,z):
        # force at point q in element frame
        # q: particle's position in element frame
        phi = full_Arctan2(y,x)
        if phi < self.ang:
            r = np.sqrt(x ** 2 + y ** 2)  # radius in x y frame
            F0 = -self.K * (r - self.rb)  # force in x y plane
            Fx = np.cos(phi) * F0
            Fy = np.sin(phi) * F0
            Fz = -self.K * z
        else:
            Fx,Fy,Fz=np.nan,np.nan,np.nan
        return Fx,Fy,Fz
    
spec = [
    ('c1', numba.float64),
    ('c2', numba.float64),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('apL', numba.float64),
    ('apR', numba.float64),
    ('apz', numba.float64),
    ('ang', numba.float64),
]
@jitclass(spec)
class CombinerIdealFieldHelper_Numba:
    def __init__(self,c1,c2,La,Lb,apL,apR,apz,ang):
        self.c1=c1
        self.c2=c2
        self.La=La
        self.Lb=Lb
        self.apL=apL
        self.apR=apR
        self.apz=apz
        self.ang=ang
    def force(self,x,y,z):
        return self._force(x,y,z,True)
    def force_NoSearchInside(self,x,y,z):
        F= self._force(x,y,z,False)
        return F
    def _force(self, x,y,z,searchIsCoordInside):
        # force at point q in element frame
        # q: particle's position in element frame
        if searchIsCoordInside==True:
            if self.is_Coord_Inside(x,y,z) == False:
                return np.nan,np.nan,np.nan
        Fx,Fy,Fz=0.0,0.0,0.0
        if 0<x < self.Lb:
            B0 = np.sqrt((self.c2 * z) ** 2 + (self.c1 + self.c2 * y) ** 2)
            Fy = SIMULATION_MAGNETON * self.c2 * (self.c1 + self.c2 * y) / B0
            Fz = SIMULATION_MAGNETON * self.c2 ** 2 * z / B0
        else:
            pass
        return Fx,Fy,Fz
    def magnetic_Potential(self, x,y,z):
        V0=0
        if 0<x < self.Lb:
            V0 = SIMULATION_MAGNETON*np.sqrt((self.c2 * z) ** 2 + (self.c1 + self.c2 * y) ** 2)
        return V0
    def is_Coord_Inside(self, x,y,z):
        # q: coordinate to test in element's frame
        if not -self.apz <= z <= self.apz:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -self.apL < y < self.apR:  # if inside the y (width) apeture
                return True
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            m = np.tan(self.ang)
            Y1 = m * x + (self.apR - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-self.apL - m * self.Lb)
            if np.sign(m)<0.0 and (y < Y1 and y > Y2 and y > Y3): #if the inlet is tilted 'down'
                return True
            elif np.sign(m)>0.0 and (y < Y1 and y < Y2 and y > Y3): #if the inlet is tilted 'up'
                return True
            else:
                return False

spec = [
    ('VArr', numba.float64[::1]),
    ('FxArr', numba.float64[::1]),
    ('FyArr', numba.float64[::1]),
    ('FzArr', numba.float64[::1]),
    ('xArr', numba.float64[::1]),
    ('yArr', numba.float64[::1]),
    ('zArr', numba.float64[::1]),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('Lm', numba.float64),
    ('space', numba.float64),
    ('apL', numba.float64),
    ('apR', numba.float64),
    ('apz', numba.float64),
    ('ang', numba.float64),
]
@jitclass(spec)
class CombinerSimFieldHelper_Numba:
    def __init__(self,fieldData,La,Lb,Lm,space,apL,apR,apz,ang):
        self.xArr,self.yArr,self.zArr,self.FxArr,self.FyArr,self.FzArr,self.VArr=fieldData
        self.La=La
        self.Lb=Lb
        self.Lm=Lm
        self.space=space
        self.apL=apL
        self.apR=apR
        self.apz=apz
        self.ang=ang
    def _force_Func(self,x,y,z):
        return vec_interp3D(x,y,z,self.xArr,self.yArr,self.zArr,self.FxArr,self.FyArr,self.FzArr)
    def _magnetic_Potential_Func(self,x,y,z):
        return scalar_interp3D(x,y,z,self.xArr,self.yArr,self.zArr,self.VArr)
    def force(self,x,y,z):
        return self._force(x,y,z,True)
    def force_NoSearchInside(self,x,y,z):
        F= self._force(x,y,z,False)
        return F
    def _force(self,x, y, z,searchIsCoordInside):
        # this function uses the symmetry of the combiner to extract the force everywhere.
        # I believe there are some redundancies here that could be trimmed to save time.
        if searchIsCoordInside == True:
            if not -self.apz <= z <= self.apz:  # if outside the z apeture (vertical)
                return np.nan, np.nan, np.nan
            elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
                # through the combiner. Simple square apeture
                if -self.apL <= y <= self.apR:  # if inside the y (width) apeture
                    pass
                else:
                    return np.nan, np.nan, np.nan
            elif x < 0:
                return np.nan, np.nan, np.nan
            else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
                # todo: better modeled as a simpler rotation?
                m = np.tan(self.ang)
                Y1 = m * x + (self.apR - m * self.Lb)  # upper limit
                Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
                Y3 = m * x + (-self.apL - m * self.Lb)
                if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                    pass
                elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                    pass
                else:
                    return np.nan, np.nan, np.nan
        xFact = 1  # value to modify the force based on symmetry
        zFact = 1
        if 0 <= x <= (self.Lm / 2 + self.space):  # if the particle is in the first half of the magnet
            if z < 0:  # if particle is in the lower plane
                z = -z  # flip position to upper plane
                zFact = -1  # z force is opposite in lower half
        elif (self.Lm / 2 + self.space) < x:  # if the particle is in the last half of the magnet
            x = (self.Lm / 2 + self.space) - (x - (self.Lm / 2 + self.space))  # use the reflection of the particle
            xFact = -1  # x force is opposite in back plane
            if z < 0:  # if in the lower plane, need to use symmetry
                z = -z
                zFact = -1  # z force is opposite in lower half
        Fx, Fy, Fz = self._force_Func(x, y, z)
        Fx = xFact * Fx
        Fz = zFact * Fz
        return Fx, Fy, Fz
    def magnetic_Potential(self, x,y,z):
        # this function uses the symmetry of the combiner to extract the magnetic potential everywhere.
        if 0 <= x <= (self.Lm / 2 + self.space):  # if the particle is in the first half of the magnet
            if z < 0:  # if particle is in the lower plane
                z = -z  # flip position to upper plane
        if (self.Lm / 2 + self.space) < x:  # if the particle is in the last half of the magnet
            x = (self.Lm / 2 + self.space) - (
                        x - (self.Lm / 2 + self.space))  # use the reflection of the particle
            if z < 0:  # if in the lower plane, need to use symmetry
                z = -z
        return self._magnetic_Potential_Func(x,y,z)

    def is_Coord_Inside(self, x,y,z):
        # q: coordinate to test in element's frame
        if not -self.apz <= z <= self.apz:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner. Simple square apeture
            if -self.apL < y < self.apR:  # if inside the y (width) apeture
                return True
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            m = np.tan(self.ang)
            Y1 = m * x + (self.apR - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-self.apL - m * self.Lb)
            if np.sign(m)<0.0 and (y < Y1 and y > Y2 and y > Y3): #if the inlet is tilted 'down'
                return True
            elif np.sign(m)>0.0 and (y < Y1 and y < Y2 and y > Y3): #if the inlet is tilted 'up'
                return True
            else:
                return False

spec = [
    ('VArr', numba.float64[::1]),
    ('FxArr', numba.float64[::1]),
    ('FyArr', numba.float64[::1]),
    ('FzArr', numba.float64[::1]),
    ('xArr', numba.float64[::1]),
    ('yArr', numba.float64[::1]),
    ('zArr', numba.float64[::1]),
    ('La', numba.float64),
    ('Lb', numba.float64),
    ('Lm', numba.float64),
    ('space', numba.float64),
    ('ap', numba.float64),
    ('ang', numba.float64),
    ('fieldFact', numba.float64),
]
# @jitclass(spec)
class CombinerHexapoleSimFieldHelper_Numba:
    def __init__(self,fieldData,La,Lb,Lm,space,ap,ang):
        self.xArr,self.yArr,self.zArr,self.FxArr,self.FyArr,self.FzArr,self.VArr=fieldData
        self.La=La
        self.Lb=Lb
        self.Lm=Lm
        self.space=space
        self.ap=ap
        self.ang=ang
        self.fieldFact=1.0
    def _force_Func(self,x,y,z):
        Fx0, Fy0, Fz0= vec_interp3D(-z,y,x,self.xArr,self.yArr,self.zArr,self.FxArr,self.FyArr,self.FzArr)
        Fx = Fz0
        Fy = Fy0
        Fz = -Fx0
        return Fx, Fy, Fz
    def _magnetic_Potential_Func(self,x,y,z):
        return scalar_interp3D(-z,y,x,self.xArr,self.yArr,self.zArr,self.VArr)
    def force(self,x,y,z):
        return self._force(x,y,z,True)
    def force_NoSearchInside(self,x,y,z):
        F= self._force(x,y,z,False)
        return F
    def _force(self,x, y, z,searchIsCoordInside):
        # this function uses the symmetry of the combiner to extract the force everywhere.
        # I believe there are some redundancies here that could be trimmed to save time.
        if searchIsCoordInside == True:
            if not -self.ap <= z <= self.ap:  # if outside the z apeture (vertical)
                return np.nan, np.nan, np.nan
            elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
                # through the combiner.
                if np.sqrt(y ** 2 + z ** 2) < self.ap:
                    pass
                else:
                    return np.nan, np.nan, np.nan
            elif x < 0:
                return np.nan, np.nan, np.nan
            else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
                # todo: For now a square aperture, update to circular. Use a simple rotation
                m = np.tan(self.ang)
                Y1 = m * x + (self.ap - m * self.Lb)  # upper limit
                Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
                Y3 = m * x + (-self.ap - m * self.Lb)
                if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                    pass
                elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                    pass
                else:
                    return np.nan, np.nan, np.nan
        FySymmetryFact = 1.0 if y >= 0.0 else -1.0  # take advantage of symmetry
        FzSymmetryFact = 1.0 if z >= 0.0 else -1.0
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        symmetryLength = self.Lm + 2 * self.space
        if 0 <= x <= symmetryLength / 2:
            x = symmetryLength / 2 - x
            Fx, Fy, Fz = self._force_Func(x, y, z)
            Fx = -Fx
        elif symmetryLength / 2 < x:
            x = x - symmetryLength / 2
            Fx, Fy, Fz = self._force_Func(x, y, z)
        else:
            raise Exception(ValueError)
        Fy = Fy * FySymmetryFact
        Fz = Fz * FzSymmetryFact
        Fx, Fy, Fz=self.fieldFact*Fx, self.fieldFact*Fy, self.fieldFact*Fz
        return Fx, Fy, Fz
    def magnetic_Potential(self, x,y,z):
        y = abs(y)  # confine to upper right quadrant
        z = abs(z)
        if self.is_Coord_Inside(x,y,z) == False:
            raise Exception(ValueError)
        symmetryLength = self.Lm + 2 * self.space
        if 0 <= x <= symmetryLength / 2:
            x = symmetryLength / 2 - x
            V = self._magnetic_Potential_Func(x, y, z)
        elif symmetryLength / 2 < x:
            x = x - symmetryLength / 2
            V = self._magnetic_Potential_Func(x, y, z)
        else:
            raise Exception(ValueError)
        return V

    def is_Coord_Inside(self, x,y,z):
        # q: coordinate to test in element's frame
        if not -self.ap <= z <= self.ap:  # if outside the z apeture (vertical)
            return False
        elif 0 <= x <= self.Lb:  # particle is in the horizontal section (in element frame) that passes
            # through the combiner.
            if np.sqrt(y ** 2 + z ** 2) < self.ap:
                pass
            else:
                return False
        elif x < 0:
            return False
        else:  # particle is in the bent section leading into combiner. It's bounded by 3 lines
            # todo: For now a square aperture, update to circular. Use a simple rotation
            m = np.tan(self.ang)
            Y1 = m * x + (self.ap - m * self.Lb)  # upper limit
            Y2 = (-1 / m) * x + self.La * np.sin(self.ang) + (self.Lb + self.La * np.cos(self.ang)) / m
            Y3 = m * x + (-self.ap - m * self.Lb)
            if np.sign(m) < 0.0 and (y < Y1 and y > Y2 and y > Y3):  # if the inlet is tilted 'down'
                return True
            elif np.sign(m) > 0.0 and (y < Y1 and y < Y2 and y > Y3):  # if the inlet is tilted 'up'
                return True
            else:
                return False