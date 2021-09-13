import numpy.linalg as npl
import numpy as np
import numba
from math import floor


def generate_3DInterp_Function_NUMBA(vx,vy,vz,xData,yData,zData):
    assert vx.shape==vy.shape and vy.shape==vz.shape
    X,Y,Z=vx.shape[0],vx.shape[1],vx.shape[2]
    min_x,max_x=xData[0],xData[-1]
    min_y,max_y=yData[0],yData[-1]
    min_z,max_z=zData[0],zData[-1]
    delta_x=(max_x-min_x)/(xData.shape[0]-1)
    delta_y=(max_y-min_y)/(yData.shape[0]-1)
    delta_z=(max_z-min_z)/(zData.shape[0]-1)
    vx=np.ravel(vx)
    vy=np.ravel(vy)
    vz=np.ravel(vz)

    @numba.njit()
    def interp3D(x,y,z):
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
            c00_x=vx[indexA]*(1-xd)+vx[indexB]*xd
            c01_x=vx[indexC]*(1-xd)+vx[indexD]*xd
            c10_x=vx[indexE]*(1-xd)+vx[indexF]*xd
            c11_x=vx[indexG]*(1-xd)+vx[indexH]*xd
            c0_x=c00_x*(1-yd)+c10_x*yd
            c1_x=c01_x*(1-yd)+c11_x*yd
            c_x=c0_x*(1-zd)+c1_x*zd

            c00_y=vy[indexA]*(1-xd)+vy[indexB]*xd
            c01_y=vy[indexC]*(1-xd)+vy[indexD]*xd
            c10_y=vy[indexE]*(1-xd)+vy[indexF]*xd
            c11_y=vy[indexG]*(1-xd)+vy[indexH]*xd
            c0_y=c00_y*(1-yd)+c10_y*yd
            c1_y=c01_y*(1-yd)+c11_y*yd
            c_y=c0_y*(1-zd)+c1_y*zd

            c00_z=vz[indexA]*(1-xd)+vz[indexB]*xd
            c01_z=vz[indexC]*(1-xd)+vz[indexD]*xd
            c10_z=vz[indexE]*(1-xd)+vz[indexF]*xd
            c11_z=vz[indexG]*(1-xd)+vz[indexH]*xd
            c0_z=c00_z*(1-yd)+c10_z*yd
            c1_z=c01_z*(1-yd)+c11_z*yd
            c_z=c0_z*(1-zd)+c1_z*zd
        else:
            raise Exception('out of bounds')
        return c_x,c_y,c_z

    return interp3D


def generate_2DInterp_Function_NUMBA(v,xData,yData):
    X,Y=v.shape[0],v.shape[1]
    min_x,max_x=xData[0],xData[-1]
    min_y,max_y=yData[0],yData[-1]
    delta_x=(max_x-min_x)/(xData.shape[0]-1)
    delta_y=(max_y-min_y)/(yData.shape[0]-1)
    v_c=np.ravel(v)

    @numba.njit(numba.float64(numba.float64,numba.float64))
    def interp2D(x,y):
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

    return interp2D