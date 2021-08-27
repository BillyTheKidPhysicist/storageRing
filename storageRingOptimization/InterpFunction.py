import numpy.linalg as npl
import numpy as np
import numba
from math import floor

def generate_3DInterp_Function_NUMBA(v,xData,yData,zData):
    X, Y, Z = v.shape[0], v.shape[1], v.shape[2]
    min_x, max_x = xData[0], xData[-1]
    min_y, max_y = yData[0], yData[-1]
    min_z, max_z = zData[0], zData[-1]
    delta_x = (max_x - min_x) / (xData.shape[0] - 1)
    delta_y = (max_y - min_y) / (yData.shape[0] - 1)
    delta_z = (max_z - min_z) / (zData.shape[0] - 1)
    v_c =np.ravel(v)

    @numba.njit(numba.float64(numba.float64,numba.float64,numba.float64))
    def interp3D(x,y,z):
        x = (x-min_x)/delta_x
        y = (y-min_y)/delta_y
        z = (z-min_z)/delta_z
        x0 = int(floor(x))
        x1 = x0 + 1
        y0 = int(floor(y))
        y1 = y0 + 1
        z0 = int(floor(z))
        z1 = z0 + 1
        xd = (x-x0)/(x1-x0)
        yd = (y-y0)/(y1-y0)
        zd = (z-z0)/(z1-z0)
        if x0 >= 0 and y0 >= 0 and z0 >= 0 and x1 < X and y1 < Y and z1 < Z:
            c00 = v_c[Y*Z*x0+Z*y0+z0]*(1-xd) + v_c[Y*Z*x1+Z*y0+z0]*xd
            c01 = v_c[Y*Z*x0+Z*y0+z1]*(1-xd) + v_c[Y*Z*x1+Z*y0+z1]*xd
            c10 = v_c[Y*Z*x0+Z*y1+z0]*(1-xd) + v_c[Y*Z*x1+Z*y1+z0]*xd
            c11 = v_c[Y*Z*x0+Z*y1+z1]*(1-xd) + v_c[Y*Z*x1+Z*y1+z1]*xd
            c0 = c00*(1-yd) + c10*yd
            c1 = c01*(1-yd) + c11*yd
            c = c0*(1-zd) + c1*zd
        else:
            raise Exception('out of bounds')
        return c
    return interp3D


def generate_2DInterp_Function_NUMBA(v,xData,yData):
    X, Y = v.shape[0], v.shape[1]
    min_x, max_x = xData[0], xData[-1]
    min_y, max_y = yData[0], yData[-1]
    delta_x = (max_x - min_x) / (xData.shape[0] - 1)
    delta_y = (max_y - min_y) / (yData.shape[0] - 1)
    v_c =np.ravel(v)

    @numba.njit(numba.float64(numba.float64,numba.float64))
    def interp2D(x,y):
        x = (x-min_x)/delta_x
        y = (y-min_y)/delta_y
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1
        xd = (x-x0)/(x1-x0)
        yd = (y-y0)/(y1-y0)
        if x0 >= 0 and y0 >= 0  and x1 < X and y1 < Y:
            c00 = v_c[Y*x0+y0]*(1-xd) + v_c[Y*x1+y0]*xd
            c10 = v_c[Y*x0+y1]*(1-xd) + v_c[Y*x1+y1]*xd
            c = c00*(1-yd) + c10*yd
        else:
            raise Exception('out of bounds')
        return c
    return interp2D