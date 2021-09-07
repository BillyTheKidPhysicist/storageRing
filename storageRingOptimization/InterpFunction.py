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
    def interp3D(x0,y0,z0):
        x = (x0-min_x)/delta_x
        y = (y0-min_y)/delta_y
        z = (z0-min_z)/delta_z
        xStart = int(floor(x))
        xEnd = xStart + 1
        yStart = int(floor(y))
        yEnd = yStart + 1
        zStart = int(floor(z))
        zEnd = zStart + 1
        xd = (x-xStart)/(xEnd-xStart)
        yd = (y-yStart)/(yEnd-yStart)
        zd = (z-zStart)/(zEnd-zStart)
        if xStart >= 0 and yStart >= 0 and zStart >= 0 and xEnd < X and yEnd < Y and zEnd < Z:
            c00 = v_c[Y*Z*xStart+Z*yStart+zStart]*(1-xd) + v_c[Y*Z*xEnd+Z*yStart+zStart]*xd
            c01 = v_c[Y*Z*xStart+Z*yStart+zEnd]*(1-xd) + v_c[Y*Z*xEnd+Z*yStart+zEnd]*xd
            c10 = v_c[Y*Z*xStart+Z*yEnd+zStart]*(1-xd) + v_c[Y*Z*xEnd+Z*yEnd+zStart]*xd
            c11 = v_c[Y*Z*xStart+Z*yEnd+zEnd]*(1-xd) + v_c[Y*Z*xEnd+Z*yEnd+zEnd]*xd
            c0 = c00*(1-yd) + c10*yd
            c1 = c01*(1-yd) + c11*yd
            c = c0*(1-zd) + c1*zd
        else:
            print(x0,y0,z0)
            print(xData.min(),xData.max())
            print(yData.min(),yData.max())
            print(zData.min(),zData.max())
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
        xStart = int(x)
        xEnd = xStart + 1
        yStart = int(y)
        yEnd = yStart + 1
        xd = (x-xStart)/(xEnd-xStart)
        yd = (y-yStart)/(yEnd-yStart)
        if xStart >= 0 and yStart >= 0  and xEnd < X and yEnd < Y:
            c00 = v_c[Y*xStart+yStart]*(1-xd) + v_c[Y*xEnd+yStart]*xd
            c10 = v_c[Y*xStart+yEnd]*(1-xd) + v_c[Y*xEnd+yEnd]*xd
            c = c00*(1-yd) + c10*yd
        else:
            raise Exception('out of bounds')
        return c
    return interp2D