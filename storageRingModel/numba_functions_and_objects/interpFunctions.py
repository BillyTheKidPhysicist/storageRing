"""
Contains interpolation routines and wrappers. Bi-linear and Tri-linear interpolation is used.
"""
import numba


@numba.njit()
def force_interp_3D(x, y, z, field_data):
    x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = field_data
    Fx, Fy, Fz = vec_interp3D(x, y, z, x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr)
    return Fx, Fy, Fz


@numba.njit()
def magnetic_potential_interp_3D(x, y, z, field_data_internal):
    x_arr, y_arr, z_arr, FxArr, FyArr, Fz_arr, V_arr = field_data_internal
    return scalar_interp3D(x, y, z, x_arr, y_arr, z_arr, V_arr)


@numba.njit()
def force_interp_2D(y: float, z: float, field_data):
    """Wrapper for interpolation of force fields of plane at center lens. see self.force"""
    y_arr, z_arr, FyArr, Fz_arr, V_arr = field_data
    Fx = 0.0
    Fy = interp2D(y, z, y_arr, z_arr, FyArr)
    Fz = interp2D(y, z, y_arr, z_arr, Fz_arr)
    return Fx, Fy, Fz


@numba.njit()
def magnetic_potential_interp_2D(x: float, y: float, z: float, field_data) -> float:
    """Wrapper for interpolation of magnetic fields of plane at center lens.see magnetic_potential"""
    y_arr, z_arr, FyArr, Fz_arr, V_arr = field_data
    V = interp2D(y, z, y_arr, z_arr, V_arr)
    return V


@numba.njit()
def scalar_interp3D(xLoc, yLoc, zLoc, xCoords, yCoords, zCoords, vec):
    X, Y, Z = len(xCoords), len(yCoords), len(zCoords)
    assert 2 < X and 2 < Y and 2 < Z, "need at least 2 points to interpolate"
    min_x, max_x = xCoords[0], xCoords[-1]
    min_y, max_y = yCoords[0], yCoords[-1]
    min_z, max_z = zCoords[0], zCoords[-1]
    delta_x = (max_x - min_x) / (xCoords.shape[0] - 1)
    delta_y = (max_y - min_y) / (yCoords.shape[0] - 1)
    delta_z = (max_z - min_z) / (zCoords.shape[0] - 1)

    x = (xLoc - min_x) / delta_x
    y = (yLoc - min_y) / delta_y
    z = (zLoc - min_z) / delta_z
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    z0 = int(z)
    z1 = z0 + 1
    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)
    index_a = Y * Z * x0 + Z * y0 + z0
    index_b = Y * Z * x1 + Z * y0 + z0
    index_c = Y * Z * x0 + Z * y0 + z1
    index_d = Y * Z * x1 + Z * y0 + z1
    index_e = Y * Z * x0 + Z * y1 + z0
    index_f = Y * Z * x1 + Z * y1 + z0
    index_g = Y * Z * x0 + Z * y1 + z1
    index_h = Y * Z * x1 + Z * y1 + z1

    if x >= 0.0 and y >= 0.0 and z >= 0.0 and x1 < X and y1 < Y and z1 < Z:
        c00 = vec[index_a] * (1 - xd) + vec[index_b] * xd
        c01 = vec[index_c] * (1 - xd) + vec[index_d] * xd
        c10 = vec[index_e] * (1 - xd) + vec[index_f] * xd
        c11 = vec[index_g] * (1 - xd) + vec[index_h] * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        c = c0 * (1 - zd) + c1 * zd
    else:
        print(xLoc, yLoc, zLoc)
        print(xCoords.min(), xCoords.max())
        print(yCoords.min(), yCoords.max())
        print(zCoords.min(), zCoords.max())
        raise Exception('out of bounds')

    return c


@numba.njit()
def vec_interp3D(xLoc, yLoc, zLoc, xCoords, yCoords, zCoords, vecX, vecY, vecZ):
    X, Y, Z = len(xCoords), len(yCoords), len(zCoords)
    assert 2 < X and 2 < Y and 2 < Z, "need at least 2 points to interpolate"
    min_x, max_x = xCoords[0], xCoords[-1]
    min_y, max_y = yCoords[0], yCoords[-1]
    min_z, max_z = zCoords[0], zCoords[-1]
    delta_x = (max_x - min_x) / (len(xCoords) - 1)
    delta_y = (max_y - min_y) / (len(yCoords) - 1)
    delta_z = (max_z - min_z) / (len(zCoords) - 1)
    x = (xLoc - min_x) / delta_x
    y = (yLoc - min_y) / delta_y
    z = (zLoc - min_z) / delta_z
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    z0 = int(z)
    z1 = z0 + 1
    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)
    index_a = Y * Z * x0 + Z * y0 + z0
    index_b = Y * Z * x1 + Z * y0 + z0
    index_c = Y * Z * x0 + Z * y0 + z1
    index_d = Y * Z * x1 + Z * y0 + z1
    index_e = Y * Z * x0 + Z * y1 + z0
    index_f = Y * Z * x1 + Z * y1 + z0
    index_g = Y * Z * x0 + Z * y1 + z1
    index_h = Y * Z * x1 + Z * y1 + z1

    if x >= 0.0 and y >= 0.0 and z >= 0.0 and x1 < X and y1 < Y and z1 < Z:
        c00_x = vecX[index_a] * (1 - xd) + vecX[index_b] * xd
        c01_x = vecX[index_c] * (1 - xd) + vecX[index_d] * xd
        c10_x = vecX[index_e] * (1 - xd) + vecX[index_f] * xd
        c11_x = vecX[index_g] * (1 - xd) + vecX[index_h] * xd
        c0_x = c00_x * (1 - yd) + c10_x * yd
        c1_x = c01_x * (1 - yd) + c11_x * yd
        c_x = c0_x * (1 - zd) + c1_x * zd
        c00_y = vecY[index_a] * (1 - xd) + vecY[index_b] * xd
        c01_y = vecY[index_c] * (1 - xd) + vecY[index_d] * xd
        c10_y = vecY[index_e] * (1 - xd) + vecY[index_f] * xd
        c11_y = vecY[index_g] * (1 - xd) + vecY[index_h] * xd
        c0_y = c00_y * (1 - yd) + c10_y * yd
        c1_y = c01_y * (1 - yd) + c11_y * yd
        c_y = c0_y * (1 - zd) + c1_y * zd
        c00_z = vecZ[index_a] * (1 - xd) + vecZ[index_b] * xd
        c01_z = vecZ[index_c] * (1 - xd) + vecZ[index_d] * xd
        c10_z = vecZ[index_e] * (1 - xd) + vecZ[index_f] * xd
        c11_z = vecZ[index_g] * (1 - xd) + vecZ[index_h] * xd
        c0_z = c00_z * (1 - yd) + c10_z * yd
        c1_z = c01_z * (1 - yd) + c11_z * yd
        c_z = c0_z * (1 - zd) + c1_z * zd
    else:
        print(xLoc, yLoc, zLoc)
        print(xCoords.min(), xCoords.max())
        print(yCoords.min(), yCoords.max())
        print(zCoords.min(), zCoords.max())
        raise Exception('out of bounds')
    return c_x, c_y, c_z


@numba.njit()
def interp2D(xLoc, yLoc, xCoords, yCoords, v_c):
    X, Y = len(xCoords), len(yCoords)
    min_x, max_x = xCoords[0], xCoords[-1]
    min_y, max_y = yCoords[0], yCoords[-1]
    delta_x = (max_x - min_x) / (xCoords.shape[0] - 1)
    delta_y = (max_y - min_y) / (yCoords.shape[0] - 1)
    x = (xLoc - min_x) / delta_x
    y = (yLoc - min_y) / delta_y
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    if x0 >= 0 and y0 >= 0 and x1 < X and y1 < Y:
        c00 = v_c[Y * x0 + y0] * (1 - xd) + v_c[Y * x1 + y0] * xd
        c10 = v_c[Y * x0 + y1] * (1 - xd) + v_c[Y * x1 + y1] * xd
        c = c00 * (1 - yd) + c10 * yd
    else:
        print(xLoc, yLoc)
        print(xCoords.min(), xCoords.max())
        print(yCoords.min(), yCoords.max())
        raise Exception('out of bounds')
    return c
