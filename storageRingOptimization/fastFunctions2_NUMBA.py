import numba
import numpy as np
import math
from math import sqrt


#---Global Variables---
BoltzmannConstant = 1.38064852e-23

mLi_kg = 1.1525801e-26
mHe_kg = 6.6464731e-27
u_kg = mLi_kg*mHe_kg/(mLi_kg+mHe_kg)

mLi = 1.
mHe = mLi * 0.576660407
u = mLi * mHe / (mLi + mHe)


@numba.njit(numba.types.UniTuple(numba.float64,3)(numba.float64,numba.float64,numba.float64,numba.float64,
                                                  numba.float64,numba.float64))
def fast_crossProduct(a0,a1,a2,b0,b1,b2):
    cX = a1*b2 - a2*b1
    cY = a2*b0 - a0*b2
    cZ = a0*b1 - a1*b0
    return cX, cY, cZ


@numba.njit(numba.float64(numba.float64,numba.float64,numba.float64,numba.float64,numba.float64,numba.float64))
def fast_dotProduct(a0,a1,a2,b0,b1,b2):
    return a0*b0 + a1*b1 + a2*b2


@numba.njit(numba.float64(numba.float64, numba.float64, numba.float64))
def fast_3DNorm(x,y,z):
    return np.sqrt(x**2+y**2+z**2)


#Computes Helium velocity vector at the lithium position
@numba.njit(numba.types.UniTuple(numba.float64, 3)(numba.float64,numba.float64,numba.float64,numba.float64))
def He_velocity(vHe, z, y, x):

    denominator = np.sqrt(z**2 + y**2 + x**2)
    vX = vHe * x / denominator
    vY = vHe * y / denominator
    vZ = vHe * z / denominator
    return vZ, vY, vX


# Relative collision energy in the center of mass frame
def collision_energy_func(vLi_z, vLi_y, vLi_x, vHe_z, vHe_y, vHe_x, vRMS):

    v_rel_mag_m = ((vLi_z-vHe_z)**2 + (vLi_y-vHe_y)**2 + (vLi_x-vHe_x)**2) / 10000
    v_rel_mag_cm = np.sqrt((vLi_z-vHe_z)**2 + (vLi_y-vHe_y)**2 + (vLi_x-vHe_x)**2)
    KE_rel = 0.5 * u_kg * (v_rel_mag_m +vRMS**2) / BoltzmannConstant
    assert KE_rel >= 0
    return KE_rel, v_rel_mag_cm


# Relative collision energy in the center of mass frame
@numba.njit(numba.float64(numba.float64, numba.float64, numba.float64, numba.float64,
                          numba.float64, numba.float64))
def collision_energy_func_randomHe(vLi_z, vLi_y, vLi_x, vHe_z, vHe_y, vHe_x):

    v_rel_mag_m = ((vLi_z-vHe_z)**2 + (vLi_y-vHe_y)**2 + (vLi_x-vHe_x)**2) / 10000
    KE_rel = 0.5 * u_kg * v_rel_mag_m / BoltzmannConstant
    assert KE_rel >= 0
    return KE_rel


# returns density in units of cm^-3
@numba.njit(numba.float64(numba.float64,numba.float64,numba.float64,numba.float64,numba.float64))
def density_function(n0, d, z, y, x):

     # angle = 0.087 * 1.5
     # z = z1 * np.cos(angle) - y1 * np.sin(angle)
     # y = z1 * np.sin(angle) + y1 * np.cos(angle)
     if z < 0:
         return 1e-12
     elif z < d * 4:
         return n0 * 0.154 * d**2 / (z**2 + y**2 + x**2) * np.cos(np.arctan(sqrt(y**2 + x**2)/z))**2
     else:
         # Density Profile from Hans Pauli, "Atom, Molecule, and Cluster Beams 1", pg. 110
         return n0 * 0.154 * d**2 / (z**2 + y**2 + x**2) * np.cos(1.15 * np.arctan(sqrt(y**2 + x**2)/z))**2


# returns cross-section in units of cm^-2. Input is relative collision energy in Kelvin.
@numba.njit(numba.float64(numba.float64))
def crossSection_func(x):

    p1 = [1.42329946e-12, -9.93554761e-11, 4.96286024e-09, -1.68532898e-07,
          3.77590116e-06, -5.51085809e-05, 5.16019065e-04, -2.98049887e-03,
          9.64933004e-03, -1.33791549e-02]

    p2 = [2.66002555e-13, -1.05138668e-12,  2.21018836e-12, -2.67261490e-12,
          1.99971752e-12, -9.56968297e-13,  2.91470398e-13, -5.25267841e-14,
          3.82004683e-15,  4.64167528e-16, -1.36258469e-16,  1.24882328e-17,
          -4.25762694e-19]

    p3 = [2.46640407e-14, -1.75975422e-15, 1.62394104e-16, -9.14362654e-18,
          3.33007974e-19, -8.13583454e-21, 1.35761750e-22, -1.54964064e-24,
          1.18910512e-26, -5.86068014e-29, 1.67506716e-31, -2.10952788e-34]

    p4 = [1.45497425e-14, -7.27856154e-18, 1.26147930e-20, -1.44874368e-23,
          9.97459630e-27, -3.97425367e-30, 8.42846428e-34, -7.34855588e-38]

    if x < 0.1448:
        return p1[0]+p1[1]*x+p1[2]*x**2+p1[3]*x**3+p1[4]*x**4+p1[5]*x**5+p1[6]*x**6+p1[7]*x**7+p1[8]*x**8+p1[9]*x**9

    elif 0.1448 <= x < 5.89:
        return p2[0]+p2[1]*x+p2[2]*x**2+p2[3]*x**3+p2[4]*x**4+p2[5]*x**5+p2[6]*x**6+p2[7]*x**7+p2[8]*x**8+p2[9]*x**9+\
        p2[10]*x**10+p2[11]*x**11+p2[12]*x**12

    elif 5.89 <= x < 129.633:
        return p3[0]+p3[1]*x+p3[2]*x**2+p3[3]*x**3+p3[4]*x**4+p3[5]*x**5+p3[6]*x**6+p3[7]*x**7+p3[8]*x**8+p3[9]*x**9+\
        p3[10]*x**10+p3[11]*x**11

    elif 129.633 <= x:
        return p4[0]+p4[1]*x+p4[2]*x**2+p4[3]*x**3+p4[4]*x**4+p4[5]*x**5+p4[6]*x**6+p4[7]*x**7

    else:
        raise ValueError


# computes final velocity of Li with scattering angle cm_angle and phi
@numba.njit(numba.types.UniTuple(numba.float64, 3)(numba.float64, numba.float64, numba.float64, numba.float64,
                                                       numba.float64, numba.float64, numba.float64, numba.float64))
def collision(vLi_z, vLi_y, vLi_x, vHe_z, vHe_y, vHe_x, cm_angle,phi):

    if cm_angle < 1e-5:
        cm_angle = 1e-5

    # Center of mass velocity and initial Lithium velocity in center of mass frame
    vCM = [(vLi_z*mLi+vHe_z*mHe)/(mLi+mHe),(vLi_y*mLi + vHe_y*mHe)/(mLi+mHe),(vLi_x*mLi+vHe_x*mHe)/(mLi+mHe)]
    vLi_CM = [u/mLi*(vLi_z-vHe_z), u/mLi*(vLi_y-vHe_y), u/mLi*(vLi_x - vHe_x)]

    # Create random vector that is not parallel to vLi_cm
    randomVector = [vLi_CM[1]*1.5 + 0.1, vLi_CM[2]*2 + 0.2, -vLi_CM[0]*3 + 0.3]

    # Generates a vector perpendicular to vLi_cm and then normalize it
    c = list(fast_crossProduct(vLi_CM[0], vLi_CM[1], vLi_CM[2], randomVector[0], randomVector[1], randomVector[2]))
    c_mag = fast_3DNorm(c[0],c[1],c[2])

    #if Helium and Lithium velocity are the exact same then collision cannot occur.
    #this is very very unlikey to happen unless the Helium velocity is not being sampled from a distribution.
    if c_mag == 0:
        return vLi_z, vLi_y, vLi_x

    k = [c[0]/c_mag, c[1]/c_mag, c[2]/c_mag]

    # Generate new vector perpendicular to both vectors
    k_cross_vLi_CM = list(fast_crossProduct(k[0], k[1], k[2], vLi_CM[0], vLi_CM[1], vLi_CM[2]))

    # Rodriguez formula to generate vector rotated by cm_angle from vLi_cm.
    term1 = [i1 * np.cos(cm_angle) for i1 in vLi_CM]
    term2 = [i2 * np.sin(cm_angle) for i2 in k_cross_vLi_CM]
    vLi_CM_f = [term1[0]+term2[0], term1[1]+term2[1], term1[2]+term2[2]]

    # Repeating process again to create a new vector rotated by phi
    a_par = [n * np.cos(cm_angle) for n in vLi_CM]
    a_perp = [vLi_CM_f[0] - a_par[0], vLi_CM_f[1] - a_par[1], vLi_CM_f[2] - a_par[2]]
    a_perp_mag = fast_3DNorm(a_perp[0], a_perp[1], a_perp[2])

    w = list(fast_crossProduct(vLi_CM[0], vLi_CM[1], vLi_CM[2], a_perp[0], a_perp[1], a_perp[2]))
    w_mag = fast_3DNorm(w[0], w[1], w[2])

    term1 = [i3 * np.cos(phi) for i3 in a_perp]
    term2 = [i4 * np.sin(phi) * a_perp_mag/w_mag for i4 in w]
    a_perp_phi = [term1[0]+term2[0], term1[1]+term2[1], term1[2]+term2[2]]

    # Final vector in cm frame is rotated by theta and phi
    vLi_CM_f = [a_perp_phi[0] + a_par[0], a_perp_phi[1] + a_par[1], a_perp_phi[2] + a_par[2]]
    vLi_CM_mag = fast_3DNorm(vLi_CM[0], vLi_CM[1], vLi_CM[2])

    # Move back into the lab frame
    vLi_f = [vLi_CM_f[0]+vCM[0], vLi_CM_f[1]+vCM[1], vLi_CM_f[2]+vCM[2]]

    # Check if anything went wrong.
    if abs(math.pi - cm_angle) > 1e-8:

        temp1 = fast_dotProduct(vLi_CM_f[0], vLi_CM_f[1], vLi_CM_f[2], vLi_CM[0], vLi_CM[1], vLi_CM[2]) / vLi_CM_mag**2
        temp = np.arccos(temp1)

        if abs(temp - cm_angle) > 1e-7 or np.isnan(temp - cm_angle):

            print(abs(temp - cm_angle))
            print('temp',temp)
            print('temp1', temp1)
            print('cm angle', cm_angle)
            print('vLi_CM', vLi_CM)
            print('vCM', vCM)
            print('w', w)
            print("vLi_CM_f", vLi_CM_f)
            print('random vector', randomVector)
            print('k cross vLi_cm', k_cross_vLi_CM)
            print('phi', phi)
            print('a perp phi', a_perp_phi)

        assert abs(temp - cm_angle) < 1e-7

    else:

        temp = fast_dotProduct(vLi_CM_f[0], vLi_CM_f[1], vLi_CM_f[2], vLi_CM[0], vLi_CM[1], vLi_CM[2]) / vLi_CM_mag**2
        assert abs(1 - abs(temp)) < 1e-8

    return vLi_f[0], vLi_f[1], vLi_f[2]


# computes change in energy of Li before/after collision to determine energy dumped into the jet.
@numba.njit(numba.float64(numba.float64, numba.float64, numba.float64, numba.float64,numba.float64, numba.float64))
def deltaKE(Vz_i, Vy_i, Vx_i, Vz_f, Vy_f, Vx_f):

    v_i_squared = Vz_i**2+Vy_i**2+Vx_i**2
    v_f_squared = Vz_f**2+Vy_f**2+Vx_f**2

    delta_vSquared = (v_f_squared - v_i_squared)/10000  # convert from cm/s^2 to m/s^2

    return -0.5 * mLi_kg * delta_vSquared / BoltzmannConstant  # negative b/c energy going into the jet