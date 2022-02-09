import numba
import numpy as np
import math
import pickle
from random import random as randomNumber

#---Global Variables---
BoltzmannConstant = 1.38064852e-23

mLi_kg = 1.1525801e-26
mHe_kg = 6.6464731e-27
u_kg = mLi_kg*mHe_kg/(mLi_kg+mHe_kg)

mLi = 1.
mHe = mLi * 0.576660407
u = mLi * mHe / (mLi + mHe)

#object for generating scattering angle given relative energy of Li-He collision. Using data from Eite
file01 = open('theta_sampler_forBilly.obj', 'rb')
theta_sampler = pickle.load(file01)
file01.close()

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


# Relative collision energy in the center of mass frame
@numba.njit(numba.float64(numba.float64, numba.float64, numba.float64, numba.float64,
                          numba.float64, numba.float64))
def collision_energy_func(vLi_z, vLi_y, vLi_x, vHe_z, vHe_y, vHe_x):

    v_rel_mag_m = ((vLi_z-vHe_z)**2 + (vLi_y-vHe_y)**2 + (vLi_x-vHe_x)**2) / 10000
    KE_rel = 0.5 * u_kg * v_rel_mag_m / BoltzmannConstant
    assert KE_rel >= 0
    return KE_rel


# computes final velocity of Li in the lab frame with scattering angle cm_angle and phi
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
    #this is very very very unlikey to happen unless the Helium velocity is not being sampled from a distribution.
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

    #z, y, and x components in that order of the final Lithium velocity in the lab frame
    return vLi_f[0], vLi_f[1], vLi_f[2]


#input is velocity components Vz,Vy,Vx in that order.
#Example is vLi = [5,1,2] and vHe = [10,-5,30].
def vLi_final(vLi_0, vHe_0):
    vLi_cm=[vi*1e2 for vi in vLi_0]
    vHe_cm=[vi*1e2 for vi in vHe_0]
    phi = randomNumber()*2*math.pi
    collision_energy = collision_energy_func(*vLi_cm, *vHe_cm)
    cm_angle = float(theta_sampler(collision_energy, randomNumber())) * math.pi / 180
    vx,vy,vz= collision(*vLi_cm, *vHe_cm, cm_angle, phi) #output is a tuple
    return [vx/1e2, vy/1e2,vz/1e2]




