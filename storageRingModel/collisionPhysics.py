import numpy as np
import numba
from numba.typed import List
from typing import Union
from constants import MASS_LITHIUM_7,BOLTZMANN_CONSTANT
import elementPT

float_kelvin=float
float_meter=float
float_mps=float
realNum=Union[float,int]
vec3D=tuple[float,float,float]
frequency=float
angle=Union[float,int]


@numba.njit()
def full_Arctan(y: realNum,x: realNum)->angle:
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(y,x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi

def collision_Rate(T: float_kelvin, rp_Meters: float_meter)-> frequency:
    """Calculate the collision rate of a beam of flux with a moving frame temperature of T confined to a fraction of
    the area rp_Meters. NOTE: This is all done in centimeters instead of meters!"""
    assert rp_Meters<.1 and T<.1 #reasonable values
    rp=rp_Meters*1e2 #convert to cm
    vRel=1e2*np.sqrt(16 * BOLTZMANN_CONSTANT*T/(3.14 *MASS_LITHIUM_7)) #cm/s
    sigma=5e-13 #cm^2
    speed=210*1e2 #cm^2
    flux=2e12*500 #1/s
    area=np.pi*(.7*rp)**2 #cm
    n=flux/(area*speed) #1/cm^3
    meanFreePath=1/(np.sqrt(2)*n*sigma)
    return vRel/meanFreePath

@numba.njit()
def momentum_sample_3D(T: float_kelvin)-> vec3D:
    """Sample momentum in 3D based on temperature T"""
    sigma=np.sqrt(BOLTZMANN_CONSTANT * T / MASS_LITHIUM_7)
    pi,pj,pk=np.random.normal(loc=0.0,scale=sigma,size=3)
    return pi,pj,pk

@numba.njit()
def collision_Partner_Momentum_Lens(s0: float_mps,T: float_kelvin)-> vec3D:
    """Calculate a collision partner's momentum for colliding with particle traveling in the lens. Collision partner
    is sampled from a gas with temperature T traveling along the lens of the lens/waveguide"""
    deltaP=momentum_sample_3D(T)
    delta_px,py,pz=deltaP
    px = s0 + delta_px
    pCollision = (px, py, pz)
    return pCollision

@numba.njit()
def collision_Partner_Momentum_Bender(qEl: vec3D,nominalSpeed:float_mps,T: float_kelvin)-> vec3D:
    """Calculate a collision partner's momentum for colliding with particle traveling in the bender. The collision
    partner is sampled assuming a random gas with nominal speeds in the bender given by geometry and angular momentum.
    """
    delta_pso,pxo,pyo=momentum_sample_3D(T)
    pso = nominalSpeed + delta_pso
    theta=full_Arctan(qEl[1],qEl[0])
    px=pxo*np.cos(theta)--pso*np.sin(theta)
    py=pxo*np.sin(theta)+-pso*np.cos(theta)
    pz=pyo
    pCollision = (px, py, pz)
    return pCollision

@numba.njit()
def apply_Collision(p: vec3D,q: vec3D,collisionParams: tuple):
    if collisionParams[0]=='STRAIGHT':
        s0,T=collisionParams[2],collisionParams[3]
        pColPartner=collision_Partner_Momentum_Lens(s0,T)
        pNew=collision(*p,*pColPartner)
        p=pNew
    elif collisionParams[0]=='SEG_BEND':
        s0,ang,T=collisionParams[2],collisionParams[3],collisionParams[4]
        theta = full_Arctan(q[1], q[0])
        if 0.0<=theta<=ang:
            pColPartner=collision_Partner_Momentum_Bender(q,s0,T)
            p = collision(*p, *pColPartner)
    elif collisionParams[0]==-1:
        pass
    else: raise NotImplementedError
    return p


def get_Collision_Params(element:elementPT.Element,atomSpeed: realNum):
    """Will be changed soon I anticipate"""
    from temp3 import MUT
    rpDrift_Fake=.03
    if type(element) in (elementPT.HalbachLensSim,elementPT.Drift):
        T, rp = MUT.T, element.rp
        rp=rpDrift_Fake if rp==np.inf else rp
        collisionRate =collision_Rate(T, rp)
        return 'STRAIGHT',collisionRate,  atomSpeed,MUT.T,np.nan
    elif type(element) is elementPT.HalbachBenderSimSegmented:
        T, rp = MUT.T, element.rp
        collisionRate = collision_Rate(T, rp)
        return 'SEG_BEND',collisionRate, atomSpeed,element.ang,MUT.T
    else: return 'NONE',np.nan,np.nan,np.nan,np.nan


@numba.njit()
def percent_difference(x,y):
    return abs((x-y) * 2 / (x+y))

@numba.njit()
def vel_comp_after_collision(v_rel):
    cos_theta = 2 * np.random.random() - 1
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = 2 * np.pi * np.random.random()

    vx_final = v_rel * sin_theta * np.cos(phi)
    vy_final = v_rel * sin_theta * np.sin(phi)
    vz_final = v_rel * cos_theta

    return vx_final, vy_final, vz_final


@numba.njit()
def collision(p1_vx, p1_vy, p1_vz,p2_vx, p2_vy, p2_vz):
    """ Elastic collision of two particles with random scattering angle phi and theta. Inputs are the two particles
        x,y,z components of their velocity. Output is the particles final velocity components. Output coordinate
        system matches whatever is used as the input so long as it's cartesian."""

    vx_cm = 0.5 * (p1_vx + p2_vx)
    vy_cm = 0.5 * (p1_vy + p2_vy)
    vz_cm = 0.5 * (p1_vz + p2_vz)

    v_rel = np.sqrt((p1_vx - p2_vx) ** 2 + (p1_vy - p2_vy) ** 2 + (p1_vz - p2_vz) ** 2)

    vx_final, vy_final, vz_final = vel_comp_after_collision(v_rel)

    p1_vx_final = vx_cm + 0.5 * vx_final
    p1_vy_final = vy_cm + 0.5 * vy_final
    p1_vz_final = vz_cm + 0.5 * vz_final

    return p1_vx_final, p1_vy_final, p1_vz_final


