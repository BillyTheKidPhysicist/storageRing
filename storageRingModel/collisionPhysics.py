"""
Functions to model the effects of collisions between lithium atoms. Current version of the model is rather simple and
described in electronic lab book. Briefly, the model assumptions are:

- lithium is uniformly distributed in magnets up to .7rp
- relative velocity comes from a combination of a minimum "geometric" value and from a thermal distirbution
- effect is only modeled in lens and arc section of bender to keep things simple

"""
# pylint: disable=too-many-locals, too-many-arguments
from typing import Union

import numba
import numpy as np

import elementPT
from constants import MASS_LITHIUM_7, BOLTZMANN_CONSTANT, SIMULATION_MAGNETON

realNum = Union[float, int]
vec3D = tuple[float, float, float]
frequency = float
angle = Union[float, int]
Element = Union[elementPT.BenderIdeal,
                elementPT.LensIdeal, elementPT.CombinerIdeal]


@numba.njit()
def clamp_value(value: float, value_min: float, value_max: float) -> float:
    """ restrict a value between a minimum and maximum value"""
    return min([max([value_min, value]), value_max])


@numba.njit()
def max_Momentum_1D_In_Trap(r, rp, F_Centrifugal) -> float:
    """
    Compute the maximum possible transverse momentum based on transverse location in hexapole magnet. This comes
    from the finite depth of the trap and the corresponding momentum to escape

    :param r: Radial position in magnet of atom
    :param rp: Bore radius of magnet
    :param F_Centrifugal: an additional constant radial force. Used to model centrifugal force
    :return:
    """
    assert abs(r) <= rp and rp > 0.0 and F_Centrifugal >= 0.0
    Bp = .75
    delta_E_Mag = Bp * SIMULATION_MAGNETON * (1 - (r / rp) ** 2)
    delta_E_Const = -F_Centrifugal * rp - -F_Centrifugal * r
    E_Escape = delta_E_Mag + delta_E_Const
    if E_Escape < 0.0:  # particle would escape according to simple model. Instead, low energy level
        E_Low = Bp * SIMULATION_MAGNETON * .5 ** 2
        vMax = np.sqrt(2 * E_Low)
    else:
        vMax = np.sqrt(2 * E_Escape)
    return vMax


@numba.njit()
def trim_Longitudinal_Momentum_To_Maximum(pLong: float, nominalSpeed: float) -> float:
    """Longitudinal momentum can only exist within a range of stability. Momentum outside that range is lost,
    and so particles with that momentum are not likely to be present in the ring in much numbers"""
    deltaPMax = 15.0  # from observations of phase space survival
    pmin, pmax = nominalSpeed - deltaPMax, nominalSpeed + deltaPMax
    return clamp_value(pLong, pmin, pmax)


@numba.njit()
def trim_Transverse_Momentum_To_Maximum(p_i: float, q_i: float, rp: float, Fcentrifugal=0.0) -> float:
    """Maximum transverse momentum is limited by the depth of the trap and centrifugal force."""
    assert abs(q_i) <= rp and rp > 0.0 and Fcentrifugal >= 0.0
    p_iMax = max_Momentum_1D_In_Trap(q_i, rp, Fcentrifugal)
    p_i = clamp_value(p_i, -p_iMax, p_iMax)
    return p_i


@numba.njit()
def full_Arctan(y: realNum, x: realNum) -> angle:
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where tan(phi)=y/x"""
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


def collision_Rate(T: float, rp_Meters: float) -> frequency:
    """Calculate the collision rate of a beam of flux with a moving frame temperature of T confined to a fraction of
    the area rp_Meters. NOTE: This is all done in centimeters instead of meters!"""
    assert 0 < rp_Meters < .1 and 0 <= T < .1  # reasonable values
    rp = rp_Meters * 1e2  # convert to cm
    vRelThermal = 1e2 * np.sqrt(16 * BOLTZMANN_CONSTANT * T / (3.14 * MASS_LITHIUM_7))  # cm/s
    # cm/s .even with zero temperature, there is still relative motion between atoms
    vRelRingDynamics = 50.0
    vRel = np.sqrt(vRelRingDynamics ** 2 + vRelThermal ** 2)  # cm/s
    sigma = 5e-13  # cm^2
    speed = 210 * 1e2  # cm^2
    flux = 2e12 * 500  # 1/s
    area = np.pi * (.7 * rp) ** 2  # cm
    n = flux / (area * speed)  # 1/cm^3
    meanFreePath = 1 / (np.sqrt(2) * n * sigma)  # cm
    return vRel / meanFreePath  # 1/s


@numba.njit()
def momentum_sample_3D(T: float) -> vec3D:
    """Sample momentum in 3D based on temperature T"""
    sigma = np.sqrt(BOLTZMANN_CONSTANT * T / MASS_LITHIUM_7)
    pi, pj, pk = np.random.normal(loc=0.0, scale=sigma, size=3)
    return pi, pj, pk


@numba.njit()
def collision_Partner_Momentum_Lens(qEl: vec3D, s0: float, T: float, rp: float) -> vec3D:
    """Calculate a collision partner's momentum for colliding with particle traveling in the lens. Collision partner
    is sampled from a gas with temperature T traveling along the lens of the lens/waveguide"""
    _, y, z = qEl
    deltaP = momentum_sample_3D(T)
    delta_px, py, pz = deltaP
    px = s0 + delta_px
    px = trim_Longitudinal_Momentum_To_Maximum(px, s0)
    py = trim_Transverse_Momentum_To_Maximum(py, y, rp)
    pz = trim_Transverse_Momentum_To_Maximum(pz, z, rp)
    pCollision = (px, py, pz)
    return pCollision


@numba.njit()
def collision_Partner_Momentum_Bender(qEl: vec3D, nominalSpeed: float, T: float, rp: float, rBend) -> vec3D:
    """Calculate a collision partner's momentum for colliding with lithium traveling in the bender. The collision
    partner is sampled assuming a random gas with nominal speeds in the bender given by geometry and angular momentum.
    """
    delta_pso, pxo, pyo = momentum_sample_3D(T)
    pso = nominalSpeed + delta_pso
    xo = np.sqrt(qEl[0] ** 2 + qEl[1] ** 2) - rBend
    yo = qEl[2]
    Fcentrigfugal = nominalSpeed ** 2 / rBend  # approximately centripetal force
    pxo = trim_Transverse_Momentum_To_Maximum(pxo, xo, rp, Fcentrifugal=Fcentrigfugal)
    pyo = trim_Transverse_Momentum_To_Maximum(pyo, yo, rp, Fcentrifugal=Fcentrigfugal)
    pso = trim_Longitudinal_Momentum_To_Maximum(pso, nominalSpeed)
    theta = full_Arctan(qEl[1], qEl[0])
    px = pxo * np.cos(theta) - -pso * np.sin(theta)
    py = pxo * np.sin(theta) + -pso * np.cos(theta)
    pz = pyo
    pCollision = (px, py, pz)
    return pCollision


@numba.njit()
def post_Collision_Momentum(p: vec3D, q: vec3D, collisionParams: tuple) -> vec3D:
    """Get the momentum after a collision. The collision partner momentum is generated, and then Jeremy's collision
    algorithm is applied to find the new momentum. There is some wonkiness here from using numba"""
    if collisionParams[0] == 'STRAIGHT':
        s0, T, rp = collisionParams[2], collisionParams[3], collisionParams[4]
        pColPartner = collision_Partner_Momentum_Lens(q, s0, T, rp)
        pNew = collision(*p, *pColPartner)
        p = pNew
    elif collisionParams[0] == 'SEG_BEND':
        s0, ang, T, rp, rb = collisionParams[2], collisionParams[3], collisionParams[4], \
                             collisionParams[5], collisionParams[6]
        theta = full_Arctan(q[1], q[0])
        if 0.0 <= theta <= ang:
            pColPartner = collision_Partner_Momentum_Bender(q, s0, T, rp, rb)
            p = collision(*p, *pColPartner)
    elif collisionParams[0] == -1:
        pass
    else:
        raise NotImplementedError
    return p


def get_Collision_Params(element: Element, atomSpeed: realNum):
    """Will be changed soon I anticipate. Dealing with numba wonkiness"""
    T = .01
    if type(element) in (elementPT.HalbachLensSim, elementPT.Drift):
        rp = element.rp
        rpDrift_Fake = .03
        rp = rpDrift_Fake if rp == np.inf else rp
        collisionRate = collision_Rate(T, rp)
        return 'STRAIGHT', collisionRate, atomSpeed, T, rp, np.nan, np.nan
    elif type(element) is elementPT.HalbachBenderSimSegmented:
        rp, rb = element.rp, element.rb
        collisionRate = collision_Rate(T, rp)
        return 'SEG_BEND', collisionRate, atomSpeed, element.ang, T, rb, rp
    else:
        return 'NONE', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


@numba.njit()
def vel_comp_after_collision(v_rel):
    """Provided by Jeremy"""
    cos_theta = 2 * np.random.random() - 1
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = 2 * np.pi * np.random.random()

    vx_final = v_rel * sin_theta * np.cos(phi)
    vy_final = v_rel * sin_theta * np.sin(phi)
    vz_final = v_rel * cos_theta

    return vx_final, vy_final, vz_final


@numba.njit()
def collision(p1_vx, p1_vy, p1_vz, p2_vx, p2_vy, p2_vz):
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
