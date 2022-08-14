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

from constants import MASS_LITHIUM_7, BOLTZMANN_CONSTANT, SIMULATION_MAGNETON
from helperTools import full_arctan2
# import latticeElements.elementPT
from latticeElements.elements import HalbachLensSim, Drift, \
    HalbachBender, Element
from typeHints import RealNum

full_arctan2 = numba.njit(full_arctan2)

vec3D = tuple[float, float, float]
frequency = float
angle = Union[float, int]


@numba.njit()
def clamp_value(value: float, value_min: float, value_max: float) -> float:
    """ restrict a value between a minimum and maximum value"""
    return min([max([value_min, value]), value_max])


@numba.njit()
def max_momentum_1D_in_trap(r, rp, F_Centrifugal) -> float:
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
    delta_E_mag = Bp * SIMULATION_MAGNETON * (1 - (r / rp) ** 2)
    delta_E_const = -F_Centrifugal * rp - -F_Centrifugal * r
    E_escape = delta_E_mag + delta_E_const
    if E_escape < 0.0:  # particle would escape according to simple model. Instead, low energy level
        E_low = Bp * SIMULATION_MAGNETON * .5 ** 2
        v_max = np.sqrt(2 * E_low)
    else:
        v_max = np.sqrt(2 * E_escape)
    return v_max


@numba.njit()
def trim_longitudinal_momentum_to_max(p_longitudinal: float, nominal_speed: float) -> float:
    """Longitudinal momentum can only exist within a range of stability. Momentum outside that range is lost,
    and so particles with that momentum are not likely to be present in the ring in much numbers"""
    delta_p_max = 15.0  # from observations of phase space survival
    p_min, p_max = nominal_speed - delta_p_max, nominal_speed + delta_p_max
    return clamp_value(p_longitudinal, p_min, p_max)


@numba.njit()
def trim_trans_momentum_to_max(p_i: float, q_i: float, rp: float, F_centrifugal=0.0) -> float:
    """Maximum transverse momentum is limited by the depth of the trap and centrifugal force."""
    assert abs(q_i) <= rp and rp > 0.0 and F_centrifugal >= 0.0
    p_i_max = max_momentum_1D_in_trap(q_i, rp, F_centrifugal)
    p_i = clamp_value(p_i, -p_i_max, p_i_max)
    return p_i


def collision_rate(T: float, rp_meters: float) -> frequency:
    """Calculate the collision rate of a beam of flux with a moving frame temperature of T confined to a fraction of
    the area rp_Meters. NOTE: This is all done in centimeters instead of meters!"""
    assert 0 < rp_meters < .1 and 0 <= T < .1  # reasonable values
    rp = rp_meters * 1e2  # convert to cm
    v_rel_thermal = 1e2 * np.sqrt(16 * BOLTZMANN_CONSTANT * T / (3.14 * MASS_LITHIUM_7))  # cm/s
    # cm/s .even with zero temperature, there is still relative motion between atoms
    v_rel_ring_dynamics = 50.0
    v_rel = np.sqrt(v_rel_ring_dynamics ** 2 + v_rel_thermal ** 2)  # cm/s
    sigma = 5e-13  # cm^2
    speed = 210 * 1e2  # cm^2
    flux = 2e12 * 500  # 1/s
    area = np.pi * (.7 * rp) ** 2  # cm
    n = flux / (area * speed)  # 1/cm^3
    mean_free_path = 1 / (np.sqrt(2) * n * sigma)  # cm
    return v_rel / mean_free_path  # 1/s


@numba.njit()
def momentum_sample_3D(T: float) -> vec3D:
    """Sample momentum in 3D based on temperature T"""
    sigma = np.sqrt(BOLTZMANN_CONSTANT * T / MASS_LITHIUM_7)
    pi, pj, pk = np.random.normal(loc=0.0, scale=sigma, size=3)
    return pi, pj, pk


@numba.njit()
def collision_partner_momentum_lens(q_el: vec3D, s0: float, T: float, rp: float) -> vec3D:
    """Calculate a collision partner's momentum for colliding with particle traveling in the lens. Collision partner
    is sampled from a gas with temperature T traveling along the lens of the lens/waveguide"""
    _, y, z = q_el
    deltaP = momentum_sample_3D(T)
    delta_px, py, pz = deltaP
    px = s0 + delta_px
    px = trim_longitudinal_momentum_to_max(px, s0)
    py = trim_trans_momentum_to_max(py, y, rp)
    pz = trim_trans_momentum_to_max(pz, z, rp)
    pCollision = (px, py, pz)
    return pCollision


@numba.njit()
def collision_partner_momentum_bender(q_el: vec3D, nominal_speed: float, T: float, rp: float, rBend) -> vec3D:
    """Calculate a collision partner's momentum for colliding with lithium traveling in the bender. The collision
    partner is sampled assuming a random gas with nominal speeds in the bender given by geometry and angular momentum.
    """
    delta_pso, pxo, pyo = momentum_sample_3D(T)
    pso = nominal_speed + delta_pso
    xo = np.sqrt(q_el[0] ** 2 + q_el[1] ** 2) - rBend
    yo = q_el[2]
    F_centrifugal = nominal_speed ** 2 / rBend  # approximately centripetal force
    pxo = trim_trans_momentum_to_max(pxo, xo, rp, F_centrifugal=F_centrifugal)
    pyo = trim_trans_momentum_to_max(pyo, yo, rp, F_centrifugal=F_centrifugal)
    pso = trim_longitudinal_momentum_to_max(pso, nominal_speed)
    theta = full_arctan2(q_el[1], q_el[0])
    px = pxo * np.cos(theta) - -pso * np.sin(theta)
    py = pxo * np.sin(theta) + -pso * np.cos(theta)
    pz = pyo
    p_collision = (px, py, pz)
    return p_collision


@numba.njit()
def post_collision_momentum(p: vec3D, q: vec3D, collision_params: tuple) -> vec3D:
    """Get the momentum after a collision. The collision partner momentum is generated, and then Jeremy's collision
    algorithm is applied to find the new momentum. There is some wonkiness here from using numba"""
    if collision_params[0] == 'STRAIGHT':
        s0, T, rp = collision_params[2], collision_params[3], collision_params[4]
        p_col_partner = collision_partner_momentum_lens(q, s0, T, rp)
        pNew = collision(*p, *p_col_partner)
        p = pNew
    elif collision_params[0] == 'SEG_BEND':
        s0, ang, T, rp, rb = collision_params[2], collision_params[3], collision_params[4], \
                             collision_params[5], collision_params[6]
        theta = full_arctan2(q[1], q[0])
        if 0.0 <= theta <= ang:
            p_col_partner = collision_partner_momentum_bender(q, s0, T, rp, rb)
            p = collision(*p, *p_col_partner)
    elif collision_params[0] == -1:
        pass
    else:
        raise NotImplementedError
    return p


def make_collision_params(element: Element, atomSpeed: RealNum):
    """Will be changed soon I anticipate. Dealing with numba wonkiness"""
    T = .01
    if type(element) in (HalbachLensSim, Drift):
        rp = element.rp
        rpDrift_Fake = .03
        rp = rpDrift_Fake if rp == np.inf else rp
        col_rate = collision_rate(T, rp)
        return 'STRAIGHT', col_rate, atomSpeed, T, rp, np.nan, np.nan
    elif type(element) is HalbachBender:
        rp, rb = element.rp, element.rb
        col_rate = collision_rate(T, rp)
        return 'SEG_BEND', col_rate, atomSpeed, element.ang, T, rb, rp
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
