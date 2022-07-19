from math import cos, sin, atan, tan, pi

import numpy as np
from shapely.geometry import LineString

from latticeElements.combiner_characterizer import make_halbach_combiner_force_function, compute_particle_trajectory
from latticeElements.elements import Drift, HalbachLensSim, CombinerHalbachLensSim, BenderIdeal, \
    HalbachBenderSimSegmented, Element
from typeHints import RealNum


def convert_center_to_orbit_coords(el: Element, s: RealNum, xc: RealNum, yc: RealNum) -> tuple[float, float, float]:
    """Convert center coordinates [s,xc,yc] to cartesian coordinates[x,y,z]"""
    arc_length = el.ang * el.ro
    if 0.0 <= s < el.L_cap:
        x, y, z = el.ro + xc, s - el.L_cap, yc
    elif el.L_cap <= s < el.L_cap + arc_length:
        theta = (s - el.L_cap) / el.ro
        r = el.ro + xc
        x, y, z = cos(theta) * r, sin(theta) * r, yc
    elif el.L_cap + arc_length <= s <= arc_length + 2 * el.L_cap:
        theta = el.ang
        r = el.ro + xc
        x0, y0 = cos(theta) * r, sin(theta) * r
        delta_s = s - (el.ang * el.ro + el.L_cap)
        theta_perp = pi + atan(-1 / tan(theta))
        x, y, z = x0 + cos(theta_perp) * delta_s, y0 + sin(theta_perp) * delta_s, yc
    else:
        raise ValueError
    return x, y, z


def combiner_halbach_xy(el: Element) -> np.ndarray:
    atom_state = 'HIGH_FIELD_SEEKER' if el.field_fact == -1 else 'LOW_FIELD_SEEKER'
    force_func = make_halbach_combiner_force_function(el)
    q_arr, _ = compute_particle_trajectory(force_func, el.PTL.speed_nominal, 0.0, 2 * el.space + el.Lm,
                                           particle_y_offset_start=el.output_offset, atom_state=atom_state)
    xy = q_arr[:, :2]
    for i, coord in enumerate(xy):
        xy[i] = el.R_Out @ coord
    xy += el.r2[:2]
    return xy


def straight_xy(el: Element) -> np.ndarray:
    xy = np.array([[0.0, 0.0], [el.L, 0.0]])
    for i, coord in enumerate(xy):
        xy[i] = el.R_Out @ coord
    xy += el.r1[:2]
    return xy


def ideal_bend_xy(el: Element) -> np.ndarray:
    angles = np.linspace(0.0, el.ang)
    xy = np.column_stack((el.ro * np.cos(angles), el.ro * np.sin(angles)))
    for i, coord in enumerate(xy):
        xy[i] = el.R_Out @ coord
    xy += el.r0[:2]
    return xy


def segmented_halbach_bender(el: Element) -> np.ndarray:
    s_vals = np.linspace(1e-6, el.Lo - 1e-6)
    xy = np.array([convert_center_to_orbit_coords(el, s, 0.0, 0.0)[:2] for s in s_vals])
    for i, coord in enumerate(xy):
        xy[i] = el.R_Out @ coord
    xy += el.r0[:2]
    return xy


def make_particle_trajectory(el: Element) -> np.ndarray:
    if type(el) in (Drift, HalbachLensSim):
        xy = straight_xy(el)
    elif type(el) is CombinerHalbachLensSim:
        xy = combiner_halbach_xy(el)
    elif type(el) is BenderIdeal:
        xy = ideal_bend_xy(el)
    elif type(el) is HalbachBenderSimSegmented:
        xy = segmented_halbach_bender(el)
    else:
        raise NotImplemented
    return xy


def make_trajectory_shape(el: Element) -> LineString:
    xy = make_particle_trajectory(el)
    return LineString(xy)
