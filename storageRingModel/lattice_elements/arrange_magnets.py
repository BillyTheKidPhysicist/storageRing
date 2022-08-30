"""
These methods deal with alinging and arranging magnets in the lattice to properly account for magnet-magnet magnetic
field interactions. It's an abomination because of how I wasn't consistent early on with element layouts.
"""

from math import cos, sin

from numpy.linalg import norm
from scipy.spatial.transform import Rotation as Rot

from field_generators import Collection
from helper_tools import *
from lattice_elements.base_element import BaseElement
from lattice_elements.elements import HalbachLensSim, Element, CombinerLensSim, Drift, BenderSim

INTERP_ELS = (HalbachLensSim, CombinerLensSim, Drift)  # ,BenderSim)  # elements that can include magnetic fields
# produced by neighboring magnets into their own interpolation
FIELD_GENERATOR_ELS = (HalbachLensSim, CombinerLensSim)#, BenderSim)  # elements that can be used to
# generate  magnetic fields
# that act on another elements

DISTANCE_BORE_RADIUS_FACT = 3.0


# I should have thought more clearly about what i was doing. Then I wouldn't have had any of these problems. I wouldn't
# have wasted so much time

def is_valid_interperable_el(el) -> bool:
    """Is 'el' and element which can use the magnetic fields of external elements to interpolate forces over?"""
    assert isinstance(el, BaseElement)
    return type(el) in INTERP_ELS


def is_valid_field_generator_el(el) -> bool:
    """Is 'el' and element that can apply external field values to other elements?"""
    return type(el) in FIELD_GENERATOR_ELS


def minimum_distance_between_elements(el1, el2) -> float:
    """Minimum distance between two elements. Reference is r1 and r2 of the element, not the actual magnetic material"""
    dr = np.array([norm(pos1 - pos2) for pos1, pos2 in itertools.product([el1.r1, el1.r2], [el2.r1, el2.r2])])
    return min(dr)


def are_els_close_enough(el1, el2) -> bool:
    """Are the two elements close enough such that the magnetic field interactions should be considered?"""
    distance = minimum_distance_between_elements(el1, el2)
    for el in [el1, el2]:
        distance_fringe = DISTANCE_BORE_RADIUS_FACT * el.rp
        if type(el) in FIELD_GENERATOR_ELS and distance_fringe > distance:
            return True
    return False


def do_geom_rule_exist(el1, el2):
    if (type(el1) is CombinerLensSim and type(el2) is BenderSim) or (
            type(el2) is CombinerLensSim and type(el1) is BenderSim):
        return False
    else:
        return True


def is_valid_neighbors(el: Element, el_neighbor: Element) -> bool:
    """is 'el_neighbor' a valid neighbooring element to 'el' for purposes of including magnetic field interactions"""
    if not is_valid_interperable_el(el) or not is_valid_field_generator_el(el_neighbor) \
            or not do_geom_rule_exist(el, el_neighbor):
        return False
    elif type(el) is BenderSim and type(el_neighbor) is BenderSim:
        return False
    elif type(el) is Drift and el.output_tilt_angle != 0 and el.input_tilt_angle != 0:
        return False
    elif el is el_neighbor:  # element is itself
        return False
    elif are_els_close_enough(el, el_neighbor):
        return True
    else:
        return False


def angle(norm) -> float:
    return full_arctan2(norm[1], norm[0])


def move_combiner_to_target_frame(el_combiner: CombinerLensSim, el_target, magnet_options) -> Collection:
    r_in_el = el_combiner.transform_lab_coords_into_element_frame(el_combiner.r1)
    magnets = el_combiner.magnet.magpylib_magnets_model(*magnet_options)
    magnets.move_meters(-r_in_el)
    theta_el = np.pi + (angle(el_combiner.ne) - angle(el_target.ne))
    R = Rot.from_rotvec([0, 0, theta_el])
    magnets.rotate(R, anchor=0)
    r1, r2 = el_combiner.r1, el_target.r1
    deltar_el = el_target.transform_lab_coords_into_element_frame(
        r1) - el_target.transform_lab_coords_into_element_frame(r2)
    magnets.move_meters(deltar_el)
    return magnets


def move_lens_to_target_el_frame(el_to_move: Element, el_target: Element, magnet_options):
    magnets = el_to_move.magnet.magpylib_magnets_model(*magnet_options)
    if type(el_target) is CombinerLensSim:  # the combiner is annoying to deal with because I defined the input
        # not at the origin, but rather along the x axis with some y offset
        # move_meters input to 0,0 in el frame
        r_input = el_target.transform_lab_coords_into_element_frame(el_target.r2)
        magnets.move_meters(r_input)
        # now rotate to correct alignment so that
        psi = angle(-el_to_move.nb) - angle(el_target.ne)
        R = Rot.from_rotvec([0, 0, np.pi + psi])
        magnets.rotate(R, anchor=(0, 0, 0))
        # now move_meters to target el output
        r_target = el_target.r2
        r_move = el_to_move.r1
        deltar_el = el_target.transform_lab_coords_into_element_frame(
            r_move) - el_target.transform_lab_coords_into_element_frame(r_target)
        magnets.move_meters(deltar_el)
        return magnets
    elif type(el_target) is BenderSim:
        shift_to_origin = (-el_to_move.L, 0, 0)
        magnets.move_meters(shift_to_origin)
        psi = -(angle(-el_target.nb) - angle(el_to_move.ne) + np.pi / 2.0 - el_target.ang)
        R = Rot.from_rotvec([0, 0, psi])
        magnets.rotate(R, anchor=(0, 0, 0))
        ang_perp = el_target.ang + np.pi / 2
        norm_out = np.array([cos(ang_perp), sin(ang_perp), 0.0])
        x = cos(el_target.ang) * el_target.ro + norm_out[0] * el_target.L_cap
        y = sin(el_target.ang) * el_target.ro + norm_out[1] * el_target.L_cap

        shift_to_bender_input = (x, y, 0)
        magnets.move_meters(shift_to_bender_input)
        r_target = el_target.r1
        r_move = el_to_move.r2
        deltar_el = el_target.transform_lab_coords_into_element_frame(
            r_move) - el_target.transform_lab_coords_into_element_frame(r_target)
        magnets.move_meters(deltar_el)
        return magnets
    elif type(el_target) in (Drift, HalbachLensSim):
        psi = angle(-el_to_move.nb) - angle(el_target.ne)
        R = Rot.from_rotvec([0, 0, psi])
        magnets.rotate(R, anchor=(0, 0, 0))
        r_target = el_target.r1
        r_move = el_to_move.r1
        deltar_el = el_target.transform_lab_coords_into_element_frame(
            r_move) - el_target.transform_lab_coords_into_element_frame(r_target)
        magnets.move_meters(deltar_el)
        return magnets
    else:
        raise NotImplementedError


def move_bender_to_target_el_frame(el_to_move, el_target, magnet_options):
    assert type(el_to_move) is BenderSim
    if type(el_target) not in (Drift, HalbachLensSim):
        raise NotImplementedError

    magnets = el_to_move.magnet.magpylib_magnets_model()
    ang_perp = el_to_move.ang + np.pi / 2
    norm_out = np.array([cos(ang_perp), sin(ang_perp), 0.0])
    x = cos(el_to_move.ang) * el_to_move.ro + norm_out[0] * el_to_move.L_cap
    y = sin(el_to_move.ang) * el_to_move.ro + norm_out[1] * el_to_move.L_cap
    delta_r = (-x, -y, 0)
    magnets.move_meters(delta_r)

    psi = angle(-el_to_move.nb) - angle(el_target.ne) + np.pi / 2.0 - el_to_move.ang
    R = Rot.from_rotvec([0, 0, psi])
    magnets.rotate(R, anchor=(0, 0, 0))
    r_target = el_target.r1
    r_move = el_to_move.r1
    deltar_el = el_target.transform_lab_coords_into_element_frame(
        r_move) - el_target.transform_lab_coords_into_element_frame(r_target)
    magnets.move_meters(deltar_el)
    return magnets


def field_generator_in_different_frame1(el_to_move: Element, el_target: Element, magnet_options: tuple):
    if type(el_to_move) is CombinerLensSim:
        return move_combiner_to_target_frame(el_to_move, el_target, magnet_options)
    elif type(el_to_move) is HalbachLensSim:
        return move_lens_to_target_el_frame(el_to_move, el_target, magnet_options)
    elif type(el_to_move) is BenderSim:
        if magnet_options[0]:
            raise NotImplementedError("magnet errors are not implemented here yet")
        return move_bender_to_target_el_frame(el_to_move, el_target, magnet_options)
    else:
        raise NotImplementedError


def collect_valid_neighboring_magpylib_magnets(el: Element, lattice) -> Optional[list[Collection]]:
    if is_valid_interperable_el(el):
        magnet_options = (lattice.use_mag_errors, lattice.include_misalignments)
        neighboring_elements = [_el for _el in lattice if is_valid_neighbors(el, _el)]
        col = [field_generator_in_different_frame1(el_neighb, el, magnet_options) for el_neighb in neighboring_elements]
        return col if len(col) != 0 else None
    else:
        return None
