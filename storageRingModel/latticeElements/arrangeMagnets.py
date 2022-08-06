"""
These methods deal with alinging and arranging magnets in the lattice to properly account for magnet-magnet magnetic
field interactions. It's an abomination because of how I wasn't consistent early on with element layouts.
"""

from scipy.spatial.transform import Rotation as Rot

from HalbachLensClass import Collection
from helperTools import *
from latticeElements.class_BaseElement import BaseElement
from latticeElements.elements import HalbachLensSim, Element, CombinerHalbachLensSim, Drift

valid_interp_els = (HalbachLensSim, CombinerHalbachLensSim, Drift)  # elements that can include magnetic fields
# produced by neighboring magnets into their own interpolation
valid_neighors = (HalbachLensSim, CombinerHalbachLensSim)  # elements that can be used to generate  magnetic fields
# that act on another elements

neighbor_index_range = 1


# I should have thought more clearly about what i was doing. Then I wouldn't have had any of these problems. I wouldn't
# have wasted so much time

def is_valid_interp_el(el) -> bool:
    assert isinstance(el, BaseElement)
    return type(el) in valid_interp_els


def is_valid_neighbors(el: Element, el_neighbor: Element) -> bool:
    if not is_valid_interp_el(el) or type(el_neighbor) not in valid_neighors:
        return False
    for index_offset in range(-neighbor_index_range, neighbor_index_range + 1):
        if index_offset != 0:
            if el.index + index_offset == el_neighbor.index:
                return True
    return False


def angle(norm) -> float:
    return full_arctan2(norm[1], norm[0])


def move_combiner_to_target_frame(el_combiner: CombinerHalbachLensSim, el_target, magnet_options) -> Collection:
    r_in_el = el_combiner.transform_lab_coords_into_element_frame(el_combiner.r1)
    magnets = el_combiner.magnet.make_magpylib_magnets(*magnet_options)
    magnets.move(-r_in_el)
    theta_el = np.pi + (angle(el_combiner.ne) - angle(el_target.ne))
    R = Rot.from_rotvec([0, 0, theta_el])
    magnets.rotate(R, anchor=0)
    r1, r2 = el_combiner.r1, el_target.r1
    deltar_el = el_target.transform_lab_coords_into_element_frame(
        r1) - el_target.transform_lab_coords_into_element_frame(r2)
    magnets.move(deltar_el)
    return magnets


def move_lens_to_target_el_frame(el_to_move: Element, el_target: Element,magnet_options):
    magnets = el_to_move.magnet.make_magpylib_magnets(*magnet_options)
    if type(el_target) is CombinerHalbachLensSim:  # the combiner is annoying to deal with because I defined the input
        # not at the origin, but rather along the x axis with some y offset
        # move input to 0,0 in el frame
        r_input = el_target.transform_lab_coords_into_element_frame(el_target.r2)
        magnets.move(r_input)
        # now rotate to correct alignment so that
        psi = angle(-el_to_move.nb) - angle(el_target.ne)
        R = Rot.from_rotvec([0, 0, np.pi + psi])
        magnets.rotate(R, anchor=(0, 0, 0))
        # now move to target el output
        r_target = el_target.r2
        r_move = el_to_move.r1
        deltar_el = el_target.transform_lab_coords_into_element_frame(
            r_move) - el_target.transform_lab_coords_into_element_frame(r_target)
        magnets.move(deltar_el)
        return magnets
    else:
        psi = angle(-el_to_move.nb) - angle(el_target.ne)
        R = Rot.from_rotvec([0, 0, psi])
        magnets.rotate(R, anchor=(0, 0, 0))
        r_target = el_target.r1
        r_move = el_to_move.r1
        deltar_el = el_target.transform_lab_coords_into_element_frame(
            r_move) - el_target.transform_lab_coords_into_element_frame(r_target)
        magnets.move(deltar_el)
        return magnets


def field_generator_in_different_frame1(el_to_move: Element, el_target: Element, magnet_options: tuple):
    if type(el_to_move) is CombinerHalbachLensSim:
        return move_combiner_to_target_frame(el_to_move, el_target,magnet_options)
    else:
        return move_lens_to_target_el_frame(el_to_move, el_target,magnet_options)


def collect_valid_neighboring_magpylib_magnets(el: Element, lattice):
    if is_valid_interp_el(el):
        magnet_options=(lattice.use_mag_errors,lattice.include_misalignments)
        neighboring_elements = [_el for _el in lattice if is_valid_neighbors(el, _el)]
        col = Collection([field_generator_in_different_frame1(el_neighb, el,magnet_options) for el_neighb in neighboring_elements])
        return col
    else:
        return None
