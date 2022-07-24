from scipy.spatial.transform import Rotation as Rot

# def solve():
from HalbachLensClass import Collection
from helperTools import *
from latticeElements.class_BaseElement import BaseElement
from latticeElements.elements import HalbachLensSim, Element

valid_els = (HalbachLensSim,)


def is_valid_interp_el(el):
    assert isinstance(el, BaseElement)
    return type(el) in valid_els


def is_valid_neighbors(el1: Element, el2: Element):
    if type(el1) not in valid_els or type(el2) not in valid_els:
        return False
    if el1 is el2:
        return False
    for index_offset in [-2, -1, 1, 2]:
        if el1.index + index_offset == el2.index:
            return True
    return False


def angle(norm):
    return full_arctan2(norm[1], norm[0])


def field_generator_in_different_frame(magnet1, magnet2):
    magnets = magnet2.make_magpylib_magnets(False)
    deltar = magnet2.r_in_lab - magnet1.r_in_lab
    theta_pos = angle(-magnet1.norm_in_el) - angle(-magnet1.norm_in_lab)
    R = Rot.from_rotvec([0, 0, theta_pos]).as_matrix()
    deltar = R @ deltar
    magnets.move(deltar)
    theta_direction = angle(magnet2.norm_in_lab) - angle(magnet1.norm_in_lab)
    R = Rot.from_rotvec([0, 0, theta_direction])
    magnets.rotate(R)
    return magnets


def collect_valid_neighboring_magnets(el, lattice):
    if is_valid_interp_el(el):
        neighboring_magnets = [_el.magnet for _el in lattice if is_valid_neighbors(el, _el)]
        col = Collection([field_generator_in_different_frame(el.magnet, magnet) for magnet in neighboring_magnets])
        return col
    else:
        return None
