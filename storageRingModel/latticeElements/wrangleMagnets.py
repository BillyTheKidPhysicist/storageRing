from scipy.spatial.transform import Rotation as Rot

# def solve():
from HalbachLensClass import Collection
from helperTools import *
from latticeElements.class_BaseElement import BaseElement
from latticeElements.elements import HalbachLensSim, Element,CombinerHalbachLensSim
from latticeElements.Magnets import MagneticLens

valid_els = (HalbachLensSim,CombinerHalbachLensSim)

def is_valid_interp_el(el):
    assert isinstance(el, BaseElement)
    return type(el) in valid_els


def is_valid_neighbors(el1: Element, el2: Element):
    if type(el1) not in valid_els or type(el2) not in valid_els:
        return False
    if el1 is el2:
        return False
    for index_offset in [-1, 1]:
        if el1.index + index_offset == el2.index:
            return True
    return False


def angle(norm):
    return full_arctan2(norm[1], norm[0])


def field_generator_in_different_frame(magnet1:MagneticLens, magnet2: MagneticLens)-> Collection:
    assert np.all(magnet1.norm_in_el==magnet2.norm_in_el) and np.all(magnet1.norm_out_el==magnet2.norm_out_el)
    magnets = magnet2.make_magpylib_magnets(False)
    magnets.move(-magnet2.r_in_el)
    psi = angle(magnet2.norm_out_lab) - angle(magnet1.norm_out_lab)
    R = Rot.from_rotvec([0, 0, psi])  # .as_matrix()
    magnets.rotate(R, anchor=(0,0,0))
    magnets.move(magnet1.r_in_el)
    deltar_lab = magnet2.r_in_lab - magnet1.r_in_lab
    theta = -angle(magnet1.norm_out_lab)
    R = Rot.from_rotvec([0, 0, theta]).as_matrix()
    deltar_el=R@deltar_lab
    magnets.move(deltar_el)
    if magnet1.combiner:
        R = Rot.from_rotvec([0, 0.0,np.pi])
        magnets.rotate(R,anchor=(magnet1.r_in_el+magnet1.r_out_el)/2.0)
    return magnets


def collect_valid_neighboring_magpylib_magnets(el, lattice):
    if is_valid_interp_el(el):
        neighboring_magnets = [_el.magnet for _el in lattice if is_valid_neighbors(el, _el)]
        col = Collection([field_generator_in_different_frame(el.magnet, magnet) for magnet in neighboring_magnets])
        return col
    else:
        return None
