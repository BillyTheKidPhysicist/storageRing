from latticeElements.elements import Drift,HalbachLensSim
def get_element_before_and_after(el_center,lattice):
    if (el_center.index == len(lattice.el_list) - 1 or el_center.index == 0) and lattice.lattice_type == 'injector':
        raise Exception('Element cannot be first or last if lattice is injector type')
    el_before_index = el_center.index - 1 if el_center.index != 0 else len(lattice.el_list) - 1
    el_after_index = el_center.index + 1 if el_center.index < len(lattice.el_list) - 1 else 0
    el_before = lattice.el_list[el_before_index]
    el_after = lattice.el_list[el_after_index]
    return el_before, el_after
def move_Injector_Element_Longitudinally(el_center,deltaL,lattice) -> None:
    el_before, el_after = get_element_before_and_after(el_center,lattice)
    assert all(type(el) is Drift  for el in (el_before,el_after)) and type(el_center) is HalbachLensSim
    el_before.set_length(el_before.L+deltaL)
    el_after.set_length(el_after.L-deltaL)
def twist_Knobs(shiftLens1,shiftLens2,lattice):
    lenses=[el for el in lattice.el_list if type(el) is HalbachLensSim]
    assert len(lenses)==2
    for el,shift in zip(lenses,[shiftLens1,shiftLens2]):
        move_Injector_Element_Longitudinally(el,shift,lattice)
    lattice.build_lattice(False)
import copy
def make_Injector_With_Longer_First_Drift(extraLength):
    lattice=copy.deepcopy(lattice_injector)
    first_el=lattice.el_list[0]
    first_el.set_length(first_el.L+extraLength)
    lattice.build_lattice(False)
    return lattice

from helperTools import parallel_evaluate
def shifted_Swarm_Cost(loadingShift):
    try:
        PTL_Injector_Long=make_Injector_With_Longer_First_Drift(loadingShift)
    except:
        return 1.0,0.0
    optimizer = StorageRingModel(lattice_ring, PTL_Injector_Long)
    return optimizer.mode_match(True) #(0.6319767830742709, 5.7820389531454275)
shifts=np.linspace(-.03,.03,9)
results=parallel_evaluate(shifted_Swarm_Cost,shifts,processes=len(shifts))