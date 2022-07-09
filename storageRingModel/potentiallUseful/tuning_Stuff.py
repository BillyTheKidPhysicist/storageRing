from latticeElements.elements import Drift,HalbachLensSim
def get_Element_Before_And_After(elCenter,lattice):
    if (elCenter.index == len(lattice.elList) - 1 or elCenter.index == 0) and lattice.latticeType == 'injector':
        raise Exception('Element cannot be first or last if lattice is injector type')
    elBeforeIndex = elCenter.index - 1 if elCenter.index != 0 else len(lattice.elList) - 1
    elAfterIndex = elCenter.index + 1 if elCenter.index < len(lattice.elList) - 1 else 0
    elBefore = lattice.elList[elBeforeIndex]
    elAfter = lattice.elList[elAfterIndex]
    return elBefore, elAfter
def move_Injector_Element_Longitudinally(elCenter,deltaL,lattice) -> None:
    elBefore, elAfter = get_Element_Before_And_After(elCenter,lattice)
    assert all(type(el) is Drift  for el in (elBefore,elAfter)) and type(elCenter) is HalbachLensSim
    elBefore.set_length(elBefore.L+deltaL)
    elAfter.set_length(elAfter.L-deltaL)
def twist_Knobs(shiftLens1,shiftLens2,lattice):
    lenses=[el for el in lattice.elList if type(el) is HalbachLensSim]
    assert len(lenses)==2
    for el,shift in zip(lenses,[shiftLens1,shiftLens2]):
        move_Injector_Element_Longitudinally(el,shift,lattice)
    lattice.build_lattice(False)
import copy
def make_Injector_With_Longer_First_Drift(extraLength):
    lattice=copy.deepcopy(lattice_injector)
    firstEl=lattice.elList[0]
    firstEl.set_length(firstEl.L+extraLength)
    lattice.build_lattice(False)
    return lattice

from helperTools import tool_Parallel_Process
def shifted_Swarm_Cost(loadingShift):
    try:
        PTL_Injector_Long=make_Injector_With_Longer_First_Drift(loadingShift)
    except:
        return 1.0,0.0
    optimizer = StorageRingModel(lattice_ring, PTL_Injector_Long)
    return optimizer.mode_match(True) #(0.6319767830742709, 5.7820389531454275)
shifts=np.linspace(-.03,.03,9)
results=tool_Parallel_Process(shifted_Swarm_Cost,shifts,processes=len(shifts))