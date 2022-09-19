import os
from particle_tracer_lattice import ParticleTracerLattice
from storage_ring_constraint_solver import solve_Floor_Plan, update_and_place_elements_from_floor_plan
from helper_tools import is_close_all,parallel_evaluate
import numpy as np

from math import isclose

testDataFolderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'testData')

#reoptimize
"""
    def cost(X):
        rb=X[0]
        Lm, rp, num_lenses = .05, .01, 62
        PTL = ParticleTracerLattice(field_dens_mult=.5)
        PTL.add_drift(1.0)
        PTL.add_segmented_halbach_bender(Lm, rp, num_lenses, rb)
        PTL.add_drift(1.0)
        PTL.add_segmented_halbach_bender(Lm, rp, num_lenses, rb)
        PTL.end_lattice()
        floor_plan = solve_Floor_Plan(PTL,False)
        posSep, normSep = floor_plan.get_end_separation_vectors()
        _cost=np.linalg.norm(posSep)+np.linalg.norm(normSep)
        print(rb,_cost)
        return _cost
    from scipy.optimize import minimize
    sol = minimize(cost, [1.0], bounds=[(.9, 1.1)], method='Nelder-Mead', options={'ftol': 1e-14})
    print(sol)
"""

def _make_Lattice_1():
    PTL = ParticleTracerLattice()
    for _ in range(4):
        PTL.add_drift(.25)
    PTL.add_bender_ideal(np.pi, 1.0, 1.0, .01)
    for _ in range(10):
        PTL.add_drift(1.0 / 10.0)
    PTL.add_bender_ideal(np.pi, 1.0, 1.0, .01)
    PTL.end_lattice(constrain=False)
    return PTL


def _make_Lattice_2():
    rb = 1.0016496743948662 # this was found by minimizing abs(angle-np.pi) as a functino of bending radus combined
    # with tweaking num_lenses
    Lm, rp, num_lenses = .05, .01, 62
    PTL = ParticleTracerLattice(field_dens_mult=.5)
    PTL.add_drift(1.0)
    PTL.add_segmented_halbach_bender(Lm, rp, num_lenses, rb)
    PTL.add_drift(1.0)
    PTL.add_segmented_halbach_bender(Lm, rp, num_lenses, rb)
    PTL.end_lattice()
    return PTL


def _test_Storage_Ring_Constraint_1():
    """Test that a few lattice with hand picked parameters are closed as anticipated"""

    PTL_List = [_make_Lattice_1(), _make_Lattice_2()]
    for PTL in PTL_List:
        floor_plan = solve_Floor_Plan(PTL,False)
        posSep, normSep = floor_plan.get_end_separation_vectors()
        assert is_close_all(posSep, np.zeros(2), 1e-9) and is_close_all(normSep, np.zeros(2), 1e-9)


def _test_Storage_Ring_Constraint_2():
    """Test that a lattice which gets constrained has expected properties of being closed"""

    rb = 1.0
    Lm, rp = .05, .01
    PTL = ParticleTracerLattice(field_dens_mult=.5)
    PTL.add_drift(.5)
    PTL.add_combiner_sim_lens(.1, .02, ap=.75 * rp, layers=2)
    PTL.add_drift(.5)
    PTL.add_segmented_halbach_bender(Lm, rp, None, rb)
    PTL.add_halbach_lens_sim(.015, None, constrain=True, ap=.75 * rp)
    PTL.add_segmented_halbach_bender(Lm, rp, None, rb)
    PTL.end_lattice(constrain=True)
    totalBendAngle = PTL.el_list[1].ang + PTL.el_list[3].ang + PTL.el_list[5].ang
    assert isclose(totalBendAngle, 2 * np.pi, abs_tol=1e-11)
    assert is_close_all(PTL.el_list[0].r1, PTL.el_list[-1].r2, 1e-11)
    assert is_close_all(PTL.el_list[0].nb, -PTL.el_list[-1].ne, 1e-11)


def _assert_Consraint_Match_Saved_Vals(PTL: ParticleTracerLattice, fileName: str):
    r1_2TestArr = np.loadtxt(os.path.join(testDataFolderPath, fileName))
    r1TestArr, r2TestArr = r1_2TestArr[:, :3], r1_2TestArr[:, 3:]
    for el, r1Test, r2Test in zip(PTL.el_list, r1TestArr, r2TestArr):
        assert is_close_all(el.r1,r1Test,1e-14) and is_close_all(el.r2, r2Test,1e-14)

def save_Results(PTL: ParticleTracerLattice, fileName: str):
    fileName=os.path.join(testDataFolderPath, fileName)
    data=[]
    for el in PTL.el_list:
        data.append(np.append(el.r1,el.r2))
    np.savetxt(fileName,data)

def _test_Storage_Ring_Constraint_3(save=False):
    """Test that the results of constructing a lattice are repeatable. For a version 1 lattice in my naming scheme"""

    PTL = ParticleTracerLattice(design_speed=200.0, lattice_type='storage_ring')
    PTL.add_drift(.02)
    PTL.add_halbach_lens_sim(.01, .5)
    PTL.add_drift(.02)
    PTL.add_combiner_sim_lens(.1, .02, layers=2)
    PTL.add_drift(.02)
    PTL.add_halbach_lens_sim(.01, .5)
    PTL.add_drift(.02)
    PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
    PTL.add_halbach_lens_sim(.01, None, constrain=True)
    PTL.add_drift(.02)
    PTL.add_halbach_lens_sim(.01, None, constrain=True)
    PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
    PTL.end_lattice(constrain=True)
    if not save:
        _assert_Consraint_Match_Saved_Vals(PTL, 'storageRingConstTest3')
    else:
        save_Results(PTL, 'storageRingConstTest3')


def _test_Storage_Ring_Constraint_4(save=False):
    """Test that the results of constructing a lattice are repeatable. For a version 3 lattice in my naming scheme"""

    PTL = ParticleTracerLattice(design_speed=200.0, lattice_type='storage_ring')
    PTL.add_drift(.02)
    PTL.add_halbach_lens_sim(.01, .5)
    PTL.add_drift(.02)
    PTL.add_combiner_sim_lens(.1, .02, layers=2)
    PTL.add_drift(.02)
    PTL.add_halbach_lens_sim(.01, .5)
    PTL.add_drift(.02)
    PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
    PTL.add_drift(.02)
    PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
    PTL.add_halbach_lens_sim(.01, None, constrain=True)
    PTL.add_drift(.02)
    PTL.add_halbach_lens_sim(.01, None, constrain=True)
    PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
    PTL.add_drift(.02)
    PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
    PTL.end_lattice(constrain=True)
    if not save:
        _assert_Consraint_Match_Saved_Vals(PTL, 'storageRingConstTest4')
    else:
        save_Results(PTL, 'storageRingConstTest4')

def save_New_Data():
    _test_Storage_Ring_Constraint_3(save=True)
    _test_Storage_Ring_Constraint_4(save=True)

def test_Storage_Ring_Constraints():
    tests = [_test_Storage_Ring_Constraint_1,
             _test_Storage_Ring_Constraint_2,
             _test_Storage_Ring_Constraint_3,
             _test_Storage_Ring_Constraint_4]

    def run(func):
        func()

    parallel_evaluate(run, tests)