import os
from ParticleTracerLatticeClass import ParticleTracerLattice
from storageRingConstraintSolver import _build_Storage_Ring_Geometry_From_PTL
from helperTools import *
from math import isclose

testDataFolderPath = os.path.join(os.getcwd(), 'testData')


def _make_Lattice_1():
    PTL = ParticleTracerLattice()
    for _ in range(4):
        PTL.add_Drift(.25)
    PTL.add_Bender_Ideal(np.pi, 1.0, 1.0, .01)
    for _ in range(10):
        PTL.add_Drift(1.0 / 10.0)
    PTL.add_Bender_Ideal(np.pi, 1.0, 1.0, .01)
    PTL.end_Lattice(constrain=False)
    return PTL


def _make_Lattice_2():
    rb = 1.0019084934446032  # this was found by minimizing abs(angle-np.pi) as a functino of bending radus combined
    # with tweaking numMagnets
    Lm, rp, numMagnets = .05, .01, 62
    # sol=minimize(ang,[1.0],bounds=[(.9,1.1)], method='Nelder-Mead',options={'ftol':1e-14} )
    PTL = ParticleTracerLattice(fieldDensityMultiplier=.5)
    PTL.add_Drift(1.0)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm, rp, numMagnets, rb)
    PTL.add_Drift(1.0)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm, rp, numMagnets, rb)
    PTL.end_Lattice()
    return PTL


def _test_Storage_Ring_Constraint_1():
    """Test that a few lattice with hand picked parameters are closed as anticipated"""

    PTL_List = [_make_Lattice_1(), _make_Lattice_2()]
    for PTL in PTL_List:
        storageRingGeom = _build_Storage_Ring_Geometry_From_PTL(PTL, False)
        posSep, normSep = storageRingGeom.get_End_Separation_Vectors()
        assert iscloseAll(posSep, normSep, 1e-10)


def _test_Storage_Ring_Constraint_2():
    """Test that a lattice which gets constrained has expected properties of being closed"""

    rb = 1.0
    Lm, rp = .05, .01
    PTL = ParticleTracerLattice(fieldDensityMultiplier=.5)
    PTL.add_Drift(.5)
    PTL.add_Combiner_Sim_Lens(.1, .02, ap=.75 * rp, layers=2)
    PTL.add_Drift(.5)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm, rp, None, rb)
    PTL.add_Halbach_Lens_Sim(.015, None, constrain=True, ap=.75 * rp)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm, rp, None, rb)
    PTL.end_Lattice(constrain=True)
    totalBendAngle = PTL.elList[1].ang + PTL.elList[3].ang + PTL.elList[5].ang
    assert isclose(totalBendAngle, 2 * np.pi, abs_tol=1e-11)
    assert iscloseAll(PTL.elList[0].r1, PTL.elList[-1].r2, 1e-11)
    assert iscloseAll(PTL.elList[0].nb, -PTL.elList[-1].ne, 1e-11)


def _assert_Consraint_Match_Saved_Vals(PTL: ParticleTracerLattice, fileName: str):
    r1_2TestArr = np.loadtxt(os.path.join(testDataFolderPath, fileName))
    r1TestArr, r2TestArr = r1_2TestArr[:, :3], r1_2TestArr[:, 3:]
    for el, r1Test, r2Test in zip(PTL.elList, r1TestArr, r2TestArr):
        assert iscloseAll(el.r1,r1Test,1e-14) and iscloseAll(el.r2, r2Test,1e-14)


def _test_Storage_Ring_Constraint_3():
    """Test that the results of constructing a lattice are repeatable. For a version 1 lattice in my naming scheme"""

    PTL = ParticleTracerLattice(v0Nominal=200.0, latticeType='storageRing')
    PTL.add_Drift(.02)
    PTL.add_Halbach_Lens_Sim(.01, .5)
    PTL.add_Drift(.02)
    PTL.add_Combiner_Sim_Lens(.1, .02, layers=2)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Lens_Sim(.01, .5)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Bender_Sim_Segmented(.0254 / 2, .01, None, 1.0, 0.0, rOffsetFact=1.015)
    PTL.add_Halbach_Lens_Sim(.01, None, constrain=True)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Lens_Sim(.01, None, constrain=True)
    PTL.add_Halbach_Bender_Sim_Segmented(.0254 / 2, .01, None, 1.0, 0.0, rOffsetFact=1.015)
    PTL.end_Lattice(constrain=True)
    _assert_Consraint_Match_Saved_Vals(PTL, 'storageRingConstTest3')


def _test_Storage_Ring_Constraint_4():
    """Test that the results of constructing a lattice are repeatable. For a version 3 lattice in my naming scheme"""

    PTL = ParticleTracerLattice(v0Nominal=200.0, latticeType='storageRing')
    PTL.add_Drift(.02)
    PTL.add_Halbach_Lens_Sim(.01, .5)
    PTL.add_Drift(.02)
    PTL.add_Combiner_Sim_Lens(.1, .02, layers=2)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Lens_Sim(.01, .5)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Bender_Sim_Segmented(.0254 / 2, .01, None, 1.0, 0.0, rOffsetFact=1.015)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Bender_Sim_Segmented(.0254 / 2, .01, None, 1.0, 0.0, rOffsetFact=1.015)
    PTL.add_Halbach_Lens_Sim(.01, None, constrain=True)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Lens_Sim(.01, None, constrain=True)
    PTL.add_Halbach_Bender_Sim_Segmented(.0254 / 2, .01, None, 1.0, 0.0, rOffsetFact=1.015)
    PTL.add_Drift(.02)
    PTL.add_Halbach_Bender_Sim_Segmented(.0254 / 2, .01, None, 1.0, 0.0, rOffsetFact=1.015)
    PTL.end_Lattice(constrain=True)
    _assert_Consraint_Match_Saved_Vals(PTL, 'storageRingConstTest4')


def test_Storage_Ring_Constraints():
    tests = [_test_Storage_Ring_Constraint_1,
             _test_Storage_Ring_Constraint_2,
             _test_Storage_Ring_Constraint_3,
             _test_Storage_Ring_Constraint_4]

    def run(func):
        func()

    tool_Parallel_Process(run, tests)