
from ParticleTracerLatticeClass import ParticleTracerLattice
from storageRingConstraintSolver import _build_Storage_Ring_Geometry_From_PTL
from helperTools import *
from math import isclose

def _make_Lattice_1():
    PTL=ParticleTracerLattice()
    for _ in range(4):
        PTL.add_Drift(.25)
    PTL.add_Bender_Ideal(np.pi,1.0,1.0,.01)
    for _ in range(10):
        PTL.add_Drift(1.0/10.0)
    PTL.add_Bender_Ideal(np.pi,1.0,1.0,.01)
    PTL.end_Lattice(constrain=False)
    return PTL

def _make_Lattice_2():
    rb=1.0019084934446032 #this was found by minimizing abs(angle-np.pi) as a functino of bending radus combined
    #with tweaking numMagnets
    Lm,rp,numMagnets=.05,.01,62
    # sol=minimize(ang,[1.0],bounds=[(.9,1.1)], method='Nelder-Mead',options={'ftol':1e-14} )
    PTL=ParticleTracerLattice(fieldDensityMultiplier=.5)
    PTL.add_Drift(1.0)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm,rp,numMagnets,rb)
    PTL.add_Drift(1.0)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm,rp,numMagnets,rb)
    PTL.end_Lattice()
    return PTL

def test_Storage_Ring_Constraint_1():
    """Test that a few lattice with hand picked parameters are closed as anticipated"""

    PTL_List=[_make_Lattice_1(),_make_Lattice_2()]
    for PTL in PTL_List:
        storageRingGeom=_build_Storage_Ring_Geometry_From_PTL(PTL,False)
        posSep,normSep=storageRingGeom.get_End_Separation_Vectors()
        assert iscloseAll(posSep,normSep,1e-10)

def test_Storage_Ring_Constraint_2():
    """Test that a lattice which gets constrained has expected properties of being closed"""

    rb = 1.0
    Lm, rp = .05, .01
    PTL = ParticleTracerLattice(fieldDensityMultiplier=.5)
    PTL.add_Drift(.5)
    PTL.add_Combiner_Sim_Lens(.1, .02, apFrac=.75)
    PTL.add_Drift(.5)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm, rp, None, rb)
    PTL.add_Halbach_Lens_Sim(.015, None, constrain=True, apFrac=.75)
    PTL.add_Halbach_Bender_Sim_Segmented(Lm, rp, None, rb)
    PTL.end_Lattice(constrain=True)
    totalBendAngle = PTL.elList[1].ang + PTL.elList[3].ang + PTL.elList[5].ang
    assert isclose(totalBendAngle, 2 * np.pi, abs_tol=1e-12)
    assert iscloseAll(PTL.elList[0].r1, PTL.elList[-1].r2, 1e-12)
    assert iscloseAll(PTL.elList[0].nb, -PTL.elList[-1].ne, 1e-12)