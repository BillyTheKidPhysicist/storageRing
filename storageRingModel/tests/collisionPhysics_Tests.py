from collisionPhysics import max_Momentum_1D_In_Trap, trim_Transverse_Momentum_To_Maximum, collision_Rate, \
    collision_Partner_Momentum_Lens, collision_Partner_Momentum_Bender, trim_Longitudinal_Momentum_To_Maximum
from math import isclose
from helperTools import *
from constants import *


def test_max_Momentum_1D_In_Trap():
    rp = .02314
    Fconst = 0.0
    Bp = .75
    assert max_Momentum_1D_In_Trap(rp * .5, rp, Fconst) == max_Momentum_1D_In_Trap(-rp * .5, rp, Fconst)
    assert max_Momentum_1D_In_Trap(rp, rp, Fconst) == 0.0
    assert max_Momentum_1D_In_Trap(0.0, rp, Fconst) == np.sqrt(2 * Bp * SIMULATION_MAGNETON)

    Fconst = 10000.0
    Bp = .75
    rMax = Fconst / (2 * Bp * SIMULATION_MAGNETON / rp ** 2)
    assert max_Momentum_1D_In_Trap(rMax, rp, Fconst) > max_Momentum_1D_In_Trap(rMax * .99, rp, Fconst)
    assert max_Momentum_1D_In_Trap(rMax, rp, Fconst) > max_Momentum_1D_In_Trap(rMax * 1.01, rp, Fconst)


def test_trim_Momentum_To_Maximum():
    assert trim_Transverse_Momentum_To_Maximum(1000.0, 0.0, 1.0) == -trim_Transverse_Momentum_To_Maximum(-1000.0, 0.0,
                                                                                                         1.0)
    assert trim_Transverse_Momentum_To_Maximum(1.0, 0.99999, 1.0) == -trim_Transverse_Momentum_To_Maximum(-1.0, 0.99999,
                                                                                                          1.0)
    assert trim_Transverse_Momentum_To_Maximum(1.0, 0.99999, 1.0,
                                               Fcentrifugal=100) != -trim_Transverse_Momentum_To_Maximum(-1.0, 0.99999,
                                                                                                         1.0)


def test_collision_Rate():
    assert collision_Rate(0.00, .01) != .0
    assert isclose(collision_Rate(.01, .01) / collision_Rate(.01, .02), 4.0)
    assert isclose(collision_Rate(.08, .01) / collision_Rate(.02, .01), 2.0, abs_tol=1e-2)
    assert 0.0 < collision_Rate(.05, .01) < 1000.0


def test_collision_Partner_Momentum_Lens():
    assert iscloseAll(collision_Partner_Momentum_Lens((0.0, 0.0, 0.0), 210.0, 0.0, .01), (210.0, 0.0, 0.0), 1e-12)
    px, py, pz = collision_Partner_Momentum_Lens((.5, 0.0, 0.0), 210.0, .09, .015)
    assert abs(py) < px and abs(pz) < px and pz != 0.0 and py != 0.0 and px != 0.0


def test_collision_Partner_Momentum_Bender():
    rBend = 1.032435
    px, py, pz = collision_Partner_Momentum_Bender((rBend + 1e-3, 0.0, 0.0), 210.0, .01, .015, rBend)
    assert py < -150 and px != 0.0 and pz != 0.0 and abs(px) < abs(py) and abs(pz) < abs(py)
