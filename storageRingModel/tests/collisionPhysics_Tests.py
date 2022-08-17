from collision_physics import max_momentum_1D_in_trap, trim_trans_momentum_to_max, collision_rate, \
    collision_partner_momentum_lens, collision_partner_momentum_bender, trim_longitudinal_momentum_to_max
from math import isclose
from helper_tools import is_close_all,parallel_evaluate
import numpy as np
from constants import SIMULATION_MAGNETON


def test_max_Momentum_1D_In_Trap():
    rp = .02314
    Fconst = 0.0
    Bp = .75
    assert max_momentum_1D_in_trap(rp * .5, rp, Fconst) == max_momentum_1D_in_trap(-rp * .5, rp, Fconst)
    assert max_momentum_1D_in_trap(rp, rp, Fconst) == 0.0
    assert max_momentum_1D_in_trap(0.0, rp, Fconst) == np.sqrt(2 * Bp * SIMULATION_MAGNETON)

    Fconst = 10000.0
    Bp = .75
    rMax = Fconst / (2 * Bp * SIMULATION_MAGNETON / rp ** 2)
    assert max_momentum_1D_in_trap(rMax, rp, Fconst) > max_momentum_1D_in_trap(rMax * .99, rp, Fconst)
    assert max_momentum_1D_in_trap(rMax, rp, Fconst) > max_momentum_1D_in_trap(rMax * 1.01, rp, Fconst)


def test_trim_Momentum_To_Maximum():
    assert trim_trans_momentum_to_max(1000.0, 0.0, 1.0) == -trim_trans_momentum_to_max(-1000.0, 0.0,
                                                                                                         1.0)
    assert trim_trans_momentum_to_max(1.0, 0.99999, 1.0) == -trim_trans_momentum_to_max(-1.0, 0.99999,
                                                                                                          1.0)
    assert trim_trans_momentum_to_max(1.0, 0.99999, 1.0,
                                      F_centrifugal=100) != -trim_trans_momentum_to_max(-1.0, 0.99999,
                                                                                        1.0)


def test_collision_Rate():
    assert collision_rate(0.00, .01) != .0
    assert isclose(collision_rate(.01, .01) / collision_rate(.01, .02), 4.0)
    assert isclose(collision_rate(.08, .01) / collision_rate(.02, .01), 2.0, abs_tol=1e-2)
    assert 0.0 < collision_rate(.05, .01) < 1000.0


def test_collision_Partner_Momentum_Lens():
    assert is_close_all(collision_partner_momentum_lens((0.0, 0.0, 0.0), 210.0, 0.0, .01), (210.0, 0.0, 0.0), 1e-12)
    px, py, pz = collision_partner_momentum_lens((.5, 0.0, 0.0), 210.0, .09, .015)
    assert abs(py) < px and abs(pz) < px and pz != 0.0 and py != 0.0 and px != 0.0


def test_collision_Partner_Momentum_Bender():
    rBend = 1.032435
    px, py, pz = collision_partner_momentum_bender((rBend + 1e-3, 0.0, 0.0), 210.0, .01, .015, rBend)
    assert py < -150 and px != 0.0 and pz != 0.0 and abs(px) < abs(py) and abs(pz) < abs(py)
