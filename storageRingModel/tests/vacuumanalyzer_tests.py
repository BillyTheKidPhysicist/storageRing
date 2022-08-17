"""Test that the vacuum analyzer system"""

from math import isclose,pi

import numpy as np

from vacuum_modeling.vacuum_analyzer import VacuumSystem, solve_vac_system,tube_conductance


def test_vacuum_results_with_hand_values():
    # test that differential pumping works as expected
    S1, S2, Q, L, D = 100.0, 1.0, .1, 1.25, .05
    gas_mass=28
    vac_sys = VacuumSystem(gas_mass_Daltons=gas_mass)
    vac_sys.add_chamber(S=0.0, Q=Q)
    vac_sys.add_tube(L, D)
    vac_sys.add_chamber(S=S1)
    vac_sys.add_tube(L, D)
    vac_sys.add_chamber(S=S2)
    solve_vac_system(vac_sys)

    # approximate values from vacuum calculations. Should be within 1%
    C=tube_conductance(gas_mass,D,L)
    P1 = Q / C
    P2 = P1 * C / S1
    P3 = P2 * C / S2
    assert isclose(P1, vac_sys.chambers()[0].P, rel_tol=.01)
    assert isclose(P2, vac_sys.chambers()[1].P, rel_tol=.01)
    assert isclose(P3, vac_sys.chambers()[2].P, rel_tol=.01)


def test_same_vals():
    vac_sys = VacuumSystem()
    vac_sys.add_chamber(S=10.0, Q=1)
    vac_sys.add_tube(.35, .1)
    vac_sys.add_tube(.35, .1)
    vac_sys.add_tube(.35, .1)
    vac_sys.add_chamber(S=.01)
    vac_sys.add_tube(3.0, .25)
    vac_sys.add_chamber(S=100.0)
    solve_vac_system(vac_sys)
    P_vals = [0.09989873972504781, 0.013653031647444466, 8.760724330472494e-06]
    assert isclose(P_vals[0], vac_sys.chambers()[0].P)
    assert isclose(P_vals[1], vac_sys.chambers()[1].P)
    assert isclose(P_vals[2], vac_sys.chambers()[2].P)


def circular_or_linear_system(is_circular):
    vac_sys = VacuumSystem(is_circular=is_circular)
    vac_sys.add_chamber(S=1.0, Q=1)
    vac_sys.add_tube(.5, .01)
    vac_sys.add_tube(.5, .01)
    vac_sys.add_chamber(S=.01)
    if is_circular:
        vac_sys.add_tube(.5, .01)
        vac_sys.add_tube(.5, .01)
    solve_vac_system(vac_sys)
    return vac_sys


def test_circular_vs_linear():
    """Test that two chambers connected by a single pipe and then by two pipes behave as expected. IE for low
    conductance of the pipes, and high gass low and pumping in chamber 1 but low pumping and no external gas load in
    chamber 2, that pressure in chamber 2 should approximately double when two pipes are used. Also compare to saved
    values"""
    vac_sys = circular_or_linear_system(False)
    P_circ = [chamber.P for chamber in vac_sys.chambers()]
    P0_circ = [0.9999876873794514, 0.0012312620548606836]

    vac_sys = circular_or_linear_system(True)
    P_straight = [chamber.P for chamber in vac_sys.chambers()]
    P0_straight = [0.999975405344194, 0.0024594655806102796]
    assert all(isclose(P, P0) for P, P0 in zip(P_circ, P0_circ))
    assert all(isclose(P, P0) for P, P0 in zip(P_straight, P0_straight))

    assert isclose(P_circ[0], P_straight[0], abs_tol=.01)
    assert isclose(P_circ[1], 2 * P_straight[1], abs_tol=.01)

def test_pressure_profile():
    """Test that the pressure profile along a periodic tube is close the value predicted by theory"""
    q = 1e-3
    D = .1
    L = 10.0
    S = 1
    c = 12.4 * D ** 3
    C_eff = 12 * c / L
    C = c / L
    S_eff = 1 / (1 / C_eff + 1 / S)
    Q = q * D * pi * L
    P_max = Q * (1 / (8 * C) + 1 / S)
    P_av = Q / S_eff

    vac_sys = VacuumSystem()
    vac_sys.add_chamber(S=S, Q=0.)
    for _ in range(10):
        vac_sys.add_tube(L, D, q=q)
        vac_sys.add_chamber(S=S, Q=.0)
    solve_vac_system(vac_sys)
    tube = vac_sys.components[11]
    assert isclose(np.mean(tube.P),P_av,rel_tol=.1) and isclose(np.max(tube.P),P_max,rel_tol=.1)
