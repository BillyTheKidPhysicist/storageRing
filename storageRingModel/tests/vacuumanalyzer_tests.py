"""Test that the vacuum analyzer system"""

from math import isclose

from vacuumanalyzer.vacuumanalyzer import VacuumSystem, solve_vac_system, tube_cond_air_fact


def test_vacuum_results_with_hand_values():
    # test that differential pumping works as expected
    S1, S2, Q, L, D = 100.0, 1.0, .1, 1.25, .05
    vac_sys = VacuumSystem()
    vac_sys.add_chamber(S=0.0, Q=Q)
    vac_sys.add_tube(L, D)
    vac_sys.add_chamber(S=S1)
    vac_sys.add_tube(L, D)
    vac_sys.add_chamber(S=S2)
    solve_vac_system(vac_sys)

    # approximate values from vacuum calculations. Should be within 1%
    C = tube_cond_air_fact * D ** 3 / L
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
    P_vals = [0.09990057569584081, 0.013623286962657424, 8.58010171965349e-06]
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
    P0_circ = [0.9999879147693604, 0.0012085230639635306]

    vac_sys = circular_or_linear_system(True)
    P_straight = [chamber.P for chamber in vac_sys.chambers()]
    P0_straight = [0.9999758590054192, 0.002414099458104502]
    assert all(isclose(P, P0) for P, P0 in zip(P_circ, P0_circ))
    assert all(isclose(P, P0) for P, P0 in zip(P_straight, P0_straight))

    assert isclose(P_circ[0], P_straight[0], abs_tol=.01)
    assert isclose(P_circ[1], 2 * P_straight[1], abs_tol=.01)
