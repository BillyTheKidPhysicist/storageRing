import time

from helper_tools import *
from storage_ring_modeler import make_optimal_solution_model
from math import isclose


def run_test_1():
    """Test that the misalignments keyword produces the same results in the optimizer"""
    t = time.time()
    model = make_optimal_solution_model('2', use_long_range_fields=False, include_misalignments=False
                                        ,build_field_helpers=False)
    model.build_field_helpers_if_unbuilt(parallel=True)
    cost, flux, = model.mode_match(parallel=True)
    T_elapsed=time.time() - t
    print(cost, flux,T_elapsed)
    cost0=0.881692127643777
    flux0= 301.248418670809
    T_elapsed0=137
    assert isclose(cost,cost0,abs_tol=1e-6)
    assert isclose(flux,flux0,abs_tol=1e-6)
    assert isclose(T_elapsed,T_elapsed0,abs_tol=10.0)
run_test_1()
