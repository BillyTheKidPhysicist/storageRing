import time

from helper_tools import *
from storage_ring_modeler import make_optimal_solution_model


def run_test_1():
    """Test that the misalignments keyword produces the same results in the optimizer"""
    t = time.time()
    np.random.seed(42)
    model = make_optimal_solution_model('2', include_mag_cross_talk=False, sim_time_max=1, include_misalignments=True)
    cost, flux, = model.mode_match(parallel=True)
    print(time.time() - t)
    print(cost, flux)


def run_test_2():
    """Test that the optimizer produces the same result"""
    t = time.time()
    np.random.seed(42)
    model = make_optimal_solution_model('2', include_mag_cross_talk=False, sim_time_max=1, include_misalignments=False)
    cost, flux, = model.mode_match(parallel=True)
    print(time.time() - t)
    print(cost, flux)


def run_test_3():
    """Test that the include_mag_cross_talk keyword produces the same results in the optimizer"""
    t = time.time()
    np.random.seed(42)
    model = make_optimal_solution_model('2', include_mag_cross_talk=True, sim_time_max=1, include_misalignments=False)
    cost, flux, = model.mode_match(parallel=True)
    print(time.time() - t)
    print(cost, flux)


def run_test_4():
    """Test that the include_mag_cross_talk and include_misalignments
     keyword produces the same results in the optimizer"""
    t = time.time()
    np.random.seed(42)
    model = make_optimal_solution_model('2', include_mag_cross_talk=True, sim_time_max=1, include_misalignments=False)
    cost, flux, = model.mode_match(parallel=True)
    print(time.time() - t)
    print(cost, flux)


run_test_1()
run_test_2()
run_test_3()
run_test_4()
