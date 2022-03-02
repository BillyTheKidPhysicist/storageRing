import HalbachLensClass_Tests
import geneticLensClass_Tests
import lattice_Tracing_Tests
import shimOptimizerOfLens_Focus_Tests
import parallel_Gradient_Descent


def test_HalbachLensClass():
    HalbachLensClass_Tests.test()
def poop_GeneticLensClass():
    geneticLensClass_Tests.test()
def test_shimOptimizerOfLens_Focus():
    shimOptimizerOfLens_Focus_Tests.run_Tests()
def test_Parallel_Gradient_Descent():
    parallel_Gradient_Descent.test1()
    parallel_Gradient_Descent.test2()
# def test_Lattice_Tracing_Full_Integration():
#     lattice_Tracing_Tests.test()
