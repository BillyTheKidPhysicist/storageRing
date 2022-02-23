import HalbachLensClass_Tests
import geneticLensClass_Tests
import lattice_Tracing_Tests
import shimOptimizerOfLens_Focus_Tests


def test_HalbachLensClass():
    HalbachLensClass_Tests.test()
def test_GeneticLensClass():
    geneticLensClass_Tests.test()
def test_shimOptimizerOfLens_Focus():
    shimOptimizerOfLens_Focus_Tests.test()
def test_Lattice_Tracing_Full_Integration():
    lattice_Tracing_Tests.test()