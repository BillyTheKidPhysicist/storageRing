import multiprocess as mp
import numpy as np
from swarm_tracer import SwarmTracer
from particle_tracer_lattice import ParticleTracerLattice
from particle import Particle, Swarm
import os
from particle_tracer import ParticleTracer

# pylint: disable=too-many-locals, too-many-arguments,missing-function-docstring

testDataFolderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'testData')
V0 = 200.0


def generate_Test_Swarm(PTL):
    swarmTracer = SwarmTracer(PTL)
    testSwarm = swarmTracer.pseudorandom_swarm(q_trans_bounds=10e-3, p_trans_bounds=10.0,delta_px_bounds= 5.0,
                                               num_particles=30, same_seed=True)
    return testSwarm


def _save_TEST_Data(PTL, testSwarm, TESTDataFileName):
    """swarm is traced through the lattice with all fancy features turned off"""
    TESTDataFilePath = os.path.join(testDataFolderPath, TESTDataFileName)
    swarmTracer = SwarmTracer(PTL)
    tracedSwarm = swarmTracer.trace_swarm_through_lattice(testSwarm, 1e-5, .25, use_fast_mode=False, parallel=False)
    testData = []
    for particle in tracedSwarm:
        pf = particle.pf if particle.pf is not None else np.nan * np.empty(3)
        qf = particle.qf if particle.qf is not None else np.nan * np.empty(3)
        revolutions = particle.revolutions
        EFinal = particle.E_vals[-1] if len(particle.E_vals) > 0 else np.nan
        testData.append(np.append(np.append(np.append(qf, pf), revolutions), EFinal))
    np.savetxt(os.path.join(testDataFolderPath, TESTDataFilePath), np.asarray(testData))


def TEST_Lattice_Tracing(PTL, testSwarm, TESTDataFileName, use_fast_mode, parallel):
    np.set_printoptions(precision=100)
    TESTDataFilePath = os.path.join(testDataFolderPath, TESTDataFileName)
    swarmTracer = SwarmTracer(PTL)
    tracedSwarm = swarmTracer.trace_swarm_through_lattice(testSwarm, 1e-5, .25, use_fast_mode=use_fast_mode,
                                                          parallel=parallel)
    testData = np.loadtxt(TESTDataFilePath)
    assert tracedSwarm.num_particles() == testSwarm.num_particles() and len(testData) == tracedSwarm.num_particles()
    eps = 1e-9  # a small number to represent changes in values that come from different kinds of operations. Because of
    # the nature of digitla computing, the same algorithm done in a different way can give slightly different answers
    # in the last few digits on different computers
    for i in range(len(tracedSwarm.particles)):
        if tracedSwarm.particles[i].T != 0.0:  # some particle get clipped right away, so don't check them
            qf = tracedSwarm.particles[i].qf
            qTest = testData[i, :3]
            pf = tracedSwarm.particles[i].pf
            pTest = testData[i, 3:6]
            revs = tracedSwarm.particles[i].revolutions
            assert not np.any(np.isnan(qf)) and not np.any(
                np.isnan(pf))  # This should never be nan for a traced particle
            # though for saved data it will nan for particles that clipped right away. But they wont enter this loop
            revsTest = testData[i, 6]
            EFinalTest = testData[i, 7]
            condition = (np.all(np.abs(qf - qTest) < eps) and np.all(np.abs(pf - pTest) < eps) and np.abs(
                revs - revsTest) < eps)
            if use_fast_mode == False:  # include energy considerations
                EFinalTraced = tracedSwarm.particles[i].E_vals[-1]
                condition = condition and np.abs(EFinalTest - EFinalTraced) < eps
            if condition == False:
                print('q:', qf)
                print('qTest:', qTest)
                print('difference:', qTest - qf)
                print('p:', pf)
                print('pTest:', pTest)
                print('difference:', pTest - pf)
                if use_fast_mode == False:
                    print('Energy: ', EFinalTraced)
                    print('EnergyTest: ', EFinalTest)
                    print('difference: ', EFinalTest - EFinalTraced)
                raise Exception('Failed on test: ' + TESTDataFileName)


def generate_Lattice(configuration):
    # a variety of lattice configurations are tested
    if configuration == '1':
        PTL = ParticleTracerLattice(design_speed=200.0)
        PTL.add_drift(.25)
        PTL.add_segmented_halbach_bender(.0254, .01, 150, 1.0)
        PTL.add_lens_ideal(1.0, 1.0, .01)
        PTL.add_halbach_lens_sim(.01, 1.0)
        PTL.add_drift(.1)
        PTL.end_lattice(constrain=False)
    elif configuration in ('2', '5'):
        PTL = ParticleTracerLattice(design_speed=200.0)
        PTL.add_drift(.25)
        PTL.add_halbach_lens_sim(.01, .5)
        PTL.add_drift(.1)
        if configuration == '2':
            PTL.add_combiner_sim(atom_state='HIGH_SEEK')
        else:
            PTL.add_combiner_sim_lens(.1, .02, layers=2,atom_state='HIGH_SEEK')
        PTL.add_halbach_lens_sim(.01, .5)
        PTL.end_lattice()
    elif configuration == '3':
        PTL = ParticleTracerLattice(design_speed=200.0)
        PTL.add_lens_ideal(1.0, 1.0, .01)
        PTL.add_bender_ideal(np.pi, 1.0, 1.0, .01)
        PTL.add_lens_ideal(1.0, 1.0, .01)
        PTL.add_bender_ideal(np.pi, 1.0, 1.0, .01)
        PTL.end_lattice()
    elif configuration in ('4', '6'):
        PTL = ParticleTracerLattice(design_speed=200.0)
        PTL.add_halbach_lens_sim(.01, .5)
        if configuration == '4':
            PTL.add_combiner_sim()
        else:
            PTL.add_combiner_sim_lens(.1, .02, layers=2)
        PTL.add_halbach_lens_sim(.01, .5)
        PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
        PTL.add_halbach_lens_sim(.01, None, constrain=True)
        PTL.add_segmented_halbach_bender(.0254 / 2, .01, None, 1.0)
        PTL.end_lattice(constrain=True)
        PTL.el_list[0].update_field_fact(.3)
        PTL.el_list[2].update_field_fact(.3)
    else:
        raise Exception('no proper configuration name provided')
    return PTL


def TEST_Lattice_Configuration(configuration, fullTest=False, save_data=False, parallel=False):
    PTL = generate_Lattice(configuration)
    testSwarm = generate_Test_Swarm(PTL)
    TESTName = 'test_' + configuration
    if save_data == True:
        _save_TEST_Data(PTL, testSwarm, TESTName)
    elif fullTest == True:
        for use_fast_mode in [True, False]:
            for parallel in [True, False]:
                TEST_Lattice_Tracing(PTL, testSwarm, TESTName, use_fast_mode, parallel)
    elif fullTest == False:
        use_fast_mode1, parallel1 = True, parallel
        TEST_Lattice_Tracing(PTL, testSwarm, TESTName, use_fast_mode1, parallel)
        use_fast_mode2, parallel2 = False, False
        TEST_Lattice_Tracing(PTL, testSwarm, TESTName, use_fast_mode2, parallel)


def _save_New_Data():
    tests = ['1', '2', '3', '4', '5', '6']
    for testNum in tests:
        print('Test number ' + testNum)
        TEST_Lattice_Configuration(testNum, save_data=True)
        print('Saved successfully')


def _full_Test():
    tests = ['1', '2', '3', '4', '5', '6']
    for testNum in tests:
        print('Test number ' + testNum)
        TEST_Lattice_Configuration(testNum, fullTest=True)
        print('Success')


def test_Lattices(parallelTesting=True, fullTest=False):
    """Keep in mind that due to the fact that a seed is used with scipy differential evolution, do not expect
    repeatable results with different machines or random number generator systems. This will only affect some of the
    lattice configurations"""

    def wrap(name):
        return TEST_Lattice_Configuration(name, fullTest=fullTest, parallel=not parallelTesting)

    testNameList = ['1', '2', '3', '4', '5', '6']
    if parallelTesting == False:
        [wrap(test) for test in testNameList]
    else:
        with mp.Pool(10) as pool:
            pool.map(wrap, testNameList)