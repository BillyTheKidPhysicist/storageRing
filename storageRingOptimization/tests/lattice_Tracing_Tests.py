import multiprocess as mp
import numpy as np
from SwarmTracerClass import SwarmTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleClass import Particle,Swarm
import os
from ParticleTracerClass import ParticleTracer
#TODO: test swarmTracer class features
testDataFolderPath=os.path.join(os.getcwd(),'testData')
V0=200.0
def generate_Test_Swarm(configuration):
    testSwarm = Swarm()
    if configuration in ('1','2','3','4','5','6'):
        testSwarm.add_Particle(pi=np.asarray([-V0, 0.0, 0.0]))
        testSwarm.add_Particle(qi=np.asarray([-1e-10, 1e-3, 0.0]),pi=np.asarray([-V0, 0.0, 0.0]))
        testSwarm.add_Particle(pi=np.asarray([-200.0, 5.0, 0.0]))
    elif configuration in (None,):
        pass
    else:
        raise Exception('no valid configuration given')
    return testSwarm

def _save_TEST_Data(PTL,testSwarm,TESTDataFileName):
    #swarm is traced through the lattice with all fancy features turned off
    TESTDataFilePath=os.path.join(testDataFolderPath,TESTDataFileName)
    if os.path.exists(TESTDataFilePath) == True:
        raise Exception("A results file already exists for that test case")
    swarmTracer=SwarmTracer(PTL)
    tracedSwarm=swarmTracer.trace_Swarm_Through_Lattice(testSwarm,1e-5,1.0,fastMode=False,parallel=False,accelerated=False)
    testData=[]
    for particle in tracedSwarm:
        pf=particle.pf
        qf=particle.qf
        revolutions=particle.revolutions
        EFinal=particle.EArr[-1]
        testData.append(np.append(np.append(np.append(qf,pf),revolutions),EFinal))
    np.savetxt(os.path.join(testDataFolderPath,TESTDataFilePath),np.asarray(testData))
    
def TEST_Lattice_Tracing(PTL,testSwarm,TESTDataFileName,fastMode,accelerated):
    np.set_printoptions(precision=100)
    TESTDataFilePath=os.path.join(testDataFolderPath,TESTDataFileName)
    swarmTracer=SwarmTracer(PTL)
    tracedSwarm=swarmTracer.trace_Swarm_Through_Lattice(testSwarm,1e-5,1.0,fastMode=fastMode,accelerated=accelerated,
                                                        parallel=False)
    assert tracedSwarm.num_Particles()==testSwarm.num_Particles()
    testData=np.loadtxt(TESTDataFilePath)
    eps=1e-9 # a small number to represent changes in values that come from different kinds of operations. Because of
    #the nature of digitla computing, the same algorithm done in a different way can give slightly different answers
    #in the last few digits on different computers
    for i in range(len(tracedSwarm.particles)):
        qf=tracedSwarm.particles[i].qf
        qTest=testData[i,:3]
        pf=tracedSwarm.particles[i].pf
        pTest=testData[i,3:6]
        revs=tracedSwarm.particles[i].revolutions
        revsTest=testData[i,6]
        EFinalTest=testData[i,7]
        condition=(np.all(np.abs(qf-qTest)<eps) and np.all(np.abs(pf-pTest)<eps) and np.abs(revs-revsTest)<eps)
        if fastMode==False: #include energy considerations
            EFinalTraced=tracedSwarm.particles[i].EArr[-1]
            condition=condition and np.abs(EFinalTest-EFinalTraced)<eps
        if condition==False:
            print('q:',qf)
            print('qTest:',qTest)
            print('difference:',qTest-qf)
            print('p:',pf)
            print('pTest:',pTest)
            print('difference:',pTest-pf)
            if fastMode==False:
                print('Energy: ',EFinalTraced)
                print('EnergyTest: ',EFinalTest)
                print('difference: ',EFinalTest-EFinalTraced)
            assert False,'Failed on test: '+TESTDataFileName
def generate_Lattice(configuration):
    #a variety of lattice configurations are tested
    if configuration=='1':
        PTL=ParticleTracerLattice(200.0,latticeType='storageRing')
        PTL.add_Drift(.25)
        PTL.add_Halbach_Bender_Sim_Segmented(.0254,.01,150,1.0,0.0,rOffsetFact=1.015)
        PTL.add_Lens_Ideal(1.0,1.0,.01)
        PTL.add_Halbach_Lens_Sim(.01,1.0)
        PTL.add_Drift(.1)
        PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
    elif configuration in ('2','5'):
        PTL=ParticleTracerLattice(200.0,latticeType='injector')
        PTL.add_Drift(.25)
        PTL.add_Halbach_Lens_Sim(.01,.5)
        PTL.add_Drift(.1)
        if configuration=='2':
            PTL.add_Combiner_Sim()
        else:
            PTL.add_Combiner_Sim_Lens(.1, .02)
        PTL.add_Halbach_Lens_Sim(.01, .5)
        PTL.end_Lattice()
    elif configuration=='3':
        PTL=ParticleTracerLattice(200.0,latticeType='storageRing')
        PTL.add_Lens_Ideal(1.0,1.0,.01)
        PTL.add_Bender_Ideal(np.pi,1.0,1.0,.01)
        PTL.add_Lens_Ideal(1.0,1.0,.01)
        PTL.add_Bender_Ideal(np.pi,1.0,1.0,.01)
        PTL.end_Lattice()
    elif configuration in ('4','6'):
        PTL=ParticleTracerLattice(200.0,latticeType='storageRing')
        PTL.add_Halbach_Lens_Sim(.01,.5)
        if configuration == '4':
            PTL.add_Combiner_Sim()
        else:
            PTL.add_Combiner_Sim_Lens(.1, .02)
        PTL.add_Halbach_Lens_Sim(.01,.5)
        PTL.add_Halbach_Bender_Sim_Segmented(.0254/2,.01,None,1.0,0.0,rOffsetFact=1.015)
        PTL.add_Halbach_Lens_Sim(.01,None,constrain=True)
        PTL.add_Halbach_Bender_Sim_Segmented(.0254/2,.01,None,1.0,0.0,rOffsetFact=1.015)
        PTL.end_Lattice(enforceClosedLattice=True,constrain=True)
        PTL.elList[0].update_Field_Fact(.3)
        PTL.elList[2].update_Field_Fact(.3)
    else:
        raise Exception('no proper configuration name provided')
    return PTL
def TEST_Lattice_Configuration(configuration,fullTest=False,saveData=False):
    PTL=generate_Lattice(configuration)
    testSwarm=generate_Test_Swarm(configuration)
    TESTName='test_'+configuration
    if saveData==True:
        _save_TEST_Data(PTL,testSwarm,TESTName)
    elif fullTest==True:
        for fastMode in [True,False]:
                for accelerated in [True,False]:
                    TEST_Lattice_Tracing(PTL,testSwarm, TESTName, fastMode, accelerated)
    elif fullTest==False:
        fastMode1,accelerated1=True,True
        TEST_Lattice_Tracing(PTL,testSwarm, TESTName, fastMode1, accelerated1)
        fastMode2,accelerated2=False,False
        TEST_Lattice_Tracing(PTL,testSwarm, TESTName, fastMode2, accelerated2)
def _save_New_Data():
    tests = ['1', '2', '3', '4', '5', '6']
    for testNum in tests:
        print('Test number ' + testNum)
        TEST_Lattice_Configuration(testNum, saveData=True)
        print('Saved successfully')
def _full_Test():
    tests = ['1', '2', '3', '4', '5', '6']
    for testNum in tests:
        print('Test number ' + testNum)
        TEST_Lattice_Configuration(testNum, fullTest=True)
        print('Success')
def run_Tests(parallelTesting=False,fullTest=False):
    def wrap(x):
        return TEST_Lattice_Configuration(x,fullTest=fullTest)
    testNameList = ['1', '2', '3', '4', '5', '6']
    if parallelTesting==False:
        [wrap(test) for test in testNameList]
    else:
        with mp.Pool(10) as pool:
            pool.map(wrap,testNameList)