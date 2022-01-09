import numpy as np
from SwarmTracerClass import SwarmTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleClass import Particle,Swarm
import os
from ParticleTracerClass import ParticleTracer
#TODO: implement testing parallel versus serial
testDataFolderPath=os.path.join(os.getcwd(),'testData')
testSwarm=Swarm()
testSwarm.add_Particle()
testSwarm.add_Particle(qi=np.asarray([-1e-10,1e-3,0.0]))
testSwarm.add_Particle(pi=np.asarray([-200.0,5.0,0.0]))

def save_TEST_Data(PTL,TESTDataFileName):
    TESTDataFilePath=os.path.join(testDataFolderPath,TESTDataFileName)
    swarmTracer=SwarmTracer(PTL)
    tracedSwarm=swarmTracer.trace_Swarm_Through_Lattice(testSwarm,1e-5,1.0,fastMode=False,parallel=False)
    testData=[]
    for particle in tracedSwarm:
        pf=particle.pf
        qf=particle.qf
        revolutions=particle.revolutions
        EFinal=particle.EArr[-1]
        testData.append(np.append(np.append(np.append(qf,pf),revolutions),EFinal))
    np.savetxt(os.path.join(testDataFolderPath,TESTDataFilePath),np.asarray(testData))
    
def TEST_Lattice_Tracing(PTL,fastMode,accelerated,TESTDataFileName):
    np.set_printoptions(precision=100)
    TESTDataFilePath=os.path.join(testDataFolderPath,TESTDataFileName)
    swarmTracer=SwarmTracer(PTL)
    tracedSwarm=swarmTracer.trace_Swarm_Through_Lattice(testSwarm,1e-5,1.0,fastMode=fastMode,parallel=False
                                                        ,accelerated=accelerated)
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
            raise Exception('Failed test')
        assert condition,'Failed on particle: '+str(i)
def generate_Lattice(configuration):
    if configuration=='1':
        PTL=ParticleTracerLattice(200.0,latticeType='storageRing')
        PTL.add_Drift(.25)
        PTL.add_Halbach_Bender_Sim_Segmented_With_End_Cap(.0254,.01,150,1.0,0.0,rOffsetFact=1.015)
        PTL.add_Lens_Ideal(1.0,1.0,.01)
        PTL.add_Halbach_Lens_Sim(.01,1.0)
        PTL.add_Drift(.1)
        PTL.end_Lattice(constrain=False,surpressWarning=True,enforceClosedLattice=False)
    elif configuration=='2':
        PTL=ParticleTracerLattice(200.0,latticeType='injector')
        PTL.add_Drift(.25)
        PTL.add_Halbach_Lens_Sim(.01,.5)
        PTL.add_Drift(.1)
        PTL.add_Combiner_Sim('combinerV3.txt')
        PTL.end_Lattice()
    elif configuration=='3':
        PTL=ParticleTracerLattice(200.0,latticeType='storageRing')
        PTL.add_Lens_Ideal(1.0,1.0,.01)
        PTL.add_Bender_Ideal(np.pi,1.0,1.0,.01)
        PTL.add_Lens_Ideal(1.0,1.0,.01)
        PTL.add_Bender_Ideal(np.pi,1.0,1.0,.01)
        PTL.end_Lattice()
    elif configuration=='4':
        PTL=ParticleTracerLattice(200.0,latticeType='storageRing')
        PTL.add_Halbach_Lens_Sim(.01,.5)
        PTL.add_Combiner_Sim('combinerV3.txt')
        PTL.add_Halbach_Lens_Sim(.01,.5)
        PTL.add_Halbach_Bender_Sim_Segmented_With_End_Cap(.0254/2,.01,None,1.0,0.0,rOffsetFact=1.015)
        PTL.add_Halbach_Lens_Sim(.01,None,constrain=True)
        PTL.add_Halbach_Bender_Sim_Segmented_With_End_Cap(.0254/2,.01,None,1.0,0.0,rOffsetFact=1.015)
        PTL.end_Lattice(enforceClosedLattice=True,constrain=True)
        PTL.elList[0].fieldFact=.3
        PTL.elList[2].fieldFact=.3
    else:
        raise Exception('no proper configuration name provided')
    return PTL
def TEST_Lattice_Configuration(configuration,saveData):
    PTL=generate_Lattice(configuration)
    TESTName='test_'+configuration
    if saveData==True:
        input("You are requesting to overwrite existing data, press enter to confirm")
        save_TEST_Data(PTL,TESTName)
    else:
        TEST_Lattice_Tracing(PTL,False,False,TESTName)
        TEST_Lattice_Tracing(PTL,True,False,TESTName)
        TEST_Lattice_Tracing(PTL,True,True,TESTName)
def TEST_1(saveData=False):
    #straight lattice
    TEST_Lattice_Configuration('1',saveData)
def TEST_2(saveData=False):
    #injector lattice of simulated fields
    TEST_Lattice_Configuration('2',saveData)
def TEST_3(saveData=False):
    #storage ring lattice of all ideal element
    TEST_Lattice_Configuration('3',saveData)
def TEST_4(saveData=False):
    #storage ring lattice with simulated elements and simulated combiner
    TEST_Lattice_Configuration('4',saveData)
def TEST_All():
    print('1')
    TEST_1()
    print('2')
    TEST_2()
    print('3')
    TEST_3()
    print('4')
    TEST_4()
TEST_All()