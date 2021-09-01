import numpy as np
from SwarmTracerClass import SwarmTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from ParticleClass import Particle,Swarm
import os
from ParticleTracerClass import ParticleTracer

testDataFolderPath=os.path.join(os.getcwd(),'testData')
testSwarm=Swarm()
testSwarm.add_Particle()
testSwarm.add_Particle(qi=np.asarray([-1e-10,1e-3,0.0]))
testSwarm.add_Particle(pi=np.asarray([-200.0,5.0,0.0]))

def save_Poop_Data(PTL,poopDataFileName):
    poopDataFilePath=os.path.join(testDataFolderPath,poopDataFileName)
    swarmTracer=SwarmTracer(PTL)
    tracedSwarm=swarmTracer.trace_Swarm_Through_Lattice(testSwarm,1e-5,1.0,fastMode=False,parallel=False)
    testData=[]
    for particle in tracedSwarm:
        p=particle.p
        q=particle.q
        revolutions=particle.revolutions
        EFinal=particle.EArr[-1]
        testData.append(np.append(np.append(np.append(q,p),revolutions),EFinal))
    np.savetxt(os.path.join(testDataFolderPath,poopDataFilePath),np.asarray(testData))
    
def poop_Lattice_Tracing(PTL,fastMode,accelerated,poopDataFileName):
    poopDataFilePath=os.path.join(testDataFolderPath,poopDataFileName)
    swarmTracer=SwarmTracer(PTL)
    tracedSwarm=swarmTracer.trace_Swarm_Through_Lattice(testSwarm,1e-5,1.0,fastMode=fastMode,parallel=False
                                                        ,accelerated=accelerated)
    testData=np.loadtxt(poopDataFilePath)
    eps=1e-9 # a small number to represent changes in values that come from different kinds of operations. Because of
    #the nature of digitla computing, the same algorithm done in a different way can give slightly different answers
    #in the last few digits
    for i in range(len(tracedSwarm.particles)):
        q=tracedSwarm.particles[i].q
        qTest=testData[i,:3]
        p=tracedSwarm.particles[i].p
        pTest=testData[i,3:6]
        revs=tracedSwarm.particles[i].revolutions
        revsTest=testData[i,6]
        EFinalTest=testData[i,7]
        condition=(np.all(np.abs(q-qTest)<eps) and np.all(np.abs(p-pTest)<eps) and np.abs(revs-revsTest)<eps)
        if fastMode==False: #include energy considerations
            EFinalTraced=tracedSwarm.particles[i].EArr[-1]
            condition=condition and np.abs(EFinalTest-EFinalTraced)<eps
        if condition==False:
            print('q:',q)
            print('qTest:',qTest)
            print('p:',p)
            print('pTest:',pTest)
            raise Exception('Failed')
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
def poop_Lattice_Configuration(configuration,saveData):
    PTL=generate_Lattice(configuration)
    poopName='test_'+configuration
    if saveData==True:
        input("You are requesting to overwrite existing data, press enter to confirm")
        save_Poop_Data(PTL,poopName)
    else:
        poop_Lattice_Tracing(PTL,False,False,poopName)
        poop_Lattice_Tracing(PTL,True,False,poopName)
        poop_Lattice_Tracing(PTL,True,True,poopName)
def poop_1(saveData=False):
    #straight lattice
    poop_Lattice_Configuration('1',saveData)
def poop_2(saveData=False):
    #injector lattice of simulated fields
    poop_Lattice_Configuration('2',saveData)
def poop_3(saveData=False):
    #storage ring lattice of all ideal element
    poop_Lattice_Configuration('3',saveData)
def poop_4(saveData=False):
    #storage ring lattice with simulated elements and simulated combiner
    poop_Lattice_Configuration('4',saveData)
def poop_All():
    print('1')
    poop_1()
    print('2')
    poop_2()
    print('3')
    poop_3()
    print('4')
    poop_4()
poop_All()