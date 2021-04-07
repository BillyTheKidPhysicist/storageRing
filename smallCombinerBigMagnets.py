#config with bigger magnets (1/4), and small combiner. Lens before combiner is a reasonable distance from the combiner
import numpy as np
from SwarmTracer import SwarmTracer
import matplotlib.pyplot as plt
from OptimizerClass import LatticeOptimizer
from particleTracerLattice import ParticleTracerLattice
from ParticleTracer import ParticleTracer
from ParticleClass import Particle
import time

def get_Lattice(trackPotential=True):
    lattice = ParticleTracerLattice(200.0)
    directory = 'smallCombinerBigMagnets_Files/'
    fileBend1 = directory + 'benderSeg1.txt'
    fileBend2 = directory + 'benderSeg2.txt'
    fileBender1Fringe = directory + 'benderFringeCap1.txt'
    fileBenderInternalFringe1 = directory + 'benderFringeInternal1.txt'
    fileBender2Fringe = directory + 'benderFringeCap2.txt'
    fileBenderInternalFringe2 = directory + 'benderFringeInternal2.txt'
    file2DLens = directory + 'lens2D.txt'
    file3DLens = directory + 'lens3D.txt'
    fileCombiner = directory + 'combinerV2.txt'
    yokeWidth = .0254 * 5 / 8
    extraSpace = 1e-3  # extra space on each ender between bender segments
    Lm = .0254  # hard edge length of segmented bender
    rp = .0125
    Llens1 = .15  # lens length before drift before combiner inlet
    Llens2 = .3
    Llens3 =0.6660736971880906
    Lcap = 0.01875
    K0 = 12000000  # 'spring' constant of field within 1%
    rb1 = 0.9991839560717193
    rb2 = 1.0011135734421053
    numMagnets1 = 110
    numMagnets2 = 110
    rOffsetFact = 1.00125

    # Llens2New=Llens2-LDrift

    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens1)
    lattice.add_Combiner_Sim(fileCombiner, sizeScale=1.0)
    #lattice.add_Drift(Llens2)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens2)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend1, fileBender1Fringe, fileBenderInternalFringe1, Lm, Lcap, rp,
                                                   K0, numMagnets1, rb1, extraSpace, yokeWidth, rOffsetFact)
    lattice.add_Lens_Sim_With_Caps(file2DLens, file3DLens, Llens3)
    lattice.add_Bender_Sim_Segmented_With_End_Cap(fileBend2, fileBender2Fringe, fileBenderInternalFringe2, Lm, Lcap, rp,
                                                   K0,
                                                   numMagnets2, rb2, extraSpace, yokeWidth, rOffsetFact)
    lattice.end_Lattice(trackPotential=trackPotential,enforceClosedLattice=True,buildLattice=True)
    #print(lattice.solve_Combiner_Constraints())
    #lattice.show_Lattice()
    return lattice

#optimizer=None
def compute_Sol(h,Revs,maxEvals):
    lattice=get_Lattice(trackPotential=True)
    #lattice.show_Lattice()
    T=Revs*lattice.totalLength/lattice.v0Nominal
    optimizer=LatticeOptimizer(lattice)

    sol=optimizer.maximize_Suvival_Through_Lattice(h, T, maxHardsEvals=maxEvals)
    return sol
# compute_Sol(1e-5,50.0,30)
# lattice=get_Lattice()
# swarmTracer=SwarmTracer(lattice)
#
#
#
#
#
#
#
# #find maximum for monte carlo integration
# import skopt
# from ParaWell import ParaWell
# helper=ParaWell()
# bounds = [(.15, .25), (.5, 1.5), (-.1, .1)]
# num=250
# sampler = skopt.sampler.Sobol()
# samples = sampler.generate(bounds, num)
#
# def wrapper(args):
#     Lo,Li,LOffset=args
#     swarm = swarmTracer.initialize_Swarm_At_Combiner_Output(Lo,Li,LOffset, labFrame=False, clipForNextApeture=True,
#                                                             numParticles=5000, fastMode=True) #5000
#     qVec,pVec=swarm.vectorize(onlyUnclipped=True)
#     pxArr=pVec[:,0]+lattice.v0Nominal
#     pyArr=pVec[:,1]
#     pzArr=pVec[:,2]
#     valMaxList=[pxArr.max(),pyArr.max(),pzArr.max()]
#     arrList=[pxArr,pyArr,pzArr]
#     integrationMaxList=[]
#     for i in range(len(valMaxList)):
#         arr=arrList[i]
#         valMax=valMaxList[i]
#         valArr=np.linspace(valMax,0)
#         for val in valArr:
#             frac=np.sum(np.abs(arr)<val)/arr.shape[0]
#             if frac<=.95: #95% survival
#                 integrationMaxList.append(val)
#                 break
#     return integrationMaxList



# argList=samples
# t=time.time()
# results=helper.parallel_Problem(wrapper,argList,onlyReturnResults=True)
# print(time.time()-t)
# pxList=[]
# pyList=[]
# pzList=[]
# for result in results:
#     pxList.append(result[0])
#     pyList.append(result[1])
#     pzList.append(result[2])
# pxArr=np.asarray(pxList)
# pyArr=np.asarray(pyList)
# # pzArr=np.asarray(pzList)
# print(np.round(pxArr.min(),5),np.round(pxArr.max(),5),np.round(np.std(pxArr),5),np.round(np.mean(pxArr),5))#2.77239 3.15021 0.08595 2.89656
# print(np.round(pyArr.min(),5),np.round(pyArr.max(),5),np.round(np.std(pyArr),5),np.round(np.mean(pyArr),5))#1.51252 5.13699 0.87522 2.80488
# print(np.round(pzArr.min(),5),np.round(pzArr.max(),5),np.round(np.std(pzArr),5),np.round(np.mean(pzArr),5))#3.28359 5.02499 0.24238 4.27949

#num=100,numParticles=1000
#2.87565 3.38391 0.12991 3.07914
#1.6302 5.19287 0.91562 2.88932
#3.26431 4.92706 0.26201 4.21084

#num=100, numParticles=5000
#2.80367 3.1185 0.07379 2.89195
#1.48238 5.11805 0.88336 2.81599
#3.4208 4.88048 0.23519 4.27259

#num=250, numParticles=1000
#2.87502 3.3896 0.1256 3.07598
#1.52445 5.16437 0.89249 2.87191
#3.37325 4.92706 0.24737 4.23559


# compute_Sol(1e-5,50.0,30)

# for particle in swarm:
# #     particle.plot_Position(plotYAxis='z')
# qVec=[]
# pVec=[]
# for particle in swarm:
#     if particle.clipped==False:
#         qVec.append(particle.q)
#         pVec.append(particle.p)
# qVec=np.asarray(qVec)
# pVec=np.asarray(pVec)
# print('-------extrema---------------')
#
# # plt.hist(pVec[:,0],bins=25)
# # plt.show()
# print(np.max(qVec[:,1]),np.max(qVec[:,2]))
# print(np.min(qVec[:,1]),np.min(qVec[:,2]))
# print(np.max(pVec[:,0]),np.max(pVec[:,1]),np.max(pVec[:,2]))
# print(np.min(pVec[:,0]),np.min(pVec[:,1]),np.min(pVec[:,2]))
# # print('py',np.mean(pVec[:,1]),np.std(pVec[:,1]))
# # print('pz',np.mean(pVec[:,2]),np.std(pVec[:,2]))
# compute_Sol(1e-5,50.0,30)
# for particle in swarm:
#     print(particle.q,particle.p,particle.clipped,particle.traced)
# lattice.show_Lattice(swarm=swarm)


# lattice=get_Lattice()
# from SwarmTracer import SwarmTracer
# swarmTracer=SwarmTracer(lattice)
# def func():
#     swarmTracer.initialize_Swarm_At_Combiner_Output(.2,1.0,0.0)
# func()
#compute_Sol(1e-5,100,100)
#
# lattice=get_Lattice()
# optimizer=LatticeOptimizer(lattice)
# optimizer.plot_Stability(h=1e-5,cutoff=8.0,gridPoints=40,savePlot=True,plotName='24',showPlot=False)

# # LDriftList=[2,4,6,8,10,12,14,16,18]
# # for LDrift in LDriftList:
# #     lattice=get_Lattice(LDrift*1e-2)
# #     optimizer=LatticeOptimizer(lattice)
# #     optimizer.plot_Stability(h=1e-5,cutoff=8.0,gridPoints=40,savePlot=True,plotName='stabilityPlot'+str(LDrift),showPlot=False)
