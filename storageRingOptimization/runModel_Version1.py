import os

os.environ['OPENBLAS_NUM_THREADS']='1'
import skopt
import numpy as np
from ParticleClass import Particle
from ParticleTracerClass import ParticleTracer
from skopt.plots import plot_objective,plot_objective_2D
from OptimizerClass import LatticeOptimizer,Solution
import dill
from ParticleTracerLatticeClass import ParticleTracerLattice
import time
import scipy.interpolate as spi
from sendMyselfEmail import send_MySelf_Email
import gc
from profilehooks import profile
import matplotlib.pyplot as plt

import sys


def solve_System(fieldBounds,num,benderMagnetStrength):
    def solve_For_Lattice_Params(X,parallel=False):
        combinerGap=5e-2
        t=time.time()
        rpBend,rpLens,Lm=X
        PTL_Ring=ParticleTracerLattice(200.0,latticeType='storageRing')
        rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm,parallel=parallel)  #25% of time here, 1.0138513851385138
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,.25)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,.25)
        PTL_Ring.add_Combiner_Sim('combinerV3.txt')
        PTL_Ring.add_Drift(combinerGap)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,.25)
        PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
        PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
        PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  #17.8 % of time here
        PTL_Ring.elList[4].fieldFact=benderMagnetStrength
        PTL_Ring.elList[6].fieldFact=benderMagnetStrength

        # PTL_Injector=ParticleTracerLattice(200.0,latticeType='injector')
        # PTL_Injector.add_Drift(.1,ap=.025)
        # PTL_Injector.add_Halbach_Lens_Sim(.025,.2) #15% of time here
        # PTL_Injector.add_Drift(.2,ap=.01)
        # PTL_Injector.add_Combiner_Sim('combinerV3.txt')
        # PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)

        # file=open('injectorFile','wb')
        # dill.dump(PTL_Injector,file)
        file=open('injectorFile','rb')
        PTL_Injector=dill.load(file)
        # print('done making lattice')

        test=LatticeOptimizer(PTL_Ring,PTL_Injector)
        test.generate_Swarms()  #33 % of time here`
        sol=test.optimize_Magnetic_Field((0,3),fieldBounds,20,maxIter=20,parallel=parallel)
        sol.xRing_L=X
        sol.description='Pseudorandom search'
        return sol
    solve_For_Lattice_Params([.01,.01,.01])
    #test speed
    t=time.time()
    solve_For_Lattice_Params([0.0080519553439631, 0.013936147854230443, 0.005370946829484897],parallel=True)
    print(time.time()-t)
    exit()

    paramBounds=[(.005,.03),(.005,.03),(.005,.025)]
    np.random.seed(42)
    initialSampleCoords=skopt.sampler.Sobol().generate(paramBounds,num)
    t=time.time()
    from ParaWell import ParaWell
    helper=ParaWell()
    solutionList=helper.parallel_Problem(solve_For_Lattice_Params,initialSampleCoords,onlyReturnResults=True,
                                         numWorkers=31)
    initialCostValues=[LatticeOptimizer.cost_Function(sol.survival) for sol in solutionList]
    print('Finished random search')

    def wrapper(X):
        print('inside wrapper')
        sol=solve_For_Lattice_Params(X,parallel=True)
        sol.description='GP Search'
        solutionList.append(sol)
        cost=LatticeOptimizer.cost_Function(sol.survival)
        return cost

    skoptSol=skopt.gp_minimize(wrapper,paramBounds,n_calls=16,n_initial_points=0,x0=initialSampleCoords,
                               y0=initialCostValues,model_queue_size=None ,n_jobs=-1,acq_optimizer='lbfgs',
                               n_points=100000 ,n_restarts_optimizer=32*10,noise=1e-6)

    argMax=np.argmax(np.asarray([sol.survival for sol in solutionList]))
    solutionOptimal=solutionList[argMax]
    print(solutionOptimal)
    print("total time: ",time.time()-t)  #original is 15600
    emailText=solutionOptimal.__str__()+'\n'
    emailText+='Run time: '+str(int(time.time()-t))+' s \n'
    send_MySelf_Email(emailText)

    file=open('solutionList1','wb')
    dill.dump(solutionList,file)
    plot_objective(skoptSol)
    plt.show()



num=32*2
fieldBounds=[(0.0,1.0),(0.0,1.0)]
singleLayerStrength=1.0
solve_System(fieldBounds,num,singleLayerStrength)



'''
vT=10.0, single layer

----------Solution-----------   
injector element spacing optimum configuration: [0.20926595 0.13467041]
 storage ring magnetic field optimum configuration: [0.17241379 0.31034483]
 storage ring spatial optimal optimum configuration: [0.01115234375, 0.01134765625, 0.0067220703125]
 optimum result: 39.47804503987378
----------------------------




'''