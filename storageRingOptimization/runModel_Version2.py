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
from joblib import Parallel,delayed
from math import ceil
from profilehooks import profile
import matplotlib.pyplot as plt

'''
This models the storage ring as a system of only permanent magnets, with two permanent magnets being moved longitudinally
for tunability
'''
def shut_Down_Joblib_Process_To_Save_Memory():
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
def mine_Solutions_Work_Around(function,bounds,num):
    #need to circumvent the fact that numba has a bug wherein compiled solutions persists in memory
    maxBatchSizeForMemory=32*3
    numBatches=int(ceil(num/maxBatchSizeForMemory))
    initialSampleCoords=skopt.sampler.Sobol().generate(bounds,num)
    solutionList=[]
    for _ in range(numBatches):
        tempSolutionList=Parallel(n_jobs=-1)(delayed(function)(coord) for coord in initialSampleCoords)
        for result in tempSolutionList: assert isinstance(result,Solution)
        solutionList.extend(tempSolutionList)
        shut_Down_Joblib_Process_To_Save_Memory()
    return solutionList

def solve_System(spacingBounds,num,benderMagnetStrength):
    def solve_For_Lattice_Params(X,parallel=False):
        combinerGap=5e-2
        tunableDriftGap=5e-2
        probeSpace=.0254
        rpLens,Lm,LLens,rpInjector,LInjector=X
        # LInjector=.15/injectorFactor
        # rpInjector=.02/injectorFactor
        rpBend=rpLens
        rpLensFirst=rpLens
        BpLens=.7
        injectorLensPhase=np.sqrt((2*800.0/200**2)*BpLens/rpInjector**2)*LInjector
        if injectorLensPhase>np.pi:
            print('bad lens phase')
            sol=Solution()
            sol.xRing_TunedParams1=X
            sol.survival=0.0
            sol.description='Pseudorandom search'
            return sol
        PTL_Ring=ParticleTracerLattice(200.0,latticeType='storageRing',parallel=parallel)
        rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm,parallel=parallel)  #25% of time here, 1.0138513851385138
        PTL_Ring.add_Drift(tunableDriftGap/2)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
        PTL_Ring.add_Drift(tunableDriftGap/2)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
        PTL_Ring.add_Combiner_Sim('combinerV3.txt')
        PTL_Ring.add_Drift(combinerGap)
        PTL_Ring.add_Halbach_Lens_Sim(rpLensFirst,LLens)
        PTL_Ring.add_Drift(tunableDriftGap/2)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
        PTL_Ring.add_Drift(tunableDriftGap/2)
        PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
        PTL_Ring.add_Drift(probeSpace)
        PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
        PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
        PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  #17.8 % of time here

        PTL_Injector=ParticleTracerLattice(200.0,latticeType='injector',parallel=parallel)
        PTL_Injector.add_Drift(.1,ap=.025)
        PTL_Injector.add_Halbach_Lens_Sim(rpInjector,LInjector,apFrac=.9)
        PTL_Injector.add_Drift(.2,ap=.01)
        PTL_Injector.add_Combiner_Sim('combinerV3.txt')
        PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)

        # file=open('injectorFile','wb')
        # dill.dump(PTL_Injector,file)
        # file=open('injectorFile','rb')
        # PTL_Injector=dill.load(file)
        # print('done making lattice')
        test=LatticeOptimizer(PTL_Ring,PTL_Injector)
        sol=test.optimize_Magnetic_Field((1,8),spacingBounds,20,'spacing',maxIter=20,parallel=parallel)
        sol.xRing_TunedParams1=X
        sol.description='Pseudorandom search'
        print(sol)
        return sol
    paramBounds=[(.005,.03),(.005,.025),(.1,.3),(.01,.03),(.1,.3)]
    np.random.seed(42)
    initialSampleCoords=skopt.sampler.Sobol().generate(paramBounds,num)
    t=time.time()

    solutionList=Parallel(n_jobs=-1)(delayed(solve_For_Lattice_Params)(coords) for coords in initialSampleCoords)
    initialCostValues=[LatticeOptimizer.cost_Function(sol.survival) for sol in solutionList]
    print('Finished random search')

    def wrapper(X):
        sol=solve_For_Lattice_Params(X,parallel=True)
        sol.description='GP Search'
        solutionList.append(sol)
        cost=LatticeOptimizer.cost_Function(sol.survival)
        return cost

    skoptSol=skopt.gp_minimize(wrapper,paramBounds,n_calls=32,n_initial_points=0,x0=initialSampleCoords,
                               y0=initialCostValues,model_queue_size=None ,n_jobs=-1,acq_optimizer='lbfgs',
                               n_points=100000 ,n_restarts_optimizer=32*10,noise=1e-6)

    argMax=np.argmax(np.asarray([sol.survival for sol in solutionList]))
    solutionOptimal=solutionList[argMax]
    print(solutionOptimal)
    print("total time: ",time.time()-t)  #original is 15600
    emailText=solutionOptimal.__str__()+'\n'
    emailText+='Run time: '+str(int(time.time()-t))+' s \n'
    send_MySelf_Email(emailText)

    file=open('solutionList_Spacing_LessVariable','wb')
    dill.dump(solutionList,file)
    plot_objective(skoptSol)
    plt.show()



num=32*2
spacingBounds=[(0.2,.8),(0.2,.8)]
singleLayerStrength=1.0
solve_System(spacingBounds,num,singleLayerStrength)