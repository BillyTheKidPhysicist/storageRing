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
    print('processes restart')
def mine_Solutions_Work_Around(function,argumentsList):
    #need to circumvent the fact that numba has a bug wherein compiled solutions persists in memory
    maxBatchSizeForMemory=32*5
    numBatches=int(ceil(len(argumentsList)/maxBatchSizeForMemory))
    solutionList=[]
    print(numBatches)
    for i in range(numBatches):
        batchArgs=argumentsList[i*maxBatchSizeForMemory:(i+1)*maxBatchSizeForMemory]
        print(len(batchArgs))
        tempSolutionList=Parallel(n_jobs=-1,batch_size=1)(delayed(function)(arg) for arg in batchArgs)
        for result in tempSolutionList: assert isinstance(result,Solution)
        solutionList.extend(tempSolutionList)
        shut_Down_Joblib_Process_To_Save_Memory()
    return solutionList

def reject_Invalid_Injector(X):
    injectorFactor,rpInjectorFactor=X
    LInjector=.15/injectorFactor
    rpInjector=rpInjectorFactor*.02/injectorFactor
    BpLens=.7
    injectorLensPhase=np.sqrt((2*800.0/200**2)*BpLens/rpInjector**2)*LInjector
    if injectorLensPhase>np.pi:
        print('bad lens phase')
        sol=Solution()
        sol.xRing_TunedParams1=X
        sol.survival=0.0
        sol.description='Pseudorandom search'
        return sol
def generate_Ring_Lattice(X,parallel=False):
    combinerGap=5e-2
    tunableDriftGap=5e-2
    probeSpace=.0254
    rpLens,Lm,LLens=X
    rpBend=rpLens
    rpLensFirst=rpLens

    PTL_Ring=ParticleTracerLattice(200.0,latticeType='storageRing',parallel=parallel)
    rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm,
                                                    parallel=parallel)  #25% of time here, 1.0138513851385138
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
    return PTL_Ring
def generate_Injector_Lattice(X,parallel=False):
    rpInjector,LInjector=X
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
    return PTL_Injector


def solve_System(spacingBounds,numInitial):
    def solve_For_Lattice_Params(X,parallel=False):
        t=time.time()
        rpLens,Lm,LLens,injectorFactor,rpInjectorFactor=X
        XInjector=injectorFactor,rpInjectorFactor
        XRing=rpLens,Lm,LLens
        reject_Invalid_Injector(XInjector)
        PTL_Ring=generate_Ring_Lattice(XRing,parallel=parallel)
        PTL_Injector=generate_Injector_Lattice(XInjector,parallel=parallel)

        test=LatticeOptimizer(PTL_Ring,PTL_Injector)
        sol=test.optimize_Magnetic_Field((1,8),spacingBounds,10,'spacing',maxIter=20,parallel=parallel)
        sol.xRing_TunedParams1=X
        sol.description='Pseudorandom search'
        print(sol)
        print(time.time()-t)
        return sol
    paramBounds=[(.005,.03),(.005,.025),(.1,.3),(.5,1.5),(.75,1.25)]
    initialSampleCoords=skopt.sampler.Sobol().generate(paramBounds,numInitial)
    np.random.seed(42)
    t=time.time()
    solutionList=mine_Solutions_Work_Around(solve_For_Lattice_Params,initialSampleCoords)
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


def run(numInitial,spacingBounds):
    solve_System(spacingBounds,numInitial)