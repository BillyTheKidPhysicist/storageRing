import os

os.environ['OPENBLAS_NUM_THREADS']='1'
import skopt
import numpy as np
from ParticleClass import Particle
from ParticleTracerClass import ParticleTracer
from skopt.plots import plot_objective,plot_objective_2D
from OptimizerClassVersion2 import LatticeOptimizer,Solution
import dill
from ParticleTracerLatticeClass import ParticleTracerLattice
import time
import scipy.interpolate as spi
from sendMyselfEmail import send_MySelf_Email
import multiprocess
import matplotlib.pyplot as plt

'''
This models the storage ring as a system of only permanent magnets, with two permanent magnets being moved longitudinally
for tunability
'''
def mine_In_Parallel(function,argumentsList):
    #need to circumvent the fact that numba has a bug wherein compiled solutions persists in memory by limiting
    ###tasks per child
    multiprocess.set_start_method('spawn')
    pool = multiprocess.Pool(processes=31,maxtasksperchild=1)
    solutionList=pool.map(function,argumentsList,chunksize=1) #8.01
    pool.close()
    file=open('solutionList_Random_Ver2','wb')  #dump to save right now
    dill.dump(solutionList,file)
    file.close()
    print('Finished random search')
    exit()
    return solutionList
def invalid_Solution(XLattice):
    assert len(XLattice)>2, "must be lattice paramters"
    sol=Solution()
    sol.xRing_TunedParams1=XLattice
    sol.survival=0.0
    sol.cost=1.0
    sol.description='Pseudorandom search'
    print(sol)
    return sol
def is_Valid_Injector(X):
    injectorFactor,rpInjectorFactor=X
    LInjector=injectorFactor*.15
    rpInjector=rpInjectorFactor*.02
    BpLens=.7
    injectorLensPhase=np.sqrt((2*800.0/200**2)*BpLens/rpInjector**2)*LInjector
    if injectorLensPhase>np.pi:
        print('bad lens phase')
        return False
    else:
        return True
def generate_Ring_Lattice(X,parallel=False):
    combinerGap=5e-2
    tunableDriftGap=2.54e-2
    probeSpace=.0254
    jeremyMagnetAp=.01
    assert len(X)==3
    rpBend,Lm,LLens=X
    rpLens=rpBend
    rpLensFirst=rpLens

    PTL_Ring=ParticleTracerLattice(200.0,latticeType='storageRing',parallel=parallel)
    rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm,
                                                    parallel=parallel)  #25% of time here, 1.0138513851385138
    if rOffsetFact is None:
        return None
    PTL_Ring.add_Drift(tunableDriftGap/2,ap=rpLens)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2,ap=rpLens)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
    PTL_Ring.add_Combiner_Sim('combinerV3.txt')
    # PTL_Ring.add_Drift(combinerGap,ap=jeremyMagnetAp)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensFirst,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    # PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
    # PTL_Ring.add_Drift(probeSpace)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented_With_End_Cap(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  #17.8 % of time here
    return PTL_Ring
def generate_Injector_Lattice(X,parallel=False):
    assert len(X)==2
    injectorFactor,rpInjectorFactor=X
    LInjector=injectorFactor*.15
    rpInjector=rpInjectorFactor*.02
    PTL_Injector=ParticleTracerLattice(200.0,latticeType='injector',parallel=parallel)
    PTL_Injector.add_Drift(.1,ap=.025)
    PTL_Injector.add_Halbach_Lens_Sim(rpInjector,LInjector,apFrac=.9)
    PTL_Injector.add_Drift(.2,ap=.01)
    PTL_Injector.add_Combiner_Sim('combinerV3.txt')
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)
    return PTL_Injector
def solve_System(numInitial):
    def solve_For_Lattice_Params(X,parallel=False):
        t=time.time()
        assert len(X)==5
        rpBend,Lm,LLens,injectorFactor,rpInjectorFactor=X
        XInjector=injectorFactor,rpInjectorFactor
        XRing=rpBend,Lm,LLens
        if is_Valid_Injector(XInjector)==False:
            # print('invalid injector:',int(time.time()-t))
            return invalid_Solution(X)
        PTL_Ring=generate_Ring_Lattice(XRing,parallel=parallel)
        if PTL_Ring is None:
            print('invalid ring',int(time.time()-t))
            return invalid_Solution(X)
        PTL_Injector=generate_Injector_Lattice(XInjector,parallel=parallel)
        optimizer=LatticeOptimizer(PTL_Ring,PTL_Injector)
        sol=optimizer.optimize((1,7),parallel=parallel)
        sol.xRing_TunedParams1=X
        sol.description='Pseudorandom search'
        print(sol)
        print(time.time()-t)
        return sol
    t=time.time()
    paramBounds=[(.005,.03),(.01,.03),(.1,.3),(.75,1.25),(.75,1.25)]
    randomSearchSampleCoords=skopt.sampler.Sobol().generate(paramBounds,numInitial)
    solve_For_Lattice_Params([0.010078125, 0.0171875, 0.228125, 0.8046875, 0.9296875],parallel=True)
    exit()
    np.random.seed(42)
    # solutionList=mine_In_Parallel(solve_For_Lattice_Params,randomSearchSampleCoords)
    # with open('solutionList_Random_Ver2','rb') as file:
    #     solutionList=dill.load(file)
    initialCostValues=[sol.cost for sol in solutionList]
    initialSampleCoords=[sol.xRing_TunedParams1 for sol in solutionList]
    def wrapper(X):
        sol=solve_For_Lattice_Params(X,parallel=True)
        sol.description='GP Search'
        solutionList.append(sol)
        return sol.cost
    skoptSol=skopt.gp_minimize(wrapper,paramBounds,n_calls=32,n_initial_points=0,x0=initialSampleCoords,
                               y0=initialCostValues ,n_jobs=-1,acq_optimizer='lbfgs',
                               n_points=100000 ,n_restarts_optimizer=32*10,noise=1e-6)

    argMax=np.argmax(np.asarray([sol.survival for sol in solutionList]))
    solutionOptimal=solutionList[argMax]
    print(solutionOptimal)
    print("total time: ",time.time()-t)  #original is 15600
    # emailText=solutionOptimal.__str__()+'\n'
    # emailText+='Run time: '+str(int(time.time()-t))+' s \n'
    # send_MySelf_Email(emailText)

    file=open('solutionList_Ver2','wb')
    dill.dump(solutionList,file)
    plot_objective(skoptSol)
    plt.show()


