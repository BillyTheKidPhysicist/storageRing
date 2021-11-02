import multiprocessing
import os
import time
import joblib

import skopt
import numpy as np
from ParticleClass import Particle
from ParticleTracerClass import ParticleTracer
from skopt.plots import plot_objective,plot_objective_2D
from OptimizerClass_Version2 import LatticeOptimizer,Solution
import dill
from ParticleTracerLatticeClass import ParticleTracerLattice
import time
import scipy.interpolate as spi
# from sendMyselfEmail import send_MySelf_Email
import multiprocess
import matplotlib.pyplot as plt

'''
This models the storage ring as a system of only permanent magnets, with two permanent magnets being moved longitudinally
for tunability
'''


# def mine_In_Parallel(function,argumentsList):
#     #need to circumvent the fact that numba has a bug wherein compiled solutions persists in memory by limiting
#     ###tasks per child
#     t=time.time()
#     multiprocess.set_start_method('spawn')
#     pool = multiprocess.Pool(processes=31,maxtasksperchild=1)
#     solutionList=pool.map(function,argumentsList,chunksize=1) #8.01
#     pool.close()
#     file=open('solutionList_Random_Ver2','wb')  #dump to save right now
#     dill.dump(solutionList,file)
#     file.close()
#     print('Finished random search. Total time: ',time.time()-t)
#     multiprocess.set_start_method('fork',force=True)
#     return solutionList
def invalid_Solution(XLattice,invalidInjector=None,invalidRing=None):
    assert len(XLattice)==7,"must be lattice paramters"
    sol=Solution()
    sol.xRing_TunedParams1=XLattice
    sol.survival=0.0
    sol.cost=1.0
    sol.description='Pseudorandom search'
    sol.invalidRing=invalidRing
    sol.invalidInjector=invalidInjector
    return sol


def is_Valid_Injector_Phase(injectorFactor,rpInjectorFactor):
    LInjector=injectorFactor*.15
    rpInjector=rpInjectorFactor*.02
    BpLens=.7
    injectorLensPhase=np.sqrt((2*800.0/200**2)*BpLens/rpInjector**2)*LInjector
    if np.pi<injectorLensPhase or injectorLensPhase<np.pi/10:
        print('bad lens phase')
        return False
    else:
        return True


def generate_Ring_Lattice(rpBend,rpLens,rpLensFirst,Lm,LLens,parallel=False):
    combinerGap=5e-2
    tunableDriftGap=2.54e-2
    probeSpace=.0254
    jeremyMagnetAp=.01
    assert type(parallel)==bool
    fringeFrac=1.5
    if LLens-2*rpLens*fringeFrac<0 or LLens-2*rpLensFirst*fringeFrac<0:  # minimum fringe length must be respected
        return None
    PTL_Ring=ParticleTracerLattice(200.0,latticeType='storageRing',parallel=parallel)
    rOffsetFact=PTL_Ring.find_Optimal_Offset_Factor(rpBend,1.0,Lm,
                                                    parallel=parallel)  # 25% of time here, 1.0138513851385138
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
    PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  # 17.8 % of time here
    return PTL_Ring


def generate_Injector_Lattice(injectorFactor,rpInjectorFactor,parallel=False):
    assert type(parallel)==bool
    if is_Valid_Injector_Phase(injectorFactor,rpInjectorFactor)==False:
        return None
    LInjector=injectorFactor*.15
    rpInjector=rpInjectorFactor*.02
    fringeFrac=1.5
    LMagnet=LInjector-2*fringeFrac*rpInjector
    if LMagnet<1e-9:  # minimum fringe length must be respected.
        return None
    PTL_Injector=ParticleTracerLattice(200.0,latticeType='injector',parallel=parallel)
    PTL_Injector.add_Drift(.1,ap=.025)
    try:
        PTL_Injector.add_Halbach_Lens_Sim(rpInjector,LInjector,apFrac=.9)
    except:
        print(LInjector,rpInjector)
    PTL_Injector.add_Drift(.2,ap=.01)
    PTL_Injector.add_Combiner_Sim('combinerV3.txt')
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)
    return PTL_Injector


def solve_For_Lattice_Params(X,parallel=False):
    assert len(X)==7
    t=time.time()
    rpBend,rpLens,rpLensFirst,Lm,LLens,injectorFactor,rpInjectorFactor=X

    PTL_Ring=generate_Ring_Lattice(rpBend,rpLens,rpLensFirst,Lm,LLens,parallel=parallel)
    if PTL_Ring is None:
        # print('invalid ring',int(time.time()-t))
        sol=invalid_Solution(X,invalidRing=True)
        # print(sol)
        return sol

    PTL_Injector=generate_Injector_Lattice(injectorFactor,rpInjectorFactor,parallel=parallel)
    if PTL_Injector is None:
        # print('invalid injector',int(time.time()-t))
        sol=invalid_Solution(X,invalidInjector=True)
        # print(sol)
        return sol

    optimizer=LatticeOptimizer(PTL_Ring,PTL_Injector)
    sol=optimizer.optimize((1,7),parallel=parallel)
    sol.xRing_TunedParams1=X
    sol.description='Pseudorandom search'
    if sol.cost<1.0:
        print(sol)
    return sol

'''
to test


----------Solution-----------   
injector element spacing optimum configuration: [0.12038327 0.28020839]
 storage ring tuned params 1 optimum configuration: [0.01367993 0.03       0.03628571 0.01369232 0.39071281 0.36664041
 0.33992824]
 storage ring tuned params 2 optimum configuration: [0.26215281 0.28339918]
 cost: 0.9969868214709979
survival: 0.30131785290020696

----------Solution-----------   
injector element spacing optimum configuration: nan
 storage ring tuned params 1 optimum configuration: [0.02133876 0.01664931 0.0373568  0.01       0.27308962 0.36114502
 0.45973645]
 storage ring tuned params 2 optimum configuration: nan
 cost: 1.0
survival: 0.0

invalid ring 92
----------Solution-----------   
injector element spacing optimum configuration: nan
 storage ring tuned params 1 optimum configuration: [0.0272293  0.01338131 0.03551544 0.03       0.32444589 0.72310314
 0.44757935]
 storage ring tuned params 2 optimum configuration: nan
 cost: 1.0
survival: 0.0

stable
invalid ring 82
----------Solution-----------   
injector element spacing optimum configuration: nan
 storage ring tuned params 1 optimum configuration: [0.02541893 0.01318074 0.04       0.03       0.3675667  0.6862384
 0.53515015]
 storage ring tuned params 2 optimum configuration: nan
 cost: 1.0
survival: 0.0


invalid injector 558
----------Solution-----------   
injector element spacing optimum configuration: nan
 storage ring tuned params 1 optimum configuration: [0.00795518 0.02129294 0.02987362 0.03       0.34199017 0.65885469
 0.25      ]
 storage ring tuned params 2 optimum configuration: nan
 cost: 1.0
survival: 0.0




'''