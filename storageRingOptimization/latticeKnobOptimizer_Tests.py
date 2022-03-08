import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from asyncDE import solve_Async
import numpy as np
import random

from latticeKnobOptimizer import LatticeOptimizer,Solution
from ParticleTracerLatticeClass import ParticleTracerLattice

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

def generate_Ring_Lattice(rpBend,rpLens,rpLensFirst,Lm,LLens,parallel=False)->ParticleTracerLattice:
    tunableDriftGap=2.54e-2
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
    PTL_Ring.add_Combiner_Sim_Lens(.2, .02)
    PTL_Ring.add_Halbach_Lens_Sim(rpLensFirst,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,LLens)
    PTL_Ring.add_Drift(tunableDriftGap/2)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    # PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
    # PTL_Ring.add_Drift(probeSpace)
    PTL_Ring.add_Halbach_Lens_Sim(rpLens,None,constrain=True)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented(Lm,rpBend,None,1.0,rOffsetFact=rOffsetFact)
    PTL_Ring.end_Lattice(enforceClosedLattice=True,constrain=True)  # 17.8 % of time here
    return PTL_Ring


def generate_Injector_Lattice(injectorFactor,rpInjectorFactor,parallel=False)->ParticleTracerLattice:
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
    # PTL_Injector.add_Combiner_Sim('combinerV3.txt')
    PTL_Injector.add_Combiner_Sim_Lens(.2,.02)
    PTL_Injector.end_Lattice(constrain=False,enforceClosedLattice=False)
    return PTL_Injector

def solve_For_Lattice_Params(X,parallel=False):
    assert len(X)==7

    rpBend,rpLens,rpLensFirst,Lm,LLens,injectorFactor,rpInjectorFactor=X

    PTL_Ring=generate_Ring_Lattice(rpBend,rpLens,rpLensFirst,Lm,LLens,parallel=parallel)
    if PTL_Ring is None:
        print('invalid ring')
        sol=invalid_Solution(X,invalidRing=True)
        return sol
    PTL_Injector=generate_Injector_Lattice(injectorFactor,rpInjectorFactor,parallel=parallel)
    if PTL_Injector is None:
        print('invalid injector')
        sol=invalid_Solution(X,invalidInjector=True)
        return sol
    optimizer=LatticeOptimizer(PTL_Ring,PTL_Injector)
    sol=optimizer.optimize((1,7),parallel=parallel)
    sol.xRing_TunedParams1=X
    sol.description='Pseudorandom search'
    print(sol)
    return sol


def TEST_Optimizer():
    '''
    ----------Solution-----------
    injector element spacing optimum configuration: [0.36331701 0.27829773]
     storage ring tuned params 1 optimum configuration: [0.015231344258015412, 0.009442441351438617, 0.028987667914122765, 0.012143163429838986, 0.2351624140159203, 0.5704592180159943, 0.8199484178151447]
     storage ring tuned params 2 optimum configuration: [0.53736145 0.30125161]
     cost: 0.9978018361866404
    survival: 0.2198163813359577
    '''
    np.random.seed(42)
    random.seed(42)
    X=[0.015231344258015412, 0.009442441351438617, 0.028987667914122765, 0.012143163429838986, 0.2351624140159203, 0.5704592180159943, 0.8199484178151447]
    rpBend, rpLens, rpLensFirst, Lm, LLens, injectorFactor, rpInjectorFactor = X
    parallel=True
    PTL_Ring = generate_Ring_Lattice(rpBend, rpLens, rpLensFirst, Lm, LLens, parallel=parallel)
    if PTL_Ring is None:
        print('invalid ring')
        sol = invalid_Solution(X, invalidRing=True)
        return sol
    PTL_Injector = generate_Injector_Lattice(injectorFactor, rpInjectorFactor, parallel=parallel)
    if PTL_Injector is None:
        print('invalid injector')
        sol = invalid_Solution(X, invalidInjector=True)
        return sol
    parallel=False
    optimizer = LatticeOptimizer(PTL_Ring, PTL_Injector)
    sol = optimizer.optimize((1, 7), parallel=parallel)
    sol.xRing_TunedParams1=X
    print(sol)
    eps=1e-12
    survival=0.2198163813359577
    cost=0.9978018361866404
    xInjector=np.asarray([0.3633170063985472, 0.27829773239844785])
    xRing=np.asarray([0.5373614548265913 ,0.3012516118361091])
    assert abs(cost-sol.cost)<eps
    assert abs(survival-sol.survival)<eps
    assert np.linalg.norm(xRing-sol.xRing_TunedParams2)<eps
    assert np.linalg.norm(xInjector-sol.xInjector_TunedParams)<eps

def main():
    TEST_Optimizer()
if __name__=='__main__':
    main()