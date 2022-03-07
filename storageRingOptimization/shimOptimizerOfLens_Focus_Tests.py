from shimAndGeneticLensOptimization import ShimOptimizer
import numpy as np
import multiprocess as mp
def test1():
    tol = 1e-9
    rp = .05
    L0 = .23
    sphereDiam=.0254
    magnetWidth = .0254
    lensBounds = [{'length': (L0 - rp, L0)}]
    lensParams = [{'rp': rp, 'width': magnetWidth}]
    lensBaseLineParams = [{'rp': rp, 'width': magnetWidth, 'length': L0}]

    shimAParamBounds = {'r': (rp, rp + magnetWidth), 'phi': (0.0, np.pi / 6), 'deltaZ': (0.0, rp),
                        'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shimALockedParams = {'diameter': sphereDiam,'shape':'sphere', 'planeSymmetry': False, 'location': 'top'}

    shimBParamBounds = {'r': (rp, rp + magnetWidth), 'phi': (0.0, np.pi / 6), 'deltaZ': (0.0, rp),
                        'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shimBLockedParams = {'diameter': sphereDiam,'shape':'sphere', 'planeSymmetry': False, 'location': 'bottom'}

    shimCParamBounds = {'r': (rp, rp + magnetWidth), 'phi': (0.0, np.pi / 6), 'deltaZ': (0.0, rp),
                        'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shimCLockedParams = {'diameter': sphereDiam,'shape':'sphere', 'planeSymmetry': True}

    shimOptimizerAB = ShimOptimizer('full',False)
    shimOptimizerAB.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizerAB.add_Shim(shimAParamBounds, shimALockedParams)
    shimOptimizerAB.add_Shim(shimBParamBounds, shimBLockedParams)
    shimOptimizerAB.initialize_Optimization()

    shimOptimizerC = ShimOptimizer('full',False)
    shimOptimizerC.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizerC.add_Shim(shimCParamBounds, shimCLockedParams)
    shimOptimizerC.initialize_Optimization()
    shimAParamBounds = shimOptimizerAB.shimsList[0].paramsBounds
    shimBParamBounds = shimOptimizerAB.shimsList[1].paramsBounds
    assert 'length0' in shimOptimizerAB.boundsKeys
    for key, val in shimAParamBounds.items():
        assert key[-1]=='0'
        assert key in shimOptimizerAB.boundsKeys
    for key, val in shimBParamBounds.items():
        assert key[-1] == '1'
        assert key  in shimOptimizerAB.boundsKeys
    args = np.ones(len(shimOptimizerAB.boundsKeys)) * .75
    args[0] = L0
    DNA_List = shimOptimizerAB.make_DNA_List_From_Args(args)
    z1 = (L0 / 2 + .75+sphereDiam/2)
    z2 = -(L0 / 2 + .75+sphereDiam/2)
    assert abs(DNA_List[1]['z'] - z1) < tol and abs(DNA_List[2]['z'] - z2) < tol
    assert len(shimOptimizerAB.bounds)==len(args)

    r0, phi0, deltaz0, theta0, psi0 = rp, np.pi / 13, .02, np.pi / 3, np.pi / 7
    argsAB = [L0, r0, phi0, deltaz0, theta0, psi0, r0, phi0, deltaz0, np.pi - theta0, psi0]
    costAB = shimOptimizerAB.cost_Function(argsAB,True,True)

    argsC = [L0, r0, phi0, deltaz0, theta0, psi0]
    costC = shimOptimizerC.cost_Function(argsC,True,True)

    costAB_0 =3.8725999982697887
    costC_0 = 3.87259999828051
    assert abs(costAB - costC) < tol
    assert abs(costAB - costAB_0) < tol and abs(costC - costC_0) < tol #failed

def test2():
    #test that layer works as expected
    tol=1e-6
    lensLongitudinalSymmetry=True
    rp,magnetWidth,L0,numLayers=.05,.02,.35,6
    lensBounds = [{}]*(numLayers//2)
    lensParamsLocked = [{'width': magnetWidth, 'length': L0 / numLayers,'rp':rp} for _ in range(numLayers//2)]
    lensBaseLineParams = [{'rp': rp, 'width': magnetWidth, 'length': L0}]
    shimOptimizer = ShimOptimizer('full',lensLongitudinalSymmetry)
    shimOptimizer.set_Lens(lensBounds, lensParamsLocked, lensBaseLineParams)
    emptyArgs=[]
    results,cost=shimOptimizer.characterize_Results(emptyArgs,display=False)
    I0=112.76966030069069
    m0=0.7871635545330284
    assert abs(results['I']-I0)<tol and abs(results['m']-m0)<tol
    assert abs(shimOptimizer.baseLineFocusDict['I']-I0)<tol and abs(shimOptimizer.baseLineFocusDict['m']-m0)<tol
def run_Tests(parallel=False):
    def run(func):
        func()
    funcList=[test1,test2]
    if parallel==True:
        with mp.Pool() as pool:
            pool.map(run,funcList)
    else:
        list(map(run,funcList))