from shimOptimizationOfLens_Focus import ShimOptimizer
import numpy as np
def test():
    tol = 1e-10
    rp = .05
    L0 = .23
    magnetWidth = .0254
    lensBounds = {'length': (L0 - rp, L0)}
    lensParams = {'rp': rp, 'width': magnetWidth}
    lensBaseLineParams = {'rp': rp, 'width': magnetWidth, 'length': L0}

    shimAParamBounds = {'r': (rp, rp + magnetWidth), 'phi': (0.0, np.pi / 6), 'deltaZ': (0.0, rp),
                        'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shimALockedParams = {'radius': .0254 / 2, 'planeSymmetry': False, 'location': 'top'}

    shimBParamBounds = {'r': (rp, rp + magnetWidth), 'phi': (0.0, np.pi / 6), 'deltaZ': (0.0, rp),
                        'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shimBLockedParams = {'radius': .0254 / 2, 'planeSymmetry': False, 'location': 'bottom'}

    shimCParamBounds = {'r': (rp, rp + magnetWidth), 'phi': (0.0, np.pi / 6), 'deltaZ': (0.0, rp),
                        'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shimCLockedParams = {'radius': .0254 / 2, 'planeSymmetry': True}

    shimOptimizerAB = ShimOptimizer()
    shimOptimizerAB.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizerAB.add_Shim(shimAParamBounds, shimALockedParams)
    shimOptimizerAB.add_Shim(shimBParamBounds, shimBLockedParams)
    shimOptimizerAB.initialize_Optimization()

    shimOptimizerC = ShimOptimizer()
    shimOptimizerC.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizerC.add_Shim(shimCParamBounds, shimCLockedParams)
    shimOptimizerC.initialize_Optimization()
    shimAParamBounds = shimOptimizerAB.shimsList[0].paramsBounds
    shimBParamBounds = shimOptimizerAB.shimsList[1].paramsBounds

    assert 'length' in shimOptimizerAB.boundsKeys
    for key, val in shimAParamBounds.items():
        assert key[-1]=='0'
        assert key in shimOptimizerAB.boundsKeys
    for key, val in shimBParamBounds.items():
        assert key[-1] == '1'
        assert key  in shimOptimizerAB.boundsKeys
    args = np.ones(len(shimOptimizerAB.boundsKeys)) * .75
    args[0] = L0
    DNA_List = shimOptimizerAB.make_DNA_List_From_Args(args)
    z1 = (L0 / 2 + .75)
    z2 = -(L0 / 2 + .75)
    assert abs(DNA_List[1]['z'] - z1) < tol and abs(DNA_List[2]['z'] - z2) < tol
    assert len(shimOptimizerAB.bounds)==len(args)

    r0, phi0, deltaz0, theta0, psi0 = rp, np.pi / 13, .02, np.pi / 3, np.pi / 7
    argsAB = [L0, r0, phi0, deltaz0, theta0, psi0, r0, phi0, deltaz0, np.pi - theta0, psi0]
    shimOptimizerAB.initialize_Baseline_Values(lensBaseLineParams)
    costAB = shimOptimizerAB.cost_Function(argsAB)

    argsC = [L0, r0, phi0, deltaz0, theta0, psi0]
    shimOptimizerC.initialize_Baseline_Values(lensBaseLineParams)
    costC = shimOptimizerC.cost_Function(argsC)

    costAB_0 =21.193482602161573
    costC_0 = 21.193482602176324
    print(costAB)
    print(costC)
    assert abs(costAB - costC) < tol
    assert abs(costAB - costAB_0) < tol and abs(costC - costC_0) < tol