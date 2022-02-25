import numpy as np
import time
from shimOptimizationOfLens_Focus import ShimOptimizer
from profilehooks import profile

rp = .05
L0 = .23
magnetWidth = .0254
lensBounds = {'length': (L0 - rp, L0)}
lensParams = {'rp': rp, 'width': magnetWidth}
lensBaseLineParams = {'rp': rp, 'width': magnetWidth, 'length': L0}


def make_Shim_Params(variableRadius,variablePhi,symmetric=True,location=None):
    shim1ParamBounds = {'r': (rp, rp + magnetWidth), 'deltaZ': (0.0, rp), 'theta': (0.0, np.pi),
                        'psi': (0.0, 2 * np.pi)}
    shim1LockedParams = {'planeSymmetry': symmetric}
    if symmetric==False:
        shim1LockedParams['location']=location
    if variableRadius == True:
        shim1ParamBounds['radius'] = (.0254 / 8, .0254)
    else:
        assert isinstance(variableRadius, float)
        shim1LockedParams['radius'] = variableRadius
    if variablePhi == True:
        shim1ParamBounds['phi'] = (0, np.pi / 6)
    else:
        assert isinstance(variablePhi, float)
        shim1LockedParams['phi'] = variablePhi
    return shim1ParamBounds, shim1LockedParams

def optimize_1Shim_Symmetric(variableRadius,variablePhi,saveData=None):
    np.random.seed(int(time.time()))
    shimParamBounds, shimLockedParams=make_Shim_Params(variableRadius,variablePhi)
    shimOptimizer = ShimOptimizer()
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimParamBounds, shimLockedParams)
    shimOptimizer.optimize(saveData=saveData)
    # shimOptimizer.characterize_Results(args0)

def optimize_2Shim_NonSymmetric(variableRadius,variablePhi,saveData=None):
    np.random.seed(int(time.time()))
    shimTopParamBounds, shimTopLockedParams = make_Shim_Params(variableRadius, variablePhi,symmetric=False,
                                                               location='top')
    shimBotParamBounds, shimBotLockedParams = make_Shim_Params(variableRadius, variablePhi,symmetric=False,
                                                               location='bottom')
    shimOptimizer = ShimOptimizer()
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimTopParamBounds, shimTopLockedParams)
    shimOptimizer.add_Shim(shimBotParamBounds, shimBotLockedParams)
    shimOptimizer.optimize(saveData=saveData)

def optimize_2Shim_Symmetric(variableRadius,variablePhi,saveData=None):
    np.random.seed(int(time.time()))
    shimTopParamBounds, shimTopLockedParams = make_Shim_Params(variableRadius, variablePhi)
    shimBotParamBounds, shimBotLockedParams = make_Shim_Params(variableRadius, variablePhi)
    shimOptimizer = ShimOptimizer()
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimTopParamBounds, shimTopLockedParams)
    shimOptimizer.add_Shim(shimBotParamBounds, shimBotLockedParams)
    shimOptimizer.optimize(saveData=saveData)



optimize_1Shim_Symmetric(True,True,saveData='run1_1Shim')
optimize_1Shim_Symmetric(True,True,saveData='run2_1Shim')
optimize_1Shim_Symmetric(True,True,saveData='run3_1Shim')

optimize_2Shim_NonSymmetric(True,True,saveData='run1_2Shim')
optimize_2Shim_NonSymmetric(True,True,saveData='run2_2Shim')
optimize_2Shim_NonSymmetric(True,True,saveData='run3_2Shim')