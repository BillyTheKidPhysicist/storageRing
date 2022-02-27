import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import time
from shimOptimizationOfLens_Focus import ShimOptimizer
from profilehooks import profile

rp = .05
L0 = .23
magnetWidth = .0254
lensBounds = {'length': (L0*.25 - rp, L0+rp*.25)}
lensParams = {'rp': rp, 'width': magnetWidth}
lensBaseLineParams = {'rp': rp, 'width': magnetWidth, 'length': L0}


def make_Shim_Params(variableRadius,variablePhi,symmetric=True,location=None):
    shim1ParamBounds = {'r': (rp, rp + magnetWidth), 'deltaZ': (0.0, rp),
                        'psi': (0.0, 2 * np.pi)}
    shim1LockedParams = {'planeSymmetry': symmetric,'theta':np.pi/2}
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
    deltaL=-.01
    lensBaseLineParams['length']-=deltaL
    shimParamBounds, shimLockedParams=make_Shim_Params(variableRadius,variablePhi)
    shimOptimizer = ShimOptimizer()
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimParamBounds, shimLockedParams)
    # shimOptimizer.optimize(saveData=saveData)
    # args0=np.array([0.22232782 -deltaL,0.05411236 ,0.01467107 ,1.69968186 ,5.40909307])
    # print(shimOptimizer.optimize1())
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
    if variablePhi==True:
        variablePhi=[True,True]
    else:
        assert len(variablePhi)==2 and isinstance(variablePhi,list)
    shimTopParamBounds, shimTopLockedParams = make_Shim_Params(variableRadius, variablePhi.pop(0))
    shimBotParamBounds, shimBotLockedParams = make_Shim_Params(variableRadius, variablePhi.pop(0))
    shimOptimizer = ShimOptimizer()
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimTopParamBounds, shimTopLockedParams)
    shimOptimizer.add_Shim(shimBotParamBounds, shimBotLockedParams)
    print(shimOptimizer.optimize())
    # args=np.array([0.2425    , 0.06432957, 0.02027348, 2.85786667, 0.0195204 ,
    #    0.06495041, 0.05      , 5.34821609, 0.0254    ])
    # print(shimOptimizer.characterize_Results(args))
# optimize_1Shim_Symmetric(.0254*3/8,np.pi/6.0)#,saveData='run1_1Shim')

optimize_2Shim_Symmetric(True,[0.0,np.pi/6])#,saveData='run1_2Shim')


