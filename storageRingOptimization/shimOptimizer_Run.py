import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import time
from shimOptimizationOfLens_Focus import ShimOptimizer
from profilehooks import profile

rp = .05
L0 = .23
magnetWidth = .0254
lensBounds = {'length': (L0- rp, L0+rp)}
lensParams = {'rp': rp, 'width': magnetWidth}
lensBaseLineParams = {'rp': rp, 'width': magnetWidth, 'length': L0}


def make_Shim_Params(variableRadius,variablePhi,symmetric=True,location=None):
    shim1ParamBounds = {'r': (rp, rp + magnetWidth), 'deltaZ': (0.0, rp),
                        'psi': (0.0, 2 * np.pi),'theta':(0.0,np.pi)}
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
    method='concentric'
    np.random.seed(int(time.time()))
    deltaL=-.01
    lensBaseLineParams['length']-=deltaL
    shimParamBounds, shimLockedParams=make_Shim_Params(variableRadius,variablePhi)
    shimOptimizer = ShimOptimizer(method)
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimParamBounds, shimLockedParams)
    shimOptimizer.optimize(saveData=saveData)
    # args0=np.array([0.21273044, 0.0754    , 0.04451173, 3.85420667, 1.55787181,
    #    0.0254    , 0.39590319])
    # shimOptimizer.initialize_Optimization()
    # shimOptimizer.cost_Function(args0,True,False)
    # print(shimOptimizer.optimize_Descent())
    # shimOptimizer.characterize_Results(args0)
def optimize_2Shim_NonSymmetric(variableRadius,variablePhi,saveData=None):
    raise Exception("does not work with using symmetric force approach")
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
    method='full'
    np.random.seed(int(time.time()))
    if variablePhi==True:
        variablePhi=[True,True]
    else:
        assert len(variablePhi)==2 and isinstance(variablePhi,list)
    shimTopParamBounds, shimTopLockedParams = make_Shim_Params(variableRadius, variablePhi.pop(0))
    shimBotParamBounds, shimBotLockedParams = make_Shim_Params(variableRadius, variablePhi.pop(0))
    shimOptimizer = ShimOptimizer(method)
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimTopParamBounds, shimTopLockedParams)
    shimOptimizer.add_Shim(shimBotParamBounds, shimBotLockedParams)
    # print(shimOptimizer.optimize())
    args=np.array([0.18508568, 0.05081386, 0.01449753, 5.10141806, 1.65634719,
       0.00769773, 0.50978621, 0.07465047, 0.03511048, 4.27038296,
       1.56204413, 0.0254    , 0.14343609])
    shimOptimizer.characterize_Results(args)
    # shimOptimizer.optimize_Descent(args)
# optimize_1Shim_Symmetric(.0254*3/8,np.pi/6.0)#,saveData='run1_1Shim')

optimize_1Shim_Symmetric(True,True)
optimize_1Shim_Symmetric(True,True)
optimize_1Shim_Symmetric(True,True)


'''
finished with total evals:  8796
---population member---- 
DNA: array([0.18820769, 0.0709656 , 0.03091131, 5.15298971, 1.56922431,
       0.0254    , 0.46719183])
cost: 0.2506882673972071
'''


