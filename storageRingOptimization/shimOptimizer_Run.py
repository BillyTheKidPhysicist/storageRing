import os
os.environ['OPENBLAS_NUM_THREADS']='1'
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
    shimTopParamBounds, shimTopLockedParams = make_Shim_Params(variableRadius, variablePhi)
    shimBotParamBounds, shimBotLockedParams = make_Shim_Params(variableRadius, variablePhi)
    shimOptimizer = ShimOptimizer()
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimTopParamBounds, shimTopLockedParams)
    shimOptimizer.add_Shim(shimBotParamBounds, shimBotLockedParams)
    print(shimOptimizer.optimize1())

optimize_1Shim_Symmetric(.0254*3/8,np.pi/6.0)#,saveData='run1_1Shim')

# optimize_2Shim_Symmetric(True,True)#,saveData='run1_2Shim')
# optimize_2Shim_Symmetric(True,True)#,saveData='run2_2Shim')
# optimize_2Shim_Symmetric(True,True)#,saveData='run3_2Shim')


'''
1 shim
Tru True



array([0.22335083, 0.05468504, 0.01778291, 1.86398975, 5.89642321]) 0.3819820289449764
array([0.22336635, 0.05398954, 0.01994955, 2.174386  , 5.1246072 ]) 0.4063641058343459
array([0.22336904, 0.05486956, 0.01967456, 1.16595304, 4.66104557]) 0.43165054561094596

array([0.22232782 ,0.05411236 ,0.01467107 ,1.69968186 ,5.40909307]) 0.33767767499002077



array([0.2228413  ,0.05462561 ,0.01132046 ,1.28740702 ,5.31414559]) 0.354635759337459



'''



'''

2 shim

True True
finished with total evals:  8137
---population member---- 
DNA: array([0.22510602, 0.05416986, 0.01163824, 1.8349294 , 5.23461587,
       0.00887285, 0.50769194, 0.05069661, 0.02116212, 1.44434705,
       6.28318531, 0.00876063, 0.22059464])
cost: 0.5067221917496517
finished with total evals:  8607
---population member---- 
DNA: array([0.22517946, 0.05562441, 0.01129363, 1.66916262, 0.        ,
       0.00920485, 0.52359878, 0.05176082, 0.04809833, 1.61851837,
       0.        , 0.00727893, 0.45137545])
cost: 0.5526184497842086
finished with total evals:  7386
---population member---- 
DNA: array([0.19284814, 0.07367982, 0.04644471, 1.72461535, 5.20144124,
       0.0254    , 0.38376053, 0.0671824 , 0.05      , 3.14159265,
       2.99035691, 0.02058372, 0.090305  ])
cost: 0.5845379567377074

'''