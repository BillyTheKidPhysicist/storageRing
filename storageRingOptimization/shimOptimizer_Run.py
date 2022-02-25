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
    # args0=np.array([0.22556419, 0.05623946, 0.01095473, 1.39222029, 0.        ])
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
optimize_1Shim_Symmetric(.0254*3/8,np.pi/6.0,saveData='run2_1Shim')
optimize_1Shim_Symmetric(.0254*3/8,np.pi/6.0,saveData='run3_1Shim')

# optimize_2Shim_Symmetric(True,True)#,saveData='run1_2Shim')
# optimize_2Shim_Symmetric(True,True)#,saveData='run2_2Shim')
# optimize_2Shim_Symmetric(True,True)#,saveData='run3_2Shim')


'''
1 shim
Tru True
finished with total evals:  5256
---population member---- 
DNA: array([0.22469908, 0.05459526, 0.01088656, 1.71126511, 5.37597293,
       0.00911954, 0.51417334])
cost: 0.5046830756884966
finished with total evals:  5556
---population member---- 
DNA: array([0.1923749 , 0.07431393, 0.03455355, 1.58521529, 5.32509403,
       0.02536473, 0.43163293])
cost: 0.5908714273303177
finished with total evals:  5471
---population member---- 
DNA: array([0.22499909, 0.05374655, 0.01137538, 1.58994084, 5.1274135 ,
       0.00848553, 0.52165971])
cost: 0.514339990084301
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