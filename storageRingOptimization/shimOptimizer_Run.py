import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import time
from shimAndGeneticLensOptimization import ShimOptimizer
from profilehooks import profile

rp = .05
L0 = .23
magnetWidth = .0254
lensBounds = {'length': (L0- rp, L0+rp)}
lensParams = {'rp': rp, 'width': magnetWidth}
lensBaseLineParams = {'rp': rp, 'width': magnetWidth, 'length': L0}


def make_Shim_Params(variableDiam,variablePhi,shape,symmetric=True,location=None):
    if shape=='sphere':
        shim1ParamBounds = {'r': (rp, rp + magnetWidth), 'deltaZ': (0.0, .75*rp),
                            'psi': (0.0, 2 * np.pi),'theta':(0.0,np.pi)}
        shim1LockedParams = {'planeSymmetry': symmetric,'shape':'sphere'}
    elif shape=='cube':
        shim1ParamBounds = {'r': (rp, rp + magnetWidth),
                            'psi': (0.0, 2 * np.pi)}
        shim1LockedParams = {'planeSymmetry': symmetric, 'shape': 'cube','deltaZ':1e-3}
        pass
    else: raise ValueError
    if symmetric==False:
        shim1LockedParams['location']=location
    if variableDiam == True:
        shim1ParamBounds['diameter'] = (.0254 / 16, .0254)
    else:
        assert isinstance(variableDiam, float)
        shim1LockedParams['diameter'] = variableDiam
    if variablePhi == True:
        shim1ParamBounds['phi'] = (0, np.pi / 6)
    else:
        assert isinstance(variablePhi, float)
        shim1LockedParams['phi'] = variablePhi
    return shim1ParamBounds, shim1LockedParams
def optimize_Radial_Profile(numlayers):
    assert numlayers%2==0
    lensLengthSymmetry=True
    lensBounds = [{'rp': (.9*rp,1.5*rp)} for _ in range(numlayers//2)]
    lensParamsLocked = [{'width': magnetWidth,'length':L0/numlayers} for _ in range(numlayers//2)]
    lensBaseLineParams = [{'rp': rp, 'width': magnetWidth, 'length': L0}]
    shimOptimizer = ShimOptimizer('full',lensLengthSymmetry)
    shimOptimizer.set_Lens(lensBounds, lensParamsLocked, lensBaseLineParams)
    shimOptimizer.optimize()
    # args0=np.array([0.05401033, 0.0507947 , 0.04620604])
    # shimOptimizer.characterize_Results(args0)
def optimize_1Shim_Symmetric(variableDiam,variablePhi,shape,saveData=None):
    method='full'
    np.random.seed(int(time.time()))
    shimParamBounds, shimLockedParams=make_Shim_Params(variableDiam,variablePhi,shape)
    shimOptimizer = ShimOptimizer(method)
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimParamBounds, shimLockedParams)
    shimOptimizer.optimize(saveData=saveData)
    # args0=np.array([0.21411904, 0.05195688, 5.63036833, 0.0250395 , 0.46592191])

    # shimOptimizer.cost_Function(args0,True,False)
    # print(shimOptimizer.optimize_Descent(Xi=args0))
    # shimOptimizer.characterize_Results(args0)
def optimize_2Shim_Symmetric(variableDiam,variablePhi,shape,saveData=None):
    method='full'
    np.random.seed(int(time.time()))
    if variablePhi==True:
        variablePhi=[True,True]
    else:
        assert len(variablePhi)==2 and isinstance(variablePhi,list)
    shimTopParamBounds, shimTopLockedParams = make_Shim_Params(variableDiam, variablePhi.pop(0),shape)
    shimBotParamBounds, shimBotLockedParams = make_Shim_Params(variableDiam, variablePhi.pop(0),shape)
    shimOptimizer = ShimOptimizer(method)
    shimOptimizer.set_Lens(lensBounds, lensParams,lensBaseLineParams)
    shimOptimizer.add_Shim(shimTopParamBounds, shimTopLockedParams)
    shimOptimizer.add_Shim(shimBotParamBounds, shimBotLockedParams)
    shimOptimizer.optimize_Shims()
    # args=np.array([0.18508568, 0.05081386, 0.01449753, 5.10141806, 1.65634719,
    #    0.00769773, 0.50978621, 0.07465047, 0.03511048, 4.27038296,
    #    1.56204413, 0.0254    , 0.14343609])
    # shimOptimizer.characterize_Results(args)
    # shimOptimizer.optimize_Descent(args)
# optimize_1Shim_Symmetric(.0254*3/8,np.pi/6.0)#,saveData='run1_1Shim')
optimize_Radial_Profile(12)

# optimize_1Shim_Symmetric(True,True,'cube')
# optimize_1Shim_Symmetric(True,True,'cube')
# optimize_1Shim_Symmetric(True,True,'cube')
# optimize_2Shim_Symmetric(True,True,'cube')
# optimize_2Shim_Symmetric(True,True,'cube')
# optimize_2Shim_Symmetric(True,True,'cube')





'''
finished with total evals:  8796
---population member---- 
DNA: array([0.18820769, 0.0709656 , 0.03091131, 5.15298971, 1.56922431,
       0.0254    , 0.46719183])
cost: 0.2506882673972071
'''



'''
squares at surface of magnet limited to inner edge at r
['length', 'r0', 'psi0', 'diameter0', 'phi0']
finished with total evals:  3882
---population member---- 
DNA: array([0.21322827, 0.05126325, 5.570695  , 0.02520376, 0.45238814])
cost: 0.31805188101011805
['length', 'r0', 'psi0', 'diameter0', 'phi0']
finished with total evals:  4012
---population member---- 
DNA: array([0.21635269, 0.05130414, 5.64479054, 0.02306725, 0.47461783])
cost: 0.319582841138564
['length', 'r0', 'psi0', 'diameter0', 'phi0']
finished with total evals:  3244
---population member---- 
DNA: array([0.21411904, 0.05195688, 5.63036833, 0.0250395 , 0.46592191])
cost: 0.3162646705994698
['length', 'r0', 'psi0', 'diameter0', 'phi0', 'r1', 'psi1', 'diameter1', 'phi1']
finished with total evals:  33844
---population member---- 
DNA: array([0.19570573, 0.05073938, 4.38488839, 0.02400479, 0.12549998,
       0.05007446, 5.84860153, 0.02422526, 0.50790065])
cost: 0.19265703380369373
'''

'''
square limited to surface of magnet, but locking the angle

'''


