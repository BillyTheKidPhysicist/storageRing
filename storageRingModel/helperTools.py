import itertools

import numpy as np
from scipy.stats.qmc import Sobol
import os
import sympy as sym
import matplotlib.pyplot as plt
import numba
import pandas as pd
import time
import multiprocess as mp
from typing import Optional, Union, Callable, Any
import math

lst_tup_arr_type = Union[list, tuple, np.ndarray]
realNumType=Union[int,float]

def tool_Parallel_Process(
    func: Callable,
    args: Any,
    resultsAsArray: bool = False,
    processes: int = -1,
    reRandomize: bool = True,
    reapplyArgs: Optional[int] = None,
    extraArgs: lst_tup_arr_type = None,
    extraKeyWordArgs: Optional[dict]=None
) -> Union[list, np.ndarray]:

    extraArgs_Constant = tuple() if extraArgs is None else extraArgs
    extraKeyWordArgs_Constant = dict() if extraKeyWordArgs is None else extraKeyWordArgs
    noPositionalArgs = True if args is None else False
    assert isinstance(extraKeyWordArgs_Constant, dict) and isinstance(
        extraArgs_Constant, (list, tuple, np.ndarray)
    )
    assert isinstance(func, Callable)

    def wrapper(arg, seed):
        if reRandomize == True:
            np.random.seed(seed)
        if noPositionalArgs == True:
            return func(*extraArgs_Constant, **extraKeyWordArgs_Constant)
        else:
            return func(arg, *extraArgs_Constant, **extraKeyWordArgs_Constant)

    assert type(processes) == int and processes >= -1
    assert (type(reapplyArgs) == int and reapplyArgs >= 1) or reapplyArgs is None
    argsIter = (args,) * reapplyArgs if reapplyArgs is not None else args
    seedArr = int(time.time()) + np.arange(len(argsIter))
    poolIter = zip(argsIter, seedArr)
    processes = mp.cpu_count() if processes == -1 else processes
    if processes > 1:
        with mp.Pool(processes,maxtasksperchild=10) as pool:
            results = pool.starmap(wrapper, poolIter)
    elif processes == 1:
        results = [func(arg, *extraArgs_Constant, **extraKeyWordArgs_Constant) for arg in argsIter]
    else:
        raise ValueError
    results = np.array(results) if resultsAsArray == True else results
    return results


def tool_Make_Image_Cartesian(
    func: Callable,
    xGridEdges: lst_tup_arr_type,
    yGridEdges: lst_tup_arr_type,
    extraArgs: lst_tup_arr_type = None,
    extraKeyWordArgs=None,
    arrInput=False,
) -> tuple[np.ndarray, list]:

    extraArgs = tuple() if extraArgs is None else extraArgs
    extraKeyWordArgs = dict() if extraKeyWordArgs is None else extraKeyWordArgs
    assert isinstance(extraKeyWordArgs, dict) and isinstance(extraArgs, (list, tuple, np.ndarray))
    assert isinstance(xGridEdges, (list, tuple, np.ndarray)) and isinstance(yGridEdges, (list, tuple, np.ndarray))
    assert isinstance(func, Callable)
    coords = np.array(np.meshgrid(xGridEdges, yGridEdges)).T.reshape(-1, 2)
    if arrInput == True:
        print("using single array input to make image data")
        vals = func(coords, *extraArgs)
    else:
        print("looping over function inputs to make image data")
        vals = np.asarray([func(*coord, *extraArgs, **extraKeyWordArgs) for coord in coords])
    image = vals.reshape(len(xGridEdges), len(yGridEdges))
    image = np.rot90(image)
    extent = [min(xGridEdges), max(xGridEdges), min(yGridEdges), max(yGridEdges)]
    return image, extent

def tool_Dense_Curve(x: lst_tup_arr_type,y: lst_tup_arr_type,numPoints: int=10_000,smoothing: float=0.0) \
        ->tuple[np.ndarray,np.ndarray]:
    import scipy.interpolate as spi
    x=np.array(x) if isinstance(x,np.ndarray)==False else x
    y=np.array(y) if isinstance(y,np.ndarray)==False else y
    FWHM_Func=spi.RBFInterpolator(x[:,None],y[:,None],smoothing=smoothing)
    xDense=np.linspace(x.min(),x.max(),numPoints)
    yDense=np.ravel(FWHM_Func(xDense[:,None]))
    return xDense,yDense

def radians(degrees: float):
    return (degrees/180)*np.pi

def degrees(radians: float):
    return (radians/np.pi)*180

def iscloseAll(a: lst_tup_arr_type,b: lst_tup_arr_type,abstol: float)->bool:
    """Test that each element in array a and b are within tolerance of each other"""
    return np.all(np.isclose(a,b,atol=abstol,equal_nan=False))

def within_Tol(a: realNumType,b: realNumType,tol: realNumType=1e-12):
    return math.isclose(a,b,abs_tol=tol,rel_tol=0.0)

def inch_To_Meter(inches: Union[float,int])-> float:
    """Convert freedom units to commie units XD"""

    return .0254*inches

def arr_Product(*args):
    """Use itertools to form a product of provided arrays, and return an array instead of iterable"""
    return np.asarray(list(itertools.product(*args)))

def full_Arctan(y,x):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi

def make_Odd(num: int)-> int:
    assert isinstance(num,int) and num!=0 and not num<0 # i didn't verify this on negative numbers
    return num + (num + 1) % 2

def low_Discrepancy_Sample(bounds: lst_tup_arr_type,num: int, seed=None)-> np.ndarray:
    """
    Make a low discrepancy sample (ie well spread)

    :param bounds: sequence of lower and upper bounds of the sampling
    :param num: number of samples. powers of two are best for nice sobol properties
    :params seed: Seed for sobol sampler. None, and no seeding. See scipy docs.
    :return:
    """
    sobolSeed = None if not seed else seed
    bounds=np.array(bounds).astype(float)
    samples = np.array(Sobol(len(bounds), scramble=True,seed=sobolSeed).random(num))
    scaleFactors=bounds[:,1]-bounds[:,0]
    offsets=bounds[:,0]
    for i,sample in enumerate(samples):
        samples[i]=sample*scaleFactors+offsets
    return samples

def test_Parallel_Process():
    tol=1e-12
    def dummy(X,a,b=None):
        b=0.0 if b is None else b['stuff']
        return np.sin(X[0])+a+np.tanh(X[1])-b
    X_Arr=np.linspace(0.0,10.0,40).reshape(-1,2)
    a0=(7.5,)
    b0=np.pi
    optionalArgs={'b':{'stuff':b0}}
    results=np.asarray([dummy(X,a0[0],b={'stuff':b0}) for X in X_Arr])
    resultsParallel=tool_Parallel_Process(dummy,X_Arr,extraArgs=a0,extraKeyWordArgs=optionalArgs,
                                                  resultsAsArray=True)
    assert np.all(np.abs(np.mean(results-resultsParallel))<tol)

    def dummy2(X,a,b=None):
        b=0.0 if b is None else b['stuff']
        return np.random.random_sample()*1e-6
    resultsParallel=np.sort(tool_Parallel_Process(dummy2,X_Arr,extraArgs=a0,extraKeyWordArgs=optionalArgs,
                                                  resultsAsArray=True,reRandomize=True))
    assert len(np.unique(resultsParallel))==len(X_Arr)

def test_tool_Make_Image_Cartesian():
    def fake_Func(x, y, z):
        assert z is None
        if np.abs(x + 1.0) < 1e-9 and np.abs(y + 1.0) < 1e-9:  # x=-1.0,y=-1.0   BL
            return 1
        if np.abs(x + 1.0) < 1e-9 and np.abs(y - 1.0) < 1e-9:  # x=-1.0, y=1.0 TL
            return 2
        if np.abs(x - 1.0) < 1e-9 and np.abs(y - 1.0) < 1e-9:  # x=1.0,y=1.0   TR
            return 3
        return 0.0

    xArr = np.linspace(-1, 1, 15)
    yArr = np.linspace(-1, 1, 5)
    image, extent = tool_Make_Image_Cartesian(fake_Func, xArr, yArr, extraArgs=[None])
    assert (
        image[len(yArr) - 1, 0] == 1
        and image[0, 0] == 2
        and image[0, len(xArr) - 1] == 3
    )
