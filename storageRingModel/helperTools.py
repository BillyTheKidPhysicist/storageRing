import itertools
import math
import time
import warnings
from typing import Optional, Union, Callable, Any
import matplotlib.pyplot as plt

import multiprocess as mp
import numpy as np
from scipy.stats.qmc import Sobol
from typeHints import RealNum
lst_tup_arr_type = Union[list, tuple, np.ndarray]


def tool_Parallel_Process(
        func: Callable,
        args: Any,
        resultsAsArray: bool = False,
        processes: int = -1,
        reRandomize: bool = True,
        reapply_args: Optional[int] = None,
        extraArgs: lst_tup_arr_type = None,
        extraKeyWordArgs: Optional[dict] = None
) -> Union[list, np.ndarray]:
    extra_args_constant = tuple() if extraArgs is None else extraArgs
    extra_key_word_args_constant = dict() if extraKeyWordArgs is None else extraKeyWordArgs
    no_positional_args = True if args is None else False
    assert isinstance(extra_key_word_args_constant, dict) and isinstance(
        extra_args_constant, (list, tuple, np.ndarray)
    )
    assert isinstance(func, Callable)

    def wrapper(arg, seed):
        if reRandomize == True:
            np.random.seed(seed)
        if no_positional_args == True:
            return func(*extra_args_constant, **extra_key_word_args_constant)
        else:
            return func(arg, *extra_args_constant, **extra_key_word_args_constant)

    assert type(processes) == int and processes >= -1
    assert (type(reapply_args) == int and reapply_args >= 1) or reapply_args is None
    args_iter = (args,) * reapply_args if reapply_args is not None else args
    seed_arr = int(time.time()) + np.arange(len(args_iter))
    pool_iter = zip(args_iter, seed_arr)
    processes = mp.cpu_count() if processes == -1 else processes
    if processes > 1:
        with mp.Pool(processes, maxtasksperchild=10) as pool:
            results = pool.starmap(wrapper, pool_iter)
    elif processes == 1:
        results = [func(arg, *extra_args_constant, **extra_key_word_args_constant) for arg in args_iter]
    else:
        raise ValueError
    results = np.array(results) if resultsAsArray == True else results
    return results


def tool_Make_Image_Cartesian(
        func: Callable,
        xGridEdges: lst_tup_arr_type,
        yGridEdges: lst_tup_arr_type,
        extra_args: lst_tup_arr_type = None,
        extra_key_word_args=None,
        arr_input=False,
) -> tuple[np.ndarray, list]:
    extra_args = tuple() if extra_args is None else extra_args
    extra_key_word_args = dict() if extra_key_word_args is None else extra_key_word_args
    assert isinstance(extra_key_word_args, dict) and isinstance(extra_args, (list, tuple, np.ndarray))
    assert isinstance(xGridEdges, (list, tuple, np.ndarray)) and isinstance(yGridEdges, (list, tuple, np.ndarray))
    assert isinstance(func, Callable)
    coords = np.array(np.meshgrid(xGridEdges, yGridEdges)).T.reshape(-1, 2)
    if arr_input == True:
        print("using single array input to make image data")
        vals = func(coords, *extra_args)
    else:
        print("looping over function inputs to make image data")
        vals = np.asarray([func(*coord, *extra_args, **extra_key_word_args) for coord in coords])
    image = vals.reshape(len(xGridEdges), len(yGridEdges))
    image = np.rot90(image)
    extent = [min(xGridEdges), max(xGridEdges), min(yGridEdges), max(yGridEdges)]
    return image, extent


def tool_Dense_Curve(x: lst_tup_arr_type, y: lst_tup_arr_type, numPoints: int = 10_000, smoothing: float = 0.0) \
        -> tuple[np.ndarray, np.ndarray]:
    import scipy.interpolate as spi
    x = np.array(x) if isinstance(x, np.ndarray) == False else x
    y = np.array(y) if isinstance(y, np.ndarray) == False else y
    FWHM_func = spi.RBFInterpolator(x[:, None], y[:, None], smoothing=smoothing)
    x_dense = np.linspace(x.min(), x.max(), numPoints)
    y_dense = np.ravel(FWHM_func(x_dense[:, None]))
    return x_dense, y_dense


def radians(degrees: float):
    return (degrees / 180) * np.pi


def degrees(radians: float):
    return (radians / np.pi) * 180


def clamp(a: float, a_min: float, a_max: float):
    return min([max([a_min, a]), a_max])


def iscloseAll(a: lst_tup_arr_type, b: lst_tup_arr_type, abstol: float) -> bool:
    """Test that each element in array a and b are within tolerance of each other"""
    return np.all(np.isclose(a, b, atol=abstol, equal_nan=False))


def within_Tol(a: RealNum, b: RealNum, tol: RealNum = 1e-12):
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.0)


def inch_To_Meter(inches: Union[float, int]) -> float:
    """Convert freedom units to commie units XD"""

    return .0254 * inches


def gauss_To_Tesla(gauss: Union[float, int]) -> float:
    """Convert units of gauss to tesla"""
    return gauss / 10_000.0


def tesla_To_Guass(tesla: Union[float, int]) -> float:
    """Convert units of tesla to gauss"""
    return tesla * 10_000.0


def arr_Product(*args):
    """Use itertools to form a product of provided arrays, and return an array instead of iterable"""
    return np.asarray(list(itertools.product(*args)))


def full_arctan2(y, x):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


def make_Odd(num: int) -> int:
    assert isinstance(num, int) and num != 0 and not num < 0  # i didn't verify this on negative numbers
    return num + (num + 1) % 2

def round_And_Make_Odd(num:RealNum) -> int:
    return make_Odd(round(num))

def low_discrepancy_sample(bounds: lst_tup_arr_type, num: int, seed=None) -> np.ndarray:
    """
    Make a low discrepancy sample (ie well spread)

    :param bounds: sequence of lower and upper bounds of the sampling
    :param num: number of samples. powers of two are best for nice sobol properties
    :param seed: Seed for sobol sampler. None, and no seeding. See scipy docs.
    :return:
    """
    sobolSeed = None if not seed else seed
    bounds = np.array(bounds).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = np.array(Sobol(len(bounds), scramble=True, seed=sobolSeed).random(num))
    scaleFactors = bounds[:, 1] - bounds[:, 0]
    offsets = bounds[:, 0]
    for i, sample in enumerate(samples):
        samples[i] = sample * scaleFactors + offsets
    return samples
#
# def test_Parallel_Process():
#     tol=1e-12
#     def dummy(X,a,b=None):
#         b=0.0 if b is None else b['stuff']
#         return np.sin(X[0])+a+np.tanh(X[1])-b
#     X_Arr=np.linspace(0.0,10.0,40).reshape(-1,2)
#     a0=(7.5,)
#     b0=np.pi
#     optionalArgs={'b':{'stuff':b0}}
#     results=np.asarray([dummy(X,a0[0],b={'stuff':b0}) for X in X_Arr])
#     resultsParallel=tool_Parallel_Process(dummy,X_Arr,extraArgs=a0,extraKeyWordArgs=optionalArgs,
#                                                   resultsAsArray=True)
#     assert np.all(np.abs(np.mean(results-resultsParallel))<tol)
#
#     def dummy2(X,a,b=None):
#         b=0.0 if b is None else b['stuff']
#         return np.random.random_sample()*1e-6
#     resultsParallel=np.sort(tool_Parallel_Process(dummy2,X_Arr,extraArgs=a0,extraKeyWordArgs=optionalArgs,
#                                                   resultsAsArray=True,reRandomize=True))
#     assert len(np.unique(resultsParallel))==len(X_Arr)
#
# def test_tool_Make_Image_Cartesian():
#     def fake_Func(x, y, z):
#         assert z is None
#         if np.abs(x + 1.0) < 1e-9 and np.abs(y + 1.0) < 1e-9:  # x=-1.0,y=-1.0   BL
#             return 1
#         if np.abs(x + 1.0) < 1e-9 and np.abs(y - 1.0) < 1e-9:  # x=-1.0, y=1.0 TL
#             return 2
#         if np.abs(x - 1.0) < 1e-9 and np.abs(y - 1.0) < 1e-9:  # x=1.0,y=1.0   TR
#             return 3
#         return 0.0
#
#     x_arr = np.linspace(-1, 1, 15)
#     y_arr = np.linspace(-1, 1, 5)
#     image, extent = tool_Make_Image_Cartesian(fake_Func, x_arr, y_arr, extraArgs=[None])
#     assert (
#         image[len(y_arr) - 1, 0] == 1
#         and image[0, 0] == 2
#         and image[0, len(x_arr) - 1] == 3
#     )
