import numpy as np
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


def tool_Parallel_Process(
    func: Callable,
    args: Any,
    resultsAsArray: bool = False,
    processes: int = -1,
    reRandomize: bool = True,
    reapplyArgs: Optional[int] = None,
    extraArgs: lst_tup_arr_type = None,
    extraKeyWordArgs=None
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
        with mp.Pool(processes) as pool:
            results = pool.starmap(wrapper, poolIter)
    elif processes == 1:
        results = [func(arg, *extraArgs, **extraKeyWordArgs) for arg in argsIter]
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
    assert isinstance(extraKeyWordArgs, dict) and isinstance(
        extraArgs, (list, tuple, np.ndarray)
    )
    assert isinstance(xGridEdges, (list, tuple, np.ndarray)) and isinstance(
        yGridEdges, (list, tuple, np.ndarray)
    )
    assert isinstance(func, Callable)
    coords = np.array(np.meshgrid(xGridEdges, yGridEdges)).T.reshape(-1, 2)
    if arrInput == True:
        print("using single array input to make image data")
        vals = func(coords, *extraArgs)
    else:
        print("looping over function inputs to make image data")
        vals = np.asarray(
            [func(*coord, *extraArgs, **extraKeyWordArgs) for coord in coords]
        )
    image = vals.reshape(len(xGridEdges), len(yGridEdges))
    image = np.rot90(image)
    extent = [xGridEdges.min(), xGridEdges.max(), yGridEdges.min(), yGridEdges.max()]
    return image, extent


def _test_tool_Make_Image_Cartesian():
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