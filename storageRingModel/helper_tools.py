import copy
import itertools
import math
import random
import time
import warnings
from contextlib import contextmanager
from typing import Optional, Union, Callable, Any, Iterable

import multiprocess as mp
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats.qmc import Sobol

from constants import MASS_LITHIUM_7, BOLTZMANN_CONSTANT
from type_hints import RealNum, sequence, ndarray


def parallel_evaluate(func: Callable, args: Any, results_as_arr: bool = False, processes: int = -1,
                      re_randomize: bool = True, reapply_args: Optional[int] = None,
                      extra_positional_args: sequence = None, parallel: bool = True,
                      extra_keyword_args: Optional[dict] = None) -> Union[list, np.ndarray]:
    extra_args_constant = tuple() if extra_positional_args is None else extra_positional_args
    extra_key_word_args_constant = dict() if extra_keyword_args is None else extra_keyword_args
    no_positional_args = True if args is None else False
    assert isinstance(extra_key_word_args_constant, dict) and isinstance(
        extra_args_constant, (list, tuple, np.ndarray))
    assert isinstance(func, Callable)

    def wrapper(arg, seed):
        if re_randomize:
            np.random.seed(seed)
        if no_positional_args:
            return func(*extra_args_constant, **extra_key_word_args_constant)
        else:
            return func(arg, *extra_args_constant, **extra_key_word_args_constant)

    assert type(processes) == int and processes >= -1
    assert (type(reapply_args) == int and reapply_args >= 1) or reapply_args is None
    args_iter = (args,) * reapply_args if reapply_args is not None else args
    seed_arr = int(time.time()) + np.arange(len(args_iter))
    pool_iter = zip(args_iter, seed_arr)
    processes = mp.cpu_count() if processes == -1 else processes
    if parallel:
        with mp.Pool(processes, maxtasksperchild=10) as pool:
            results = pool.starmap(wrapper, pool_iter)
    else:
        results = [func(arg, *extra_args_constant, **extra_key_word_args_constant) for arg in args_iter]
    results = np.array(results) if results_as_arr else results
    return results


def make_image_cartesian(func: Callable, x_grid_edges: sequence, y_grid_edges: sequence,
                         extra_positional_args: sequence = None, extra_keyword_args=None, arr_input=False,
                         ) -> tuple[np.ndarray, list[float]]:
    extra_positional_args = tuple() if extra_positional_args is None else extra_positional_args
    extra_keyword_args = dict() if extra_keyword_args is None else extra_keyword_args
    assert isinstance(extra_keyword_args, dict) and isinstance(extra_positional_args, (list, tuple, np.ndarray))
    assert isinstance(x_grid_edges, (list, tuple, np.ndarray)) and isinstance(y_grid_edges, (list, tuple, np.ndarray))
    assert isinstance(func, Callable)
    coords = np.array(np.meshgrid(x_grid_edges, y_grid_edges)).T.reshape(-1, 2)
    if arr_input:
        print("using single array input to make image data")
        vals = func(coords, *extra_positional_args)
    else:
        print("looping over function inputs to make image data")
        vals = np.asarray([func(*coord, *extra_positional_args, **extra_keyword_args) for coord in coords])
    image = vals.reshape(len(x_grid_edges), len(y_grid_edges))
    image = np.rot90(image)
    extent = [min(x_grid_edges), max(x_grid_edges), min(y_grid_edges), max(y_grid_edges)]
    return image, extent


def make_dense_curve_1D_RBF(x: sequence, y: sequence, num_points: int = 10_000, smoothing: RealNum = 0.0) \
        -> tuple[np.ndarray, np.ndarray]:
    import scipy.interpolate as spi
    x = np.array(x) if isinstance(x, np.ndarray) == False else x
    y = np.array(y) if isinstance(y, np.ndarray) == False else y
    FWHM_func = spi.RBFInterpolator(x[:, None], y[:, None], smoothing=smoothing)
    x_dense = np.linspace(x.min(), x.max(), num_points)
    y_dense = np.ravel(FWHM_func(x_dense[:, None]))
    return x_dense, y_dense


def make_dense_curve_1D_linear(x: np.ndarray, y: np.ndarray, num_points: int = 10_000):
    x_dense = np.linspace(x.min(), x.max(), num_points)
    y_dense = interp1d(x, y)(x_dense)
    return x_dense, y_dense


def multiply_matrices(matrices: sequence, M_start=None, reverse=False, num_iters_cycling=None):
    """Given a sequence of matrices, multiply them together start to end and return the result, M_total.
    M_total=Mn@Mn-1 ... M1@M0. """
    assert matrices[0].shape[0] == matrices[0].shape[1] and matrices[0].ndim == 2
    matrix_size = matrices[0].shape[0]
    if reverse:
        matrices = reversed(matrices)
    if num_iters_cycling is not None:
        assert isinstance(num_iters_cycling, int) and num_iters_cycling > 0
        matrices = itertools.islice(itertools.cycle(matrices), num_iters_cycling)
    M_total = np.eye(matrix_size) if M_start is None else M_start
    for i, M in enumerate(matrices):
        M_total = M @ M_total if i > 0 else M
    return M_total


def shrink_bounds_around_vals(bounds: sequence, vals: sequence, shrink_frac: float) -> sequence:
    """Contract bounds about coordinates by a factor"""
    assert shrink_frac > 0
    bounds = copy.deepcopy(bounds)
    for bound, val in zip(bounds, vals):
        if val > bound[1] or val < bound[0] or not bound[1] >= bound[0]:
            raise ValueError
        bound_width = shrink_frac * (bound[1] - bound[0])
        bound_lower = val - bound_width / 2.0
        bound_upper = val + bound_width / 2.0
        bound[0] = bound_lower if bound[0] < bound_lower else bound[0]
        bound[1] = bound_upper if bound[1] > bound_upper else bound[1]
    return bounds


def grid_from_bounds(bounds: Iterable, num_points_edge: int) -> ndarray:
    """Return coordinates of points in a grid within specified bounds, and 'num_points_edge' points per dim. returned
    shaped is (dimensionality^num_points_edge,len(bounds))"""
    arrays = []
    for (lower, upper) in bounds:
        assert upper >= lower
        arr = np.linspace(lower, upper, num_points_edge)
        arrays.append(arr)
    grid_points = arr_product(*arrays)
    return grid_points


def random_num_for_seeding() -> int:
    """Return a random number appropriate to use as a seed with numpy or python"""
    MAX_32_BIT = int(2 ** 31) - 1  # biggest seed size for numpy random
    return np.random.randint(MAX_32_BIT)


def is_even(x: int) -> bool:
    """Return True if a number is even, False if it is odd"""
    assert type(x) is int and x > 0
    return True if x % 2 == 0 else False


def is_odd(x: int) -> bool:
    """Return True if a number is odd, False if it is even"""
    return not is_even(x)


def is_odd_length(values: sequence) -> bool:
    """Return True if the length of a sequence is odd, False if it is even"""
    return is_odd(len(values))


def radians(value_in_degrees: RealNum) -> float:
    return (value_in_degrees / 180) * np.pi


def degrees(value_in_radians: RealNum) -> float:
    return (value_in_radians / np.pi) * 180


def simulation_energy_to_mk(E):
    return 1e3 * E * MASS_LITHIUM_7 / BOLTZMANN_CONSTANT


def clamp(a: RealNum, a_min: RealNum, a_max: RealNum) -> RealNum:
    return min([max([a_min, a]), a_max])


def RMS(vals: sequence) -> float:
    """Simple RMS function"""
    vals = np.array(vals) if not isinstance(vals, np.ndarray) else vals
    return np.sqrt(np.sum(vals ** 2) / len(vals))


def is_ascending(vals: sequence) -> bool:
    """Return True if vals is in ascending order, else False"""
    if isinstance(vals, np.ndarray):
        return np.all(np.sort(vals) == vals)
    else:
        return all(val1 == val2 for val1, val2 in zip(vals, sorted(vals)))


def is_close_all(a: sequence, b: sequence, abstol: RealNum) -> bool:
    """Test that each element in array a and b are within tolerance of each other"""
    return np.all(np.isclose(a, b, atol=abstol, equal_nan=False))


def within_tol(a: RealNum, b: RealNum, tol: RealNum = 1e-12):
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.0)


def inch_to_meter(value_in_inches: RealNum) -> float:
    """Convert freedom units to commie units XD"""
    return .0254 * value_in_inches


def values_up_to_cutoff_frac(values: np.ndarray, fraction: float) -> np.ndarray:
    """Return sorted subset of 'values' up to some fractional cutoff of the maximum"""
    assert 0.0 <= fraction <= 1.0 and len(values) > 0
    arg_at_frac = round(len(values) * fraction)
    return np.sort(values)[:arg_at_frac]


def val_at_cumulative_fraction(values: np.ndarray, fraction: float) -> float:
    """Sort 'values', and return the nearest element to 'fraction' of the total length"""
    return values_up_to_cutoff_frac(values, fraction)[-1]


def meter_to_cm(value_in_meters: RealNum) -> float:
    return value_in_meters * 100.0


def gauss_to_tesla(value_in_gauss: RealNum) -> float:
    """Convert units of gauss to tesla"""
    return value_in_gauss / 10_000.0


def tesla_to_guass(value_in_tesla: RealNum) -> float:
    """Convert units of tesla to gauss"""
    return value_in_tesla * 10_000.0


def arr_product(*args):
    """Use itertools to form a product of provided arrays, and return an array instead of iterable"""
    return np.asarray(list(itertools.product(*args)))


def full_arctan2(y, x):
    """Compute angle spanning 0 to 2pi degrees as expected from x and y where q=numpy.array([x,y,z])"""
    phi = np.arctan2(y, x)
    if phi < 0:  # confine phi to be between 0 and 2pi
        phi += 2 * np.pi
    return phi


def make_odd(num: int) -> int:
    assert isinstance(num, int) and num != 0 and not num < 0  # i didn't verify this on negative numbers
    return num + (num + 1) % 2


def round_and_make_odd(num: RealNum) -> int:
    return make_odd(round(num))


@contextmanager
def temporary_seed(seed: int) -> None:
    """context manager that temporarily overrides numpy and python random
    generators by seeding with the value 'seed', then restores the previous state"""
    if seed is not None:
        state_np = np.random.get_state()
        state_py = random.getstate()
        np.random.seed(seed=seed)
        random.seed(a=seed)
        yield
        np.random.set_state(state_np)
        random.setstate(state_py)
    else:
        yield


def low_discrepancy_sample(bounds: sequence, num: int, seed=None) -> np.ndarray:
    """
    Make a low discrepancy sample (ie well spread)

    :param bounds: sequence of lower and upper bounds of the sampling
    :param num: number of samples. powers of two are best for nice sobol properties
    :param seed: Seed for sobol sampler. None, and no seeding. See scipy docs.
    :return:
    """
    bounds = np.array(bounds).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = np.array(Sobol(len(bounds), scramble=True, seed=seed).random(num))
    scale_factors = bounds[:, 1] - bounds[:, 0]
    offsets = bounds[:, 0]
    for i, sample in enumerate(samples):
        samples[i] = sample * scale_factors + offsets
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
#     resultsParallel=parallel_evaluate(dummy,X_Arr,extra_positional_args=a0,extra_keyword_args=optionalArgs,
#                                                   results_as_arr=True)
#     assert np.all(np.abs(np.mean(results-resultsParallel))<tol)
#
#     def dummy2(X,a,b=None):
#         b=0.0 if b is None else b['stuff']
#         return np.random.random_sample()*1e-6
#     resultsParallel=np.sort(parallel_evaluate(dummy2,X_Arr,extra_positional_args=a0,extra_keyword_args=optionalArgs,
#                                                   results_as_arr=True,re_randomize=True))
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
#     image, extent = make_image_cartesian(fake_Func, x_arr, y_arr, extra_positional_args=[None])
#     assert (
#         image[len(y_arr) - 1, 0] == 1
#         and image[0, 0] == 2
#         and image[0, len(x_arr) - 1] == 3
#     )
