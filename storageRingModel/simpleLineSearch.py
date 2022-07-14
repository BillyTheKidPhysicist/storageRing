from helperTools import *


def change_value_along_dimension(xi, dim, delta, bounds):
    x_new = np.array(xi)
    x_new[dim] += delta
    x_upper = np.array(bounds)[:, 1]
    x_lower = np.array(bounds)[:, 0]
    x_new = np.clip(x_new, a_min=x_lower, a_max=x_upper)
    return x_new


def get_search_params(xi, delta, bounds):
    list_of_params = []
    for i in range(len(xi)):
        for shift in [-delta, delta]:
            list_of_params.append(change_value_along_dimension(xi, i, shift, bounds))
    return list_of_params


def find_optimal_value(func, xi, delta, bounds, processes):
    assert len(bounds) == len(xi)
    list_of_params = get_search_params(xi, delta, bounds)
    results = parallel_evaluate(func, list_of_params, processes=processes)
    return list_of_params[np.argmin(results)], np.min(results)


def line_search(cost_func, xi, delta_initial, bounds, cost_initial=None, processes=-1):
    x_opt, delta = xi, delta_initial
    frac_reduction=.75
    cost_opt = cost_initial if cost_initial is not None else cost_func(xi)
    i,failures, maxiters = 0,0, 10
    while failures < maxiters:
        i += 1
        print('results', cost_opt, delta, repr(x_opt))
        x_opt_new, cost_opt_new = find_optimal_value(cost_func, x_opt, delta, bounds, processes)
        if cost_opt_new >= cost_opt:
            delta *= frac_reduction
        else:
            x_opt = x_opt_new
            cost_opt = cost_opt_new
            failures+=1
    return x_opt, cost_opt
