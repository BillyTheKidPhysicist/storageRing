from typing import Optional

from latticeModels.injectorModel_any import injector_param_bounds, injector_params_optimal
from latticeModels.injectorModel_any import make_injector_lattice as make_injector_lattice_any
from latticeModels.latticeModelUtilities import assert_combiners_are_same
from latticeModels.ringModelSurrogate_any import make_ring_surrogate_for_injector
from latticeModels.ringModel_1 import make_ring_lattice as make_ring_lattice_1
from latticeModels.ringModel_1 import ring_param_bounds as ring_param_bounds_1
from latticeModels.ringModel_1 import ring_params_optimal as ring_params_optimal_1
from latticeModels.ringModel_2 import make_ring_lattice as make_ring_lattice_2
from latticeModels.ringModel_2 import ring_param_bounds as ring_param_bounds_2
from latticeModels.ringModel_2 import ring_params_optimal as ring_params_optimal_2


def get_ring_bounds(ring_version: str):
    if ring_version == '1':
        return ring_param_bounds_1
    elif ring_version == '2':
        return ring_param_bounds_2
    else:
        raise NotImplementedError


def get_optimal_ring_params(ring_version: str):
    if ring_version == '1':
        return ring_params_optimal_1
    elif ring_version == '2':
        return ring_params_optimal_2
    else:
        raise NotImplementedError


def get_optimal_injector_params():
    return injector_params_optimal


def get_injector_bounds():
    return injector_param_bounds


def make_injector_lattice(injector_params, system_options):
    return make_injector_lattice_any(injector_params, options=system_options)


def make_ring_lattice(ring_params, ring_version: str, system_options: dict):
    if ring_version == '1':
        return make_ring_lattice_1(ring_params, options=system_options)
    elif ring_version == '2':
        return make_ring_lattice_2(ring_params, options=system_options)
    else:
        raise NotImplementedError


def make_surrogate_ring_for_injector(injector_params, ring_version, system_options: dict):
    assert ring_version in ('1', '2')
    return make_ring_surrogate_for_injector(injector_params, system_options)


def make_system_model(ring_params, injector_params, ring_version: str, system_options: Optional[dict]):
    lattice_ring = make_ring_lattice(ring_params, ring_version, system_options)
    lattice_injector = make_injector_lattice_any(injector_params, options=system_options)
    assert_combiners_are_same(lattice_injector, lattice_ring)
    return lattice_ring, lattice_injector


def make_optimal_ring_and_injector_lattice(ring_version):
    ring_params_optimal = get_optimal_ring_params(ring_version)
    injector_params_optimal = get_optimal_injector_params()
    return make_system_model(ring_params_optimal, injector_params_optimal, ring_version, None)
