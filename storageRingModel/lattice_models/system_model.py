from typing import Optional

import lattice_models.ring_model_1 as ring_model_1
import lattice_models.ring_model_1 as ring_model_2
import lattice_models.ring_model_1 as ring_model_3
from particle_tracer_lattice import ParticleTracerLattice
from lattice_models.injector_model_any import injector_param_bounds, injector_params_optimal
from lattice_models.injector_model_any import make_injector_lattice as make_injector_lattice_any
from lattice_models.ring_model_surrogate_any import make_ring_surrogate_for_injector
from lattice_models.utilities import assert_combiners_are_same, LockedDict

ring_models = {'1': ring_model_1, '2': ring_model_2, '3': ring_model_3}


def get_ring_bounds(ring_version: str) -> LockedDict:
    return ring_models[ring_version].ring_param_bounds


def get_optimal_ring_params(ring_version: str) -> LockedDict:
    return ring_models[ring_version].ring_params_optimal


def get_optimal_injector_params() -> LockedDict:
    return injector_params_optimal


def get_injector_bounds() -> LockedDict:
    return injector_param_bounds


def make_injector_lattice(injector_params, system_options) -> ParticleTracerLattice:
    return make_injector_lattice_any(injector_params, options=system_options)


def make_ring_lattice(ring_params, ring_version: str, system_options: dict) -> ParticleTracerLattice:
    return ring_models[ring_version].make_ring_lattice(ring_params, options=system_options)


def make_surrogate_ring_for_injector(injector_params, ring_version, system_options: dict) -> ParticleTracerLattice:
    assert ring_version in ('1', '2')
    return make_ring_surrogate_for_injector(injector_params, system_options)


def make_system_model(ring_params, injector_params, ring_version: str,
                      system_options: Optional[dict]) -> tuple[ParticleTracerLattice, ParticleTracerLattice]:
    lattice_ring = make_ring_lattice(ring_params, ring_version, system_options)
    lattice_injector = make_injector_lattice_any(injector_params, options=system_options)
    assert_combiners_are_same(lattice_injector, lattice_ring)
    return lattice_ring, lattice_injector


def make_optimal_ring_and_injector_lattice(ring_version: str) -> tuple[ParticleTracerLattice, ParticleTracerLattice]:
    ring_params_optimal = get_optimal_ring_params(ring_version)
    injector_params_optimal = get_optimal_injector_params()
    return make_system_model(ring_params_optimal, injector_params_optimal, ring_version, None)
