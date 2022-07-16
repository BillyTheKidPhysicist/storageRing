from latticeModels.injectorModel_1 import make_injector_lattice
from latticeModels.latticeModelUtilities import assert_combiners_are_same
from latticeModels.ringModel_1 import make_ring_lattice


def make_system_model(ring_params, injector_params, options):
    lattice_ring = make_ring_lattice(ring_params, options=options)
    lattice_injector = make_injector_lattice(injector_params, options=options)
    assert_combiners_are_same(lattice_injector, lattice_ring)
    return lattice_ring, lattice_injector
