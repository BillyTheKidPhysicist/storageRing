from storageRingModeler import StorageRingModel
import storageRingModeler
import numpy as np
from shapely.geometry import LineString
from ParticleTracerLatticeClass import ParticleTracerLattice
from helperTools import is_close_all
from math import isclose


def test_Modeler():
    lattice_ring = ParticleTracerLattice()
    lattice_ring.add_halbach_lens_sim(.02, .1)
    lattice_ring.add_drift(.02)
    lattice_ring.add_combiner_sim_lens(.15, .03)
    lattice_ring.add_drift(.1)
    lattice_ring.end_lattice(constrain=False)

    lattice_injector = ParticleTracerLattice(lattice_type='injector')
    lattice_injector.add_drift(.1)
    lattice_injector.add_halbach_lens_sim(.01, .1)
    lattice_injector.add_drift(.1)
    lattice_injector.add_halbach_lens_sim(.01, .1)
    lattice_injector.add_drift(.12)
    lattice_injector.add_combiner_sim_lens(.15, .03)
    lattice_injector.end_lattice()
    storageRingModeler.ELEMENTS_MODE_MATCHER=tuple([type(el) for el in lattice_injector])
    model = StorageRingModel(lattice_ring, lattice_injector)

    assert model.floor_plan_cost() == 0  # no overlap between lenses
    model.swarm_injector_initial.particles = model.swarm_injector_initial.particles[:500]
    swarm_injector_traced = model.swarm_tracer_injector.trace_swarm_through_lattice(
        model.swarm_injector_initial.quick_copy(), 1e-5, 1.0,
        use_fast_mode=False, copy_swarm=False, log_el_phase_space_coords=True, accelerated=True)
    swarmRingInitial = model.transform_swarm_from_injector_to_ring_frame(swarm_injector_traced,
                                                                               copy_particles=True)
    swarmRingTraced = model.swarm_tracer_ring.trace_swarm_through_lattice(swarmRingInitial, 1e-5, 1, use_fast_mode=False,
                                                                        accelerated=True)

    clippable_elements = model.clippable_elements_in_ring()
    for particle_injector, particle_ring in zip(swarm_injector_traced, swarmRingTraced):
        assert not (particle_injector.clipped and not particle_ring.clipped)  # this wouldn't make sense

        if particle_injector.q_vals is not None and len(particle_injector.q_vals) > 1:
            qInj_RingFrame = np.array([model.convert_position_injector_to_ring_frame(q) for q in particle_injector.q_vals])
            line = LineString(qInj_RingFrame[:, :2])
            assert any(lens.SO_outer.intersects(line) for lens in clippable_elements) == \
                   model.does_ring_clip_injector_particle(particle_injector)  # test that the
            # method of looking for particle clipping with shapely and without logging agrees with this more
            # straightforward method

            if particle_ring.q_vals is not None and len(particle_ring.q_vals) > 1:
                # assert particles are on top of each other at handoff between injector and ring, and that they are
                # very collinear by comparing angle injector with last two position steps to angle of ring by
                # momentum. This is valid because injector particle is in end of combiner with almost no field, and ring
                # particle is in drift region
                assert is_close_all(particle_ring.q_vals[0], qInj_RingFrame[-1], 1e-12)
                slopeInjEnd = (qInj_RingFrame[-1][1] - qInj_RingFrame[-2][1]) / (
                            qInj_RingFrame[-1][0] - qInj_RingFrame[-2][0])
                slopeRing = particle_ring.p_vals[-1][1] / particle_ring.p_vals[-1][0]
                assert isclose(slopeRing, slopeInjEnd, abs_tol=5e-6)
    r2Inj = lattice_injector.combiner.r2
    r2Ring = lattice_ring.combiner.r2
    # test that the transform moved coordinates as expected
    assert is_close_all(model.convert_position_injector_to_ring_frame(r2Inj), r2Ring, 1e-12)