from storageRingModeler import StorageRingModel
import numpy as np
from shapely.geometry import LineString
from ParticleTracerLatticeClass import ParticleTracerLattice
from helperTools import iscloseAll
from math import isclose
import os


def test_Modeler():
    PTL_Ring = ParticleTracerLattice()
    PTL_Ring.add_Halbach_Lens_Sim(.02, .1)
    PTL_Ring.add_Drift(.02)
    PTL_Ring.add_Combiner_Sim_Lens(.15, .03)
    PTL_Ring.add_Drift(.1)
    PTL_Ring.end_Lattice(constrain=False)

    PTL_Injector = ParticleTracerLattice(latticeType='injector')
    PTL_Injector.add_Drift(.1)
    PTL_Injector.add_Halbach_Lens_Sim(.01, .1)
    PTL_Injector.add_Drift(.1)
    PTL_Injector.add_Halbach_Lens_Sim(.01, .1)
    PTL_Injector.add_Drift(.12)
    PTL_Injector.add_Combiner_Sim_Lens(.15, .03)
    PTL_Injector.end_Lattice()

    model = StorageRingModel(PTL_Ring, PTL_Injector)

    assert model.floor_Plan_Cost() == 0  # no overlap between lenses
    model.swarmInjectorInitial.particles = model.swarmInjectorInitial.particles[:500]
    swarmInjectorTraced = model.swarmTracerInjector.trace_Swarm_Through_Lattice(
        model.swarmInjectorInitial.quick_Copy(), 1e-5, 1.0,
        fastMode=False, copySwarm=False, logPhaseSpaceCoords=True, accelerated=True)
    swarmRingInitial = model.transform_Swarm_From_Injector_Frame_To_Ring_Frame(swarmInjectorTraced,
                                                                               copyParticles=True, onlyUnclipped=False)
    swarmRingTraced = model.swarmTracerRing.trace_Swarm_Through_Lattice(swarmRingInitial, 1e-5, 1, fastMode=False,
                                                                        accelerated=True)

    lenses = model.get_Lenses_Before_Combiner_Ring()
    for particleInj, particleRing in zip(swarmInjectorTraced, swarmRingTraced):
        assert not (particleInj.clipped and not particleRing.clipped)  # this wouldn't make sense

        if particleInj.qArr is not None and len(particleInj.qArr) > 1:
            qInj_RingFrame = np.array([model.convert_Pos_Injector_Frame_To_Ring_Frame(q) for q in particleInj.qArr])
            line = LineString(qInj_RingFrame[:, :2])
            assert any(lens.SO_Outer.intersects(line) for lens in lenses) == \
                   model.does_Injector_Particle_Clip_On_Ring(particleInj)  # test that the
            # method of looking for particle clipping with shapely and without logging agrees with this more
            # straightforward method

            if particleRing.qArr is not None and len(particleRing.qArr) > 1:
                # assert particles are on top of each other at handoff between injector and ring, and that they are
                # very collinear by comparing angle injector with last two position steps to angle of ring by
                # momentum. This is valid because injector particle is in end of combiner with almost no field, and ring
                # particle is in drift region
                assert iscloseAll(particleRing.qArr[0], qInj_RingFrame[-1], 1e-12)
                slopeInjEnd = (qInj_RingFrame[-1][1] - qInj_RingFrame[-2][1]) / (
                            qInj_RingFrame[-1][0] - qInj_RingFrame[-2][0])
                slopeRing = particleRing.pArr[-1][1] / particleRing.pArr[-1][0]
                assert isclose(slopeRing, slopeInjEnd, abs_tol=5e-6)
    r2Inj = PTL_Injector.combiner.r2
    r2Ring = PTL_Ring.combiner.r2
    # test that the transform moved coordinates as expected
    assert iscloseAll(model.convert_Pos_Injector_Frame_To_Ring_Frame(r2Inj), r2Ring, 1e-12)
