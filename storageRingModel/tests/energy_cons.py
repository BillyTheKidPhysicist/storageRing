import numpy as np

from helper_tools import is_close_all
from particle_class import Particle
from particle_tracer import ParticleTracer
from particle_tracer_lattice import ParticleTracerLattice


def test_energy_conservation():
    """Crude test that I get the same results"""
    lattice = ParticleTracerLattice(use_mag_errors=False, field_dens_mult=1, include_mag_cross_talk=True)
    lattice.add_halbach_lens_sim(.03, .15)
    lattice.add_drift(.05)
    lattice.add_combiner_sim_lens(.2, .05)
    lattice.add_drift(.1)
    lattice.end_lattice()

    pt = ParticleTracer(lattice)

    particle = Particle(qi=np.array([-1e-10, 15e-3, 5.0e-3]))
    particle = pt.trace(particle, 5e-7, 1.0, fast_mode=True)

    np.set_printoptions(precision=100)
    qf0 = np.array([-0.6502345445954198, 0.013901055149699137, -0.002979473557867293])
    assert is_close_all(particle.qf, qf0, 0.0)
