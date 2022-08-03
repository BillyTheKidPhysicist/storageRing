import numpy as np

from ParticleClass import Particle
from ParticleTracerClass import ParticleTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from helperTools import is_close_all
import matplotlib.pyplot as plt

def poop_energy_conservation():
    """Crude test that I get the same results"""
    lattice = ParticleTracerLattice(use_mag_errors=False, field_dens_mult=1, include_mag_cross_talk=True)
    lattice.add_halbach_lens_sim(.03, .15)
    # lattice.add_halbach_lens_sim(.03, .15)
    lattice.add_drift(.1)
    lattice.add_halbach_lens_sim(.03, .15)
    lattice.add_drift(.1)
    lattice.add_combiner_sim_lens(.2, .05)
    lattice.add_halbach_lens_sim(.03, .15)
    lattice.add_drift(.1)
    lattice.end_lattice()

    pt = ParticleTracer(lattice)

    particle = Particle(qi=np.array([-1e-10, 15e-3, 5.0e-3]))
    particle = pt.trace(particle, 5e-7, 1.0, fast_mode=False)

    np.set_printoptions(precision=100)
    qf0 = np.array([-0.8297543499818717  , -0.002187290013948691,-0.004025337437755215])
    print(repr(particle.qf))
    assert is_close_all(particle.qf, qf0, 0.0)
    # print(particle.E_vals[0]-particle.E_vals[-1]) #5.517820803561335

    # plt.semilogy(-particle.q_vals[:,0],particle.V_vals)
    # plt.axvline(x=.15,linestyle=':',c='r')
    # plt.show()
poop_energy_conservation()