from ParticleClass import Particle,Swarm
from ParticleTracerClass import ParticleTracer
from ParticleTracerLatticeClass import ParticleTracerLattice
from constants import DEFAULT_ATOM_SPEED
from matrixMethodTracing import Lattice
from SwarmTracerClass import SwarmTracer
from helperTools import *
from math import isclose

def test_sim_and_matrix_similiarity():
    rp=.05
    L=.2
    L_drift=.1
    PTL=ParticleTracerLattice(lattice_type='injector')
    for _ in range(3):
        PTL.add_halbach_lens_sim(rp,L)
        PTL.add_drift(L_drift)
    PTL.end_lattice()
    lattice=Lattice()
    lattice.build_matrix_lattice_from_sim_lattice(PTL)
    st=SwarmTracer(PTL)


    yi_vals=2*(np.random.random(1000)-.5)*rp*.3
    pyi_vals=2*(np.random.random(1000)-.5)*5.0
    swarm_initial=Swarm()
    for yi,pyi in zip(yi_vals,pyi_vals):
        swarm_initial.add_new_particle(qi=np.array([-1e-10,yi,0.0]),pi=np.array([-DEFAULT_ATOM_SPEED,pyi,0.0]) )
    swarm=st.trace_swarm_through_lattice(swarm_initial,1e-5,1.0,copy_swarm=True)


    yf_vals=[]
    yf_ang_vals=[]
    for particle in swarm_initial:
        Xi=[particle.qi[1],particle.pi[1]/DEFAULT_ATOM_SPEED]
        yf,yf_ang=lattice.trace(Xi)
        yf_vals.append(yf)
        yf_ang_vals.append(yf_ang)
    yf_std_matrix,yf_ang_std_matrix=np.std(yf_vals),np.std(yf_ang_vals)


    yf_vals=[]
    yf_ang_vals=[]
    for particle in swarm:
        if not particle.clipped:
            yf,yf_ang=particle.qf[1],particle.pf[1]/DEFAULT_ATOM_SPEED
            yf_vals.append(yf)
            yf_ang_vals.append(yf_ang)
    np.std(yf_vals),np.std(yf_ang_vals)
    yf_std_sim,yf_ang_std_sim=np.std(yf_vals),np.std(yf_ang_vals)
    assert isclose(yf_std_sim,yf_std_matrix,rel_tol=.01)
    assert isclose(yf_std_sim,yf_std_matrix,rel_tol=.01)