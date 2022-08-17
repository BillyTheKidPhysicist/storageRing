from Particle_tracer_lattice import ParticleTracerLattice
from lattice_elements.elements import HalbachLensSim

assert HalbachLensSim.fringe_frac_outer == 1.5

swarmShift_x = 0.0372 * 1.5 - 0.03
swarmShift_y=0.012

#todo: paramaterize the below with this
L_lens1=0.321
L_lens2=0.1564
L_gap=0.1274
rp1=0.0242
rp2=0.0183

def add_Kevin_Bumper_Elements(PTL: ParticleTracerLattice):  # creates an identical bumper, but does not end the lattice
    assert PTL.initial_location[0] == 0.0 and PTL.initial_location[1] == 0.0
    intitialValues = (PTL.use_standard_mag_size, PTL.use_standard_tube_OD)
    PTL.use_standard_mag_size, PTL.use_standard_tube_OD = (False, False)
    PTL.initial_location = (0.0, swarmShift_y)
    PTL.add_halbach_lens_sim((0.0242, 0.0372), 0.321, magnet_width=(0.0127, 0.01905))
    PTL.add_drift(0.1274, .04, input_tilt_angle=0.1803, output_tilt_angle=0.0753)
    PTL.add_halbach_lens_sim(0.0183, 0.1564)
    PTL.add_drift(0.085, .04)
    PTL.use_standard_mag_size, PTL.use_standard_tube_OD = intitialValues
    return PTL

# from ParticleTracerLatticeClass import ParticleTracerLattice
# PTL=ParticleTracerLattice()
# add_Kevin_Bumper_Elements(PTL)
# PTL.end_lattice()

#  the focus of the big collector magnet is 3 cm before the first magnet, so the initial swarm would look
#  more like


# def create_swarm(PTL, focus_offset, n):  # this should work
#     st = SwarmTracer(PTL)  # it seems like it needs a PTL to work
#     swarm = st.initialize_simulated_collector_focus_swarm(n)
#     for particle in swarm:
#         particle.obj_qi = particle.qi.copy()
#         t = (0.0372 * 1.5 + focus_offset - particle.qi[0]) / particle.pi[0]
#         particle.qi[0] = -10 ** -4
#         particle.qi[1] = particle.qi[1] + t * particle.pi[1]
#         particle.qi[2] = particle.qi[2] + t * particle.pi[2]
#     return swarm
#
#
# exmaple_bumper = add_Kevin_Bumper_Elements(ParticleTracerLattice(lattice_type='injector', magnet_grade='N52'))
# exmaple_bumper.end_lattice()
# example_swarm_3cm = create_swarm(exmaple_bumper, -0.03, 500)
