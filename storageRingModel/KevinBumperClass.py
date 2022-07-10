from latticeElements.elements import HalbachLensSim
from SwarmTracerClass import SwarmTracer
from ParticleTracerLatticeClass import ParticleTracerLattice

assert HalbachLensSim.fringe_frac_outer == 1.5

swarmShift_x=0.0372 * 1.5 - 0.03
def add_Kevin_Bumper_Elements(PTL: ParticleTracerLattice):  # creates an identical bumper, but does not end the lattice
    assert PTL.initialLocation[0] == 0.0 and PTL.initialLocation[1] == 0.0
    intitialValues=(PTL.use_standard_mag_size,PTL.use_standard_tube_OD)
    PTL.use_standard_mag_size, PTL.use_standard_tube_OD = (False,False)
    PTL.initialLocation = (0.0, 0.012)
    PTL.add_Halbach_Lens_Sim((0.0242, 0.0372), 0.321, magnetWidth=(0.0127, 0.01905))
    PTL.add_Drift(0.1274, .04, inputTiltAngle=0.1803, outputTiltAngle=0.0753)
    PTL.add_Halbach_Lens_Sim(0.0183, 0.1564)
    PTL.add_Drift(0.085, .04)
    PTL.use_standard_mag_size, PTL.use_standard_tube_OD = intitialValues
    return PTL

# from ParticleTracerLatticeClass import ParticleTracerLattice
# PTL=ParticleTracerLattice()
# add_Kevin_Bumper_Elements(PTL)
# PTL.end_Lattice()

#  the focus of the big collector magnet is 3 cm before the first magnet, so the initial swarm would look
#  more like


# def create_swarm(PTL, focus_offset, n):  # this should work
#     st = SwarmTracer(PTL)  # it seems like it needs a PTL to work
#     swarm = st.initialize_Simulated_Collector_Focus_Swarm(n)
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
# exmaple_bumper.end_Lattice()
# example_swarm_3cm = create_swarm(exmaple_bumper, -0.03, 500)
