from particle_tracer_lattice import ParticleTracerLattice
from lattice_elements.elements import HalbachLensSim

assert HalbachLensSim.fringe_frac_outer == 1.5

L_lens1=0.3020
L_lens2=0.1869

rp1=0.0242
rp2=0.0372

swarmShift_x = rp2 * HalbachLensSim.fringe_frac_outer - 0.03
swarmShift_y=-0.0114

def add_Kevin_Bumper_Elements(PTL: ParticleTracerLattice):  # creates an identical bumper, but does not end the lattice
    # note, somewhere in the code you will need to specify the magnet grades for the lens
    # the first magnet is N42 and the second N52
    intitialValues = PTL.use_standard_tube_OD
    assert PTL.initial_location[0] == 0.0 and PTL.initial_location[1] == 0.0
    PTL.add_halbach_lens_sim((rp1, rp2), L_lens1,
                             magnet_width=(0.0127, 0.01905),magnet_grade='N42')
    PTL.add_drift(0.3312, .04, input_tilt_angle=0.1107, output_tilt_angle=0.0167)
    PTL.add_halbach_lens_sim(0.0242, L_lens2, magnet_width=0.0127,magnet_grade='N52')
    PTL.add_drift(0.182, .04)
    PTL.use_standard_tube_OD = intitialValues
    return PTL

