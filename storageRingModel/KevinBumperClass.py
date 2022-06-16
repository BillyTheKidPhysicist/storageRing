import numpy as np
from ParticleTracerLatticeClass import ParticleTracerLattice
from SwarmTracerClass import SwarmTracer
from helperTools import *
from shapely.geometry import Polygon, MultiPolygon
from sklearn.neighbors import NearestNeighbors
from latticeElements.elements import  HalbachLensSim

assert HalbachLensSim.fringeFracOuter==1.5

#  made 6/13/2022, after two rounds of optimizing.
#  use KevinBumper.clone_bumper() to create an identical bumper without an end, so more elements can be added.
#  (or use KevinBumperOpen, the same thing but maybe more convenient)

#  if you want to add the KevinBumper to something else, you would need to consider the offset drift before it.
#  if you want the bumper to be point left and start at zero, call KevinBumper.clone_bumper(angle=np.pi, long_off=0)

#  the focus should be at the variable np.abs(KevinBumper.start) above the center of the entrance of the first lens.

#  be careful when tracing the swarm in the cloned bumper, you would need to flip the velocities and project the
#  particles backwards a bit. look at KevinBumper.trace_simulated_focus for the flip


mu = 9.274 * 10 ** -24  # J/T
m_li7 = 1.165 * 10 ** -26  # kg
v = 210  # m/s
good_field_1 = 0.8  # percent of the radius of the first magnet that is usable
good_field_2 = 0.9


def y0_max(r, b_max):
    numerator = good_field_1 ** 2 * mu * b_max - 0.5 * m_li7 * (v * 0.0521) ** 2
    return r * np.sqrt(numerator / (mu * b_max)) - 0.0075


r_norm = 0.05  # constants I use so that the gradient descent parameters are all on similar scales
l_norm = 0.5
d_norm = 0.5
L_norm = 0.05
phi_norm = 0.2

opt_norm = [0.53227244, 0.50382043, 0.78162133, -0.64865529, 0.34259073, 0.41882729, 0.17864541, 0.40304075]
opt_p = [opt_norm[0] * r_norm, opt_norm[1] * l_norm, opt_norm[2] * d_norm, opt_norm[3] * L_norm - 0.00045,
         opt_norm[4] * phi_norm + 0.020, opt_norm[5] * r_norm, opt_norm[6] * l_norm, opt_norm[7] * d_norm]
r1p1, l1, d1, L, phi, r2, l2, d2=opt_p
start = y0_max(r1p1, 0.9)
magwidth1 = r1p1 * np.tan(2 * np.pi / 24) * 2
r1p2 = r1p1 + magwidth1
magwidth2 = r1p2 * np.tan(2 * np.pi / 24) * 2
swarmShift_x=1.5*r1p2
# KevinBumper = Bumper(opt_p[0], opt_p[1], opt_p[2], opt_p[3], opt_p[4], opt_p[5], opt_p[6], opt_p[7], 0,
#                      leftwards=True, long_off=0, trace=False)
# KevinBumperOpen = KevinBumper.clone_bumper()

def add_Kevin_Bumper_Elements(PTL):  # creates an identical bumper, but does not end the lattice
    assert PTL.initialLocation[0]==0.0 and PTL.initialLocation[1]==0.0
    delta_x = r2 * 1.5 * np.cos(phi)
    delta_y = r2 * 1.5 * np.sin(phi)
    a1 = np.tan((L - delta_y) / (d1 - r1p2 * 1.5 - delta_x))
    a2 = a1 + phi
    d_fix = np.sqrt((L - delta_y) ** 2 + (d1 - r1p2 * 1.5 - delta_x) ** 2)
    l1_plus_fringe = l1 + r1p2 * 3
    l2_plus_fringe = l2 + r2 * 3


    PTL.initialLocation=(0.0, start)
    PTL.add_Halbach_Lens_Sim((r1p1, r1p2), l1_plus_fringe,
                             magnetWidth=(magwidth1, magwidth2))
    PTL.add_Drift(d_fix, .04, inputTiltAngle=-a1, outputTiltAngle=-a2)
    PTL.add_Halbach_Lens_Sim(r2, l2_plus_fringe)
    PTL.add_Drift(d2, .04)
    return PTL