"""
This file holds constants that I use throughout the simulation. 

I use a different form of units where the mass of a lithium 7 atom is taken as 1 kg. This changes the value of the bohr
magneton. Every other constant is the same
"""
MASS_LITHIUM_7=1.165034676538e-26 #mass of lithium 7, kg
BHOR_MAGNETON=9.274009994e-24 #Bohr magneton, SI
BOLTZMANN_CONSTANT=1.38064852e-23 #Boltzman constant, SI
MAGNETIC_PERMEABILITY=1.25663706212*1e-6
SIMULATION_MAGNETON=BHOR_MAGNETON/MASS_LITHIUM_7