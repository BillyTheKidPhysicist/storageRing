"""
This file holds constants that I use throughout the simulation. 

I use a different form of units where the mass of a lithium 7 atom is taken as 1 kg. This changes the value of the bohr
magneton. Every other constant I use is the same.

The new units are derived by:
F=ma=uB*grad(V) -> m*1kg*a=uB*grad(V) -> 1kg*a=(uB/m)*grad(V) ->1kg*a=uB' *grad(V)

where V is the magnetic potential energy of a lithium atom, uB is the Bohr magnetic, m is the mass of lithium
divided by 1kg, and uB' is the new Bohr magneton. uB' ~ 796.0
"""
MASS_LITHIUM_7: float=1.165034676538e-26 #mass of lithium 7, kg
MASS_HELIUM: float=6.64216e-27 #mass of lithium 7, kg
BHOR_MAGNETON: float=9.274009994e-24 #Bohr magneton, SI
BOLTZMANN_CONSTANT: float=1.38064852e-23 #Boltzman constant, SI
MAGNETIC_PERMEABILITY: float=1.25663706212*1e-6
SIMULATION_MAGNETON=BHOR_MAGNETON/MASS_LITHIUM_7