from latticeModels.latticeModelUtilities import LockedDict

# outgassing rates, in units of torr L /s, from stainless steel with high temperature heat treating procedure
outgassing_rates = LockedDict({'H2': 5e-13})

#different species have different pumping speeds. From Duniway manual for ion pump lineup

# collision rate coefficents
rate_coefficients = LockedDict({'H2': 6e-9, 'He': 1.46e-9})

_small_ion_pump_speed=10.0
small_ion_pump_speed=LockedDict({'H2':2.0*_small_ion_pump_speed,'He':.1*_small_ion_pump_speed})

_big_ion_pump_speed=10.0
big_ion_pump_speed=LockedDict({'H2':2.0*_big_ion_pump_speed,'He':.1*_big_ion_pump_speed})
turbo_pump_speed=100.0
diffusion_pump_speed=2000.0

big_chamber_pressure=2e-6

