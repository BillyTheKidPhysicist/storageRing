from latticeModels.latticeModelUtilities import LockedDict

# outgassing rates, in units of torr L /s, from stainless steel with high temperature heat treating procedure
outgassing_rates = LockedDict({'H2': 5e-13})

#different species have different pumping speeds. From Duniway manual for ion pump lineup
ion_pump_speed_factors=LockedDict({'H2': 2.0, 'He':.1})

# collision rate coefficents
rate_coefficients = LockedDict({'H2': 6e-9, 'He': 1.46e-9})
