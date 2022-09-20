"""
Contains surrogate model of the ring. This is just the combiner with a lens before for enforcing geometric constraints,
and a lens after for testing particle survival. Used when optimization occurs in a two step process to reduce
compute time
"""
from lattice_models.lattice_model_functions import check_and_add_default_values, add_combiner_and_OP_ring
from lattice_models.lattice_model_parameters import system_constants, atom_characteristics
from lattice_models.ring_model_1 import ring_constants
from lattice_models.utilities import LockedDict
from particle_tracer_lattice import ParticleTracerLattice

surrogate_params = LockedDict({'rp_lens1': ring_constants['rp_lens1'],
                               'L_Lens1': .5,
                               'L_Lens2': .3})


def make_ring_surrogate_for_injector(injector_params: dict,
                                     options: dict = None) -> ParticleTracerLattice:
    """Surrogate model of storage to aid in realism of independent injector optimizing. Benders are exluded. Model
    serves to represent geometric constraints between injector and ring, and to test that if particle that travel
    into combiner can make it to the next element. Since injector is optimized independent of ring, the parameters
    for the ring surrogate are typically educated guesses"""
    injector_params = LockedDict(injector_params)
    options = check_and_add_default_values(options)

    lattice = ParticleTracerLattice(design_speed=atom_characteristics["nominalDesignSpeed"],
                                    use_solenoid_field=options['use_solenoid_field'],
                                    use_standard_tube_OD=options['use_standard_tube_OD'])

    # ----lens before combiner
    lattice.add_halbach_lens_sim(surrogate_params['rp_lens1'], surrogate_params['L_Lens1'])

    # ---combiner + OP magnet-----
    add_combiner_and_OP_ring(lattice, system_constants['rp_combiner'], injector_params['Lm_combiner'],
                             injector_params['load_beam_offset'], ring_constants['rp_lens2'], options, 'Injection')

    # ---lens after combiner---

    lattice.add_halbach_lens_sim(ring_constants['rp_lens2'], surrogate_params['L_Lens2'])

    surrogate_params.assert_All_Entries_Accessed_And_Reset_Counter()
    lattice.end_lattice(constrain=False)
    return lattice
