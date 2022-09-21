from optimizer import optimize
from lattice_models.ring_model_2 import injector_params_optimal

def main():
    # optimize('both', 'global', '2', save_population='final_population', time_out_seconds=23 * 3600)
    # optimize('injector_Surrogate_Ring', 'global', '2', time_out_seconds=23 * 3600)
    optimize('both', 'global', '2', time_out_seconds=47 * 3600)


if __name__ == '__main__':
    main()
