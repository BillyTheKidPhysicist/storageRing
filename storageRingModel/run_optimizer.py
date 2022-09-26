from optimizer import optimize
from lattice_models.ring_model_2 import injector_params_optimal

def main():
    # x0=(0.029939266743197166, 0.009366032835856075, 0.007394813254581377, 0.057462658388666564,
    #     0.2593652797436143, 0.47411631456318243, 0.1923582590838, 0.017106838745615468)
    optimize('both', 'global', '2', time_out_seconds=23 * 3600,save_population='mem_hist')


if __name__ == '__main__':
    main()
