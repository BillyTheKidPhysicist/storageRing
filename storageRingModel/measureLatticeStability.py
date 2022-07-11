import os
from typing import Optional

import numpy as np
from ringOptimizer import solution_From_Lattice

from ParticleTracerLatticeClass import ParticleTracerLattice
from helperTools import parallel_evaluate
from latticeElements.elements import CombinerHalbachLensSim, CombinerIdeal, CombinerSim, LensIdeal, HalbachLensSim
from latticeModels import make_Ring_And_Injector_Version3

combinerTypes = (CombinerHalbachLensSim, CombinerIdeal, CombinerSim)


class StabilityAnalyzer:
    def __init__(self, paramsOptimal: np.ndarray, alignmentTol: float = 1e-3,
                 machineTolerance: float = 250e-6):
        """
        Analyze stability of ring and injector system. Elements are perturbed by random amplitudes by sampling from
        a gaussian

        :param paramsOptimal: Optimal parameters of a lattice solution.
        :param alignmentTol: Maximum displacement from ideal trajectory perpindicular to vacuum tube in one direction.
        This is our accepted alignmentTol
        """

        self.paramsOptimal = paramsOptimal
        self.alignmentTol = alignmentTol
        self.machineTolerance = machineTolerance
        self.jitterableElements = (CombinerHalbachLensSim, LensIdeal, HalbachLensSim)

    def generate_Ring_And_Injector_Lattice(self, use_mag_errors: bool,
                                           combiner_seed: int = None) \
            -> tuple[ParticleTracerLattice, ParticleTracerLattice, np.ndarray]:
        # params=self.apply_Machining_Errors(self.paramsOptimal) if useMachineError==True else self.paramsOptimal
        params = self.paramsOptimal
        lattice_ring, lattice_injector = make_Ring_And_Injector_Version3(params, use_mag_errors=use_mag_errors,
                                                                         combiner_seed=combiner_seed)
        # if misalign:
        #     self.jitter_System(lattice_ring,lattice_injector)
        return lattice_ring, lattice_injector, params

    def apply_Machining_Errors(self, params: np.ndarray) -> np.ndarray:
        deltaParams = 2 * (np.random.random_sample(params.shape) - .5) * self.machineTolerance
        params_Error = params + deltaParams
        return params_Error

    def make_Jitter_Amplitudes(self, element, randomOverRide: Optional[tuple]) -> tuple[float, ...]:
        angle_max = np.arctan(self.alignmentTol / element.L)
        randomNum = np.random.random_sample() if randomOverRide is None else randomOverRide[0]
        random4Nums = np.random.random_sample(4) if randomOverRide is None else randomOverRide[1]
        fractionAngle, fractionShift = randomNum, 1 - randomNum
        angleAmp, shiftAmp = angle_max * fractionAngle, self.alignmentTol * fractionShift
        angleAmp = angleAmp / np.sqrt(2)  # consider both dimensions being misaligned
        shiftAmp = shiftAmp / np.sqrt(2)  # consider both dimensions being misaligned
        rotAngleY = 2 * (random4Nums[0] - .5) * angleAmp
        rotAngleZ = 2 * (random4Nums[1] - .5) * angleAmp
        shift_y = 2 * (random4Nums[2] - .5) * shiftAmp
        shift_z = 2 * (random4Nums[3] - .5) * shiftAmp
        return shift_y, shift_z, rotAngleY, rotAngleZ

    def jitter_Lattice(self, PTL, combinerRandomOverride):
        for el in PTL:
            if any(validElementType == type(el) for validElementType in self.jitterableElements):
                if any(type(el) == elType for elType in combinerTypes):
                    shift_y, shift_z, rot_angle_y, rot_angle_z = self.make_Jitter_Amplitudes(el,
                                                                                             randomOverRide=combinerRandomOverride)
                else:
                    shift_y, shift_z, rot_angle_y, rot_angle_z = self.make_Jitter_Amplitudes(el, None)
                el.perturb_element(shift_y, shift_z, rot_angle_y, rot_angle_z)

    def jitter_System(self, lattice_ring: ParticleTracerLattice, lattice_injector: ParticleTracerLattice) -> None:
        combinerRandomOverride = (np.random.random_sample(), np.random.random_sample(4))
        self.jitter_Lattice(lattice_ring, combinerRandomOverride)
        self.jitter_Lattice(lattice_injector, combinerRandomOverride)

    def dejitter_System(self, lattice_ring, lattice_injector):
        # todo: possibly useless
        tolerance0 = self.alignmentTol
        self.alignmentTol = 0.0
        self.jitter_Lattice(lattice_ring, None)
        self.jitter_Lattice(lattice_injector, None)
        self.alignmentTol = tolerance0

    def inject_And_Trace_Through_Ring(self, use_mag_errors: bool, combiner_seed: int = None):
        lattice_ring, lattice_injector, params = self.generate_Ring_And_Injector_Lattice(use_mag_errors,
                                                                                         combiner_seed=combiner_seed)
        sol = solution_From_Lattice(lattice_ring, lattice_injector)
        sol.params = params
        return sol

    def measure_Sensitivity(self) -> None:

        # todo: now that i'm doing much more than just jittering elements, I should refactor this. Previously
        # it worked by reusing the lattice over and over again. Maybe I still want that to be a feature? Or maybe
        # that's dumb

        # todo: I need to figure out what all this is doing

        def flux_Multiplication(i):
            np.random.seed(i)
            if i == 0:
                sol = self.inject_And_Trace_Through_Ring(False)
            else:
                sol = self.inject_And_Trace_Through_Ring(True, combiner_seed=i)
            # print('seed',i)
            print(i, sol.flux_mult)
            return sol.cost, sol.flux_mult

        indices = list(range(1, 17))
        results = parallel_evaluate(flux_Multiplication, indices, processes=8, results_as_arr=True)
        os.system("say 'done bitch!'")
        # print('cost',np.mean(results[:,0]),np.std(results[:,0]))
        print('flux', np.mean(results[:, 1]), np.std(results[:, 1]))
        # plt.hist(results[:,1])
        # plt.show()
        print(repr(results[:, 1]))
        # np.savetxt('data',results)
        # print(repr(results))
        # _cost(1)


x = np.array([0.02477938, 0.01079024, 0.04059919, 0.010042, 0.07175166, 0.51208528])
sa = StabilityAnalyzer(x)
sa.measure_Sensitivity()
