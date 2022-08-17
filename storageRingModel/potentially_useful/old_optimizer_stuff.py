def catch_Optimizer_Errors(self, tuningBounds: list_array_tuple, tuningElementIndices: list_array_tuple,
                           tuningChoice: str, whichKnobs: str) -> None:
    if max(tuningElementIndices) >= len(self.lattice_ring.el_list) - 1: raise Exception(
        "element indices out of bounds")
    if len(tuningBounds) != len(tuningElementIndices): raise Exception(
        "Bounds do not match number of tuned elements")
    combinerRing, combinerLat = self.lattice_ring.combiner, self.lattice_injector.combiner
    if not (combinerRing.Lm == combinerLat.Lm and combinerRing.ap == combinerLat.ap and combinerRing.output_offset ==
            combinerLat.output_offset):
        raise Exception('Combiners are different between the two lattices')
    injectorTuningElements = [self.lattice_injector.el_list[index] for index in self.injectTuneElIndices]
    if not all(isinstance(el, Drift) for el in injectorTuningElements):
        raise Exception("injector tuning elements must be drift region")
    if tuningChoice == 'field':
        for el_index in tuningElementIndices:
            el = self.lattice_ring.el_list[el_index]
            if (isinstance(el, LensIdeal) and isinstance(el, HalbachLensSim)) != True:
                raise Exception("For field tuning elements must be LensIdeal or HalbachLensSim")
    elif tuningChoice == 'spacing':
        for el_index in tuningElementIndices:
            el_before, el_after = self.lattice_ring.get_element_before_and_after(self.lattice_ring.el_list[el_index])
            tunableLength = (el_before.L + el_after.L) - 2 * self.minElementLength
            if (isinstance(el_before, Drift) and isinstance(el_after, Drift)) != True:
                raise Exception("For spacing tuning neighboring elements must be Drift elements")
            if tunableLength < 0.0:
                raise Exception("Tunable elements are too short for length tuning. Min total length is "
                                + str(2 * self.minElementLength))
    else:
        raise Exception('No proper tuning choice provided')
    if whichKnobs not in ('all', 'ring'):
        raise Exception('Knobs must be either \'all\' (full system) or \'ring\' (only storage ring)')


def initialize_Optimizer(self, tuningElementIndices: list_array_tuple, tuningChoice: str, whichKnobs: str,
                         ringTuningBounds: list_array_tuple, injectorTuningBounds: list_array_tuple) -> None:
    assert tuningChoice in ('spacing', 'field') and whichKnobs in ('all', 'ring')
    assert all(isinstance(arg, Iterable) for arg in (tuningElementIndices, ringTuningBounds, injectorTuningBounds))
    self.whichKnobs = whichKnobs
    self.tuningBounds = ringTuningBounds.copy()
    if self.whichKnobs == 'all':
        self.tuningBounds.extend(injectorTuningBounds)
    self.tunedElementList = [self.lattice_ring.el_list[index] for index in tuningElementIndices]
    self.tuningChoice = tuningChoice
    if self.sameSeedForSearch == True:
        np.random.seed(42)


def test_Lattice_Stability(self, ringTuningBounds: list_array_tuple, injectorTuningBounds: list_array_tuple,
                           numEdgePoints: int = 30, parallel: bool = False) -> bool:
    assert len(ringTuningBounds) == 2 and len(injectorTuningBounds) == 2
    ringKnob1Arr = np.linspace(ringTuningBounds[0][0], ringTuningBounds[0][1], numEdgePoints)
    ringKnob2Arr = np.linspace(ringTuningBounds[1][0], ringTuningBounds[1][1], numEdgePoints)
    injectorKnob1Arr_Constant = ringTuningBounds[0][1] * np.ones(numEdgePoints ** 2)
    injectorKnobA2rr_Constant = ringTuningBounds[1][1] * np.ones(numEdgePoints ** 2)
    testCoords = np.asarray(np.meshgrid(ringKnob1Arr, ringKnob2Arr)).T.reshape(-1, 2)
    if self.whichKnobs == 'all':
        testCoords = np.column_stack((testCoords, injectorKnob1Arr_Constant, injectorKnobA2rr_Constant))
    if parallel == False:
        stabilityList = [self.is_Stable(coords) for coords in testCoords]
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            stabilityList = pool.map(self.is_Stable, testCoords)
    assert len(stabilityList) == numEdgePoints ** 2
    if sum(stabilityList) == 0:
        return False
    else:
        return True


def _fast_Minimize(self):
    # less accurate method that minimizes with a smaller surrogate swarm then uses the full swarm for the final
    # value
    useSurrogate, use_energy_correction = [True, False]
    sol_Surrogate = spo.differential_evolution(self.mode_Match_Cost, self.tuningBounds, tol=self.tolerance,
                                               polish=False, args=(useSurrogate, use_energy_correction),
                                               maxiter=self.max_evals // (
                                                       self.optimalPopSize * len(self.tuningBounds)),
                                               popsize=self.optimalPopSize, init='halton')
    return sol_Surrogate


def _accurate_Minimize(self):
    # start first by quickly randomly searching with a surrogate swarm.
    randomSamplePoints = 128
    use_energy_correction = False  # leave this off here, apply later once
    useSurrogateRoughPass = True
    samples = skopt.sampler.Sobol().generate(self.tuningBounds, randomSamplePoints)
    vals = [self.mode_Match_Cost(sample, useSurrogateRoughPass, use_energy_correction) for sample in samples]
    XInitial = samples[np.argmin(vals)]
    useSurrogateScipyOptimer = False
    sol = spo.differential_evolution(self.mode_Match_Cost, self.tuningBounds, polish=False, x0=XInitial,
                                     tol=self.tolerance,
                                     maxiter=self.max_evals // (self.optimalPopSize * len(self.tuningBounds)),
                                     args=(useSurrogateScipyOptimer, use_energy_correction), popsize=self.optimalPopSize,
                                     init='halton')
    return sol


def _minimize(self) -> Solution:
    if self.fastSolver == True:
        scipySol = self._fast_Minimize()
    else:
        scipySol = self._accurate_Minimize()
    useSurrogate, use_energy_correction = [False, True]
    cost_Most_Accurate = self.mode_Match_Cost(scipySol.x, useSurrogate, use_energy_correction)
    sol = Solution()
    sol.scipyMessage = scipySol.message
    sol.cost = cost_Most_Accurate
    optimalConfig = scipySol.x
    sol.stable = True
    sol.xRing_TunedParams2 = optimalConfig[:2]
    if self.whichKnobs == 'all':
        sol.xInjector_TunedParams = optimalConfig[2:]
    sol.invalidInjector = False
    sol.invalidRing = False
    return sol


def optimize(self, tuningElementIndices, ringTuningBounds=None, injectorTuningBounds=None, tuningChoice='spacing'
             , whichKnobs='all', parallel=False, fastSolver=False) -> Solution:
    self.fastSolver = fastSolver
    if ringTuningBounds is None:
        ringTuningBounds = [(.2, .8), (.2, .8)]
    if injectorTuningBounds is None:
        injectorTuningBounds = [(.1, .4), (.1, .4)]
    self.catch_Optimizer_Errors(ringTuningBounds, tuningElementIndices, tuningChoice, whichKnobs)
    self.initialize_Optimizer(tuningElementIndices, tuningChoice, whichKnobs, ringTuningBounds,
                              injectorTuningBounds)
    if self.test_Lattice_Stability(ringTuningBounds, injectorTuningBounds, parallel=parallel) == False:
        sol = Solution()
        sol.flux_mult = 0.0
        sol.cost = 1.0
        sol.stable = False
        return sol
    sol = self._minimize()
    return sol
def update_Ring_And_Injector(self, X: Optional[list_array_tuple]):
    if X is None:
        pass
    elif self.whichKnobs == 'all':
        assert len(X) == 4
        XRing, XInjector = X[:2], X[2:]
        self.update_Ring_Lattice(XRing)
        self.update_Injector_Lattice(XInjector)
    elif self.whichKnobs == 'ring':
        assert len(X) == 2
        self.update_Ring_Lattice(X)
    else:
        raise ValueError
def update_Injector_Lattice(self, X: list_array_tuple):
    # modify lengths of drift regions in injector
    raise NotImplementedError
    assert len(X) == 2
    assert X[0] > 0.0 and X[1] > 0.0
    self.lattice_injector.el_list[self.injectTuneElIndices[0]].set_length(X[0])
    self.lattice_injector.el_list[self.injectTuneElIndices[1]].set_length(X[1])
    self.lattice_injector.build_lattice()
def update_Ring_Lattice(self, X: list_array_tuple) -> None:
    assert len(X) == 2
    if self.tuningChoice == 'field':
        self.update_Ring_Field_Values(X)
    elif self.tuningChoice == 'spacing':
        self.update_Ring_Spacing(X)
    else:
        raise ValueError
def update_Ring_Field_Values(self, X: list_array_tuple) -> None:
    raise NotImplementedError
    for el, arg in zip(self.tunedElementList, X):
        el.field_fact = arg
def update_Ring_Spacing(self, X: list_array_tuple) -> None:
    raise NotImplementedError
    for el_center, spaceFracElBefore in zip(self.tunedElementList, X):
        self.move_Element_Longitudinally(el_center, spaceFracElBefore)
    self.lattice_ring.build_lattice()
def move_Element_Longitudinally(self, el_center: Element, spaceFracElBefore: float) -> None:
    assert 0 <= spaceFracElBefore <= 1.0
    el_before, el_after = self.lattice_ring.get_element_before_and_after(el_center)
    assert isinstance(el_before, Drift) and isinstance(el_after, Drift)
    totalBorderingElLength = el_before.L + el_after.L
    tunableLength = (el_before.L + el_after.L) - 2 * self.minElementLength
    LBefore = spaceFracElBefore * tunableLength + self.minElementLength
    LAfter = totalBorderingElLength - LBefore
    el_before.set_length(LBefore)
    el_after.set_length(LAfter)
def is_Stable(self, X: list_array_tuple, minRevsToTest=5.0) -> bool:
    self.update_Ring_And_Injector(X)
    maxInitialTransversePos = 1e-3
    T_Max = 1.1 * minRevsToTest * self.lattice_ring.total_length / self.lattice_ring.speed_nominal
    swarmTestInitial = self.swarm_tracer_ring.initialize_stablity_testing_swarm(maxInitialTransversePos)
    swarmTestAtCombiner = self.swarm_tracer_ring.move_swarm_to_combiner_output(swarmTestInitial)
    swarmTestTraced = self.swarm_tracer_ring.trace_swarm_through_lattice(swarmTestAtCombiner, self.h,
                                                                       T_Max, accelerated=False, use_fast_mode=True,
                                                                       use_collisions=self.use_collisions)
    stable = False
    for particle in swarmTestTraced:
        if particle.revolutions > minRevsToTest:  # any stable particle, consider lattice stable
            stable = True
    return stable