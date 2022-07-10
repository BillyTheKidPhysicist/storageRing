def find_Optimal_Offset_Factor(self, rp: float, rb: float, Lm: float, parallel: bool = False) -> float:
    # How far exactly to offset the bending segment from linear segments is exact for an ideal bender, but for an
    # imperfect segmented bender it needs to be optimized.
    raise NotImplementedError  # this doesn't do much with my updated approach, and needs to be reframed in terms
    # of shifting the particle over to improve performance. It's also bloated. ALso, it's not accurate
    assert rp < rb / 2.0  # geometry argument, and common mistake
    num_magnetsHalfBend = int(np.pi * rb / Lm)
    # todo: this should be self I think
    lattice_ring = ParticleTracerLattice(lattice_type='injector')
    lattice_ring.add_drift(.05)
    lattice_ring.add_segmented_halbach_bender(Lm, rp, num_magnetsHalfBend, rb)
    lattice_ring.end_lattice(enforceClosedLattice=False, constrain=False)

    def errorFunc(offset):
        h = 5e-6
        particle = Particle(qi=np.array([-1e-10, offset, 0.0]), pi=np.array([-self.speed_nominal, 0.0, 0.0]))
        particleTracer = ParticleTracer(lattice_ring)
        particle = particleTracer.trace(particle, h, 1.0, use_fast_mode=False)
        qo_arr = particle.qo_arr
        particleAngEl = np.arctan2(qo_arr[-1][1],
                                   qo_arr[-1][0])  # very close to zero, or negative, if particle made it to
        # end
        if particleAngEl < .01:
            error = np.std(1e6 * particle.qo_arr[:, 1])
            return error
        else:
            return np.nan

    output_offsetFactArr = np.linspace(-3e-3, 3e-3, 100)

    if parallel:
        njobs = -1
    else:
        njobs = 1
    errorArr = np.asarray(
        Parallel(n_jobs=njobs)(delayed(errorFunc)(output_offset) for output_offset in output_offsetFactArr))
    rOffsetOptimal = self._find_rOptimal(output_offsetFactArr, errorArr)
    return rOffsetOptimal

def _find_rOptimal(self, output_offsetFactArr: np.ndarray, errorArr: np.ndarray) -> Optional[float]:
    test = errorArr.copy()[1:]
    test = np.append(test, errorArr[0])
    numValidSolutions = np.sum(~np.isnan(errorArr))
    numNanInitial = np.sum(np.isnan(errorArr))
    numNanAfter = np.sum(np.isnan(test + errorArr))
    valid = True
    if numNanAfter - numNanInitial > 1:
        valid = False
    elif numValidSolutions < 4:
        valid = False
    elif numNanInitial > 0:
        if (np.isnan(errorArr[0]) == False and np.isnan(errorArr[-1]) == False):
            valid = False
    if valid == False:
        return None
    # trim out invalid points
    output_offsetFactArr = output_offsetFactArr[~np.isnan(errorArr)]
    errorArr = errorArr[~np.isnan(errorArr)]
    fit = spi.RBFInterpolator(output_offsetFactArr[:, np.newaxis], errorArr)
    output_offsetFactArrDense = np.linspace(output_offsetFactArr[0], output_offsetFactArr[-1], 10_000)
    errorArrDense = fit(output_offsetFactArrDense[:, np.newaxis])
    rOptimal = output_offsetFactArrDense[np.argmin(errorArrDense)]
    rMinDistFromEdge = np.min(output_offsetFactArr[1:] - output_offsetFactArr[:-1]) / 4
    if rOptimal > output_offsetFactArr[-1] - rMinDistFromEdge or rOptimal < output_offsetFactArr[
        0] + rMinDistFromEdge:
        # print('Invalid solution, rMin very near edge. ')
        return None
    return rOptimal