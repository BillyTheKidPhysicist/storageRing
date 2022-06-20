def find_Optimal_Offset_Factor(self, rp: float, rb: float, Lm: float, parallel: bool = False) -> float:
    # How far exactly to offset the bending segment from linear segments is exact for an ideal bender, but for an
    # imperfect segmented bender it needs to be optimized.
    raise NotImplementedError  # this doesn't do much with my updated approach, and needs to be reframed in terms
    # of shifting the particle over to improve performance. It's also bloated. ALso, it's not accurate
    assert rp < rb / 2.0  # geometry argument, and common mistake
    numMagnetsHalfBend = int(np.pi * rb / Lm)
    # todo: this should be self I think
    PTL_Ring = ParticleTracerLattice(latticeType='injector')
    PTL_Ring.add_Drift(.05)
    PTL_Ring.add_Halbach_Bender_Sim_Segmented(Lm, rp, numMagnetsHalfBend, rb)
    PTL_Ring.end_Lattice(enforceClosedLattice=False, constrain=False)

    def errorFunc(offset):
        h = 5e-6
        particle = Particle(qi=np.array([-1e-10, offset, 0.0]), pi=np.array([-self.v0Nominal, 0.0, 0.0]))
        particleTracer = ParticleTracer(PTL_Ring)
        particle = particleTracer.trace(particle, h, 1.0, fastMode=False)
        qoArr = particle.qoArr
        particleAngEl = np.arctan2(qoArr[-1][1],
                                   qoArr[-1][0])  # very close to zero, or negative, if particle made it to
        # end
        if particleAngEl < .01:
            error = np.std(1e6 * particle.qoArr[:, 1])
            return error
        else:
            return np.nan

    outputOffsetFactArr = np.linspace(-3e-3, 3e-3, 100)

    if parallel:
        njobs = -1
    else:
        njobs = 1
    errorArr = np.asarray(
        Parallel(n_jobs=njobs)(delayed(errorFunc)(outputOffset) for outputOffset in outputOffsetFactArr))
    rOffsetOptimal = self._find_rOptimal(outputOffsetFactArr, errorArr)
    return rOffsetOptimal