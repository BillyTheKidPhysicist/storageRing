from math import isclose, sqrt, pi

from scipy.spatial.transform import Rotation as Rot

from constants import SIMULATION_MAGNETON, FLAT_WALL_VACUUM_THICKNESS
from helperTools import *
from latticeElements.elements import CombinerIdeal, CombinerHalbachLensSim, CombinerSim
from numbaFunctionsAndObjects.combinerIdealFieldHelper import combiner_Ideal_Force


def compute_Particle_Trajectory(forceFunc, speed, xStart, xStop, particleOutputOffsetStart: float = 0.0,
                                atomState='LOW_FIELD_SEEKING') -> tuple[np.ndarray, np.ndarray]:
    # this computes the output angle and offset for a combiner magnet.
    # NOTE: for the ideal combiner this gives slightly inaccurate results because of lack of conservation of energy!
    # NOTE: for the simulated bender, this also give slightly unrealisitc results because the potential is not allowed
    # to go to zero (finite field space) so the the particle will violate conservation of energy
    # limit: how far to carry the calculation for along the x axis. For the hard edge magnet it's just the hard edge
    # length, but for the simulated magnets, it's that plus twice the length at the ends.
    # h: timestep
    # lowField: wether to model low or high field seekers
    h = 5e-6
    particleOutputOffsetStart = -particleOutputOffsetStart  # temporary
    assert atomState in ('LOW_FIELD_SEEKING', 'HIGH_FIELD_SEEKING')
    stateFact = 1 if atomState == 'LOW_FIELD_SEEKING' else -1
    Force = lambda x: forceFunc(x) * stateFact
    q = np.asarray([xStart, particleOutputOffsetStart, 0.0])
    p = np.asarray([speed, 0.0, 0.0])
    qList, pList = [q], [p]

    forcePrev = Force(q)  # recycling the previous force value cut simulation time in half
    while True:
        F = forcePrev
        q_n = q + p * h + .5 * F * h ** 2
        if q_n[0] > xStop:  # if overshot, go back and walk up to the edge assuming no force
            dr = xStop - q[0]
            dt = dr / p[0]
            qFinal = q + p * dt
            F_n = Force(q_n)
            assert not np.any(np.isnan(F_n))
            pFinal = p + .5 * (F + F_n) * h
            qList.append(qFinal)
            pList.append(pFinal)
            break
        F_n = Force(q_n)
        assert not np.any(np.isnan(F_n))
        p_n = p + .5 * (F + F_n) * h
        q, p = q_n, p_n
        forcePrev = F_n
        qList.append(q)
        pList.append(p)
    assert qFinal[2] == 0.0  # only interested in xy plane bending, expected to be zero
    qArr = np.asarray(qList)
    pArr = np.asarray(pList)
    return qArr, pArr


def calculateTrajectory_Length(qTracedArr: np.ndarray) -> float:
    assert np.all(np.sort(qTracedArr[:, 0]) == qTracedArr[:, 0])  # monotonically increasing
    return float(np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1))))


def make_Halbahc_Combiner_Force_Function(el) -> Callable:
    lens = el.make_Lens()
    assert all(val == 0.0 for val in [*lens.orientation.as_rotvec(), *lens.position])
    orientation = Rot.from_rotvec([0, pi / 2, 0.0])
    lens.rotate(orientation)
    lens.move((el.space + el.Lm / 2, 0, 0))

    def force_Func(q):
        if el.space < q[0] < el.Lm + el.space:
            assert sqrt(q[1] ** 2 + q[2] ** 2) < el.ap
        F = -SIMULATION_MAGNETON * lens.BNorm_Gradient(q)
        F[2] = 0.0
        return F

    return force_Func


def input_Angle(pArr) -> float:
    px, py, _ = pArr[-1]
    return np.arctan(py / px)


def closet_Approach_To_Lens_Corner(el: CombinerHalbachLensSim, qArr: np.ndarray):
    lensCorner = np.array([el.space + el.Lm + FLAT_WALL_VACUUM_THICKNESS, -el.ap, 0.0])
    return np.min(np.linalg.norm(qArr - lensCorner, axis=1))


def characterize_CombinerIdeal(el: CombinerIdeal):
    assert type(el) is CombinerIdeal

    def force(q):
        assert abs(q[2]) < el.apz and -el.apL < q[1] < el.apR
        return np.array(combiner_Ideal_Force(*q, el.Lm, el.c1, el.c2))

    qArr, pArr = compute_Particle_Trajectory(force, el.PTL.v0Nominal, 0.0, el.Lm)
    assert isclose(qArr[-1, 0], el.Lm) and isclose(qArr[0, 0], 0.0)
    trajectoryLength = calculateTrajectory_Length(qArr)
    inputAngle = input_Angle(pArr)
    inputOffset = qArr[-1, 1]
    assert trajectoryLength > el.Lm
    return inputAngle, inputOffset, trajectoryLength


def characterize_CombinerHalbach(el: CombinerHalbachLensSim, atomState, particleOffset=0.0):
    force_Func = make_Halbahc_Combiner_Force_Function(el)
    qArr, pArr = compute_Particle_Trajectory(force_Func, el.PTL.v0Nominal, 0.0, 2 * el.space + el.Lm,
                                             particleOutputOffsetStart=particleOffset, atomState=atomState)

    assert isclose(qArr[-1, 0], el.Lm + 2 * el.space) and isclose(qArr[0, 0], 0.0)
    minBeamLensSep = closet_Approach_To_Lens_Corner(el, qArr)
    trajectoryLength = calculateTrajectory_Length(qArr)
    inputAngle = input_Angle(pArr)
    inputOffset = qArr[-1, 1]
    return inputAngle, inputOffset, trajectoryLength, minBeamLensSep


def characterize_CombinerSim(el: CombinerSim):
    def force_Func(q):
        if el.space < q[0] < el.Lm + el.space:
            assert abs(q[2]) < el.apz and -el.apL < q[1] < el.apR
        F = np.array(el.fastFieldHelper.numbaJitClass.force_Without_isInside_Check(*q))
        F[2] = 0.0
        return F

    qArr, pArr = compute_Particle_Trajectory(force_Func, el.PTL.v0Nominal, 0.0, 2 * el.space + el.Lm)
    assert isclose(qArr[-1, 0], 2 * el.space + el.Lm) and isclose(qArr[0, 0], 0.0)
    trajectoryLength = calculateTrajectory_Length(qArr)
    inputAngle = input_Angle(pArr)
    inputOffset = qArr[-1, 1]
    assert trajectoryLength > el.Lm
    return inputAngle, inputOffset, trajectoryLength
