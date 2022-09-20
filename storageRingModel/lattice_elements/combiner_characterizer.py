"""
Contains functions for characterizing combiner's input angle and offsets with particle tracing. These
functions are called everytime a combiner element is made.
"""

from math import isclose, sqrt

from constants import SIMULATION_MAGNETON, FLAT_WALL_VACUUM_THICKNESS
from helper_tools import *
from lattice_elements.elements import CombinerIdeal, CombinerLensSim, CombinerSim
from numba_functions_and_objects.combiner_ideal_numba_function import combiner_Ideal_Force


def compute_particle_trajectory(force_func, speed, xStart, xStop, particle_y_offset_start: float = 0.0,
                                atom_state='LOW_FIELD_SEEKER', h=5e-6) -> tuple[np.ndarray, np.ndarray]:
    # TODO: WHAT IS THE DEAL WITH THIS?
    particle_y_offset_start = -particle_y_offset_start  # temporary
    assert atom_state in ('LOW_FIELD_SEEKER', 'HIGH_FIELD_SEEKER')
    state_fact = 1 if atom_state == 'LOW_FIELD_SEEKER' else -1

    def force(x):
        return force_func(x) * state_fact

    q = np.asarray([xStart, particle_y_offset_start, 0.0])
    p = np.asarray([speed, 0.0, 0.0])
    q_list, p_list = [q], [p]

    force_prev = force(q)  # recycling the previous force value cut simulation time in half
    while True:
        F = force_prev
        q_n = q + p * h + .5 * F * h ** 2
        if q_n[0] > xStop:  # if overshot, go back and walk up to the edge assuming no force
            dr = xStop - q[0]
            dt = dr / p[0]
            q_final = q + p * dt
            F_n = force(q_n)
            assert not np.any(np.isnan(F_n))
            pFinal = p + .5 * (F + F_n) * h
            q_list.append(q_final)
            p_list.append(pFinal)
            break
        F_n = force(q_n)
        assert not np.any(np.isnan(F_n))
        p_n = p + .5 * (F + F_n) * h
        q, p = q_n, p_n
        force_prev = F_n
        q_list.append(q)
        p_list.append(p)
    assert q_final[2] == 0.0  # only interested in xy plane bending, expected to be zero
    q_arr = np.asarray(q_list)
    p_arr = np.asarray(p_list)
    return q_arr, p_arr


def calculate_trajectory_length(qTracedArr: np.ndarray) -> float:
    assert np.all(np.sort(qTracedArr[:, 0]) == qTracedArr[:, 0])  # monotonically increasing
    return float(np.sum(np.sqrt(np.sum((qTracedArr[1:] - qTracedArr[:-1]) ** 2, axis=1))))


def make_halbach_combiner_force_function(el) -> Callable:
    lens = el.magnet.magpylib_magnets_model(False, False)

    def force_func(q):
        if el.space < q[0] < el.Lm + el.space:
            assert sqrt(q[1] ** 2 + q[2] ** 2) < el.ap
        F = -SIMULATION_MAGNETON * lens.B_norm_grad(q)
        F[2] = 0.0
        return F

    return force_func


def make_combiner_force_func(el) -> Callable:
    if type(el) is CombinerLensSim:
        return make_halbach_combiner_force_function(el)


def input_angle(p_arr) -> float:
    px, py, _ = p_arr[-1]
    return np.arctan(py / px)


def closet_approach_to_lens_corner(el: CombinerLensSim, q_arr: np.ndarray):
    lens_corner_coords = np.array([el.space + el.Lm + FLAT_WALL_VACUUM_THICKNESS, -el.ap, 0.0])
    return np.min(np.linalg.norm(q_arr - lens_corner_coords, axis=1))


def characterize_combiner_ideal(el: CombinerIdeal):
    assert type(el) is CombinerIdeal

    def force(q):
        assert abs(q[2]) < el.apz and -el.ap_left < q[1] < el.ap_right
        return np.array(combiner_Ideal_Force(*q, el.Lm, el.c1, el.c2))

    q_arr, p_arr = compute_particle_trajectory(force, el.PTL.design_speed, 0.0, el.Lm)
    assert isclose(q_arr[-1, 0], el.Lm) and isclose(q_arr[0, 0], 0.0)
    trajectory_length = calculate_trajectory_length(q_arr)
    input_ang = input_angle(p_arr)
    input_offset = q_arr[-1, 1]
    assert trajectory_length > el.Lm
    return input_ang, input_offset, trajectory_length


def characterize_combiner_halbach(el: CombinerLensSim, atom_state=None, particleOffset=None):
    atom_state = (
        'HIGH_FIELD_SEEKER' if el.field_fact == -1 else 'LOW_FIELD_SEEKER') if atom_state is None else atom_state
    particle_y_offset_start = el.output_offset if particleOffset is None else particleOffset
    force_func = make_halbach_combiner_force_function(el)
    q_arr, p_arr = compute_particle_trajectory(force_func, el.PTL.design_speed, 0.0, 2 * el.space + el.Lm,
                                               particle_y_offset_start=particle_y_offset_start, atom_state=atom_state)

    assert isclose(q_arr[-1, 0], el.Lm + 2 * el.space) and isclose(q_arr[0, 0], 0.0)
    min_beam_lens_sep = closet_approach_to_lens_corner(el, q_arr)
    trajectory_length = calculate_trajectory_length(q_arr)
    input_ang = input_angle(p_arr)
    input_offset = q_arr[-1, 1]
    return input_ang, input_offset, trajectory_length, min_beam_lens_sep


def characterize_combiner_sim(el: CombinerSim):
    from numba_functions_and_objects.combiner_quad_sim_numba_function import force_without_isinside_check
    params = (np.nan, np.nan, el.Lb, el.Lm, el.apz, el.ap_left, el.ap_right, el.space, el.field_fact)

    field_data = el.open_and_shape_field_data()

    def force_func(q):
        if el.space < q[0] < el.Lm + el.space:
            assert abs(q[2]) < el.apz and -el.ap_left < q[1] < el.ap_right
        F = np.array(force_without_isinside_check(*q, params, field_data))
        F[2] = 0.0
        return F

    q_arr, p_arr = compute_particle_trajectory(force_func, el.PTL.design_speed, 0.0, 2 * el.space + el.Lm)
    assert isclose(q_arr[-1, 0], 2 * el.space + el.Lm) and isclose(q_arr[0, 0], 0.0)
    trajectory_length = calculate_trajectory_length(q_arr)
    input_ang = input_angle(p_arr)
    input_offset = q_arr[-1, 1]
    assert trajectory_length > el.Lm
    return input_ang, input_offset, trajectory_length
