import numpy as np
from scipy.spatial.transform import Rotation as Rot

from particle_class import Particle, Swarm
from helper_tools import parallel_evaluate
from storage_ring_modeler import StorageRingModel
from type_hints import RealNum


def timestep_particle_to_zero(particle: Particle):
    startX = -1e-10
    dx = startX - particle.qi[0]
    dt = dx / particle.pi[0]
    particle.qi += dt * particle.pi


def displace_swarm(swarm: Swarm, dx: RealNum, dy: RealNum, dz: RealNum, move_to_zero: bool = True):
    delta = np.array([dx, dy, dz])
    for particle in swarm:
        particle.qi += delta
        if move_to_zero:
            timestep_particle_to_zero(particle)


def change_swarm_speed(swarm: Swarm, delta_speed: RealNum):
    for particle in swarm:
        speed0 = np.linalg.norm(particle.pi)
        velocity_normalized = particle.pi / speed0
        new_speed = speed0 + delta_speed
        particle.pi = velocity_normalized * new_speed


def rotate_swarm_momentum(swarm: Swarm, angle_y: RealNum, angle_z: RealNum):
    Rz = Rot.from_rotvec([0, 0, angle_z]).as_matrix()
    Ry = Rot.from_rotvec([0, angle_y, 0]).as_matrix()
    for particle in swarm:
        particle.pi = Rz @ particle.pi
        particle.pi = Ry @ particle.pi


class JitteredSwarmModel:
    def __init__(self, model, dx, dy, dz, delta_speed, angle_y, angle_z):
        self.model, self.dx, self.dy, self.dz, self.delta_speed, self.angle_y, self.angle_z = model, dx, dy, dz, delta_speed, angle_y, angle_z
        self.swarm_original = self.model.swarm_injector_initial.copy()  # copy original swarm so it can be reset

    def __enter__(self) -> StorageRingModel:
        displace_swarm(self.model.swarm_injector_initial, self.dx, self.dy, self.dz)
        rotate_swarm_momentum(self.model.swarm_injector_initial, self.angle_y, self.angle_z)
        change_swarm_speed(self.model.swarm_injector_initial, self.delta_speed)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.swarm_injector_initial = self.swarm_original  # put back original swarm


def solve_with_jittered_swarm(model: StorageRingModel, dx: RealNum, dy: RealNum, dz: RealNum, delta_speed: RealNum,
                              angle_y: RealNum, angle_z: RealNum) -> tuple[float, float]:
    with JitteredSwarmModel(model, dx, dy, dz, delta_speed, angle_y, angle_z) as jitteredModel:
        results = jitteredModel.mode_match()
    return results


def plot_trajectories_with_jittered_swarm(model: StorageRingModel, dx: RealNum, dy: RealNum, dz: RealNum,
                                          delta_speed: RealNum, angle_y: RealNum,
                                          angle_z: RealNum):
    with JitteredSwarmModel(model, dx, dy, dz, delta_speed, angle_y, angle_z) as jittered_model:
        jittered_model.show_floor_plan_with_trajectories()


def make_increasing_3D_arr_along_axis(amplitude: RealNum, num: int, axis: int) -> np.ndarray:
    arr = np.zeros((num, 3))
    arr[:, axis] = np.linspace(-amplitude, amplitude, num)
    return arr


def make_random_arr(amplitude: RealNum, num: int) -> np.ndarray:
    return amplitude * 2 * (np.random.random(num) - .5)


def make_random_arr_in_circle(radius: RealNum, num: int) -> np.ndarray:
    samples = []
    while len(samples) < num:
        x, y = make_random_arr(radius, 2)
        if np.sqrt(x ** 2 + y ** 2) <= radius:
            samples.append((x, y))
    return np.array(samples)


def make_misalignment_params(delta_x_max: RealNum, delta_r_max: RealNum, delta_speed: RealNum, angle_max: RealNum,
                             num: int) -> np.ndarray:
    shifts_x = make_random_arr(delta_x_max, num)
    shifts_y, shifts_z = make_random_arr_in_circle(delta_r_max, num).T
    angle_y, angle_z = make_random_arr_in_circle(angle_max, num).T
    delta_speeds = make_random_arr(delta_speed, num)
    params = np.column_stack((shifts_x, shifts_y, shifts_z, delta_speeds, angle_y, angle_z))
    return params


def solve_misaligned(model: StorageRingModel, delta_x_max: RealNum, delta_r_max: RealNum, delta_speed_max,
                     angle_max: RealNum, num: int) -> list[tuple[float, float]]:
    params = make_misalignment_params(delta_x_max, delta_r_max, delta_speed_max, angle_max, num)
    solve = lambda X: solve_with_jittered_swarm(model, *X)
    results = parallel_evaluate(solve, params)
    return results


dimIndex = {'x': 0, 'y': 1, 'z': 2, 'rot_angle_y': 3, 'rot_angle_z': 4}


def get_results_misaligned_dim(model: StorageRingModel, whichDim: str, amplitude: RealNum, num: int):
    misalign_values = np.linspace(-amplitude, amplitude, num)
    index = dimIndex[whichDim]

    def solve(val):
        params = [0] * len(dimIndex)
        params[index] = val
        return solve_with_jittered_swarm(model, *params)

    results = parallel_evaluate(solve, misalign_values)
    return misalign_values, results
