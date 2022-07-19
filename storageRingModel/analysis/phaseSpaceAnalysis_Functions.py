import matplotlib.pyplot as plt

from ParticleClass import Swarm, Particle
from helperTools import *
from phaseSpaceAnalyzer import SwarmSnapShot
from storageRingModeler import StorageRingModel


def trace_swarm_through_ring(model: StorageRingModel, swarm_to_trace: Swarm, h=7.5e-6, T=10.0,
                             parallel=True) -> tuple[Swarm, Swarm]:
    swarm_injector_traced = model.swarm_tracer_injector.trace_swarm_through_lattice(
        swarm_to_trace, h, 1.0, use_fast_mode=False, copy_swarm=True,
        log_phase_space_coords=True, accelerated=True, use_energy_correction=True)
    swarm_ring_initial = model.transform_swarm_from_injector_to_ring_frame(swarm_injector_traced,
                                                                           copy_particles=True)
    swarm_traced_ring = model.swarm_tracer_ring.trace_swarm_through_lattice(swarm_ring_initial, h, T,
                                                                            use_fast_mode=False, accelerated=True,
                                                                            copy_swarm=True,
                                                                            parallel=parallel, steps_per_logging=4,
                                                                            use_energy_correction=True)

    xoStart = sum(model.lattice_ring.el_list[i].Lo for i in range(0, model.lattice_ring.combiner_index + 1))

    for particle in swarm_traced_ring:
        assert particle.clipped is not None
        if particle.qo_arr is not None:
            particle.qo_arr[:, 0] -= particle.qo_arr[0, 0]
            particle.qo_arr[:, 0] += xoStart

    return swarm_injector_traced, swarm_traced_ring


def emmitance(qi: np.ndarray, pi: np.ndarray) -> float:
    var_qi = np.mean(qi ** 2) - np.mean(qi) ** 2
    var_pi = np.mean(pi ** 2) - np.mean(pi) ** 2
    var_qipi = np.mean(pi * qi) - np.mean(pi) * np.mean(qi)
    return np.sqrt(var_qi * var_pi - var_qipi ** 2)


def emittance_from_particles(particles: list) -> tuple[float, float]:
    y = 1e3 * np.array([particle.qo[1] for particle in particles])
    py = np.array([particle.po[1] for particle in particles])
    z = 1e3 * np.array([particle.qo[2] for particle in particles])
    pz = np.array([particle.po[2] for particle in particles])
    epsy = emmitance(y, py)
    epsz = emmitance(z, pz)
    return epsy, epsz


def plot_ring_lattice_with_stops(model: StorageRingModel, phase_space_info, savefig=None, dpi=100):
    model.lattice_ring.show_lattice(show_immediately=False)

    for i, particle_stops in enumerate(phase_space_info):
        first_stop = particle_stops[1]
        x_vals, y_vals = [], []
        for particle in first_stop:
            if particle.clipped is False:
                x, y, z = particle.q
                x_vals.append(x)
                y_vals.append(y)
        plt.scatter(np.mean(x_vals), np.mean(y_vals), marker='x', s=200, label='stop ' + str(i + 1), zorder=100)

    plt.legend()
    if savefig is not None:
        plt.savefig(savefig, dpi=dpi)
    plt.show()


def transfer_particles_to_new_swarm(swarm_original: Swarm, num_particles: int) -> Swarm:
    assert isinstance(num_particles, int) and num_particles >= 0
    swarm_new = Swarm()
    while len(swarm_new) < num_particles and len(swarm_original) > 0:
        swarm_new.add(swarm_original.particles.pop(0))
    return swarm_new


def initial_phase_space_info(model: StorageRingModel, info: float, h: float) -> list[Particle]:
    num_particle = len(info[0])
    swarm_to_trace = model.generate_swarm(num_particle)
    swarm_injector_traced, _ = trace_swarm_through_ring(model, swarm_to_trace, h=h, T=1e-6, parallel=True)
    x_start = max([particle.qo_arr[0][0] for particle in swarm_injector_traced])
    # match up paremter
    for particle_ring, particle_injector in zip(info[0], swarm_injector_traced):
        particle_injector.T += particle_ring.T
        particle_injector.revolutions = particle_ring.revolutions

    snap = SwarmSnapShot(swarm_injector_traced, x_start + 1e-9)
    return snap.particles


def get_phase_space_info_at_positions(model: StorageRingModel, xPositions: list, numParticles: int, h: float = 7.5e-6,
                                      T: float = 10.0, parallel=True) -> list:
    phase_space_info = [[[] for _ in xStops] for xStops in xPositions]
    swarm_initial = model.generate_swarm(numParticles)
    numParticlesLeft = numParticles
    worksize = 100
    while numParticlesLeft > 0:
        swarm_to_trace = transfer_particles_to_new_swarm(swarm_initial, worksize)
        _, swarm_traced = trace_swarm_through_ring(model, swarm_to_trace, h=h, T=T, parallel=parallel)
        for phase_space_stop, x_stops in zip(phase_space_info, xPositions):
            for info, x in zip(phase_space_stop, x_stops):
                snap = SwarmSnapShot(swarm_traced, x)
                info.extend(snap.particles)
        numParticlesLeft -= worksize if worksize < numParticlesLeft else numParticlesLeft

    for info in phase_space_info:
        phase_space_initial = initial_phase_space_info(model, info, h)
        info.insert(0, phase_space_initial)
    return phase_space_info


def make_phase_space_x_positions(model: StorageRingModel, x0, numStops, Tmax) -> list:
    x_max = model.lattice_injector.speed_nominal * Tmax
    revs_max = x_max / model.lattice_ring.total_length - 1
    if numStops != -1:
        assert revs_max / numStops > 1
        revs_stops = np.linspace(0, revs_max, numStops).astype(int)
    else:
        revs_stops = np.arange(0, revs_max, 1.0).astype(int)
    x_positions = []
    for x in x0:
        x_stops = [x + revs * model.lattice_ring.total_length for revs in revs_stops]
        x_positions.append(x_stops)
    return x_positions


def make_phase_space_info(model: StorageRingModel, x0, numStops, Tmax, numParticles) -> tuple[list, list]:
    x0 = (x0,) if isinstance(x0, (float, int)) else x0
    x_positions = make_phase_space_x_positions(model, x0, numStops, Tmax)
    phase_space_info = get_phase_space_info_at_positions(model, x_positions, numParticles, T=Tmax + 1e-3)
    for x_stop in x_positions:
        x_stop.insert(0, 0)
    return phase_space_info, x_positions


def get_emittances(phase_space_info: list, periodic_stop_index: int, T_min: float = None, revs_min: int = None,
                   truncate_x_to_min_survived_T_min=True, extra_constraint_func=None) \
        -> tuple[list[float], list[float]]:
    assert all((val >= 0.0 if val is not None else True) for val in [T_min, revs_min])
    x_min = None
    survived_particles_x = [particle.qo[0] for particle in phase_space_info[0][-1] if particle.T > T_min]
    if truncate_x_to_min_survived_T_min and len(survived_particles_x) != 0:
        x_min = min(survived_particles_x)
        print(x_min)

    def is_valid(particle):
        if particle.clipped:
            return False
        elif T_min is not None and particle.T < T_min:
            return False
        elif revs_min is not None and particle.revolutions < revs_min:
            return False
        elif x_min is not None and particle.qo[0] > x_min:
            return False
        elif extra_constraint_func is not None and not extra_constraint_func(particle):
            return False
        else:
            return True

    epsy_stops, epsz_stops = [], []
    for particles_at_stop in phase_space_info[periodic_stop_index]:
        particles_valid = list(filter(is_valid, particles_at_stop))
        if len(particles_valid) == 0:
            epsy, epsz = np.nan, np.nan
        else:
            epsy, epsz = emittance_from_particles(particles_valid)
        epsy_stops.append(epsy)
        epsz_stops.append(epsz)
    return epsy_stops, epsz_stops
