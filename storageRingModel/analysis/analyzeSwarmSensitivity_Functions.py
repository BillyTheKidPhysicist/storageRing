import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ParticleClass import Particle, Swarm
from storageRingModeler import StorageRingModel

from helperTools import tool_Parallel_Process


def timeStep_Particle_To_Zero(particle: Particle):
    startX = -1e-10
    dx = startX - particle.qi[0]
    dt = dx / particle.pi[0]
    particle.qi += dt * particle.pi


def displace_Swarm(swarm: Swarm, dx: float, dy: float, dz: float, moveToZero: bool = True):
    delta = np.array([dx, dy, dz])
    for particle in swarm:
        particle.qi += delta
        if moveToZero:
            timeStep_Particle_To_Zero(particle)


def rotate_Swarm_Momentum(swarm: Swarm, angleY: float, angleZ: float):
    Rz = Rot.from_rotvec([0, 0, angleZ]).as_matrix()
    Ry = Rot.from_rotvec([0, angleY, 0]).as_matrix()
    for particle in swarm:
        particle.pi = Rz @ particle.pi
        particle.pi = Ry @ particle.pi


def solve_With_Jittered_Swarm(model: StorageRingModel, dx: float, dy: float, dz: float, angleY: float, angleZ: float) \
        -> tuple[float, float]:
    swarmOriginal=model.swarmInjectorInitial.copy() #copy original swarm so it can be reset
    displace_Swarm(model.swarmInjectorInitial, dx, dy, dz)
    rotate_Swarm_Momentum(model.swarmInjectorInitial, angleY, angleZ)
    results= model.mode_Match()
    model.swarmInjectorInitial=swarmOriginal #put back original swarm
    return results


def make_Increasing_3D_Arr_Along_Axis(amplitude: float, num: int, axis: int) -> np.ndarray:
    arr = np.zeros((num, 3))
    arr[:, axis] = np.linspace(-amplitude, amplitude, num)
    return arr

def make_Random_Arr(amplitude: float, num: int) -> np.ndarray:
    return amplitude * 2 * (np.random.random(num) - .5)


def make_Random_Arr_In_Circle(radius: float, num: int) -> np.ndarray:
    samples = []
    while len(samples) < num:
        x, y = make_Random_Arr(radius, 2)
        if np.sqrt(x ** 2 + y ** 2) <= radius:
            samples.append((x, y))
    return np.array(samples)


def make_Misalignment_Params(deltaxMax: float, deltarMax: float, angleMax: float, num: int) -> np.ndarray:
    shiftsx = make_Random_Arr(deltaxMax, num)
    shiftsy, shiftsz = make_Random_Arr_In_Circle(deltarMax, num).T
    rotsY, rotsZ = make_Random_Arr_In_Circle(angleMax, num).T
    params = np.column_stack((shiftsx, shiftsy, shiftsz, rotsY, rotsZ))
    return params

def solve_Misaligned(model: StorageRingModel,deltaxMax: float, deltarMax: float, angleMax: float, num: int)\
        -> list[tuple[float, float]]:
    params=make_Misalignment_Params(deltaxMax,deltarMax,angleMax,num)
    solve=lambda X: solve_With_Jittered_Swarm(model,*X)
    results=tool_Parallel_Process(solve,params)
    return results


dimIndex={'x':0,'y':1,'z':2,'roty':3,'rotz':4}
def get_Results_Misaligned_Dimension(model: StorageRingModel,whichDim: str,amplitude: float,num: int, parallel: bool=True):
    misalignValues=np.linspace(-amplitude,amplitude,num)
    index=dimIndex[whichDim]
    def solve(val):
        params=[0]*len(dimIndex)
        params[index]=val
        return solve_With_Jittered_Swarm(model,*params)
    results=tool_Parallel_Process(solve,misalignValues)
    return misalignValues,results