import copy
import warnings
from math import pi
from typing import Union, Iterable, Any

import numpy as np

'''
Methods and objects for computing vacuum system performance. A model of a vacuum is created component by component,
then the system of equations of conductance and pump speed are converted to a matrix and solved. Currently only
supports linear vacuum systems 
'''

tube_cond_air_fact = 12.1  # assumes cm and torr

RealNum = Union[float, int]

R = 8.3145


def split_length_into_equal_offset_array(L: RealNum, num_points: int) -> np.ndarray:
    num_splits = num_points + 1
    start = L / num_splits
    stop = L - start
    return np.linspace(start, stop, num_points)


def append_or_extend_to_list(item: Union[Any, list], item_list: list) -> None:
    if has_len(item):
        item_list.extend(item)
    else:
        item_list.append(item)


def has_len(obj: Any) -> bool:
    return hasattr(obj, '__len__')


def nan_array(num: int) -> np.ndarray:
    return np.ones(num) * np.nan


def is_valid_pressure(value: Union[np.ndarray, RealNum]) -> bool:
    if type(value) is np.ndarray:
        return np.all(value >= 0.0)
    else:
        return value >= 0.0


class _Component:
    def __init__(self, name: str, P: Union[np.ndarray, RealNum]):
        self.name = name
        self.P = P

    def __setattr__(self, name, value):
        if name == 'P':
            if name in self.__dict__:
                assert is_valid_pressure(value)
        self.__dict__[name] = value

    def __str__(self):
        string = '--Component information-- \n'
        string += 'component type: ' + type(self).__name__ + '\n'
        string += 'name: ' + self.name + '\n'

        for key, value in self.__dict__.items():
            if key != 'name':
                string += key + ': ' + str(value) + '\n'
        return string


class Tube(_Component):
    def __init__(self, L: RealNum, inside_diam: RealNum, q: RealNum = 0.0, num_profile_points: int = 1,
                 name: str = 'unassigned'):
        assert L > 0.0 and inside_diam > 0.0
        if not L >= 3 * inside_diam:
            warnings.warn('Tube length should be several times longer than diam for more accurate results')
        super().__init__(name, nan_array(num_profile_points))
        self.P_x_vals = split_length_into_equal_offset_array(L, num_profile_points)
        self.L = L
        self.inside_diam = inside_diam
        self.Q = q * self.inside_diam * pi * self.L

    def C(self) -> float:
        geometric_factor = self.inside_diam ** 3 / self.L
        # geometric_factor*=1/(1+(4/3)*self.inside_diam/self.L)
        return tube_cond_air_fact * geometric_factor


class Chamber(_Component):
    def __init__(self, S: RealNum, Q: RealNum, name: str = 'unassigned'):
        assert S >= 0.0 and Q >= 0.0
        super().__init__(name, np.nan)
        self.S = S
        self.Q = Q


Component = Union[Tube, Chamber]


def node_index(node, nodes: list) -> int:
    indices = [i for i, _node in enumerate(nodes) if _node is node]
    assert len(indices) == 1
    return indices[0]


def get_branch(nodes: list, stop_cond):
    if len(nodes) == 0:
        return []
    branch = []
    for node in nodes:
        branch.append(node)
        if stop_cond(node):
            break
    return branch


def reverse(some_list):
    some_list_copy = some_list.copy()
    some_list_copy.reverse()
    return some_list_copy


def backward_and_forward_nodes(branch_node, nodes, is_circular) -> tuple[list[Component], list[Component]]:
    index = node_index(branch_node, nodes)
    nodes_forward_to_end = [nodes[i] for i in range(index + 1, len(nodes))]
    nodes_backward_to_end = [nodes[i] for i in range(index - 1, -1, -1)]
    if is_circular:
        nodes_forward = [*nodes_forward_to_end, *reverse(nodes_backward_to_end)]
        nodes_backward = [*nodes_backward_to_end, *reverse(nodes_forward_to_end)]
    else:
        nodes_forward, nodes_backward = nodes_forward_to_end, nodes_backward_to_end
    return nodes_backward, nodes_forward


def get_branches_from_node(branch_node, nodes: list, stop_cond, is_circular) -> tuple[list, list]:
    nodes_backward, nodes_forward = backward_and_forward_nodes(branch_node, nodes, is_circular)
    branch_forward = get_branch(nodes_forward, stop_cond)
    branch_backward = get_branch(nodes_backward, stop_cond)
    return branch_backward, branch_forward


class VacuumSystem:
    def __init__(self, is_circular=False):
        self.components: list[Component] = []
        self.is_circular = is_circular

    def add_tube(self, L: float, inside_diam: float, q: float = 0.0, num_profile_points=1, name: str = 'unassigned'):
        component = Tube(L, inside_diam, q=q, num_profile_points=num_profile_points, name=name)
        self.components.append(component)

    def add_chamber(self, S: float = 0.0, Q: float = 0.0, name: str = 'unassigned'):
        component = Chamber(S, Q, name=name)
        self.components.append(component)

    def num_components(self) -> int:
        return len(self.components)

    def chambers(self) -> tuple[Chamber]:
        return tuple([component for component in self if type(component) is Chamber])

    def __iter__(self) -> Iterable[Component]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, index: int) -> Component:
        return self.components[index]

    def branches(self, chamber: Chamber) -> tuple[list[Component], list[Component]]:
        assert type(chamber) is Chamber

        def stop_condition(node):
            return type(node) is Chamber

        branch_1, branch_2 = get_branches_from_node(chamber, self.components, stop_condition, self.is_circular)
        return branch_1, branch_2


class SolverVacuumSystem(VacuumSystem):
    def __init__(self, vacuum_system: VacuumSystem):
        super().__init__(is_circular=vacuum_system.is_circular)
        self.components, self.component_map = self.solver_components_and_map(vacuum_system)
        self.matrix_index: dict[Component, int] = self.solver_matrix_index_dict()

    def solver_matrix_index_dict(self) -> dict[Component, int]:
        chambers = self.chambers()
        indices = range(len(chambers))
        matrix_index = dict(zip(chambers, indices))
        return matrix_index

    def split_tube(self, tube: Tube) -> list[Tube, Chamber]:
        num_profile_points = len(tube.P)
        L_split = tube.L / (num_profile_points + 1)
        Q_splits = tube.Q / num_profile_points
        component_initializers = [Tube, Chamber] * num_profile_points
        component_initializers.append(Tube)
        new_components = []
        for component_init in component_initializers:
            if component_init is Tube:
                new_components.append(component_init(L_split, tube.inside_diam, name=tube.name))
            elif component_init is Chamber:
                new_components.append(component_init(0.0, Q_splits))
            else:
                raise NotImplementedError

        return new_components

    def convert_component(self, component: Component) -> Union[list[Component], Component]:
        if type(component) is Chamber:
            solver_version = copy.copy(component)
        elif type(component) is Tube:
            if has_len(component.P):
                solver_version = self.split_tube(component)
            else:
                solver_version = copy.copy(component)
        else:
            raise NotImplementedError
        return solver_version

    def solver_components_and_map(self, vacuum_system: VacuumSystem) -> tuple[list[Component], dict]:
        solver_components = []
        component_map = {}
        for component in vacuum_system:
            solver_version = self.convert_component(component)
            append_or_extend_to_list(solver_version, solver_components)
            component_map[component] = solver_version
        return solver_components, component_map


def total_conductance(tubes: list[Tube]) -> float:
    return 1 / sum([1 / tube.C() for tube in tubes])


def make_Q_vec(vac_sys: SolverVacuumSystem) -> np.ndarray:
    Q_vec = np.array([chamber.Q for chamber in vac_sys.chambers()])
    return Q_vec


def is_all_tubes(components: Iterable[Component]) -> bool:
    return all(type(component) is Tube for component in components)


def make_C_matrix(solver_vac_sys: SolverVacuumSystem) -> np.ndarray:
    C_matrix = np.zeros((len(solver_vac_sys.chambers()),) * 2)
    for chamber_a in solver_vac_sys.chambers():
        idx_a = solver_vac_sys.matrix_index[chamber_a]
        C_matrix[idx_a, idx_a] += chamber_a.S
        branches = solver_vac_sys.branches(chamber_a)
        for branch in branches:
            assert len(branch) != 1  # either no branch, or at least one tube, then a vacuum chamber
            if len(branch) != 0:
                tubes, chamber_b = branch[:-1], branch[-1]
                assert is_all_tubes(tubes) and type(chamber_b) is Chamber
                idx_b = solver_vac_sys.matrix_index[chamber_b]
                C_total = total_conductance(tubes)
                C_matrix[idx_a, idx_b] += -C_total
                C_matrix[idx_a, idx_a] += C_total
    return C_matrix


def map_pressure_to_tube(tube: Tube, tube_split_components: list[Component]) -> None:
    pressure_profile = [comp.P for comp in tube_split_components if type(comp) is Chamber]
    assert len(tube.P) == len(pressure_profile)
    tube.P[:] = pressure_profile


def update_vacuum_system_with_results(vacuum_system: VacuumSystem, solver_vac_sys: SolverVacuumSystem) -> None:
    for component in vacuum_system:
        if type(component) is Chamber:
            component.P = solver_vac_sys.component_map[component].P
        elif type(component) is Tube:
            tube_split_components = solver_vac_sys.component_map[component]
            map_pressure_to_tube(component, tube_split_components)


def solve_vac_system(vacuum_system: VacuumSystem) -> None:
    solver_vac_sys = SolverVacuumSystem(vacuum_system)
    Q_mat = make_Q_vec(solver_vac_sys)
    C = make_C_matrix(solver_vac_sys)
    P = np.linalg.inv(C) @ Q_mat
    for idx, chamber in enumerate(solver_vac_sys.chambers()):
        chamber.P = P[idx]

    update_vacuum_system_with_results(vacuum_system, solver_vac_sys)
