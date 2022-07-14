import warnings
from typing import Union, Iterable

import numpy as np

'''
Methods and objects for computing vacuum system performance. A model of a vacuum is created component by component,
then the system of equations of conductance and pump speed are converted to a matrix and solved. Currently only
supports linear vacuum systems 
'''

tube_cond_air_fact = 12.1

RealNum = Union[float, int]


class _Component:
    def __init__(self, name: str):
        self.name = name
        self.P = None

    def __setattr__(self, name, value):
        if name == 'P':
            assert value >= 0.0 if value is not None else True
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
    def __init__(self, L: RealNum, inside_diam: RealNum, name: str):
        assert L > 0.0 and inside_diam > 0.0
        if not L >= 3 * inside_diam:
            warnings.warn('Tube length should be several times longer than diam for more accurate results')
        super().__init__(name)
        self.L = L
        self.inside_diam = inside_diam

    def C(self) -> float:
        geometric_factor = self.inside_diam ** 3 / self.L
        # geometric_factor*=1/(1+(4/3)*self.inside_diam/self.L)
        return tube_cond_air_fact * geometric_factor


class Chamber(_Component):
    def __init__(self, S: RealNum, Q: RealNum, name: str):
        assert S >= 0.0 and Q >= 0.0
        super().__init__(name)
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
        self.matrix_index: dict[Component, int] = {}
        self.is_circular = is_circular

    def add_tube(self, L: float, inside_diam: float, name: str = 'unassigned'):
        component = Tube(L, inside_diam, name)
        self.components.append(component)

    def add_chamber(self, S: float = 0.0, Q: float = 0.0, name: str = 'unassigned'):
        component = Chamber(S, Q, name)
        self.components.append(component)
        self.matrix_index[component] = len(self.matrix_index)

    def num_components(self) -> int:
        return len(self.components)

    def num_chambers(self) -> int:
        return len(self.chambers())

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


def total_conductance(tubes: list[Tube]) -> float:
    return 1 / sum([1 / tube.C() for tube in tubes])


def make_Q_matrix(vac_sys: VacuumSystem) -> np.ndarray:
    Q_mat = np.zeros(vac_sys.num_chambers())
    for chamber in vac_sys.chambers():
        idx = vac_sys.matrix_index[chamber]
        Q_mat[idx] = chamber.Q
    return Q_mat


def is_all_tubes(components: Iterable[Component]) -> bool:
    return all(type(component) is Tube for component in components)


def make_C_matrix(vac_sys: VacuumSystem) -> np.ndarray:
    M = np.zeros((vac_sys.num_chambers(),) * 2)
    for chamber_a in vac_sys.chambers():
        idx_a = vac_sys.matrix_index[chamber_a]
        M[idx_a, idx_a] += chamber_a.S
        branches = vac_sys.branches(chamber_a)
        for branch in branches:
            assert len(branch) != 1  # either no branch, or at least one tube, then a vacuum chamber
            if len(branch) != 0:
                tubes, chamber_b = branch[:-1], branch[-1]
                assert is_all_tubes(tubes) and type(chamber_b) is Chamber
                idx_b = vac_sys.matrix_index[chamber_b]
                C_total = total_conductance(tubes)
                M[idx_a, idx_b] += -C_total
                M[idx_a, idx_a] += C_total
    return M


def solve_vac_system(vac_sys: VacuumSystem):
    Q_mat = make_Q_matrix(vac_sys)
    C = make_C_matrix(vac_sys)
    P = np.linalg.inv(C) @ Q_mat
    for chamber in vac_sys.chambers():
        idx = vac_sys.matrix_index[chamber]
        chamber.P = P[idx]
