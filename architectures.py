from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from qiskit.providers.fake_provider import FakeMumbaiV2, FakeLagosV2, FakeGuadalupeV2


class Architecture(ABC):
    def __init__(self, system_size: int, name: str):
        self.system_size = system_size
        self.name = name
        self.coupling_map = self.get_topology()

    @abstractmethod
    def get_topology(self):
           pass


class Grid(Architecture):
    def __init__(self, system_size: int, m: int, n:int):
        assert (system_size == m * n, "System size does not match the defined grid structure.")
        self.m = m
        self.n = n
        super().__init__(system_size, "grid")

    def get_topology(self):
        graph = nx.grid_2d_graph(m=self.m, n=self.n)
        graph = nx.convert_node_labels_to_integers(graph)
        graph =  [list(e) for e in graph.edges]
        to_add = []
        for e in graph:
            i, j = e
            to_add.append([j,i])
        graph = graph + to_add
        return graph


class SquareGrid(Grid):
    def __init__(self, system_size: int):
        if not np.sqrt(system_size).is_integer():
            exit("The system size {} is not valid for square grid topology.".format(self.system_size))

        n = int(np.sqrt(system_size))
        super().__init__(system_size, m=n, n=n)


class LineArchitecture(Architecture):
    def __init__(self,system_size: int):
        super().__init__(system_size, "line")

    def get_topology(self):
        graph = nx.path_graph(self.system_size)
        return [list(e) for e in graph.edges]


class HeavyHexArchitecture(Architecture):
    def __init__(self, system_size: int):
        # TODO: There is no fake backend for Hummingbird model (IBM Ithaca)
        available_system_sizes = {7, 16, 27}
        if system_size not in available_system_sizes:
            exit("System size {} not available in IBM's heavyhex devices. Available system sizes are: {}".format(system_size, available_system_sizes))
        super().__init__(system_size, "heavyhex")

    def get_topology(self):
        available_systems = {7: FakeLagosV2(), 16:FakeGuadalupeV2(), 27:FakeMumbaiV2()}
        coupling_list = list(available_systems[self.system_size].coupling_map.get_edges())
        coupling_list = [list(t) for t in coupling_list]
        return coupling_list
