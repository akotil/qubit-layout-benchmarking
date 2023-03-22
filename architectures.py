from abc import ABC, abstractmethod
import numpy as np
import networkx as nx

class Architecture(ABC):
    def __init__(self, system_size: int):
        self.system_size = system_size
        self.coupling_map = self.get_topology()

    @abstractmethod
    def get_topology(self):
           pass

    @property
    @abstractmethod
    def name(self):
        pass


class Grid(Architecture):
    def __init__(self, system_size: int, m: int, n:int):
        self.m = m
        self.n = n
        super().__init__(system_size)

    def get_topology(self):
        graph = nx.grid_2d_graph(m=self.m, n=self.n)
        graph = nx.convert_node_labels_to_integers(graph)
        return [list(e) for e in graph.edges]

    @property
    def name(self):
        return "grid"

class SquareGrid(Grid):
    def __init__(self, system_size: int):
        if not np.sqrt(system_size).is_integer():
            exit("The system size {} is not valid for square grid topology.".format(self.system_size))

        n = int(np.sqrt(self.system_size))
        super().__init__(system_size, m=n, n=n)

    @Grid.name.getter
    def name(self):
        return "square_grid"


class LineArchitecture(Architecture):
    def __init__(self,system_size: int):
        super().__init__(system_size)

    def get_topology(self):
        graph = nx.path_graph(self.system_size)
        return [list(e) for e in graph.edges]

    @property
    def name(self):
        return "line"


class LineArchitecture3d(Architecture):
    def __init__(self, system_size: int):
        super().__init__(system_size)

    #TODO
    def get_topology(self):
        pass

    @property
    def name(self):
        return "line_3d"