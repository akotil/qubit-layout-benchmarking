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


class SquareGrid(Architecture):
    def __init__(self, system_size: int):
        super().__init__(system_size)

    def get_topology(self):
        if not np.sqrt(self.system_size).is_integer():
            exit("The system size {} is not valid for square grid topology.".format(self.system_size))

        n = int(np.sqrt(self.system_size))
        graph = nx.grid_2d_graph(m=n, n=n)
        graph = nx.convert_node_labels_to_integers(graph)
        return [e for e in graph.edges]

class LineArchitecture(Architecture):
    def __init__(self,system_size: int):
        super().__init__(system_size)

    def get_topology(self):
        graph = nx.path_graph(self.system_size)
        return [e for e in graph.edges]


class LineArchitecture3d(Architecture):
    def __init__(self, system_size: int):
        super().__init__(system_size)

    #TODO
    def get_topology(self):
        pass