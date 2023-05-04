from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from qiskit.providers.fake_provider import FakeMumbaiV2, FakeLagosV2, FakeGuadalupeV2, FakeWashingtonV2, FakeQuitoV2


class Architecture(ABC):
    def __init__(self, system_size: int, name: str):
        self.system_size = system_size
        self.name = name
        self.coupling_map = self.get_topology()

    @abstractmethod
    def get_topology(self):
        pass


class Grid(Architecture):
    def __init__(self, system_size: int, m: int, n: int):
        assert system_size == m * n, "System size does not match the defined grid structure."
        self.m = m
        self.n = n
        super().__init__(system_size, "grid")

    def get_topology(self):
        graph = nx.grid_2d_graph(m=self.m, n=self.n)
        graph = nx.convert_node_labels_to_integers(graph)
        graph = [list(e) for e in graph.edges]
        to_add = []
        for e in graph:
            i, j = e
            to_add.append([j, i])
        graph = graph + to_add
        return graph


class SquareGrid(Grid):
    def __init__(self, system_size: int):
        if not np.sqrt(system_size).is_integer():
            exit("The system size {} is not valid for square grid topology.".format(self.system_size))

        n = int(np.sqrt(system_size))
        super().__init__(system_size, m=n, n=n)


class LineArchitecture(Architecture):
    def __init__(self, system_size: int):
        super().__init__(system_size, "line")

    def get_topology(self):
        graph = nx.path_graph(self.system_size)
        return [list(e) for e in graph.edges]


class HeavyHexArchitecture(Architecture):
    def __init__(self, system_size: int):
        available_system_sizes = {5, 7, 16, 27, 65, 127}
        if system_size not in available_system_sizes:
            exit("System size {} not available in IBM's heavyhex devices. Available system sizes are: {}".format(
                system_size, available_system_sizes))
        super().__init__(system_size, "heavyhex")

    def get_topology(self):
        if self.system_size == 65:
            return self.get_hummingbird_topology()
        available_systems = {5: FakeQuitoV2(), 7: FakeLagosV2(), 16: FakeGuadalupeV2(), 27: FakeMumbaiV2(),
                             127: FakeWashingtonV2()}
        coupling_list = list(available_systems[self.system_size].coupling_map.get_edges())
        coupling_list = [list(t) for t in coupling_list]
        return coupling_list

    def get_hummingbird_topology(self):
        coupling_list = []
        coupling_list.extend(self._create_connectivity_chain(0, 9))
        coupling_list.extend(self._create_connectivity_chain(13, 23))
        coupling_list.extend(self._create_connectivity_chain(27, 37))
        coupling_list.extend(self._create_connectivity_chain(41, 51))
        coupling_list.extend(self._create_connectivity_chain(55, 64))

        starting_sets = [[0, 10, 13], [15, 24, 29], [27, 38, 41], [43, 52, 56]]
        for s in starting_sets:
            q1, q2, q3 = s
            coupling_list.extend(self._create_vertical_chain(s))
            coupling_list.extend(self._create_vertical_chain([q1 + 4, q2 + 1, q3 + 4]))
            coupling_list.extend(self._create_vertical_chain([q1 + 8, q2 + 2, q3 + 8]))

        return coupling_list

    def _create_connectivity_chain(self, begin: int, end: int):
        chain = []
        for idx in range(begin, end):
            chain.append([idx, idx + 1])
            chain.append([idx + 1, idx])
        return chain

    def _create_vertical_chain(self, indices):
        chain = []
        for k in range(len(indices) - 1):
            chain.append([indices[k], indices[k + 1]])
            chain.append([indices[k + 1], indices[k]])
        return chain


class RigettiArchitecture(Architecture):
    def __init__(self, system_size: int, m: int, n: int):
        self.m = m
        self.n = n
        assert system_size == m * n * 8, "System size of ring architecture do not match the number of rows and columns."
        super().__init__(system_size, "rigetti_rings")

    def get_topology(self):
        rows = []
        for row_idx in range(self.m):
            row = self._create_row_rings(row_idx)
            rows.extend(row)

            if row_idx != 0:
                # connect rows with each other
                for ring_idx in range(self.n):
                    prev_node = (ring_idx + row_idx * self.n) * 8
                    curr_node = (ring_idx + (row_idx - 1) * self.n) * 8

                    rows.append([prev_node + 5, curr_node])
                    rows.append([curr_node, prev_node + 5])

                    rows.append([prev_node + 4, curr_node + 1])
                    rows.append([curr_node + 1, prev_node + 4])

        return rows

    def _create_row_rings(self, row_idx: int):
        row_ring = []
        for k in range(self.n):
            # create n rings next to each other
            ring = []
            for i in range(8):
                idx = k * 8 + i
                if i == 7:
                    ring.append([idx, idx - 7])
                    ring.append([idx - 7, idx])
                else:
                    ring.append([idx, idx + 1])
                    ring.append([idx + 1, idx])

            row_ring.extend(ring)

            # append to the latest ring
            if k != 0:
                row_ring.append([(k - 1) * 8 + 2, k * 8 + 7])
                row_ring.append([k * 8 + 7, (k - 1) * 8 + 2])

                row_ring.append([(k - 1) * 8 + 3, k * 8 + 6])
                row_ring.append([k * 8 + 6, (k - 1) * 8 + 3])

        row_ring = [[row_idx * self.n * 8 + x, row_idx * self.n * 8 + y] for x, y in row_ring]
        return row_ring
