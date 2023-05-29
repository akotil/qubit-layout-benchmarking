from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
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
            exit("The system size {} is not valid for square grid topology.".format(system_size))

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
        available_system_sizes = {5, 7, 16, 27, 65, 127, 433}
        if system_size not in available_system_sizes:
            exit("System size {} not available in IBM's heavyhex devices. Available system sizes are: {}".format(
                system_size, available_system_sizes))
        super().__init__(system_size, "heavyhex")

    def get_topology(self):
        if self.system_size == 65:
            return self.get_hummingbird_topology()
        elif self.system_size == 433:
            return self.get_osprey_topology()

        available_systems = {5: FakeQuitoV2(), 7: FakeLagosV2(), 16: FakeGuadalupeV2(), 27: FakeMumbaiV2(),
                             127: FakeWashingtonV2()}
        coupling_list = list(available_systems[self.system_size].coupling_map.get_edges())
        coupling_list = [list(t) for t in coupling_list]
        return coupling_list

    def get_osprey_topology(self):
        coupling_list = []
        no_rows = 12
        prev_path = None
        intermediate_nodes = list(range(433 -no_rows * 7, 433))
        cursor = 0
        for k in range(no_rows):
            if prev_path is None:
                prev_path = nx.path_graph(27)
                coupling_list.extend([list(e) for e in prev_path.edges])

            curr_path = nx.path_graph(27)
            curr_path = nx.relabel_nodes(curr_path, lambda x: x + (k+1)*27)
            coupling_list.extend([list(e) for e in curr_path.edges])

            curr_path_nodes = list(curr_path.nodes)
            prev_path_nodes = list(prev_path.nodes)

            if k % 2 == 0:
                connection_range = (0, 25, 4)
            else:
                connection_range = (2, 27, 4)

            for m in range(*connection_range):
                intermediate_node = intermediate_nodes[cursor]
                print(intermediate_node)
                coupling_list.append([curr_path_nodes[m], intermediate_node])
                coupling_list.append([intermediate_node, curr_path_nodes[m]])
                coupling_list.append([prev_path_nodes[m], intermediate_node])
                coupling_list.append([intermediate_node, prev_path_nodes[m]])
                cursor += 1
            prev_path = curr_path

            if k == 1 or k ==2:
                G = nx.Graph()
                G.add_edges_from(coupling_list)
                #pos = nx.planar_layout(G)
                pos = nx.nx_agraph.graphviz_layout(G)
                nx.draw(G, pos=pos)
                plt.show()


        # remove two nodes
        coupling_list.remove([25,26])
        coupling_list.remove([no_rows*27, no_rows*27+1])

        # make topology symmetric
        symmetric_edges = []
        for e in coupling_list:
            e1, e2 = e
            symmetric_edges.append([e2, e1])
        coupling_list.extend(symmetric_edges)

        return coupling_list

    def _get_row(self, row_idx: int, m: int, n: int):
        row = []
        # todo: is it symmetric?
        # 12 * 7 intermediate nodes
        # 433 -12 * 7 path nodes
        path = nx.path_graph(27)



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
