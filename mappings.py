import copy
import itertools
import pickle
from abc import abstractmethod, ABC
import random

from typing import Optional, Union

import qiskit
from pytket._tket.passes import PlacementPass, NaivePlacementPass, SequencePass
from pytket._tket.predicates import CompilationUnit
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.placement import LinePlacement, GraphPlacement

from qiskit.providers.fake_provider import FakeGuadalupeV2

from pytket.architecture import Architecture
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import TrivialLayout

import architectures
import utils


class InitialLayout(ABC):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, name: str):
        """
        The abstract base class for all initial layout implementations.

        :param no_virt_qubits: The number of virtual qubits in the quantum circuit.
        :param no_phys_qubits: The number of physical qubits in the QPU.
        :param name: Unique name of the layout method.
        """

        self.no_virt_qubits = no_virt_qubits
        self.no_phys_qubits = no_phys_qubits
        assert self.no_phys_qubits >= self.no_virt_qubits, "The size of the architecture cannot be smaller than" \
                                                           " the circuit size."

        self.name = name
        self.virtual_layout = None
        self.physical_layout = None

    @abstractmethod
    def get_physical_layout(self):
        pass

    @abstractmethod
    def get_virtual_layout(self):
        pass

class LayoutByExhaustiveSearch(InitialLayout):

    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, backend: architectures.Architecture,
                 qc: QuantumCircuit, method):
        '''
        This layout class is responsible for finding the best layout in terms of swap count by exhaustive search.

        :param no_virt_qubits: The number of virtual qubits in the quantum circuit.
        :param no_phys_qubits: The number of physical qubits in the QPU.
        :param routing_method: The routing method used to obtain swap gates.
        '''

        super().__init__(no_virt_qubits, no_phys_qubits, method)
        self.v2p: dict[int, int] = None  # TODO: This should be in super class
        self.p2v: dict[int, Optional[int]] = None
        self.qc = qc
        self.backend = backend
        self.coupling = backend.coupling_map

        self.result_dict = None
        self.worst_layout = None
        self.best_layout = None
        self.seed = None


    def get_physical_layout(self):
        pass

    def get_virtual_layout(self):
        if self.best_layout is not None and self.worst_layout is not None and self.seed is None:
            return self.best_layout, self.worst_layout

        pickle_data = self.load_from_pickle(self.qc.name, "_" + str(self.seed) + "_" + self.backend.name)
        if pickle_data is not None:
            best_perm, worst_perm = utils.find_layout_bounds(pickle_data)
            self.best_layout = best_perm
            self.worst_layout = worst_perm
            return self.best_layout, self.worst_layout

        print("pickle data was none")
        perms = list(itertools.permutations(list(range(self.no_phys_qubits))))
        result_dict = utils.get_results_dict(perms, self.qc, self.coupling, self.seed)
        self.result_dict = result_dict

        best_perm, worst_perm = utils.find_layout_bounds(self.result_dict)
        self.best_layout = best_perm
        self.worst_layout = worst_perm
        self.save(result_dict, self.qc.name, "_" + str(self.seed) + "_" + self.backend.name)
        return self.best_layout, self.worst_layout


    def save(self, result_dict, qc_name: str, arc_name: str):
        filename = "layout_bins/{}_{}_{}_{}.pickle".format("ExhaustiveSearch", self.no_phys_qubits, qc_name, arc_name)
        with open(filename, 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(self, qc_name: str, arc_name: str):
        filename = "layout_bins/{}_{}_{}_{}.pickle".format("ExhaustiveSearch", self.no_phys_qubits, qc_name, arc_name)
        try:
            with open(filename, 'rb') as handle:
                result_dict = pickle.load(handle)
                return result_dict
        except:
            print("File {} does not exist. Performing layout search from start.".format(filename))
            return None


class BestLayout(LayoutByExhaustiveSearch):

    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, backend: architectures.Architecture,
                 qc: QuantumCircuit):
        super().__init__(no_virt_qubits, no_phys_qubits, backend, qc, "BestLayout")


    def get_physical_layout(self):
        pass

    def get_virtual_layout(self):
        super().get_virtual_layout()
        return self.best_layout


class WorstLayout(LayoutByExhaustiveSearch):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, backend: architectures.Architecture,
                 qc: QuantumCircuit):
        super().__init__(no_virt_qubits, no_phys_qubits, backend, qc, "WorstLayout")
        self.backend = backend
        self.qc = qc

    def get_physical_layout(self):
        pass

    def get_virtual_layout(self):
        super().get_virtual_layout()
        return self.worst_layout


######################## Random Initial Layout ########################

class RandomInitialLayout(InitialLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, seed: int = None):
        """
        RandomInitialLayout is responsible for creating randomized layouts.
        Randomization can be fixed by a seed per initialization.
        """
        self.seed = seed
        self.v2p: dict[int, int] = None  # TODO: This should be in super class
        self.p2v: dict[int, Optional[int]] = None
        super().__init__(no_virt_qubits, no_phys_qubits, "random")


    def get_physical_layout(self) -> list[Optional[int]]:
        """
        Creates a mapping from physical qubits to virtual qubits randomly.

        :return: List of size self.no_phys_qubits where every entry corresponds to the physical qubit to which
        the virtual qubit is assigned. The virtual qubits are assumed to be ordered by the list indices.
        """
        if self.physical_layout is not None:
            return self.physical_layout
        else:
            physical_layout = list[sorted(self.p2v).values()]
            return physical_layout

    def get_virtual_layout(self) -> list[int]:
        """
        Creates a mapping from virtual qubits to physical qubits randomly.

        :return: List of size self.no_virt_qubits where every entry corresponds to the physical qubit to which
        the virtual qubit is assigned. The virtual qubits are assumed to be ordered by the list indices.
        """
        if self.virtual_layout is not None:
            return self.virtual_layout

        else:
            if self.seed is not None:
                random.seed(self.seed)
            virtual_layout = random.sample(range(self.no_phys_qubits), self.no_virt_qubits)
            self.v2p = dict(zip(range(self.no_virt_qubits), virtual_layout))
            self.p2v = self._set_p2v_from_v2p()
            return virtual_layout

    def _set_p2v_from_v2p(self) -> dict[int, Optional[int]]:
        p2v = dict()
        for key_v in self.v2p:
            p2v[self.v2p[key_v]] = key_v

        for idx in range(self.no_phys_qubits):
            if idx not in p2v:
                p2v[idx] = None

        return p2v

    def set_layout_from_physical_qubits(self, permutation: list[Optional[int]]):
        assert len(permutation) == self.no_phys_qubits
        raise NotImplementedError

    def set_layout_from_virtual_qubits(self, permutation: list[int]):
        assert len(permutation) == self.no_virt_qubits
        raise NotImplementedError


######################## Tket's Initial Layouts ########################

class TketPlacementLayout(InitialLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, method: str,
                 backend: Union[qiskit.providers.BackendV2, Architecture] = None,
                 qc: QuantumCircuit = None):
        super().__init__(no_virt_qubits, no_phys_qubits, method)
        """
        TketPlacementLayout is an abstract class for Tket-specific placements.
        """
        self.v2p: dict[int, int] = None
        self.p2v: dict[int, Optional[int]] = None
        self.backend = backend
        self.qc = qc
        self.tket_qc = qiskit_to_tk(self.qc)

        self.v2p: dict[int, int] = None
        self.p2v: dict[int, Optional[int]] = None
        self.method = method
        self.arc = Architecture(
            connections=self.backend.coupling_map)
        # TODO: Generalize to other arcs too. When using qiskit,
        # this becomes list(self.backend.coupling_map.get_edges()). The class should only receive couplings.

    def get_physical_layout(self) -> list[Optional[int]]:
        pass

    def get_virtual_layout(self) -> list[int]:

        if self.virtual_layout is not None:
            return self.virtual_layout

        if self.method == "LinePlacement":
            initial_placement_pass = PlacementPass(LinePlacement(self.arc))
        elif self.method == "GraphPlacement":
            initial_placement_pass = PlacementPass(GraphPlacement(self.arc))
        else:
            exit("{} is not a valid placement method for Tket.".format(self.method))

        naive_placement_pass = NaivePlacementPass(self.arc)

        cu = CompilationUnit(copy.deepcopy(self.tket_qc))
        # TODO: tket routing placement + dynamic routing vs. placement + naive + routing
        # Wenn der Unterschied recht groß ist, dann backtracking sonst lassen
        # TODO: Optimierung ausschalten

        # Apply a placement method first and then initialize the unlabeled qubits with naive approach.
        seq_pass = SequencePass([initial_placement_pass, naive_placement_pass])
        seq_pass.apply(cu)

        virtual_layout = []
        for node in cu.final_map.values():
            virtual_layout.append(node.index[0])
        self.virtual_layout = virtual_layout
        return virtual_layout


class LinePlacementLayout(TketPlacementLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int,
                 backend: Union[qiskit.providers.BackendV2, Architecture] = None,
                 qc: QuantumCircuit = None):
        """
        LinePlacementLayout delegates to PyTket's LinePlacement.
        """
        super().__init__(no_virt_qubits, no_phys_qubits, "LinePlacement", backend, qc)


class GraphPlacementLayout(TketPlacementLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int,
                 backend: Union[qiskit.providers.BackendV2, Architecture],
                 qc: QuantumCircuit = None):
        """
        GraphPlacementLayout delegates to PyTket's GraphPlacement.
        """
        super().__init__(no_virt_qubits, no_phys_qubits, "GraphPlacement", backend, qc)

######################## Qiskit's Initial Layouts ########################

class QiskitTrivialLayout(InitialLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int,
                 backend: Union[qiskit.providers.BackendV2, Architecture] = None, qc: QuantumCircuit = None):
        super().__init__(no_virt_qubits, no_phys_qubits, "TrivialLayout")
        self.backend = backend
        self.qc = qc

    def get_physical_layout(self) -> list[Optional[int]]:
        return TrivialLayout(self.backend.coupling_map).coupling_map # TODO: What format does this have?

    def get_virtual_layout(self) -> list[int]:
        pass


class QiskitSabreLayout(InitialLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int,
                 backend: Union[qiskit.providers.BackendV2, Architecture] = None, qc: QuantumCircuit = None):
        super().__init__(no_virt_qubits, no_phys_qubits, "SabreLayout")
        self.backend = backend
        self.qc = qc

    def get_physical_layout(self) -> list[Optional[int]]:
        pass

    def get_virtual_layout(self) -> list[int]:
        pass
