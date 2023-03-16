from abc import abstractmethod, ABC
import random
from typing import Optional, Union

import qiskit
from mqt.bench import get_benchmark
from pytket._tket.circuit import UnitType, Node
from pytket._tket.passes import PlacementPass, NaivePlacementPass, SequencePass
from pytket._tket.predicates import ConnectivityPredicate, CompilationUnit
from pytket.extensions.qiskit import IBMQBackend, tk_to_qiskit, qiskit_to_tk
from pytket.placement import LinePlacement, GraphPlacement

from qiskit.providers.fake_provider import FakeGuadalupeV2

from pytket.architecture import Architecture
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers

import architectures


class InitialLayout(ABC):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int):
        """
        The abstract base class for all initial layout implementations.

        :param no_virt_qubits: The number of virtual qubits in the quantum circuit.
        :param no_phys_qubits: The number of physical qubits in the QPU.
        """

        self.no_virt_qubits = no_virt_qubits
        self.no_phys_qubits = no_phys_qubits
        assert self.no_phys_qubits >= self.no_virt_qubits, "The size of the architecture cannot be smaller than" \
                                                           " the circuit size."

        self.virtual_layout = None
        self.physical_layout = None

    @abstractmethod
    def get_physical_layout(self):
        pass

    @abstractmethod
    def get_virtual_layout(self):
        pass

######################## Random Initial Layout ########################

class RandomInitialLayout(InitialLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, seed: int = None):
        """
        RandomInitialLayout is responsible for creating randomized layouts.
        Randomization can be fixed by a seed per initialization.
        """
        self.seed = seed
        self.v2p: dict[int, int] = None
        self.p2v: dict[int, Optional[int]] = None
        super().__init__(no_virt_qubits, no_phys_qubits)

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
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, method :str, backend: Union[qiskit.providers.BackendV2, Architecture] = None,
                 qc: QuantumCircuit = None):
        """
        TketPlacementLayout is an abstract class for Tket-specific placements.
        """
        self.v2p: dict[int, int] = None
        self.p2v: dict[int, Optional[int]] = None
        self.backend = backend
        self.qc = qc
        self.v2p: dict[int, int] = None
        self.p2v: dict[int, Optional[int]] = None
        self.method = method
        self.arc = Architecture(
            connections=self.backend.coupling_map)  # TODO: Generalize to other arcs too. When using qiskit,
        # this becomes list(self.backend.coupling_map.get_edges()). The class should only receive couplings.
        super().__init__(no_virt_qubits, no_phys_qubits)

    def get_physical_layout(self) -> list[Optional[int]]:
        pass

    def get_virtual_layout(self) -> list[int]:
        tket_qc = qiskit_to_tk(self.qc)
        if self.method == "LinePlacement":
            initial_placement_pass = PlacementPass(LinePlacement(self.arc))
        elif self.method == "GraphPlacement":
            initial_placement_pass = PlacementPass(GraphPlacement(self.arc))
        else:
            exit("{} is not a valid placement method for Tket.".format(self.method))

        naive_placement_pass = NaivePlacementPass(self.arc)
        cu = CompilationUnit(tket_qc)

        # Apply a placement method first and then initialize the unlabeled qubits with naive approach.
        seq_pass = SequencePass([initial_placement_pass, naive_placement_pass])
        seq_pass.apply(cu)
        print(cu.final_map)

class LinePlacementLayout(TketPlacementLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, backend: Union[qiskit.providers.BackendV2, Architecture] = None,
                 qc: QuantumCircuit = None):
        """
        LinePlacementLayout delegates to PyTket's LinePlacement.
        """
        super().__init__(no_virt_qubits, no_phys_qubits, "LinePlacement", backend, qc)


class GraphPlacementLayout(TketPlacementLayout):
    def __init__(self, no_virt_qubits: int, no_phys_qubits: int, backend: IBMQBackend = None,
                 qc: QuantumCircuit = None):
        """
        GraphPlacementLayout delegates to PyTket's GraphPlacement.
        """
        super().__init__(no_virt_qubits, no_phys_qubits, "GraphPlacement", backend, qc)


######################## Qiskit's Initial Layouts ########################


no_qubits=9
backend = architectures.SquareGrid(no_qubits)
circ = get_benchmark("dj", "indep", 9)
#circ = RemoveBarriers()(circ)
circ.remove_final_measurements()

layout = GraphPlacementLayout(9, no_qubits, backend, circ)
layout.get_virtual_layout()
