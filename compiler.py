from typing import Union

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeGuadalupeV2
import qiskit

from architectures import Architecture
from mappings import InitialLayout, QiskitSabreLayout


class Compiler:
    def __init__(self, arc: Union[qiskit.providers.BackendV2, Architecture], circuit: QuantumCircuit, no_qubits: int):
        self.arc = arc
        self.no_qubits = no_qubits
        self.circ = circuit

        self.coupling_map = arc.coupling_map
        self.gate_counts = None

    def compile(self, layout_provider: Union[InitialLayout, tuple], seed:int = None, opt_level: int = 0):
        layout_method = None
        if isinstance(layout_provider, InitialLayout):
            initial_layout = layout_provider.get_virtual_layout()
        elif isinstance(layout_provider, QiskitSabreLayout):
            initial_layout = None
            layout_method = "sabre"
        else:
            initial_layout = layout_provider

        qc_transpiled = transpile(self.circ, initial_layout=initial_layout, layout_method = layout_method,
                                  coupling_map=self.coupling_map, optimization_level=opt_level, routing_method="sabre", seed_transpiler=seed)
        self.gate_counts = qc_transpiled.count_ops()
        return qc_transpiled

