from typing import Union

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import FakeGuadalupeV2
import qiskit

from architectures import Architecture
from mappings import InitialLayout


class Compiler:
    def __init__(self, arc: Union[qiskit.providers.BackendV2, Architecture], circuit: QuantumCircuit, no_qubits: int):
        self.arc = arc
        self.no_qubits = no_qubits
        self.circ = circuit

        self.coupling_map = arc.coupling_map
        self.gate_counts = None

    def compile(self, layout_provider: Union[InitialLayout, tuple], seed:int = None, opt_level: int = 0):
        if isinstance(layout_provider, InitialLayout):
            initial_layout = layout_provider.get_virtual_layout()
        else:
            initial_layout = layout_provider

        #print("Compiling started for layout={} and system size={}".format(layout_provider.name, self.no_qubits))

        qc_transpiled = transpile(self.circ, initial_layout=initial_layout,
                                  coupling_map=self.coupling_map, optimization_level=opt_level, routing_method="sabre", seed_transpiler=seed)
        self.gate_counts = qc_transpiled.count_ops()
        return qc_transpiled

