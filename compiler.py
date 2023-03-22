from typing import Union

from qiskit.compiler import transpile
from mqt.bench import get_benchmark
from qiskit.providers.fake_provider import FakeGuadalupeV2
import qiskit
import matplotlib.pyplot as plt
from tqdm import tqdm

from architectures import Architecture
from mappings import InitialLayout, RandomInitialLayout


class Compiler:
    def __init__(self, arc: Union[qiskit.providers.BackendV2, Architecture], alg: str, no_qubits: int):
        self.arc = arc
        self.no_qubits = no_qubits
        self.alg = alg

        self.coupling_map = arc.coupling_map
        self.qc = get_benchmark(self.alg, "alg", self.no_qubits)
        self.gate_counts = None

    def compile(self, layout_provider: InitialLayout, opt_level: int = 0):
        initial_layout = layout_provider.get_virtual_layout()
        qc_transpiled = transpile(self.qc, initial_layout=initial_layout,
                                  coupling_map=self.coupling_map, optimization_level=opt_level)
        self.gate_counts = qc_transpiled.count_ops()


def main():
    arc = FakeGuadalupeV2()
    no_qubits = 10
    compiler = Compiler(arc=arc, alg="dj", no_qubits=no_qubits)

    results = []
    with tqdm(total=100, desc="Compiling circuits", bar_format="{l_bar}{bar} [ time left: {remaining} ]",
              colour="blue") as pbar:
        for i in range(100):
            initial_layout = RandomInitialLayout(no_virt_qubits=no_qubits, no_phys_qubits=arc.num_qubits)
            compiler.compile(initial_layout)
            results.append((compiler.gate_counts["swap"], initial_layout.get_virtual_layout()))
            pbar.update(1)

    plt.plot([res[0] for res in results])
    plt.show()

    best_10 = sorted(results, key=lambda x: x[0])[:5]
    print(best_10)


if __name__ == "__main__":
    main()
