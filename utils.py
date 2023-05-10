import random

import numpy as np
from joblib import Parallel, delayed
from mqt.bench.utils import get_examplary_max_cut_qp
from qiskit import transpile
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from tqdm import tqdm

def get_results_dict(perms, qc, coupling, seed):

    inputs = [(perm, qc, coupling) for perm in perms]

    def func(inp):
        perm, qc, coupling = inp
        r = transpile(qc, coupling_map=coupling, initial_layout=perm, optimization_level=0,
                      routing_method="sabre", seed_transpiler=seed).count_ops()
        if "swap" in r:
            swap_count = r["swap"]
        else:
            swap_count = 0
        return swap_count

    parallel = Parallel(n_jobs=8)
    outputs = parallel(delayed(func)(inp) for inp in tqdm(inputs))
    result_dict = dict(zip([inp[0] for inp in inputs], outputs))

    return result_dict


def get_qaoa_circuit(num_qubits: int):
    qp = get_examplary_max_cut_qp(num_qubits)
    qaoa = QAOA(sampler=Sampler(), reps=2, optimizer=SLSQP(maxiter=0))
    qaoa_result = qaoa.compute_minimum_eigenvalue(qp.to_ising()[0])
    qc = qaoa.ansatz.bind_parameters(qaoa_result.eigenvalue)
    qc.name = "qaoa"
    qc = qc.decompose().decompose()
    return qc

def get_vqe_circuit(num_qubits: int):
    # TODO: Look at the circuit again
    ansatz = RealAmplitudes(num_qubits, reps=2)
    ansatz.bind_parameters(np.random.uniform(low=-2, high=2, size=(ansatz.num_parameters,)))
    ansatz.name = "vqe"
    ansatz = ansatz.decompose()
    return ansatz


def find_layout_bounds(result_dict: dict):
    # Find the best layout
    min = float("inf")
    max = float('-inf')
    best_perm = None
    worst_perm = None
    for p in result_dict:
        if result_dict[p] < min:
            min = result_dict[p]
            best_perm = p

        if result_dict[p] > max:
            max = result_dict[p]
            worst_perm = p

    return best_perm, worst_perm