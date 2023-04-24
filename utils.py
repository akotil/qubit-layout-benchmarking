from qiskit import transpile
from tqdm import tqdm


def get_results_dict(perms, qc, coupling, seed):

    inputs = [(perm, qc, coupling) for perm in perms]
    result_dict = {}

    for inp in tqdm(inputs, total=len(inputs),
                           desc="Compiling circuits", bar_format="{l_bar}{bar} [ time left: {remaining} ]",
                            colour="blue"):
        perm, qc, coupling = inp
        r = transpile(qc, coupling_map=coupling, initial_layout=perm, optimization_level=0,
                      routing_method="sabre", seed_transpiler=seed).count_ops()["swap"]
        result_dict[perm] = r

    return result_dict


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

    # TODO: Figure out experimentally a reasonable number of runs
    return best_perm, worst_perm