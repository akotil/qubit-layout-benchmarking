import pickle
from typing import List, Dict

from matplotlib import pyplot as plt
from mqt.bench import get_benchmark

import architectures
from compiler import Compiler
from mappings import LinePlacementLayout, GraphPlacementLayout, BestLayout, LayoutByExhaustiveSearch, WorstLayout, \
    InitialLayout


def calculate_results(layouts, algos, arcs:List[architectures.Architecture], max_seed):
    for alg in algos:
        for lay in layouts:
            for arc in arcs:
                circ = get_benchmark(alg, "indep", arc.system_size)
                circ.remove_final_measurements()
                compiler = Compiler(arc, circ, arc.system_size)
                l = lay(arc.system_size, arc.system_size, arc, circ)
                for s in range(max_seed):
                    # Route the circuit and get results
                    if isinstance(l, LayoutByExhaustiveSearch):
                        l.seed = s
                    print("Compiling backend {} with layout {} and system size {}.".format(arc.name, l.name,
                                                                                           arc.system_size))
                    qc_transpiled = compiler.compile(l, seed=s)
                    filename = "transpiled_qc_bins/{}_{}_{}_{}_{}.pickle".format(l.name, arc.system_size, circ.name, arc.name,
                                                                                 s)
                    pickle.dump(qc_transpiled, open(filename, "wb"))


def load_and_plot_results(layouts: List[str], algos: List[str], arcs:List[architectures.Architecture], max_seed: int):
    result_dict = {}
    for alg in algos:
        for lay in layouts:
            result_dict[lay] = dict()
            result_dict[lay][alg] = dict()
            for arc in arcs:

                best_of_seeds = float("inf")
                worst_of_seeds = float("-inf")
                for s in range(max_seed):
                    filename = "transpiled_qc_bins/{}_{}_{}_{}_{}.pickle".format(lay, arc.system_size, alg,
                                                                             arc.name,
                                                                             s)
                    try:
                        with open(filename, 'rb') as handle:
                            transpiled_qc = pickle.load(handle)
                    except:
                        print("File {} does not exist!".format(filename))

                    swap_count = transpiled_qc.count_ops()["swap"]
                    if swap_count < best_of_seeds:
                        best_of_seeds = swap_count

                    if swap_count > worst_of_seeds:
                        worst_of_seeds = swap_count
                if lay == "WorstLayout":
                    result_dict[lay][alg][arc.name] = worst_of_seeds
                else:
                    result_dict[lay][alg][arc.name] = best_of_seeds

    # TODO: Plot with pandas?

arc_grid_structure = {4:(2,2), 6:(2,3), 8:(2,4), 9:(3,3)}
architectures = [architectures.Grid(4, 2, 2), architectures.Grid(6, 2, 3), architectures.Grid(8, 2, 4),
                 architectures.Grid(9, 3, 3), architectures.HeavyHexArchitecture(7),
                 architectures.HeavyHexArchitecture(16), architectures.HeavyHexArchitecture(27)]
# TODO: For Sabre, you need to route and see
layouts = {"LinePlacement": LinePlacementLayout, "GraphPlacement": GraphPlacementLayout, "BestLayout": BestLayout,
           "WorstLayout": WorstLayout}
algorithms = ["dj"]

calculate_results(layouts.values(), algorithms, architectures, max_seed=10)
load_and_plot_results(layouts.keys(), algorithms, architectures, max_seed=10)

'''
x_ticks = arc_grid_structure.keys()
for k in r.keys():
    plt.plot(x_ticks, r[k], label=k, marker="o")

plt.xticks(list(x_ticks))
plt.legend()
plt.show()
'''

# TODO: Theoretical background of line and graph placement
# TODO: Leave out one of qiskit's graphplacement and tket graphplacement
# TODO: Is Qiskit using SABRE as default?


# TODO: Add SabreLayout
# TODO: Add TrivialLayout
# TODO: More data points
# TODO: Bigger simulations without best and worst
# Algos: Grover, arithmetic, vqe or qaoa mit maxcut (near team algos)
# Architekturen: Heavyhex from IBM, rigetti rings
# Everything with depth also
# Fidelity?


# Add classical runtime dimension to the plots -> find a tradeoff between 'how good the mapping is' and how fast it is