import pickle
import statistics
from typing import List

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from mqt.bench import get_benchmark

import architectures
import utils
from compiler import Compiler
from mappings import LinePlacementLayout, GraphPlacementLayout, BestLayout, LayoutByExhaustiveSearch, WorstLayout, \
    QiskitTrivialLayout, QiskitSabreLayout

import os.path


def calculate_results(layouts, alg, arcs:List[architectures.Architecture], max_seed):
    for lay in layouts:
        for arc in arcs:
            # Construct the circuit
            if alg == "qaoa":
                circ = utils.get_qaoa_circuit(arc.system_size)
            elif alg == "vqe":
                circ = utils.get_vqe_circuit(arc.system_size)
            else:
                circ = get_benchmark(alg, "indep", arc.system_size)

            circ.remove_final_measurements()

            compiler = Compiler(arc, circ, arc.system_size)
            l = lay(arc.system_size, arc.system_size, arc, circ)

            for s in range(max_seed):
                # Route the circuit and get results
                if isinstance(l, LayoutByExhaustiveSearch):
                    l.seed = s
                filename = "transpiled_qc_bins/{}_{}_{}_{}_{}.pickle".format(l.name, arc.system_size, circ.name, arc.name,
                                                                             s)
                if not os.path.isfile(filename):
                    print("Compiling backend {} with layout {} and system size {} for seed {}.".format(arc.name, l.name,
                                                                                           arc.system_size, s))
                    qc_transpiled = compiler.compile(l, seed=s)
                    pickle.dump(qc_transpiled, open(filename, "wb"))
                else:
                    print("Using already compiled circuit for backend {} with layout {} and system size {} for seed {}."
                          .format(arc.name, l.name, arc.system_size, s))


def load_results(layouts: List[str], alg: str, arcs:List[architectures.Architecture], max_seed: int):
    result_dict = {}
    for lay in layouts:
        result_dict[lay] = dict()
        result_dict[lay][alg] = dict()
        for arc in arcs:
            swap_results = []
            depth_results = []
            for s in range(max_seed):
                filename = "transpiled_qc_bins/{}_{}_{}_{}_{}.pickle".format(lay, arc.system_size, alg,
                                                                         arc.name, s)

                # Load the transpiled circuit if it exists, exit otherwise
                try:
                    with open(filename, 'rb') as handle:
                        transpiled_qc = pickle.load(handle)
                except:
                    print("File {} does not exist!".format(filename))

                depth_count = transpiled_qc.depth()
                if "swap" in transpiled_qc.count_ops():
                    swap_count = transpiled_qc.count_ops()["swap"]
                else:
                    swap_count = 0

                swap_results.append(swap_count)
                depth_results.append(depth_count)

            if lay == "WorstLayout":
                result_dict[lay][alg][arc.system_size] = {"swap": max(swap_results), "depth": max(depth_results)}

            elif lay == "BestLayout":
                result_dict[lay][alg][arc.system_size] = {"swap": min(swap_results), "depth": min(depth_results)}
            else:
                result_dict[lay][alg][arc.system_size] = {"swap": swap_results, "depth": depth_results}

    return result_dict


def plot_results(result_dict: dict, layouts: List[str], arcs: List[architectures.Architecture], benchmark_entity: str, alg:str):
    styling = {"color":{"LinePlacement":"darkkhaki", "GraphPlacement":"palevioletred", "TrivialLayout": "cornflowerblue", "SabreLayout":"lightseagreen"},
               "linestyle": {"LinePlacement":"dashed", "GraphPlacement":"dashdot", "TrivialLayout": "-", "SabreLayout":"dotted"}
    }
    plot_dict = {}
    for lay in layouts:
        plot_dict[lay] = []
        x_arr = [arc.system_size for arc in sorted(arcs, key=lambda x: x.system_size)]
        # TODO: Find a better way to sort
        for arc in sorted(arcs, key=lambda x: x.system_size):
            #plot_dict[lay].append(result_dict[lay][alg][arc.system_size])
            r = result_dict[lay][alg][arc.system_size][benchmark_entity]
            min_val = min(r)
            max_val = max(r)
            avg_val = statistics.fmean(r)
            r_tuple = (min_val, avg_val, max_val)
            plot_dict[lay].append(r_tuple)

        # plot the average values
        plt.scatter(x_arr, [t[1] for t in plot_dict[lay]], marker="o", color=styling["color"][lay])

        # plot the deviation ranges
        if lay not in ["WorstLayout", "BestLayout"]:
            lower = [t[0] for t in plot_dict[lay]]
            upper = [t[2] for t in plot_dict[lay]]
            middle = [t[1] for t in plot_dict[lay]]
            y_l = np.array(middle) - np.array(lower)
            y_u = np.array(upper) - np.array(middle)
            errors = [y_l, y_u]
            plt.errorbar(x_arr, y= middle, yerr=errors, color=styling["color"][lay], linestyle=styling["linestyle"][lay], label=lay)

    plt.grid()
    plt.legend()
    title = "{} comparison for {}".format(benchmark_entity, alg)
    plt.title(title)
    plt.savefig("plots/{}_{}.png".format(alg, arcs[0].name), dpi=1500)
    plt.show()



def get_heavyhex_arcs(system_sizes : List[int]):
    arcs = []
    for size in system_sizes:
        arcs.append(architectures.HeavyHexArchitecture(size))
    return arcs

def get_rigetti_arcs():
    sizes = [(1,1), (1,2), (1,5), (1,6), (2,5), (2,6), (3,5),(3,6), (4,5),(4,6), (5,5), (5,6), (6,6), (7,6), (8,6), (9,6)]
    arcs = []
    for size in sizes:
        m, n = size
        system_size = m * n * 8
        arcs.append(architectures.RigettiArchitecture(system_size=system_size, m=m, n=n))
    return arcs

def get_square_grid_arcs(max_n:int):
    arcs = []
    for i in range(2, max_n+1):
        arcs.append(architectures.SquareGrid(i ** 2))
    return arcs

if __name__ == "__main__":
    archs = []

    ibm_archs = get_heavyhex_arcs([5,7,16,27,65,127])
    rigetti_arcs = get_rigetti_arcs()
    square_arcs = get_square_grid_arcs(20)

    archs.extend(rigetti_arcs)

    layouts = {"LinePlacement": LinePlacementLayout, "GraphPlacement": GraphPlacementLayout,
               #"BestLayout": BestLayout,
               #"WorstLayout": WorstLayout,
               "TrivialLayout": QiskitTrivialLayout,
               "SabreLayout": QiskitSabreLayout}

    # Set the parameters
    algorithm = "vqe"
    max_seed = 10
    entity = "swap"
    calculate_results(layouts.values(), algorithm, archs, max_seed=max_seed)
    result_dict = load_results(layouts.keys(), algorithm, archs, max_seed=max_seed)
    plot_results(result_dict, layouts.keys(), archs, entity, algorithm)