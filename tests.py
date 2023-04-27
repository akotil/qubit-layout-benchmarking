import pickle
from typing import List, Dict

from matplotlib import pyplot as plt
from mqt.bench import get_benchmark

import architectures
import utils
from compiler import Compiler
from mappings import LinePlacementLayout, GraphPlacementLayout, BestLayout, LayoutByExhaustiveSearch, WorstLayout, \
    InitialLayout, QiskitTrivialLayout, QiskitSabreLayout

import os.path


def calculate_results(layouts, algos, arcs:List[architectures.Architecture], max_seed):
    for alg in algos:
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


def load_and_plot_results(layouts: List[str], algos: List[str], arcs:List[architectures.Architecture], benchmark_entity:str, max_seed: int):
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
                                                                             arc.name, s)

                    # Load the transpiled circuit if it exists, exit otherwise
                    try:
                        with open(filename, 'rb') as handle:
                            transpiled_qc = pickle.load(handle)
                    except:
                        print("File {} does not exist!".format(filename))

                    assert benchmark_entity in ["swap", "depth"]
                    if benchmark_entity == "swap" and "swap" in transpiled_qc.count_ops():
                        count = transpiled_qc.count_ops()["swap"]
                    elif benchmark_entity == "depth":
                        count = transpiled_qc.depth()
                    else:
                        print("Benchmark entity {} doesn't exist for {}_{}_{}_{}_{}"
                              .format(benchmark_entity,lay, arc.system_size, alg, arc.name,s))
                        count = 0

                    if count < best_of_seeds:
                        best_of_seeds = count
                    if count > worst_of_seeds:
                        worst_of_seeds = count

                if lay == "WorstLayout":
                    result_dict[lay][alg][arc.system_size] = worst_of_seeds
                else:
                    result_dict[lay][alg][arc.system_size] = best_of_seeds

    plot_dict = {}
    for lay in layouts:
        plot_dict[lay] = []
        # TODO: Find a better way to sort
        for arc in sorted(arcs, key=lambda x: x.system_size):
            # Fix the algorithm
            plot_dict[lay].append(result_dict[lay]["vqe"][arc.system_size])
        plt.plot([arc.system_size for arc in sorted(arcs, key=lambda x: x.system_size)], plot_dict[lay],
                 label=lay, marker="o")

    plt.grid()
    plt.legend()
    title = "{} comparison for {}".format(benchmark_entity, algos)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    archs = [
                     architectures.HeavyHexArchitecture(7),
                     architectures.HeavyHexArchitecture(16),
                     architectures.HeavyHexArchitecture(27),
                     ]

    for i in range(2, 13):
        archs.append(architectures.SquareGrid(i**2))

    layouts = {"LinePlacement": LinePlacementLayout, "GraphPlacement": GraphPlacementLayout,
               #"BestLayout": BestLayout,
               #"WorstLayout": WorstLayout,
               "TrivialLayout": QiskitTrivialLayout,
               "SabreLayout": QiskitSabreLayout}
    algorithms = ["vqe"]

    max_seed = 10
    entity = "depth"
    calculate_results(layouts.values(), algorithms, archs, max_seed=max_seed)
    load_and_plot_results(layouts.keys(), algorithms, archs, benchmark_entity="depth", max_seed=max_seed)

    # TODO: Theoretical background of line and graph placement
    # TODO: Leave out one of qiskit's graphplacement and tket graphplacement
    # TODO: Is Qiskit using SABRE as default?
    # TODO: Bigger simulations without best and worst
    # Add classical runtime dimension to the plots -> find a tradeoff between 'how good the mapping is' and how fast it is