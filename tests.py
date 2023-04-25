import pickle
from typing import List, Dict

import pandas as pd
from matplotlib import pyplot as plt
from mqt.bench import get_benchmark

import architectures
from compiler import Compiler
from mappings import LinePlacementLayout, GraphPlacementLayout, BestLayout, LayoutByExhaustiveSearch, WorstLayout, \
    InitialLayout, QiskitTrivialLayout, QiskitSabreLayout

import os.path



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


def load_and_plot_results(layouts: List[str], algos: List[str], arcs:List[architectures.Architecture], benchmark_measure:str, max_seed: int):
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
                    result_dict[lay][alg][arc.system_size] = worst_of_seeds
                else:
                    result_dict[lay][alg][arc.system_size] = best_of_seeds

    if benchmark_measure=="arc":
        plot_dict = {}
        for lay in layouts:
            plot_dict[lay] = []
            # TODO: Find a better way to sort
            for arc in sorted(arcs, key=lambda x: x.system_size):
                # Fix the algorithm
                print(arc.system_size)
                plot_dict[lay].append(result_dict[lay]["dj"][arc.system_size])
            plt.plot([arc.system_size for arc in sorted(arcs, key=lambda x: x.system_size)], plot_dict[lay],
                     label=lay, marker="o")

        plt.grid()
        plt.legend()
        plt.title(algos)
        plt.show()

arc_grid_structure = {4:(2,2), 6:(2,3), 8:(2,4), 9:(3,3)}
architectures = [architectures.Grid(4, 2, 2), architectures.Grid(6, 2, 3), architectures.Grid(8, 2, 4),
                 architectures.Grid(9, 3, 3), architectures.Grid(12, 3, 4), architectures.Grid(20, 4, 5),
                 architectures.HeavyHexArchitecture(7),
                 architectures.HeavyHexArchitecture(16),
                 architectures.HeavyHexArchitecture(27),
                 architectures.Grid(25, 5, 5), architectures.Grid(30, 6, 5), architectures.Grid(36, 6, 6),
                 architectures.Grid(42, 6, 7), architectures.Grid(49, 7, 7)
                 ]

layouts = {"LinePlacement": LinePlacementLayout, "GraphPlacement": GraphPlacementLayout,
           #"BestLayout": BestLayout,
           #"WorstLayout": WorstLayout,
           "TrivialLayout": QiskitTrivialLayout,
           "SabreLayout": QiskitSabreLayout}
algorithms = ["grover-noancilla"]

calculate_results(layouts.values(), algorithms, architectures, max_seed=10)
load_and_plot_results(layouts.keys(), algorithms, architectures, benchmark_measure="arc", max_seed=10)

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