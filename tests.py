from mqt.bench import get_benchmark

import architectures
from compiler import Compiler
from mappings import LinePlacementLayout, GraphPlacementLayout, BestLayout


arc_grid_structure = {4:(2,2), 6: (2,3), 8: (2,4), 9: (3,3)}
layouts = [LinePlacementLayout, GraphPlacementLayout, BestLayout]
algorithm = "dj"

for sys_size in arc_grid_structure:
    # Specify the architecture
    backend = architectures.Grid(sys_size, *arc_grid_structure[sys_size])
    circ = get_benchmark("dj", "indep", sys_size)
    circ.remove_final_measurements()

    # Specify the layout
    for layout in layouts:
        virt_layout = layout(sys_size, sys_size, backend, circ).get_virtual_layout()
        #layout.save("dj", "grid")

        # Route the circuit and get results
        compiler = Compiler(backend, algorithm, sys_size)
        compiler.compile(virt_layout)
        print(compiler.gate_counts)



# TODO: Theoretical background of line and graph placement
# TODO: Leave out one of qiskit's graphplacement and tket graphplacement
# TODO: Is Qiskit using SABRE as default?
# TODO: How large is the difference between the best and the worst layout?
