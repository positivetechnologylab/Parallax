"""
Altered version of graphine, returns params needed for atomic array manipulation. 
"""
import numpy as np
import networkx as nx
from scipy.optimize import dual_annealing

import matplotlib as mpl
import matplotlib.pyplot as mp
import matplotlib.patches as mpatches

import qiskit.qasm3 as qasm3
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_dagdependency, dag_to_circuit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.visualization import dag_drawer
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOpNode, DAGOutNode
from qiskit import transpile

def graphine(input_file):
    params = {
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.texsystem': 'xelatex',
        'pgf.preamble': r'\usepackage{fontspec,physics}',
    }

    # mpl.rcParams.update(params)
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Create a dictionary of all connections in the circuit
    connections = {}

    # Number of qubits in the circuit
    num_qubits = None

    """
    "connections" will contain a set of keys (q1,q2), and a set of values (int number of connections between the two qubits)
    num_qubits = number of qubits (wires) in the circuit
    """
    with open(input_file, 'r') as file:
        qasm = file.readlines()

        # For each row in the QASM file
        for row in qasm:

            #get number of qubits out of the qreg file
            if 'qreg' in row:
                num_qubits = int(row.split('[')[1].split(']')[0])

            # Calculate the number of qubits in the op
            num_op_qubits = row.count('[')

            # If there are more than 1 qubits
            if num_op_qubits > 1:

                # Verify the number of qubits
                assert num_op_qubits == row.count('q')
                assert num_op_qubits == row.count(']')

                # Get list of all the qubits as ints involved in the operation
                qubits = [int(row.split('[')[q+1].split(']')[0]) for q in range(num_op_qubits)]

                # Sort the qubits
                qubits.sort()

                # For all two-qubit combinations within the operation
                for i1 in range(len(qubits)-1):
                    for i2 in range(i1+1, len(qubits)):
                        q1, q2 = qubits[i1], qubits[i2]

                        # If qubits not in the dictionary, create an entry
                        if (q1, q2) not in connections:
                            connections[(q1, q2)] = 0

                        # Increase the weight of the entry by 1
                        connections[(q1, q2)] += 1
    connect_count = connections
    
    """
    Objective function for dual annealing (should find location of a given qubit)
    """
    def score_configuration(x):

        # Get number of qubits and qubit positions
        num_qubits = int(len(x)/2)
        qubit_positions = [(x[i*2], x[i*2+1]) for i in range(num_qubits)]

        # v1 is the edge weights and v2 is the distance for each qubit pair
        v1 = []
        v2 = []

        # For each qubit pair
        for q1 in range(num_qubits-1):
            for q2 in range(q1+1, num_qubits):

                # If the qubits are connected, append edge weight, otherwise 0
                if (q1, q2) in connections.keys():
                    v1.append(connections[(q1, q2)])
                else:
                    v1.append(0)

                # Calculate the distance
                distance = ((qubit_positions[q1][0]-qubit_positions[q2][0])**2 +
                            (qubit_positions[q1][1]-qubit_positions[q2][1])**2)**0.5

                # If the distance is too small, return high score
                if distance < (1 / num_qubits):
                    return 1

                # Append distance
                v2.append(distance)

        # Calculate correlation between v1 and v2 and normalize between 0 and 1
        correlation = np.corrcoef(v1, v2)[0][1]
        correlation = (correlation + 1) / 2

        # Return the correlation
        return correlation

    # Bounds for the position of each qubit
    bounds = (0, 1)
    position_bounds = [bounds for q in range(num_qubits*2)]

    # Run the black box optimization
    results = dual_annealing(score_configuration,
                             bounds=position_bounds,
                             initial_temp=5.e4,
                             maxiter=int(1e4),
                             maxfun=1e8)

    # Get the lowest scoring qubit positions
    qubit_positions = [(results.x[i*2], results.x[i*2+1]) for i in range(num_qubits)]

    # To calculate the radius, get all the qubit distances
    dists = []
    distances = [[0]*num_qubits for q in range(num_qubits)]
    for q1 in range(num_qubits-1):
        for q2 in range(q1+1, num_qubits):

            # Calculate distance
            distance = ((qubit_positions[q1][0]-qubit_positions[q2][0])**2 +
                        (qubit_positions[q1][1]-qubit_positions[q2][1])**2)**0.5

            # Record distance
            distances[q1][q2] = distance
            distances[q2][q1] = distance
            dists.append(distance)

    # Sort the distances
    dists.sort()

    # Select the lowest distance such as all qubits are within reach of one another as the radius
    radius = None
    for dist in dists:

        # All qubits within reach
        qubits_in_range = [0]

        # Maintain a frontier of nodes
        frontier = [0]

        # Until there are qubits in the frontier
        while frontier:

            # New frontier
            new_frontier = []

            # Add all qubits within reach to the new frontier
            for q1 in frontier:
                for q2 in range(num_qubits):
                    if q2 not in qubits_in_range and q1 != q2 and distances[q1][q2] < dist:
                        qubits_in_range.append(q2)
                        new_frontier.append(q2)

            # Reset the frontier
            frontier = new_frontier

        # If all qubits are in reach, select the distance as the radius
        if len(qubits_in_range) == num_qubits:

            edges = []
            for q1 in range(num_qubits-1):
                for q2 in range(q1+1, num_qubits):
                    if distances[q1][q2] < dist:
                        edges.append((q1, q2))

            graph = nx.Graph()
            graph.add_edges_from(edges)

            lengths = dict(nx.all_pairs_shortest_path_length(graph))

            max_length = np.max([list(v1.values()) for v1 in lengths.values()])

            if max_length > num_qubits**0.5:
                continue

            radius = dist
            break

    """
    Draw graph for result
    """
    # Print the lowest score, radius, and final qubit positions
#     print('Score:', results.fun)
#     print('Radius:', radius)
#     print('Positions:', qubit_positions)

#     # Plot the positions on the 1x1 grid
#     fig = mp.figure(figsize=(4, 3.84))
#     fig.subplots_adjust(left=0.145, top=0.97916, right=0.965, bottom=0.125)

#     ax = fig.add_subplot(111)
#     ax.set_axisbelow(True)
#     ax.grid(linestyle=':', color='grey', linewidth=0.5)

#     ax.set_xlim(bounds)
#     ax.set_ylim(bounds)

#     # Draw circles around qubit that are half the radius
#     for pos in qubit_positions:
#         circle = mpatches.Circle(pos, radius/2, facecolor='turquoise', alpha=0.2, zorder=0, transform=ax.transAxes)
#         ax.add_patch(circle)

#     # Draw edges between qubits with connections
#     # Edge width proportional to the number of connections
#     max_conn = max(connections.values())
#     for key in connections.keys():
#         for i1 in range(len(key)-1):
#             for i2 in range(i1+1, len(key)):
#                 q1, q2 = key[i1], key[i2]
#                 ax.plot([qubit_positions[q1][0], qubit_positions[q2][0]], [qubit_positions[q1][1], qubit_positions[q2][1]],
#                         linewidth=connections[key]/max_conn*2, zorder=1, color='black')

#     # Draw qubits
#     ax.scatter([qubit_positions[q][0] for q in range(num_qubits)], [qubit_positions[q][1] for q in range(num_qubits)],
#                 color='turquoise', edgecolor='black', zorder=2)

#     mp.setp(ax.get_xticklabels(), fontsize=14)
#     mp.setp(ax.get_yticklabels(), fontsize=14)

#     ax.set_xlabel('X Coordinates', fontsize=14)
#     ax.set_ylabel('Y Coordinates', fontsize=14)

#     # Save the the output file correxponding to the algorithm
#     mp.savefig('./GRAPHINE_altered/benchmarks/' + input_file.split('/')[-1].split('.')[0] + '.pdf')
#     mp.close()

    def get_connections(num_qubits, qubit_positions):

        connections = []
        qubit_conns = {q:[] for q in range(num_qubits)}
        for q1 in range(num_qubits-1):
            for q2 in range(q1+1, num_qubits):

                # Calculate distance
                distance = ((qubit_positions[q1][0]-qubit_positions[q2][0])**2 +
                            (qubit_positions[q1][1]-qubit_positions[q2][1])**2)**0.5

                # Record distance
                if distance < radius + 1e-5:
                    connections.append([q1, q2])
                    connections.append([q2, q1])

                    qubit_conns[q1].append(q2)
                    qubit_conns[q2].append(q1)

        return connections, qubit_conns

    def compile(num_qubits, connections, base_circuit):

        min_cz = float('inf')
        min_circuit = None
        for trial in range(5):
            circuit = transpile(base_circuit,
                                initial_layout=list(range(num_qubits)),
                                coupling_map=connections,
                                basis_gates=['u3', 'cz'],
                                optimization_level=3)
            if circuit.count_ops()['cz'] < min_cz:
                min_cz = circuit.count_ops()['cz']
                min_circuit = circuit

        return min_circuit

    """
    Returns tuple of num pulses, num critical pulses
    """
    def get_pulse_counts(num_qubits, qubit_conns, circuit):

        ops = []
        qubit_ops = [[] for q in range(num_qubits)]
        for row in circuit.qasm().split('\n'):

            num_op_qubits = row.count('[')

            if not num_op_qubits or 'reg' in row:
                continue

            if num_op_qubits > 1:
                qubits = [int(row.split('[')[q+1].split(']')[0]) for q in range(num_op_qubits)]
                qubits.sort()
                for qubit in qubits:
                    qubit_ops[qubit].append(len(ops))
                ops.append(qubits)
            else:
                qubit = int(row.split('[')[1].split(']')[0])
                qubit_ops[qubit].append(len(ops))
                ops.append([qubit])

        num_ops = [len(qubit_ops[q]) for q in range(num_qubits)]

        num_total_pulses = 0
        num_critical_pulses = 0
        frontiers = [0]*num_qubits
        while any([frontiers[q] != num_ops[q] for q in range(num_qubits)]):

            blocked_qubits = set()
            sched_qubits = []
            for q1 in range(num_qubits):
                if q1 in blocked_qubits:
                    continue
                if frontiers[q1] == num_ops[q1]:
                    continue
                if len(ops[qubit_ops[q1][frontiers[q1]]]) == 1:
                    continue
                q2 = ops[qubit_ops[q1][frontiers[q1]]][0]
                if q2 == q1:
                    q2 = ops[qubit_ops[q1][frontiers[q1]]][1]
                assert frontiers[q2] < num_ops[q2]
                if q2 in blocked_qubits:
                    continue
                if qubit_ops[q1][frontiers[q1]] != qubit_ops[q2][frontiers[q2]]:
                    continue

                num_total_pulses += 3
                for q in qubit_conns[q1]:
                    blocked_qubits.add(q)
                for q in qubit_conns[q2]:
                    blocked_qubits.add(q)
                sched_qubits.append(q1)
                sched_qubits.append(q2)
                frontiers[q1] += 1
                frontiers[q2] += 1

            if sched_qubits:
                num_critical_pulses += 3
                for q in range(num_qubits):
                    if q in sched_qubits:
                        continue
                    for n_sched in range(3):
                        if frontiers[q] == num_ops[q]:
                            break
                        if len(ops[qubit_ops[q][frontiers[q]]]) > 1:
                            break
                        num_total_pulses += 1
                        frontiers[q] += 1
            else:
                num_critical_pulses += 1
                for q in range(num_qubits):
                    if frontiers[q] == num_ops[q]:
                        continue
                    if len(ops[qubit_ops[q][frontiers[q]]]) > 1:
                        continue
                    num_total_pulses += 1
                    frontiers[q] += 1

        return num_total_pulses, num_critical_pulses


    with open(input_file, 'r') as f:
        qasm_str = f.read()
    base_circuit = QuantumCircuit.from_qasm_str(qasm_str)

    connections, qubit_conns = get_connections(num_qubits, qubit_positions)

    circuit = compile(num_qubits, connections, base_circuit)

    ops = circuit.count_ops()
    pulses = get_pulse_counts(num_qubits, qubit_conns, circuit)

    return (qubit_conns, qubit_positions, ops, pulses, connect_count, radius)
