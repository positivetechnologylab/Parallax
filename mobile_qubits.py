"""
Selector function for mobile qubits, given an architecture.
Note: At least one atom must be selected for AOD placement if this function is called.
"""

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

#The proportion of qubits that will be on the AOD. Note that if the value used than the number of AOD rows/cols, it will simply take the number of rows/cols as the number of AOD qubits rather than the fraction defined here (by default set to 1 to just use all available AOD rows/cols)
MOBILE_PROPORITON = 1.0 

"""
Function that selects which qubits will be placed on the AOD
Note that the last arg can be used to test different aod counts.
"""
def select_mobile_qubits(mapped_points, qubit_edge_counts, radius, qasm_str, aod_array_ct, aod_testing_ct=0):    
    
    # Helper function to compute the distance  between two points
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    def count_multi_gates(interference_count, qasm, mapped_points, radius):
        # Number of qubits in the circuit
        num_qubits = None
        list_of_multigates = []

        # For each row in the QASM file
        for row in qasm.splitlines():
            #get number of qubits out of the qreg file
            if "OPENQASM" in row or "include" in row:
                continue
                
            if 'qreg' in row:
                num_qubits = int(row.split('[')[1].split(']')[0])
                continue

            # Calculate the number of qubits in the op
            num_op_qubits = row.count('[')

            # If there are more than 1 qubits, add gate to list of multi qubit gates
            if num_op_qubits > 1:
                assert num_op_qubits == row.count('q')
                assert num_op_qubits == row.count(']')
                # Get list of all the qubits as ints involved in the operation
                qubits = [int(row.split('[')[q+1].split(']')[0]) for q in range(num_op_qubits)]
                qubits.sort()
                list_of_multigates.append(qubits)

        #For all combinations of multiqubit gates, check if they interfere with each other
        #O(num_qubits^2) since list_of_multigates is proportional to num_qubits size in worst case
        for i1 in range(len(list_of_multigates)-1):
            for i2 in range(i1+1, len(list_of_multigates)):
                gate1 = list_of_multigates[i1]
                gate2 = list_of_multigates[i2]
                for qubit in gate1:
                    for qubit2 in gate2:
                        #If distance between the two qubits is less than 2.5*rydberg range, they interfere
                        #Increment each interfering qubits interference count
                        if np.linalg.norm(np.array((mapped_points[qubit]))-np.array(mapped_points[qubit2])) < 2.5 * radius:
                            interference_count[qubit] += 1
                            interference_count[qubit2] += 1

    """
    Get counts of times that each qubit participates in a multiqubit gate that interferes with another multiqubit from Rydberg blockading.
    Note: This (and the above function "count_multi_gates" are optional when computing AOD weight selection, as they effectively only factor in to break ties between qubits with the same out-of-range CZ gate count
    """
    def get_interference_count(qasm_str, mapped_points, radius):
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        dag = circuit_to_dag(qc)

        multi_multigate = []
        qubit_inds = list(range(dag.num_qubits()))
        qubit_multigate_counts = {key: 0 for key in qubit_inds}

        # Create a dictionary of counts of all interfering connections in the circuit for each qubit
        interference_count = {key: 0 for key in qubit_inds}
        for i in dag.layers():
            if len(i['graph'].two_qubit_ops()) + len(i['graph'].multi_qubit_ops()) > 1:
                count_multi_gates(interference_count, dag_to_circuit(i['graph']).qasm(), mapped_points, radius)

        return interference_count
    
    # Helper function to compute the average distance from the current qubit to all selected qubits
    def distance_from_selected(qubit, selected_qubits, mapped_points):
        total_distance = 0
        for selected in selected_qubits:
            total_distance += distance(mapped_points[qubit], mapped_points[selected])
        return total_distance / len(selected_qubits) if selected_qubits else 0

    # Compute the weight value for AOD selection for a qubit (i.e. score used to determine which qubits will be placed in AOD)
    # O(num_qubits^2*num_aod)
    def compute_weight(qubit, radius, qubit_edge_counts, interference_count, sum_interference, sum_dist_edges, mobile_qubits):
        #Weighting on distant edges ("distant" defined as outside of Rydberg interaction distance)
        distant_edge_weight = 0.99
        
        # Interference count (normalized to total amount of interference in circuit)
        if sum_interference != 0:
            interference_weight = interference_count[qubit]/sum_interference
        else:
            interference_weight = 0

        # Number of edges connecting to distant nodes (main criteria for AOD qubit selection (99%))
        count = 0
        for pair in qubit_edge_counts:
            if qubit in pair:
                if all(q not in mobile_qubits for q in pair):
                    if distance(mapped_points[pair[0]],mapped_points[pair[1]]) > radius:
                        count += qubit_edge_counts[pair]
        
        if sum_dist_edges > 0:
            distant_edges = count/sum_dist_edges
        else:
            distant_edges = 0
        
        # Combine the weights
        return distant_edge_weight * distant_edges + (1 - distant_edge_weight) * interference_weight
    
    #Initial selection of first qubit to be placed on AOD
    qubits = list(range(len(mapped_points)))
    interference_count = get_interference_count(qasm_str,mapped_points,radius)
    mobile_qubits = []
    sum_interference = sum(interference_count.values())
    
    sum_dist_edges = sum(qubit_edge_counts[pair] for pair in qubit_edge_counts if distance(mapped_points[pair[0]],mapped_points[pair[1]]) > radius)
    
    weights = {q: compute_weight(q, radius, qubit_edge_counts, interference_count, sum_interference, sum_dist_edges, []) for q in qubits}
    
    initial_qubit = max(weights, key=weights.get)
    mobile_qubits = [initial_qubit]

    selectable_qubits = set(qubits)
    selectable_qubits.remove(initial_qubit)
    
    #This is to test different AOD row/col counts 
    if aod_testing_ct > 0:
        num_mobile = min(math.ceil(len(qubits) * MOBILE_PROPORITON), aod_testing_ct)
    else:
        num_mobile = min(math.ceil(len(qubits) * MOBILE_PROPORITON), aod_array_ct)
    
    #Keep adding atoms to AOD as long as there is room
    while len(mobile_qubits) < num_mobile:
        # Update weights considering the interference count and distant edges
        weights = {q: compute_weight(q, radius, qubit_edge_counts, interference_count, sum_interference, sum_dist_edges, mobile_qubits) for q in selectable_qubits}

        # Select qubit with highest weight from selectable qubits
        next_qubit = max(selectable_qubits, key=weights.get)
        mobile_qubits.append(next_qubit)
        selectable_qubits.remove(next_qubit)
    
    return mobile_qubits