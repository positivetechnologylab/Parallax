import neutralatomcompilation as nac
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
import qiskit.qasm3 as qasm3
import numpy as np
import os
import time
import pickle
import re

def ld_qasm(algo_name):
    #Get qasm input file
    input_file = +algo_name+'.qasm'
    with open(input_file, 'r') as f:
        qasm_str = f.read()
    return qasm_str, input_file

runtimes = {}
comp_circs = {}
directory_path = './benchmarks_retrans/'
for filename in os.listdir(directory_path):
    if filename.endswith('.qasm'):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
            start = time.time()
            print("loading ", full_path)
            with open(full_path, 'r') as f:
                qasm_str = f.read()
            print("hw ", full_path)
            hw = nac.Hardware(num_dimensions=2, dimensions_length=(35, 35), dimensions_spacing=(1,1))
            print("im ", full_path)
            im = nac.InteractionModel(hardware=hw, d_to_r=lambda x: x / 2, max_int_dist=3)
            print("qc ", full_path)
            qc = QuantumCircuit.from_qasm_str(qasm_str)
            comp = nac.LookaheadCompiler(interaction_model=im, hardware=hw)
            print("comp", full_path)
            compiled_circ, l2p, p2l, dag = comp.compile(qc, lookahead_distance=float('inf'), weighting_function=lambda x: np.e ** (-x))
            unique_numbers_in_brackets = set(map(int, re.findall(r'\[(\d+)\]', compiled_circ.qasm())))
            qd = compiled_circ.qasm()
            l2p = {key: (value.dim_loc(0), value.dim_loc(1)) for key, value in l2p.items()}
            dag = [( (hq.dim_loc(0), hq.dim_loc(1)), qb) for hq, qb in dag]
            Create a dictionary to map coordinates to Qubit indices in l2p
            coordinate_to_index_l2p = {coord: qubit.index for qubit, coord in l2p.items()}

            # Initialize an empty list to store the mappings
            mappings = []
            unique_numbers_in_brackets = set(map(int, re.findall(r'\[(\d+)\]', qasm_str)))
            print(unique_numbers_in_brackets)
            # Iterate through the list of tuples in dag
            for coordinate, qubit in dag:
                # Find the corresponding Qubit index based on the coordinates in l2p
                qubit_index_l2p = coordinate_to_index_l2p.get(coordinate, None)
                if qubit_index_l2p is not None:
                    # Append the mapping as a tuple of indices
                    mappings.append((qubit_index_l2p, qubit.index))

            # Iterate through the mappings and create a dictionary
            # where the second index is the key and the first index is the value
            index_mappings = {second_index: first_index for first_index, second_index in mappings}

            # Define a function to replace integers in brackets with the first index from the mappings
            def replace_indices(match):
                integer = int(match.group(1))
                if integer in index_mappings:
                    # Include the brackets in the replacement
                    return '[' + str(index_mappings[integer]) + ']'
                return match.group(0)
            unique_numbers_in_brackets_2 = set(map(int, re.findall(r'\[(\d+)\]', compiled_circ.qasm())))
            if unique_numbers_in_brackets!=unique_numbers_in_brackets_2:
                print(unique_numbers_in_brackets,unique_numbers_in_brackets_2)
            # Use regular expressions to find and replace integers in brackets
            qasm_modified = re.sub(r'\[(\d+)\]', replace_indices, compiled_circ.qasm())
            # Regular expression pattern to match the line
            pattern = r"qreg q\d+\[\d+\];"

            # Replacement pattern, inserting the unique number
            replacement = f"qreg q[{len(unique_numbers_in_brackets)-1}];"

            # Performing the replacement
            qasm_modified = re.sub(pattern, replacement, qasm_modified)
            
            pattern = r"q\d+\["

            # Replacement pattern (remove the integer between q and [)
            replacement = "q["

            # Performing the replacement
            qasm_modified = re.sub(pattern, replacement, qasm_modified)
            
            #print(qasm_modified)
            comp_circs[full_path] = [qasm_modified, l2p]
            #print(compiled_circ.qasm())
            print("done ",full_path)
            end = time.time()
            runtimes[full_path] = end - start
            #print(qasm_modified)
with open('baker_res_atom.pkl', 'wb') as file:
    pickle.dump(comp_circs, file)
with open('runtimes_atom.pkl', 'wb') as file:
    pickle.dump(runtimes, file)