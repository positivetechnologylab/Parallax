"""
Script that takes in qasm files and converts ELDI output into  qasm files and coordinates in the atom grid 
that can be used to compile the circuit.
"""
# HDWR = 'Quera'
HDWR = 'Atom'

import neutralatomcompilation as nac
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
import qiskit.qasm3 as qasm3
import numpy as np
import os
import time
import pickle
import re
import math
from qiskit.converters import circuit_to_dag

#load qasm file based on algo name
def ld_qasm(algo_name):
    input_file = +algo_name+'.qasm'
    with open(input_file, 'r') as f:
        qasm_str = f.read()
    return qasm_str, input_file

runtimes = {}
comp_circs = {}
directory_path = './benchmarks_retrans/'

#execute the ELDI procedure to map circuit qubits
for filename in os.listdir(directory_path):
    if filename.endswith('.qasm'):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
            start = time.time()
            with open(full_path, 'r') as f:
                qasm_str = f.read()
            if HDWR == 'Quera':
                hw = nac.Hardware(num_dimensions=2, dimensions_length=(16, 16), dimensions_spacing=(1,1))
            elif HDWR == 'Atom':
                hw = nac.Hardware(num_dimensions=2, dimensions_length=(35, 35), dimensions_spacing=(1,1))
            else:
                print("Error on Hardware")
                break
            im = nac.InteractionModel(hardware=hw, d_to_r=lambda x: x / 2, max_int_dist=2)
            qc = QuantumCircuit.from_qasm_str(qasm_str)
            comp = nac.LookaheadCompiler(interaction_model=im, hardware=hw)
            compiled_circ, l2p, p2l, logic_phys_map = comp.compile(qc, lookahead_distance=float('inf'), weighting_function=lambda x: np.e ** (-x))
            comp_circs[full_path] = compiled_circ
            end = time.time()
            runtimes[full_path] = end - start

circs = {}
for key, value in comp_circs.items():
    new_key = key.replace('./benchmarks_retrans/', '')
    circs[new_key] = value

qasms = {}
new_atom_locs_dict = {}
for circ in circs.keys():
    qasm_string = circs[circ].qasm()
    
    # remove starting lines
    lines = qasm_string.split('\n')
    remaining_lines = lines[3:]
    qasm_string = '\n'.join(remaining_lines)
    
    #get unique qubit indices in qasm str
    unique_inds = set(map(int, re.findall(r'\[(\d+)\]', qasm_string)))
    if HDWR == 'Atom':
        dim_sz = 35
    elif HDWR == 'Quera':
        dim_sz = 16
    else:
        print("Error: unknown HDWR var")
    atom_locs = {}
    for i in unique_inds:
        integer_part = i // dim_sz
        remainder = i % dim_sz
        atom_locs[i] = (remainder, integer_part) #(Y_loc * dim_size) + X_loc = location of qubit
    
    #convert keys to inrementing indices
    sorted_keys = sorted(atom_locs.keys())
    new_atom_locs = {i: atom_locs[old_key] for i, old_key in enumerate(atom_locs.keys())}
    
    new_atom_locs_dict[circ] = new_atom_locs
    old_to_new_mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(atom_locs.keys()))}

    # replaces old keys with new keys in the QASM string and format qasm string
    def replace_keys(match):
        old_key = int(match.group(1))
        return f"[{old_to_new_mapping.get(old_key, old_key)}]"
    pattern = r'\[(\d+)\]'
    qasm_string = re.sub(pattern, replace_keys, qasm_string)
    pattern = r'q\d+\['
    qasm_string = re.sub(pattern, 'q[', qasm_string)
    lines = qasm_string.split('\n')
    swap_count = 0
    new_qasm_string = ""
    for line in lines:
        if 'swap' in line:
            swap_count += 1
        else:
            new_qasm_string += line + '\n'
    new_qasm_string = new_qasm_string.rstrip()
    qasm_string = new_qasm_string
    
    intro_ind = len(unique_inds)
    intro = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{intro_ind}];\n"

    qasm_string = intro + qasm_string
    qasms[circ] = [qasm_string,swap_count]

data_dict = {}
for key in qasms.keys():
    data_dict[key] = [new_atom_locs_dict[key],qasms[key]]

if HDWR == 'Atom':
    file_path = "../../eldi_res/eldi_res_atom.pkl"
elif HDWR == 'Quera':
    file_path = "../../eldi_res/eldi_res_quera.pkl"
    
# save to a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(data_dict, file)
    
print("Done creating aggregated .pkl file for "+HDWR+" hardware (non-parallelized circuits)")
    
    
"""
------------
Begin Parallelized Circuit Data Generation (this data is always generated for Atom hardware (1,225 qubits))
------------
"""

runtimes = {}
comp_circs = {}
directory_path = './benchmarks_retrans_par/'

#Dict mapping parallelization factor (num copies of circuit simultaneously on Atom comp) to dimension size per circ
par_dim_sizes_ADV = {121:3,100:3,81:3,64:4,49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_KNN = {49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_QV = {25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_SECA = {64:4,49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_SQRT = {49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_WST = {25:7,16:8,9:11,4:17,1:35}

#Files used to generate parallelized result in paper (Fig 11)
file_to_dims = {"./benchmarks_retrans_par/advantage_9.qasm" : par_dim_sizes_ADV,
"./benchmarks_retrans_par/knn_n25.qasm" : par_dim_sizes_KNN,
"./benchmarks_retrans_par/qv_32.qasm" : par_dim_sizes_QV,
"./benchmarks_retrans_par/seca_n11.qasm" : par_dim_sizes_SECA,
"./benchmarks_retrans_par/sqrt_18.qasm" : par_dim_sizes_SQRT,
"./benchmarks_retrans_par/wstate_27.qasm" : par_dim_sizes_WST,
}

#Run ELDI
for filename in os.listdir(directory_path):
    if filename.endswith('.qasm'):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
            for par_fact,dim_sz in file_to_dims[full_path].items():
                start = time.time()
                with open(full_path, 'r') as f:
                    qasm_str = f.read()
                hw = nac.Hardware(num_dimensions=2, dimensions_length=(dim_sz, dim_sz), dimensions_spacing=(1,1))
                im = nac.InteractionModel(hardware=hw, d_to_r=lambda x: x / 2, max_int_dist=2)
                qc = QuantumCircuit.from_qasm_str(qasm_str)
                comp = nac.LookaheadCompiler(interaction_model=im, hardware=hw)
                compiled_circ, l2p, p2l, logic_phys_map = comp.compile(qc, lookahead_distance=float('inf'), weighting_function=lambda x: np.e ** (-x))
                comp_circs[full_path+'_'+str(par_fact)+'_'+str(dim_sz)] = compiled_circ
                end = time.time()
                runtimes[full_path] = end - start

circs = {}
for key, value in comp_circs.items():
    new_key = key.replace('./benchmarks_retrans_par/', '')
    circs[new_key] = value
    
qasms = {}
new_atom_locs_dict = {}

#process qasm strings
for circ in circs.keys():
    qasm_string = circs[circ].qasm()
    
    lines = qasm_string.split('\n')
    remaining_lines = lines[3:]
    qasm_string = '\n'.join(remaining_lines)
    
    unique_inds = set(map(int, re.findall(r'\[(\d+)\]', qasm_string)))
    
    parts = circ.split('_')
    dim_sz = int(parts[-1])

    atom_locs = {}
    for i in unique_inds:
        integer_part = i // dim_sz
        remainder = i % dim_sz
        atom_locs[i] = (remainder, integer_part) #(Y_loc * dim_size) + X_loc = location of qubit
    
    sorted_keys = sorted(atom_locs.keys())
    new_atom_locs = {i: atom_locs[old_key] for i, old_key in enumerate(atom_locs.keys())}
    
    new_atom_locs_dict[circ] = new_atom_locs
    old_to_new_mapping = {old_key: new_key for new_key, old_key in enumerate(sorted(atom_locs.keys()))}

    def replace_keys(match):
        old_key = int(match.group(1))
        return f"[{old_to_new_mapping.get(old_key, old_key)}]"

    pattern = r'\[(\d+)\]'
    qasm_string = re.sub(pattern, replace_keys, qasm_string)
    
    pattern = r'q\d+\['
    qasm_string = re.sub(pattern, 'q[', qasm_string)
    lines = qasm_string.split('\n')

    swap_count = 0
    new_qasm_string = ""
    for line in lines:
        if 'swap' in line:
            swap_count += 1
        else:
            new_qasm_string += line + '\n'

    new_qasm_string = new_qasm_string.rstrip()
    qasm_string = new_qasm_string
    
    intro_ind = len(unique_inds)
    intro = f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{intro_ind}];\n"

    qasm_string = intro + qasm_string
    qasms[circ] = [qasm_string,swap_count]
    
#separate/save data based on the qasm file
for key in qasms.keys():
    initial_string, num1, num2 = key.rsplit('_', 2)
    initial_string = initial_string.rsplit('.', 1)[0]
    
    directory_path = f'../../eldi_par_data/{initial_string}'
    filename = f'{num1}_{num2}.qasm'
    full_path = os.path.join(directory_path, filename)
    
    os.makedirs(directory_path, exist_ok=True)
    
    data_to_pickle = [new_atom_locs_dict[key], qasms[key]]
    
    with open(full_path, 'wb') as f:
        pickle.dump(data_to_pickle, f)
    print("Done creating parallelized circuit data stored in "+key)