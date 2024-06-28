# HDWR = 'Quera'
HDWR = 'Atom'

import copy
import numpy as np
import math
import random
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import matplotlib as mpl
import matplotlib.pyplot as mp
from matplotlib.patches import Circle
from mobile_qubits import select_mobile_qubits
import sys
import os
import pickle
import time
import neutralatomcompilation
from eldi_disc_comp import NA_Architecture

# dt_dct = [dict_of_ind_to_coords,[qasmstr, num_swaps_removed]]
def ld_qasm(algo_name, dt_dct):
    input_file = algo_name+'.qasm'
    return dt_dct[input_file][1][0]

def ld_topology(algo_name, dt_dct):
    input_file = algo_name+'.qasm'
    top_dict = dt_dct[input_file][0]
    sorted_values = [value for key, value in sorted(top_dict.items())]

    return sorted_values

if HDWR == 'Atom':
    file = 'eldi_res_atom.pkl'
elif HDWR == 'Quera':
    file = 'eldi_res_quera.pkl'
with open('eldi_res/'+file, 'rb') as f:
    all_data = pickle.load(f)

if HDWR == 'Atom':
    COMP_TYPE = 'atom/'
elif HDWR == 'Quera':
    COMP_TYPE = 'quera/'
    
def compilation_save(algo_name, data):
    with open('./disc_eldi_results/'+COMP_TYPE+algo_name+'_res.pkl', 'wb') as file:
        pickle.dump(data, file)

def calc_runtime(all_layers, swap_counts):
    CZ_TIME = 0.8 #us
    U3_TIME = 2 #us
    total_time = 0
    
    #Add time for all layers, NOT including additional SWAPs
    #We do it like this since U3 gates take longer than CZ gates
    for layer in all_layers:
        min_length = min(len(gate) for gate in layer)
        if min_length == 1:
            total_time += U3_TIME
        elif min_length == 2:
            total_time += CZ_TIME
    
    #Add time for swaps that execute out of layers (SWAP_time=3*CZ_time)
    total_time += swap_counts*3*CZ_TIME
    
    return total_time

def main_loop(algo):
    print("Running Discretized ELDI",algo,"...")
    start_time = time.time()
    algo_name = algo
    qasm_str = ld_qasm(algo_name, all_data)
    mapped_points = ld_topology(algo_name, all_data)
    radius = 2.0
    res = None
    implicit_swap_counts = all_data[algo+'.qasm'][1][1]

    AOD_ROWS = 20 #AOD row count
    AOD_COLS = 20 #AOD col count
    if HDWR == 'Atom':
        ARR_WIDTH = 35
        ARR_HEIGHT = 35
    elif HDWR == 'Quera':
        ARR_WIDTH = 16
        ARR_HEIGHT = 16

    start_time = time.time()
    na = NA_Architecture([AOD_ROWS, AOD_COLS], [ARR_WIDTH, ARR_HEIGHT], mapped_points, radius, qasm_str)
    all_layers, swap_counts, cz_count, u_count = na.compile_circuit()
    runtime = calc_runtime(all_layers, swap_counts+implicit_swap_counts)

    end_time = time.time()
    compile_time = end_time - start_time
    end_time = time.time()

    compilation_save(algo_name, [all_layers, swap_counts, cz_count+3*implicit_swap_counts, u_count, runtime, compile_time])
    
algo_list = ['adder_9',
             'advantage_9',
             'gcm_h6_13',
             'heisenberg_16',
              'hlf_10',
             'knn_n25',
              'multiplier_10',
              'qaoa_10',
             'qec9xz_n17',
              'qft_10',
             'qugan_n39',
               'qv_32',
              'sat_11',
                'seca_n11',
              'sqrt_18',
             'tfim_128',
#             'vqe_uccsd_n28', #VQE IS NOT RUN FOR ELDI AS IT TAKES TOO LONG
              'wstate_27'
            ]
for algo in algo_list:
    main_loop(algo)