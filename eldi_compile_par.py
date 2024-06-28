
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
from eldi_disc_comp import NA_Architecture

def load_data_from_pickle(file_path):
    _, file_name = os.path.split(file_path)
    num1, num2_with_ext = file_name.split('_')
    dim_size = int(num2_with_ext.split('.')[0])  

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    sorted_dict_values = [value for key, value in sorted(loaded_data[0].items())]
    qasm_string = loaded_data[1][0]
    
    implicit_swap_counts = loaded_data[1][1]

    return sorted_dict_values, qasm_string, dim_size, implicit_swap_counts

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


def compilation_save(algo_name, data, file_path):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(file_path)


def main_loop(algo):
    base_dir = './eldi_par_data'
    algo_dir = os.path.join(base_dir, algo)
    output_base_dir = './par_eldi_res'  
    
    if not os.path.isdir(algo_dir):
        print(f"Directory not found: {algo_dir}")
        return

    for file_name in os.listdir(algo_dir):
        if file_name.endswith(".qasm"):  
            file_path = os.path.join(algo_dir, file_name)
            mapped_points, qasm_str, dim_size, implicit_swap_counts = load_data_from_pickle(file_path)
            start_time = time.time()
            algo_name = algo
            radius = 2.0
            res = None
            AOD_ROWS = 20 #AOD row count DOES NOT MATTER HERE
            AOD_COLS = 20 #AOD col count DOES NOT MATTER HERE
            ARR_WIDTH = dim_size
            ARR_HEIGHT = dim_size
            
            start_time = time.time()
            na = NA_Architecture([AOD_ROWS, AOD_COLS], [ARR_WIDTH, ARR_HEIGHT], mapped_points, radius, qasm_str)
            all_layers, swap_counts, cz_count, u_count = na.compile_circuit()
            runtime = calc_runtime(all_layers, swap_counts+implicit_swap_counts) 
            end_time = time.time()
            compile_time = end_time - start_time
            end_time = time.time()
            output_file_path = os.path.join(output_base_dir, algo, file_name.replace('.qasm', '.pkl'))
            compilation_save(algo_name, [all_layers, swap_counts, cz_count+3*implicit_swap_counts, u_count, runtime, compile_time], output_file_path)
            
algo_list = ['advantage_9', 'knn_n25', 'qv_32', 'seca_n11', 'sqrt_18', 'wstate_27']
for algo in algo_list:
    main_loop(algo)