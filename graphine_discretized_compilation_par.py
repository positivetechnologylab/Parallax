"""
File for compiling circuits using base Graphine with SWAPs and parallelizing circuits to fit hardware (no AOD movement)
"""
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
from GRAPHINE_disc_comp import NA_Architecture
CT = 0
def ld_qasm(algo_name):
    input_file = './benchmarks/'+algo_name+'.qasm'
    with open(input_file, 'r') as f:
        qasm_str = f.read()
    return qasm_str, input_file


def compilation_save(algo_name, num_copies, dim_size, data):
    # Define the file path using the algo_name, num_copies, and dim_size
    file_path = f"./par_graphine_res/{algo_name}/{num_copies}_{dim_size}.pkl"
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {file_path}")

def load_list_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

class NoRoomError(Exception):
    pass


def map_to_bounded_integer(points, width, height, radius):
    filled_locations = set()
    hold_list = []
    mapped_points = []

    # Function to find the closest empty discrete location
    def find_closest_empty(x, y):
        for dx in range(max(width, height)):
            for dy in range(max(width, height)):
                for nx, ny in [(x+dx, y+dy), (x+dx, y-dy), (x-dx, y+dy), (x-dx, y-dy)]:
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in filled_locations:
                        return nx, ny
        return None

    # Attempt to map each point
    for (x, y) in points:
        mapped_x = int(x * width)
        mapped_y = int(y * height)
        if (mapped_x, mapped_y) in filled_locations:
            hold_list.append((x, y))
        else:
            mapped_points.append((mapped_x, mapped_y))
            filled_locations.add((mapped_x, mapped_y))

    # Process points in the hold list
    for (x, y) in hold_list:
        closest_empty = find_closest_empty(int(x * width), int(y * height))
        if closest_empty is None:
            raise NoRoomError("Not enough room in SLM for all qubits to be loaded.")
        else:
            mapped_points.append(closest_empty)
            filled_locations.add(closest_empty)

    radius = radius * max(width, height)
    return mapped_points, radius

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

par_dim_sizes_ADV = {121:3,100:3,81:3,64:4,49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_KNN = {49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_QV = {25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_SECA = {64:4,49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_SQRT = {49:5,36:5,25:7,16:8,9:11,4:17,1:35}
par_dim_sizes_WST = {25:7,16:8,9:11,4:17,1:35}

file_to_dims = {"advantage_9" : par_dim_sizes_ADV,
"knn_n25" : par_dim_sizes_KNN,
"qv_32" : par_dim_sizes_QV,
"seca_n11" : par_dim_sizes_SECA,
"sqrt_18" : par_dim_sizes_SQRT,
"wstate_27" : par_dim_sizes_WST,
}

def main_loop(algo, num_copies, dim_size):
    print("Running Discretized Graphine",algo,"...")
    start_time = time.time()
    algo_name = algo
    qasm_str, qasm_file = ld_qasm(algo_name)
    res = None

    res = load_list_from_file('./graphine_results/'+algo_name+'_res.pkl')

    if res == None:
        print("GRAPHINE result not generated.")
    else:
        AOD_ROWS = 20 #AOD row count
        AOD_COLS = 20 #AOD col count
        ARR_WIDTH = dim_size
        ARR_HEIGHT = dim_size
        

        rydberg_connect, points, gate_counts, pulse_counts, connect_count, radius = res[0],res[1],res[2],res[3],res[4],res[5]
        try:
            mapped_points, radius = map_to_bounded_integer(points, ARR_WIDTH, ARR_HEIGHT, radius)
        except NoRoomError as e:
            print(e)
        start_time = time.time()
        na = NA_Architecture([AOD_ROWS, AOD_COLS], [ARR_WIDTH, ARR_HEIGHT], mapped_points, connect_count, radius, qasm_str)
        print("Compiling circuit...")
        all_layers, swap_counts, cz_count, u_count = na.compile_circuit()
        runtime = calc_runtime(all_layers, swap_counts)

        end_time = time.time()
        compile_time = end_time - start_time
        end_time = time.time()
        
        compilation_save(algo_name, num_copies, dim_size,[all_layers, swap_counts, cz_count, u_count, runtime, compile_time])

algo_list = ['advantage_9', 'knn_n25', 'qv_32', 'seca_n11', 'sqrt_18', 'wstate_27']
# Iterate through each algorithm and its corresponding parameter dimension sizes
for algo, dim_sizes in file_to_dims.items():
    for k, v in dim_sizes.items():
        main_loop(algo, k, v)