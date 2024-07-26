import sys
import os
import pickle
import time
import math
 
from na_arch_aod_ch import NA_Architecture
from graphine import graphine

HDWR = 'Atom'
# HDWR = 'Quera'

def ld_qasm(algo_name):
    #Get qasm input file
    input_file = './benchmarks/'+algo_name+'.qasm'
    with open(input_file, 'r') as f:
        qasm_str = f.read()
    return qasm_str, input_file

def graphine_save(algo_name, qasm_file):
    print("GRAPHINE underway...")
    res = graphine(qasm_file)
    print("GRAPHINE Complete")
    with open('./graphine_results/'+algo_name+'_res.pkl', 'wb') as file:
        pickle.dump(res, file)
    return res

def compilation_save(algo_name, arr_dim_sz, data, aod_dim=None): 
    results_dir = './aod_count_changing_results/'
    algo_dir = os.path.join(results_dir, algo_name)
    if not os.path.exists(algo_dir):
        os.makedirs(algo_dir)
    data_file_path = os.path.join(algo_dir, f'{algo_name}_{aod_dim}_res.pkl')

    with open(data_file_path, 'wb') as file:
        print("Saving in ",data_file_path)
        pickle.dump(data, file)

        
def load_list_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def calc_runtime(all_layers, total_max_moved_dist, swap_trap_count, trap_dist):
    CZ_TIME = 0.8 #us
    U3_TIME = 2 #us
    AVG_TRAP_VELOCITY = 55 #um/us
    TRAP_TIME = 50 #us
    US_PER_UNIT = 4.05 #2 us min sep distance, x2, +0.05 for padding for AOD movement between qubits
    total_time = 0
    
    #Add time for all layers
    for layer in all_layers:
        min_length = min(len(gate) for gate in layer)
        if min_length == 1:
            total_time += U3_TIME
        elif min_length == 2:
            total_time += CZ_TIME
    
    #Add time for moves (ONLY for moves without trap changing)
    total_time += (US_PER_UNIT*total_max_moved_dist)/AVG_TRAP_VELOCITY #total time added from moving that distance
    
    #Add time for trap changes/moves
    total_time += swap_trap_count*TRAP_TIME*2 #x2 for trap/detrap, this line just the trap time
    total_time += (US_PER_UNIT*trap_dist)/AVG_TRAP_VELOCITY
    
    return total_time


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


def main_loop(algo):
    print("Running w/AOD ablation ",algo,"...")
    aod_ablation = [1,5,10,20,40]
    algo_name = algo
    #Load qasm file
    input_file = './benchmarks/'+algo+'.qasm'
    with open(input_file, 'r') as f:
        qasm_str = f.read()
    qasm_str, qasm_file = qasm_str, input_file
    res = None

    #If second command line arg > 1, then call GRAPHINE and save result to algo_name+"_res.pkl".
    #Note: This needs to be done once to get an initial graphine result
    #Note: It is recommended to use arg==0 once this has been performed one time, as it dramatically reduces runtimes
    if (int(sys.argv[1]) > 0):
        res = graphine_save(algo_name, qasm_file)
    else:
        res = load_list_from_file('./graphine_results/'+algo_name+'_res.pkl')

    for aod_ct in aod_ablation:
        if res == None:
            print("GRAPHINE result not generated.")
        else:
            if HDWR == 'Atom':
                ARR_WIDTH = 35
                ARR_HEIGHT = 35            
            elif HDWR == 'Quera':
                ARR_WIDTH = 16
                ARR_HEIGHT = 16
            else:
                print("Error: Invalid hardware type"+HDWR)
    
            rydberg_connect, points, gate_counts, pulse_counts, connect_count, radius = res[0],res[1],res[2],res[3],res[4],res[5]
            arr_width = ARR_WIDTH
            arr_height = ARR_HEIGHT
            try:
                mapped_points, radius = map_to_bounded_integer(points, arr_width, arr_height, radius)
            except NoRoomError as e:
                print(e)
            AOD_ROWS = aod_ct
            AOD_COLS = aod_ct
            start_time = time.time()
            na = NA_Architecture([AOD_ROWS, AOD_COLS], [arr_width, arr_height], mapped_points, connect_count, radius, qasm_str)
            frontiers, move_count, total_max_moved_dist, cz_count, u_count, swap_trap_count, trap_dist, swap_trap_gates = na.compile_circuit()
            end_time = time.time()
            elapsed_time = end_time - start_time
            runtime = calc_runtime(frontiers, total_max_moved_dist, swap_trap_count, trap_dist)
            if ARR_WIDTH == 16:
                hrdwr = 'QuEra'
            elif ARR_WIDTH == 35:
                hrdwr = 'Atom'
            else:
                hrdwr = ''
            compilation_save(algo_name, arr_width, [frontiers, move_count, total_max_moved_dist, cz_count, u_count, swap_trap_count, trap_dist, runtime, elapsed_time, hrdwr], AOD_ROWS)

# All algos
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
             'vqe_uccsd_n28',
             'wstate_27'
]


for algo in algo_list:
    main_loop(algo)