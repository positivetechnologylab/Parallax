"""
Version of Parallax that runs each input circuit many times on varying grid sizes, which is meant
to simulate running multiple copies of the circuit in parallel on a single quantum computer.
"""
import sys
import os
import pickle
import time
import math
from na_arch import NA_Architecture
from graphine import graphine

HDWR = 'Atom' #Choose this for a 35x35 qubit computer; for parallelized results this is the only option used 

def graphine_save(algo_name, qasm_file):
    print("GRAPHINE underway...")
    res = graphine(qasm_file)
    print("GRAPHINE Complete")
    #Save result
    with open('./graphine_results/'+algo_name+'_res.pkl', 'wb') as file:
        pickle.dump(res, file)
    return res

"""
Saves results in files for each algorithm.
The file names are in the format: [Number of copies of the circuit runable in parallel]_[Array dimension size]_res.pkl
"""
def compilation_save(algo_name, arr_dim_sz, num_copies, data, aod_dim=None): 
    directory = f'./parallax_par_res/{algo_name}/'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = directory + f'{str(num_copies)}_{str(arr_dim_sz)}_res.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print("Saved to:", file_path)
        
def load_list_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

#Computes the circuit execution time using Parallax
def calc_runtime(all_layers, total_max_moved_dist, swap_trap_count, trap_dist):
    CZ_TIME = 0.8 #us
    U3_TIME = 2 #us
    AVG_TRAP_VELOCITY = 55 #um/us
    TRAP_TIME = 50 #us
    US_PER_UNIT = 4.05 #2 us min sep distance, x2, +0.05 for padding for AOD movement between qubits
    total_time = 0
    
    #Add time for all layers
    #Note: U3 gates take longer than CZ gates
    for layer in all_layers:
        min_length = min(len(gate) for gate in layer)
        if min_length == 1:
            total_time += U3_TIME
        elif min_length == 2:
            total_time += CZ_TIME
    
    #Add time for moves (ONLY for moves without traps)
    total_time += (US_PER_UNIT*total_max_moved_dist)/AVG_TRAP_VELOCITY #total time added from moving that distance
    
    #Add time for traps/moves
    total_time += swap_trap_count*TRAP_TIME*2 #adds time to switch traps; x2 for trap into AOD/untrap
    total_time += (US_PER_UNIT*trap_dist)/AVG_TRAP_VELOCITY #adds time to move the AOD for this trap-swapped atom
    
    return total_time
    
class NoRoomError(Exception):
    pass

"""
Discretizes output of Graphine to the size of the specified atom grid.
Computations are done assuming possible SLM sites are about 2 * minimum-separation-distance units apart.
If a mapped point is already occupied, the algo will cycle around the desired point, attempting to find the closest location to store the atom.
This parallelized version is slightly modified from the baseline version: Given the potentially very small atom grid sizes, 
we adjust the Rydberg radius to be slightly larger for smaller grid sizes. This is to avoid degenerate AOD movement behavior.
"""
def map_to_bounded_integer(points, width, height, radius):
    # Initialize a set to keep track of filled locations
    filled_locations = set()
    # Initialize a list to hold points that couldn't be placed immediately
    hold_list = []

    mapped_points = []

    # Function to find the closest empty discrete location
    def find_closest_empty(x, y):
        # Check all possible locations in increasing distance
        for dx in range(max(width, height)):
            for dy in range(max(width, height)):
                # Check in all four directions
                for nx, ny in [(x+dx, y+dy), (x+dx, y-dy), (x-dx, y+dy), (x-dx, y-dy)]:
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in filled_locations:
                        return nx, ny
        # If no empty location is found
        return None

    # Expand Rydberg radius proportionally for smaller array sizes to avoid degenerate AOD movement behavior
    expansion_cutoff = 17
    if min(width, height) < expansion_cutoff:
        radius = radius * min(width, height)
        
    # Attempt to map each point
    for (x, y) in points:
        mapped_x = int(x * width)
        mapped_y = int(y * height)
        # If the location is already filled, add the point to the hold list
        if (mapped_x, mapped_y) in filled_locations:
            hold_list.append((x, y))
        else:
            # Place the point and mark the location as filled
            mapped_points.append((mapped_x, mapped_y))
            filled_locations.add((mapped_x, mapped_y))
    
    # Process points in the hold list
    for (x, y) in hold_list:
        closest_empty = find_closest_empty(int(x * width), int(y * height))
        if closest_empty is None:
            # If there are no empty locations left, raise an exception
            raise NoRoomError("Not enough room in SLM for all qubits to be loaded.")
        else:
            # Place the point from the hold list to the closest empty location
            mapped_points.append(closest_empty)
            filled_locations.add(closest_empty)
        
    return mapped_points, radius


def main_loop(algo):
    print("Running ",algo,"...")
    start_time = time.time()
    algo_name = algo
    input_file = "./benchmarks/"+algo+'.qasm'
    with open(input_file, 'r') as f:
        qasm_str = f.read()
    qasm_str, qasm_file = qasm_str, input_file
    res = None

    #If second command line arg > 0, then call GRAPHINE and save result to algo_name+"_res.pkl"
    if (int(sys.argv[1]) > 0):
        res = graphine_save(algo_name, qasm_file)
    else:
        res = load_list_from_file('./graphine_results/'+algo_name+'_res.pkl')

    AOD_ROWS = 20 #AOD row count
    AOD_COLS = 20 #AOD col count
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
        
        #Generate grid sizes to run algo with
        #par_dims contains dimensions used in the grid array
        #num_circs is the number of circuits run in parallel
        par_dims = []
        num_circs = []
        square = 1
        while True:
            test_val = min(ARR_WIDTH,ARR_HEIGHT)**2 / float(square ** 2)
            if test_val < len(points):
                break
            sqrt_number = math.sqrt(test_val)
            floored_sqrt = math.floor(sqrt_number)
            nearest_square = floored_sqrt ** 2
            if nearest_square < len(points):
                break
            par_dims.append(floored_sqrt)
            num_circs.append(square**2)
            square += 1
            
        #For each dimension size, compile the circuit and save the results
        for dm in range(len(par_dims)):
            arr_width = par_dims[dm]
            arr_height = par_dims[dm]
            nc = num_circs[dm]
            try:
                mapped_points, radius = map_to_bounded_integer(points, arr_width, arr_height, radius)
            except NoRoomError as e:
                print(e)
            
            """
            The args for the NA_Architecture object are:
            0 - [number_AOD_rows, number_AOD_cols] - The size of the AOD
            1 - [atoms_in_x_axis, atoms_in_y_axis] - The number of atoms in the computer (ex: 35x35 for the Atom computer)
            2 - Discretized coordinate list for the qubits involved in the circuit
            3 - List of counts of gates between qubits
            4 - Rydberg Radius
            5 - qasm string that represents the input circuit 
            """
            na = NA_Architecture([AOD_ROWS, AOD_COLS], [arr_width, arr_height], mapped_points, connect_count, radius, qasm_str)

            """
            na.compile_circuit will return the following outputs:
            0 - list of layers of gates 
            1 - number of moves made
            2 - sequential AOD distance moved (used for time for AOD movement; does not include distance from swap traps(see below))
            3 - CZ gate count, 
            4 - U3 gate count 
            5 - number of swap traps = Number of times where qubits needed to change from SLM to AOD to execute a CZ
            6 - distance traveled during swap traps
            7 - list of CZ gates that needed swap traps
            """
            frontiers, move_count, total_max_moved_dist, cz_count, u_count, swap_trap_count, trap_dist, swap_trap_gates = na.compile_circuit()
            runtime = calc_runtime(frontiers, total_max_moved_dist, swap_trap_count, trap_dist)

            end_time = time.time()
            elapsed_time = end_time - start_time

            #Save results to .pkl file
            compilation_save(algo_name, arr_width, nc, [frontiers, move_count, total_max_moved_dist, cz_count, u_count, swap_trap_count, trap_dist, HDWR, elapsed_time, runtime , (par_dims,num_circs)])
            print("Compilation of " +algo+ " complete.")

algo_list = [#'adder_9',
            'advantage_9',
#              'gcm_h6_13',
#              'heisenberg_16',
#              'hlf_10',
             'knn_n25',
#              'multiplier_10',
#              'qaoa_10',
#              'qec9xz_n17',
#              'qft_10',
#              'qugan_n39',
             'qv_32',
#              'sat_11',
             'seca_n11',
             'sqrt_18',
#              'tfim_128',
#              'vqe_uccsd_n28',
             'wstate_27'
]

for algo in algo_list:
    main_loop(algo)