"""
File that is used to perform baseline runs of Parallax.
This file is used to run Parallax on a set of benchmark circuits and save the results to a .pkl file.
Note: Change "HDWR" variable to change the simulated hardware size used for the run.
"""
import sys
import pickle
import time
 
from na_arch import NA_Architecture
from graphine import graphine

HDWR = 'Atom' #Choose this for a 35x35 qubit computer (1,225 qubits)
# HDWR = 'Quera' #Choose this for a 16x16 qubit computer (256 qubits)

def graphine_save(algo_name, qasm_file):
    print("GRAPHINE underway...")
    res = graphine(qasm_file)
    print("GRAPHINE Complete")
    #Save result
    with open('./graphine_results/'+algo_name+'_res.pkl', 'wb') as file:
        pickle.dump(res, file)
    return res

def compilation_save(algo_name, arr_dim_sz, data, aod_dim=None): 
    if HDWR == 'Atom':
        hdf = 'atom/'
        with open('./parallax_results/'+hdf+algo_name+'_res.pkl', 'wb') as file:
            pickle.dump(data, file)        
        print("Saved to: ",'./parallax_results/'+hdf+algo_name+'_res.pkl')
    elif HDWR == 'Quera':
        hdf = 'quera/'
        with open('./parallax_results/'+hdf+algo_name+'_res.pkl', 'wb') as file:
            pickle.dump(data, file)        
        print("Saved to: ",'./parallax_results/'+hdf+algo_name+'_res.pkl')
    else:
        print("Error: Invalid hardware type"+HDWR)
        
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
"""
#O(n^2), n=#points
def map_to_bounded_integer(points, width, height, radius):
    # Initialize a set to keep track of filled locations
    filled_locations = set()
    # Initialize a list to hold points that couldn't be placed immediately
    hold_list = []
    # Initialize the list of mapped points
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

    # Expand Rydberg radius proportionally to longer of the two dimensions
    radius = radius * max(width, height)
    return mapped_points, radius


def main_loop(algo):
    print("Running ",algo,"...")
    start_time = time.time()
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

    AOD_ROWS = 20 #AOD row count
    AOD_COLS = 20 #AOD col count
    print("Running with AOD Count: "+str(AOD_ROWS))
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
            print("Error: Invalid hardware "+HDWR)

        rydberg_connect, points, gate_counts, pulse_counts, connect_count, radius = res[0],res[1],res[2],res[3],res[4],res[5]
        arr_width = ARR_WIDTH
        arr_height = ARR_HEIGHT
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
        end_time = time.time()
        #elapsed_time == runtime for Parallax
        elapsed_time = end_time - start_time
        print(algo+" runtime: " ,elapsed_time)
        #runtime == Estimated circuit execution time
        runtime = calc_runtime(frontiers, total_max_moved_dist, swap_trap_count, trap_dist)
        if ARR_WIDTH == 16:
            hrdwr = 'QuEra'
        elif ARR_WIDTH == 35:
            hrdwr = 'Atom'
        else:
            hrdwr = ''
        #Save results to .pkl file
        compilation_save(algo_name, arr_width, [frontiers, move_count, total_max_moved_dist, cz_count, u_count, swap_trap_count, trap_dist, runtime, elapsed_time, hrdwr], AOD_ROWS)
        
        print("Compilation of " +algo+ " complete.")

# All algos; feel free to comment in/out desired algos (beloew are the ones used for benchmarks) 
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