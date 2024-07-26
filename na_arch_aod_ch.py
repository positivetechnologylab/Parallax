"""
File for class defs of different needed abstractions for a neutral atom qc's hardware 
"""

import copy
import numpy as np
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as mp
from matplotlib.patches import Circle
from mobile_qubits import select_mobile_qubits

#EPSILON controls how far each AOD row/col can be from each other(not the atoms themselves, but the rows/cols)
EPSILON = 0.1

"""
Atom class used as abstraction of atom stored in atom grid.
initial_position stores a beginning location of the atom (it might stay here if it is in SLM for whole program execution)
row, col will store the AOD row/col where the atom is stored if/when it is on the AOD
"""
class AOD_Atom:
    def __init__(self, atom_id, initial_position):
        self.atom_id = atom_id
        self.initial_position = initial_position  # Store the initial position from topology
        self.row = None
        self.col = None
    
    @property
    def position(self):
        if self.row and self.col:
            return (self.col.position, self.row.position)
        return self.initial_position

    def __repr__(self):
        if self.row == None or self.col == None:
            return f"Atom(id: {self.atom_id}, initial position: {self.initial_position})"
        else:
            return f"Atom(id: {self.atom_id}, position: {self.position})"

"""
AOD row class
position indicates the row's topological Y-location
atom_objects lists atoms in the row
moved is a bool flag for whether the row has been moved in a given iteration of compilation
"""
class Row:
    def __init__(self, position):
        self._position = position
        self.atom_objects = []  # List of AOD_Atom objects
        #self.moved = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = round(value, 4)

    def __repr__(self):
        return f"[{' '.join(map(str, [atom.atom_id for atom in self.atom_objects]))}] at position {self.position}"

    
"""
AOD col class
position indicates the col's topological X-location
atom_objects lists atoms in the col
moved is a bool flag for whether the col has been moved in a given iteration of compilation
"""
class Column:
    def __init__(self, position):
        self._position = position
        self.atom_objects = []  # List of AOD_Atom objects
        #self.moved = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = round(value, 4)

    def __repr__(self):
        return f"[{' '.join(map(str, [atom.atom_id for atom in self.atom_objects]))}] at position {self.position}"

"""
AOD class
rows/columns contain the atom(s) in each row/column 
atom_by_id stores a dict mapping atom_id to atom
"""
class AODGrid:
    def __init__(self, grid_dims, array_dims):
        # Initialize a list to store the positions
        self.rows = []
        self.columns = []
        self.atom_by_id = {}  # Dictionary to store atoms by their ID
        
        #Below computes initial position for rows cols s.t. they are equidistant from each other
        # Calculate spacing between each row and column
        row_space = array_dims[1] // grid_dims[0]
        col_space = array_dims[0] // grid_dims[1] 

        #loop to compute initial row positions
        for row in range(grid_dims[0]):
            y_position = (row_space // 2) + row * row_space
            
            if row == grid_dims[0] - 1:
                y_position = min(y_position, array_dims[1] - (row_space // 2))
            
            self.rows.append(Row(y_position))
 
        #loop to compute initial col positions
        for col in range(grid_dims[1]):
            x_position = (col_space // 2) + col * col_space

            if col == grid_dims[1] - 1:
                x_position = min(x_position, array_dims[0] - (col_space // 2))

            self.columns.append(Column(x_position))
            
    def add_atom(self, atom):
        self.atom_by_id[atom.atom_id] = atom
        
    def __repr__(self):
        rows_repr = self.rows
        columns_repr = self.columns
        return f"Rows:\n{rows_repr}\nColumns:\n{columns_repr}"

    
"""
Class that contains all of the core functionality of Parallax.

BIG-O:
From mobility computations:
O((number_of_gates*num_qubits^2) + (max(AOD_ROWS,AOD_COLS)*num_qubits^3)) 

From circuit scheduling:
O(num_gates * ((num_qubits^2) + (max(num_AOD_rows,num_AOD_cols)^2) * (slm_atoms+max(num_AOD_rows,num_AOD_cols))))
"""
class NA_Architecture:
    class PositionException(Exception):
        pass

    def __init__(self, grid_dims, array_dims, qubit_topology, connect_count, radius, qasm_str):
        self.grid_dims = grid_dims #AOD grid dimensions (number of rows/cols)
        self.array_dims = array_dims #Atom array dimensions (size of the computer)
        self.min_sep_dist = 0.49 #Represents atom distance from each other
        self.slm = qubit_topology #List of qubit positions
        self.radius = radius #Rydberg interaction radius
        self.qasm_str = qasm_str #qasm file that contains the circuit
        self.atom_locs = {i: position for i, position in enumerate(qubit_topology)} #Dict of qubit positions
        self.aod = AODGrid(grid_dims, array_dims)
        
        #Use for AOD testing: This is used to test impact of different AOD row/col counts (last argument changes AOD count)
        self.mobile_qubits = select_mobile_qubits(qubit_topology, connect_count, radius, qasm_str, min(self.grid_dims[0],self.grid_dims[1]),self.grid_dims[0])
        
        self.initialize_atoms_on_aod()
        self.moved_atoms_list = [] #Keeps track of recursion in atom movement
        self.recurse_count = 0
        self.trap_dist = 0
        self.graph_count = 0
    
    def get_qubit_position(self, qubit_ind):
        """Retrieve the qubit's position based on its location (AOD or SLM)."""
        qubit_reference = self.atom_locs[qubit_ind]
        
        if isinstance(qubit_reference, AOD_Atom):
            return qubit_reference.position  # Return AOD qubit's position
        else:
            return qubit_reference  # Return SLM qubit's position

    #Moves AOD rows/columns that are not being used out of range of the grid to prevent interference
    def move_empty_rows_cols_out_of_frame(self):
        #Can change to any location out of bounds
        OUT_OF_FRAME_POSITION = -3 

        # Move rows/cols out of range
        for row in self.aod.rows:
            if not row.atom_objects:
                row.position = OUT_OF_FRAME_POSITION
        for col in self.aod.columns:
            if not col.atom_objects:
                col.position = OUT_OF_FRAME_POSITION

    #Places all the AOD-selected atoms onto the AOD grid
    def initialize_atoms_on_aod(self):
        #Sort mobile atoms by their Y axis position, place in appropriate rows
        sorted_mobile_qubits = sorted(self.mobile_qubits, key=lambda idx: self.slm[idx][1])
        for row in self.aod.rows:
            qubit_ind = sorted_mobile_qubits.pop()
            atom = AOD_Atom(qubit_ind, self.slm[qubit_ind])
            row.position = self.slm[qubit_ind][1]
            row.atom_objects.append(atom)
            atom.row = row
            self.aod.add_atom(atom)
            if not sorted_mobile_qubits:
                break

        #Sort mobile atoms by their X axis position, place in appropriate cols
        sorted_mobile_qubits = sorted(self.mobile_qubits, key=lambda idx: self.slm[idx][0])
        for col in self.aod.columns:
            qubit_ind = sorted_mobile_qubits.pop()
            atom = self.aod.atom_by_id[qubit_ind]
            col.position = self.slm[qubit_ind][0]
            col.atom_objects.append(atom)
            atom.col = col
            if not sorted_mobile_qubits:
                break
        
        # First, update the atom_locs for the qubits moved to AOD
        for ind in self.mobile_qubits:
            self.atom_locs[ind] = self.aod.atom_by_id[ind]
            
        #Then remove from SLM device
        self.mobile_qubits.sort(reverse=True)
        for ind in self.mobile_qubits:
            del self.slm[ind]
        
        #Move all rows/cols out of frame
        self.move_empty_rows_cols_out_of_frame()
        #Adjust AOD rows/columns such that they satisfy minimum separation distance
        self.adjust_overlapping_rows()
        self.adjust_overlapping_columns()

    #Adjust AOD rows to avoid them overlapping after initialization
    #O(entity^2)
    def adjust_overlapping_rows(self):
        position_dict = {}

        # check rows against each other, add any that overlap to dict
        for row in self.aod.rows:
            if row.atom_objects:  # Check if row has atoms
                if row.position not in position_dict:
                    position_dict[row.position] = []
                position_dict[row.position].append(row)

        # adjust positions of overlapping rows
        for position, rows in position_dict.items():
            # If more than one row at this position, adjust them
            if len(rows) > 1:
                increment = 0
                for row in rows:  # Sort to move topmost row the most
                    row.position += increment
                    increment += EPSILON
    
    #Adjust AOD cols to avoid them overlapping after initialization
    #O(entity^2)
    def adjust_overlapping_columns(self):
        position_dict = {}

        # check cols against each other, add any that overlap to dict
        for col in self.aod.columns:
            if col.atom_objects:  # Check if column has atoms
                if col.position not in position_dict:
                    position_dict[col.position] = []
                position_dict[col.position].append(col)

         # adjust positions of overlapping cols
        for position, cols in position_dict.items():
            # If more than one col at this position, adjust them
            if len(cols) > 1:
                increment = 0
                for col in cols:  # Sort to move rightmost column the most
                    col.position += increment
                    increment += EPSILON
    

    """
    Function used to perform recursive movement on an AOD row/column
    This function essentially has two steps: 
    1. recursively move AOD rows/cols for atom min separation distance
    2. recursively move AOD rows/cols that themselves are too close together
    """
    def _move(self, entity_list, index, step, max_dimension, moving_atom_id = None):
        """Helper function to move an entity (row/column) recursively."""
        max_rec_flag = True

        # Create a list to store the intended final positions
        intended_positions = [entity.position for entity in entity_list]

        # Calculate the intended position for the current entity
        intended_positions[index] += step

        # Determine if we are moving in the x or y direction
        is_moving_x = isinstance(entity_list[0], Column)

        # The following logic moves AOD atoms that are in the way of the moving atom
        if is_moving_x:
            new_loc = (intended_positions[index],self.aod.atom_by_id[moving_atom_id].position[1])
        else:
            new_loc = (self.aod.atom_by_id[moving_atom_id].position[0],intended_positions[index])
        for atom_id in self.mobile_qubits:
            atom = self.aod.atom_by_id[atom_id]
            #Don't compare atom with itself
            if atom_id == moving_atom_id:
                continue
            
            #For each AOD atom, move if it is blocking AOD movement
            if atom:
                # Compute the distance of the atom to the moving atoms' new location
                distance = ((new_loc[0] - atom.position[0]) ** 2 + 
                            (new_loc[1] - atom.position[1]) ** 2) ** 0.5
                #If the new distance is too close (violates min separation distance)
                if distance < self.min_sep_dist:
                    
                    # Determine direction to move the interfering atom
                    if step > 0:  # Moving right or up
                        if is_moving_x:
                            move_dir = (atom.position[0] > intended_positions[index]) * 2 - 1
                        else:
                            move_dir = (atom.position[1] > entity_list[index].position) * 2 - 1
                    else:  # Moving left or down
                        if is_moving_x:
                            move_dir = (atom.position[0] < intended_positions[index]) * 2 - 1
                        else:
                            move_dir = (atom.position[1] < entity_list[index].position) * 2 - 1

                    # Determine whether to move in x or y direction, and recursively move the AOD atom that is blocking
                    if is_moving_x:
                        # Moving in x direction, so move interfering atom in y direction
                        max_rec_flag = self.move_atom(atom_id, atom.position[0], atom.position[1] + move_dir * self.min_sep_dist) and max_rec_flag
                    else:
                        # Moving in y direction, so move interfering atom in x direction
                        max_rec_flag = self.move_atom(atom_id, atom.position[0] + move_dir * self.min_sep_dist, atom.position[1]) and max_rec_flag
        
        # Adjust the positions of neighboring entities (rows/cols) if there's overlap.
        # Get entities (rows/cols) based on their positions
        current_entity = entity_list[index]
        entity_final_pos = current_entity.position + step
        sorted_entities = sorted(entity_list, key=lambda entity: entity.position)
        if step > 0:  # Moving right or up
            sorted_entities = sorted(entity_list, key=lambda entity: entity.position)
            sorted_entities = [entity for entity in sorted_entities if entity.position > current_entity.position]
        else:  # Moving left or down
            sorted_entities = sorted(entity_list, key=lambda entity: entity.position, reverse=True)
            sorted_entities = [entity for entity in sorted_entities if entity.position < current_entity.position and entity.position >= 0]
        #Recursively move AOD rows/cols blocking movement: 
        if step > 0: # Moving right or up
            for i in range(len(sorted_entities)):
                next_entity = sorted_entities[i]
                if next_entity.position <= 0:  # Ignore out-of-bound entities (i.e. AOD rows/cols not being used)
                    continue
                if entity_final_pos + EPSILON >= next_entity.position:
                    intended_position = entity_final_pos + EPSILON
                    # Retrieve the atom id from the next entity and move it
                    atom_id = next_entity.atom_objects[0].atom_id
                    if is_moving_x:
                        max_rec_flag = self.move_atom(atom_id, intended_position, self.aod.atom_by_id[atom_id].position[1]) and max_rec_flag
                    else:
                        max_rec_flag = self.move_atom(atom_id, self.aod.atom_by_id[atom_id].position[0], intended_position) and max_rec_flag
                else:
                    break
        
        else: # Moving left or down
            for i in range(len(sorted_entities)):
                prev_entity = sorted_entities[i]
                if prev_entity.position <= 0:  # Ignore out-of-bound entities
                    continue
                if entity_final_pos - EPSILON <= prev_entity.position:
                    intended_position = entity_final_pos - EPSILON
                    # Retrieve the atom id from the previous entity and move it
                    atom_id = prev_entity.atom_objects[0].atom_id
                    if is_moving_x:
                        max_rec_flag = self.move_atom(atom_id, intended_position, self.aod.atom_by_id[atom_id].position[1]) and max_rec_flag
                    else:
                        max_rec_flag = self.move_atom(atom_id, self.aod.atom_by_id[atom_id].position[0], intended_position) and max_rec_flag
                else:
                    break

        entity_list[index].position = intended_positions[index]
        return max_rec_flag

    #moves AOD row 
    def _move_row(self, row_index, step, moving_atom_id):
        if step != 0:
            return self._move(self.aod.rows, row_index, step, self.array_dims[1], moving_atom_id)
        else:
            return True

    #moves AOD col
    def _move_col(self, col_index, step, moving_atom_id):
        if step != 0:
            return self._move(self.aod.columns, col_index, step, self.array_dims[0], moving_atom_id)
        else:
            return True
    
    #O((max(num_AOD_rows,num_AOD_cols)^2) * (slm_atoms+max(AOD_rows,AOD_cols)))
    def move_atom(self, atom_id, x2, y2):
        """Move an AOD_Atom in the X direction, then in the Y direction.
        We limit the amount of recursion to a constant 80 steps.  
        In our testing, this can be reduced to 20 without a major impact to performance.
        """
        recurse_limit = 80
        self.recurse_count += 1
        if self.recurse_count > 2 * max(len(self.aod.rows),len(self.aod.columns),recurse_limit):
            return False
        if atom_id not in [atom[0] for atom in self.moved_atoms_list]:
            self.moved_atoms_list.append((atom_id, self.get_qubit_position(atom_id)))

        atom = self.aod.atom_by_id[atom_id]
        if not atom:
            raise ValueError(f"No atom found with ID {atom_id}")
            
        # Adjust x2, y2 if they are within min_sep_dist of any static atom in slm
        for stationary_atom_position in self.slm:
            # Compute the Euclidean distance between the destination and the stationary atom
            distance = ((x2 - stationary_atom_position[0]) ** 2 + 
                        (y2 - stationary_atom_position[1]) ** 2) ** 0.5

            # If too cloise, adjust x2, y2 to be just outside the min_sep_dist circumference of the stationary atom
            if distance < self.min_sep_dist:
                if x2 - atom.position[0] != 0:
                    direction_x = (x2 - atom.position[0]) / abs(x2 - atom.position[0])
                else:
                    direction_x = 0
                if y2 - atom.position[1] != 0:
                    direction_y = (y2 - atom.position[1]) / abs(y2 - atom.position[1])
                else:
                    direction_y = 0
                    
                while distance < self.min_sep_dist:
                    x2 += max(direction_x,0.01)
                    y2 += max(direction_y,0.01)
                    distance = ((x2 - stationary_atom_position[0]) ** 2 + 
                                (y2 - stationary_atom_position[1]) ** 2) ** 0.5
                    

        y_diff = y2 - atom.position[1]
        x_diff = x2 - atom.position[0]

        row_index = self.aod.rows.index(atom.row)
        col_index = self.aod.columns.index(atom.col)
        
        # Move in the x (col) direction
        if self._move_col(col_index, x_diff, atom.atom_id) == False:
            return False
        # Move in the y (row) direction
        if self._move_row(row_index, y_diff, atom.atom_id) == False:
            return False
        
        return True

    #Function used to display the state of the atom grid
    def show_array(self):
        x_coords = [x for x, y in self.slm]
        y_coords = [y for x, y in self.slm]

        # Get aod coordinates
        aod_coords = [atom.position for atom in self.aod.atom_by_id.values()]
        x_coords_2 = [x for x, y in aod_coords]
        y_coords_2 = [y for x, y in aod_coords]

        cmap = mp.cm.get_cmap('tab20')
        fig, ax = mp.subplots()

        # Plotting the slm points with black edges
        ax.scatter(x_coords, y_coords, color='black', edgecolors='black')  

        # Plotting the aod points and their corresponding lines
        for i, (x, y) in enumerate(aod_coords):
            color = cmap(i % len(self.grid_dims[0]))
            ax.scatter(x, y, color=color, edgecolors='black', s=105)
            #ax.axhline(y, color=color, linestyle='--')  # Horizontal line from y-coordinate
            #ax.axvline(x, color=color, linestyle='--')  # Vertical line from x-coordinate

            #Rydberg radius-sized circle around each point
            #circle_radius = Circle((x, y), self.radius, color=color, fill=False)
            #ax.add_patch(circle_radius)

            #Minimum separation-sized circle around each point
            #circle_min_sep = Circle((x, y), self.min_sep_dist, color=color, fill=False, linestyle=':')
            #ax.add_patch(circle_min_sep)
            
            #Rydberg-interference circle around each point
            #circle_min_sep = Circle((x, y), self.radius*2.5, color=color, fill=False, linestyle='dashdot')
            #ax.add_patch(circle_min_sep)
        ax.set_xlim(-1, 35)
        ax.set_ylim(-1, 35)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Atom Grid')
        #Change to desired directory. graph_count ensures images saved using show_array are not overwritten in a single run
        fig.savefig('graph_image'+str(self.graph_count)+'.png')
        self.graph_count+=1

    def __repr__(self):
        return self.show_array()
    
    #Dummy function to represent when gate execution would occur
    def execute_frontier(self, frontier):
        pass

    #Function that returns gates simply as lists of qubits (currently assuming only CZ gates for multigates)
    def get_gates(self, qasm):
        num_qubits = None
        list_of_gates = []
        # For each row in the QASM file
        for row in qasm.splitlines():
            #get number of qubits out of the qreg file
            if "OPENQASM" in row or "include" in row:
                continue
            if 'qreg' in row:
                num_qubits = int(row.split('[')[1].split(']')[0])
                continue
                
            # Get list of all the qubits as ints involved in the operation
            qubits = [int(row.split('[')[q+1].split(']')[0]) for q in range(row.count('['))]
            list_of_gates.append(qubits)
        return list_of_gates
    
    #Returns dict of gates for each qubit, by its identifier: {1:[[1,2],[1]]...}
    def categorize_by_qubit_id(self, gates):
        result_dict = {}
        for gate in gates:
            for qubit_id in gate:
                if qubit_id not in result_dict:
                    result_dict[qubit_id] = []
                result_dict[qubit_id].append(gate)
        return result_dict

    #Boolean check for Rydberg blockade interference between two gates
    def check_interference(self, gate1, gate2):
        for qubit in gate1:
            for qubit2 in gate2:
                if np.linalg.norm(np.array(self.get_qubit_position(qubit)) - np.array(self.get_qubit_position(qubit2))) < 2.5 * self.radius:
                    return True
        return False
    
    #Processes a layer of gates to ensure none of them are in Rydberg blockade of one another
    def process_interference(self, curr_layer, gate_lists):
        index = 0
        #O(num_qubits)
        while index < len(curr_layer):
            gate_of_interest = curr_layer[index]
            if len(gate_of_interest) == 2:
                # Use a flag to track if the gate_of_interest should be incremented
                increment_gate = True
                # Iterating in reverse order from end of the list to current gate, check for Rydberg blockade and remove gates from layer if interfering
                #O(num_qubits)
                for other_index in range(len(curr_layer) - 1, index, -1):
                    other_gate = curr_layer[other_index]
                    if len(other_gate) == 2:
                        if self.check_interference(gate_of_interest, other_gate):
                            for qubit in other_gate:
                                gate_lists[qubit].insert(0, other_gate)
                            curr_layer.pop(other_index)
                            # Since we removed an interfering gate, we do not increment the gate_of_interest yet
                            increment_gate = False
                # If no interference was detected for gate of interest, move to the next gate
                if increment_gate:
                    index += 1
            else:
                index += 1

    #Get location for atom to move to
    def get_move_loc(self, q1, slm_pos_0, slm_pos_1):
        x2,y2 = float(slm_pos_0), float(slm_pos_1)
        x1,y1 = float(self.get_qubit_position(q1)[0]),float(self.get_qubit_position(q1)[1]) #(x1,y1)
        point_1 = (x1, y1)
        point_2 = (x2, y2)

        # Compute the vector from point_2 to point_1
        v = (x1 - x2, y1 - y2)

        # Compute the distance
        length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # Normalize the vector
        v_norm = (v[0] / float(length), v[1] / float(length))

        # Adjust point_2 closer to point_1 by min_sep_dist
        new_point_2 = (x2 + v_norm[0] * (self.min_sep_dist+.001), y2 + v_norm[1] * (self.min_sep_dist+.001))
        #print("new x2,y2",new_point_2[0],new_point_2[1])
        return new_point_2[0],new_point_2[1]
                
    """
    Function that actually compiles/runs the circuit.
    
    """
    #O(#gates * ((num_qubits^2) + (max(num_AOD_rows,num_AOD_cols)^2) * (slm_atoms+max(num_AOD_rows,num_AOD_cols))))
    def compile_circuit(self):
        gate_list = self.get_gates(self.qasm_str)
        gate_lists = self.categorize_by_qubit_id(self.get_gates(self.qasm_str)) #of form {1:[[1,2],[1]]...}
        all_layers = []
        num_qubits = len(self.atom_locs.keys())
        aod_reposition_counter = math.floor(math.log(len(gate_lists.keys())))
        move_count = 0
        swap_trap_count = 0
        execute_count = 0
        total_max_moved_dist = 0
        swap_trap_gates = []
        
        #While there are any gates left to execute, continue iterating and creating parallel layers
        while any(gate_lists[qubit_id] for qubit_id in gate_lists):
            curr_layer = []
            movement_done = False
            
            qubit_list = list(gate_lists.keys())
            processed_qubits = set()
            gates_to_pop = []
            #For each qubit, get its next gate (if not already processed in this layer)
            for qubit_id in qubit_list:
                
                if qubit_id in processed_qubits:
                    continue  # Skip this iteration if qubit_id is already processed

                if gate_lists[qubit_id]:
                    # Single qubit gate: Can always execute, just add to layer
                    if len(gate_lists[qubit_id][0]) == 1:
                        gates_to_pop.append(qubit_id)

                    # Two qubit gate: 
                    elif len(gate_lists[qubit_id][0]) == 2:
                        other_qubit_id = gate_lists[qubit_id][0][0] if gate_lists[qubit_id][0][0] != qubit_id else gate_lists[qubit_id][0][1]
                        # Add two qubit gate if neither has been processed yet this layer
                        if gate_lists[other_qubit_id] and gate_lists[qubit_id][0] == gate_lists[other_qubit_id][0]:
                            gates_to_pop.append(qubit_id)
                            gates_to_pop.append(other_qubit_id)

                            # Add both qubit_ids to the processed set
                            processed_qubits.add(qubit_id)
                            processed_qubits.add(other_qubit_id)
                            
            # Now, pop the gates identified in the previous loop
            # Create a set to keep track of gates we've already added to curr_layer
            added_to_curr_layer = set()
            for qubit_id in gates_to_pop:
                gate = gate_lists[qubit_id].pop(0)

                # If the gate hasn't been added to curr_layer yet, append it
                if tuple(gate) not in added_to_curr_layer:
                    curr_layer.append(gate)
                    added_to_curr_layer.add(tuple(gate))

            # Shuffle the list of keys randomly to ensure interfering gates not biased towards high/low qubit indices
            keys_list = list(gate_lists.keys())
            random.shuffle(curr_layer)
            
            #Next, for all 2-qubit gates in curr_layer, check if they need to be moved into range. For the first one that does (if any), see if one qubit in gate can be moved (i.e. is in mobile_qubits). If one can, move it. Finally, break of loop
            for i in reversed(range(len(curr_layer))):
                #If gate is 2-qubit, and out of range, we 
                # Check for movement conditions
                gate = curr_layer[i]
                moved_atom_id = None
                if len(gate) > 1: 
                    q1, q2 = gate
                    #if qubits are out of range and neither is mobile, add to swap_trap_count (the number of times we need to add/remove atoms to/from SLM/AOD to move into range)
                    if np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2))) > self.radius and q1 not in self.mobile_qubits and q2 not in self.mobile_qubits:
                        move_count += 1
                        swap_trap_count += 1
                        self.trap_dist += np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2)))
                        swap_trap_gates.append(i)
                        curr_layer.pop(i)
                        continue
                    #If it's a multi-qubit gate but movement is done for this layer, need to re-add the gate to the gatelist
                    elif movement_done and np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2))) > self.radius:
                        for q in gate:
                            gate_lists[q].insert(0, gate)
                        curr_layer.pop(i)
                    #Else move the mobile qubit into range
                    elif not movement_done:  
                        if np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2))) > self.radius: 
                            if q1 in self.mobile_qubits or q2 in self.mobile_qubits:
                                move_count += 1
                                if q1 in self.mobile_qubits:
                                    moved_atom_id = q1
                                    slm_pos = self.get_qubit_position(q2)
                                    x2,y2 = self.get_move_loc(q1, slm_pos[0], slm_pos[1])
                                    #Recursion failed, need to switch an atom into a trap and move in its own layer for execution
                                    if self.move_atom(q1, x2, y2) == False:
                                        for moved_atom_id, (original_x, original_y) in reversed(self.moved_atoms_list):
                                            moved_atom = self.aod.atom_by_id[moved_atom_id]
                                            moved_atom.row.position = original_y
                                            moved_atom.col.position = original_x
                                        # Clear the list as we've reverted the moves
                                        self.moved_atoms_list = []
                                        self.recurse_count = 0
                                        swap_trap_count += 1
                                        self.trap_dist += np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2)))
                                        swap_trap_gates.append(i)
                                        curr_layer.pop(i)
                                        continue
                                        
                                    #Check if any atoms tried to move out of range of the atom grid
                                    flg = False
                                    for mv_id, (og_x, og_y) in reversed(self.moved_atoms_list):
                                        mv_at = self.aod.atom_by_id[mv_id]
                                        cl,rw = mv_at.position[0],mv_at.position[1]
                                        if cl < -1.0 or rw < -1.0 or cl > self.array_dims[0] + 1 or rw > self.array_dims[1] + 1:
                                            for moved_atom_id, (original_x, original_y) in reversed(self.moved_atoms_list):
                                                moved_atom = self.aod.atom_by_id[moved_atom_id]
                                                moved_atom.row.position = original_y
                                                moved_atom.col.position = original_x
                                            # Clear the list as we've reverted the moves
                                            self.moved_atoms_list = []
                                            self.recurse_count = 0
                                            swap_trap_count += 1
                                            self.trap_dist += np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2)))
                                            swap_trap_gates.append(i)
                                            curr_layer.pop(i)
                                            flg = True
                                            break
                                    if flg == True:
                                        continue
                                #Do the same as above on qubit 2 in the gate                       
                                else:
                                    moved_atom_id = q2
                                    slm_pos = self.get_qubit_position(q1)
                                    x2,y2 = self.get_move_loc(q2, float(slm_pos[0]), float(slm_pos[1]))
                                    if self.move_atom(q2, x2, y2) == False:
                                        for moved_atom_id, (original_x, original_y) in reversed(self.moved_atoms_list):
                                            moved_atom = self.aod.atom_by_id[moved_atom_id]
                                            moved_atom.row.position = original_y
                                            moved_atom.col.position = original_x
                                        self.moved_atoms_list = []
                                        self.recurse_count = 0
                                        swap_trap_count += 1
                                        self.trap_dist += np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2)))
                                        swap_trap_gates.append(i)
                                        curr_layer.pop(i)                                        
                                        continue
                                    #Check if any atoms tried to move out of range of the atom grid
                                    flg = False
                                    for mv_id, (og_x, og_y) in reversed(self.moved_atoms_list):
                                        mv_at = self.aod.atom_by_id[mv_id]
                                        cl,rw = mv_at.position[0],mv_at.position[1]
                                        if cl < -1.0 or rw < -1.0 or cl > self.array_dims[0] + 1 or rw > self.array_dims[1] + 1:
                                            for moved_atom_id, (original_x, original_y) in reversed(self.moved_atoms_list):
                                                moved_atom = self.aod.atom_by_id[moved_atom_id]
                                                moved_atom.row.position = original_y
                                                moved_atom.col.position = original_x
                                            self.moved_atoms_list = []
                                            self.recurse_count = 0
                                            swap_trap_count += 1
                                            self.trap_dist += np.linalg.norm(np.array(self.get_qubit_position(q1)) - np.array(self.get_qubit_position(q2)))
                                            swap_trap_gates.append(i)
                                            curr_layer.pop(i)
                                            flg = True
                                            break
                                    if flg == True:
                                        continue
                                        
                                self.recurse_count = 0
                                movement_done = True
                               
            # Check for interference. Return gates that interfere back to gate_lists.
            self.process_interference(curr_layer, gate_lists)
            
            # Execute the gates in the frontier (currently a dummy function)
            self.execute_frontier(curr_layer)
            #Reset position of moved qubit (if any)
            if self.moved_atoms_list:
                moves = []
                for i in self.moved_atoms_list:
                    #append move
                    moves.append(np.linalg.norm(np.array(self.get_qubit_position(i[0])) - np.array((i[1][0],i[1][1]))))
                    
                    #These two lines determines whether home location is used: Without these, the atom stays in current location
                    #Comment out to remove home bases for AOD atoms (used to generate results/fig with returning home vs not)
                    self.aod.atom_by_id[i[0]].row.position = i[1][1]
                    self.aod.atom_by_id[i[0]].col.position = i[1][0]
                
                #Add max move distance from this layer to total distance moved (since AOD movement can be done in parallel, this can be used as proxy for total move time for this layer)
                total_max_moved_dist += max(moves) * 2
            self.moved_atoms_list = []
            
            #len can be 0 if all gates were swap_trap gates; if it is, we just ignore this on returned layer list.
            if len(curr_layer) > 0:
                all_layers.append(curr_layer)
            execute_count += len(curr_layer)
            for gate in curr_layer:
                gate_list.remove(gate)
        
        #Get gate counts
        cz_count = 0
        u_count = 0
        for f in all_layers:
            for g in f:
                if len(g) == 1:
                    u_count += 1
                else:
                    cz_count += 1
        
        return all_layers, move_count, total_max_moved_dist, cz_count+swap_trap_count, u_count, swap_trap_count, self.trap_dist, swap_trap_gates
        